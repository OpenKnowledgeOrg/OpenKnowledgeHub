"""Archive one Semantic Scholar dataset of one pinned release into the raw bucket."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from contextlib import aclosing
from datetime import datetime

from pydantic import BaseModel, ValidationError

from openknowledge_contracts.datasets import (
    DIFF_INVENTORY_NAME,
    INVENTORY_NAME,
    ArchiveDocument,
    DatasetInventory,
    DiffInventory,
    DiffPart,
    ReleaseRecord,
    ShardRef,
    SignedMaterialError,
)
from openknowledge_contracts.manifests import (
    SCHEMA_VERSION,
    SOURCE,
    DatasetManifest,
    ManifestEntry,
    ObjectPath,
    Producer,
    ShardReceipt,
)
from openknowledge_core.ingestion.commands import (
    DownloadDataset,
    DownloadDatasetResult,
    DownloadDiff,
    DownloadShard,
    DownloadShardResult,
    PinnedRelease,
    PinRelease,
)
from openknowledge_core.ingestion.errors import (
    ArchiveConflict,
    GzipInvalid,
    IncompleteTransfer,
    IntegrityError,
    InventoryChanged,
    LinkExpired,
    ManifestIncomplete,
    ReceiptInvalid,
    ShardNotListed,
    SourceRejected,
    StoredCopyMismatch,
    StoreUnavailable,
)
from openknowledge_core.ingestion.integrity import IntegrityReport, StreamCheck
from openknowledge_core.ingestion.models import (
    ByteStream,
    CheckedStream,
    DiffListing,
    PinnedDataset,
    PinnedDiff,
    ShardLink,
    ShardOutcome,
    StoredObject,
    dataset_file_path,
    diff_file_path,
    release_record_path,
)
from openknowledge_core.ingestion.ports import ArtifactStore, ReleaseCatalog, ShardDownloader

log = logging.getLogger(__name__)

JSON = "application/json"
GZIP = "application/gzip"
RECEIPT_LIMIT = 64 * 1024
DOCUMENT_LIMIT = 16 * 1024 * 1024
AGAIN = "rerun later, and report it to Semantic Scholar if it repeats"
SERVED_NOW = "what Semantic Scholar serves now"


class DownloadDatasetService:
    def __init__(
        self,
        *,
        catalog: ReleaseCatalog,
        downloads: ShardDownloader,
        store: ArtifactStore,
        producer: Producer,
        clock: Callable[[], datetime],
    ) -> None:
        self._catalog = catalog
        self._downloads = downloads
        self._store = store
        self._producer = producer
        self._clock = clock
        self._links: dict[tuple[str | None, str, str], dict[str, ShardLink]] = {}

    @property
    def catalog(self) -> ReleaseCatalog:
        return self._catalog

    async def pin_release(self, command: PinRelease) -> PinnedRelease:
        """``latest`` is resolved here and nowhere else; a stored record of a release wins."""
        if command.release != "latest":
            path = release_record_path(command.bucket, command.prefix, command.release)
            if await self._stored_release(path) == command.release:
                return PinnedRelease(release_id=command.release, record=path)
        metadata = await self._catalog.resolve_release(command.release)
        if command.release != "latest" and metadata.release_id != command.release:
            raise SourceRejected(
                f"Semantic Scholar sent release {metadata.release_id} when asked for "
                f"{command.release}; rerun later"
            )
        path = release_record_path(command.bucket, command.prefix, metadata.release_id)
        record = ReleaseRecord(retrieved_at=self._clock(), metadata=metadata)
        await self._archive(
            path,
            record,
            lambda stored: _differing(
                ("release id", stored.metadata.release_id != metadata.release_id)
            ),
            compare_with=SERVED_NOW,
        )
        return PinnedRelease(release_id=metadata.release_id, record=path)

    async def pin_dataset(self, command: DownloadDataset) -> PinnedDataset:
        """A stored inventory wins over a fresh listing, which only has to name the same shards."""
        await self.pin_release(
            PinRelease(release=command.release, bucket=command.bucket, prefix=command.prefix)
        )
        listing = await self._catalog.list_dataset(command.release, command.dataset)
        if listing.summary.name != command.dataset:
            raise SourceRejected(
                f"Semantic Scholar sent dataset {listing.summary.name} when asked for "
                f"{command.dataset}; rerun later"
            )
        listed = tuple(link.shard for link in listing.links)
        try:
            inventory = DatasetInventory(
                release_id=command.release,
                retrieved_at=self._clock(),
                dataset=listing.summary,
                shards=listed,
            )
        except ValidationError:
            raise SourceRejected(
                f"the listing of {command.dataset} {command.release} names no shard, or two "
                f"shards with the same path or file name; {AGAIN}"
            ) from None
        path = dataset_file_path(
            command.bucket, command.prefix, command.release, command.dataset, INVENTORY_NAME
        )
        stored = await self._archive(
            path,
            inventory,
            lambda stored: _differing(
                ("release id", stored.release_id != command.release),
                ("dataset", stored.dataset.name != command.dataset),
                ("set of shards", _identities(stored.shards) != _identities(listed)),
            ),
            compare_with=SERVED_NOW,
        )
        pinned = PinnedDataset(
            release_id=command.release,
            dataset=command.dataset,
            bucket=command.bucket,
            prefix=command.prefix,
            shards=stored.shards,
        )
        self._links[(None, pinned.release_id, pinned.dataset)] = _by_path(listing.links)
        return pinned

    async def pin_diff(self, command: DownloadDiff) -> PinnedDiff:
        """A stored diff.json wins over a fresh listing, which only has to name the same files."""
        await self.pin_release(
            PinRelease(release=command.release, bucket=command.bucket, prefix=command.prefix)
        )
        diff = await self._diff(command.from_release, command.release, command.dataset)
        updates = tuple(link.shard for link in diff.updates)
        deletes = tuple(link.shard for link in diff.deletes)
        try:
            inventory = DiffInventory(
                from_release=command.from_release,
                release_id=command.release,
                dataset=command.dataset,
                retrieved_at=self._clock(),
                updates=updates,
                deletes=deletes,
            )
        except ValidationError:
            raise SourceRejected(
                f"the diff of {command.dataset} from {command.from_release} to {command.release} "
                f"names two files with the same path or file name; {AGAIN}"
            ) from None
        path = diff_file_path(
            command.bucket,
            command.prefix,
            command.from_release,
            command.release,
            command.dataset,
            DIFF_INVENTORY_NAME,
        )
        stored = await self._archive(
            path,
            inventory,
            lambda stored: _differing(
                ("dataset", stored.dataset != command.dataset),
                ("set of update files", _identities(stored.updates) != _identities(updates)),
                ("set of delete files", _identities(stored.deletes) != _identities(deletes)),
            ),
            compare_with=SERVED_NOW,
        )
        key = (command.from_release, command.release, command.dataset)
        self._links[key] = _by_path(diff.updates + diff.deletes)
        return PinnedDiff(
            updates=_part(command, "updates", stored.updates),
            deletes=_part(command, "deletes", stored.deletes),
        )

    async def archive_shard(self, pinned: PinnedDataset, shard: ShardRef) -> ShardOutcome:
        """Leave ``shard`` archived with a valid receipt: verified, recovered or downloaded."""
        if shard not in pinned.shards:
            raise ShardNotListed(f"{shard.file_name} is not in {pinned.inventory_path}")
        receipt = await self._store.stat(pinned.receipt_path(shard))
        if receipt is not None:
            await self._verify_receipt(pinned, shard, receipt)
            return ShardOutcome.VERIFIED
        target = pinned.object_path(shard)
        existing = await self._store.stat(target)
        if existing is None:
            if await self._download(pinned, shard):
                return ShardOutcome.CREATED
            existing = await self._store.stat(target)
            if existing is None:
                raise _vanished(target)
        await self._recover(pinned, shard, existing)
        return ShardOutcome.RECOVERED

    async def archive_shard_at(self, command: DownloadShard) -> DownloadShardResult:
        if command.from_release is None or command.part is None:
            pinned = await self.pin_dataset(
                DownloadDataset(
                    release=command.release,
                    dataset=command.dataset,
                    bucket=command.bucket,
                    prefix=command.prefix,
                )
            )
        else:
            diff = await self.pin_diff(
                DownloadDiff(
                    from_release=command.from_release,
                    release=command.release,
                    dataset=command.dataset,
                    bucket=command.bucket,
                    prefix=command.prefix,
                )
            )
            pinned = diff.part(command.part)
        if command.shard >= len(pinned.shards):
            last = len(pinned.shards) - 1
            numbered = (
                "no shard"
                if last < 0
                else "one shard, numbered 0"
                if last == 0
                else f"shards numbered 0 to {last}"
            )
            among = "" if pinned.part is None else f" among its {pinned.part}"
            raise ShardNotListed(
                f"{pinned.inventory_path} has {numbered}{among}, so there is no shard "
                f"{command.shard}"
            )
        shard = pinned.shards[command.shard]
        outcome = await self.archive_shard(pinned, shard)
        return DownloadShardResult(
            release_id=pinned.release_id,
            dataset=pinned.dataset,
            index=command.shard,
            shard=shard,
            outcome=outcome,
            receipt=pinned.receipt_path(shard),
        )

    async def unfinished(self, pinned: PinnedDataset) -> tuple[int, ...]:
        """Indexes of the shards without a receipt."""
        receipts = [await self._store.stat(pinned.receipt_path(shard)) for shard in pinned.shards]
        return tuple(index for index, receipt in enumerate(receipts) if receipt is None)

    async def write_manifest(self, pinned: PinnedDataset) -> ObjectPath:
        """Write the manifest after every receipt checks out, or keep an earlier one that agrees."""
        entries: list[ManifestEntry] = []
        for shard in pinned.shards:
            stored = await self._store.stat(pinned.receipt_path(shard))
            if stored is None:
                raise ManifestIncomplete(
                    f"{pinned.receipt_path(shard)} is missing, and the manifest needs every "
                    "receipt; resume the dataset download before writing its manifest"
                )
            receipt = await self._verify_receipt(pinned, shard, stored)
            entries.append(
                ManifestEntry(
                    object_ref=receipt.object_ref,
                    size=receipt.size,
                    sha256=receipt.sha256,
                    md5=receipt.md5,
                    crc32c=receipt.crc32c,
                    records=receipt.records,
                )
            )
        manifest = DatasetManifest(
            schema_version=SCHEMA_VERSION,
            release_id=pinned.release_id,
            dataset=pinned.dataset,
            entries=tuple(entries),
            producer=self._producer,
            completed_at=self._clock(),
        )
        await self._archive(
            pinned.manifest_path,
            manifest,
            lambda stored: _differing(
                ("release id", stored.release_id != manifest.release_id),
                ("dataset", stored.dataset != manifest.dataset),
                ("entry list", stored.entries != manifest.entries),
            ),
            compare_with="the receipts beside it",
        )
        return pinned.manifest_path

    def result(
        self, pinned: PinnedDataset, outcomes: Iterable[ShardOutcome], manifest: ObjectPath
    ) -> DownloadDatasetResult:
        counts = Counter(outcomes)
        return DownloadDatasetResult(
            release_id=pinned.release_id,
            dataset=pinned.dataset,
            shards=len(pinned.shards),
            created=counts[ShardOutcome.CREATED],
            recovered=counts[ShardOutcome.RECOVERED],
            verified=counts[ShardOutcome.VERIFIED],
            inventory=pinned.inventory_path,
            manifest=manifest,
        )

    async def _download(self, pinned: PinnedDataset, shard: ShardRef) -> bool:
        """Stream the shard into a new object and receipt it, or return False if one appeared."""
        target = pinned.object_path(shard)
        source = await self._open(pinned, shard)
        async with aclosing(source.chunks) as chunks:
            if source.expected_size is None:
                raise IncompleteTransfer(
                    f"{shard.file_name}: the file host sent no length, so nothing was uploaded; "
                    "rerun later"
                )
            if source.expected_size == 0:
                raise GzipInvalid(
                    f"{shard.file_name}: the file host says the shard is empty, which no gzip "
                    "file is, so nothing was uploaded; rerun later"
                )
            check = StreamCheck()
            body = _verified(chunks, check, source.expected_size)
            try:
                stored = await self._store.create_from_stream(target, body, content_type=GZIP)
            except IntegrityError as exc:
                raise type(exc)(
                    f"{shard.file_name}: {exc}; nothing was stored, so a rerun downloads it again"
                ) from None
        if stored is None:
            return False
        if not body.completed or check.report is None:
            raise StoredCopyMismatch(
                f"{target} was finalized before its stream passed the checks; "
                f"{_delete_hint(target)}"
            )
        await self._write_receipt(pinned, shard, stored, check.report)
        return True

    async def _open(self, pinned: PinnedDataset, shard: ShardRef) -> ByteStream:
        try:
            return await self._downloads.open(await self._link(pinned, shard, refresh=False))
        except LinkExpired:
            log.info("%s: the link expired, fetching a fresh listing", shard.file_name)
        try:
            return await self._downloads.open(await self._link(pinned, shard, refresh=True))
        except LinkExpired as exc:
            message = f"{shard.file_name}: a freshly listed link was refused too ({exc})"
        raise LinkExpired(f"{message}; {AGAIN}")

    async def _link(self, pinned: PinnedDataset, shard: ShardRef, *, refresh: bool) -> ShardLink:
        key = (pinned.from_release, pinned.release_id, pinned.dataset)
        links = None if refresh else self._links.get(key)
        if links is None:
            listed, every = await self._listed(pinned)
            if _identities(link.shard for link in listed) != _identities(pinned.shards):
                raise InventoryChanged(
                    f"a fresh listing of {_about(pinned)} names other shards "
                    f"than {pinned.inventory_path}"
                )
            links = self._links[key] = _by_path(every)
        return links[shard.source_path]

    async def _listed(
        self, pinned: PinnedDataset
    ) -> tuple[tuple[ShardLink, ...], tuple[ShardLink, ...]]:
        """The links to ``pinned``'s shards as listed now, and every link of that listing."""
        if pinned.from_release is None:
            listing = await self._catalog.list_dataset(pinned.release_id, pinned.dataset)
            return listing.links, listing.links
        diff = await self._diff(pinned.from_release, pinned.release_id, pinned.dataset)
        listed = diff.updates if pinned.part == "updates" else diff.deletes
        return listed, diff.updates + diff.deletes

    async def _diff(self, from_release: str, release: str, dataset: str) -> DiffListing:
        chain = await self._catalog.list_diffs(from_release, release, dataset)
        steps = [(diff.from_release, diff.release_id) for diff in chain.diffs]
        if chain.dataset != dataset or steps != [(from_release, release)]:
            raise SourceRejected(
                f"Semantic Scholar did not list exactly one diff of {dataset} from "
                f"{from_release} to {release}; check that {release} is the release right after "
                f"{from_release}"
            )
        return chain.diffs[0]

    async def _recover(
        self, pinned: PinnedDataset, shard: ShardRef, existing: StoredObject
    ) -> None:
        """Receipt a stored object only if its length still matches what the source declares."""
        path = existing.ref.path
        log.info("%s: %s has no receipt, checking the stored copy", shard.file_name, path)
        declared = await self._open(pinned, shard)
        async with aclosing(declared.chunks):
            pass  # only the declared length is needed, not the body
        if declared.expected_size is None:
            raise IncompleteTransfer(
                f"{shard.file_name}: the file host sent no length, so {path} cannot be checked; "
                "rerun later"
            )
        # A shortened body can still be a whole gzip stream, so the length must match too.
        if declared.expected_size != existing.size:
            raise StoredCopyMismatch(
                f"{path} holds {existing.size} bytes, but the source declares "
                f"{declared.expected_size}; {_delete_hint(path)}"
            )
        source = await self._store.open_read(existing.ref)
        check = StreamCheck()
        try:
            async with aclosing(source.chunks) as chunks:
                async for chunk in chunks:
                    check.update(chunk)
            report = check.finish(existing.size)
        except IntegrityError as exc:
            raise type(exc)(
                f"the stored copy {path} fails its check ({exc}); {_delete_hint(path)}"
            ) from None
        await self._write_receipt(pinned, shard, existing, report)

    async def _write_receipt(
        self, pinned: PinnedDataset, shard: ShardRef, stored: StoredObject, report: IntegrityReport
    ) -> None:
        if stored.md5 is None:
            raise StoredCopyMismatch(
                f"{stored.ref.path} has no MD5, which GCS omits for composite objects; "
                f"{_delete_hint(stored.ref.path)}"
            )
        if stored.size != report.size or stored.md5 != report.md5:
            raise StoredCopyMismatch(
                f"{stored.ref.path} holds {stored.size} bytes with MD5 {stored.md5}, but "
                f"{report.size} bytes with MD5 {report.md5} passed the checks; "
                f"{_delete_hint(stored.ref.path)}"
            )
        receipt = ShardReceipt(
            schema_version=SCHEMA_VERSION,
            source=SOURCE,
            release_id=pinned.release_id,
            dataset=pinned.dataset,
            shard=shard,
            object_ref=stored.ref,
            size=report.size,
            sha256=report.sha256,
            md5=report.md5,
            crc32c=stored.crc32c,
            schema_ref=_schema_ref(pinned),
            gzip_members=report.gzip_members,
            records=report.records,
            uncompressed_size=report.uncompressed_size,
            producer=self._producer,
            completed_at=self._clock(),
        )
        path = pinned.receipt_path(shard)
        if await self._store.create(path, receipt.to_json(), content_type=JSON) is not None:
            return
        existing = await self._store.stat(path)
        if existing is None:
            raise _vanished(path)
        await self._verify_receipt(pinned, shard, existing)

    async def _verify_receipt(
        self, pinned: PinnedDataset, shard: ShardRef, stored: StoredObject
    ) -> ShardReceipt:
        """Accept a receipt only when it and the object it names agree on every field."""
        path = stored.ref.path
        data = await read_small(self._store, stored, RECEIPT_LIMIT)
        if data is None:
            raise ReceiptInvalid(f"{path} is larger than a receipt can be; {_delete_hint(path)}")
        try:
            receipt = ShardReceipt.from_json(data)
        except ValueError as exc:
            raise ReceiptInvalid(
                f"{path} is not a valid receipt ({_reason(exc, ShardReceipt)}); "
                f"{_delete_hint(path)}"
            ) from None
        target = pinned.object_path(shard)
        expected: dict[str, object] = {
            "release": pinned.release_id,
            "dataset": pinned.dataset,
            "shard": shard,
            "object": target,
            "schema": _schema_ref(pinned),
        }
        named: dict[str, object] = {
            "release": receipt.release_id,
            "dataset": receipt.dataset,
            "shard": receipt.shard,
            "object": receipt.object_ref.path,
            "schema": receipt.schema_ref,
        }
        wrong = [field for field, value in expected.items() if named[field] != value]
        if wrong:
            raise ReceiptInvalid(f"{path} is for another {_listed(wrong)}; {_delete_hint(path)}")
        current = await self._store.stat(target)
        if current is None:
            raise ReceiptInvalid(
                f"{path} names {target}, which does not exist; {_delete_hint(path)}"
            )
        differ = [
            field
            for field, ours, theirs in (
                ("generation", current.ref.generation, receipt.object_ref.generation),
                ("size", current.size, receipt.size),
                ("MD5", current.md5, receipt.md5),
                ("CRC32C", current.crc32c, receipt.crc32c),
            )
            if ours != theirs
        ]
        if differ:
            raise ReceiptInvalid(
                f"{path} and {target} disagree on {_listed(differ)}; {_delete_hint(path)}"
            )
        return receipt

    async def _stored_release(self, path: ObjectPath) -> str | None:
        """The release id of a readable record at ``path``; otherwise pin_release resolves."""
        stored = await self._store.stat(path)
        data = None if stored is None else await read_small(self._store, stored, DOCUMENT_LIMIT)
        try:
            return None if data is None else ReleaseRecord.from_json(data).metadata.release_id
        except ValueError:
            return None

    async def _archive[D: ArchiveDocument](
        self,
        path: ObjectPath,
        document: D,
        differing: Callable[[D], list[str]],
        *,
        compare_with: str,
    ) -> D:
        """Create ``document``, or return the stored one when ``differing`` names no field."""
        kind = type(document).__name__
        try:
            data = document.to_json()
        except SignedMaterialError as exc:
            raise SourceRejected(f"the {kind} for {path} was refused: {exc}") from None
        if len(data) > DOCUMENT_LIMIT:
            raise SourceRejected(
                f"the {kind} for {path} would be {len(data)} bytes, more than a rerun reads back"
            )
        if await self._store.create(path, data, content_type=JSON) is not None:
            return document
        stored = await self._store.stat(path)
        if stored is None:
            raise _vanished(path)
        existing_data = await read_small(self._store, stored, DOCUMENT_LIMIT)
        if existing_data is None:
            raise ArchiveConflict(f"{path} is larger than a {kind} can be; {_delete_hint(path)}")
        try:
            existing = type(document).from_json(existing_data)
        except ValueError as exc:
            raise ArchiveConflict(
                f"{path} is not a valid {kind} ({_reason(exc, type(document))}); "
                f"{_delete_hint(path)}"
            ) from None
        wrong = differing(existing)
        if wrong:
            raise ArchiveConflict(
                f"{path} holds a {kind} with a different {_listed(wrong)} than this run's; "
                f"it stays as it is, so compare it with {compare_with} before anyone removes it"
            )
        return existing


async def read_small(store: ArtifactStore, stored: StoredObject, limit: int) -> bytes | None:
    if stored.size > limit:
        return None
    source = await store.open_read(stored.ref)
    data = bytearray()
    async with aclosing(source.chunks) as chunks:
        async for chunk in chunks:
            data += chunk
            if len(data) > limit:
                return None
    return bytes(data)


def _verified(chunks: AsyncIterator[bytes], check: StreamCheck, size: int) -> CheckedStream:
    async def checked() -> AsyncGenerator[bytes, None]:
        async for chunk in chunks:
            check.update(chunk)
            yield chunk
        check.finish(size)
        body.completed = True

    body = CheckedStream(size=size, chunks=checked())
    return body


def _by_path(links: Iterable[ShardLink]) -> dict[str, ShardLink]:
    return {link.shard.source_path: link for link in links}


def _differing(*fields: tuple[str, bool]) -> list[str]:
    return [name for name, differs in fields if differs]


def _identities(shards: Iterable[ShardRef]) -> frozenset[tuple[str, str]]:
    """Shards as a set, so two listings compare by identity and never by order."""
    return frozenset((shard.source_path, shard.file_name) for shard in shards)


def _schema_ref(pinned: PinnedDataset) -> str:
    ref = f"{SOURCE}/{pinned.dataset}@{pinned.release_id}"
    return f"{ref}/deletes" if pinned.part == "deletes" else ref


def _about(pinned: PinnedDataset) -> str:
    if pinned.from_release is None:
        return f"{pinned.dataset} {pinned.release_id}"
    return (
        f"the {pinned.part} of {pinned.dataset} from {pinned.from_release} to {pinned.release_id}"
    )


def _part(command: DownloadDiff, part: DiffPart, shards: tuple[ShardRef, ...]) -> PinnedDataset:
    return PinnedDataset(
        release_id=command.release,
        dataset=command.dataset,
        bucket=command.bucket,
        prefix=command.prefix,
        shards=shards,
        from_release=command.from_release,
        part=part,
    )


def _reason(exc: ValueError, kind: type[BaseModel]) -> str:
    """Why a stored document failed to parse, naming at most a field or JSON path, never a value."""
    if isinstance(exc, ValidationError):
        problems: set[str] = set()
        for error in exc.errors(include_input=False, include_url=False):
            top = error["loc"][0] if error["loc"] else None
            if top is None:
                problems.add("not an object")
            elif top not in kind.model_fields:
                problems.add("unexpected fields")
            else:
                problems.add(f"{'missing' if error['type'] == 'missing' else 'invalid'} {top}")
        return ", ".join(sorted(problems))
    return str(exc)  # from_json's own errors name at most a JSON path


def _delete_hint(path: ObjectPath) -> str:
    return f"okp never replaces an object, so delete {path} with an account that can, then rerun"


def _vanished(path: ObjectPath) -> StoreUnavailable:
    return StoreUnavailable(
        f"GCS refused to create {path} because it exists, then could not find it; rerun"
    )


def _listed(names: list[str]) -> str:
    *rest, last = names
    return f"{', '.join(rest)} and {last}" if rest else last
