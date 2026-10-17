"""Immutable archive completion anchored to an explicitly selected base."""

from collections.abc import Callable
from datetime import datetime
from hashlib import sha256
from typing import Literal, Self

from pydantic import AwareDatetime, model_validator

from openknowledge_contracts.datasets import (
    ArchiveDocument,
    DatasetInventory,
    DatasetName,
    DiffInventory,
    ReleaseId,
)
from openknowledge_contracts.manifests import (
    DatasetManifest,
    ObjectPath,
    ObjectRef,
    Producer,
    Sha256Hex,
)
from openknowledge_core.ingestion.commands import (
    BatchDiffDownloadResult,
    DownloadDataset,
    DownloadDiff,
)
from openknowledge_core.ingestion.errors import ArchiveConflict
from openknowledge_core.ingestion.models import PinnedDataset
from openknowledge_core.ingestion.ports import ArtifactStore
from openknowledge_core.ingestion.service import DOCUMENT_LIMIT, DownloadDatasetService, read_small


class HashedRef(ArchiveDocument):
    ref: ObjectRef
    sha256: Sha256Hex


class ArchiveCompletion(ArchiveDocument):
    schema_version: Literal[1] = 1
    dataset: DatasetName
    kind: Literal["base", "diff"]
    release: ReleaseId
    from_release: ReleaseId | None
    inventory: HashedRef
    manifests: tuple[HashedRef, ...]
    predecessor: HashedRef | None
    completed_at: AwareDatetime
    producer: Producer

    @model_validator(mode="after")
    def _identity(self) -> Self:
        if self.kind == "base":
            if (
                self.from_release is not None
                or self.predecessor is not None
                or len(self.manifests) != 1
            ):
                raise ValueError("base completion requires one manifest and no predecessor")
        elif (
            self.from_release is None
            or self.from_release >= self.release
            or self.predecessor is None
            or len(self.manifests) > 2
        ):
            raise ValueError("diff completion requires an ordered transition and predecessor")
        return self


def completion_path(command: DownloadDataset | DownloadDiff) -> ObjectPath:
    parts = (
        ("diffs", command.from_release, command.release)
        if isinstance(command, DownloadDiff)
        else ("bases", command.release)
    )
    return ObjectPath.under(
        command.bucket,
        command.prefix,
        "archive-progress",
        command.dataset,
        *parts[:-1],
        parts[-1] + ".json",
    )


def _completion_identity(
    command: DownloadDataset | DownloadDiff,
    reference: HashedRef,
    record: ArchiveCompletion,
) -> None:
    if (record.dataset, record.release, record.from_release, reference.ref.path) != (
        command.dataset,
        command.release,
        command.from_release if isinstance(command, DownloadDiff) else None,
        completion_path(command),
    ):
        raise ArchiveConflict("archive completion identity mismatch")
    if record.kind != ("diff" if isinstance(command, DownloadDiff) else "base"):
        raise ArchiveConflict("archive completion kind mismatch")


class ArchiveProgress:
    def __init__(
        self,
        store: ArtifactStore,
        downloads: DownloadDatasetService,
        producer: Producer,
        clock: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._downloads = downloads
        self._producer = producer
        self._clock = clock

    async def reference(self, path: ObjectPath) -> HashedRef:
        stored = await self._store.stat(path)
        if stored is None:
            raise ArchiveConflict(f"missing archive object {path}")
        data = await self._bytes(stored.ref)
        return HashedRef(ref=stored.ref, sha256=sha256(data).hexdigest())

    async def read[D: ArchiveDocument](self, reference: HashedRef, kind: type[D]) -> D:
        data = await self._bytes(reference.ref)
        if sha256(data).hexdigest() != reference.sha256:
            raise ArchiveConflict(f"content hash changed for {reference.ref.path}")
        try:
            return kind.from_json(data)
        except ValueError:
            raise ArchiveConflict(f"invalid archive document {reference.ref.path}") from None

    async def _bytes(self, ref: ObjectRef) -> bytes:
        stored = await self._store.stat(ref.path)
        if stored is None or stored.ref != ref:
            raise ArchiveConflict(f"archive generation changed: {ref.path}")
        data = await read_small(self._store, stored, DOCUMENT_LIMIT)
        if data is None:
            raise ArchiveConflict(f"archive document exceeds size limit: {ref.path}")
        return data

    async def _inventory_parts(
        self,
        command: DownloadDataset | DownloadDiff,
        inventory: HashedRef,
    ) -> tuple[PinnedDataset, ...]:
        seed = PinnedDataset(
            shards=(),
            release_id=command.release,
            dataset=command.dataset,
            bucket=command.bucket,
            prefix=command.prefix,
            from_release=command.from_release if isinstance(command, DownloadDiff) else None,
            part="updates" if isinstance(command, DownloadDiff) else None,
        )
        if inventory.ref.path != seed.inventory_path:
            raise ArchiveConflict("archive inventory path mismatch")
        if isinstance(command, DownloadDiff):
            diff = await self.read(inventory, DiffInventory)
            if (diff.dataset, diff.from_release, diff.release_id) != (
                command.dataset,
                command.from_release,
                command.release,
            ):
                raise ArchiveConflict("diff inventory identity mismatch")
            return tuple(
                seed.model_copy(update={"shards": shards, "part": part})
                for part, shards in (("updates", diff.updates), ("deletes", diff.deletes))
                if shards
            )
        base = await self.read(inventory, DatasetInventory)
        if (base.dataset.name, base.release_id) != (command.dataset, command.release):
            raise ArchiveConflict("base inventory identity mismatch")
        return (seed.model_copy(update={"shards": base.shards}),)

    async def _artifacts(
        self,
        command: DownloadDataset | DownloadDiff,
    ) -> tuple[HashedRef, tuple[HashedRef, ...]]:
        seed = PinnedDataset(
            shards=(),
            release_id=command.release,
            dataset=command.dataset,
            bucket=command.bucket,
            prefix=command.prefix,
            from_release=command.from_release if isinstance(command, DownloadDiff) else None,
            part="updates" if isinstance(command, DownloadDiff) else None,
        )
        inventory = await self.reference(seed.inventory_path)
        manifests: list[HashedRef] = []
        for part in await self._inventory_parts(command, inventory):
            # Creation/adoption reuses full receipt and object validation.
            path = await self._downloads.write_manifest(part)
            manifests.append(await self.reference(path))
        return inventory, tuple(manifests)

    async def baseline_manifest(
        self,
        reference: HashedRef,
        dataset: str,
        bucket: str,
        prefix: str,
    ) -> DatasetManifest:
        manifest = await self.read(reference, DatasetManifest)
        if manifest.dataset != dataset:
            raise ArchiveConflict("configured baseline identifies another dataset")
        command = DownloadDataset(
            release=manifest.release_id,
            dataset=dataset,
            bucket=bucket,
            prefix=prefix,
        )
        _, manifests = await self._artifacts(command)
        if manifests != (reference,):
            raise ArchiveConflict("configured baseline differs from verified base manifest")
        return manifest

    async def finalize(
        self,
        command: DownloadDataset | DownloadDiff,
        predecessor: HashedRef | None = None,
    ) -> HashedRef:
        if isinstance(command, DownloadDiff):
            if predecessor is None:
                raise ArchiveConflict("diff completion requires an anchored predecessor")
            previous = await self.read(predecessor, ArchiveCompletion)
            if previous.dataset != command.dataset or previous.release != command.from_release:
                raise ArchiveConflict("diff predecessor identifies another dataset or release")
        elif predecessor is not None:
            raise ArchiveConflict("base completion cannot have a predecessor")
        inventory, manifests = await self._artifacts(command)
        document = ArchiveCompletion(
            dataset=command.dataset,
            kind="diff" if isinstance(command, DownloadDiff) else "base",
            release=command.release,
            from_release=command.from_release if isinstance(command, DownloadDiff) else None,
            inventory=inventory,
            manifests=manifests,
            predecessor=predecessor,
            completed_at=self._clock(),
            producer=self._producer,
        )
        path = completion_path(command)
        await self._store.create(path, document.to_json(), content_type="application/json")
        reference = await self.reference(path)
        existing = await self.read(reference, ArchiveCompletion)
        fields = {"completed_at", "producer"}
        if existing.model_dump(exclude=fields) != document.model_dump(exclude=fields):
            raise ArchiveConflict(f"immutable completion disagrees with archive: {path}")
        return reference

    async def head(self, base: DownloadDataset, anchor: HashedRef) -> tuple[str, HashedRef]:
        chain = await self._chain(base, anchor)
        reference, record = chain[-1]
        return record.release, reference

    async def _chain(
        self,
        base: DownloadDataset,
        anchor: HashedRef,
    ) -> list[tuple[HashedRef, ArchiveCompletion]]:
        root = await self.read(anchor, ArchiveCompletion)
        if root.kind != "base":
            raise ArchiveConflict("configured anchor is not a base")
        await self._validate(base, anchor, root)
        prefix = ObjectPath.under(
            base.bucket, base.prefix, "archive-progress", base.dataset, "diffs"
        )
        records: dict[str, tuple[HashedRef, ArchiveCompletion]] = {}
        for stored in await self._store.list_objects(prefix):
            reference = await self.reference(stored.ref.path)
            record = await self.read(reference, ArchiveCompletion)
            if record.kind != "diff" or record.from_release is None:
                raise ArchiveConflict("non-diff record in diff progress")
            command = DownloadDiff(
                **base.model_dump(exclude={"release"}),
                from_release=record.from_release,
                release=record.release,
            )
            _completion_identity(command, reference, record)
            # A replacement base supersedes transitions ending at or before it.
            if record.release <= base.release:
                continue
            if record.from_release < base.release:
                raise ArchiveConflict("archive transition crosses the selected base boundary")
            await self._validate(command, reference, record)
            if record.from_release in records:
                raise ArchiveConflict("conflicting archive progress branches")
            records[record.from_release] = (reference, record)
        chain = [(anchor, root)]
        current, reference = base.release, anchor
        while current in records:
            next_ref, record = records.pop(current)
            if record.predecessor != reference:
                raise ArchiveConflict("archive predecessor reference mismatch")
            chain.append((next_ref, record))
            current, reference = record.release, next_ref
        if records:
            raise ArchiveConflict("disconnected archive progress; explicit base selection required")
        return chain

    async def manual_diff(
        self,
        base: DownloadDataset,
        anchor: HashedRef,
        command: DownloadDiff,
    ) -> tuple[HashedRef, BatchDiffDownloadResult | None]:
        if (base.dataset, base.bucket, base.prefix) != (
            command.dataset,
            command.bucket,
            command.prefix,
        ):
            raise ArchiveConflict("manual diff baseline identifies another dataset/location")
        chain = await self._chain(base, anchor)
        for _, record in chain:
            if (record.from_release, record.release) == (command.from_release, command.release):
                assert record.predecessor is not None
                inventory = await self.read(record.inventory, DiffInventory)
                return record.predecessor, BatchDiffDownloadResult(
                    from_release=command.from_release,
                    release_id=command.release,
                    dataset=command.dataset,
                    updates=len(inventory.updates),
                    deletes=len(inventory.deletes),
                    manifests=tuple(item.ref.path for item in record.manifests),
                    jobs=(),
                )
        reference, record = chain[-1]
        if record.release == command.from_release:
            return reference, None
        raise ArchiveConflict(
            "manual archive diff must extend the anchored head or retry a completed transition"
        )

    async def _validate(
        self,
        command: DownloadDataset | DownloadDiff,
        reference: HashedRef,
        record: ArchiveCompletion,
    ) -> None:
        _completion_identity(command, reference, record)
        parts = await self._inventory_parts(command, record.inventory)
        if tuple(item.ref.path for item in record.manifests) != tuple(
            part.manifest_path for part in parts
        ):
            raise ArchiveConflict("archive completion manifest parts/paths mismatch")
        for part, reference in zip(parts, record.manifests, strict=True):
            manifest = await self.read(reference, DatasetManifest)
            if (manifest.dataset, manifest.release_id) != (command.dataset, command.release):
                raise ArchiveConflict("archive manifest identity mismatch")
            if tuple(entry.object_ref.path for entry in manifest.entries) != tuple(
                part.object_path(shard) for shard in part.shards
            ):
                raise ArchiveConflict("archive manifest entries disagree with pinned inventory")
