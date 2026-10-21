"""In-memory core ports, with FakeStore checked against the GCS adapter by test_port_contracts."""

from __future__ import annotations

import gzip
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from pydantic import SecretStr

from openknowledge_contracts.datasets import DatasetSummary, ReleaseMetadata, ShardRef
from openknowledge_contracts.manifests import ObjectPath, ObjectRef, Producer
from openknowledge_contracts.tables.contract import UpdateMode
from openknowledge_core.ingestion.commands import DownloadDataset, DownloadDiff, DownloadShard
from openknowledge_core.ingestion.errors import (
    IncompleteTransfer,
    IngestionError,
    LinkExpired,
    ObjectChanged,
    SourceUnavailable,
)
from openknowledge_core.ingestion.families import ArchiveFamily
from openknowledge_core.ingestion.models import (
    AppliedRelease,
    ByteStream,
    CheckedStream,
    DatasetListing,
    DiffChain,
    DiffListing,
    Duplicates,
    JobOutcome,
    LoadObject,
    LoadRecord,
    Operation,
    ShardJob,
    ShardLink,
    StoredObject,
    TableState,
)
from openknowledge_core.ingestion.service import DownloadDatasetService
from openknowledge_core.search.errors import BulkRefused
from openknowledge_core.search.models import (
    AbstractRow,
    BulkOperation,
    CorpusSlice,
    DeleteDocument,
    PaperRow,
    SourceRows,
)
from openknowledge_core.search.service import document_operation
from tests.fakes.checksums import crc32c_b64, md5_b64

RELEASE = "2026-10-06"
PRODUCER = Producer(
    transform="verbatim", transform_version=1, software_version="0.1.0", image_digest=None
)
STEP = 1000


def gz(payload: bytes) -> bytes:
    return gzip.compress(payload, mtime=0)


PAPERS = {"part-0.gz": gz(b'{"id": 1}\n'), "part-1.gz": gz(b'{"id": 2}\n' * 50)}
COMMAND = DownloadDataset(release=RELEASE, dataset="papers", bucket="raw", prefix="s2")
NEXT = "2026-10-13"
DIFF = DownloadDiff(from_release=RELEASE, release=NEXT, dataset="papers", bucket="raw", prefix="s2")


def clock() -> datetime:
    return datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def applied_release(table: str, release: str, age: timedelta) -> AppliedRelease:
    """A record of ``release`` applied to ``table``, ``age`` before ``clock()``."""
    return AppliedRelease(
        target=table,
        target_uuid=UUID(int=1),
        release=release,
        from_release=None,
        software_version=PRODUCER.software_version,
        image_digest=None,
        applied_at=clock() - age,
    )


def body_of(data: bytes, *, error: Exception | None = None, complete: bool = True) -> CheckedStream:
    """A body as core hands it to a store: ``completed`` is set after the last chunk."""

    async def chunks() -> AsyncGenerator[bytes, None]:
        for start in range(0, len(data), STEP):
            yield data[start : start + STEP]
        if error is not None:
            raise error
        body.completed = complete

    body = CheckedStream(size=len(data), chunks=chunks())
    return body


class FakeSource:
    """Semantic Scholar's API and its file host: the catalog and the downloader in one."""

    def __init__(self) -> None:
        self.releases = [RELEASE]
        self.files: dict[str, dict[str, bytes]] = {}
        self.readme = "Terms of use for the release."
        self.listings = 0
        self.resolved: list[str] = []
        self.opened: list[str] = []
        self.expired_listings: set[int] = set()
        self.unavailable: list[str] = []
        self.cut: set[str] = set()
        self.declared: dict[str, int | None] = {}
        self.reorder = False
        self.diffs: dict[str, list[tuple[str, str, list[str], list[str]]]] = {}
        self.diff_files: dict[str, bytes] = {}

    def add(self, dataset: str, name: str, data: bytes) -> None:
        self.files.setdefault(dataset, {})[name] = data

    def add_diff(
        self, dataset: str, old: str, new: str, updates: dict[str, bytes], deletes: dict[str, bytes]
    ) -> None:
        for part, files in (("updates", updates), ("deletes", deletes)):
            for name, data in files.items():
                self.diff_files[diff_ref(part, old, new, dataset, name).source_path] = data
        self.diffs.setdefault(dataset, []).append((old, new, list(updates), list(deletes)))

    async def list_releases(self) -> tuple[str, ...]:
        return tuple(self.releases)

    async def resolve_release(self, selector: str | Literal["latest"]) -> ReleaseMetadata:
        self.resolved.append(selector)
        release_id = self.releases[-1] if selector == "latest" else selector
        return ReleaseMetadata.model_validate(
            {
                "release_id": release_id,
                "README": self.readme,
                "license_notes": "ODC-BY for most datasets",
                "datasets": [
                    {"name": name, "description": f"The {name} dataset", "README": f"About {name}"}
                    for name in self.files
                ],
            }
        )

    async def list_dataset(self, release_id: str, dataset: str) -> DatasetListing:
        self.listings += 1
        names = list(self.files[dataset])
        if self.reorder and self.listings > 1:
            names.reverse()
        links = tuple(
            ShardLink(
                shard=shard_ref(dataset, name),
                url=SecretStr(
                    f"https://files.test{shard_ref(dataset, name).source_path}"
                    f"?AWSAccessKeyId=AKIAFAKE&Signature=secret{self.listings}"
                ),
            )
            for name in names
        )
        summary = DatasetSummary.model_validate(
            {
                "name": dataset,
                "description": f"The {dataset} dataset",
                "README": f"About {dataset}",
                "citation": "Kinney et al. 2023",
            }
        )
        return DatasetListing(summary=summary, links=links)

    async def list_diffs(self, start: str, end: str, dataset: str) -> DiffChain:
        self.listings += 1
        end = self.releases[-1] if end == "latest" else end
        return DiffChain(
            dataset=dataset,
            diffs=tuple(
                DiffListing(
                    from_release=old,
                    release_id=new,
                    updates=tuple(
                        self._link(diff_ref("updates", old, new, dataset, name)) for name in updates
                    ),
                    deletes=tuple(
                        self._link(diff_ref("deletes", old, new, dataset, name)) for name in deletes
                    ),
                )
                for old, new, updates, deletes in self.diffs.get(dataset, [])
                if start <= old and new <= end
            ),
        )

    def _link(self, shard: ShardRef) -> ShardLink:
        url = f"https://files.test{shard.source_path}?Signature=secret{self.listings}"
        return ShardLink(shard=shard, url=SecretStr(url))

    async def open(self, link: ShardLink) -> ByteStream:
        shard = link.shard
        self.opened.append(shard.file_name)
        listing = int(link.url.get_secret_value().rsplit("secret", 1)[1])
        if listing in self.expired_listings:
            raise LinkExpired(f"HTTP 403 from https://files.test{shard.source_path}")
        if shard.file_name in self.unavailable:
            self.unavailable.remove(shard.file_name)
            raise SourceUnavailable(f"HTTP 503 from https://files.test{shard.source_path}")
        dataset = shard.source_path.split("/")[-2]
        data = self.diff_files.get(shard.source_path)
        if data is None:
            data = self.files[dataset][shard.file_name]
        size = self.declared.get(shard.file_name, len(data))
        return ByteStream(expected_size=size, chunks=self._chunks(shard.file_name, data))

    async def _chunks(self, name: str, data: bytes) -> AsyncGenerator[bytes, None]:
        if name in self.cut:
            self.cut.discard(name)
            yield data[: len(data) // 2]
            raise SourceUnavailable(f"GET https://files.test/{name}: ReadError while streaming")
        for start in range(0, len(data), STEP):
            yield data[start : start + STEP]


def shard_ref(dataset: str, name: str) -> ShardRef:
    return ShardRef(source_path=f"/staging/{RELEASE}/{dataset}/{name}", file_name=name)


def diff_ref(part: str, old: str, new: str, dataset: str, name: str) -> ShardRef:
    return ShardRef(source_path=f"/{part}/{old}-to-{new}/{dataset}/{name}", file_name=name)


@dataclass
class _Object:
    data: bytes
    generation: int


class FakeStore:
    """GCS as the create-only account sees it, where ``put`` and ``remove`` play another writer."""

    def __init__(self) -> None:
        self.objects: dict[str, _Object] = {}
        self.content_types: dict[str, str] = {}
        self.reads: list[str] = []
        self.md5_override: dict[str, str | None] = {}
        self.crc32c_override: dict[str, str] = {}
        self.appears_during_upload: dict[str, bytes] = {}
        self.appears_before_create: dict[str, bytes] = {}
        self._generation = 1_000

    def put(self, path: ObjectPath, data: bytes) -> int:
        self._generation += 1
        self.objects[str(path)] = _Object(data, self._generation)
        return self._generation

    def remove(self, path: ObjectPath) -> None:
        del self.objects[str(path)]

    def data(self, path: ObjectPath) -> bytes:
        return self.objects[str(path)].data

    def generation(self, path: ObjectPath) -> int:
        return self.objects[str(path)].generation

    def _stored(self, path: ObjectPath) -> StoredObject:
        stored = self.objects[str(path)]
        return StoredObject(
            ref=ObjectRef(path=path, generation=stored.generation),
            size=len(stored.data),
            md5=self.md5_override.get(str(path), md5_b64(stored.data)),
            crc32c=self.crc32c_override.get(str(path), crc32c_b64(stored.data)),
        )

    async def list_objects(self, prefix: ObjectPath) -> tuple[StoredObject, ...]:
        return tuple(
            self._stored(ObjectPath(bucket=prefix.bucket, name=key.split("/", 3)[3]))
            for key in self.objects
            if key.startswith(str(prefix) + "/")
        )

    async def stat(self, path: ObjectPath) -> StoredObject | None:
        return self._stored(path) if str(path) in self.objects else None

    async def create(
        self, path: ObjectPath, data: bytes, *, content_type: str
    ) -> StoredObject | None:
        other = self.appears_before_create.pop(str(path), None)
        if other is not None:
            self.put(path, other)
        if str(path) in self.objects:
            return None
        self.put(path, data)
        self.content_types[str(path)] = content_type
        return self._stored(path)

    async def create_from_stream(
        self, path: ObjectPath, body: CheckedStream, *, content_type: str
    ) -> StoredObject | None:
        received = bytearray()
        async with aclosing(body.chunks) as chunks:
            if str(path) in self.objects:
                return None
            async for chunk in chunks:
                received += chunk
        if not body.completed or len(received) != body.size:
            raise IncompleteTransfer("the body ended before its checks completed")
        other = self.appears_during_upload.pop(str(path), None)
        if other is not None:
            self.put(path, other)
            return None
        self.put(path, bytes(received))
        self.content_types[str(path)] = content_type
        return self._stored(path)

    async def open_read(self, ref: ObjectRef) -> ByteStream:
        stored = self.objects.get(str(ref.path))
        if stored is None or stored.generation != ref.generation:
            raise ObjectChanged(f"{ref.path} no longer has generation {ref.generation}")
        self.reads.append(str(ref.path))
        return ByteStream(expected_size=len(stored.data), chunks=self._chunks(ref, stored.data))

    async def _chunks(self, ref: ObjectRef, data: bytes) -> AsyncGenerator[bytes, None]:
        for start in range(0, len(data), STEP):
            yield data[start : start + STEP]
        current = self.objects.get(str(ref.path))
        if current is None or current.generation != ref.generation:
            raise ObjectChanged(f"{ref.path} changed generation while it was read")


class FakeTables:
    """ClickHouse as the loader sees it, counting rows per object."""

    def __init__(self, store: FakeStore) -> None:
        self.store = store
        self.uuid = UUID(int=1)
        self.updates: UpdateMode = "diffs"
        self.rows: dict[str, int] = {}
        self.records: dict[str, LoadRecord] = {}
        self.loads: list[str] = []
        self.operations: list[Operation] = []
        self.failures: list[IngestionError] = []
        self.short_by = 0
        self.collapsed = 0
        self.found = Duplicates(identical_rows=0, conflicting_keys=0, examples=())
        self.scanned: list[str] = []
        self.flagged = 0
        self.marked: list[str] = []

    async def table(self, dataset: str) -> TableState:
        return TableState(
            name=dataset,
            uuid=self.uuid,
            has_rows=bool(self.rows),
            load_sql_sha256="0" * 64,
            delete_sql_sha256="1" * 64,
            updates=self.updates,
        )

    async def loaded(self, dataset: str, source_object: ObjectPath) -> LoadRecord | None:
        return self.records.get(str(source_object))

    async def load(self, request: LoadObject) -> int:
        path = request.object_ref.path
        self.loads.append(path.name.rsplit("/", 1)[-1])
        self.operations.append(request.operation)
        if self.failures:
            raise self.failures.pop(0)
        lines = gzip.decompress(self.store.data(path)).count(b"\n")
        self.rows[str(path)] = lines - self.short_by
        return self.rows[str(path)]

    async def record(self, record: LoadRecord) -> None:
        self.records[record.source_object] = record

    async def live_rows(self, dataset: str, release: str) -> int:
        return sum(self.rows.values()) - self.collapsed

    async def duplicates(self, dataset: str, objects: Sequence[ObjectPath]) -> Duplicates:
        self.scanned = [path.name.rsplit("/", 1)[-1] for path in objects]
        return self.found

    async def mark_removed(self, dataset: str, release: str) -> int:
        self.marked.append(release)
        return self.flagged


class FakeApplied:
    """The ``applied`` table, newest record last."""

    def __init__(self) -> None:
        self.records: list[AppliedRelease] = []

    async def record(self, applied: AppliedRelease) -> None:
        self.records.append(applied)

    async def latest(self, target: str) -> AppliedRelease | None:
        return next((found for found in reversed(self.records) if found.target == target), None)


class FakeJobs:
    """Archives a job's shards in this process, failing each shard in ``failing`` once."""

    def __init__(self, service: DownloadDatasetService, failing: set[int]) -> None:
        self.service = service
        self.failing = failing
        self.submitted: list[tuple[int, ...]] = []
        self.jobs: list[ShardJob] = []
        self.parts: list[str | None] = []
        self.outcomes: dict[str, JobOutcome] = {}
        self.left: dict[str, bool] = {}
        self.deleted: list[str] = []
        self.waited: list[str] = []

    async def running(self, family: ArchiveFamily = ArchiveFamily.CORE) -> dict[str, bool]:
        self.family = family
        left, self.left = self.left, {}
        return left

    async def delete(self, name: str) -> None:
        self.deleted.append(name)

    async def submit(self, job: ShardJob) -> str:
        self.jobs.append(job)
        self.submitted.append(job.shards)
        self.parts.append(job.part)
        outcome = JobOutcome.SUCCEEDED
        for index in job.shards:
            if index in self.failing:
                self.failing.discard(index)
                outcome = JobOutcome.FAILED
                continue
            await self.service.archive_shard_at(
                DownloadShard(
                    release=job.release_id,
                    dataset=job.dataset,
                    shard=index,
                    bucket=job.bucket,
                    prefix=job.prefix,
                    from_release=job.from_release,
                    part=job.part,
                )
            )
        name = f"job-{len(self.submitted)}"
        self.outcomes[name] = outcome
        return name

    async def wait(self, name: str) -> JobOutcome:
        self.waited.append(name)
        return self.outcomes[name]


class FakeSearchSource:
    """The two tables as the search source reads them: joined rows for a base or a diff."""

    def __init__(self) -> None:
        self.base_rows: list[SourceRows] = []
        self.diff_rows: list[SourceRows] = []
        self.asked: list[tuple[str | None, str, CorpusSlice]] = []

    async def base(self, release: str, corpus_slice: CorpusSlice) -> AsyncGenerator[SourceRows]:
        self.asked.append((None, release, corpus_slice))
        return self._stream(self.base_rows)

    async def diff(
        self, from_release: str, release: str, corpus_slice: CorpusSlice
    ) -> AsyncGenerator[SourceRows]:
        self.asked.append((from_release, release, corpus_slice))
        return self._stream(self.diff_rows)

    async def _stream(self, rows: list[SourceRows]) -> AsyncGenerator[SourceRows]:
        for row in rows:
            yield row


class FakeSearchSink:
    """An index per name, keeping each id's latest version and its source, or None once deleted."""

    def __init__(self) -> None:
        self.indices: dict[str, dict[str, tuple[int, dict[str, Any] | None]]] = {}
        self.created: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self.requests: list[tuple[str, int]] = []
        self.refreshed: list[str] = []
        self.refused: str | None = None
        """A corpus ID whose item every bulk request refuses."""

    async def exists(self, index: str) -> bool:
        return index in self.indices

    async def create_index(
        self, index: str, mapping: Mapping[str, Any], settings: Mapping[str, Any]
    ) -> None:
        self.indices[index] = {}
        self.created[index] = (dict(mapping), dict(settings))

    async def bulk(self, index: str, operations: Sequence[BulkOperation]) -> None:
        self.requests.append((index, len(operations)))
        documents = self.indices.setdefault(index, {})
        for operation in operations:
            current = documents.get(operation.id)
            if operation.id == self.refused or (current and operation.version < current[0]):
                raise BulkRefused(
                    f"{index}: 1 of {len(operations)} bulk items refused; "
                    f"corpusid {operation.id}: version_conflict_engine_exception"
                )
            source = None if isinstance(operation, DeleteDocument) else operation.document.source()
            documents[operation.id] = (operation.version, source)

    async def refresh(self, index: str) -> None:
        self.refreshed.append(index)

    async def count(self, index: str) -> int:
        return sum(1 for _, source in self.indices[index].values() if source is not None)


def paper_row(corpusid: int, version: int, **values: Any) -> PaperRow:  # noqa: ANN401
    """A live ``papers`` row at ``version`` with ``values`` over the defaults."""
    return PaperRow.model_validate(
        {
            "corpusid": corpusid,
            "version": version,
            "source_release": RELEASE,
            "is_deleted": False,
            **values,
        }
    )


def abstract_row(corpusid: int, version: int, text: str = "A synthetic abstract.") -> AbstractRow:
    return AbstractRow(
        corpusid=corpusid, version=version, source_release=RELEASE, is_deleted=False, abstract=text
    )


def search_document(paper: PaperRow, abstract: AbstractRow | None = None) -> BulkOperation:
    """The write the core builds for ``paper`` and ``abstract`` at contract version 1."""
    operation = document_operation(
        SourceRows(corpusid=paper.corpusid, paper=paper, abstract=abstract), 1
    )
    assert operation is not None
    return operation
