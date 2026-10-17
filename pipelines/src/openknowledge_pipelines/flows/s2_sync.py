"""Sequential coordinator composition; each source call owns exactly one S2 lease."""

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

from prefect import get_run_logger

from openknowledge_contracts.manifests import DatasetManifest
from openknowledge_contracts.tables.s2 import CONTRACTS
from openknowledge_core.ingestion.archive_progress import ArchiveProgress
from openknowledge_core.ingestion.commands import (
    DownloadDataset,
    DownloadDiff,
    LoadDataset,
    LoadDiff,
)
from openknowledge_core.ingestion.coordinator import (
    CORE_ORDER,
    LARGE_ORDER,
    DatasetSyncResult,
    SyncCommand,
    SyncIncomplete,
    SyncResult,
    SyncTransition,
    guard_diff,
)
from openknowledge_core.ingestion.discovery import ReleaseDiscovery, contiguous
from openknowledge_core.ingestion.errors import (
    BatchTimedOut,
    IngestionError,
    SharedCapacityError,
)
from openknowledge_core.ingestion.families import ArchiveFamily, archive_family
from openknowledge_core.ingestion.models import held_release
from openknowledge_core.ingestion.ports import (
    AppliedReleases,
    ReleaseCatalog,
    ShardJobs,
    TableLoader,
)
from openknowledge_pipelines.flows.s2_batch_download import S2BatchDownloadFlows, recover_jobs
from openknowledge_pipelines.flows.s2_load import S2LoadFlows


@dataclass(frozen=True)
class SyncServices:
    jobs: ShardJobs
    catalog: ReleaseCatalog
    tables: TableLoader
    applied: AppliedReleases
    progress: ArchiveProgress
    downloads: dict[str, S2BatchDownloadFlows]
    loads: S2LoadFlows
    s2_lease: Callable[[], AbstractAsyncContextManager[None]]
    load_lease: Callable[[], AbstractAsyncContextManager[None]]


@asynccontextmanager
async def _source_operation(
    services: SyncServices,
    family: ArchiveFamily = ArchiveFamily.CORE,
) -> AsyncGenerator[None]:
    async with services.s2_lease():
        await recover_jobs(services.jobs, family)
        yield


async def run_sync(command: SyncCommand, services: SyncServices) -> SyncResult:
    async with _source_operation(services):
        discovery = ReleaseDiscovery(services.catalog, await services.catalog.list_releases())
    result = SyncResult()
    # Freeze every core target before any archive work. Ordinary failures remain per-dataset.
    for dataset in CORE_ORDER:
        item = DatasetSyncResult(dataset=dataset)
        result.datasets.append(item)
        try:
            table = await services.tables.table(dataset)
            item.start = held_release(table, await services.applied.latest(table.name))
            item.final = item.start
            async with _source_operation(services):
                item.target = await discovery.target(dataset, item.start)
            if item.start is None:
                item.outcome, item.reason = "needs_base", "no valid applied base for current table"
            elif item.target == item.start:
                item.outcome = "current"
        except (SharedCapacityError, BatchTimedOut):
            raise
        except IngestionError as exc:
            item.reason = str(exc)
    for item in result.datasets:
        if item.reason is not None or item.outcome == "current":
            continue
        try:
            await _core(command, services, item)
            item.outcome = "synced"
        except (SharedCapacityError, BatchTimedOut):
            raise
        except IngestionError as exc:
            item.outcome, item.reason = "failed", str(exc)
    if any(item.outcome in ("failed", "needs_base") for item in result.datasets):
        _incomplete(result)
    for dataset in LARGE_ORDER:
        item = DatasetSyncResult(dataset=dataset)
        result.datasets.append(item)
        if dataset not in command.enabled_archives:
            item.outcome = "disabled"
            continue
        try:
            baseline = command.archives.get(dataset)
            if baseline is None:
                raise IngestionError(
                    "enabled archive requires an explicit verified base completion"
                )
            if (baseline.base.dataset, baseline.base.bucket, baseline.base.prefix) != (
                dataset,
                command.bucket,
                command.prefix,
            ):
                raise IngestionError(
                    "archive baseline configuration identifies another location/dataset"
                )
            item.start, predecessor = await services.progress.head(
                baseline.base, baseline.completion
            )
            item.final = item.start
            async with _source_operation(services, archive_family(dataset)):
                item.target = await discovery.target(dataset, item.start)
            if item.target == item.start:
                item.outcome = "current"
                continue
            for old, new in await _transitions(services, dataset, item.start, item.target):
                download = DownloadDiff(
                    bucket=command.bucket,
                    prefix=command.prefix,
                    dataset=dataset,
                    from_release=old,
                    release=new,
                )
                async with _source_operation(services, archive_family(dataset)):
                    await services.downloads[dataset].diff.fn(download)
                    predecessor = await services.progress.finalize(download, predecessor)
                item.archived.append(SyncTransition(start=old, release=new, objects=(predecessor,)))
                item.final = new
            item.outcome = "synced"
        except (SharedCapacityError, BatchTimedOut):
            raise
        except IngestionError as exc:
            item.reason = str(exc)
            _incomplete(result)
    return result


async def _transitions(
    services: SyncServices,
    dataset: str,
    start: str,
    target: str | None,
) -> tuple[tuple[str, str], ...]:
    if target is None:
        raise IngestionError("no advertised dataset target")
    async with _source_operation(services, archive_family(dataset)):
        chain = await services.catalog.list_diffs(start, target, dataset)
    return contiguous(chain, dataset, start, target)


async def _core(command: SyncCommand, services: SyncServices, item: DatasetSyncResult) -> None:
    if item.start is None or item.target is None:
        raise IngestionError("core target/base missing")
    common = dict(bucket=command.bucket, prefix=command.prefix, dataset=item.dataset)
    if CONTRACTS[item.dataset].updates == "snapshot":
        download = DownloadDataset(**common, release=item.target)
        async with _source_operation(services):
            archived = await services.downloads[item.dataset].dataset.fn(download)
        reference = await services.progress.reference(archived.manifest)
        transition = SyncTransition(start=item.start, release=item.target, objects=(reference,))
        item.archived.append(transition)
        async with services.load_lease():
            await services.loads.dataset.fn(LoadDataset(**common, release=item.target))
        item.loaded.append(transition)
        item.final = item.target
        return
    transitions = await _transitions(services, item.dataset, item.start, item.target)
    baseline = command.core_baselines.get(item.dataset)
    if baseline is None:
        raise IngestionError("size guard requires an explicit baseline manifest reference")
    base = await services.progress.baseline_manifest(
        baseline, item.dataset, command.bucket, command.prefix
    )
    for old, new in transitions:
        diff = DownloadDiff(**common, from_release=old, release=new)
        async with _source_operation(services):
            archived = await services.downloads[item.dataset].diff.fn(diff)
        refs = tuple([await services.progress.reference(path) for path in archived.manifests])
        transition = SyncTransition(start=old, release=new, objects=refs)
        item.archived.append(transition)
        manifests = [await services.progress.read(ref, DatasetManifest) for ref in refs]
        guard_diff(
            base, item.dataset, sum(entry.size for part in manifests for entry in part.entries)
        )
        async with services.load_lease():
            await services.loads.diff.fn(LoadDiff(**common, from_release=old, release=new))
        item.loaded.append(transition)
        item.final = new


def _incomplete(result: SyncResult) -> None:
    get_run_logger().error("S2 sync partial results: %s", result.model_dump_json())
    raise SyncIncomplete(result)
