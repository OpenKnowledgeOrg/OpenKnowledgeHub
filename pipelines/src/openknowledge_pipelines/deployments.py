"""The flows the control VM serves, each reading its settings from the run's environment."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager

from prefect import flow
from prefect import serve as prefect_serve
from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import DeploymentScheduleCreate, GlobalConcurrencyLimitCreate
from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy
from prefect.client.schemas.schedules import CronSchedule
from prefect.concurrency.asyncio import concurrency
from prefect.deployments.runner import RunnerDeployment
from prefect.exceptions import ObjectNotFound
from prefect.types.entrypoint import EntrypointType

from openknowledge_core.ingestion import (
    BatchDiffDownloadResult,
    BatchDownloadResult,
    DownloadDataset,
    DownloadDiff,
    LoadDataset,
    LoadDatasetResult,
    LoadDiff,
    LoadDiffResult,
    WatchdogResult,
)
from openknowledge_core.ingestion.coordinator import SyncResult
from openknowledge_core.ingestion.errors import IngestionError
from openknowledge_core.ingestion.families import ArchiveFamily, require_family
from openknowledge_pipelines import bootstrap
from openknowledge_pipelines.flows.s2_sync import run_sync
from openknowledge_pipelines.settings import (
    BatchSettings,
    LoadSettings,
    ScheduleSettings,
    SyncSettings,
    WatchdogSettings,
)

DEPLOYMENT = "control-vm"
# Keep the existing process budget. Busy downloads fail acquisition promptly to release slots
# for loads and the watchdog; deployment queues alone do not reserve runner capacity.
RUNS_AT_ONCE = 3
S2_LIMIT = "semantic-scholar"
LOAD_LIMIT = "clickhouse"
WATCHDOG_CRON = "0 */6 * * *"


def one_s2_user() -> AbstractAsyncContextManager[None]:
    # No per-family limits: the key and Batch task ceiling are shared. On lease loss, stop
    # submitting work. A later controller must inspect orphaned jobs before submitting.
    return concurrency(
        S2_LIMIT,
        strict=True,
        timeout_seconds=1,
        max_retries=0,
        raise_on_lease_renewal_failure=True,
    )


def one_clickhouse_loader() -> AbstractAsyncContextManager[None]:
    # A missing limit must fail. Keep a load running through a failed renewal so an insert
    # of hours is not interrupted.
    return concurrency(LOAD_LIMIT, strict=True, raise_on_lease_renewal_failure=False)


@flow(name="s2-load", persist_result=False)
async def s2_load(command: LoadDataset) -> LoadDatasetResult:
    settings = LoadSettings()  # pyright: ignore[reportCallIssue]  # fields come from the environment
    async with one_clickhouse_loader(), bootstrap.open_load_service(settings) as service:
        # fn runs the body in this flow run, so its tasks do not start a subflow.
        return await bootstrap.load_flows(service, settings).dataset.fn(command)


@flow(name="s2-load-diff", persist_result=False)
async def s2_load_diff(command: LoadDiff) -> LoadDiffResult:
    settings = LoadSettings()  # pyright: ignore[reportCallIssue]
    async with one_clickhouse_loader(), bootstrap.open_load_service(settings) as service:
        return await bootstrap.load_flows(service, settings).diff.fn(command)


async def _download(command: DownloadDataset, family: ArchiveFamily) -> BatchDownloadResult:
    require_family(command.dataset, family)
    settings = BatchSettings()  # pyright: ignore[reportCallIssue]
    async with one_s2_user(), bootstrap.open_batch_download(settings) as (service, jobs):
        result = await bootstrap.batch_download_flows(service, jobs, settings, family).dataset.fn(
            command
        )
        if family != ArchiveFamily.CORE:
            async with bootstrap.open_archive_progress(settings) as progress:
                await progress.finalize(command)
        return result


async def _download_diff(command: DownloadDiff, family: ArchiveFamily) -> BatchDiffDownloadResult:
    require_family(command.dataset, family)
    settings = BatchSettings()  # pyright: ignore[reportCallIssue]
    async with one_s2_user():
        predecessor = None
        if family != ArchiveFamily.CORE:
            baseline = settings.sync.archives.get(command.dataset) if settings.sync else None
            if baseline is None:
                raise IngestionError("manual archive diff requires OKP_SYNC archive baseline")
            if (baseline.base.dataset, baseline.base.bucket, baseline.base.prefix) != (
                command.dataset,
                command.bucket,
                command.prefix,
            ):
                raise IngestionError("manual diff baseline identifies another dataset/location")
            async with bootstrap.open_archive_progress(settings) as progress:
                predecessor, completed = await progress.manual_diff(
                    baseline.base, baseline.completion, command
                )
                if completed is not None:
                    return completed
        async with bootstrap.open_batch_download(settings) as (service, jobs):
            result = await bootstrap.batch_download_flows(service, jobs, settings, family).diff.fn(
                command
            )
        if family != ArchiveFamily.CORE:
            async with bootstrap.open_archive_progress(settings) as progress:
                await progress.finalize(command, predecessor)
        return result


@flow(name="s2-batch-download", persist_result=False)
async def s2_batch_download(command: DownloadDataset) -> BatchDownloadResult:
    return await _download(command, ArchiveFamily.CORE)


@flow(name="s2-batch-download-diff", persist_result=False)
async def s2_batch_download_diff(command: DownloadDiff) -> BatchDiffDownloadResult:
    return await _download_diff(command, ArchiveFamily.CORE)


@flow(name="s2-batch-download-fulltext", persist_result=False)
async def s2_batch_download_fulltext(command: DownloadDataset) -> BatchDownloadResult:
    return await _download(command, ArchiveFamily.FULLTEXT)


@flow(name="s2-batch-download-fulltext-diff", persist_result=False)
async def s2_batch_download_fulltext_diff(command: DownloadDiff) -> BatchDiffDownloadResult:
    return await _download_diff(command, ArchiveFamily.FULLTEXT)


@flow(name="s2-batch-download-embeddings", persist_result=False)
async def s2_batch_download_embeddings(command: DownloadDataset) -> BatchDownloadResult:
    return await _download(command, ArchiveFamily.EMBEDDINGS)


@flow(name="s2-batch-download-embeddings-diff", persist_result=False)
async def s2_batch_download_embeddings_diff(command: DownloadDiff) -> BatchDiffDownloadResult:
    return await _download_diff(command, ArchiveFamily.EMBEDDINGS)


@flow(name="s2-sync", persist_result=False, retries=0)
async def s2_sync() -> SyncResult:
    settings = SyncSettings()  # pyright: ignore[reportCallIssue]
    if settings.sync is None:
        raise IngestionError("OKP_SYNC must configure the coordinator bucket and baselines")
    async with bootstrap.open_sync(settings, one_s2_user, one_clickhouse_loader) as services:
        return await run_sync(settings.sync, services)


def sync_deployment() -> RunnerDeployment:
    settings = ScheduleSettings()
    return RunnerDeployment.from_flow(
        s2_sync,
        DEPLOYMENT,
        entrypoint_type=EntrypointType.MODULE_PATH,
        schedules=[
            DeploymentScheduleCreate(
                schedule=CronSchedule(cron="30 1 * * *", timezone="Europe/Zurich"),
                active=settings.sync_schedule_enabled,
            )
        ],
        concurrency_limit=ConcurrencyLimitConfig(
            limit=1,
            collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW,
        ),
    )


@flow(name="watchdog", persist_result=False)
async def watchdog() -> WatchdogResult:
    settings = WatchdogSettings()  # pyright: ignore[reportCallIssue]
    async with bootstrap.open_watchdog(settings) as service:
        return await bootstrap.watchdog_flow(service, settings).fn()


def ensure_limits() -> None:
    """Create the global limits the flows take, since a strict limit must exist before a run."""
    with get_client(sync_client=True) as client:
        for name in (S2_LIMIT, LOAD_LIMIT):
            try:
                client.read_global_concurrency_limit_by_name(name)
            except ObjectNotFound:
                client.create_global_concurrency_limit(
                    GlobalConcurrencyLimitCreate(name=name, limit=1)
                )


def serve() -> None:
    ensure_limits()
    prefect_serve(
        sync_deployment(),
        # One run of each load at a time, held in Prefect's queue, so a second run of the same load
        # holds no serve slot. LOAD_LIMIT keeps the two deployments apart, so no two served loads
        # hit ClickHouse at once; a load waiting on it does hold a slot.
        # The deployment limit gives a running load a lease that Prefect renews every 225 seconds,
        # up to three attempts per renewal, and cancels the run when all three fail. The global
        # limit's raise_on_lease_renewal_failure=False does not cover that lease.
        RunnerDeployment.from_flow(
            s2_load, DEPLOYMENT, entrypoint_type=EntrypointType.MODULE_PATH, concurrency_limit=1
        ),
        RunnerDeployment.from_flow(
            s2_load_diff,
            DEPLOYMENT,
            entrypoint_type=EntrypointType.MODULE_PATH,
            concurrency_limit=1,
        ),
        # Each download deployment queues its own runs. All six compete for S2_LIMIT; busy
        # runs fail promptly so they do not occupy runner slots while waiting globally.
        *(
            RunnerDeployment.from_flow(
                download,
                DEPLOYMENT,
                entrypoint_type=EntrypointType.MODULE_PATH,
                concurrency_limit=1,
            )
            for download in (
                s2_batch_download,
                s2_batch_download_diff,
                s2_batch_download_fulltext,
                s2_batch_download_fulltext_diff,
                s2_batch_download_embeddings,
                s2_batch_download_embeddings_diff,
            )
        ),
        # Every 6 hours, for a log alert; it needs no S2 key, so it takes no limit.
        RunnerDeployment.from_flow(
            watchdog,
            DEPLOYMENT,
            entrypoint_type=EntrypointType.MODULE_PATH,
            cron=WATCHDOG_CRON,
        ),
        limit=RUNS_AT_ONCE,
    )
