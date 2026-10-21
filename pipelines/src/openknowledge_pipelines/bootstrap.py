"""Wires adapters to the core ports for one run, as the only module that imports them."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from typing import TYPE_CHECKING

from openknowledge_adapters import batch, clickhouse, gcs, opensearch, semantic_scholar
from openknowledge_contracts.manifests import TRANSFORM_VERSION, Producer
from openknowledge_core.ingestion import (
    DownloadDatasetService,
    LoadDatasetService,
    ShardJobs,
    WatchdogService,
)
from openknowledge_core.ingestion.archive_progress import ArchiveProgress
from openknowledge_core.ingestion.consumers import ArchiveConsumer
from openknowledge_core.ingestion.coordinator import CORE_ORDER, LARGE_ORDER
from openknowledge_core.ingestion.families import ArchiveFamily
from openknowledge_core.ingestion.measurement import LoadObservation
from openknowledge_core.ingestion.ports import LoadTelemetry
from openknowledge_core.search import SearchIndexService
from openknowledge_pipelines.settings import (
    BatchSettings,
    CommonSettings,
    LoadSettings,
    PipelineSettings,
    SearchSettings,
    SyncSettings,
    WatchdogSettings,
)

if TYPE_CHECKING:
    from openknowledge_pipelines.flows.s2_batch_download import S2BatchDownloadFlows
    from openknowledge_pipelines.flows.s2_download import S2DownloadFlow
    from openknowledge_pipelines.flows.s2_load import S2LoadFlows
    from openknowledge_pipelines.flows.s2_sync import SyncServices
    from openknowledge_pipelines.flows.watchdog import WatchdogFlow


@asynccontextmanager
async def open_service(settings: PipelineSettings) -> AsyncGenerator[DownloadDatasetService, None]:
    s2_config = semantic_scholar.SemanticScholarConfig(
        api_key=settings.s2_api_key,
        api_url=settings.s2_api_url,
        request_interval_seconds=settings.s2_request_interval_seconds,
        attempts=settings.s2_attempts,
        backoff_seconds=settings.s2_backoff_seconds,
        timeout_seconds=settings.s2_timeout_seconds,
        download_connect_timeout_seconds=settings.download_connect_timeout_seconds,
        download_read_timeout_seconds=settings.download_read_timeout_seconds,
    )
    async with semantic_scholar.connect(s2_config) as s2, gcs.connect(_gcs(settings)) as store:
        yield DownloadDatasetService(
            catalog=s2.catalog,
            downloads=s2.downloads,
            store=store,
            producer=producer(settings),
            clock=lambda: datetime.now(UTC),
        )


@asynccontextmanager
async def open_load_service(settings: LoadSettings) -> AsyncGenerator[LoadDatasetService, None]:
    async with (
        gcs.connect(_gcs(settings)) as store,
        clickhouse.connect(_clickhouse(settings, settings.clickhouse_timeout_seconds)) as ch,
    ):
        yield LoadDatasetService(
            store=store,
            tables=ch.tables,
            applied=ch.applied,
            producer=producer(settings),
            clock=lambda: datetime.now(UTC),
        )


@asynccontextmanager
async def open_watchdog(settings: WatchdogSettings) -> AsyncGenerator[WatchdogService, None]:
    config = _clickhouse(settings, settings.watchdog_timeout_seconds)
    async with clickhouse.connect(config) as ch:
        yield WatchdogService(applied=ch.applied, clock=lambda: datetime.now(UTC))


@asynccontextmanager
async def open_search_service(settings: SearchSettings) -> AsyncGenerator[SearchIndexService]:
    async with (
        clickhouse.connect(_clickhouse(settings, settings.clickhouse_timeout_seconds)) as ch,
        opensearch.connect(_opensearch(settings)) as sink,
    ):
        yield SearchIndexService(source=ch.search, sink=sink, shards=settings.index_shards)


def _gcs(settings: CommonSettings) -> gcs.GcsConfig:
    return gcs.GcsConfig(
        request_timeout_seconds=settings.gcs_request_timeout_seconds,
        min_transfer_rate=settings.gcs_min_transfer_bytes_per_second,
        api_root=settings.gcs_api_root,
    )


def _clickhouse(settings: LoadSettings, timeout_seconds: float) -> clickhouse.ClickHouseConfig:
    return clickhouse.ClickHouseConfig(
        url=settings.clickhouse_url,
        user=settings.clickhouse_user,
        password=settings.clickhouse_password,
        raw_bucket=settings.clickhouse_raw_bucket,
        timeout_seconds=timeout_seconds,
    )


def _opensearch(settings: SearchSettings) -> opensearch.OpenSearchConfig:
    return opensearch.OpenSearchConfig(
        url=settings.opensearch_url,
        user=settings.opensearch_user,
        password=settings.opensearch_password,
        timeout_seconds=settings.opensearch_timeout_seconds,
    )


def producer(settings: CommonSettings) -> Producer:
    return Producer(
        transform="verbatim",
        transform_version=TRANSFORM_VERSION,
        software_version=version("openknowledge-pipelines"),
        image_digest=settings.image_digest,
    )


def download_flow(service: DownloadDatasetService, settings: PipelineSettings) -> S2DownloadFlow:
    # Imported here so that `okp s2 pin` and `okp s2 download-shard` never load Prefect.
    from openknowledge_pipelines.flows.s2_download import s2_download_flow

    return s2_download_flow(
        service,
        task_retries=settings.task_retries,
        retry_delay_seconds=settings.task_retry_delay_seconds,
    )


def load_flows(service: LoadDatasetService, settings: LoadSettings) -> S2LoadFlows:
    from openknowledge_pipelines.flows.s2_load import s2_load_flows

    return s2_load_flows(
        service,
        task_retries=settings.task_retries,
        retry_delay_seconds=settings.task_retry_delay_seconds,
    )


def watchdog_flow(service: WatchdogService, settings: WatchdogSettings) -> WatchdogFlow:
    from openknowledge_pipelines.flows import watchdog

    return watchdog.watchdog_flow(
        service,
        max_age=timedelta(days=settings.watchdog_max_age_days),
        task_retries=settings.task_retries,
        retry_delay_seconds=settings.task_retry_delay_seconds,
    )


@asynccontextmanager
async def open_batch_download(
    settings: BatchSettings,
) -> AsyncGenerator[tuple[DownloadDatasetService, ShardJobs], None]:
    config = batch.BatchConfig(
        project=settings.batch_project,
        region=settings.batch_region,
        service_account=settings.batch_service_account,
        network=settings.batch_network,
        subnetwork=settings.batch_subnetwork,
        image=f"{settings.batch_image_repository}@{settings.image_digest}",
        machine_type=settings.batch_machine_type,
        parallelism=settings.batch_parallelism,
        provisioning_model=settings.batch_provisioning_model,
        max_run_seconds=settings.batch_max_run_seconds,
        poll_seconds=settings.batch_poll_seconds,
        wait_timeout_seconds=settings.batch_wait_timeout_seconds,
        environment={"OKP_IMAGE_DIGEST": settings.image_digest},
        secrets={"S2_API_KEY": settings.batch_s2_key_secret},
    )
    async with open_service(settings) as service, batch.connect(config) as jobs:
        yield service, jobs


def batch_download_flows(
    service: DownloadDatasetService,
    jobs: ShardJobs,
    settings: BatchSettings,
    family: ArchiveFamily = ArchiveFamily.CORE,
) -> S2BatchDownloadFlows:
    from openknowledge_pipelines.flows.s2_batch_download import s2_batch_download_flows

    return s2_batch_download_flows(
        service,
        jobs,
        family=family,
        rounds=settings.task_retries + 1,
        task_retries=settings.task_retries,
        retry_delay_seconds=settings.task_retry_delay_seconds,
    )


@asynccontextmanager
async def open_archive_progress(
    settings: PipelineSettings,
) -> AsyncGenerator[ArchiveProgress, None]:
    async with open_service(settings) as service, gcs.connect(_gcs(settings)) as store:
        yield ArchiveProgress(store, service, producer(settings), lambda: datetime.now(UTC))


@asynccontextmanager
async def open_sync(
    settings: SyncSettings,
    s2_lease: Callable[[], AbstractAsyncContextManager[None]],
    load_lease: Callable[[], AbstractAsyncContextManager[None]],
) -> AsyncGenerator[SyncServices, None]:
    from openknowledge_core.ingestion.families import archive_family
    from openknowledge_pipelines.flows.s2_sync import SyncServices

    async with (
        open_batch_download(settings) as (service, jobs),
        gcs.connect(_gcs(settings)) as store,
        clickhouse.connect(_clickhouse(settings, settings.clickhouse_timeout_seconds)) as ch,
    ):
        progress = ArchiveProgress(store, service, producer(settings), lambda: datetime.now(UTC))
        loader = LoadDatasetService(
            store=store,
            tables=ch.tables,
            applied=ch.applied,
            producer=producer(settings),
            clock=lambda: datetime.now(UTC),
        )
        yield SyncServices(
            jobs=jobs,
            catalog=service.catalog,
            tables=ch.tables,
            applied=ch.applied,
            progress=progress,
            downloads={
                dataset: batch_download_flows(service, jobs, settings, archive_family(dataset))
                for dataset in (*CORE_ORDER, *LARGE_ORDER)
            },
            loads=load_flows(loader, settings),
            s2_lease=s2_lease,
            load_lease=load_lease,
        )


@asynccontextmanager
async def open_archive_consumer(settings: CommonSettings) -> AsyncGenerator[ArchiveConsumer]:
    async with gcs.connect(_gcs(settings)) as store:
        yield ArchiveConsumer(store)


@asynccontextmanager
async def open_measured_load(
    settings: LoadSettings,
) -> AsyncGenerator[tuple[LoadDatasetService, LoadObservation, LoadTelemetry]]:
    observation = LoadObservation()
    async with (
        gcs.connect(_gcs(settings)) as store,
        clickhouse.connect(_clickhouse(settings, settings.clickhouse_timeout_seconds)) as ch,
    ):
        yield (
            LoadDatasetService(
                store=store,
                tables=ch.tables,
                applied=ch.applied,
                producer=producer(settings),
                clock=lambda: datetime.now(UTC),
                observer=observation,
            ),
            observation,
            ch.telemetry,
        )
