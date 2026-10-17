"""Managed entry points and the real Prefect shared-budget behavior (loopback test server)."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from prefect.concurrency.asyncio import ConcurrencySlotAcquisitionError

from openknowledge_core.ingestion import DownloadDatasetService, ShardJobs
from openknowledge_pipelines import bootstrap, deployments
from openknowledge_pipelines.settings import BatchSettings
from tests.fakes.ports import COMMAND, DIFF, FakeSource, FakeStore


@pytest.mark.parametrize("dataset", ["s2orc_v2", "embeddings-specter_v2", "s2orc", "unknown"])
async def test_core_mismatch_precedes_settings_and_clients(
    monkeypatch: pytest.MonkeyPatch, dataset: str
) -> None:
    monkeypatch.delenv("S2_API_KEY", raising=False)
    with pytest.raises(ValueError, match="belongs to|Unsupported archive dataset"):
        await deployments.s2_batch_download.fn(COMMAND.model_copy(update={"dataset": dataset}))
    with pytest.raises(ValueError, match="belongs to|Unsupported archive dataset"):
        await deployments.s2_batch_download_diff.fn(DIFF.model_copy(update={"dataset": dataset}))


async def test_archive_mismatch_precedes_client_construction() -> None:
    for base in (deployments.s2_batch_download_fulltext, deployments.s2_batch_download_embeddings):
        with pytest.raises(ValueError, match="belongs to core"):
            await base.fn(COMMAND)
    for diff in (
        deployments.s2_batch_download_fulltext_diff,
        deployments.s2_batch_download_embeddings_diff,
    ):
        with pytest.raises(ValueError, match="belongs to core"):
            await diff.fn(DIFF)


@pytest.mark.usefixtures("prefect_server")
async def test_managed_routes_share_budget_and_release_after_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Clients must never consume settings when the real shared budget is busy.
    monkeypatch.setattr(deployments, "BatchSettings", BatchSettings.model_construct)

    @asynccontextmanager
    async def forbidden(
        settings: BatchSettings,
    ) -> AsyncGenerator[tuple[DownloadDatasetService, ShardJobs]]:
        pytest.fail("busy controllers must not open clients")
        yield  # pragma: no cover

    monkeypatch.setattr(bootstrap, "open_batch_download", forbidden)
    deployments.ensure_limits()
    async with deployments.one_s2_user():
        with pytest.raises(ConcurrencySlotAcquisitionError, match="semantic-scholar"):
            await deployments.s2_batch_download_fulltext.fn(
                COMMAND.model_copy(update={"dataset": "s2orc_v2"})
            )
        with pytest.raises(ConcurrencySlotAcquisitionError, match="semantic-scholar"):
            await deployments.s2_batch_download_embeddings_diff.fn(
                DIFF.model_copy(update={"dataset": "embeddings-specter_v2"})
            )
    with pytest.raises(RuntimeError, match="job wait failed"):
        async with deployments.one_s2_user():
            raise RuntimeError("job wait failed")
    async with deployments.one_s2_user():
        pass


@pytest.mark.usefixtures("prefect_server")
async def test_missing_shared_limit_refuses_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deployments, "S2_LIMIT", "missing-archive-budget-test")
    with pytest.raises(ConcurrencySlotAcquisitionError, match="must be created"):
        async with deployments.one_s2_user():
            pytest.fail("a missing budget must not permit work")


async def test_managed_manual_diff_retries_only_anchored_completions(
    monkeypatch: pytest.MonkeyPatch,
    service: DownloadDatasetService,
    source: FakeSource,
    store: FakeStore,
) -> None:
    from openknowledge_core.ingestion.archive_progress import (
        ArchiveCompletion,
        ArchiveProgress,
        completion_path,
    )
    from openknowledge_core.ingestion.coordinator import ArchiveBaseline, SyncCommand
    from openknowledge_core.ingestion.errors import IngestionError
    from tests.fakes.ports import NEXT, PAPERS, PRODUCER, clock

    progress = ArchiveProgress(store, service, PRODUCER, clock)
    base = COMMAND.model_copy(update={"dataset": "s2orc_v2"})
    diff = DIFF.model_copy(update={"dataset": base.dataset})
    source.add(base.dataset, "part.gz", PAPERS["part-0.gz"])
    pinned = await service.pin_dataset(base)
    await service.archive_shard(pinned, pinned.shards[0])
    anchor = await progress.finalize(base)
    config = SyncCommand(
        bucket=base.bucket,
        prefix=base.prefix,
        archives={base.dataset: ArchiveBaseline(base=base, completion=anchor)},
    )

    @asynccontextmanager
    async def lease() -> AsyncGenerator[None]:
        yield

    @asynccontextmanager
    async def opened(settings: BatchSettings) -> AsyncGenerator[ArchiveProgress]:
        yield progress

    @asynccontextmanager
    async def forbidden(settings: BatchSettings) -> AsyncGenerator[None]:
        pytest.fail("completed manual retries must not open Batch or submit jobs")
        yield

    monkeypatch.setattr(deployments, "one_s2_user", lease)
    monkeypatch.setattr(
        deployments, "BatchSettings", lambda: BatchSettings.model_construct(sync=config)
    )
    monkeypatch.setattr(bootstrap, "open_archive_progress", opened)
    monkeypatch.setattr(bootstrap, "open_batch_download", forbidden)
    predecessor = anchor
    completion_manifest = None
    for command in (diff, diff.model_copy(update={"from_release": NEXT, "release": "2026-10-20"})):
        source.add_diff(
            base.dataset,
            command.from_release,
            command.release,
            {"update.gz": PAPERS["part-0.gz"]},
            {},
        )
        pinned_diff = await service.pin_diff(command)
        for shard in pinned_diff.updates.shards:
            await service.archive_shard(pinned_diff.updates, shard)
        if command == diff:
            completion_manifest = pinned_diff.updates.manifest_path
        predecessor = await progress.finalize(command, predecessor)
        result = await deployments.s2_batch_download_fulltext_diff.fn(diff)
        assert (
            result.from_release,
            result.release_id,
            result.updates,
            result.deletes,
            result.jobs,
        ) == (
            diff.from_release,
            diff.release,
            1,
            0,
            (),
        )
        assert result.manifests == (completion_manifest,)

    baseline = config.archives[base.dataset]
    for field, value in (("dataset", "papers"), ("bucket", "other"), ("prefix", "other")):
        config.archives[base.dataset] = baseline.model_copy(
            update={"base": base.model_copy(update={field: value})}
        )
        with pytest.raises(IngestionError, match="another dataset/location"):
            await deployments.s2_batch_download_fulltext_diff.fn(diff)
    config.archives[base.dataset] = baseline

    # An existing completion detached from this anchor cannot make a retry successful.
    record = await progress.read(predecessor, ArchiveCompletion)
    store.put(completion_path(command), record.model_copy(update={"predecessor": anchor}).to_json())
    with pytest.raises(IngestionError, match="predecessor"):
        await deployments.s2_batch_download_fulltext_diff.fn(diff)
