"""The s2-batch-download flows: they pin and write manifests, and Batch archives the shards."""

from __future__ import annotations

from dataclasses import dataclass
from types import CoroutineType
from typing import Any

from prefect import Flow, flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from openknowledge_contracts.manifests import ObjectPath
from openknowledge_core.ingestion import (
    BatchDiffDownloadResult,
    BatchDownloadResult,
    DownloadDataset,
    DownloadDatasetService,
    DownloadDiff,
    JobOutcome,
    PinnedDataset,
    PinnedDiff,
    ShardJob,
    ShardJobs,
)
from openknowledge_core.ingestion.errors import ManifestIncomplete
from openknowledge_core.ingestion.families import ArchiveFamily, require_family
from openknowledge_pipelines.flows.s2_download import retry_transient

type S2BatchDownloadFlow = Flow[[DownloadDataset], CoroutineType[Any, Any, BatchDownloadResult]]
type S2BatchDiffDownloadFlow = Flow[
    [DownloadDiff], CoroutineType[Any, Any, BatchDiffDownloadResult]
]


@dataclass(frozen=True)
class S2BatchDownloadFlows:
    dataset: S2BatchDownloadFlow
    diff: S2BatchDiffDownloadFlow


def _flow_name(family: ArchiveFamily, *, diff: bool = False) -> str:
    suffix = "" if family == ArchiveFamily.CORE else f"-{family}"
    return f"s2-batch-download{suffix}{'-diff' if diff else ''}"


def s2_batch_download_flows(
    service: DownloadDatasetService,
    jobs: ShardJobs,
    *,
    family: ArchiveFamily = ArchiveFamily.CORE,
    rounds: int,
    task_retries: int,
    retry_delay_seconds: float,
) -> S2BatchDownloadFlows:
    @task(
        name="pin-dataset",
        retries=task_retries,
        retry_delay_seconds=retry_delay_seconds,
        retry_condition_fn=retry_transient,
        cache_policy=NO_CACHE,
        persist_result=False,
    )
    async def pin_dataset(command: DownloadDataset) -> PinnedDataset:
        return await service.pin_dataset(command)

    @task(
        name="pin-diff",
        retries=task_retries,
        retry_delay_seconds=retry_delay_seconds,
        retry_condition_fn=retry_transient,
        cache_policy=NO_CACHE,
        persist_result=False,
    )
    async def pin_diff(command: DownloadDiff) -> PinnedDiff:
        return await service.pin_diff(command)

    @task(
        name="find-unfinished",
        retries=task_retries,
        retry_delay_seconds=retry_delay_seconds,
        retry_condition_fn=retry_transient,
        cache_policy=NO_CACHE,
        persist_result=False,
    )
    async def find_unfinished(pinned: PinnedDataset) -> tuple[int, ...]:
        return await service.unfinished(pinned)

    @task(
        name="clear-earlier-jobs",
        retries=task_retries,
        retry_delay_seconds=retry_delay_seconds,
        retry_condition_fn=retry_transient,
        cache_policy=NO_CACHE,
        persist_result=False,
    )
    async def clear_earlier_jobs() -> None:
        """Not part of run-job, whose retry could find its own queued job and delete it."""
        logger = get_run_logger()
        for earlier, queued in (await jobs.running(family)).items():
            if queued:
                logger.info("Deleting %s, a download job that has not started", earlier)
                await jobs.delete(earlier)
            else:
                logger.info("Waiting for %s, a download job that has not ended", earlier)
            await jobs.wait(earlier)

    @task(
        name="run-job",
        task_run_name="round {round_}",
        retries=task_retries,
        retry_delay_seconds=retry_delay_seconds,
        retry_condition_fn=retry_transient,
        cache_policy=NO_CACHE,
        persist_result=False,
    )
    async def run_job(job: ShardJob, round_: int) -> str:
        logger = get_run_logger()
        logger.info("Submitting Batch operation %s", job.operation_id)
        name = await jobs.submit(job)
        logger.info("%s archives shards %s", name, ", ".join(map(str, job.shards)))
        if await jobs.wait(name) is JobOutcome.FAILED:
            logger.warning("%s failed; the next round runs the shards still unfinished", name)
        return name

    @task(
        name="write-manifest",
        retries=task_retries,
        retry_delay_seconds=retry_delay_seconds,
        retry_condition_fn=retry_transient,
        cache_policy=NO_CACHE,
        persist_result=False,
    )
    async def write_manifest(pinned: PinnedDataset) -> ObjectPath:
        try:
            return await service.write_manifest(pinned)
        except ManifestIncomplete as exc:
            deployment = _flow_name(family, diff=pinned.part is not None)
            raise ManifestIncomplete(
                f"{exc}. Start a new {deployment}/control-vm run with the same parameters "
                "to archive the remaining shards"
            ) from None

    async def archive(pinned: PinnedDataset) -> tuple[list[str], ObjectPath]:
        names: list[str] = []
        for round_ in range(1, rounds + 1):
            await clear_earlier_jobs()
            shards = await find_unfinished(pinned)
            if not shards:
                break
            job = ShardJob(
                release_id=pinned.release_id,
                dataset=pinned.dataset,
                bucket=pinned.bucket,
                prefix=pinned.prefix,
                shards=shards,
                from_release=pinned.from_release,
                part=pinned.part,
            )
            names.append(await run_job(job, round_))
        return names, await write_manifest(pinned)

    @flow(name=_flow_name(family), persist_result=False)
    async def s2_batch_download(command: DownloadDataset) -> BatchDownloadResult:
        require_family(command.dataset, family)
        await clear_earlier_jobs()
        pinned = await pin_dataset(command)
        names, manifest = await archive(pinned)
        return BatchDownloadResult(
            release_id=pinned.release_id,
            dataset=pinned.dataset,
            shards=len(pinned.shards),
            jobs=tuple(names),
            manifest=manifest,
        )

    @flow(name=_flow_name(family, diff=True), persist_result=False)
    async def s2_batch_download_diff(command: DownloadDiff) -> BatchDiffDownloadResult:
        require_family(command.dataset, family)
        await clear_earlier_jobs()
        pinned = await pin_diff(command)
        names: list[str] = []
        manifests: list[ObjectPath] = []
        for part in (pinned.updates, pinned.deletes):
            if part.shards:
                part_names, manifest = await archive(part)
                names += part_names
                manifests.append(manifest)
        return BatchDiffDownloadResult(
            from_release=command.from_release,
            release_id=command.release,
            dataset=command.dataset,
            updates=len(pinned.updates.shards),
            deletes=len(pinned.deletes.shards),
            jobs=tuple(names),
            manifests=tuple(manifests),
        )

    return S2BatchDownloadFlows(dataset=s2_batch_download, diff=s2_batch_download_diff)
