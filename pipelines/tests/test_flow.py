import asyncio
import base64
from pathlib import Path

import pytest
from prefect.client.orchestration import get_client
from prefect.client.schemas.sorting import FlowRunSort
from prefect.settings import PREFECT_RESULTS_LOCAL_STORAGE_PATH, temporary_settings

from openknowledge_contracts.manifests import DatasetManifest
from openknowledge_core.ingestion import (
    DownloadDatasetService,
    DownloadShard,
    JobOutcome,
    LoadDataset,
    LoadDatasetService,
    ShardJob,
)
from openknowledge_core.ingestion.errors import (
    BatchRejected,
    BatchUnavailable,
    ContractViolation,
    GzipInvalid,
    ManifestIncomplete,
    SourceRejected,
    TableUnavailable,
)
from openknowledge_core.ingestion.families import ArchiveFamily, archive_family
from openknowledge_pipelines.flows.s2_batch_download import s2_batch_download_flows
from openknowledge_pipelines.flows.s2_download import S2DownloadFlow, s2_download_flow
from openknowledge_pipelines.flows.s2_load import S2LoadFlow, s2_load_flows
from tests.fakes.ports import (
    COMMAND,
    DIFF,
    NEXT,
    PAPERS,
    PRODUCER,
    RELEASE,
    FakeApplied,
    FakeSource,
    FakeStore,
    FakeTables,
    clock,
)

pytestmark = pytest.mark.usefixtures("prefect_server")


def build(service: DownloadDatasetService) -> S2DownloadFlow:
    return s2_download_flow(service, task_retries=2, retry_delay_seconds=0)


async def test_the_flow_archives_the_dataset_then_writes_its_manifest(
    service: DownloadDatasetService, source: FakeSource, store: FakeStore
) -> None:
    result = await build(service)(COMMAND)
    assert (result.shards, result.created, result.recovered, result.verified) == (2, 2, 0, 0)
    manifest = DatasetManifest.from_json(store.data(result.manifest))
    assert len(manifest.entries) == 2
    rerun = DownloadDatasetService(
        catalog=source, downloads=source, store=store, producer=PRODUCER, clock=clock
    )
    again = await build(rerun)(COMMAND)
    assert (again.created, again.verified, again.manifest) == (0, 2, result.manifest)


async def test_transient_errors_are_retried(
    service: DownloadDatasetService, source: FakeSource
) -> None:
    source.unavailable = ["part-0.gz"]
    result = await build(service)(COMMAND)
    assert result.created == 2
    assert source.opened.count("part-0.gz") == 2


async def test_integrity_errors_are_not_retried(
    service: DownloadDatasetService, source: FakeSource
) -> None:
    source.files["papers"]["part-0.gz"] = PAPERS["part-0.gz"][:-4]
    with pytest.raises(GzipInvalid):
        await build(service)(COMMAND)
    assert source.opened.count("part-0.gz") == 1


async def test_only_the_command_reaches_prefect(service: DownloadDatasetService) -> None:
    await build(service)(COMMAND)
    async with get_client() as client:
        [flow_run] = await client.read_flow_runs(sort=FlowRunSort.START_TIME_DESC, limit=1)
    assert flow_run.parameters == {"command": COMMAND.model_dump(mode="json")}


async def test_a_failing_run_persists_no_signed_link(
    service: DownloadDatasetService, source: FakeSource, tmp_path: Path
) -> None:
    """Prefect writes a failed flow's exception once, and that record must hold no signed link."""
    source.expired_listings = {1}
    source.files["papers"]["part-1.gz"] = PAPERS["part-1.gz"][:-4]
    with temporary_settings(updates={PREFECT_RESULTS_LOCAL_STORAGE_PATH: tmp_path}):
        with pytest.raises(GzipInvalid):
            await build(service)(COMMAND)
    [record] = await asyncio.to_thread(lambda: [path.read_bytes() for path in tmp_path.iterdir()])
    pickled = base64.b64decode(record)
    assert b"GzipInvalid" in pickled  # the record is the pickled exception, so the checks see it
    for blob in (record, pickled):
        assert b"secret" not in blob and b"Signature" not in blob


LOAD = LoadDataset(**COMMAND.model_dump())


def load_flow(store: FakeStore, tables: FakeTables) -> S2LoadFlow:
    loader = LoadDatasetService(
        store=store, tables=tables, applied=FakeApplied(), producer=PRODUCER, clock=clock
    )
    return s2_load_flows(loader, task_retries=2, retry_delay_seconds=0).dataset


async def test_the_load_flow_loads_each_object_once(
    service: DownloadDatasetService, store: FakeStore
) -> None:
    await build(service)(COMMAND)
    tables = FakeTables(store)
    first = await load_flow(store, tables)(LOAD)
    again = await load_flow(store, tables)(LOAD)
    assert (first.loaded, again.already_loaded, len(tables.loads)) == (2, 2, 2)


async def test_only_transient_load_errors_are_retried(
    service: DownloadDatasetService, store: FakeStore
) -> None:
    await build(service)(COMMAND)
    tables = FakeTables(store)
    tables.failures = [TableUnavailable("try again"), ContractViolation("bad record")]
    with pytest.raises(ContractViolation):
        await load_flow(store, tables)(LOAD)
    assert tables.loads == ["part-0.gz", "part-0.gz"]


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


async def test_batch_rounds_run_only_unfinished_shards(
    service: DownloadDatasetService, store: FakeStore
) -> None:
    jobs = FakeJobs(service, failing={1})
    flows = s2_batch_download_flows(service, jobs, rounds=3, task_retries=0, retry_delay_seconds=0)
    batch = flows.dataset
    result = await batch(COMMAND)
    assert jobs.submitted == [(0, 1), (1,)]
    assert jobs.jobs[0].operation_id != jobs.jobs[1].operation_id
    assert (result.shards, result.jobs) == (2, ("job-1", "job-2"))
    assert len(DatasetManifest.from_json(store.data(result.manifest)).entries) == 2
    again = await batch(COMMAND)
    assert (jobs.submitted[2:], again.jobs) == ([], ())


async def test_a_diff_runs_a_job_for_each_part_with_files(
    service: DownloadDatasetService, source: FakeSource, store: FakeStore
) -> None:
    source.add_diff("papers", RELEASE, NEXT, dict(PAPERS), {})
    jobs = FakeJobs(service, failing=set())
    flows = s2_batch_download_flows(service, jobs, rounds=1, task_retries=0, retry_delay_seconds=0)
    result = await flows.diff(DIFF)
    assert (jobs.parts, jobs.submitted) == (["updates"], [(0, 1)])
    assert (result.updates, result.deletes, result.jobs) == (2, 0, ("job-1",))
    [manifest] = result.manifests
    assert manifest.name == "s2/diffs/2026-10-06/2026-10-13/papers/updates/manifest.json"
    assert len(DatasetManifest.from_json(store.data(manifest)).entries) == 2


async def test_a_round_first_deletes_queued_download_jobs_and_waits_for_running_ones(
    service: DownloadDatasetService,
) -> None:
    jobs = FakeJobs(service, failing=set())
    jobs.left = {"queued": True, "running": False}
    jobs.outcomes |= {"queued": JobOutcome.FAILED, "running": JobOutcome.FAILED}
    flows = s2_batch_download_flows(service, jobs, rounds=1, task_retries=0, retry_delay_seconds=0)
    await flows.dataset(COMMAND)
    assert jobs.deleted == ["queued"]
    assert jobs.waited == ["queued", "running", "job-1"]


async def test_a_lost_submission_reply_retries_the_same_operation(
    service: DownloadDatasetService,
) -> None:
    class LostReplyJobs(FakeJobs):
        def __init__(self) -> None:
            super().__init__(service, failing=set())
            self.attempts: list[ShardJob] = []

        async def submit(self, job: ShardJob) -> str:
            self.attempts.append(job)
            if len(self.attempts) > 1:
                return "job-1"
            await super().submit(job)
            raise BatchUnavailable("the accepted submission's reply was lost")

    jobs = LostReplyJobs()
    flows = s2_batch_download_flows(service, jobs, rounds=1, task_retries=1, retry_delay_seconds=0)
    result = await flows.dataset(COMMAND)
    first, retry = jobs.attempts
    assert first == retry
    assert (jobs.submitted, result.jobs) == ([(0, 1)], ("job-1",))


@pytest.mark.parametrize("dataset", ["s2orc_v2", "embeddings-specter_v2"])
async def test_archive_families_resume_base_and_both_diff_parts(
    dataset: str, service: DownloadDatasetService, source: FakeSource, store: FakeStore
) -> None:
    # Synthetic gzip fixtures exercise the archive protocol, not either dataset's real schema.
    for name, data in PAPERS.items():
        source.add(dataset, name, data)
    source.add_diff(dataset, RELEASE, NEXT, dict(PAPERS), dict(PAPERS))
    family = archive_family(dataset)
    jobs = FakeJobs(service, failing={1})
    flows = s2_batch_download_flows(
        service, jobs, family=family, rounds=1, task_retries=0, retry_delay_seconds=0
    )
    base = COMMAND.model_copy(update={"dataset": dataset})
    with pytest.raises(ManifestIncomplete):
        await flows.dataset(base)
    result = await flows.dataset(base)
    assert jobs.submitted == [(0, 1), (1,)]
    assert jobs.family == family
    assert result.manifest.name == f"s2/{RELEASE}/{dataset}/manifest.json"
    assert len(DatasetManifest.from_json(store.data(result.manifest)).entries) == 2

    jobs.failing = {1}
    diff = DIFF.model_copy(update={"dataset": dataset})
    with pytest.raises(ManifestIncomplete):
        await flows.diff(diff)
    resumed = await flows.diff(diff)
    assert (resumed.updates, resumed.deletes) == (2, 2)
    assert [job.part for job in jobs.jobs[-2:]] == ["updates", "deletes"]
    assert [job.shards for job in jobs.jobs[-2:]] == [(1,), (0, 1)]
    assert len(resumed.manifests) == 2
    assert (await flows.diff(diff)).jobs == ()


@pytest.mark.parametrize("dataset", ["s2orc_v2", "embeddings-specter_v2"])
async def test_missing_diff_does_not_create_manifest_or_submit_jobs(
    dataset: str, service: DownloadDatasetService, store: FakeStore
) -> None:
    jobs = FakeJobs(service, failing=set())
    flows = s2_batch_download_flows(
        service,
        jobs,
        family=archive_family(dataset),
        rounds=1,
        task_retries=0,
        retry_delay_seconds=0,
    )
    before = dict(store.objects)
    with pytest.raises(SourceRejected, match="did not list exactly one diff"):
        await flows.diff(DIFF.model_copy(update={"dataset": dataset}))
    assert jobs.submitted == []
    assert all(path.endswith("/release.json") for path in set(store.objects) - set(before))


@pytest.mark.parametrize("diff", [False, True])
async def test_refused_inventory_never_cleans_up_or_pins(
    diff: bool, service: DownloadDatasetService, store: FakeStore
) -> None:
    class RefusedJobs(FakeJobs):
        async def running(self, family: ArchiveFamily = ArchiveFamily.CORE) -> dict[str, bool]:
            # The adapter has found an own-family queued job on one page, followed by a
            # foreign/legacy job on another. It refuses the whole inventory before returning.
            raise BatchRejected("foreign job must be drained")

    jobs = RefusedJobs(service, failing=set())
    flows = s2_batch_download_flows(
        service,
        jobs,
        family=ArchiveFamily.FULLTEXT,
        rounds=1,
        task_retries=0,
        retry_delay_seconds=0,
    )
    with pytest.raises(BatchRejected, match="must be drained"):
        if diff:
            await flows.diff(DIFF.model_copy(update={"dataset": "s2orc_v2"}))
        else:
            await flows.dataset(COMMAND.model_copy(update={"dataset": "s2orc_v2"}))
    assert (jobs.deleted, jobs.waited, jobs.submitted) == ([], [], [])
    assert not store.objects
