import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest
from aiohttp import web
from gcloud.aio.auth import Token

from openknowledge_adapters import batch
from openknowledge_adapters.batch.jobs import job_id, job_spec
from openknowledge_contracts.datasets import DiffPart
from openknowledge_core.ingestion.errors import BatchRejected, BatchTimedOut, BatchUnavailable
from openknowledge_core.ingestion.families import ArchiveFamily
from openknowledge_core.ingestion.models import JobOutcome, ShardJob

PARENT = "projects/p/locations/europe-west4"
SECRET = "projects/p/secrets/s2-api-key/versions/latest"
CONFIG = batch.BatchConfig(
    project="p",
    region="europe-west4",
    service_account="download@p.iam.gserviceaccount.com",
    network="projects/p/global/networks/n",
    subnetwork="projects/p/regions/europe-west4/subnetworks/download",
    image="repo/pipelines@sha256:" + "a" * 64,
    machine_type="e2-standard-2",
    parallelism=8,
    max_run_seconds=3600,
    poll_seconds=0.01,
    environment={"OKP_IMAGE_DIGEST": "sha256:" + "a" * 64},
    secrets={"S2_API_KEY": SECRET},
)
JOB = ShardJob(
    release_id="2026-10-06", dataset="s2orc_v2", bucket="raw", prefix="s2", shards=(3, 7)
)


class FakeToken:
    async def get(self) -> str:
        return "token"


@asynccontextmanager
async def serving(app: web.Application) -> AsyncGenerator[str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def batch_api(
    created: web.Response,
    states: list[int | str],
    pages: list[list[dict[str, Any]]] | None = None,
) -> AsyncGenerator[tuple[str, list[dict[str, Any]]]]:
    posted: list[dict[str, Any]] = []
    pages = pages or [[]]

    async def create(request: web.Request) -> web.Response:
        posted.append({**request.query, **await request.json(), **request.headers})
        return created

    async def get(request: web.Request) -> web.Response:
        state = states.pop(0) if len(states) > 1 else states[0]
        if isinstance(state, int):
            return web.Response(status=state)
        return web.json_response({"status": {"state": state}})

    async def delete(request: web.Request) -> web.Response:
        states[:] = [404]
        return web.json_response({})

    async def list_jobs(request: web.Request) -> web.Response:
        page = int(request.query.get("pageToken", "0"))
        last = page == len(pages) - 1
        return web.json_response(
            {"jobs": pages[page], "nextPageToken": "" if last else str(page + 1)}
        )

    app = web.Application()
    app.router.add_post(f"/{PARENT}/jobs", create)
    app.router.add_get(f"/{PARENT}/jobs", list_jobs)
    app.router.add_get(f"/{PARENT}/jobs/{{job_id}}", get)
    app.router.add_delete(f"/{PARENT}/jobs/{{job_id}}", delete)
    async with serving(app) as root:
        yield root, posted


async def test_a_job_runs_one_task_per_shard_and_is_polled_until_it_ends() -> None:
    async with (
        batch_api(web.json_response({}), ["QUEUED", 503, "RUNNING", "SUCCEEDED"]) as (root, posted),
        batch.connect(CONFIG, api_root=root, token=cast(Token, FakeToken())) as jobs,
    ):
        name = await jobs.submit(JOB)
        assert await jobs.wait(name) is JobOutcome.SUCCEEDED
    [request] = posted
    assert name == f"{PARENT}/jobs/{request['jobId']}"
    assert request["jobId"].startswith("s2-s2orc-v2-2026-10-06-")
    assert request["requestId"] == str(JOB.operation_id)
    assert request["Authorization"] == "Bearer token"
    [group] = request["taskGroups"]
    assert [env["variables"] for env in group["taskEnvironments"]] == [
        {"OKP_SHARD": "3"},
        {"OKP_SHARD": "7"},
    ]
    assert group["parallelism"] == 2  # at most one per shard
    one = job_spec(CONFIG.model_copy(update={"parallelism": 1}), JOB)
    assert one["taskGroups"][0]["parallelism"] == 1
    assert group["taskSpec"]["environment"]["secretVariables"] == {"S2_API_KEY": SECRET}
    [runnable] = group["taskSpec"]["runnables"]
    assert runnable["container"]["imageUri"] == CONFIG.image
    assert "--shard" not in runnable["container"]["commands"]
    policy = request["allocationPolicy"]
    assert policy["network"]["networkInterfaces"][0]["subnetwork"] == CONFIG.subnetwork
    assert policy["instances"][0]["policy"]["provisioningModel"] == "SPOT"


async def test_running_names_download_jobs_that_have_not_ended_on_every_page() -> None:
    def job(name: str, state: str, labels: dict[str, str]) -> dict[str, Any]:
        return {"name": name, "labels": labels, "status": {"state": state}}

    download = {"operation-id": "1", "archive-family": "core"}
    pages = [
        [
            job("a", "RUNNING", download),
            job("b", "SUCCEEDED", download),
            job("legacy", "SUCCEEDED", {"operation-id": "old"}),
            job("foreign", "FAILED", {"operation-id": "f", "archive-family": "embeddings"}),
        ],
        [
            job("c", "RUNNING", {}),
            job("d", "QUEUED", download),
            job("legacy-running", "RUNNING", {"operation-id": "old", "dataset": "papers"}),
            job("legacy-queued", "QUEUED", {"operation-id": "old2", "dataset": "authors"}),
        ],
    ]
    async with (
        batch_api(web.json_response({}), [], pages) as (root, _),
        batch.connect(CONFIG, api_root=root, token=cast(Token, FakeToken())) as jobs,
    ):
        assert await jobs.running() == {
            "a": False,
            "d": True,
            "legacy-running": False,
            "legacy-queued": True,
        }


async def test_a_deleted_job_ends_its_wait() -> None:
    name = f"{PARENT}/jobs/j"
    async with (
        batch_api(web.json_response({}), ["QUEUED"]) as (root, _),
        batch.connect(CONFIG, api_root=root, token=cast(Token, FakeToken())) as jobs,
    ):
        await jobs.delete(name)
        assert await asyncio.wait_for(jobs.wait(name), timeout=1) is JobOutcome.FAILED


async def test_a_refused_job_says_why() -> None:
    refused = web.json_response({"error": {"message": "actAs denied"}}, status=403)
    async with (
        batch_api(refused, []) as (root, _),
        batch.connect(CONFIG, api_root=root, token=cast(Token, FakeToken())) as jobs,
    ):
        with pytest.raises(BatchRejected, match="HTTP 403: actAs denied"):
            await jobs.submit(JOB)


async def test_a_wait_is_bounded_even_while_batch_is_unreachable() -> None:
    config = CONFIG.model_copy(update={"wait_timeout_seconds": 0.05, "poll_seconds": 3600})
    async with (
        batch_api(web.json_response({}), [503]) as (root, _),
        batch.connect(config, api_root=root, token=cast(Token, FakeToken())) as jobs,
    ):
        with pytest.raises(BatchTimedOut, match=f"{PARENT}/jobs/j.*may still be running"):
            await asyncio.wait_for(jobs.wait(f"{PARENT}/jobs/j"), timeout=1)


def test_a_diff_part_names_the_release_it_starts_from_and_its_part() -> None:
    job = JOB.model_copy(update={"from_release": "2026-09-29", "part": "deletes"})
    spec = batch.job_spec(CONFIG, job)
    command = spec["taskGroups"][0]["taskSpec"]["runnables"][0]["container"]["commands"]
    assert command[-4:] == ["--from-release", "2026-09-29", "--part", "deletes"]
    assert spec["labels"]["part"] == "deletes"
    assert job_id(job).startswith("s2-s2orc-v2-2026-10-06-deletes-")


@pytest.mark.parametrize("replay", [False, True])
async def test_a_lost_create_reply_resumes_exactly_one_remote_job(replay: bool) -> None:
    posted: list[dict[str, str]] = []
    remote: dict[str, Any] = {}

    async def create(request: web.Request) -> web.Response:
        posted.append(dict(request.query))
        id_ = request.query["jobId"]
        if id_ in remote:
            if replay:
                return web.json_response(remote[id_])
            return web.json_response({"error": {"message": "already exists"}}, status=409)
        remote[id_] = await request.json()
        # Batch created the job, but its reply never arrives intact.
        return web.Response(text="{", content_type="application/json")

    app = web.Application()
    app.router.add_post(f"/{PARENT}/jobs", create)
    async with (
        serving(app) as root,
        batch.connect(CONFIG, api_root=root, token=cast(Token, FakeToken())) as jobs,
    ):
        with pytest.raises(BatchUnavailable, match=f"{PARENT}/jobs/{job_id(JOB)}.*may be running"):
            await jobs.submit(JOB)
        assert await jobs.submit(JOB) == f"{PARENT}/jobs/{job_id(JOB)}"
    assert len(remote) == 1
    assert posted == [{"jobId": job_id(JOB), "requestId": str(JOB.operation_id)}] * 2


@pytest.mark.parametrize(
    "family,foreign,state,dataset",
    [
        ("core", "fulltext", "QUEUED", "papers"),
        ("fulltext", "embeddings", "RUNNING", "papers"),
        ("embeddings", None, "QUEUED", None),
        ("fulltext", None, "RUNNING", "papers"),
        ("core", None, "QUEUED", "unknown"),
        ("core", None, "RUNNING", "s2orc-v2"),
        ("core", None, "QUEUED", None),
    ],
)
async def test_foreign_or_legacy_jobs_refuse_cleanup_across_pages(
    family: str, foreign: str | None, state: str, dataset: str | None
) -> None:
    own = {"operation-id": "own", "archive-family": family}
    labels = {"operation-id": "other"}
    if dataset is not None:
        labels["dataset"] = dataset
    if foreign is not None:
        labels["archive-family"] = foreign
    pages = [
        [{"name": "own", "labels": own, "status": {"state": "QUEUED"}}],
        [{"name": "other", "labels": labels, "status": {"state": state}}],
    ]
    async with (
        batch_api(web.json_response({}), [], pages) as (root, posted),
        batch.connect(CONFIG, api_root=root, token=cast(Token, FakeToken())) as jobs,
    ):
        with pytest.raises(BatchRejected, match="drain it to a terminal state"):
            await jobs.running(ArchiveFamily(family))
        assert posted == []


@pytest.mark.parametrize(
    "dataset,family,part",
    [
        ("papers", "core", None),
        ("s2orc_v2", "fulltext", "updates"),
        ("embeddings-specter_v2", "embeddings", "deletes"),
    ],
)
def test_base_and_diff_labels_and_task_budget(
    dataset: str, family: str, part: DiffPart | None
) -> None:
    job = ShardJob(
        release_id="2026-10-13",
        dataset=dataset,
        bucket="raw",
        prefix="s2",
        shards=tuple(range(20)),
        from_release="2026-10-06" if part else None,
        part=part,
    )
    spec = job_spec(CONFIG, job)
    assert spec["labels"]["archive-family"] == family
    assert spec["taskGroups"][0]["parallelism"] == 8


def test_batch_worker_context_identifies_the_admitted_operation() -> None:
    spec = job_spec(CONFIG, JOB)
    variables = spec["taskGroups"][0]["taskSpec"]["environment"]["variables"]
    assert variables == {**CONFIG.environment, "OKP_BATCH_OPERATION_ID": JOB.operation_id.hex}
