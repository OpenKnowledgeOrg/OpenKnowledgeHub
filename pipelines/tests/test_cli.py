import json
import logging
import os
import socket
import subprocess
import sys
from collections.abc import AsyncGenerator, Awaitable
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner
from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import DeploymentScheduleCreate
from prefect.client.schemas.objects import State
from prefect.client.schemas.schedules import CronSchedule
from prefect.context import use_profile
from prefect.deployments.runner import RunnerDeployment
from prefect.flows import load_flow_from_entrypoint
from prefect.settings import PREFECT_API_URL, Profile, temporary_settings

from openknowledge_core.ingestion import (
    DownloadDataset,
    DownloadDatasetService,
    DownloadDiff,
    LoadDatasetService,
    WatchdogResult,
    WatchdogService,
)
from openknowledge_core.search import SearchIndexService, SourceRows
from openknowledge_pipelines import bootstrap, cli, deployments
from openknowledge_pipelines.settings import (
    LoadSettings,
    PipelineSettings,
    SearchSettings,
    WatchdogSettings,
)
from tests.fakes.ports import (
    PAPERS,
    PRODUCER,
    RELEASE,
    FakeApplied,
    FakeSearchSink,
    FakeSearchSource,
    FakeSource,
    FakeStore,
    FakeTables,
    applied_release,
    clock,
    paper_row,
)

ROOT = Path(__file__).resolve().parents[2]
TARGET = ["--bucket", "raw", "--prefix", "s2"]
LOAD_ENV = {
    "OKP_CLICKHOUSE_URL": "http://clickhouse:8123",
    "CLICKHOUSE_PASSWORD": "dev",
    "OKP_CLICKHOUSE_RAW_BUCKET": "raw",
}
SEARCH_ENV = {
    "OKP_OPENSEARCH_URL": "http://opensearch:9200",
    "OPENSEARCH_PASSWORD": "dev",
    "OKP_INDEX_SHARDS": "2",
}
BATCH_ENV = {
    "S2_API_KEY": "k",
    "OKP_IMAGE_DIGEST": "sha256:" + "a" * 64,
    "OKP_BATCH_PROJECT": "p",
    "OKP_BATCH_REGION": "europe-west4",
    "OKP_BATCH_SERVICE_ACCOUNT": "download@p.iam.gserviceaccount.com",
    "OKP_BATCH_NETWORK": "n",
    "OKP_BATCH_SUBNETWORK": "s",
    "OKP_BATCH_IMAGE_REPOSITORY": "repo/pipelines",
    "OKP_BATCH_S2_KEY_SECRET": "projects/p/secrets/s2-api-key/versions/latest",
}


def okp(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(cli.main, ["s2", *args])
    return result.exit_code, result.output


@pytest.mark.usefixtures("wired")
def test_pin_prints_the_explicit_release() -> None:
    code, output = okp("pin", *TARGET)
    assert code == 0, output
    assert json.loads(output) == {
        "release_id": "2026-10-06",
        "record": {"bucket": "raw", "name": "s2/2026-10-06/release.json"},
    }


@pytest.mark.usefixtures("wired", "prefect_server")
def test_download_runs_the_flow_and_prints_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in BATCH_ENV:
        if key != "S2_API_KEY":
            monkeypatch.delenv(key, raising=False)
    code, output = okp("download", "--release", "2026-10-06", "--dataset", "papers", *TARGET)
    assert code == 0, output
    result = json.loads(output.splitlines()[-1])
    assert list(result) == [
        "release_id",
        "dataset",
        "shards",
        "created",
        "recovered",
        "verified",
        "inventory",
        "manifest",
    ]
    assert (result["shards"], result["created"]) == (2, 2)


@pytest.mark.usefixtures("wired")
def test_download_shard_archives_one_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OKP_SHARD", "1")
    code, output = okp("download-shard", "--release", "2026-10-06", "--dataset", "papers", *TARGET)
    assert code == 0, output
    result = json.loads(output)
    assert (result["index"], result["outcome"]) == (1, "created")
    assert result["receipt"]["name"] == "s2/2026-10-06/papers/part-1.gz.receipt.json"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["download", "--dataset", "papers", *TARGET], "Missing option '--release'"),
        (
            ["download", "--release", "latest", "--dataset", "papers", *TARGET],
            "--release must look like 2026-10-06",
        ),
        (
            ["download", "--release", "2026-10-06", "--dataset", "Papers", *TARGET],
            "--dataset must look like publication-venues",
        ),
        (
            ["download-shard", "--release", "2026-10-06", "--dataset", "papers", *TARGET],
            "Missing option '--shard'",
        ),
        (["pin", "--bucket", "Raw"], "--bucket must be a bucket name"),
        (["pin", "--bucket", "raw", "--prefix", "a//b"], "--prefix must look like"),
        (
            ["download-shard", "--release", "2026-10-06", "--dataset", "papers", "--shard", "-1"]
            + TARGET,
            "--shard must be a whole number like 0",
        ),
        (
            ["download-shard", "--release", "2026-10-06", "--dataset", "papers", "--shard", "x"]
            + TARGET,
            "--shard must be a whole number like 0",
        ),
        (
            ["download-shard", "--release", "2026-10-06", "--dataset", "papers", "--shard", "0"]
            + ["--part", "updates", *TARGET],
            "a diff file needs both the release it starts from and its part",
        ),
    ],
)
@pytest.mark.usefixtures("wired")
def test_bad_options_exit_2(args: list[str], message: str) -> None:
    code, output = okp(*args)
    assert code == 2
    assert message in output


@pytest.mark.usefixtures("prefect_server")
def test_load_runs_the_flow_without_an_s2_key(
    monkeypatch: pytest.MonkeyPatch, wired: DownloadDatasetService, store: FakeStore
) -> None:
    assert okp("download", "--release", "2026-10-06", "--dataset", "papers", *TARGET)[0] == 0

    @asynccontextmanager
    async def open_load_service(settings: LoadSettings) -> AsyncGenerator[LoadDatasetService]:
        yield LoadDatasetService(
            store=store,
            tables=FakeTables(store),
            applied=FakeApplied(),
            producer=PRODUCER,
            clock=clock,
        )

    monkeypatch.setattr(bootstrap, "open_load_service", open_load_service)
    monkeypatch.delenv("S2_API_KEY")
    for name, value in LOAD_ENV.items():
        monkeypatch.setenv(name, value)
    code, output = okp("load", "--release", "2026-10-06", "--dataset", "papers", *TARGET)
    assert code == 0, output
    assert json.loads(output.splitlines()[-1])["loaded"] == 2


@pytest.mark.usefixtures("prefect_server")
def test_serve_offers_the_load_flow_by_module_path(monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[RunnerDeployment] = []

    def fake_serve(*found: RunnerDeployment, limit: int) -> None:
        assert limit == 3
        served.extend(found)

    monkeypatch.setattr(deployments, "prefect_serve", fake_serve)
    for name, value in {**LOAD_ENV, **BATCH_ENV}.items():
        monkeypatch.setenv(name, value)
    result = CliRunner().invoke(cli.main, ["serve"])
    assert result.exit_code == 0, result.output
    assert [load_flow_from_entrypoint(str(found.entrypoint)) for found in served] == [
        deployments.s2_sync,
        deployments.s2_load,
        deployments.s2_load_diff,
        deployments.s2_batch_download,
        deployments.s2_batch_download_diff,
        deployments.s2_batch_download_fulltext,
        deployments.s2_batch_download_fulltext_diff,
        deployments.s2_batch_download_embeddings,
        deployments.s2_batch_download_embeddings_diff,
        deployments.watchdog,
    ]
    assert [found.concurrency_limit for found in served] == [1] * 9 + [None]
    _coordinator, *unscheduled, watchdog = served
    assert [found.schedules for found in unscheduled] == [[]] * 8
    [scheduled] = cast("list[DeploymentScheduleCreate]", watchdog.schedules)
    assert isinstance(scheduled.schedule, CronSchedule)
    # from_flow stores no timezone for a cron, and the server reads None as UTC.
    assert (scheduled.schedule.cron, scheduled.schedule.timezone, scheduled.active) == (
        "0 */6 * * *",
        None,
        True,
    )
    with get_client(sync_client=True) as client:
        for name in (deployments.S2_LIMIT, deployments.LOAD_LIMIT):
            assert client.read_global_concurrency_limit_by_name(name).limit == 1


@pytest.mark.usefixtures("prefect_server")
async def test_the_watchdog_deployment_runs_over_the_wired_service(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeApplied()
    fake.records.append(applied_release("papers", RELEASE, timedelta(days=4)))

    @asynccontextmanager
    async def open_watchdog(settings: WatchdogSettings) -> AsyncGenerator[WatchdogService]:
        yield WatchdogService(applied=fake, clock=clock)

    monkeypatch.setattr(bootstrap, "open_watchdog", open_watchdog)
    monkeypatch.delenv("S2_API_KEY", raising=False)
    for name, value in LOAD_ENV.items():
        monkeypatch.setenv(name, value)
    caplog.set_level(logging.INFO)
    # Flow is invariant in its return type, so pyright cannot pick the awaitable overload here.
    state = await cast("Awaitable[State[WatchdogResult]]", deployments.watchdog(return_state=True))
    assert state.is_completed()
    result = await state.aresult()
    assert (result.stale, len(result.never_loaded)) == ((), 5)
    assert "watchdog checked 6 tables: 0 older than 14 days, 5 never loaded" in caplog.messages


def test_a_missing_key_fails_before_anything_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def must_not_open(settings: PipelineSettings) -> AsyncGenerator[DownloadDatasetService]:
        pytest.fail("no service may open without a key")
        yield  # pragma: no cover

    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setattr(bootstrap, "open_service", must_not_open)
    code, output = okp("pin", *TARGET)
    assert code == 1
    assert "S2_API_KEY is not set" in output


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OKP_S2_ATTEMPTS", "many-topsecret"),
        ("OKP_IMAGE_DIGEST", "sha256:topsecret"),
        ("S2_API_KEY", "topsecret\u00a0"),
    ],
)
@pytest.mark.usefixtures("wired")
def test_a_bad_setting_is_named_without_its_value(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    code, output = okp("pin", *TARGET)
    assert code == 1
    assert name in output and "topsecret" not in output and "Value error" not in output


@pytest.mark.usefixtures("wired")
def test_an_option_error_names_the_option_once_with_a_valid_value() -> None:
    code, output = okp("pin", "--release", "2026-1-6", *TARGET)
    assert code == 2
    assert output.count("--release") == 1
    assert "must be latest or look like 2026-10-06" in output and "pattern" not in output


def test_the_group_help_shows_every_summary_whole() -> None:
    result = CliRunner().invoke(cli.main, ["s2", "--help"], terminal_width=60)
    code, output = result.exit_code, result.output
    assert code == 0
    listed = " ".join(output.split("Commands:")[1].split())
    assert all(
        text in listed for text in (cli.PIN_HELP, cli.DOWNLOAD_HELP, cli.DOWNLOAD_SHARD_HELP)
    )


def test_an_ingestion_error_exits_1_with_its_message(
    wired: DownloadDatasetService, source: FakeSource
) -> None:
    source.files["papers"]["part-0.gz"] = PAPERS["part-0.gz"][:-4]
    code, output = okp(
        "download-shard", "--release", "2026-10-06", "--dataset", "papers", "--shard", "0", *TARGET
    )
    assert code == 1
    assert "Error: part-0.gz: the gzip stream" in output and "rerun" in output


def test_importing_the_package_turns_prefect_telemetry_off() -> None:
    env = {"PATH": os.environ["PATH"]}
    code = (
        "import os, openknowledge_pipelines; print(os.environ['PREFECT_SERVER_ANALYTICS_ENABLED'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "false"


NO_PREFECT = """
import sys
from contextlib import asynccontextmanager
from click.testing import CliRunner
from openknowledge_core.ingestion import DownloadDatasetService
from openknowledge_pipelines import bootstrap, cli
from tests.fakes.ports import PAPERS, PRODUCER, FakeSource, FakeStore, clock

source = FakeSource()
for name, data in PAPERS.items():
    source.add("papers", name, data)

@asynccontextmanager
async def open_service(settings):
    yield DownloadDatasetService(
        catalog=source, downloads=source, store=FakeStore(), producer=PRODUCER, clock=clock
    )

bootstrap.open_service = open_service
args = ["s2", "download-shard", "--release", "2026-10-06", "--dataset", "papers", "--shard", "0",
        "--bucket", "raw"]
result = CliRunner().invoke(cli.main, args, env={"S2_API_KEY": "k"})
print(result.exit_code, "prefect" in sys.modules)
"""


def test_download_shard_never_loads_prefect() -> None:
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", NO_PREFECT], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["0", "False"]


@pytest.mark.usefixtures("prefect_server")
@pytest.mark.parametrize(
    "name,dataset,family",
    [
        ("archive-fulltext", "s2orc_v2", "fulltext"),
        ("archive-specter2", "embeddings-specter_v2", "embeddings"),
    ],
)
@pytest.mark.parametrize("diff", [False, True])
def test_archive_cli_routes_explicit_commands(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    dataset: str,
    family: str,
    diff: bool,
) -> None:
    deployments.ensure_limits()
    seen: list[DownloadDataset | DownloadDiff] = []

    async def archive(command: DownloadDataset | DownloadDiff) -> DownloadDataset | DownloadDiff:
        seen.append(command)
        return command

    route = f"s2_batch_download_{family}{'_diff' if diff else ''}"
    monkeypatch.setattr(deployments, route, archive)
    for key, value in BATCH_ENV.items():
        monkeypatch.setenv(key, value)
    args = [name, "--release", "2026-10-13", *TARGET]
    if diff:
        args += ["--from-release", "2026-10-06"]
    monkeypatch.delenv("PREFECT_API_URL", raising=False)
    profile = Profile(
        name="shared-archive-test", settings={PREFECT_API_URL: PREFECT_API_URL.value()}
    )
    with use_profile(profile):
        code, output = okp(*args)
    assert code == 0, output
    [command] = seen
    assert command.dataset == dataset
    assert command.release == "2026-10-13"
    assert (command.bucket, command.prefix) == ("raw", "s2")
    assert isinstance(command, DownloadDiff) == diff
    if isinstance(command, DownloadDiff):
        assert command.from_release == "2026-10-06"


@pytest.mark.parametrize(
    "arguments", [["archive-fulltext", "--release", "2026-10-13", *TARGET], ["sync"]]
)
def test_managed_requires_effective_server_before_flow(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    async def forbidden(command: DownloadDataset | None = None) -> DownloadDataset:
        pytest.fail("must validate server before starting the flow")

    monkeypatch.setattr(deployments, "s2_batch_download_fulltext", forbidden)
    monkeypatch.setattr(deployments, "s2_sync", forbidden)
    with temporary_settings({PREFECT_API_URL: None}):
        code, output = okp(*arguments)
    assert code == 1
    assert "shared control VM Prefect server" in output
    assert "PREFECT_API_URL" in output


@pytest.mark.usefixtures("prefect_server")
def test_archive_missing_limit_fails_before_flow_without_console_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    for key, value in BATCH_ENV.items():
        monkeypatch.setenv(key, value)

    def forbidden(settings: object) -> None:
        pytest.fail("missing limit must not open cloud clients")

    monkeypatch.setattr(bootstrap, "open_batch_download", forbidden)
    deployments.ensure_limits()
    with get_client(sync_client=True) as client:
        client.delete_global_concurrency_limit_by_name(deployments.S2_LIMIT)
        before = {run.id for run in client.read_flow_runs()}
        try:
            caplog.clear()
            result = CliRunner().invoke(
                cli.main, ["s2", "archive-fulltext", "--release", "2026-10-13", *TARGET]
            )
            console = capfd.readouterr()
            assert result.exit_code == 1
            assert "Required shared Prefect limit 'semantic-scholar' is missing" in result.output
            assert "shared control VM server" in result.output
            assert "okp serve" in result.output
            assert "Traceback" not in result.output + console.out + console.err + caplog.text
            assert not any(record.exc_info for record in caplog.records)
            assert {run.id for run in client.read_flow_runs()} == before
        finally:
            deployments.ensure_limits()


@pytest.mark.usefixtures("prefect_server")
def test_archive_unrelated_body_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    deployments.ensure_limits()
    for key, value in BATCH_ENV.items():
        monkeypatch.setenv(key, value)

    async def broken(command: DownloadDataset) -> DownloadDataset:
        raise RuntimeError("download failed")

    monkeypatch.setattr(deployments, "s2_batch_download_fulltext", broken)
    result = CliRunner().invoke(
        cli.main, ["s2", "archive-fulltext", "--release", "2026-10-13", *TARGET]
    )
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "download failed"


def test_unreachable_managed_api_fails_before_flow_or_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("unreachable preflight must not start a flow or cloud client")

    monkeypatch.setattr(deployments, "s2_batch_download_fulltext", forbidden)
    monkeypatch.setattr(bootstrap, "open_batch_download", forbidden)
    # A bound, non-listening socket reserves the port and reliably refuses connections.
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{endpoint.getsockname()[1]}/api"
        with temporary_settings({PREFECT_API_URL: url}):
            result = CliRunner().invoke(
                cli.main, ["s2", "archive-fulltext", "--release", "2026-10-13", *TARGET]
            )
    assert result.exit_code == 1
    assert "Cannot reach the configured shared control API" in result.output
    assert "PREFECT_API_URL" in result.output
    assert "Traceback" not in result.output
    assert "127.0.0.1" not in result.output


def test_read_base_cli_without_source_key_or_control_api(
    monkeypatch: pytest.MonkeyPatch,
    service: DownloadDatasetService,
    source: FakeSource,
    store: FakeStore,
) -> None:
    import asyncio
    import gzip

    from openknowledge_core.ingestion.archive_progress import ArchiveProgress
    from openknowledge_core.ingestion.consumers import ConsumeBase

    command = ConsumeBase(dataset="s2orc_v2", release="2026-10-13", bucket="raw", prefix="s2")
    source.add(
        "s2orc_v2",
        "part.gz",
        gzip.compress(b'{"corpusid":7,"body":{"text":"Synthetic body","annotations":null}}\n'),
    )

    async def archive() -> None:
        pinned = await service.pin_dataset(command)
        await service.archive_shard(pinned, pinned.shards[0])
        await ArchiveProgress(store, service, PRODUCER, clock).finalize(command)

    asyncio.run(archive())
    before = dict(store.objects)

    @asynccontextmanager
    async def open_store(config: object) -> AsyncGenerator[FakeStore]:
        yield store

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("read-base must not create source/download clients or write manifests")

    monkeypatch.setattr("openknowledge_adapters.gcs.connect", open_store)
    monkeypatch.setattr("openknowledge_adapters.semantic_scholar.connect", forbidden)
    monkeypatch.setattr(service, "write_manifest", forbidden)
    monkeypatch.delenv("S2_API_KEY", raising=False)
    with temporary_settings({PREFECT_API_URL: None}):
        code, output = okp(
            "read-base",
            "--dataset",
            "s2orc_v2",
            "--release",
            command.release,
            "--max-records",
            "1",
            *TARGET,
        )
    assert code == 0, output
    row = json.loads(output)
    assert row["paper"]["corpusid"] == 7
    assert row["source"]["dataset"] == "s2orc_v2"
    assert row["source"]["release"] == command.release
    assert row["source"]["object_ref"]["generation"] > 0
    assert store.objects == before


def test_search_index_creates_the_index_and_prints_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, sink = FakeSearchSource(), FakeSearchSink()
    source.base_rows = [SourceRows(corpusid=101, paper=paper_row(101, 41464, title="A Paper"))]

    @asynccontextmanager
    async def open_search_service(settings: SearchSettings) -> AsyncGenerator[SearchIndexService]:
        assert settings.opensearch_user == "hub-indexer"
        yield SearchIndexService(source=source, sink=sink, shards=settings.index_shards)

    monkeypatch.setattr(bootstrap, "open_search_service", open_search_service)
    monkeypatch.delenv("S2_API_KEY", raising=False)
    for name, value in {**LOAD_ENV, **SEARCH_ENV}.items():
        monkeypatch.setenv(name, value)
    result = CliRunner().invoke(
        cli.main, ["search", "index", "--index", "papers-v1", "--release", "2026-10-06"]
    )
    assert result.exit_code == 0, result.output
    printed = json.loads(result.output.splitlines()[-1])
    assert (printed["created"], printed["written"], printed["documents"]) == (True, 1, 1)
    assert (printed["part"], printed["parts"], printed["from_release"]) == (1, 1, None)
    assert sink.created["papers-v1"][1]["index"]["number_of_shards"] == 2
    result = CliRunner().invoke(
        cli.main,
        ["search", "index", "--index", "papers-v1", "--release", "2026-10-06", "--range", "3/2"],
    )
    assert result.exit_code == 2
    assert "the part must not exceed the number of parts" in result.output
