"""``okp s2 load`` against ClickHouse and a fake GCS in Docker, which CI must have."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import shutil
import socket
import subprocess
from collections.abc import Awaitable, Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner
from prefect.testing.utilities import prefect_test_harness

from openknowledge_adapters import gcs
from openknowledge_core.ingestion import (
    DownloadDataset,
    DownloadDatasetService,
    DownloadDiff,
    LoadDatasetResult,
    LoadDatasetService,
    LoadOutcome,
    PinnedDataset,
    PreparedLoad,
)
from openknowledge_pipelines import cli
from tests.clickhouse_local import MIGRATIONS, section, up
from tests.fakes.ports import PRODUCER, FakeSource, clock

E2E = Path(__file__).parent / "e2e"
FIXTURE = (
    Path(__file__).parents[1] / "packages/contracts/tests/fixtures/s2/publication-venues.jsonl"
)
BUCKET = "openknowledge-508216-raw"
DATASET = "publication-venues"
TLDRS = "tldrs"
DOCKER = shutil.which("docker")

pytestmark = pytest.mark.skipif(
    DOCKER is None and not os.environ.get("CI"), reason="no docker on PATH"
)


def _free_port() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


@pytest.fixture(scope="module")
def ports() -> Iterator[dict[str, str]]:
    ports = {name: _free_port() for name in ("CLICKHOUSE_PORT", "GCS_PORT", "OPENSEARCH_PORT")}
    env = {**os.environ, **ports}
    compose = [
        DOCKER or "docker",
        "compose",
        "-p",
        f"okp-e2e-{os.getpid()}",
        "-f",
        str(E2E / "compose.yaml"),
    ]
    try:
        subprocess.run([*compose, "up", "-d", "--wait"], env=env, check=True)
        subprocess.run(
            [
                *compose,
                "exec",
                "-T",
                "clickhouse",
                "clickhouse-client",
                "--multiquery",
            ],
            input=f"CREATE DATABASE openknowledge;\nUSE openknowledge;\n{up()}",
            text=True,
            env=env,
            check=True,
        )
        gcs_root = f"http://127.0.0.1:{ports['GCS_PORT']}"
        httpx.post(
            f"{gcs_root}/storage/v1/b", params={"project": "e2e"}, json={"name": BUCKET}
        ).raise_for_status()
        with prefect_test_harness():
            yield ports
    finally:
        subprocess.run([*compose, "down", "--volumes"], env=env, check=True)


def packed(lines: list[str]) -> bytes:
    return gzip.compress("".join(f"{line}\n" for line in lines).encode(), mtime=0)


def archiving(
    ports: dict[str, str],
    source: FakeSource,
    work: Callable[[DownloadDatasetService], Awaitable[None]],
) -> None:
    """Run ``work`` on the download service that archives ``source`` into the fake GCS."""
    config = gcs.GcsConfig(
        request_timeout_seconds=60,
        min_transfer_rate=1024,
        api_root=f"http://127.0.0.1:{ports['GCS_PORT']}",
    )

    async def run() -> None:
        async with gcs.connect(config) as store:
            await work(
                DownloadDatasetService(
                    catalog=source, downloads=source, store=store, producer=PRODUCER, clock=clock
                )
            )

    asyncio.run(run())


async def archive_part(service: DownloadDatasetService, pinned: PinnedDataset) -> None:
    for shard in pinned.shards:
        await service.archive_shard(pinned, shard)
    await service.write_manifest(pinned)


def archive(ports: dict[str, str], release: str, lines: list[str], dataset: str = DATASET) -> None:
    """Archive one shard per line through the download service, as okp s2 download would."""
    source = FakeSource()
    source.releases = [release]
    for index, line in enumerate(lines):
        source.add(dataset, f"part-{index}.gz", packed([line]))

    async def work(service: DownloadDatasetService) -> None:
        command = DownloadDataset(release=release, dataset=dataset, bucket=BUCKET)
        await archive_part(service, await service.pin_dataset(command))

    archiving(ports, source, work)


def archive_diff(
    ports: dict[str, str],
    from_release: str,
    release: str,
    updates: list[str],
    deletes: list[str],
    dataset: str = TLDRS,
) -> None:
    """Archive one update file and one delete file with the given records."""
    source = FakeSource()
    source.releases = [release]
    source.add_diff(
        dataset,
        from_release,
        release,
        {"part-0.gz": packed(updates)},
        {"part-0.gz": packed(deletes)},
    )

    async def work(service: DownloadDatasetService) -> None:
        command = DownloadDiff(
            from_release=from_release, release=release, dataset=dataset, bucket=BUCKET
        )
        pinned = await service.pin_diff(command)
        await archive_part(service, pinned.updates)
        await archive_part(service, pinned.deletes)

    archiving(ports, source, work)


def okp(ports: dict[str, str], *args: str) -> tuple[int, str]:
    env = {
        "OKP_CLICKHOUSE_URL": f"http://127.0.0.1:{ports['CLICKHOUSE_PORT']}",
        "CLICKHOUSE_PASSWORD": "openknowledge",
        "OKP_CLICKHOUSE_RAW_BUCKET": BUCKET,
        "OKP_GCS_API_ROOT": f"http://127.0.0.1:{ports['GCS_PORT']}",
        "OKP_TASK_RETRY_DELAY_SECONDS": "0",
    }
    result = CliRunner().invoke(cli.main, ["s2", *args, "--bucket", BUCKET], env=env)
    return result.exit_code, result.output


def okp_load(ports: dict[str, str], release: str) -> tuple[int, str]:
    return okp(ports, "load", "--release", release, "--dataset", DATASET)


def ran(ports: dict[str, str], *args: str) -> dict[str, Any]:
    code, output = okp(ports, *args)
    assert code == 0, output
    return json.loads(output.splitlines()[-1])


def loaded(ports: dict[str, str], release: str) -> dict[str, Any]:
    return ran(ports, "load", "--release", release, "--dataset", DATASET)


def query(ports: dict[str, str], sql: str) -> str:
    response = httpx.post(
        f"http://127.0.0.1:{ports['CLICKHOUSE_PORT']}",
        params={"database": "openknowledge"},
        headers={"X-ClickHouse-User": "openknowledge", "X-ClickHouse-Key": "openknowledge"},
        content=sql,
    )
    response.raise_for_status()
    return response.text


def test_a_load_is_repeatable_and_rebuilt_from_raw(ports: dict[str, str]) -> None:
    archive(ports, "2026-10-06", FIXTURE.read_text().splitlines())
    first = loaded(ports, "2026-10-06")
    assert (first["loaded"], first["live_rows"]) == (2, 2)
    assert loaded(ports, "2026-10-06")["already_loaded"] == 2
    query(ports, "TRUNCATE TABLE loads")
    again = loaded(ports, "2026-10-06")
    assert (again["loaded"], again["live_rows"]) == (2, 2)


@pytest.mark.parametrize("dataset, count", [("authors", 2), ("papers", 5)])
def test_nullable_metadata_loads_and_replays(
    ports: dict[str, str], dataset: str, count: int
) -> None:
    lines = FIXTURE.with_name(f"{dataset}.jsonl").read_text().splitlines()
    if dataset == "authors":
        # Fail on the unknown count before any other object can be checkpointed.
        lines.reverse()
    archive(ports, "2026-10-13", lines, dataset)
    args = ("load", "--release", "2026-10-13", "--dataset", dataset)
    if dataset == "authors":
        migration = MIGRATIONS / "00011_make_authors_papercount_nullable.sql"
        query(ports, "ALTER TABLE authors MODIFY COLUMN papercount UInt32 DEFAULT 0")
        query(ports, "ALTER TABLE authors MODIFY COLUMN papercount REMOVE DEFAULT")
        try:
            code, output = okp(ports, *args)
            assert code == 1
            assert "code 349" in output
            assert query(ports, "SELECT count() FROM loads WHERE target = 'authors'") == "0\n"
            assert query(ports, "SELECT count() FROM applied WHERE target = 'authors'") == "0\n"
        finally:
            # The migration has two statements; the HTTP endpoint accepts one at a time.
            for statement in section(migration, "Up").split(";"):
                if statement.strip():
                    query(ports, statement)
    first = ran(ports, *args)
    assert (first["loaded"], first["live_rows"]) == (count, count)
    again = ran(ports, *args)
    assert (again["loaded"], again["already_loaded"], again["live_rows"]) == (0, count, count)
    expected: list[dict[str, Any]]
    if dataset == "authors":
        columns, key = "authorid, papercount", "authorid"
        expected = [{"authorid": "1001", "papercount": 3}, {"authorid": "1002", "papercount": None}]
    else:
        columns, key = (
            "corpusid, url, title, venue, authors, authors_is_null, journal, journal_is_null",
            "corpusid",
        )
        expected = [
            {
                "corpusid": row["corpusid"],
                "url": row.get("url"),
                "title": row.get("title"),
                "venue": row.get("venue"),
                "authors": row["authors"] or [],
                "authors_is_null": row["authors"] is None,
                "journal": row["journal"] or {"name": None, "pages": None, "volume": None},
                "journal_is_null": row["journal"] is None,
            }
            for row in sorted((json.loads(line) for line in lines), key=lambda row: row["corpusid"])
        ]
    records = query(
        ports, f"SELECT {columns} FROM {dataset} FINAL ORDER BY {key} FORMAT JSONEachRow"
    )
    assert [json.loads(line) for line in records.splitlines()] == expected


def test_a_record_that_breaks_the_contract_fails_the_load(ports: dict[str, str]) -> None:
    record = json.loads(FIXTURE.read_text().splitlines()[0])
    archive(ports, "2026-10-13", [json.dumps({**record, "surprise": 1})])
    code, output = okp_load(ports, "2026-10-13")
    assert code == 1
    assert "a record has a key the contract does not declare; the object stays unloaded" in output


def test_a_key_repeated_identically_is_counted(ports: dict[str, str]) -> None:
    record = json.loads(FIXTURE.read_text().splitlines()[0])
    archive(ports, "2026-10-20", [json.dumps(record), json.dumps(dict(reversed(record.items())))])
    result = loaded(ports, "2026-10-20")
    assert (result["live_rows"], result["identical_duplicates"]) == (1, 1)


def test_citation_context_overlap_resumes_finalization_from_checkpoints(
    ports: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    release = "2026-10-13"
    row = {
        "citationid": 1577791790,
        "citingcorpusid": 139809936,
        "citedcorpusid": None,
        "isinfluential": False,
        "contexts": ["A representative citation context."],
        "intents": [["background"]],
    }
    # Load the rich row first, so a load-order winner would incorrectly retain null.
    archive(ports, release, [json.dumps(row), json.dumps({**row, "contexts": None})], "citations")
    args = ("load", "--release", release, "--dataset", "citations")

    async def interrupted(
        self: LoadDatasetService, prepared: PreparedLoad, outcomes: Iterable[LoadOutcome]
    ) -> LoadDatasetResult:
        raise RuntimeError("simulated interruption before finalization")

    # Only inject the interruption: archive access, inserts, and checkpoints are all real.
    with monkeypatch.context() as patch:
        patch.setattr(LoadDatasetService, "finish", interrupted)
        code, _ = okp(ports, *args)
        assert code == 1
    assert query(ports, "SELECT count() FROM loads FINAL WHERE target = 'citations'") == "2\n"
    assert query(ports, "SELECT count() FROM applied WHERE target = 'citations'") == "0\n"
    result = ran(ports, *args)
    assert (result["loaded"], result["already_loaded"], result["live_rows"]) == (0, 2, 1)
    assert (result["identical_duplicates"], result["enriched_duplicates"]) == (0, 1)
    query(ports, "OPTIMIZE TABLE citations FINAL")
    records = query(
        ports, "SELECT contexts, contexts_is_null FROM citations FINAL FORMAT JSONEachRow"
    )
    assert [json.loads(line) for line in records.splitlines()] == [
        {"contexts": row["contexts"], "contexts_is_null": False}
    ]


def test_a_diff_upserts_and_deletes_and_is_repeatable(ports: dict[str, str]) -> None:
    old, new = "2026-11-03", "2026-11-10"

    def tldr(key: int, text: str) -> str:
        return json.dumps({"corpusid": key, "model": "tldr@v2.0.0", "text": text})

    archive(ports, old, [tldr(1, "old"), tldr(2, "gone")], TLDRS)
    ran(ports, "load", "--release", old, "--dataset", TLDRS)
    archive_diff(
        ports, old, new, updates=[tldr(1, "new"), tldr(3, "added")], deletes=['{"corpusid": 2}']
    )
    diff = ["load-diff", "--from-release", old, "--release", new, "--dataset", TLDRS]
    first = ran(ports, *diff)
    assert (first["loaded"], first["upserted"], first["deleted"]) == (2, 2, 1)
    assert query(ports, "SELECT corpusid, text FROM tldrs FINAL ORDER BY corpusid") == (
        "1\tnew\n3\tadded\n"
    )
    applied = "SELECT release, from_release FROM applied WHERE target = 'tldrs'"
    assert query(ports, f"{applied} ORDER BY applied_at DESC LIMIT 1") == f"{new}\t{old}\n"
    again = ran(ports, *diff)
    assert (again["loaded"], again["already_loaded"]) == (0, 2)


def test_measured_single_load_and_replay(ports: dict[str, str], tmp_path: Path) -> None:
    release = "2027-01-05"
    archive(ports, release, FIXTURE.read_text().splitlines())
    reports = [tmp_path / "first.json", tmp_path / "replay.json"]
    for report in reports:
        ran(
            ports,
            "load",
            "--release",
            release,
            "--dataset",
            DATASET,
            "--measurement-report",
            str(report),
            "--measurement-kind",
            "synthetic-validation",
        )
    first, replay = [json.loads(path.read_text()) for path in reports]
    assert first["status"] == replay["status"] == "completed"
    assert first["validation_kind"] == "synthetic-validation"
    assert first["counters"]["acknowledged_written_rows"] == 2
    assert first["counters"]["attempted_compressed_source_bytes"] > 0
    assert first["coverage"] == "full-manifest"
    assert first["before"]["active_table_bytes"] is not None
    assert first["after"]["active_table_bytes"] > 0
    assert first["before"]["server_memory_tracking_bytes"] is not None
    assert first["after"]["table_running_merges"] is not None
    assert replay["counters"]["acknowledged_written_rows"] == 0
    assert replay["acknowledged_rows_per_second"] == 0
    assert replay["counters"]["attempted_compressed_source_bytes"] == 0
    assert replay["coverage"] == "partial-or-resumed"


def test_measured_failure_retains_partial_writes_and_failure(
    ports: dict[str, str],
    tmp_path: Path,
) -> None:
    release = "2027-01-12"
    record = json.loads(FIXTURE.read_text().splitlines()[0])
    archive(ports, release, [json.dumps(record), json.dumps({**record, "surprise": 1})])
    report_file = tmp_path / "failed.json"
    code, output = okp(
        ports,
        "load",
        "--release",
        release,
        "--dataset",
        DATASET,
        "--measurement-report",
        str(report_file),
        "--measurement-kind",
        "synthetic-validation",
    )
    assert code == 1 and "contract" in output
    report = json.loads(report_file.read_text())
    assert report["status"] == "failed" and report["error_type"] == "ContractViolation"
    assert report["counters"]["acknowledged_written_rows"] == 1
    assert report["counters"]["attempts_without_acknowledged_row_count"] == 1
    assert report["coverage"] == "partial-or-resumed"
    assert report["before"]["active_table_bytes"] is not None
    assert report["after"]["active_table_bytes"] > 0
    assert query(ports, f"SELECT count() FROM applied WHERE release = '{release}'") == "0\n"
