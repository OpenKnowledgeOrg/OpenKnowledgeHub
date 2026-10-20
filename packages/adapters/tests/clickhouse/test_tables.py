import json

import httpx
import pytest

from openknowledge_adapters import clickhouse
from openknowledge_contracts.tables.clickhouse import version_expression
from openknowledge_contracts.tables.s2 import CITATIONS
from openknowledge_core.ingestion import LoadDataset, LoadDatasetService
from openknowledge_core.ingestion.errors import (
    ContractViolation,
    CredentialRejected,
    IngestionError,
    TableRejected,
    TableUnavailable,
)
from tests.fakes.ports import PRODUCER, FakeApplied, FakeStore, clock

CONFIG = clickhouse.ClickHouseConfig.model_validate(
    {
        "url": "http://clickhouse:8123",
        "user": "openknowledge",
        "password": "hunter2",
        "raw_bucket": "raw",
        "timeout_seconds": 5,
    }
)
VIOLATION = (
    "Code: 395. DB::Exception: papers: corpusid breaks the contract: while executing "
    "'FUNCTION throwIf(...)'. (FUNCTION_THROW_IF_VALUE_IS_NON_ZERO) (version 26.8.22.13)"
)


@pytest.mark.parametrize(
    ("status", "code", "body", "error", "message"),
    [
        (500, "395", VIOLATION, ContractViolation, "papers: corpusid breaks the contract; "),
        (403, "516", "Code: 516. DB::Exception: wrong password", CredentialRejected, "ClickHouse"),
        (503, "", "", TableUnavailable, "ClickHouse query"),
        (404, "60", "Code: 60. DB::Exception: Unknown table. (UNKNOWN_TABLE)", TableRejected, ""),
        (0, "", "", TableUnavailable, "ClickHouse query"),
    ],
)
async def test_errors_map_by_exception_code(
    status: int, code: str, body: str, error: type[IngestionError], message: str
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert "hunter2" not in str(request.url)
        assert request.headers["X-ClickHouse-Key"] == "hunter2"
        if not status:
            raise httpx.ConnectError("refused")
        return httpx.Response(status, headers={"X-ClickHouse-Exception-Code": code}, text=body)

    async with clickhouse.connect(CONFIG, transport=httpx.MockTransport(handle)) as ch:
        with pytest.raises(error) as raised:
            await ch.tables.live_rows("papers", "2026-10-06")
    assert str(raised.value).startswith(message)
    assert "hunter2" not in str(raised.value)


@pytest.mark.parametrize("current", [False, True])
async def test_citations_require_enrichment_versions_before_checkpoint_reuse(current: bool) -> None:
    queries: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        query = request.content.decode()
        queries.append(query)
        if "system.tables" in query:
            row = {"uuid": "00000000-0000-0000-0000-000000000001", "total_rows": 100}
        else:
            assert "system.columns" in query
            row = {
                "default_expression": version_expression(CITATIONS)
                if current
                else "(toUInt64(source_release) * 2) + is_deleted"
            }
        return httpx.Response(200, text=json.dumps(row))

    async with clickhouse.connect(CONFIG, transport=httpx.MockTransport(handle)) as ch:
        if current:
            assert (await ch.tables.table("citations")).has_rows
        else:
            service = LoadDatasetService(
                store=FakeStore(),
                tables=ch.tables,
                applied=FakeApplied(),
                producer=PRODUCER,
                clock=clock,
            )
            command = LoadDataset(release="2026-10-13", dataset="citations", bucket="raw")
            with pytest.raises(TableRejected, match="populated legacy table must be preserved"):
                await service.prepare(command)
    assert len(queries) == 2  # No checkpoint lookup, archive scan, or INSERT on legacy versions.
