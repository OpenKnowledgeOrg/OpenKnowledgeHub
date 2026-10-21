import json
from base64 import b64encode
from collections.abc import Collection, Mapping
from typing import Any

import pytest
from opensearchpy import AsyncConnection
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from openknowledge_adapters import opensearch
from openknowledge_core.ingestion.errors import IngestionError
from openknowledge_core.search.errors import BulkRefused, SearchRejected, SearchUnavailable
from tests.fakes.ports import paper_row, search_document

CONFIG = opensearch.OpenSearchConfig.model_validate(
    {
        "url": "http://opensearch:9200",
        "user": "hub-indexer",
        "password": "hunter2",
        "timeout_seconds": 5,
    }
)
AUTHORIZATION = "Basic " + b64encode(b"hub-indexer:hunter2").decode()
VERSION = 41464


def item(corpusid: str, status: int, error: dict[str, str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"_index": "papers", "_id": corpusid, "status": status}
    if error is not None:
        result["error"] = error
    return {"index": result}


CONFLICT = {
    "type": "version_conflict_engine_exception",
    "reason": "[102]: version conflict, current version [41478] is higher",
}
REJECTED = {"type": "es_rejected_execution_exception", "reason": "rejected execution of bulk"}
REFUSED = {"errors": True, "items": [item("101", 201), item("102", 409, CONFLICT)]}
BUSY = {"errors": True, "items": [item("101", 429, REJECTED), item("102", 429, REJECTED)]}
ECHO = "<html>Bad gateway; the request said Authorization: Basic hunter2 and more</html>"


def answering(status: int, answer: object) -> type[AsyncConnection]:
    """A connection class whose every request gets ``status`` with ``answer``, or no connection."""

    class FakeConnection(AsyncConnection):
        headers: dict[str, str]

        async def perform_request(
            self,
            method: str,
            url: str,
            params: Mapping[str, Any] | None = None,
            body: bytes | None = None,
            timeout: int | float | None = None,  # noqa: ASYNC109  # the library's signature
            ignore: Collection[int] = (),
            headers: Mapping[str, str] | None = None,
        ) -> tuple[int, dict[str, str], str]:
            assert "hunter2" not in url
            assert self.headers["authorization"] == AUTHORIZATION
            if status == 0:
                raise OpenSearchConnectionError("N/A", "connection refused", None)
            text = answer if isinstance(answer, str) else json.dumps(answer)
            content_type = "text/html" if isinstance(answer, str) else "application/json"
            if not 200 <= status < 300 and status not in ignore:
                self._raise_error(status, text, content_type)
            return status, {"content-type": content_type}, text

        async def close(self) -> None:
            return None

    return FakeConnection


@pytest.mark.parametrize(
    ("status", "answer", "error", "message"),
    [
        (200, REFUSED, BulkRefused, "papers: 1 of 2 bulk items refused; corpusid 102: version_"),
        (200, BUSY, SearchUnavailable, "papers: 2 of 2 bulk items refused; corpusid 101: es_"),
        (
            0,
            {},
            SearchUnavailable,
            "OpenSearch bulk write to papers failed (ConnectionError); rerun",
        ),
        (401, {"error": "Unauthorized"}, SearchRejected, "OpenSearch refused the user or password"),
        (403, {"error": "Forbidden"}, SearchRejected, "OpenSearch denied the user to bulk write"),
        (
            503,
            {"error": "unavailable"},
            SearchUnavailable,
            "OpenSearch bulk write to papers failed with status 503: unavailable; rerun",
        ),
        (
            502,
            ECHO,
            SearchUnavailable,
            "OpenSearch bulk write to papers failed with status 502: an answer that is not",
        ),
    ],
)
async def test_bulk_failures_are_mapped_and_a_refused_item_is_named(
    status: int,
    answer: object,
    error: type[IngestionError],
    message: str,
) -> None:
    operations: list[Any] = [
        search_document(paper_row(101, VERSION, title="A Synthetic Paper")),
        search_document(paper_row(102, VERSION)),
    ]
    async with opensearch.connect(CONFIG, connection_class=answering(status, answer)) as sink:
        with pytest.raises(error) as raised:
            await sink.bulk("papers", operations)
    assert str(raised.value).startswith(message), str(raised.value)
    assert "hunter2" not in str(raised.value)
    assert raised.value.__context__ is None
