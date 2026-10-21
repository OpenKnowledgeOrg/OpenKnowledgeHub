"""The search sink over OpenSearch: index creation, versioned bulk writes, refresh and count.

Every bulk item carries its version with ``external_gte``, so the same operations can be sent
again. The client's own retries cover a dropped connection and the 502 to 504 statuses; a bulk
request sent twice converges for the same reason. A failure repeats what OpenSearch said only as
an error type and the first line of a reason, never a whole answer, because an answer from the
wrong host can echo the request and its credentials.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, cast

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import (
    AuthenticationException,
    AuthorizationException,
    SSLError,
    TransportError,
)
from opensearchpy.exceptions import (
    ConnectionError as OpenSearchConnectionError,
)

from openknowledge_core.ingestion.errors import TRANSIENT_STATUSES, IngestionError
from openknowledge_core.search.errors import BulkRefused, SearchRejected, SearchUnavailable
from openknowledge_core.search.models import BulkOperation, DeleteDocument, IndexName

VERSION_TYPE = "external_gte"
# Refused items a failure names, at most.
NAMED = 3
# An item OpenSearch could not take right now, which a rerun sends again.
TOO_MANY_REQUESTS = 429
_ERROR_TYPE = re.compile(r"[A-Za-z0-9_.]{1,100}")
_REASON_LIMIT = 300


class OpenSearchSink:
    def __init__(self, client: AsyncOpenSearch) -> None:
        self._client = client

    async def exists(self, index: IndexName) -> bool:
        return bool(await self._call("look up", index, self._client.indices.exists(index=index)))

    async def create_index(
        self, index: IndexName, mapping: Mapping[str, Any], settings: Mapping[str, Any]
    ) -> None:
        body = {"mappings": dict(mapping), "settings": dict(settings)}
        await self._call("create", index, self._client.indices.create(index=index, body=body))

    async def bulk(self, index: IndexName, operations: Sequence[BulkOperation]) -> None:
        """Send every operation in one request and check every item of the answer.

        Items refused with 429 alone are transient: OpenSearch could not take them right now,
        and a rerun sends them again. Any other refusal fails the pass for good.
        """
        if not operations:
            return
        body: list[dict[str, Any]] = []
        for operation in operations:
            action = {
                "_index": index,
                "_id": operation.id,
                "version": operation.version,
                "version_type": VERSION_TYPE,
            }
            if isinstance(operation, DeleteDocument):
                body.append({"delete": action})
            else:
                body.append({"index": action})
                body.append(operation.document.source())
        response: object = await self._call("bulk write to", index, self._client.bulk(body=body))
        items = _items(response)
        if items is None or len(items) != len(operations):
            raise SearchRejected(
                f"OpenSearch answered a bulk write to {index} without one item per operation"
            )
        refused = [
            (operation.id, refusal)
            for operation, item in zip(operations, items, strict=True)
            if (refusal := _refusal(item)) is not None
        ]
        if not refused:
            return
        named = "; ".join(f"corpusid {id_}: {reason}" for id_, (_, reason) in refused[:NAMED])
        message = f"{index}: {len(refused)} of {len(operations)} bulk items refused; {named}"
        if all(status == TOO_MANY_REQUESTS for _, (status, _) in refused):
            raise SearchUnavailable(f"{message}; rerun")
        raise BulkRefused(message)

    async def refresh(self, index: IndexName) -> None:
        await self._call("refresh", index, self._client.indices.refresh(index=index))

    async def count(self, index: IndexName) -> int:
        answer: object = await self._call("count", index, self._client.count(index=index))
        count = cast("dict[str, object]", answer).get("count") if isinstance(answer, dict) else None
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SearchRejected(f"OpenSearch answered the count of {index} without a count")
        return count

    async def _call[T](self, what: str, index: str, request: Awaitable[T]) -> T:
        """Await one client call, mapping its failures.

        The failure is raised outside the handler, unchained, so neither its message nor its
        context can carry the request and its credentials.
        """
        error: IngestionError
        try:
            return await request
        except SSLError as exc:
            error = SearchRejected(
                f"OpenSearch's certificate was refused ({type(exc).__name__}); cannot {what} "
                f"{index}; check OKP_OPENSEARCH_URL and the certificate"
            )
        except OpenSearchConnectionError as exc:  # a timeout is one of these
            error = SearchUnavailable(
                f"OpenSearch {what} {index} failed ({type(exc).__name__}); rerun"
            )
        except AuthenticationException:
            error = SearchRejected(
                f"OpenSearch refused the user or password; cannot {what} {index}"
            )
        except AuthorizationException:
            error = SearchRejected(
                f"OpenSearch denied the user to {what} {index}; check its permissions"
            )
        except TransportError as exc:
            status = exc.status_code
            failure = (
                f"OpenSearch {what} {index} failed with status {status}: "
                f"{_said(exc.info, exc.error)}"
            )
            if status in TRANSIENT_STATUSES:
                error = SearchUnavailable(f"{failure}; rerun")
            else:
                error = SearchRejected(failure)
        raise error


def _items(response: object) -> list[object] | None:
    """The items of a bulk answer, or None when the answer has no such list."""
    if not isinstance(response, dict):
        return None
    items = cast("dict[str, object]", response).get("items")
    return cast("list[object]", items) if isinstance(items, list) else None


def _refusal(item: object) -> tuple[int, str] | None:
    """A refused item's status and reason, or None when it was applied, as a not found delete is."""
    if not isinstance(item, dict):
        return 0, "an item that is not an object"
    results = list(cast("dict[str, object]", item).values())
    result = results[0] if results else None
    if not isinstance(result, dict):
        return 0, "an item without a result"
    found = cast("dict[str, object]", result)
    error = found.get("error")
    if error is None:
        return None
    status = found.get("status")
    if isinstance(status, bool) or not isinstance(status, int):
        status = 0
    return status, _reason(error)


def _said(info: object, error: object) -> str:
    """What OpenSearch said, as its error type and the first line of its reason, cut short.

    opensearch-py sets ``error`` to the answer's whole text when it is not JSON, which the wrong
    host could fill with the request itself, so such an answer is only named, never repeated.
    """
    if isinstance(info, dict):
        found = cast("dict[str, object]", info).get("error")
        if found is not None:
            return _reason(found)
    return "an answer that is not OpenSearch's"


def _reason(error: object) -> str:
    if isinstance(error, dict):
        detail = cast("dict[str, object]", error)
        kind = _ERROR_TYPE.match(str(detail.get("type", "")))
        reason = detail.get("reason")
        text = _line(reason) if isinstance(reason, str) else "no reason given"
        return f"{kind.group() if kind else 'error'}: {text}"
    if isinstance(error, str):
        return _line(error)
    return "an error that is not an object"


def _line(text: str) -> str:
    return (text.strip().splitlines() or [""])[0][:_REASON_LIMIT]
