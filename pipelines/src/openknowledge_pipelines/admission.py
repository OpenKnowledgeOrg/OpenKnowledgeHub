"""Nonleased admission survives runner replacement and uncertain external work."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

import httpx
from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import (
    GlobalConcurrencyLimitCreate,
    GlobalConcurrencyLimitUpdate,
)
from prefect.client.schemas.responses import GlobalConcurrencyLimitResponse
from prefect.exceptions import ObjectNotFound
from prefect.settings import PREFECT_API_URL, PREFECT_CLIENT_MAX_RETRIES, temporary_settings
from pydantic import BaseModel

NAME = "deployment-admission"
OPEN_LIMIT = 1_000_000
_LOG = logging.getLogger(__name__)


class AdmissionError(RuntimeError):
    """A sanitized refusal safe to show at the command boundary."""


class AdmissionStatus(BaseModel):
    supported: bool = True
    closed: bool
    active_slots: int


def _require_server() -> None:
    if not PREFECT_API_URL.value():
        raise AdmissionError("Admission requires the shared PREFECT_API_URL")


def _validate(current: GlobalConcurrencyLimitResponse) -> None:
    if not current.active or current.slot_decay_per_second != 0:
        raise AdmissionError(
            "Admission must be active without slot decay; operator repair required"
        )
    if current.limit not in (0, OPEN_LIMIT):
        raise AdmissionError("Unexpected admission capacity; operator repair required")


def _check_response(response: httpx.Response) -> None:
    # Prefect retries maintenance responses indefinitely, even with max_retries=0.
    # Abort rejected requests before its retry loop. Preserve the GET responses
    # used for missing-limit detection and discovery of disabled CSRF protection.
    request = response.request
    expected = request.method == "GET" and (
        response.status_code == 404
        or (request.url.path.endswith("/csrf-token") and response.status_code == 422)
    )
    if not response.is_success and not expected:
        raise AdmissionError(
            "Admission API request refused; state may be uncertain; inspect status before retrying"
        )


async def _check_async_response(response: httpx.Response) -> None:
    _check_response(response)


@asynccontextmanager
async def execution() -> AsyncGenerator[None]:
    _require_server()
    owner = uuid4().hex
    # The pinned client's retry loop reads this setting on each send. Never replay a
    # counter mutation: a response can be lost after the database commits it.
    async with get_client(
        httpx_settings={"timeout": 10, "event_hooks": {"response": [_check_async_response]}}
    ) as client:
        _LOG.warning("Admission acquisition started owner=%s", owner)
        try:
            with temporary_settings({PREFECT_CLIENT_MAX_RETRIES: 0}):
                async with asyncio.timeout(30):
                    current = await client.read_global_concurrency_limit_by_name(NAME)
                    _validate(current)
                    response = await client.increment_concurrency_slots([NAME], 1, "concurrency")
                    if response.json() != [
                        {"id": str(current.id), "name": NAME, "limit": OPEN_LIMIT}
                    ]:
                        raise AdmissionError("Admission acquisition unconfirmed")
                    # Prefect returns inactive limits too, even when no slot was taken.
                    # Check again before work; never compensate an uncertain acquisition.
                    confirmed = await client.read_global_concurrency_limit_by_name(NAME)
                    _validate(confirmed)
                    if confirmed.id != current.id or confirmed.active_slots < 1:
                        raise AdmissionError("Admission acquisition unconfirmed")
        except BaseException as exc:
            _LOG.error("Admission acquisition unconfirmed; reconcile owner=%s", owner)
            if not isinstance(exc, Exception):
                raise
            raise AdmissionError(
                f"Admission closed, unavailable or invalid; execution refused; "
                f"reconcile uncertain occupancy (owner={owner})"
            ) from None
        _LOG.warning("Admission acquired owner=%s", owner)
        try:
            yield
        except BaseException:
            # Even an ordinary error may leave Batch/native/database work alive.
            _LOG.error("Execution failed or interrupted; admission retained owner=%s", owner)
            raise
        else:
            try:
                with temporary_settings({PREFECT_CLIENT_MAX_RETRIES: 0}):
                    async with asyncio.timeout(30):
                        confirmed = await client.read_global_concurrency_limit_by_name(NAME)
                        _validate(confirmed)
                        if confirmed.id != current.id:
                            raise AdmissionError("Admission identity changed")
                        # Occupancy averaging divides by capacity in Prefect 3.8.8.
                        # Omit that optional statistic so release also works closed.
                        response = await client.request(
                            "POST",
                            "/v2/concurrency_limits/decrement",
                            json={"names": [NAME], "slots": 1},
                        )
                        # Closing can race this request after the pre-read.
                        if response.json() not in (
                            [{"id": str(current.id), "name": NAME, "limit": capacity}]
                            for capacity in (0, OPEN_LIMIT)
                        ):
                            raise AdmissionError("Admission release unconfirmed")
            except BaseException as exc:
                _LOG.error("Admission release unconfirmed; reconcile owner=%s", owner)
                if not isinstance(exc, Exception):
                    raise
                raise AdmissionError(
                    f"Admission release uncertain; operator reconciliation required (owner={owner})"
                ) from None
            _LOG.warning("Admission released owner=%s", owner)


def ensure_closed() -> None:
    """Initial install requires explicit operator admission after verification."""
    _require_server()
    try:
        with (
            temporary_settings({PREFECT_CLIENT_MAX_RETRIES: 0}),
            get_client(
                sync_client=True,
                httpx_settings={"timeout": 10, "event_hooks": {"response": [_check_response]}},
            ) as client,
        ):
            try:
                current = client.read_global_concurrency_limit_by_name(NAME)
            except ObjectNotFound:
                client.create_global_concurrency_limit(
                    GlobalConcurrencyLimitCreate(name=NAME, limit=0)
                )
            else:
                _validate(current)
    except AdmissionError:
        raise
    except Exception:
        raise AdmissionError(
            "Cannot initialize shared admission; check the control API before starting serve"
        ) from None


def status(action: Literal["status", "close", "open"] = "status") -> AdmissionStatus:
    _require_server()
    with (
        temporary_settings({PREFECT_CLIENT_MAX_RETRIES: 0}),
        get_client(
            sync_client=True,
            httpx_settings={"timeout": 10, "event_hooks": {"response": [_check_response]}},
        ) as client,
    ):
        current = client.read_global_concurrency_limit_by_name(NAME)
        _validate(current)
        if action == "open" and current.active_slots != 0:
            raise AdmissionError("Admission still occupied; reconcile work before reopening")
        if action in ("close", "open"):
            client.update_global_concurrency_limit(
                NAME, GlobalConcurrencyLimitUpdate(limit=0 if action == "close" else OPEN_LIMIT)
            )
            current = client.read_global_concurrency_limit_by_name(NAME)
            _validate(current)
        return AdmissionStatus(closed=current.limit == 0, active_slots=current.active_slots)
