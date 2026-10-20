"""The real pinned Prefect API admission boundary with disposable server/database."""

import asyncio
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas.actions import GlobalConcurrencyLimitUpdate
from prefect.client.schemas.responses import GlobalConcurrencyLimitResponse
from prefect.settings import PREFECT_API_URL, temporary_settings

from openknowledge_pipelines import admission


@pytest.fixture
def closed_admission(prefect_server: None) -> Generator[None]:
    with get_client(sync_client=True) as client:
        client.update_global_concurrency_limit(
            admission.NAME, GlobalConcurrencyLimitUpdate(limit=0, active=True, active_slots=0)
        )
    yield
    with get_client(sync_client=True) as client:
        client.update_global_concurrency_limit(
            admission.NAME,
            GlobalConcurrencyLimitUpdate(limit=admission.OPEN_LIMIT, active=True, active_slots=0),
        )


async def test_close_blocks_concurrent_submission_without_cancelling_existing(
    closed_admission: None,
) -> None:
    admission.status("open")
    async with admission.execution():
        with ThreadPoolExecutor(1) as pool:
            closed = pool.submit(admission.status, "close").result(timeout=10)
        assert closed.closed and closed.active_slots == 1
        with pytest.raises(RuntimeError, match="execution refused"):
            async with admission.execution():
                pytest.fail("closed gate admitted new execution")
    assert admission.status().active_slots == 0
    assert admission.status().closed


async def test_missing_and_inactive_limits_refuse_execution(closed_admission: None) -> None:
    with get_client(sync_client=True) as client:
        client.update_global_concurrency_limit(
            admission.NAME, GlobalConcurrencyLimitUpdate(active=False)
        )
    with pytest.raises(RuntimeError, match="execution refused"):
        async with admission.execution():
            pytest.fail("inactive gate admitted execution")
    with get_client(sync_client=True) as client:
        client.delete_global_concurrency_limit_by_name(admission.NAME)
    with pytest.raises(RuntimeError, match="execution refused"):
        async with admission.execution():
            pytest.fail("missing gate admitted execution")
    admission.ensure_closed()
    assert admission.status().closed


async def test_unreleased_slot_survives_close_and_serve_initialization(
    closed_admission: None,
) -> None:
    admission.status("open")
    async with get_client() as client:
        await client.increment_concurrency_slots([admission.NAME], 1, "concurrency")
    admission.status("close")
    admission.ensure_closed()
    assert admission.status().active_slots == 1
    assert admission.status().closed
    with pytest.raises(RuntimeError, match="still occupied"):
        admission.status("open")


@pytest.mark.parametrize("interruption", [asyncio.CancelledError, KeyboardInterrupt])
async def test_interruption_retains_nonleased_slot_for_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    client = AsyncMock(spec=PrefectClient)
    client.__aenter__.return_value = client
    current = GlobalConcurrencyLimitResponse(
        id=UUID(int=1),
        name=admission.NAME,
        limit=admission.OPEN_LIMIT,
        active_slots=1,
        slot_decay_per_second=0,
    )
    client.read_global_concurrency_limit_by_name.return_value = current
    client.increment_concurrency_slots.return_value = httpx.Response(
        200, json=[{"id": str(current.id), "name": admission.NAME, "limit": admission.OPEN_LIMIT}]
    )

    def make_client(**kwargs: object) -> AsyncMock:
        return client

    monkeypatch.setattr(admission, "get_client", make_client)
    with temporary_settings({PREFECT_API_URL: "http://disposable.test/api"}):
        with pytest.raises(interruption):
            async with admission.execution():
                raise interruption()
    client.increment_concurrency_slots.assert_awaited_once()
    client.request.assert_not_awaited()


@pytest.mark.parametrize("attempt", range(4))
async def test_close_races_acquisition_without_losing_occupancy(
    closed_admission: None, attempt: int
) -> None:
    admission.status("open")
    release = asyncio.Event()
    ready = [asyncio.Event() for _ in range(4)]
    admitted: list[int] = []

    async def contender(index: int) -> None:
        try:
            async with admission.execution():
                admitted.append(index)
                ready[index].set()
                await release.wait()
        except admission.AdmissionError:
            ready[index].set()

    tasks = [asyncio.create_task(contender(index)) for index in range(4)]
    try:
        await asyncio.to_thread(admission.status, "close")
        async with asyncio.timeout(40):
            await asyncio.gather(*(event.wait() for event in ready))
        observed = await asyncio.to_thread(admission.status)
        assert observed.closed
        assert observed.active_slots == len(admitted)
        with pytest.raises(admission.AdmissionError):
            async with admission.execution():
                pytest.fail("new work crossed the closed boundary")
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert admission.status().active_slots == 0


async def test_cancellation_retains_real_occupancy(closed_admission: None) -> None:
    admission.status("open")
    entered = asyncio.Event()

    async def work() -> None:
        async with admission.execution():
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(work())
    try:
        async with asyncio.timeout(40):
            await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        admission.status("close")
        admission.ensure_closed()
        assert admission.status().active_slots == 1
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_refusal_holds_no_slot(
    closed_admission: None, caplog: pytest.LogCaptureFixture
) -> None:
    with pytest.raises(admission.AdmissionError) as refused:
        async with admission.execution():
            pytest.fail("closed gate admitted execution")
    assert "reconcile" not in str(refused.value)
    assert not [record for record in caplog.records if record.levelname == "ERROR"]
    assert admission.status().active_slots == 0


async def test_release_frees_held_slots_only_while_closed(closed_admission: None) -> None:
    admission.status("open")
    with pytest.raises(RuntimeError, match="external job may survive"):
        async with admission.execution():
            raise RuntimeError("external job may survive")
    with pytest.raises(admission.AdmissionError, match="Close admission"):
        admission.release(1)
    admission.status("close")
    with pytest.raises(admission.AdmissionError, match="holds 1 slots, not 2"):
        admission.release(2)
    assert admission.release(1).active_slots == 0
    admission.status("open")
