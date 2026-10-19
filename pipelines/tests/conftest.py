from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager

import pytest
from prefect.testing.utilities import prefect_test_harness

from openknowledge_core.ingestion import DownloadDatasetService
from openknowledge_pipelines import admission, bootstrap
from openknowledge_pipelines.settings import PipelineSettings
from tests.fakes.ports import PAPERS, PRODUCER, FakeSource, FakeStore, clock


@pytest.fixture(scope="session")
def prefect_server() -> Generator[None, None, None]:
    with prefect_test_harness():
        admission.ensure_closed()
        admission.status("open")
        yield


@pytest.fixture
def source() -> FakeSource:
    fake = FakeSource()
    for name, data in PAPERS.items():
        fake.add("papers", name, data)
    return fake


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def service(source: FakeSource, store: FakeStore) -> DownloadDatasetService:
    return DownloadDatasetService(
        catalog=source, downloads=source, store=store, producer=PRODUCER, clock=clock
    )


@pytest.fixture
def wired(
    monkeypatch: pytest.MonkeyPatch, service: DownloadDatasetService
) -> DownloadDatasetService:
    @asynccontextmanager
    async def open_service(settings: PipelineSettings) -> AsyncGenerator[DownloadDatasetService]:
        yield service

    @asynccontextmanager
    async def admitted() -> AsyncGenerator[None]:
        yield

    monkeypatch.setattr(admission, "execution", admitted)
    monkeypatch.setattr(bootstrap, "open_service", open_service)
    monkeypatch.setenv("S2_API_KEY", "test-key")
    monkeypatch.setenv("OKP_TASK_RETRY_DELAY_SECONDS", "0")
    return service
