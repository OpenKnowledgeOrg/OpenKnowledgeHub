"""The Semantic Scholar Datasets API and its signed file host behind one httpx MockTransport."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from pydantic import HttpUrl, SecretStr

from openknowledge_adapters.semantic_scholar import SemanticScholarConfig

API_HOST = "api.semanticscholar.org"
FILE_HOST = "files.test"
KEY = "test-key"
LINK_TTL = 3600


def s2_config(key: str = KEY) -> SemanticScholarConfig:
    return SemanticScholarConfig(
        api_key=SecretStr(key),
        api_url=HttpUrl(f"https://{API_HOST}/datasets/v1/"),
        request_interval_seconds=1.0,
        attempts=8,
        backoff_seconds=1.0,
        timeout_seconds=30.0,
        download_connect_timeout_seconds=30.0,
        download_read_timeout_seconds=300.0,
    )


class FakeClock:
    """Monotonic time that moves only when something sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class CutStream(httpx.AsyncByteStream):
    """A response body that breaks off after its first piece."""

    def __init__(self, head: bytes) -> None:
        self.head = head

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.head
        raise httpx.ReadError("connection lost")


class FakeS2:
    def __init__(self) -> None:
        self.releases = ["2026-09-29", "2026-10-06"]
        self.files: dict[str, dict[str, bytes]] = {}
        self.extra_fields: dict[str, object] = {"license": "ODC-BY"}
        self.now = 1_000_000
        self.api_calls: list[str] = []
        self.keys_sent: list[str | None] = []
        self.fail_next: list[int | Exception | httpx.Response] = []
        self.cut_next: set[str] = set()
        self.diffs: dict[str, list[tuple[str, str, list[str], list[str]]]] = {}
        self.transport = httpx.MockTransport(self.handle)

    def add(self, dataset: str, name: str, data: bytes) -> None:
        self.files.setdefault(dataset, {})[name] = data

    def sign(self, dataset: str, name: str, release: str) -> str:
        return self.sign_path(f"staging/{release}/{dataset}/{name}")

    def sign_path(self, path: str) -> str:
        return (
            f"https://{FILE_HOST}/{path}"
            f"?AWSAccessKeyId=AKIAFAKE&Signature=c2lnbmF0dXJl&Expires={self.now + LINK_TTL}"
        )

    def expire_links(self) -> None:
        self.now += LINK_TTL + 1

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == API_HOST:
            return self._api(request)
        if request.url.host == FILE_HOST:
            return self._file(request)
        return httpx.Response(502)

    def _api(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/datasets/v1/")
        self.api_calls.append(request.url.raw_path.decode().removeprefix("/datasets/v1/"))
        key = request.headers.get("x-api-key")
        self.keys_sent.append(key)
        if self.fail_next:
            failure = self.fail_next.pop(0)
            if isinstance(failure, Exception):
                raise failure
            if isinstance(failure, httpx.Response):
                return failure
            return httpx.Response(failure)
        parts = path.strip("/").split("/")
        if parts == ["release"]:
            return httpx.Response(200, json=self.releases)
        if len(parts) == 2 and parts[0] == "release":
            release = self.releases[-1] if parts[1] == "latest" else parts[1]
            if release not in self.releases:
                return httpx.Response(404, json={"error": f"No release {release}"})
            datasets = [
                {"name": name, "description": f"The {name} dataset", "README": f"About {name}"}
                for name in self.files
            ]
            body = {"release_id": release, "README": "Terms of use", "datasets": datasets}
            return httpx.Response(200, json={**body, **self.extra_fields})
        if len(parts) == 4 and parts[0] == "release" and parts[2] == "dataset":
            if key != KEY:
                return httpx.Response(403, json={"message": "Forbidden"})
            release, dataset = parts[1], parts[3]
            if release not in self.releases or dataset not in self.files:
                error = f"No dataset named {dataset} in release {release}"
                return httpx.Response(404, json={"error": error})
            files = [self.sign(dataset, name, release) for name in self.files[dataset]]
            body = {"name": dataset, "description": "", "README": f"About {dataset}"}
            return httpx.Response(200, json={**body, **self.extra_fields, "files": files})
        if len(parts) == 5 and parts[0] == "diffs" and parts[2] == "to":
            if key != KEY:
                return httpx.Response(403, json={"message": "Forbidden"})
            start, end, dataset = parts[1], parts[3], parts[4]
            end = self.releases[-1] if end == "latest" else end
            diffs = [
                {
                    "from_release": old,
                    "to_release": new,
                    "update_files": [
                        self.sign_path(f"updates/{old}-to-{new}/{dataset}/{name}")
                        for name in updates
                    ],
                    "delete_files": [
                        self.sign_path(f"deletes/{old}-to-{new}/{dataset}/{name}")
                        for name in deletes
                    ],
                }
                for old, new, updates, deletes in self.diffs.get(dataset, [])
                if start <= old and new <= end
            ]
            body = {"dataset": dataset, "start_release": start, "end_release": end, "diffs": diffs}
            return httpx.Response(200, json=body)
        return httpx.Response(404, json={"error": "unknown path"})

    def _file(self, request: httpx.Request) -> httpx.Response:
        segments = request.url.path.split("/")
        dataset, name = segments[-2], segments[-1]
        if int(request.url.params["Expires"]) < self.now:
            return httpx.Response(403, text="<Error><Code>AccessDenied</Code></Error>")
        data = self.files[dataset][name]
        if name in self.cut_next:
            self.cut_next.discard(name)
            return httpx.Response(200, stream=CutStream(data[: len(data) // 2]))
        return httpx.Response(
            200, headers={"content-length": str(len(data))}, stream=httpx.ByteStream(data)
        )
