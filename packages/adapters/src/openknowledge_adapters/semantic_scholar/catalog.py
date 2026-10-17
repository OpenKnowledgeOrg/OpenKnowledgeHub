"""Semantic Scholar Datasets API client (https://api.semanticscholar.org/api-docs/datasets)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Literal
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, TypeAdapter, ValidationError

from openknowledge_adapters.semantic_scholar.rate_limit import AsyncRateLimiter
from openknowledge_contracts.datasets import (
    DatasetName,
    DatasetSummary,
    ReleaseId,
    ReleaseMetadata,
    ShardRef,
)
from openknowledge_core.ingestion.errors import (
    TRANSIENT_STATUSES,
    CredentialRejected,
    SourceRejected,
    SourceUnavailable,
)
from openknowledge_core.ingestion.models import DatasetListing, DiffChain, DiffListing, ShardLink

log = logging.getLogger(__name__)

KEY_REFUSED_STATUSES = frozenset({401, 403})
AGAIN = "rerun later, and report it to Semantic Scholar if it repeats"
MAX_RETRY_AFTER_SECONDS = 120.0


class _DatasetBody(BaseModel):
    """The dataset endpoint's answer, whose links' query strings leave only inside SecretStr."""

    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)

    name: str
    files: list[str]


class _DiffBody(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)

    from_release: str
    to_release: str
    update_files: list[str]
    delete_files: list[str]


class _DiffChainBody(BaseModel):
    """The diffs endpoint's answer, whose links leave only inside SecretStr, like a listing's."""

    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)

    dataset: str
    diffs: list[_DiffBody]


class SemanticScholarCatalog:
    """Release metadata and dataset listings, spaced by the shared rate limiter."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        limiter: AsyncRateLimiter,
        *,
        api_key: SecretStr,
        attempts: int,
        backoff: float,
        sleep: Callable[[float], Awaitable[None]],
        uniform: Callable[[float, float], float],
    ) -> None:
        self._http = http
        self._limiter = limiter
        self._key = api_key
        self._attempts = attempts
        self._backoff = backoff
        self._sleep = sleep
        self._uniform = uniform

    async def list_releases(self) -> tuple[ReleaseId, ...]:
        body = await self._get("release", "available releases", authenticated=False)
        try:
            return TypeAdapter(tuple[ReleaseId, ...]).validate_python(body)
        except ValidationError:
            raise SourceRejected("Semantic Scholar sent malformed release IDs") from None

    async def resolve_release(self, selector: ReleaseId | Literal["latest"]) -> ReleaseMetadata:
        about = f"release {selector}"
        body = await self._get(_path("release", selector), about, authenticated=False)
        try:
            return ReleaseMetadata.model_validate(body)
        except ValidationError:
            pass
        raise SourceRejected(
            f"the metadata Semantic Scholar sent for {about} has missing or malformed fields; "
            f"{AGAIN}"
        )

    async def list_dataset(self, release_id: ReleaseId, dataset: DatasetName) -> DatasetListing:
        about = f"dataset {dataset} in release {release_id}"
        path = _path("release", release_id, "dataset", dataset)
        body = await self._get(path, about, authenticated=True)
        try:
            parsed = _DatasetBody.model_validate(body)
            summary = DatasetSummary.model_validate(parsed.model_dump(exclude={"files"}))
        except ValidationError:
            parsed = summary = None
        if parsed is None or summary is None:
            raise SourceRejected(
                f"the listing Semantic Scholar sent for {about} has missing or malformed fields; "
                f"{AGAIN}"
            )
        return DatasetListing(summary=summary, links=tuple(_link(url) for url in parsed.files))

    async def list_diffs(
        self, start: ReleaseId, end: ReleaseId | Literal["latest"], dataset: DatasetName
    ) -> DiffChain:
        about = f"diffs of {dataset} from {start} to {end}"
        body = await self._get(_path("diffs", start, "to", end, dataset), about, authenticated=True)
        try:
            parsed = _DiffChainBody.model_validate(body)
            chain = DiffChain(
                dataset=parsed.dataset,
                diffs=tuple(
                    DiffListing(
                        from_release=diff.from_release,
                        release_id=diff.to_release,
                        updates=tuple(_link(url) for url in diff.update_files),
                        deletes=tuple(_link(url) for url in diff.delete_files),
                    )
                    for diff in parsed.diffs
                ),
            )
        except ValidationError:
            chain = None
        if chain is None:
            raise SourceRejected(
                f"the {about} Semantic Scholar sent have missing or malformed fields; {AGAIN}"
            )
        return chain

    async def _get(self, path: str, about: str, *, authenticated: bool) -> object:
        headers = {"x-api-key": self._key.get_secret_value()} if authenticated else {}
        last = ""
        for attempt in range(1, self._attempts + 1):
            await self._limiter.wait()
            retry_after = 0.0
            unreadable = ""
            try:
                response = await self._http.get(path, headers=headers)
            except httpx.TransportError as exc:
                last = type(exc).__name__
            except httpx.HTTPError as exc:
                # Its request holds the key, so only the error's name leaves this module.
                unreadable = type(exc).__name__
            else:
                if response.is_success:
                    return _json(response, about)
                status = response.status_code
                last = f"HTTP {status}"
                if authenticated and status in KEY_REFUSED_STATUSES:
                    raise CredentialRejected(
                        f"Semantic Scholar refused S2_API_KEY with HTTP {status}; check the key "
                        "in ~/.config/openknowledge/.env, and request a new one if it sat unused "
                        "for about 60 days"
                    )
                if status == 404:
                    raise SourceRejected(
                        f"Semantic Scholar has no {about}; check the release id and dataset name"
                    )
                if status not in TRANSIENT_STATUSES:
                    raise SourceRejected(
                        f"Semantic Scholar refused the request for {about} with HTTP {status}; "
                        f"{AGAIN}"
                    )
                retry_after = _retry_after(response)
            if unreadable:
                raise SourceRejected(
                    f"Semantic Scholar's answer for {about} could not be read ({unreadable}); "
                    "rerun later"
                )
            if attempt < self._attempts:
                # Full jitter, so tasks that hit a 429 together do not retry together.
                wait = max(retry_after, self._uniform(0, self._backoff * 2**attempt))
                log.warning(
                    "Semantic Scholar request for %s failed with %s; retrying in %.1fs",
                    about,
                    last,
                    wait,
                )
                await self._sleep(wait)
        raise SourceUnavailable(
            f"Semantic Scholar failed the request for {about} {self._attempts} times, "
            f"the last time with {last}; rerun later"
        )


def _json(response: httpx.Response, about: str) -> object:
    try:
        body: object = response.json()
    except (ValueError, RecursionError):
        body = None
    if body is None:
        raise SourceRejected(f"Semantic Scholar's answer for {about} is not JSON; rerun later")
    return body


def _link(url: str) -> ShardLink:
    """Accept only https links with plain file names, since the name reaches the bucket and logs."""
    try:
        parts = urlsplit(url)
        name = parts.path.rsplit("/", 1)[-1]
        plain = parts.scheme == "https" and url.isprintable()
        shard = ShardRef(source_path=parts.path, file_name=name) if plain else None
    except ValueError:
        shard = None
    if shard is None:
        raise SourceRejected(
            f"a file link in the listing is not an https URL with a plain file name; {AGAIN}"
        )
    return ShardLink(shard=shard, url=SecretStr(url))


def _path(*segments: str) -> str:
    """Join path segments, percent-encoding each so an id cannot change the route."""
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError("release ids and dataset names cannot be empty or dots")
    return "/".join(quote(segment, safe="") for segment in segments)


def _retry_after(response: httpx.Response) -> float:
    """The Retry-After delay in whole seconds, capped so one answer cannot stall a run."""
    value = response.headers.get("retry-after", "")
    return min(float(value), MAX_RETRY_AFTER_SECONDS) if value.isdecimal() else 0.0
