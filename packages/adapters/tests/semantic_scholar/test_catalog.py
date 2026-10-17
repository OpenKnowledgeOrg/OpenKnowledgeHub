import httpx
import pytest

from openknowledge_adapters.semantic_scholar import SemanticScholar, connect
from openknowledge_core.ingestion.errors import (
    CredentialRejected,
    SourceRejected,
    SourceUnavailable,
)
from tests.fakes.s2_http import KEY, FakeClock, FakeS2, s2_config


async def test_latest_resolves_to_an_explicit_release_without_the_key(
    client: SemanticScholar, s2: FakeS2
) -> None:
    metadata = await client.catalog.resolve_release("latest")
    assert metadata.release_id == "2026-10-06"
    assert metadata.model_dump()["license"] == "ODC-BY"
    assert (await client.catalog.resolve_release("2026-09-29")).release_id == "2026-09-29"
    assert s2.keys_sent == [None, None]


async def test_a_listing_keeps_signed_links_out_of_its_metadata(
    client: SemanticScholar, s2: FakeS2
) -> None:
    s2.add("papers", "part-0.gz", b"")
    s2.add("papers", "part-1.gz", b"")
    listing = await client.catalog.list_dataset("2026-10-06", "papers")
    assert [link.shard.file_name for link in listing.links] == ["part-0.gz", "part-1.gz"]
    assert listing.links[0].shard.source_path == "/staging/2026-10-06/papers/part-0.gz"
    assert listing.links[0].url.get_secret_value().startswith("https://files.test/staging/")
    assert "Signature" not in repr(listing) and "Signature" not in listing.model_dump_json()
    summary = listing.summary.model_dump()
    assert "files" not in summary
    assert (summary["README"], summary["license"]) == ("About papers", "ODC-BY")
    assert s2.keys_sent == [KEY]


async def test_a_diff_chain_keeps_its_signed_links_secret(
    client: SemanticScholar, s2: FakeS2
) -> None:
    s2.releases.append("2026-10-13")
    s2.diffs["tldrs"] = [
        ("2026-09-29", "2026-10-06", ["u-0.gz", "u-1.gz"], ["d-0.gz"]),
        ("2026-10-06", "2026-10-13", ["u-2.gz"], []),
    ]
    chain = await client.catalog.list_diffs("2026-09-29", "latest", "tldrs")
    steps = [(diff.from_release, diff.release_id) for diff in chain.diffs]
    assert steps == [("2026-09-29", "2026-10-06"), ("2026-10-06", "2026-10-13")]
    first = chain.diffs[0]
    assert [link.shard.source_path for link in first.deletes] == [
        "/deletes/2026-09-29-to-2026-10-06/tldrs/d-0.gz"
    ]
    assert [link.shard.file_name for link in first.updates] == ["u-0.gz", "u-1.gz"]
    assert "Signature" not in repr(chain) and "Signature" not in chain.model_dump_json()
    assert (s2.api_calls, s2.keys_sent) == (["diffs/2026-09-29/to/latest/tldrs"], [KEY])


async def test_a_rejected_key_is_not_retried_and_never_shown(s2: FakeS2, clock: FakeClock) -> None:
    config = s2_config(key="wrong-key")
    async with connect(config, transport=s2.transport, clock=clock, sleep=clock.sleep) as client:
        with pytest.raises(CredentialRejected) as info:
            await client.catalog.list_dataset("2026-10-06", "papers")
    assert "wrong-key" not in str(info.value)
    assert "S2_API_KEY with HTTP 403" in str(info.value)
    assert len(s2.api_calls) == 1


async def test_an_unknown_dataset_is_a_plain_error(client: SemanticScholar) -> None:
    with pytest.raises(SourceRejected) as info:
        await client.catalog.list_dataset("2026-10-06", "nothing")
    assert str(info.value) == (
        "Semantic Scholar has no dataset nothing in release 2026-10-06; "
        "check the release id and dataset name"
    )


async def test_one_request_per_second(client: SemanticScholar, clock: FakeClock) -> None:
    for _ in range(3):
        await client.catalog.resolve_release("latest")
    assert clock.sleeps == [1.0, 1.0]


async def test_retries_wait_with_awaited_backoff(
    client: SemanticScholar, s2: FakeS2, clock: FakeClock
) -> None:
    s2.fail_next = [429, httpx.ConnectError("reset"), 503]
    assert (await client.catalog.resolve_release("latest")).release_id == "2026-10-06"
    assert clock.sleeps == [2.0, 4.0, 8.0]
    assert len(s2.api_calls) == 4


async def test_gives_up_within_about_four_minutes(
    client: SemanticScholar, s2: FakeS2, clock: FakeClock
) -> None:
    s2.fail_next = [429] * 8
    with pytest.raises(SourceUnavailable, match="8 times, the last time with HTTP 429"):
        await client.catalog.resolve_release("latest")
    assert len(s2.api_calls) == 8
    assert clock.sleeps == [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    assert sum(clock.sleeps) == 254.0


async def test_retry_after_is_honoured(
    client: SemanticScholar, s2: FakeS2, clock: FakeClock
) -> None:
    s2.fail_next = [httpx.Response(429, headers={"retry-after": "7"})]
    await client.catalog.resolve_release("latest")
    assert clock.sleeps == [7.0]


async def test_a_retry_waits_a_random_part_of_its_backoff_and_says_so(
    s2: FakeS2, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    s2.fail_next = [httpx.Response(429, headers={"retry-after": "7"}), 503]
    async with connect(
        s2_config(),
        transport=s2.transport,
        clock=clock,
        sleep=clock.sleep,
        uniform=lambda low, high: low,
    ) as client:
        await client.catalog.resolve_release("latest")
    # Retry-After is a floor, and the rate limiter still spaces the calls.
    assert clock.sleeps == [7.0, 0.0, 1.0]
    assert [record.getMessage() for record in caplog.records] == [
        "Semantic Scholar request for release latest failed with HTTP 429; retrying in 7.0s",
        "Semantic Scholar request for release latest failed with HTTP 503; retrying in 0.0s",
    ]


async def test_a_body_that_is_not_json_is_an_error(client: SemanticScholar, s2: FakeS2) -> None:
    s2.fail_next = [httpx.Response(200, text="<html>maintenance</html>")]
    with pytest.raises(SourceRejected, match="not JSON"):
        await client.catalog.resolve_release("latest")


async def test_metadata_without_the_expected_fields_is_refused_without_echo(
    client: SemanticScholar, s2: FakeS2
) -> None:
    s2.fail_next = [httpx.Response(200, json={"release_id": "topsecret"})]
    with pytest.raises(SourceRejected, match="missing or malformed fields") as info:
        await client.catalog.resolve_release("latest")
    assert "topsecret" not in str(info.value)
    assert info.value.__context__ is None


async def test_ids_are_quoted_into_the_path(client: SemanticScholar, s2: FakeS2) -> None:
    with pytest.raises(SourceRejected, match="has no dataset"):
        await client.catalog.list_dataset("2026-10-06/../x", "papers?fields=all")
    assert s2.api_calls == ["release/2026-10-06%2F..%2Fx/dataset/papers%3Ffields%3Dall"]


@pytest.mark.parametrize("release", ["", ".", ".."])
async def test_dots_and_empty_ids_are_refused(client: SemanticScholar, release: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty or dots"):
        await client.catalog.list_dataset(release, "papers")


@pytest.mark.parametrize(
    "url",
    [
        "http://files.test/staging/r/d/part-0.gz?x=1",
        "https://files.test/staging/r/d/",
        "https://files.test/staging/r/d/..",
        "https://files.test/staging/r/d/part\u200b0.gz",
        "https://files.test/staging/r/d/.hidden.gz",
        "https://files.test/staging/r/d/part 0.gz",
        "https://files.test/staging/r/d/part-0.gz?Signature=s\u00a0x",
        "https://files.test/staging/r/d/part-0.gz?Signature=s\u202ex",
        "https://files.test/staging/r/d/dataset.json",
        "https://[::1/staging/part-0.gz",
    ],
)
async def test_links_must_be_https_with_a_plain_name_and_printable(
    client: SemanticScholar, s2: FakeS2, url: str
) -> None:
    body = {"name": "papers", "description": "", "README": "", "files": [url]}
    s2.fail_next = [httpx.Response(200, json=body)]
    with pytest.raises(SourceRejected, match="plain file name"):
        await client.catalog.list_dataset("2026-10-06", "papers")


async def test_a_listing_for_another_dataset_name_is_refused(
    client: SemanticScholar, s2: FakeS2
) -> None:
    s2.fail_next = [httpx.Response(200, json={"name": "Not A Name", "files": []})]
    with pytest.raises(SourceRejected, match="missing or malformed fields"):
        await client.catalog.list_dataset("2026-10-06", "papers")


async def test_a_body_that_does_not_decode_is_refused_at_once_and_leaks_no_key(
    client: SemanticScholar, s2: FakeS2
) -> None:
    s2.add("papers", "part-0.gz", b"")
    body = httpx.ByteStream(b"not gzip at all")
    s2.fail_next = [httpx.Response(200, headers={"content-encoding": "gzip"}, stream=body)]
    with pytest.raises(SourceRejected, match="DecodingError") as info:
        await client.catalog.list_dataset("2026-10-06", "papers")
    assert len(s2.api_calls) == 1
    assert KEY not in str(info.value) and KEY not in repr(info.value)
    assert info.value.__context__ is None


async def test_a_body_nested_beyond_reason_is_refused(client: SemanticScholar, s2: FakeS2) -> None:
    s2.fail_next = [httpx.Response(200, content=b"[" * 100_000 + b"]" * 100_000)]
    with pytest.raises(SourceRejected, match="not JSON"):
        await client.catalog.resolve_release("latest")


async def test_retry_after_is_capped(client: SemanticScholar, s2: FakeS2, clock: FakeClock) -> None:
    s2.fail_next = [httpx.Response(429, headers={"retry-after": "86400"})]
    await client.catalog.resolve_release("latest")
    assert clock.sleeps == [120.0]


async def test_release_listing_validates_ids_and_preserves_source_errors(
    client: SemanticScholar,
    s2: FakeS2,
) -> None:
    assert await client.catalog.list_releases() == tuple(s2.releases)
    assert s2.keys_sent == [None]
    s2.fail_next = [httpx.Response(200, json=["malformed"])]
    with pytest.raises(SourceRejected, match="malformed release IDs"):
        await client.catalog.list_releases()
    s2.fail_next = [httpx.Response(403)]
    with pytest.raises(SourceRejected):
        await client.catalog.list_releases()
