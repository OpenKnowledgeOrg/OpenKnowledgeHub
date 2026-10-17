"""Scoped listing follows all pages without using a socket."""

from typing import cast
from unittest.mock import AsyncMock, Mock

from gcloud.aio.storage import Storage

from openknowledge_adapters.gcs import GcsArtifactStore
from openknowledge_contracts.manifests import ObjectPath


async def test_progress_listing_follows_page_token_and_keeps_scope() -> None:
    metadata = dict(bucket="raw", generation="1", size="2", crc32c="AAAAAA==")
    listing = AsyncMock(
        side_effect=[
            {
                "items": [{**metadata, "name": "s2/archive-progress/papers/diffs/a.json"}],
                "nextPageToken": "page-two",
            },
            {"items": [{**metadata, "name": "s2/archive-progress/papers/diffs/b.json"}]},
        ]
    )
    storage = Mock(spec=Storage)
    storage.list_objects = listing
    store = GcsArtifactStore(cast(Storage, storage), request_timeout=60, min_transfer_rate=1024)
    prefix = ObjectPath(bucket="raw", name="s2/archive-progress/papers/diffs")
    objects = await store.list_objects(prefix)
    assert [item.ref.path.name.rsplit("/", 1)[-1] for item in objects] == ["a.json", "b.json"]
    assert [call.kwargs["params"] for call in listing.await_args_list] == [
        {"prefix": prefix.name + "/"},
        {"prefix": prefix.name + "/", "pageToken": "page-two"},
    ]
