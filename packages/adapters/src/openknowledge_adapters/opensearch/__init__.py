"""OpenSearch over opensearch-py's asyncio client, the only package that speaks its API."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from opensearchpy import AsyncConnection, AsyncHttpConnection, AsyncOpenSearch

from openknowledge_adapters.opensearch.config import OpenSearchConfig
from openknowledge_adapters.opensearch.sink import OpenSearchSink


@asynccontextmanager
async def connect(
    config: OpenSearchConfig, *, connection_class: type[AsyncConnection] = AsyncHttpConnection
) -> AsyncGenerator[OpenSearchSink, None]:
    """One client per run, closed with its sessions when the run ends.

    The credentials go into an Authorization header set once on the client, because
    opensearch-py 3.2 hands ``http_auth`` to aiohttp as the ``auth`` argument that aiohttp 3.14
    deprecates. pydantic keeps the brackets of an IPv6 literal, which opensearch-py adds itself.
    """
    url = config.url
    path = url.path or ""
    credentials = f"{config.user}:{config.password.get_secret_value()}".encode()
    client = AsyncOpenSearch(
        hosts=[
            {
                "host": (url.host or "").strip("[]"),
                "port": url.port,
                "use_ssl": url.scheme == "https",
                "url_prefix": "" if path == "/" else path,
            }
        ],
        connection_class=connection_class,
        headers={"Authorization": "Basic " + b64encode(credentials).decode()},
        verify_certs=True,
        http_compress=True,
        timeout=config.timeout_seconds,
    )
    try:
        yield OpenSearchSink(client)
    finally:
        await client.close()


__all__ = ["OpenSearchConfig", "OpenSearchSink", "connect"]
