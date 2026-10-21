"""ClickHouse over HTTP, the only package that sends SQL."""

from __future__ import annotations

import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import NamedTuple

import httpx

from openknowledge_adapters.clickhouse.client import ClickHouseClient
from openknowledge_adapters.clickhouse.config import ClickHouseConfig
from openknowledge_adapters.clickhouse.search import ClickHouseSearchSource
from openknowledge_adapters.clickhouse.tables import ClickHouseAppliedReleases, ClickHouseTables
from openknowledge_adapters.clickhouse.telemetry import ClickHouseLoadTelemetry


def keepalive_options() -> list[tuple[int, int, int]]:
    """Probe idle inserts before GCP drops their connection, using this platform's options."""
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle = getattr(socket, "TCP_KEEPIDLE", None)
    if idle is None:
        idle = getattr(socket, "TCP_KEEPALIVE", None)
    if idle is not None:
        options.append((socket.IPPROTO_TCP, idle, 60))
    interval = getattr(socket, "TCP_KEEPINTVL", None)
    if interval is not None:
        options.append((socket.IPPROTO_TCP, interval, 30))
    return options


class ClickHouse(NamedTuple):
    tables: ClickHouseTables
    applied: ClickHouseAppliedReleases
    telemetry: ClickHouseLoadTelemetry
    search: ClickHouseSearchSource


@asynccontextmanager
async def connect(
    config: ClickHouseConfig, *, transport: httpx.AsyncBaseTransport | None = None
) -> AsyncGenerator[ClickHouse, None]:
    http = httpx.AsyncClient(
        base_url=str(config.url),
        headers={
            "X-ClickHouse-User": config.user,
            "X-ClickHouse-Key": config.password.get_secret_value(),
        },
        timeout=httpx.Timeout(config.timeout_seconds, connect=10.0),
        follow_redirects=False,
        transport=transport or httpx.AsyncHTTPTransport(socket_options=keepalive_options()),
    )
    async with http:
        client = ClickHouseClient(http, database=config.database)
        yield ClickHouse(
            tables=ClickHouseTables(client, raw_bucket=config.raw_bucket),
            applied=ClickHouseAppliedReleases(client),
            telemetry=ClickHouseLoadTelemetry(client),
            search=ClickHouseSearchSource(client),
        )


__all__ = [
    "ClickHouse",
    "ClickHouseAppliedReleases",
    "ClickHouseConfig",
    "ClickHouseSearchSource",
    "ClickHouseTables",
    "connect",
]
