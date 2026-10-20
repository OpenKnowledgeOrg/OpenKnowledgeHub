"""The loader's side of ClickHouse, over its HTTP interface."""

from __future__ import annotations

from collections.abc import Sequence

from openknowledge_adapters.clickhouse.client import ClickHouseClient
from openknowledge_adapters.clickhouse.load_sql import (
    delete_query,
    duplicates_parameters,
    duplicates_query,
    query_parameters,
    removed_query,
    sql_sha256,
    upsert_query,
)
from openknowledge_contracts.datasets import DatasetName, ReleaseId
from openknowledge_contracts.manifests import ObjectPath
from openknowledge_contracts.tables import DatasetContract
from openknowledge_contracts.tables.clickhouse import version_expression
from openknowledge_contracts.tables.contract import REMOVED
from openknowledge_contracts.tables.s2 import CONTRACTS
from openknowledge_core.ingestion.errors import TableRejected
from openknowledge_core.ingestion.models import (
    AppliedRelease,
    Duplicates,
    LoadObject,
    LoadRecord,
    TableState,
)

RECORD_COLUMNS = ", ".join(LoadRecord.model_fields)
APPLIED_COLUMNS = ", ".join(AppliedRelease.model_fields)


class ClickHouseTables:
    def __init__(self, client: ClickHouseClient, *, raw_bucket: str) -> None:
        self._client = client
        self._raw_bucket = raw_bucket

    async def table(self, dataset: DatasetName) -> TableState:
        contract = _contract(dataset)
        rows = await self._client.rows(
            "SELECT uuid, total_rows FROM system.tables "
            "WHERE database = currentDatabase() AND name = {table:String}",
            table=contract.table,
        )
        if not rows:
            raise TableRejected(f"table {contract.table} does not exist; run the migrations")
        if contract.enriched is not None:
            version = await self._client.rows(
                "SELECT default_expression FROM system.columns "
                "WHERE database = currentDatabase() AND table = {table:String} "
                "AND name = 'version'",
                table=contract.table,
            )
            if not version or version[0]["default_expression"] != version_expression(contract):
                raise TableRejected(
                    f"{contract.table} needs the version migration that ranks "
                    f"{contract.enriched}; a populated legacy table must be preserved and "
                    "swapped by hand, as migration 00014 describes"
                )
        return TableState(
            name=contract.table,
            uuid=rows[0]["uuid"],
            has_rows=bool(rows[0]["total_rows"]),
            load_sql_sha256=sql_sha256(upsert_query(contract)),
            delete_sql_sha256=sql_sha256(delete_query(contract)),
            updates=contract.updates,
        )

    async def loaded(self, dataset: DatasetName, source_object: ObjectPath) -> LoadRecord | None:
        rows = await self._client.rows(
            f"SELECT {RECORD_COLUMNS} FROM loads FINAL "
            "WHERE target = {target:String} AND source_object = {source_object:String}",
            target=_contract(dataset).table,
            source_object=str(source_object),
        )
        return LoadRecord.model_validate(rows[0]) if rows else None

    async def load(self, request: LoadObject) -> int:
        contract = _contract(request.dataset)
        path = self._readable(request.object_ref.path)
        query = delete_query(contract) if request.operation == "delete" else upsert_query(contract)
        parameters = query_parameters(
            source_file=path.name, release=request.release, source_object=str(path)
        )
        return await self._client.written(query, **parameters)

    async def record(self, record: LoadRecord) -> None:
        await self._client.post(
            "INSERT INTO loads FORMAT JSONEachRow", data=record.model_dump_json()
        )

    async def live_rows(self, dataset: DatasetName, release: ReleaseId) -> int:
        contract = _contract(dataset)
        query = (
            "SELECT count() AS rows FROM {table:Identifier} FINAL "
            "WHERE source_release = {release:Date}"
        )
        if contract.updates == "snapshot":
            query += f" AND NOT {REMOVED}"
        rows = await self._client.rows(query, table=contract.table, release=release)
        return int(rows[0]["rows"])

    async def duplicates(self, dataset: DatasetName, objects: Sequence[ObjectPath]) -> Duplicates:
        names = [self._readable(path).name for path in objects]
        rows = await self._client.rows(
            duplicates_query(_contract(dataset), len(names)), **duplicates_parameters(names)
        )
        return Duplicates.model_validate(rows[0])

    async def mark_removed(self, dataset: DatasetName, release: ReleaseId) -> int:
        return await self._client.written(removed_query(_contract(dataset)), source_release=release)

    def _readable(self, path: ObjectPath) -> ObjectPath:
        if path.bucket != self._raw_bucket:
            raise TableRejected(f"ClickHouse reads only {self._raw_bucket}, not {path}")
        return path


class ClickHouseAppliedReleases:
    def __init__(self, client: ClickHouseClient) -> None:
        self._client = client

    async def record(self, applied: AppliedRelease) -> None:
        await self._client.post(
            "INSERT INTO applied FORMAT JSONEachRow", data=applied.model_dump_json()
        )

    async def latest(self, target: str) -> AppliedRelease | None:
        rows = await self._client.rows(
            f"SELECT {APPLIED_COLUMNS} FROM applied "
            "WHERE target = {target:String} ORDER BY applied_at DESC LIMIT 1",
            target=target,
        )
        return AppliedRelease.model_validate(rows[0]) if rows else None


def _contract(dataset: DatasetName) -> DatasetContract:
    try:
        return CONTRACTS[dataset]
    except KeyError:
        raise TableRejected(f"no table contract for dataset {dataset}") from None
