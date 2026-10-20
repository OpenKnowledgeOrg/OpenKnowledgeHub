"""SQL that loads one archived object into its dataset's table.

ClickHouse reads the object as text, one record per line, and checks each record against the
contract with JSON functions before it extracts a value, because its JSON input formats accept
missing fields, quoted numbers and overflowing integers. A failing record stops the insert with
``<table>: <field> breaks the contract`` (``throwIf``, code 395), which never holds the data.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

import pyarrow as pa

from openknowledge_contracts.tables import PROVENANCE, DatasetContract
from openknowledge_contracts.tables.clickhouse import NULL_FLAG_SUFFIX, table_columns
from openknowledge_contracts.tables.contract import REMOVED
from openknowledge_contracts.tables.fields import ArrowField, fields_of

type Source = Literal["s3", "file"]
type Check = tuple[str, str]

# Repeated keys a failing load names, at most.
EXAMPLES = 10

_RELEASE = "{source_release:Date}"
_PLAIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TYPES: dict[pa.DataType, str] = {
    pa.bool_(): "Bool",
    pa.string(): "String",
    pa.int8(): "Int8",
    pa.int16(): "Int16",
    pa.int32(): "Int32",
    pa.int64(): "Int64",
    pa.uint8(): "UInt8",
    pa.uint16(): "UInt16",
    pa.uint32(): "UInt32",
    pa.uint64(): "UInt64",
}
_RANGES: dict[pa.DataType, tuple[int, int]] = {
    pa.int8(): (-(2**7), 2**7 - 1),
    pa.int16(): (-(2**15), 2**15 - 1),
    pa.int32(): (-(2**31), 2**31 - 1),
    pa.uint8(): (0, 2**8 - 1),
    pa.uint16(): (0, 2**16 - 1),
    pa.uint32(): (0, 2**32 - 1),
}


def upsert_query(contract: DatasetContract, source: Source = "s3") -> str:
    """Insert every record of one object as a current row."""
    fields = fields_of(contract.source)
    return _load(contract, fields, [f.name for f in fields if not f.nullable], 0, source)


def delete_query(contract: DatasetContract, source: Source = "s3") -> str:
    """Insert a tombstone for every record of one object of delete records.

    A delete record holds the contract's key fields and nothing else, so the tombstone takes the
    table's defaults in every other column.
    """
    keys = [field for field in fields_of(contract.source) if field.name in contract.key]
    return _load(contract, keys, contract.key, 1, source)


def removed_query(contract: DatasetContract) -> str:
    """Flag every key that an older release holds and the loaded release no longer lists.

    Each such live row is copied with ``removed`` set and ``source_release`` moved to the loaded
    release, so a later release that lists the key again wins with ``removed`` unset.
    """
    if contract.updates != "snapshot":
        raise ValueError(f"{contract.name}: only a snapshot contract flags removed keys")
    columns = [column.name for column in table_columns(contract) if not column.materialized]
    replaced = {"source_release": _RELEASE, REMOVED: "true"}
    selected = ",\n    ".join(replaced.get(name, _identifier(name)) for name in columns)
    table = _identifier(contract.table)
    return (
        f"INSERT INTO {table} ({', '.join(_identifier(name) for name in columns)})\n"
        f"SELECT\n    {selected}\n"
        f"FROM {table} FINAL\n"
        f"WHERE source_release < {_RELEASE} AND NOT {REMOVED}"
    )


def problems_query(contract: DatasetContract, source: Source = "s3") -> str:
    """Every line of one object that fails the upsert's checks, with the first check it fails."""
    query = _Query(contract)
    fields = fields_of(contract.source)
    checks = query.record(fields, required=[f.name for f in fields if not f.nullable])
    checks += [query.field(field, _raw(field.name)) for field in fields]
    branches = ",\n        ".join(
        f"NOT ({condition}), {_literal(message)}" for condition, message in checks
    )
    return (
        f"SELECT line, problem FROM (\n{_with(fields)}"
        f"SELECT line, multiIf(\n        {branches},\n        ''\n    ) AS problem\n"
        f"FROM {_source(source)})\nWHERE problem != ''"
    )


def duplicates_query(contract: DatasetContract, objects: int, source: Source = "s3") -> str:
    """Keys that repeat across the objects ``source_file_0`` to ``source_file_<objects - 1>``.

    Records repeat identically when every stored value is the same, whatever the key order or
    spacing of their lines. The values are hashed as one tuple, because a null argument would
    make the hash null and ``min`` and ``max`` skip it. A contract's ``enriched`` field is hashed
    apart from the other values, so that its records may differ there alone.
    """
    query = _Query(contract)
    fields = fields_of(contract.source)
    values = dict(column for field in fields for column in _stored(query, field))
    key = ", ".join(values[name] for name in contract.key)
    lines = "\n    UNION ALL\n    ".join(
        f"SELECT line FROM {_source(source, f'source_file_{index}')}" for index in range(objects)
    )
    if contract.enriched is not None:
        return _enriched_duplicates(fields, values, key, lines, contract.enriched)
    records = (
        f"{_with(fields)}SELECT tuple({key}) AS record_key, "
        f"cityHash64(tuple({', '.join(values.values())})) AS record_hash\n"
        f"FROM (\n    {lines}\n)"
    )
    keys = (
        "SELECT record_key, count() AS n, min(record_hash) != max(record_hash) AS conflicting\n"
        f"FROM (\n{records}\n)\nGROUP BY record_key\nHAVING n > 1"
    )
    return (
        "SELECT\n"
        "    sumIf(n - 1, NOT conflicting) AS identical_rows,\n"
        "    countIf(conflicting) AS conflicting_keys,\n"
        f"    groupArrayIf({EXAMPLES})(toString(record_key), conflicting) AS examples\n"
        f"FROM (\n{keys}\n)"
    )


def _enriched_duplicates(
    fields: Sequence[ArrowField], values: dict[str, str], key: str, lines: str, enriched: str
) -> str:
    """Only a missing ``enriched`` value may yield to one agreed non-null value of the same key.

    Hashes only ever compare records of one key, so 64 bits suffice and keep the per-key state
    of a full release small. Nullable values stay inside tuples so aggregates cannot silently
    skip them. As with the other datasets, a collision could theoretically hide a conflict.
    """
    flag = enriched + NULL_FLAG_SUFFIX
    other = ", ".join(value for name, value in values.items() if name not in (enriched, flag))
    records = (
        f"{_with(fields)}SELECT tuple({key}) AS record_key,\n"
        f"    cityHash64(tuple({other})) AS other_hash,\n"
        f"    cityHash64(tuple({values[enriched]})) AS enriched_hash,\n"
        f"    {values[flag]} AS enriched_null\n"
        f"FROM (\n    {lines}\n)"
    )
    keys = (
        "SELECT record_key, count() AS n, countIf(enriched_null) AS nulls,\n"
        "    min(other_hash) != max(other_hash)\n"
        "        OR minIf(enriched_hash, NOT enriched_null)\n"
        "            != maxIf(enriched_hash, NOT enriched_null)\n"
        "        AS conflicting,\n"
        "    (nulls > 0 AND nulls < n) AS enriched\n"
        f"FROM (\n{records}\n) GROUP BY record_key HAVING n > 1"
    )
    return (
        "SELECT\n"
        "    sumIf(n - 1 - enriched, NOT conflicting) AS identical_rows,\n"
        "    countIf(enriched AND NOT conflicting) AS enriched_rows,\n"
        "    countIf(conflicting) AS conflicting_keys,\n"
        f"    groupArrayIf({EXAMPLES})(toString(record_key), conflicting) AS examples\n"
        f"FROM (\n{keys}\n)"
    )


def query_parameters(*, source_file: str, release: str, source_object: str) -> dict[str, str]:
    return {"source_file": source_file, "source_release": release, "source_object": source_object}


def duplicates_parameters(source_files: Sequence[str]) -> dict[str, str]:
    return {f"source_file_{index}": name for index, name in enumerate(source_files)}


def sql_sha256(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


class _Query:
    def __init__(self, contract: DatasetContract) -> None:
        self._contract = contract
        self._variables = 0

    def record(self, fields: Sequence[ArrowField], *, required: Sequence[str]) -> list[Check]:
        table = self._contract.table
        keys = "JSONExtractKeys(line)"
        return [
            ("JSONType(line) = 'Object'", f"{table}: a line is not a JSON object"),
            (f"length({keys}) = length(arrayDistinct({keys}))", f"{table}: a record repeats a key"),
            (
                f"hasAll({_array(f.name for f in fields)}, {keys})",
                f"{table}: a record has a key the contract does not declare",
            ),
            (f"hasAll({keys}, {_array(required)})", f"{table}: a record lacks a required field"),
        ]

    def field(self, field: ArrowField, raw: str) -> Check:
        message = f"{self._contract.table}: {field.name} breaks the contract"
        return self._check(field, raw, field.name), message

    def value(self, field: ArrowField, raw: str) -> str:
        """The value to store, where a null composite is stored empty."""
        type_ = field.type
        if isinstance(type_, pa.MapType):
            map_type = cast("pa.MapType[pa.DataType, pa.DataType]", type_)
            pairs = f"JSONExtractKeysAndValuesRaw({raw})"
            pair = self._variable()
            value = self.value(map_type.item_field, f"tupleElement({pair}, 2)")
            return (
                f"mapFromArrays(arrayMap({pair} -> tupleElement({pair}, 1), {pairs}), "
                f"arrayMap({pair} -> {value}, {pairs}))"
            )
        if isinstance(type_, pa.ListType):
            item = cast("pa.ListType[pa.DataType]", type_).value_field
            variable = self._variable()
            value = self.value(item, variable)
            return f"arrayMap({variable} -> {value}, JSONExtractArrayRaw({raw}))"
        if isinstance(type_, pa.StructType):
            values = [
                self.value(f, f"JSONExtractRaw({raw}, {_literal(f.name)})")
                for f in fields_of(type_)
            ]
            return f"tuple({', '.join(values)})"
        if isinstance(type_, pa.JsonType):
            return f"if(JSONType({raw}) = 'Null', NULL, {raw})" if field.nullable else raw
        name = _TYPES[_scalar(type_)]
        return f"JSONExtract({raw}, {_literal(f'Nullable({name})' if field.nullable else name)})"

    def _check(self, field: ArrowField, raw: str, path: str) -> str:
        condition = self._typed(field.type, raw, path)
        if path in self._contract.null_as_empty:
            condition = f"{condition} AND {self._not_empty(field.type, raw, path)}"
        if field.nullable:
            return f"(JSONType({raw}) = 'Null' OR ({condition}))"
        return f"({condition})"

    def _typed(self, type_: pa.DataType, raw: str, path: str) -> str:
        if isinstance(type_, pa.MapType):
            map_type = cast("pa.MapType[pa.DataType, pa.DataType]", type_)
            item = map_type.item_field
            pair = self._variable()
            value = self._check(item, f"tupleElement({pair}, 2)", f"{path}.{item.name}")
            keys = f"JSONExtractKeys({raw})"
            return (
                f"JSONType({raw}) = 'Object' AND length({keys}) = length(arrayDistinct({keys})) "
                f"AND arrayAll({pair} -> {value}, JSONExtractKeysAndValuesRaw({raw}))"
            )
        if isinstance(type_, pa.ListType):
            item = cast("pa.ListType[pa.DataType]", type_).value_field
            variable = self._variable()
            check = self._check(item, variable, f"{path}.{item.name}")
            return (
                f"JSONType({raw}) = 'Array' "
                f"AND arrayAll({variable} -> {check}, JSONExtractArrayRaw({raw}))"
            )
        if isinstance(type_, pa.StructType):
            return self._object(raw, fields_of(type_), path)
        if isinstance(type_, pa.JsonType):
            return f"JSONType({raw}) != 'Null'"
        scalar = _scalar(type_)
        if scalar == pa.string():
            return f"JSONType({raw}) = 'String'"
        if scalar == pa.bool_():
            return f"JSONType({raw}) = 'Bool'"
        if scalar == pa.int64():
            return f"JSONType({raw}) = 'Int64'"
        if scalar == pa.uint64():
            # JSONType says UInt64 only above the Int64 range, and neither for 2**64 and above.
            return (
                f"(JSONType({raw}) = 'UInt64' "
                f"OR JSONType({raw}) = 'Int64' AND JSONExtractInt({raw}) >= 0)"
            )
        low, high = _RANGES[scalar]
        return f"JSONType({raw}) = 'Int64' AND JSONExtractInt({raw}) BETWEEN {low} AND {high}"

    def _object(self, raw: str, fields: Sequence[ArrowField], path: str) -> str:
        keys = f"JSONExtractKeys({raw})"
        required = [f.name for f in fields if not f.nullable]
        parts = [
            f"JSONType({raw}) = 'Object'",
            f"length({keys}) = length(arrayDistinct({keys}))",
            f"hasAll({_array(f.name for f in fields)}, {keys})",
        ]
        if required:
            parts.append(f"hasAll({keys}, {_array(required)})")
        for field in fields:
            nested = f"JSONExtractRaw({raw}, {_literal(field.name)})"
            parts.append(self._check(field, nested, f"{path}.{field.name}"))
        return " AND ".join(parts)

    def _not_empty(self, type_: pa.DataType, raw: str, path: str) -> str:
        if isinstance(type_, pa.ListType):
            return f"length(JSONExtractArrayRaw({raw})) > 0"
        if isinstance(type_, pa.MapType):
            return f"length(JSONExtractKeys({raw})) > 0"
        raise TypeError(f"{self._contract.name}: an empty {type_} at {path} cannot stand for null")

    def _variable(self) -> str:
        self._variables += 1
        return f"x{self._variables}"


def _load(
    contract: DatasetContract,
    fields: Sequence[ArrowField],
    required: Sequence[str],
    is_deleted: int,
    source: Source,
) -> str:
    query = _Query(contract)
    checks = query.record(fields, required=required)
    values: list[tuple[str, str]] = []
    for field in fields:
        checks.append(query.field(field, _raw(field.name)))
        values += _stored(query, field)
    values += _provenance(is_deleted)
    return _insert(contract, fields, values, checks, source)


def _insert(
    contract: DatasetContract,
    fields: Sequence[ArrowField],
    values: Sequence[tuple[str, str]],
    checks: Sequence[Check],
    source: Source,
) -> str:
    columns = ", ".join(_identifier(name) for name, _ in values)
    selected = ",\n    ".join(expression for _, expression in values)
    where = "\n    AND ".join(
        f"NOT throwIf(NOT ({condition}), {_literal(message)})" for condition, message in checks
    )
    return (
        f"INSERT INTO {_identifier(contract.table)} ({columns})\n"
        # Refuse an unmigrated destination rather than checkpointing coerced NULLs.
        "SETTINGS insert_null_as_default = 0\n"
        f"{_with(fields)}"
        f"SELECT\n    {selected}\n"
        f"FROM {_source(source)}\n"
        f"WHERE {where}"
    )


def _with(fields: Sequence[ArrowField]) -> str:
    raws = ",\n    ".join(
        f"JSONExtractRaw(line, {_literal(f.name)}) AS {_raw(f.name)}" for f in fields
    )
    return f"WITH\n    {raws}\n"


def _stored(query: _Query, field: ArrowField) -> list[tuple[str, str]]:
    """The columns a field fills: its value, and whether a nullable composite was null."""
    raw = _raw(field.name)
    columns = [(field.name, query.value(field, raw))]
    if field.nullable and _is_composite(field.type):
        columns.append((field.name + NULL_FLAG_SUFFIX, f"JSONType({raw}) = 'Null'"))
    return columns


def _source(source: Source, parameter: str = "source_file") -> str:
    """The table function that reads the object a query parameter names."""
    name = "{" + parameter + ":String}"
    if source == "s3":
        return (
            f"s3(raw, filename = {name}, format = 'LineAsString', "
            "structure = 'line String', compression_method = 'gzip')"
        )
    return f"file({name}, 'LineAsString', 'line String', 'gzip')"


def _provenance(is_deleted: int) -> list[tuple[str, str]]:
    values = {
        "source_release": _RELEASE,
        "source_object": "{source_object:String}",
        "is_deleted": str(is_deleted),
    }
    return [(name, values[name]) for name in PROVENANCE.names]


def _scalar(type_: pa.DataType) -> pa.DataType:
    if isinstance(type_, pa.DictionaryType):
        dictionary = cast("pa.DictionaryType[Any, Any]", type_)
        type_ = dictionary.value_type
    if type_ not in _TYPES:
        raise TypeError(f"no load check for {type_}")
    return type_


def _is_composite(type_: pa.DataType) -> bool:
    return isinstance(type_, pa.ListType | pa.MapType | pa.StructType)


def _raw(name: str) -> str:
    if not _PLAIN_NAME.fullmatch(name):
        raise ValueError(f"field {name!r} cannot name a query alias")
    return f"raw_{name}"


def _identifier(name: str) -> str:
    return "`" + name.replace("\\", "\\\\").replace("`", "\\`") + "`"


def _literal(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _array(names: Iterable[str]) -> str:
    return "[" + ", ".join(_literal(name) for name in names) + "]"
