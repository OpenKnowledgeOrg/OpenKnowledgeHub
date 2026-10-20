"""The ClickHouse columns a contract's table must have.

Migrations are written by hand. This module says what they must produce, and a test compares
the two through ``system.columns``, so types and expressions are spelled the way ClickHouse
prints them.

ClickHouse cannot wrap an array, map or tuple in ``Nullable``. A nullable one at the top of a
record stores an empty value in its column and the null in a ``Bool`` column named
``<field>_is_null`` right after it. Inside another value there is no room for a flag, so such a
path must be listed in the contract's ``null_as_empty``. JSON text stays a ``String``; readers
parse it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pyarrow as pa

from openknowledge_contracts.tables.contract import REMOVED, VERSION, DatasetContract
from openknowledge_contracts.tables.fields import ArrowField, fields_of

NULL_FLAG_SUFFIX = "_is_null"

_SCALARS: dict[pa.DataType, str] = {
    pa.bool_(): "Bool",
    pa.int8(): "Int8",
    pa.int16(): "Int16",
    pa.int32(): "Int32",
    pa.int64(): "Int64",
    pa.uint8(): "UInt8",
    pa.uint16(): "UInt16",
    pa.uint32(): "UInt32",
    pa.uint64(): "UInt64",
    pa.float32(): "Float32",
    pa.float64(): "Float64",
    pa.string(): "String",
    pa.date32(): "Date",
}


@dataclass(frozen=True)
class Column:
    """One column as ``system.columns`` lists it."""

    name: str
    type: str
    materialized: str = ""
    """The expression the table computes the column from, for a column a load does not write."""
    default: str = ""
    """The expression of a ``DEFAULT`` column, which a load leaves to the table."""


_REMOVED = Column(REMOVED, "Bool", default="false")


def version_expression(contract: DatasetContract) -> str:
    """How the table ranks a key's rows: release, then deletion, then an enriched value."""
    if contract.enriched is None:
        return "(toUInt64(source_release) * 2) + is_deleted"
    return (
        "((toUInt64(source_release) * 4) + (toUInt64(is_deleted) * 2)) + "
        f"toUInt64(NOT {contract.enriched}{NULL_FLAG_SUFFIX})"
    )


def table_columns(contract: DatasetContract) -> tuple[Column, ...]:
    """The columns of ``contract.table`` in order, null flags and ``version`` included.

    A snapshot contract adds ``removed`` last, where ``ALTER TABLE ... ADD COLUMN`` puts it.
    """
    renderer = _Renderer(contract)
    columns: list[Column] = []
    for field in fields_of(contract.schema):
        columns.append(Column(field.name, renderer.render(field.type, field.nullable, field.name)))
        if field.nullable and _is_composite(field.type):
            columns.append(Column(field.name + NULL_FLAG_SUFFIX, "Bool"))
    columns.append(Column(VERSION, "UInt64", materialized=version_expression(contract)))
    if contract.updates == "snapshot":
        columns.append(_REMOVED)
    names = [column.name for column in columns]
    if contract.enriched is not None and contract.enriched + NULL_FLAG_SUFFIX not in names:
        raise ValueError(f"{contract.name}: enriched {contract.enriched!r} has no null flag")
    if duplicates := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"{contract.name}: columns {duplicates} appear twice")
    if unused := sorted(contract.null_as_empty - renderer.collapsed):
        raise ValueError(f"{contract.name}: null_as_empty paths {unused} hold no nullable value")
    return tuple(columns)


def _is_composite(type_: pa.DataType) -> bool:
    return isinstance(type_, pa.ListType | pa.MapType | pa.StructType)


def _nullable(name: str, nullable: bool) -> str:
    return f"Nullable({name})" if nullable else name


class _Renderer:
    """Renders types for one contract and records which ``null_as_empty`` paths it used."""

    def __init__(self, contract: DatasetContract) -> None:
        self._contract = contract
        self.collapsed: set[str] = set()

    def render(self, type_: pa.DataType, nullable: bool, path: str) -> str:
        """The ClickHouse type for ``type_``. ``nullable`` only matters for scalars."""
        # isinstance leaves a generic type's parameters unknown, so each branch casts.
        if isinstance(type_, pa.MapType):
            map_type = cast("pa.MapType[pa.DataType, pa.DataType]", type_)
            key = self._inner(map_type.key_field, f"{path}.{map_type.key_field.name}")
            value = self._inner(map_type.item_field, f"{path}.{map_type.item_field.name}")
            return f"Map({key}, {value})"
        if isinstance(type_, pa.ListType):
            list_type = cast("pa.ListType[pa.DataType]", type_)
            item = list_type.value_field
            return f"Array({self._inner(item, f'{path}.{item.name}')})"
        if isinstance(type_, pa.StructType):
            items = ", ".join(
                f"{f.name} {self._inner(f, f'{path}.{f.name}')}" for f in fields_of(type_)
            )
            return f"Tuple({items})"
        if isinstance(type_, pa.DictionaryType):
            dictionary = cast("pa.DictionaryType[Any, Any]", type_)
            if dictionary.value_type == pa.string():
                return f"LowCardinality({_nullable('String', nullable)})"
        if isinstance(type_, pa.JsonType):
            return _nullable("String", nullable)
        if type_ in _SCALARS:
            return _nullable(_SCALARS[type_], nullable)
        raise TypeError(f"{self._contract.name}: no ClickHouse type for {path} ({type_})")

    def _inner(self, field: ArrowField, path: str) -> str:
        if field.nullable and _is_composite(field.type):
            if path not in self._contract.null_as_empty:
                raise ValueError(
                    f"{self._contract.name}: {path} is a nullable {field.type} inside another "
                    "value, which ClickHouse cannot store; list it in null_as_empty if an "
                    "empty value can stand for null"
                )
            self.collapsed.add(path)
        return self.render(field.type, field.nullable, path)
