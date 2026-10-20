import pyarrow as pa
import pytest

from openknowledge_contracts.tables import DatasetContract
from openknowledge_contracts.tables.clickhouse import Column, table_columns
from openknowledge_contracts.tables.fields import (
    ArrowField,
    list_of,
    optional,
    required,
    string_map,
    struct_of,
)

PROVENANCE_COLUMNS = [
    Column("source_release", "Date"),
    Column("source_object", "LowCardinality(String)"),
    Column("is_deleted", "UInt8"),
    Column("version", "UInt64", materialized="(toUInt64(source_release) * 2) + is_deleted"),
]


def columns(*fields: ArrowField, null_as_empty: frozenset[str] = frozenset()) -> list[Column]:
    source = pa.schema([required("id", pa.uint64()), *fields])
    contract = DatasetContract("example", "example", ("id",), source, null_as_empty)
    rendered = list(table_columns(contract))
    assert rendered[-len(PROVENANCE_COLUMNS) :] == PROVENANCE_COLUMNS
    return rendered[1 : -len(PROVENANCE_COLUMNS)]


def test_nullable_scalars_become_nullable_columns() -> None:
    assert columns(required("n", pa.uint32()), optional("year", pa.int16())) == [
        Column("n", "UInt32"),
        Column("year", "Nullable(Int16)"),
    ]


def test_a_nullable_composite_keeps_its_null_in_a_flag_column() -> None:
    assert columns(optional("tags", list_of(pa.string()))) == [
        Column("tags", "Array(String)"),
        Column("tags_is_null", "Bool"),
    ]


def test_a_required_composite_has_no_flag() -> None:
    assert columns(required("tags", list_of(pa.string()))) == [Column("tags", "Array(String)")]


def test_structs_and_maps_render_with_their_inner_nullability() -> None:
    journal = struct_of(required("name", pa.string()), optional("pages", pa.string()))
    assert columns(
        required("journal", journal), required("ids", string_map(nullable_values=True))
    ) == [
        Column("journal", "Tuple(name String, pages Nullable(String))"),
        Column("ids", "Map(String, Nullable(String))"),
    ]


def test_json_text_stays_a_string() -> None:
    assert columns(optional("raw", pa.json_())) == [Column("raw", "Nullable(String)")]


def test_a_nested_nullable_composite_needs_a_declared_collapse() -> None:
    intents = optional("intents", list_of(list_of(pa.string()), nullable_items=True))
    with pytest.raises(ValueError, match="intents.item"):
        columns(intents)
    assert columns(intents, null_as_empty=frozenset({"intents.item"})) == [
        Column("intents", "Array(Array(String))"),
        Column("intents_is_null", "Bool"),
    ]


def test_an_unused_collapse_is_refused() -> None:
    with pytest.raises(ValueError, match="tags.item"):
        columns(required("tags", list_of(pa.string())), null_as_empty=frozenset({"tags.item"}))


def test_a_flag_may_not_collide_with_a_source_field() -> None:
    with pytest.raises(ValueError, match="tags_is_null"):
        columns(optional("tags", list_of(pa.string())), required("tags_is_null", pa.bool_()))


def test_types_without_a_lossless_mapping_are_refused() -> None:
    with pytest.raises(TypeError, match="stamp"):
        columns(required("stamp", pa.timestamp("ms")))


@pytest.mark.parametrize(
    ("field", "error"),
    [
        (required("text", list_of(pa.string())), "nullable field"),
        (optional("text", pa.string()), "no null flag"),
    ],
)
def test_an_enriched_field_must_be_a_nullable_composite(field: ArrowField, error: str) -> None:
    source = pa.schema([required("id", pa.uint64()), field])
    with pytest.raises(ValueError, match=error):
        table_columns(DatasetContract("example", "example", ("id",), source, enriched="text"))
