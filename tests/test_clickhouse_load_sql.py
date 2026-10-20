"""The load SQL against every contract, run with ``clickhouse local`` on gzip files."""

from __future__ import annotations

import copy
import gzip
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pytest

from openknowledge_adapters.clickhouse.load_sql import (
    delete_query,
    duplicates_parameters,
    duplicates_query,
    problems_query,
    query_parameters,
    removed_query,
    upsert_query,
)
from openknowledge_contracts.tables import PROVENANCE, DatasetContract
from openknowledge_contracts.tables.clickhouse import NULL_FLAG_SUFFIX, table_columns
from openknowledge_contracts.tables.fields import ArrowField, fields_of
from openknowledge_contracts.tables.s2 import (
    AUTHORS,
    CITATIONS,
    CONTRACTS,
    PAPERS,
    PUBLICATION_VENUES,
)
from tests.clickhouse_local import (
    MIGRATIONS,
    clickhouse_local,
    needs_clickhouse,
    run,
    section,
    up,
)

FIXTURES = Path(__file__).resolve().parents[1] / "packages" / "contracts" / "tests" / "fixtures"
RELEASE = "2026-10-06"
OBJECT = "gs://raw/s2/2026-10-06/dataset/part-0.gz"
TEXT = 'café "quoted" \\ tab\t'

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Location = tuple[str | int, ...]

contracts = pytest.mark.parametrize("contract", CONTRACTS.values(), ids=list(CONTRACTS))

_BITS: dict[pa.DataType, int] = {
    pa.int8(): 8,
    pa.int16(): 16,
    pa.int32(): 32,
    pa.int64(): 64,
    pa.uint8(): 8,
    pa.uint16(): 16,
    pa.uint32(): 32,
    pa.uint64(): 64,
}


def _range(type_: pa.DataType) -> tuple[int, int]:
    bits = _BITS[type_]
    if pa.types.is_signed_integer(type_):
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def _kind(type_: pa.DataType) -> str:
    kinds = (("map", pa.MapType), ("list", pa.ListType), ("struct", pa.StructType))
    return next((kind for kind, class_ in kinds if isinstance(type_, class_)), "scalar")


def _item(type_: pa.DataType) -> ArrowField:
    """A list's item or a map's value."""
    if isinstance(type_, pa.MapType):
        return cast("pa.MapType[pa.DataType, pa.DataType]", type_).item_field
    return cast("pa.ListType[pa.DataType]", type_).value_field


def _members(type_: pa.DataType) -> list[ArrowField]:
    return fields_of(cast("pa.StructType", type_))


def _scalar(type_: pa.DataType) -> pa.DataType:
    if isinstance(type_, pa.DictionaryType):
        return cast("pa.DictionaryType[Any, Any]", type_).value_type
    return type_


def full(type_: pa.DataType) -> Json:
    """A value with every optional part present, integers at their highest."""
    kind = _kind(type_)
    if kind == "map":
        return {"DOI": full(_item(type_).type), "MAG": full(_item(type_).type)}
    if kind == "list":
        return [full(_item(type_).type), full(_item(type_).type)]
    if kind == "struct":
        return {f.name: full(f.type) for f in _members(type_)}
    if isinstance(type_, pa.JsonType):
        return {"DBLP": ["A. Doe", TEXT], "ORCID": None, "n": -1.5}
    type_ = _scalar(type_)
    if type_ == pa.string():
        return TEXT
    if type_ == pa.bool_():
        return True
    return _range(type_)[1]


def nulled(field: ArrowField, *, outer: bool) -> Json:
    """Null at the outermost nullable value, or at the innermost ones."""
    kind = _kind(field.type)
    if field.nullable and (outer or kind == "scalar"):
        return None
    if kind == "map":
        return {"DOI": nulled(_item(field.type), outer=outer)}
    if kind == "list":
        return [nulled(_item(field.type), outer=outer)] * 2
    if kind == "struct":
        return {f.name: nulled(f, outer=outer) for f in _members(field.type)}
    return full(field.type)


def absent(fields: list[ArrowField]) -> dict[str, Json]:
    """Every optional key left out, at every depth."""

    def value(type_: pa.DataType) -> Json:
        kind = _kind(type_)
        if kind == "map":
            return {"DOI": value(_item(type_).type)}
        if kind == "list":
            return [value(_item(type_).type)]
        if kind == "struct":
            return absent(_members(type_))
        return full(type_)

    return {f.name: value(f.type) for f in fields if not f.nullable}


def accepted(contract: DatasetContract) -> list[dict[str, Json]]:
    source = fields_of(contract.source)
    records: list[dict[str, Json]] = [
        {f.name: full(f.type) for f in source},
        {f.name: nulled(f, outer=True) for f in source},
        {f.name: nulled(f, outer=False) for f in source},
        absent(source),
    ]
    for index, record in enumerate(records):
        for name in contract.key:
            record[name] = f"key-{index}" if isinstance(record[name], str) else index
    return records


def stored(field: ArrowField, value: Json) -> Json:
    """A null composite is stored empty, and a required scalar in it as its default."""
    kind = _kind(field.type)
    if kind == "map":
        values = cast("dict[str, Json]", value or {})
        return {k: stored(_item(field.type), v) for k, v in values.items()}
    if kind == "list":
        return [stored(_item(field.type), v) for v in cast("list[Json]", value or [])]
    if kind == "struct":
        members = cast("dict[str, Json]", value or {})
        return {f.name: stored(f, members.get(f.name)) for f in _members(field.type)}
    if value is None and not field.nullable:
        scalar = _scalar(field.type)
        return "" if scalar == pa.string() else False if scalar == pa.bool_() else 0
    return value


def rows(contract: DatasetContract, records: list[dict[str, Json]]) -> list[dict[str, Json]]:
    expected: list[dict[str, Json]] = []
    for record in records:
        row: dict[str, Json] = {}
        for field in fields_of(contract.source):
            row[field.name] = stored(field, record.get(field.name))
            if field.nullable and _kind(field.type) != "scalar":
                row[field.name + NULL_FLAG_SUFFIX] = record.get(field.name) is None
        provenance = {"source_release": RELEASE, "source_object": OBJECT, "is_deleted": 0}
        expected.append({**row, **provenance})
    return sorted(expected, key=lambda r: [r[k] for k in contract.key])


@dataclass(frozen=True)
class Refused:
    name: str
    line: str
    problem: str


class _Missing:
    """A key that is deleted rather than given a value."""


MISSING = _Missing()


def wrong_types(type_: pa.DataType) -> Iterator[tuple[str, Json]]:
    kind = _kind(type_)
    if kind in ("map", "struct"):
        yield "array for object", ["x"]
    elif kind == "list":
        yield "object for array", {"x": 1}
    elif isinstance(type_, pa.JsonType):
        return
    elif _scalar(type_) == pa.string():
        yield "number for string", 1
        yield "object for string", {"x": "y"}
    elif _scalar(type_) == pa.bool_():
        yield "number for bool", 1
        yield "string for bool", "true"
    else:
        low, high = _range(_scalar(type_))
        yield "string for integer", "1"
        yield "float for integer", 1.5
        yield "bool for integer", True
        yield "above the range", high + 1
        yield "below the range", low - 1


def breaks(
    contract: DatasetContract, field: ArrowField, path: str, at: Location
) -> Iterator[tuple[str, Location, Json | _Missing]]:
    """Every break of the value at ``at``, and of every value inside it."""
    kind = _kind(field.type)
    if not field.nullable:
        yield f"{path} null", at, None
    for name, value in wrong_types(field.type):
        yield f"{path} {name}", at, value
    if path in contract.null_as_empty:
        yield f"{path} empty", at, [] if kind == "list" else {}
    if kind in ("map", "list"):
        item = _item(field.type)
        step = "DOI" if kind == "map" else 1
        yield from breaks(contract, item, f"{path}.{item.name}", (*at, step))
    elif kind == "struct":
        yield f"{path} unknown key", (*at, "surprise"), 1
        for member in _members(field.type):
            if not member.nullable:
                yield f"{path}.{member.name} missing", (*at, member.name), MISSING
            yield from breaks(contract, member, f"{path}.{member.name}", (*at, member.name))


def changed(record: dict[str, Json], at: Location, value: Json | _Missing) -> str:
    copied = copy.deepcopy(record)
    target: Any = copied
    for step in at[:-1]:
        target = target[step]
    if isinstance(value, _Missing):
        del target[at[-1]]
    else:
        target[at[-1]] = value
    return json.dumps(copied)


def refused(contract: DatasetContract) -> list[Refused]:
    table = contract.table
    record = accepted(contract)[0]
    line = json.dumps(record)
    first = next(iter(record))
    not_an_object = f"{table}: a line is not a JSON object"
    cases = [
        Refused("blank line", "", not_an_object),
        Refused("array", "[]", not_an_object),
        Refused("trailing text", line + " x", not_an_object),
        Refused(
            "repeated key",
            f'{{"{first}": {json.dumps(record[first])}, {line[1:]}',
            f"{table}: a record repeats a key",
        ),
        Refused(
            "unknown key",
            changed(record, ("surprise",), 1),
            f"{table}: a record has a key the contract does not declare",
        ),
    ]
    for field in fields_of(contract.source):
        if not field.nullable:
            missing = changed(record, (field.name,), MISSING)
            lacks = f"{table}: a record lacks a required field"
            cases.append(Refused(f"{field.name} missing", missing, lacks))
        problem = f"{table}: {field.name} breaks the contract"
        for name, at, value in breaks(contract, field, field.name, (field.name,)):
            cases.append(Refused(name, changed(record, at, value), problem))
    return cases


def gz(path: Path, lines: list[str]) -> Path:
    path.write_bytes(gzip.compress("".join(line + "\n" for line in lines).encode(), mtime=0))
    return path


def params(source: Path) -> dict[str, str]:
    return query_parameters(source_file=str(source), release=RELEASE, source_object=OBJECT)


def load(contract: DatasetContract, workdir: Path, lines: list[str]) -> list[dict[str, Any]]:
    workdir.mkdir()
    source = gz(workdir / "part-0.gz", lines)
    columns = ", ".join(
        c.name for c in table_columns(contract) if not (c.materialized or c.default)
    )
    sql = f"""
        {up()}
        {upsert_query(contract, "file")};
        SELECT {columns} FROM {contract.table}
        ORDER BY {", ".join(contract.key)} FORMAT JSONEachRow;
    """
    loaded = clickhouse_local(sql, workdir / "db", **params(source))
    for field in fields_of(contract.source):
        if isinstance(field.type, pa.JsonType):
            for row in loaded:
                row[field.name] = row[field.name] and json.loads(row[field.name])
    return loaded


@needs_clickhouse
@contracts
def test_accepted_records_land_as_their_lines_read(
    contract: DatasetContract, tmp_path: Path
) -> None:
    fixture = (FIXTURES / "s2" / f"{contract.name}.jsonl").read_text().splitlines()
    generated = [json.dumps(record) for record in accepted(contract)]
    for name, lines in (("fixture", fixture), ("generated", generated)):
        expected = rows(contract, [json.loads(line) for line in lines])
        assert load(contract, tmp_path / name, lines) == expected


@needs_clickhouse
@contracts
def test_every_generated_break_is_found_with_its_field(
    contract: DatasetContract, tmp_path: Path
) -> None:
    cases = refused(contract)
    lines = [case.line for case in cases]
    assert len(set(lines)) == len(lines), "two cases share a line, so one would hide the other"
    source = gz(tmp_path / "part-0.gz", [json.dumps(accepted(contract)[0]), *lines])
    found = clickhouse_local(
        problems_query(contract, "file") + "\nFORMAT JSONEachRow", tmp_path, **params(source)
    )
    problems = {row["line"]: row["problem"] for row in found}
    assert {case.name: problems.get(case.line) for case in cases} == {
        case.name: case.problem for case in cases
    }
    assert len(found) == len(cases), "an accepted line was listed as a problem"


@needs_clickhouse
@contracts
def test_an_upsert_stops_at_a_refused_record(contract: DatasetContract, tmp_path: Path) -> None:
    case = next(c for c in refused(contract) if c.name.endswith(" null"))
    source = gz(tmp_path / "part-0.gz", [json.dumps(accepted(contract)[0]), case.line])
    sql = f"{up()}\n{upsert_query(contract, 'file')};"
    result = run(sql, tmp_path, **params(source))
    assert f"Code: 395. DB::Exception: {case.problem}: " in result.stderr


@needs_clickhouse
@pytest.mark.parametrize("missing", [False, True], ids=["null", "missing"])
@pytest.mark.parametrize(
    "contract, field, migration_name, error",
    [
        (AUTHORS, "papercount", "00011_make_authors_papercount_nullable.sql", "Code: 349."),
        (
            PAPERS,
            "url",
            "00013_make_papers_metadata_nullable.sql",
            "No such column authors_is_null",
        ),
    ],
    ids=["authors", "papers"],
)
def test_unknown_value_requires_the_nullable_migration(
    contract: DatasetContract,
    field: str,
    migration_name: str,
    error: str,
    missing: bool,
    tmp_path: Path,
) -> None:
    record = accepted(contract)[0]
    record[field] = None
    if missing:
        del record[field]
    source = gz(tmp_path / "part-0.gz", [json.dumps(record)])
    migration = MIGRATIONS / migration_name
    clickhouse_local(f"{up()}\n{section(migration, 'Down')}", tmp_path)
    insert = upsert_query(contract, "file")
    rejected = run(insert, tmp_path, **params(source))
    assert rejected.returncode != 0
    assert error in rejected.stderr
    assert clickhouse_local(
        f"{section(migration, 'Up')}\n{insert};\n"
        f"SELECT {field} FROM {contract.table} FINAL FORMAT JSONEachRow;",
        tmp_path,
        **params(source),
    ) == [{field: None}]


@needs_clickhouse
@contracts
def test_a_delete_hides_its_upsert_and_leaves_every_other_column_at_its_default(
    contract: DatasetContract, tmp_path: Path
) -> None:
    record = accepted(contract)[0]
    key = {name: record[name] for name in contract.key}
    provenance = {"source_release": RELEASE, "source_object": OBJECT, "is_deleted": 1}
    upserts = gz(tmp_path / "upserts.gz", [json.dumps(record)])
    deletes = gz(tmp_path / "deletes.gz", [json.dumps(key)])
    columns = ", ".join([*contract.key, *PROVENANCE.names])
    table = contract.table
    db = tmp_path / "db"
    upsert = f"{up()}\n{upsert_query(contract, 'file')};"
    clickhouse_local(upsert, db, **params(upserts))
    tombstone, bare, live = clickhouse_local(
        f"""
        {delete_query(contract, "file")};
        CREATE TABLE bare AS {table};
        INSERT INTO bare ({columns}) SELECT {columns} FROM {table} WHERE is_deleted = 1;
        SELECT * FROM {table} WHERE is_deleted = 1 FORMAT JSONEachRow;
        SELECT * FROM bare FORMAT JSONEachRow;
        SELECT count() AS live FROM {table} FINAL FORMAT JSONEachRow;
        """,
        db,
        **params(deletes),
    )
    assert {name: tombstone[name] for name in (*key, *provenance)} == {**key, **provenance}
    assert tombstone == bare
    assert live == {"live": 0}


@needs_clickhouse
@contracts
def test_a_delete_record_holds_the_key_and_nothing_else(
    contract: DatasetContract, tmp_path: Path
) -> None:
    record = accepted(contract)[0]
    key = {name: record[name] for name in contract.key}
    types = {field.name: field.type for field in fields_of(contract.source)}
    table = contract.table
    extra = next(name for name in record if name not in key)
    cases = [
        Refused(
            f"{extra} beside the key",
            json.dumps({**key, extra: record[extra]}),
            f"{table}: a record has a key the contract does not declare",
        )
    ]
    for name in key:
        without = {other: value for other, value in key.items() if other != name}
        _, wrong = next(wrong_types(types[name]))
        cases += [
            Refused(
                f"{name} missing", json.dumps(without), f"{table}: a record lacks a required field"
            ),
            Refused(
                f"{name} of a wrong type",
                json.dumps({**key, name: wrong}),
                f"{table}: {name} breaks the contract",
            ),
        ]
    sql = f"{up()}\n{delete_query(contract, 'file')};"
    for case in cases:
        source = gz(tmp_path / "deletes.gz", [case.line])
        result = run(sql, tmp_path, **params(source))
        assert f"Code: 395. DB::Exception: {case.problem}: " in result.stderr, case.name


@needs_clickhouse
@contracts
def test_a_repeated_key_is_identical_or_conflicting(
    contract: DatasetContract, tmp_path: Path
) -> None:
    """Key order and spacing do not tell repeats apart, and nulls do; a changed value does."""
    first, nulls, inner_nulls, last = accepted(contract)
    changed = next(
        f.name
        for f in fields_of(contract.source)
        if f.name not in contract.key
        and _kind(f.type) == "scalar"
        and not isinstance(f.type, pa.JsonType)
    )
    value = first[changed]
    other = (
        not value
        if isinstance(value, bool)
        else value - 1
        if isinstance(value, int)
        else f"{value}!"
    )
    swapped = [
        json.dumps(dict(reversed(record.items())), separators=(",", ":"))
        for record in (nulls, inner_nulls)
    ]
    files = [
        gz(tmp_path / "part-0.gz", [json.dumps(r) for r in (first, nulls, inner_nulls)]),
        gz(
            tmp_path / "part-1.gz",
            [*swapped, json.dumps({**first, changed: other}), json.dumps(last)],
        ),
    ]
    found = clickhouse_local(
        duplicates_query(contract, len(files), "file") + "\nFORMAT JSONEachRow",
        tmp_path,
        **duplicates_parameters([str(file) for file in files]),
    )
    key = first[contract.key[0]]
    expected = {"identical_rows": 2, "conflicting_keys": 1, "examples": [f"({key!r})"]}
    if contract.enriched is not None:
        expected["enriched_rows"] = 0
    assert found == [expected]


@needs_clickhouse
def test_citation_duplicate_comparison_preserves_nulls_arrays_and_identity(tmp_path: Path) -> None:
    """Only contexts may enrich a citation; other stored values and nulls remain exact."""
    baseline: dict[str, Json] = {
        "citationid": 1577791790,
        "citingcorpusid": 139809936,
        "citedcorpusid": None,
        "isinfluential": False,
        "contexts": None,
        "intents": [["background"]],
    }
    # The first two enrich contexts. Every other change must still fail.
    changes: list[dict[str, Json]] = [
        {"contexts": ["A representative citation context."]},
        {"contexts": []},
        {"citedcorpusid": 0},
        {"citingcorpusid": 139809937},
        {"isinfluential": True},
        {"intents": None},
        {"intents": []},
        {"intents": [None]},
        {"intents": [["methodology"]]},
    ]
    first = [dict(baseline, citationid=1577791790 + i) for i in range(len(changes))]
    second = [{**record, **change} for record, change in zip(first, changes, strict=True)]
    # Replays of either variant are identical repeats; enrichment counts one lost variant.
    first.extend([first[0], first[0]])
    second.extend([second[0], second[0]])
    # Missing nullable fields and explicit null store identically; JSON order is immaterial.
    identical = dict(baseline, citationid=6011687425, intents=[None])
    omitted = {name: value for name, value in identical.items() if value is not None}
    first.append(identical)
    second.append(dict(reversed(omitted.items())))
    # Distinct citation IDs may share endpoints and must remain distinct.
    second.append(dict(baseline, citationid=6011687426))
    files = [
        gz(tmp_path / f"part-{i}.gz", [json.dumps(record) for record in records])
        for i, records in enumerate((first, second))
    ]
    found = clickhouse_local(
        duplicates_query(CITATIONS, len(files), "file") + "\nFORMAT JSONEachRow",
        tmp_path,
        **duplicates_parameters([str(file) for file in files]),
    )[0]
    assert found["identical_rows"] == 5
    assert found["enriched_rows"] == 2
    assert found["conflicting_keys"] == len(changes) - 2
    assert set(found["examples"]) == {
        f"({record['citationid']})" for record in first[2 : len(changes)]
    }


@needs_clickhouse
@pytest.mark.parametrize("other", [[], ["b", "a"], ["a"]])
def test_citation_enrichment_refuses_different_nonnull_contexts(
    tmp_path: Path, other: list[str]
) -> None:
    row = {**accepted(CITATIONS)[0], "contexts": ["a", "b"]}
    source = gz(
        tmp_path / "conflicts.gz",
        [json.dumps({**row, "contexts": contexts}) for contexts in (None, ["a", "b"], other)],
    )
    found = clickhouse_local(
        duplicates_query(CITATIONS, 1, "file") + " FORMAT JSONEachRow",
        tmp_path,
        **duplicates_parameters([str(source)]),
    )[0]
    assert (found["conflicting_keys"], found["enriched_rows"], found["identical_rows"]) == (1, 0, 0)


@needs_clickhouse
@pytest.mark.parametrize("reverse", [False, True])
def test_citation_context_versions_survive_merges_replays_releases_and_deletes(
    tmp_path: Path, reverse: bool
) -> None:
    row: dict[str, Json] = {**accepted(CITATIONS)[0], "contexts": ["a context"], "intents": [None]}
    db = tmp_path / "db"
    clickhouse_local(up(), db)

    def insert(record: dict[str, Json], release: str, *, delete: bool = False) -> None:
        source = gz(tmp_path / "rows.gz", [json.dumps(record)])
        sql = delete_query(CITATIONS, "file") if delete else upsert_query(CITATIONS, "file")
        clickhouse_local(
            sql,
            db,
            **query_parameters(source_file=str(source), release=release, source_object=OBJECT),
        )

    null = {**row, "contexts": None}
    records = [row, null] if reverse else [null, row]
    for record in records * 2:
        insert(record, RELEASE)
    clickhouse_local("OPTIMIZE TABLE citations FINAL", db)
    select = "SELECT contexts, contexts_is_null FROM citations FINAL FORMAT JSONEachRow"
    assert clickhouse_local(select, db) == [{"contexts": ["a context"], "contexts_is_null": False}]
    # A later release can deliberately clear the field. Older enrichment cannot resurrect it.
    insert(null, "2026-10-13")
    insert(row, RELEASE)
    assert clickhouse_local(select, db) == [{"contexts": [], "contexts_is_null": True}]
    key = {"citationid": row["citationid"]}
    insert(key, "2026-10-13", delete=True)
    insert(row, "2026-10-13")
    assert clickhouse_local(select, db) == []
    insert(row, "2026-10-20")
    assert clickhouse_local(select, db) == [{"contexts": ["a context"], "contexts_is_null": False}]


@needs_clickhouse
def test_citation_version_migration_requires_rebuild_and_refuses_populated_rollback(
    tmp_path: Path,
) -> None:
    migration = MIGRATIONS / "00014_order_citation_context_enrichment.sql"
    upgrade, downgrade = section(migration, "Up"), section(migration, "Down")
    legacy = section(MIGRATIONS / "00007_create_citations.sql", "Up")
    db = tmp_path / "db"
    clickhouse_local("CREATE DATABASE rebuild ENGINE = Atomic", db)

    def query(sql: str) -> list[dict[str, Any]]:
        return clickhouse_local("USE rebuild;" + sql, db)

    query(
        legacy
        + section(MIGRATIONS / "00008_create_loads.sql", "Up")
        + "INSERT INTO citations (citationid, source_release) VALUES (1, '2026-10-13');",
    )
    assert "swap the populated legacy table" in run("USE rebuild;" + upgrade, db).stderr
    # A rebuild preserves the old table; the new UUID invalidates its shard checkpoints.
    old_uuid = query(
        "SELECT uuid FROM system.tables WHERE database = currentDatabase() "
        "AND name = 'citations' FORMAT JSONEachRow"
    )[0]["uuid"]
    query(
        "INSERT INTO loads (target, target_uuid) "
        f"VALUES ('citations', '{old_uuid}'); TRUNCATE TABLE citations;"
    )
    assert "swap the populated legacy table" in run("USE rebuild;" + upgrade, db).stderr
    query("INSERT INTO citations (citationid, source_release) VALUES (1, '2026-10-13');")
    query("RENAME TABLE citations TO citations_preserved;" + legacy + upgrade)
    new_uuid = query(
        "SELECT uuid FROM system.tables WHERE database = currentDatabase() "
        "AND name = 'citations' FORMAT JSONEachRow"
    )[0]["uuid"]
    assert new_uuid != old_uuid
    query(
        "INSERT INTO citations (citationid, source_release) VALUES (1, '2026-10-13');" + upgrade,
    )
    assert "rollback requires an empty replacement" in run("USE rebuild;" + downgrade, db).stderr
    assert query("SELECT count() AS n FROM citations_preserved FORMAT JSONEachRow") == [{"n": 1}]


@contracts
def test_an_upsert_writes_every_column_the_table_does_not_fill(contract: DatasetContract) -> None:
    match = re.match(r"INSERT INTO `\w+` \(([^)]*)\)", upsert_query(contract))
    assert match is not None
    written = [name.strip("`") for name in match.group(1).split(", ")]
    assert written == [c.name for c in table_columns(contract) if not (c.materialized or c.default)]


@needs_clickhouse
def test_a_snapshot_flags_the_keys_a_newer_release_no_longer_lists(tmp_path: Path) -> None:
    venue = {**accepted(PUBLICATION_VENUES)[0], "issn": None}
    db = tmp_path / "db"
    first, second, third, fourth = "2026-09-29", "2026-10-06", "2026-10-13", "2026-10-20"
    upsert = upsert_query(PUBLICATION_VENUES, "file")
    flag = removed_query(PUBLICATION_VENUES)

    def at(release: str, ids: list[str], sql: str) -> None:
        source = gz(tmp_path / f"{release}.gz", [json.dumps({**venue, "id": id_}) for id_ in ids])
        arguments = query_parameters(source_file=str(source), release=release, source_object=OBJECT)
        clickhouse_local(sql, db, **arguments)

    def table() -> tuple[int, dict[str, tuple[str, bool]]]:
        """The rows written with ``removed`` set, and the live row of each key."""
        flagged, *live = clickhouse_local(
            "SELECT count() AS n FROM publication_venues WHERE removed FORMAT JSONEachRow;"
            "SELECT id, toString(source_release) AS release, removed FROM publication_venues FINAL"
            " FORMAT JSONEachRow;",
            db,
        )
        return flagged["n"], {row["id"]: (row["release"], row["removed"]) for row in live}

    clickhouse_local(up(), db)
    at(first, ["a", "b"], upsert)
    at(second, ["a"], upsert)
    at(second, [], flag)
    flagged = (1, {"a": (second, False), "b": (second, True)})
    assert table() == flagged
    (kept,) = clickhouse_local(
        "SELECT * EXCEPT (source_release, removed) FROM publication_venues FINAL WHERE id = 'b'"
        " FORMAT JSONEachRow",
        db,
    )
    (original,) = rows(PUBLICATION_VENUES, [{**venue, "id": "b"}])
    assert kept == {name: value for name, value in original.items() if name != "source_release"}
    at(second, [], flag)
    assert table() == flagged
    at(third, ["a"], upsert)
    at(third, [], flag)
    assert table() == (1, {"a": (third, False), "b": (second, True)})
    at(fourth, ["b"], upsert)
    assert table()[1]["b"] == (fourth, False)
