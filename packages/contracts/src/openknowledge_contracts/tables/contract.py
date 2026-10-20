"""A dataset contract: the source schema, its key and the table that holds the current rows.

The source schema mirrors one record as the provider delivers it, field for field. Nothing is
renamed, cast or dropped, so a record that does not fit is news, not noise. Arrow lets unknown
keys and nulls in required fields through, so a load refuses both itself, at every depth.

A load writes the source fields and three provenance columns:

- ``source_release``: the release the row came from.
- ``source_object``: the archived object it was read from.
- ``is_deleted``: 1 for a tombstone, which carries the key and defaults elsewhere.

The table computes ``version`` from ``source_release`` and ``is_deleted`` and keeps the row with
the highest version per key: the latest release wins, and within a release a delete outranks an
upsert, because a diff applies its updates before its deletes. A version depends only on its own
row, so replayed batches converge in any order, as long as a key appears at most once per release
and operation. A contract with an ``enriched`` field ranks a row holding it above one without it,
after release and deletion.

A snapshot contract's table also has a ``removed`` column, which a load never sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pyarrow as pa

from openknowledge_contracts.datasets import DatasetName
from openknowledge_contracts.tables.fields import fields_of, required

PROVENANCE = pa.schema(
    [
        required("source_release", pa.date32()),
        required("source_object", pa.dictionary(pa.int32(), pa.string())),
        required("is_deleted", pa.uint8()),
    ]
)
VERSION = "version"
REMOVED = "removed"

type UpdateMode = Literal["diffs", "snapshot"]


@dataclass(frozen=True)
class DatasetContract:
    """One source dataset and the table that holds its current rows.

    Attributes:
        name: The dataset's name at its source.
        table: The table holding one current row per key.
        key: Fields that identify a record.
        source: One record as delivered.
        null_as_empty: Paths below the top level, such as ``intents.item``, where a store
            that cannot hold null keeps an empty value instead. A load must count empty values
            at these paths in the raw data and fail on any; otherwise the stored empty no
            longer tells which one the source sent.
        updates: How the dataset follows its releases. ``diffs`` applies each release's
            diff. ``snapshot`` loads every release whole and flags the keys a release no
            longer lists in a ``removed`` column that loads never set.
        enriched: A nullable field that some records of a key may lack within one release
            while others hold it. A key may then repeat with that difference alone, as long
            as every record holding it agrees, and the table keeps a record holding it.
    """

    name: DatasetName
    table: str
    key: tuple[str, ...]
    source: pa.Schema
    null_as_empty: frozenset[str] = frozenset()
    updates: UpdateMode = "diffs"
    enriched: str | None = None

    def __post_init__(self) -> None:
        """Refuse a missing or nullable key, and field names that repeat or are taken."""
        if not self.key:
            raise ValueError(f"{self.name}: a contract needs a key")
        fields = {field.name: field for field in fields_of(self.source)}
        if len(fields) < len(self.source.names):
            raise ValueError(f"{self.name}: source field names repeat")
        for name in self.key:
            field = fields.get(name)
            if field is None or field.nullable:
                raise ValueError(f"{self.name}: key {name!r} must be a required source field")
        if clash := sorted(fields.keys() & {*PROVENANCE.names, VERSION, REMOVED}):
            raise ValueError(f"{self.name}: source fields {clash} shadow table columns")
        if self.enriched is not None and not (
            self.enriched in self.source.names and fields[self.enriched].nullable
        ):
            raise ValueError(f"{self.name}: enriched {self.enriched!r} must be a nullable field")

    @property
    def schema(self) -> pa.Schema:
        """What a load writes: the source fields, then provenance."""
        return pa.schema([*fields_of(self.source), *fields_of(PROVENANCE)])
