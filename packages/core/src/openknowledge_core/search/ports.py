"""What the search use cases need from outside: the current rows, and an index to write."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any, Protocol

from openknowledge_contracts.datasets import ReleaseId
from openknowledge_core.search.models import BulkOperation, CorpusSlice, IndexName, SourceRows


class SearchSource(Protocol):
    async def base(
        self, release: ReleaseId, corpus_slice: CorpusSlice
    ) -> AsyncGenerator[SourceRows]:
        """The live rows of both tables at ``release``, one item per corpus ID in the slice.

        The tables keep one row per key, so ``release`` must be the one they hold: an older
        state cannot be read back once a later release replaced a row.
        """
        ...

    async def diff(
        self, from_release: ReleaseId, release: ReleaseId, corpus_slice: CorpusSlice
    ) -> AsyncGenerator[SourceRows]:
        """The latest row per key in both tables, tombstones included, for the changed keys.

        A key counts as changed when either table has a row for it at a release after
        ``from_release`` and up to ``release``.
        """
        ...


class SearchSink(Protocol):
    async def exists(self, index: IndexName) -> bool: ...

    async def create_index(
        self, index: IndexName, mapping: Mapping[str, Any], settings: Mapping[str, Any]
    ) -> None: ...

    async def bulk(self, index: IndexName, operations: Sequence[BulkOperation]) -> None:
        """Apply every operation, or raise BulkRefused naming a refused corpus ID and why."""
        ...

    async def refresh(self, index: IndexName) -> None: ...

    async def count(self, index: IndexName) -> int: ...
