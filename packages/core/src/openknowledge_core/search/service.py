"""Write one corpus ID slice of the paper search documents, as a base pass or a diff pass.

A document's version is the larger ``version`` of the rows it was built from, a delete carries
the same, and every write uses OpenSearch's ``external_gte`` versioning. So a slice can be run
again at the same or a newer release: an equal version is accepted again, and a pass over an
older release is refused rather than applied, as long as OpenSearch still remembers a deleted
document's version, which it drops after ``index.gc_deletes``.
"""

from __future__ import annotations

from contextlib import aclosing

from openknowledge_contracts.search import PAPERS, SearchContract, index_body
from openknowledge_core.search.commands import IndexPapers, IndexPapersResult
from openknowledge_core.search.errors import SearchRejected
from openknowledge_core.search.models import (
    BulkOperation,
    DeleteDocument,
    ExternalIds,
    FieldOfStudy,
    IndexDocument,
    IndexName,
    Journal,
    SearchDocument,
    SourceRows,
)
from openknowledge_core.search.ports import SearchSink, SearchSource

BULK_BYTES = 5 * 1024 * 1024


def document_operation(rows: SourceRows, contract_version: int) -> BulkOperation | None:
    """The write one corpus ID needs, or None for an abstract whose paper the table never had.

    A papers tombstone becomes a delete at the larger version of the two rows, which is the
    tombstone's unless an abstracts row is newer, so the delete is never below the document it
    removes. An abstracts tombstone leaves the document without an abstract, at that version.
    """
    paper = rows.paper
    if paper is None:
        return None
    abstract = rows.abstract
    version = paper.version if abstract is None else max(paper.version, abstract.version)
    if paper.is_deleted:
        return DeleteDocument(id=str(paper.corpusid), version=version)
    known = ExternalIds.model_fields
    document = SearchDocument(
        corpusid=str(paper.corpusid),
        title=paper.title,
        abstract=None if abstract is None or abstract.is_deleted else abstract.abstract,
        authors=paper.authors,
        venue=paper.venue,
        publicationvenueid=paper.publicationvenueid,
        year=paper.year,
        publicationdate=paper.publicationdate,
        journal=None if paper.journal_name is None else Journal(name=paper.journal_name),
        externalids=ExternalIds.model_validate(
            {kind: value for kind, value in paper.externalids.items() if kind in known}
        ),
        s2fieldsofstudy=tuple(
            FieldOfStudy(category=category) for category in dict.fromkeys(paper.s2fieldsofstudy)
        ),
        publicationtypes=paper.publicationtypes,
        referencecount=paper.referencecount,
        citationcount=paper.citationcount,
        influentialcitationcount=paper.influentialcitationcount,
        isopenaccess=paper.isopenaccess,
        url=paper.url,
        contract_version=contract_version,
    )
    return IndexDocument(id=document.corpusid, version=version, document=document)


class SearchIndexService:
    def __init__(
        self,
        *,
        source: SearchSource,
        sink: SearchSink,
        shards: int,
        contract: SearchContract = PAPERS,
        bulk_bytes: int = BULK_BYTES,
    ) -> None:
        self._source = source
        self._sink = sink
        self._shards = shards
        self._contract = contract
        self._bulk_bytes = bulk_bytes

    async def index(self, command: IndexPapers) -> IndexPapersResult:
        """One pass over one slice, in bulk requests of about ``bulk_bytes`` each.

        A base pass asks the source for its rows before it creates a missing index, so a table
        that cannot be reached leaves OpenSearch untouched. A diff pass only follows a base pass,
        so it refuses a missing index before it asks for anything.
        """
        if command.from_release is None:
            rows = await self._source.base(command.release, command.corpus_slice)
            created = await self._ensure(command.index)
        else:
            if not await self._sink.exists(command.index):
                raise SearchRejected(
                    f"{command.index} does not exist; a base pass must create it before a diff "
                    "pass writes into it"
                )
            rows = await self._source.diff(
                command.from_release, command.release, command.corpus_slice
            )
            created = False
        written = deleted = requests = 0
        batch: list[BulkOperation] = []
        size = 0
        async with aclosing(rows):
            async for found in rows:
                operation = document_operation(found, self._contract.version)
                if operation is None:
                    continue
                batch.append(operation)
                size += len(operation.model_dump_json())
                if size >= self._bulk_bytes:
                    sent = await self._flush(command.index, batch)
                    written, deleted, requests = written + sent[0], deleted + sent[1], requests + 1
                    batch, size = [], 0
        if batch:
            sent = await self._flush(command.index, batch)
            written, deleted, requests = written + sent[0], deleted + sent[1], requests + 1
        await self._sink.refresh(command.index)
        return IndexPapersResult(
            index=command.index,
            release=command.release,
            from_release=command.from_release,
            part=command.part,
            parts=command.parts,
            created=created,
            written=written,
            deleted=deleted,
            requests=requests,
            documents=await self._sink.count(command.index),
        )

    async def _ensure(self, index: IndexName) -> bool:
        """Create the index with the contract's mapping unless it exists; True when created.

        Only a base pass calls this: it fills a new index whole, where a diff pass would leave
        a partial one.
        """
        if await self._sink.exists(index):
            return False
        body = index_body(self._contract, shards=self._shards)
        await self._sink.create_index(index, body.mappings, body.settings)
        return True

    async def _flush(self, index: IndexName, batch: list[BulkOperation]) -> tuple[int, int]:
        """Send one bulk request and return how many documents it wrote and deleted."""
        await self._sink.bulk(index, batch)
        deleted = sum(1 for operation in batch if isinstance(operation, DeleteDocument))
        return len(batch) - deleted, deleted
