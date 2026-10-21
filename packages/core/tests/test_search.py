import pytest

from openknowledge_contracts.search import PAPERS
from openknowledge_core.search import (
    AbstractRow,
    BulkRefused,
    CorpusSlice,
    IndexPapers,
    SearchIndexService,
    SearchRejected,
    SourceRows,
)
from tests.fakes.ports import (
    NEXT,
    RELEASE,
    FakeSearchSink,
    FakeSearchSource,
    abstract_row,
    paper_row,
)

V1, V2 = 41464, 41478  # the versions of a live row at RELEASE and at NEXT
INDEX = "papers-v1"


def service(
    source: FakeSearchSource, sink: FakeSearchSink, bulk_bytes: int = 5 * 1024 * 1024
) -> SearchIndexService:
    return SearchIndexService(source=source, sink=sink, shards=2, bulk_bytes=bulk_bytes)


def tombstone(corpusid: int, version: int) -> SourceRows:
    paper = paper_row(corpusid, version, is_deleted=True, source_release=NEXT)
    return SourceRows(corpusid=corpusid, paper=paper)


async def test_a_base_pass_writes_every_document_with_the_larger_version() -> None:
    source, sink = FakeSearchSource(), FakeSearchSink()
    paper = paper_row(
        101,
        V1,
        title="A Synthetic Paper",
        externalids={"DOI": "10.1000/EXAMPLE.1", "CorpusId": "101", "ACL": None},
        s2fieldsofstudy=["Computer Science", "Computer Science"],
        journal_name="Example Journal",
    )
    source.base_rows = [
        SourceRows(corpusid=101, paper=paper, abstract=abstract_row(101, V2)),
        SourceRows(corpusid=102, paper=paper_row(102, V1)),
    ]
    result = await service(source, sink, bulk_bytes=1).index(IndexPapers(index=INDEX, release=NEXT))
    assert (result.created, result.written, result.deleted, result.documents) == (True, 2, 0, 2)
    assert result.requests == 2 and sink.requests == [(INDEX, 1), (INDEX, 1)]
    mapping, settings = sink.created[INDEX]
    assert set(mapping["properties"]) == PAPERS.names
    assert settings["index"]["number_of_shards"] == 2
    version, source_101 = sink.indices[INDEX]["101"]
    assert version == V2 and source_101 is not None
    assert set(source_101) <= PAPERS.names and source_101["contract_version"] == 1
    assert source_101["abstract"] == "A synthetic abstract."
    assert source_101["externalids"] == {"DOI": "10.1000/EXAMPLE.1"}
    assert source_101["s2fieldsofstudy"] == [{"category": "Computer Science"}]
    assert source_101["journal"] == {"name": "Example Journal"}
    version, source_102 = sink.indices[INDEX]["102"]
    assert version == V1 and source_102 is not None and "abstract" not in source_102
    assert source.asked == [(None, NEXT, CorpusSlice(part=1, parts=1))]


async def test_a_diff_pass_deletes_a_papers_tombstone_and_drops_a_tombstoned_abstract() -> None:
    source, sink = FakeSearchSource(), FakeSearchSink()
    sink.indices[INDEX] = {"102": (V1, {"corpusid": "102", "abstract": "old"}), "103": (V1, {})}
    gone = AbstractRow(corpusid=102, version=V2 + 1, source_release=NEXT, is_deleted=True)
    newer = tombstone(105, V2 + 1).model_copy(update={"abstract": abstract_row(105, V2 + 2)})
    source.diff_rows = [
        SourceRows(corpusid=102, paper=paper_row(102, V1), abstract=gone),
        tombstone(103, V2 + 1),
        SourceRows(corpusid=104, abstract=abstract_row(104, V2)),
        newer,
    ]
    command = IndexPapers(index=INDEX, release=NEXT, from_release=RELEASE, part=2, parts=3)
    result = await service(source, sink).index(command)
    assert (result.created, result.written, result.deleted, result.documents) == (False, 1, 2, 1)
    assert sink.indices[INDEX]["103"] == (V2 + 1, None)
    assert sink.indices[INDEX]["105"] == (V2 + 2, None)
    version, source_102 = sink.indices[INDEX]["102"]
    assert version == V2 + 1 and source_102 is not None and "abstract" not in source_102
    assert "104" not in sink.indices[INDEX]
    assert source.asked == [(RELEASE, NEXT, CorpusSlice(part=2, parts=3))]
    # Over an index that does not exist, a diff pass refuses before it asks for anything.
    missing = FakeSearchSink()
    with pytest.raises(SearchRejected, match=f"{INDEX} does not exist; a base pass must create"):
        await service(source, missing).index(command)
    assert (missing.indices, missing.requests, missing.refreshed) == ({}, [], [])
    assert len(source.asked) == 1


async def test_a_refused_bulk_item_fails_the_pass() -> None:
    source, sink = FakeSearchSource(), FakeSearchSink()
    source.base_rows = [SourceRows(corpusid=101, paper=paper_row(101, V1))]
    sink.refused = "101"
    with pytest.raises(BulkRefused, match="corpusid 101: version_conflict_engine_exception"):
        await service(source, sink).index(IndexPapers(index=INDEX, release=RELEASE))
    assert sink.refreshed == []


async def test_a_slice_with_no_rows_writes_nothing() -> None:
    source, sink = FakeSearchSource(), FakeSearchSink()
    command = IndexPapers(index=INDEX, release=RELEASE, part=3, parts=4)
    result = await service(source, sink).index(command)
    assert (result.written, result.deleted, result.requests, result.documents) == (0, 0, 0, 0)
    assert (sink.requests, sink.refreshed, result.created) == ([], [INDEX], True)
    # The parts tile the keys once, and a part with no key of its own has no range at all.
    top = 2**64 - 1
    assert [CorpusSlice(part=p, parts=3).bounds(101, top) for p in (1, 2, 3)] == [
        (101, 101 + (top - 100) // 3 - 1),
        (101 + (top - 100) // 3, 101 + 2 * (top - 100) // 3 - 1),
        (101 + 2 * (top - 100) // 3, top),
    ]
    assert [CorpusSlice(part=p, parts=3).bounds(0, 0) for p in (1, 2, 3)] == [
        (0, -1),
        (0, -1),
        (0, 0),
    ]
