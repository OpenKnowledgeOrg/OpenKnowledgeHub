"""Values the search feature passes around; none is written to storage."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    model_validator,
)

from openknowledge_contracts.datasets import ReleaseId

# OpenSearch takes lowercase names without the characters it reserves, under 256 bytes.
IndexName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]{0,254}$")]


class CorpusSlice(BaseModel):
    """Part ``part`` of the corpus IDs split into ``parts`` equal ranges, counting from 1."""

    model_config = ConfigDict(frozen=True)

    part: PositiveInt = 1
    parts: PositiveInt = 1

    @model_validator(mode="after")
    def _within(self) -> Self:
        if self.part > self.parts:
            raise ValueError("the part must not exceed the number of parts")
        return self

    def bounds(self, low: int, high: int) -> tuple[int, int]:
        """Both ends of this part of the keys ``low`` to ``high``, ends included.

        The parts tile the keys once, and the last ends at ``high``. With fewer keys than
        parts, some parts are empty, and an empty part's start comes after its end.
        """
        if low > high:
            raise ValueError(f"the keys {low} to {high} are an empty range")
        span = high - low + 1
        start = low + span * (self.part - 1) // self.parts
        return start, low + span * self.part // self.parts - 1


class Author(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    authorId: str | None = None


class PaperRow(BaseModel):
    """One row of ``papers`` as a search source reads it; a tombstone carries defaults."""

    model_config = ConfigDict(frozen=True)

    corpusid: NonNegativeInt
    version: NonNegativeInt
    source_release: ReleaseId
    is_deleted: bool
    externalids: dict[str, str | None] = Field(default_factory=dict)
    url: str | None = None
    title: str | None = None
    authors: tuple[Author, ...] = ()
    venue: str | None = None
    publicationvenueid: str | None = None
    year: int | None = None
    referencecount: NonNegativeInt = 0
    citationcount: NonNegativeInt = 0
    influentialcitationcount: NonNegativeInt = 0
    isopenaccess: bool = False
    s2fieldsofstudy: tuple[str, ...] = ()
    """The categories, in the row's order."""
    publicationtypes: tuple[str, ...] = ()
    publicationdate: str | None = None
    journal_name: str | None = None


class AbstractRow(BaseModel):
    """One row of ``abstracts`` as a search source reads it; a tombstone carries defaults."""

    model_config = ConfigDict(frozen=True)

    corpusid: NonNegativeInt
    version: NonNegativeInt
    source_release: ReleaseId
    is_deleted: bool
    abstract: str = ""


class SourceRows(BaseModel):
    """The rows of one corpus ID in both tables, each absent where the table has none."""

    model_config = ConfigDict(frozen=True)

    corpusid: NonNegativeInt
    paper: PaperRow | None = None
    abstract: AbstractRow | None = None


class Journal(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str


class FieldOfStudy(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: str


class ExternalIds(BaseModel):
    """The identifier kinds the index maps, spelled as Semantic Scholar spells them."""

    model_config = ConfigDict(frozen=True)

    DOI: str | None = None
    ArXiv: str | None = None
    PubMed: str | None = None
    PubMedCentral: str | None = None
    MAG: str | None = None
    ACL: str | None = None
    DBLP: str | None = None


class SearchDocument(BaseModel):
    """One document as the index holds it: the search contract's fields and nothing else."""

    model_config = ConfigDict(frozen=True)

    corpusid: str
    title: str | None = None
    abstract: str | None = None
    authors: tuple[Author, ...] = ()
    venue: str | None = None
    publicationvenueid: str | None = None
    year: int | None = None
    publicationdate: str | None = None
    journal: Journal | None = None
    externalids: ExternalIds = ExternalIds()
    s2fieldsofstudy: tuple[FieldOfStudy, ...] = ()
    publicationtypes: tuple[str, ...] = ()
    referencecount: NonNegativeInt = 0
    citationcount: NonNegativeInt = 0
    influentialcitationcount: NonNegativeInt = 0
    isopenaccess: bool = False
    url: str | None = None
    contract_version: PositiveInt

    def source(self) -> dict[str, Any]:
        """The document as JSON data, without the keys that hold nothing."""
        return self.model_dump(mode="json", exclude_none=True)


class IndexDocument(BaseModel):
    """Write ``document`` under ``id`` unless the index holds a newer ``version`` of it."""

    model_config = ConfigDict(frozen=True)

    operation: Literal["index"] = "index"
    id: str
    version: NonNegativeInt
    document: SearchDocument


class DeleteDocument(BaseModel):
    """Delete the document under ``id`` unless the index holds a newer ``version`` of it."""

    model_config = ConfigDict(frozen=True)

    operation: Literal["delete"] = "delete"
    id: str
    version: NonNegativeInt


type BulkOperation = IndexDocument | DeleteDocument
