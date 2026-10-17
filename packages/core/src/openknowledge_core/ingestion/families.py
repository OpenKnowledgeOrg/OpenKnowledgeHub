"""Archive routing shared by controllers and Batch workers."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from openknowledge_contracts.tables.s2 import CONTRACTS


class ArchiveFamily(StrEnum):
    CORE = "core"
    FULLTEXT = "fulltext"
    EMBEDDINGS = "embeddings"


ARCHIVE_DATASETS: Mapping[str, ArchiveFamily] = MappingProxyType(
    {
        **dict.fromkeys(CONTRACTS, ArchiveFamily.CORE),
        "s2orc_v2": ArchiveFamily.FULLTEXT,
        "embeddings-specter_v2": ArchiveFamily.EMBEDDINGS,
    }
)


def archive_family(dataset: str) -> ArchiveFamily:
    try:
        return ARCHIVE_DATASETS[dataset]
    except KeyError:
        raise ValueError(f"Unsupported archive dataset: {dataset}") from None


def require_family(dataset: str, family: ArchiveFamily) -> None:
    actual = archive_family(dataset)
    if actual != family:
        raise ValueError(f"Dataset {dataset} belongs to {actual}, not {family}")
