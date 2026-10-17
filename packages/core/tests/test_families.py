import pytest

from openknowledge_contracts.tables.s2 import CONTRACTS
from openknowledge_core.ingestion.families import (
    ARCHIVE_DATASETS,
    ArchiveFamily,
    archive_family,
)


def test_archive_families_cover_exactly_the_supported_datasets() -> None:
    assert ARCHIVE_DATASETS == {
        **dict.fromkeys(CONTRACTS, ArchiveFamily.CORE),
        "s2orc_v2": ArchiveFamily.FULLTEXT,
        "embeddings-specter_v2": ArchiveFamily.EMBEDDINGS,
    }


@pytest.mark.parametrize("dataset", ["s2orc", "embeddings-specter_v1", "unknown"])
def test_legacy_and_unknown_datasets_are_not_managed(dataset: str) -> None:
    with pytest.raises(ValueError, match="Unsupported archive dataset"):
        archive_family(dataset)
