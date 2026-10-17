"""Fixed sync policy and durable per-dataset partial outcomes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from openknowledge_contracts.datasets import DatasetName, ReleaseId
from openknowledge_contracts.manifests import BucketName, DatasetManifest, ObjectPrefix
from openknowledge_contracts.tables.s2 import CONTRACTS
from openknowledge_core.ingestion.archive_progress import HashedRef
from openknowledge_core.ingestion.commands import DownloadDataset
from openknowledge_core.ingestion.errors import IngestionError

CORE_ORDER = (
    "papers",
    "paper-ids",
    "abstracts",
    "tldrs",
    "authors",
    "publication-venues",
    "citations",
)
if set(CORE_ORDER) != set(CONTRACTS):
    raise RuntimeError("coordinator core order disagrees with contracts")
LARGE_ORDER = ("s2orc_v2", "embeddings-specter_v2")


class ArchiveBaseline(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    base: DownloadDataset
    completion: HashedRef


class SyncCommand(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    bucket: BucketName
    prefix: ObjectPrefix = "semantic-scholar"
    core_baselines: dict[DatasetName, HashedRef] = Field(default_factory=lambda: {})
    archives: dict[DatasetName, ArchiveBaseline] = Field(default_factory=lambda: {})
    # Listing a baseline alone cannot enable a large archive.
    enabled_archives: frozenset[Literal["s2orc_v2", "embeddings-specter_v2"]] = frozenset()


class SyncTransition(BaseModel):
    model_config = ConfigDict(frozen=True)
    start: ReleaseId
    release: ReleaseId
    objects: tuple[HashedRef, ...] = ()


class DatasetSyncResult(BaseModel):
    dataset: DatasetName
    start: ReleaseId | None = None
    target: ReleaseId | None = None
    final: ReleaseId | None = None
    archived: list[SyncTransition] = Field(default_factory=lambda: [])
    loaded: list[SyncTransition] = Field(default_factory=lambda: [])
    outcome: Literal["current", "synced", "needs_base", "failed", "disabled"] = "failed"
    reason: str | None = None


class SyncResult(BaseModel):
    datasets: list[DatasetSyncResult] = Field(default_factory=lambda: [])


class SyncIncomplete(IngestionError):
    def __init__(self, result: SyncResult) -> None:
        self.result = result
        super().__init__(
            "S2 sync incomplete: "
            + "; ".join(
                f"{item.dataset}: {item.reason or item.outcome}"
                for item in result.datasets
                if item.outcome in ("failed", "needs_base")
            )
        )


def guard_diff(base: DatasetManifest, dataset: str, diff_bytes: int) -> None:
    baseline_bytes = sum(entry.size for entry in base.entries)
    if base.dataset != dataset or baseline_bytes <= 0:
        raise IngestionError("size guard requires a verified, nonzero baseline for this dataset")
    if diff_bytes * 10 > baseline_bytes * 3:
        raise IngestionError(
            f"archived diff {diff_bytes} bytes exceeds 30% of configured base "
            f"{baseline_bytes} bytes; load refused"
        )
