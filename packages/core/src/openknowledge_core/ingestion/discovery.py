"""Dataset-specific target selection from one run's frozen release catalogue."""

from openknowledge_contracts.datasets import ReleaseId, ReleaseMetadata
from openknowledge_core.ingestion.errors import SourceRejected
from openknowledge_core.ingestion.models import DiffChain
from openknowledge_core.ingestion.ports import ReleaseCatalog


class ReleaseDiscovery:
    def __init__(self, catalog: ReleaseCatalog, releases: tuple[ReleaseId, ...]) -> None:
        self._catalog = catalog
        self._releases = sorted(set(releases), reverse=True)
        self._metadata: dict[ReleaseId, ReleaseMetadata] = {}

    async def target(self, dataset: str, held: ReleaseId | None) -> ReleaseId | None:
        for release in self._releases:
            if held is not None and release <= held:
                break
            if release not in self._metadata:
                metadata = await self._catalog.resolve_release(release)
                if metadata.release_id != release:
                    raise SourceRejected("release metadata identifies another release")
                self._metadata[release] = metadata
            if any(item.name == dataset for item in self._metadata[release].datasets):
                return release
        return held


def contiguous(
    chain: DiffChain, dataset: str, start: str, target: str
) -> tuple[tuple[str, str], ...]:
    if chain.dataset != dataset:
        raise SourceRejected("diff chain identifies another dataset")
    current = start
    pairs: list[tuple[str, str]] = []
    for diff in chain.diffs:
        if diff.from_release != current or not current < diff.release_id <= target:
            raise SourceRejected("diff chain has a gap, fork or unordered transition")
        pairs.append((current, diff.release_id))
        current = diff.release_id
    if current != target:
        raise SourceRejected("required diff path is unavailable; operator action required")
    return tuple(pairs)
