import pytest

from openknowledge_contracts.datasets import ReleaseMetadata
from openknowledge_core.ingestion.discovery import ReleaseDiscovery, contiguous
from openknowledge_core.ingestion.errors import CredentialRejected, SourceRejected
from openknowledge_core.ingestion.models import DiffChain, DiffListing
from tests.fakes.ports import NEXT, RELEASE, FakeSource


async def test_selection_has_no_short_lookback_and_propagates_access_errors() -> None:
    class Lagging(FakeSource):
        async def resolve_release(self, selector: str) -> ReleaseMetadata:
            if selector == NEXT:
                raise CredentialRejected("access refused")
            return await super().resolve_release(selector)

    source = Lagging()
    discovery = ReleaseDiscovery(source, (RELEASE, NEXT))
    with pytest.raises(CredentialRejected):
        await discovery.target("s2orc_v2", RELEASE)
    source = FakeSource()
    source.add("papers", "part.gz", b"")
    discovery = ReleaseDiscovery(source, (RELEASE, NEXT))
    assert await discovery.target("papers", RELEASE) == NEXT
    assert await discovery.target("authors", RELEASE) == RELEASE
    assert source.resolved == [NEXT]  # metadata is cached across datasets


def test_contiguous_path_refuses_gap_and_missing_target() -> None:
    chain = DiffChain(
        dataset="papers",
        diffs=(
            DiffListing(
                from_release=RELEASE,
                release_id=NEXT,
                updates=(),
                deletes=(),
            ),
        ),
    )
    assert contiguous(chain, "papers", RELEASE, NEXT) == ((RELEASE, NEXT),)
    with pytest.raises(SourceRejected, match="unavailable"):
        contiguous(chain, "papers", RELEASE, "2026-10-20")
    with pytest.raises(SourceRejected, match="gap"):
        contiguous(chain, "papers", "2026-09-29", NEXT)
