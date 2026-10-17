from datetime import timedelta

import pytest

from openknowledge_contracts.manifests import ObjectPath
from openknowledge_core.ingestion import DownloadDatasetService
from openknowledge_core.ingestion.archive_progress import (
    ArchiveCompletion,
    ArchiveProgress,
    completion_path,
)
from openknowledge_core.ingestion.errors import ArchiveConflict, ManifestIncomplete, ObjectChanged
from openknowledge_core.ingestion.models import StoredObject
from tests.fakes.ports import COMMAND, DIFF, NEXT, PAPERS, PRODUCER, FakeSource, FakeStore, clock


async def test_adoption_interruption_and_empty_diff(
    service: DownloadDatasetService,
    source: FakeSource,
    store: FakeStore,
) -> None:
    progress = ArchiveProgress(store, service, PRODUCER, clock)
    for name, data in PAPERS.items():
        source.add("papers", name, data)
    pinned = await service.pin_dataset(COMMAND)
    with pytest.raises(ManifestIncomplete):
        await progress.finalize(COMMAND)
    assert await store.stat(completion_path(COMMAND)) is None
    for shard in pinned.shards:
        await service.archive_shard(pinned, shard)
    await service.write_manifest(pinned)  # completed base from an earlier manual operation
    opened = list(source.opened)
    anchor = await progress.finalize(COMMAND)
    retry = ArchiveProgress(store, service, PRODUCER, lambda: clock() + timedelta(days=1))
    assert await retry.finalize(COMMAND) == anchor
    assert source.opened == opened
    source.add_diff("papers", COMMAND.release, NEXT, {}, {})
    await service.pin_diff(DIFF)
    tip = await progress.finalize(DIFF, anchor)
    assert (await progress.read(tip, ArchiveCompletion)).manifests == ()
    assert await retry.head(COMMAND, anchor) == (NEXT, tip)
    predecessor, completed = await retry.manual_diff(COMMAND, anchor, DIFF)
    assert predecessor == anchor and completed is not None
    assert await retry.finalize(DIFF, anchor) == tip


@pytest.mark.parametrize(
    "invalid",
    [
        "disconnected",
        "fork",
        "predecessor",
        "generation",
        "dataset",
        "hash",
        "manifest_parts",
        "inventory_path",
    ],
)
async def test_invalid_progress_is_refused(
    service: DownloadDatasetService,
    source: FakeSource,
    store: FakeStore,
    invalid: str,
) -> None:
    progress = ArchiveProgress(store, service, PRODUCER, clock)
    source.add("papers", "part.gz", PAPERS["part-0.gz"])
    pinned = await service.pin_dataset(COMMAND)
    await service.archive_shard(pinned, pinned.shards[0])
    anchor = await progress.finalize(COMMAND)
    source.add_diff("papers", COMMAND.release, NEXT, {}, {})
    await service.pin_diff(DIFF)
    tip = await progress.finalize(DIFF, anchor)
    record = await progress.read(tip, ArchiveCompletion)
    if invalid in ("disconnected", "fork"):
        old = "2026-10-20" if invalid == "disconnected" else COMMAND.release
        other = DIFF.model_copy(update={"from_release": old, "release": "2026-10-27"})
        source.diffs["papers"] = []
        source.add_diff("papers", old, other.release, {}, {})
        # An interruption/foreign writer left an otherwise valid record outside the chain.
        other_record = record.model_copy(update={"from_release": old, "release": other.release})
        inventory = await progress.reference((await service.pin_diff(other)).updates.inventory_path)
        store.put(
            completion_path(other),
            other_record.model_copy(update={"inventory": inventory}).to_json(),
        )
    elif invalid == "generation":
        store.put(record.inventory.ref.path, store.data(record.inventory.ref.path))
    elif invalid in ("hash", "manifest_parts", "inventory_path"):
        root = await progress.read(anchor, ArchiveCompletion)
        changes = {
            "hash": {"inventory": record.inventory.model_copy(update={"sha256": "0" * 64})},
            "manifest_parts": {"manifests": root.manifests},
            "inventory_path": {"inventory": root.inventory},
        }
        store.put(tip.ref.path, record.model_copy(update=changes[invalid]).to_json())
    else:
        changed = {"predecessor": tip} if invalid == "predecessor" else {"dataset": "authors"}
        store.put(tip.ref.path, record.model_copy(update=changed).to_json())
    with pytest.raises((ArchiveConflict, ObjectChanged)):
        await progress.head(COMMAND, anchor)


async def test_guard_boundary_uses_verified_base_bytes(
    service: DownloadDatasetService,
    source: FakeSource,
    store: FakeStore,
) -> None:
    from openknowledge_contracts.manifests import DatasetManifest
    from openknowledge_core.ingestion.coordinator import guard_diff
    from openknowledge_core.ingestion.errors import IngestionError

    progress = ArchiveProgress(store, service, PRODUCER, clock)
    source.add("papers", "part.gz", PAPERS["part-0.gz"])
    pinned = await service.pin_dataset(COMMAND)
    await service.archive_shard(pinned, pinned.shards[0])
    reference = await progress.reference(await service.write_manifest(pinned))
    manifest = await progress.read(reference, DatasetManifest)
    # Scale to an exact boundary, independent of gzip encoder output lengths.
    manifest = manifest.model_copy(
        update={"entries": (manifest.entries[0].model_copy(update={"size": 100}),)}
    )
    guard_diff(manifest, "papers", 30)
    with pytest.raises(IngestionError, match="diff 31 bytes exceeds 30%.*base 100 bytes"):
        guard_diff(manifest, "papers", 31)


async def test_replacement_base_supersedes_old_chain_without_shard_scans(
    service: DownloadDatasetService,
    source: FakeSource,
    store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = ArchiveProgress(store, service, PRODUCER, clock)
    source.add("papers", "part.gz", PAPERS["part-0.gz"])
    pinned = await service.pin_dataset(COMMAND)
    await service.archive_shard(pinned, pinned.shards[0])
    old_anchor = await progress.finalize(COMMAND)
    source.add_diff("papers", COMMAND.release, NEXT, {}, {})
    await service.pin_diff(DIFF)
    old_tip = await progress.finalize(DIFF, old_anchor)
    replacement = COMMAND.model_copy(update={"release": NEXT})
    pinned = await service.pin_dataset(replacement)
    await service.archive_shard(pinned, pinned.shards[0])
    anchor = await progress.finalize(replacement)
    # A selected replacement must not depend on artifacts in its superseded chain.
    old_record = await progress.read(old_tip, ArchiveCompletion)
    store.remove(old_record.inventory.ref.path)
    old_base = await progress.read(old_anchor, ArchiveCompletion)
    store.put(old_base.manifests[0].ref.path, b"obsolete manifest changed")

    async def forbidden_manifest(pinned: object) -> None:
        pytest.fail("progress lookup must not rebuild manifests")

    original_stat = store.stat

    async def small_only(path: ObjectPath) -> StoredObject | None:
        assert not path.name.endswith((".gz", ".receipt.json"))
        return await original_stat(path)

    monkeypatch.setattr(store, "stat", small_only)
    monkeypatch.setattr(service, "write_manifest", forbidden_manifest)
    store.reads.clear()
    assert await progress.head(replacement, anchor) == (NEXT, anchor)
    assert not any("receipt.json" in path or path.endswith(".gz") for path in store.reads)
    assert await progress.read(old_tip, ArchiveCompletion)  # old immutable history is retained

    # A transition starting at the selected release must reference the selected base,
    # not the superseded chain's completion at the same release.
    following = DIFF.model_copy(update={"from_release": NEXT, "release": "2026-10-20"})
    source.add_diff("papers", NEXT, following.release, {}, {})
    await service.pin_diff(following)
    await progress.finalize(following, old_tip)
    with pytest.raises(ArchiveConflict, match="predecessor"):
        await progress.head(replacement, anchor)

    store.remove(completion_path(following))
    crossing = DIFF.model_copy(update={"release": "2026-10-20"})
    source.diffs["papers"] = []
    source.add_diff("papers", crossing.from_release, crossing.release, {}, {})
    await service.pin_diff(crossing)
    await progress.finalize(crossing, old_anchor)
    with pytest.raises(ArchiveConflict, match="crosses"):
        await progress.head(replacement, anchor)
