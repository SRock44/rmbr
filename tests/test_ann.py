import numpy as np
import pytest

from rmbr.ann import AnnIndex


def unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_add_and_search_returns_nearest_first():
    idx = AnnIndex(dim=4)
    idx.add(
        [1, 2, 3],
        [unit([1, 0, 0, 0]), unit([0, 1, 0, 0]), unit([0.9, 0.1, 0, 0])],
    )
    results = idx.search(unit([1, 0, 0, 0]), k=3)
    ids = [id_ for id_, _score in results]
    assert ids[0] == 1  # exact match ranks first
    assert set(ids) == {1, 2, 3}


def test_search_respects_k():
    idx = AnnIndex(dim=4)
    idx.add([1, 2, 3, 4], [unit(np.random.rand(4)) for _ in range(4)])
    results = idx.search(unit([1, 0, 0, 0]), k=2)
    assert len(results) == 2


def test_search_empty_index_returns_nothing():
    idx = AnnIndex(dim=4)
    assert idx.search(unit([1, 0, 0, 0]), k=5) == []


def test_remove_excludes_from_future_searches():
    idx = AnnIndex(dim=4)
    idx.add([1, 2], [unit([1, 0, 0, 0]), unit([0, 1, 0, 0])])
    idx.remove([1])
    results = idx.search(unit([1, 0, 0, 0]), k=5)
    ids = [id_ for id_, _score in results]
    assert 1 not in ids
    assert len(idx) == 1


def test_add_duplicate_id_raises():
    idx = AnnIndex(dim=4)
    idx.add([1], [unit([1, 0, 0, 0])])
    with pytest.raises(Exception):
        idx.add([1], [unit([0, 1, 0, 0])])


def test_replace_overwrites_existing_id():
    idx = AnnIndex(dim=4)
    idx.add([1], [unit([1, 0, 0, 0])])
    idx.replace([1], [unit([0, 1, 0, 0])])
    assert len(idx) == 1
    results = idx.search(unit([0, 1, 0, 0]), k=1)
    assert results[0][0] == 1


def test_serialize_roundtrip_preserves_search_results():
    idx = AnnIndex(dim=4)
    idx.add([1, 2, 3], [unit([1, 0, 0, 0]), unit([0, 1, 0, 0]), unit([0, 0, 1, 0])])
    blob = idx.to_bytes()

    restored = AnnIndex.from_bytes(blob, dim=4)
    assert len(restored) == 3
    results = restored.search(unit([1, 0, 0, 0]), k=1)
    assert results[0][0] == 1


def test_reload_after_full_removal_then_add_does_not_crash():
    """Regression test for #20: usearch (>=2.9, confirmed through 2.26.0)
    leaves a tombstoned node in the HNSW graph after remove(), even once the
    index is back down to zero vectors. Serializing that state, reloading it
    in a fresh AnnIndex, and adding a new vector segfaulted the whole
    process (native access violation, not a catchable Python exception)
    before AnnIndex started compacting on serialize - so this test's real
    assertion is that the process survives to reach the asserts at all.
    """
    idx = AnnIndex(dim=4)
    idx.add([1], [unit([1, 0, 0, 0])])
    idx.remove([1])
    blob = idx.to_bytes()

    restored = AnnIndex.from_bytes(blob, dim=4)
    restored.add([2], [unit([0, 1, 0, 0])])  # segfaulted the process pre-fix
    assert len(restored) == 1
    results = restored.search(unit([0, 1, 0, 0]), k=1)
    assert results[0][0] == 2


def test_reload_after_partial_removal_then_add_preserves_surviving_vectors():
    idx = AnnIndex(dim=4)
    idx.add([1, 2, 3], [unit([1, 0, 0, 0]), unit([0, 1, 0, 0]), unit([0, 0, 1, 0])])
    idx.remove([2])
    blob = idx.to_bytes()

    restored = AnnIndex.from_bytes(blob, dim=4)
    restored.add([4], [unit([0, 0, 0, 1])])
    assert len(restored) == 3
    assert sorted(restored.ids()) == [1, 3, 4]
    results = restored.search(unit([1, 0, 0, 0]), k=1)
    assert results[0][0] == 1


def test_compact_is_skipped_when_nothing_was_removed():
    """The rebuild-on-serialize guard should be a no-op (same underlying
    _index object) when remove() was never called, so the common
    append-only case doesn't pay a full rebuild on every save."""
    idx = AnnIndex(dim=4)
    idx.add([1, 2], [unit([1, 0, 0, 0]), unit([0, 1, 0, 0])])
    original_index_obj = idx._index
    idx.to_bytes()
    assert idx._index is original_index_obj


def test_recall_at_scale_meets_floor():
    """Regression guard for a real bug: usearch's own defaults
    (expansion_search=64) measured recall@5=0.68 on a 5,000-vector
    corpus in production benching - AnnIndex's tuned defaults
    (expansion_add/expansion_search=256) brought that to 0.94+. This
    uses a smaller corpus for CI speed, with a floor loose enough to
    absorb run-to-run noise but tight enough to catch a real regression
    back toward the untuned defaults.
    """
    rng = np.random.default_rng(0)
    n, dim, k = 2000, 64, 5

    doc_vectors = rng.standard_normal((n, dim)).astype(np.float32)
    doc_vectors /= np.linalg.norm(doc_vectors, axis=1, keepdims=True)

    idx = AnnIndex(dim=dim)
    idx.add(list(range(n)), list(doc_vectors))

    query_vectors = doc_vectors[:200] + rng.standard_normal((200, dim)).astype(np.float32) * 0.1
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

    true_top_k = np.argsort(-(query_vectors @ doc_vectors.T), axis=1)[:, :k]

    hits = 0
    for query_vector, truth in zip(query_vectors, true_top_k):
        predicted = {id_ for id_, _sim in idx.search(query_vector, k)}
        hits += len(predicted & set(truth.tolist()))
    recall = hits / (len(query_vectors) * k)

    assert recall > 0.85, f"recall@{k}={recall:.3f} - below the floor, check expansion_search/expansion_add"
