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
