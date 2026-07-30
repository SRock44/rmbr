import numpy as np
import pytest

from rmbr.ann import AnnIndex
from rmbr.embed import CachingEmbedder
from rmbr.search import hybrid_search
from rmbr.store import Store


class MappedEmbedder:
    """Returns a caller-controlled vector per exact text — lets tests decide
    which queries should look "semantically similar" without depending on
    a real model's opinion of similarity."""

    model_name = "mapped"

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = {text: np.asarray(vec, dtype=np.float32) for text, vec in mapping.items()}

    def embed(self, texts):
        return [self.mapping[t] for t in texts]


BASE = [1.0, 0.0, 0.0, 0.0]
NEAR = [0.99, 0.01, 0.0, 0.0]  # cosine sim to BASE > 0.99, well above the 0.95 threshold
FAR = [0.0, 1.0, 0.0, 0.0]  # orthogonal to BASE, cosine sim 0.0


@pytest.fixture
def rig(tmp_path):
    store = Store(tmp_path / "test.db")
    mapping = {"query one": BASE, "query near": NEAR, "query far": FAR, "seed": [0.5, 0.5, 0.5, 0.5]}
    embedder = CachingEmbedder(MappedEmbedder(mapping), store)
    ann_index = AnnIndex(dim=4)

    doc_id = store.insert_document("researcher")
    [chunk_id] = store.insert_chunks(doc_id, "researcher", ["seed content for search"])
    ann_index.add([chunk_id], [embedder.embed_one("seed")])

    yield store, embedder, ann_index
    store.close()


def run_search(store, embedder, ann_index, query, **kwargs):
    return hybrid_search(
        query=query,
        namespaces=["researcher"],
        k=kwargs.pop("k", 5),
        fts_search=store.search_chunks_fts,
        fetch_records=store.get_chunks,
        ann_index=ann_index,
        embedder=embedder,
        query_cache=store,
        **kwargs,
    )


def test_first_search_is_not_from_cache(rig):
    store, embedder, ann_index = rig
    hits = run_search(store, embedder, ann_index, "query one")
    assert hits.from_cache is False
    assert "bm25_ms" in hits.timings


def test_semantically_near_query_hits_cache(rig):
    store, embedder, ann_index = rig
    run_search(store, embedder, ann_index, "query one")  # populates the cache

    hits = run_search(store, embedder, ann_index, "query near")
    assert hits.from_cache is True
    assert "bm25_ms" not in hits.timings  # retrieval skipped entirely on a cache hit
    assert "cache_ms" in hits.timings


def test_dissimilar_query_does_not_hit_cache(rig):
    store, embedder, ann_index = rig
    run_search(store, embedder, ann_index, "query one")

    hits = run_search(store, embedder, ann_index, "query far")
    assert hits.from_cache is False


def test_expired_cache_entry_is_not_reused(rig):
    store, embedder, ann_index = rig
    run_search(store, embedder, ann_index, "query one", cache_ttl_seconds=0)

    hits = run_search(store, embedder, ann_index, "query near", cache_ttl_seconds=0)
    assert hits.from_cache is False


def test_write_invalidates_cache(rig):
    store, embedder, ann_index = rig
    run_search(store, embedder, ann_index, "query one")

    doc_id = store.insert_document("researcher")
    store.insert_chunks(doc_id, "researcher", ["freshly added content"])
    store.clear_query_cache()  # what Memory/Index call internally on every write

    hits = run_search(store, embedder, ann_index, "query near")
    assert hits.from_cache is False


def test_cache_hit_requires_sufficient_k(rig):
    store, embedder, ann_index = rig
    run_search(store, embedder, ann_index, "query one", k=2)

    # A request for more results than what's cached must not reuse the
    # shallower cached list.
    hits = run_search(store, embedder, ann_index, "query near", k=5)
    assert hits.from_cache is False


def test_cache_hit_is_sliced_to_requested_k(rig):
    store, embedder, ann_index = rig
    run_search(store, embedder, ann_index, "query one", k=5)

    hits = run_search(store, embedder, ann_index, "query near", k=1)
    assert hits.from_cache is True
    assert len(hits) <= 1
