import pytest

from rmbr.ann import AnnIndex
from rmbr.embed import CachingEmbedder, FakeEmbedder
from rmbr.search import hybrid_search
from rmbr.store import Store


@pytest.fixture
def rig(tmp_path):
    """A minimal store + embedder + ann index wired together, no Index/Memory facade."""
    store = Store(tmp_path / "test.db")
    embedder = CachingEmbedder(FakeEmbedder(dimension=16), store)
    ann_index = AnnIndex(dim=16)

    def add_chunk(namespace: str, text: str, metadata: dict | None = None) -> int:
        doc_id = store.insert_document(namespace)
        [chunk_id] = store.insert_chunks(doc_id, namespace, [text], [metadata or {}])
        vector = embedder.embed_one(text)
        ann_index.add([chunk_id], [vector])
        return chunk_id

    yield store, embedder, ann_index, add_chunk
    store.close()


def search(store, embedder, ann_index, query, namespaces, **kwargs):
    return hybrid_search(
        query=query,
        namespaces=namespaces,
        k=kwargs.pop("k", 5),
        fts_search=store.search_chunks_fts,
        fetch_records=store.get_chunks,
        ann_index=ann_index,
        embedder=embedder,
        **kwargs,
    )


def test_bm25_only_finds_keyword_match(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "the deployment guide covers docker setup")
    add_chunk("researcher", "a recipe for baking sourdough bread")

    hits = search(store, embedder, ann_index, "docker deployment", ["researcher"])
    assert len(hits) >= 1
    assert "docker" in hits[0].text


def test_hits_have_timings(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "some searchable content here")

    hits = search(store, embedder, ann_index, "searchable", ["researcher"])
    assert "bm25_ms" in hits.timings
    assert "fusion_ms" in hits.timings
    assert "total_ms" in hits.timings
    assert "embed_ms" in hits.timings  # vector stage ran, no budget set
    assert hits.degraded is False


def test_namespace_filtering_excludes_other_namespaces(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "shared topic about rockets")
    add_chunk("coder", "shared topic about rockets")

    hits = search(store, embedder, ann_index, "rockets", ["researcher"])
    assert all(h.namespace == "researcher" for h in hits)
    assert len(hits) == 1


def test_budget_ms_degrades_to_bm25_only(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "some searchable content here")

    hits = search(store, embedder, ann_index, "searchable", ["researcher"], budget_ms=0)
    assert hits.degraded is True
    assert "embed_ms" not in hits.timings
    assert "ann_ms" not in hits.timings
    assert "bm25_ms" in hits.timings  # bm25 always runs; it's the cheap stage


def test_use_vectors_false_skips_vector_stage(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "some searchable content here")

    hits = search(store, embedder, ann_index, "searchable", ["researcher"], use_vectors=False)
    assert "embed_ms" not in hits.timings
    assert hits.degraded is False  # not degraded — vectors were never requested


def test_use_bm25_false_gives_pure_vector_search(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "some searchable content here")

    hits = search(store, embedder, ann_index, "searchable", ["researcher"], use_bm25=False)
    assert "bm25_ms" not in hits.timings
    assert "embed_ms" in hits.timings
    assert "ann_ms" in hits.timings
    assert len(hits) == 1


def test_use_bm25_and_use_vectors_both_false_returns_no_hits(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "some searchable content here")

    hits = search(store, embedder, ann_index, "searchable", ["researcher"], use_bm25=False, use_vectors=False)
    assert len(hits) == 0


def test_rrf_ranks_dual_matches_above_single_signal_matches(rig):
    store, embedder, ann_index, add_chunk = rig
    # This chunk should match both BM25 (keyword) and, being semantically
    # close to itself, the vector query too — it's the query text verbatim.
    dual_match_id = add_chunk("researcher", "quarterly revenue report")
    # This one only shares no keywords and is a different fake-embedding vector,
    # so it should rank behind the dual match.
    add_chunk("researcher", "unrelated content about gardening")

    hits = search(store, embedder, ann_index, "quarterly revenue report", ["researcher"])
    assert hits[0].id == dual_match_id


def test_deleted_chunk_id_in_stale_ann_index_is_dropped(rig):
    store, embedder, ann_index, add_chunk = rig
    chunk_id = add_chunk("researcher", "temporary content")
    # Simulate the document being deleted without the ANN index being
    # updated yet (e.g. a crash between the two writes) — search must not
    # blow up on a dangling id.
    store.conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    store.conn.commit()

    hits = search(store, embedder, ann_index, "temporary content", ["researcher"])
    assert all(h.id != chunk_id for h in hits)


def test_where_filters_to_matching_metadata(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "deployment guide for docker", metadata={"source": "docs/deploy.md"})
    add_chunk("researcher", "deployment guide for kubernetes", metadata={"source": "docs/k8s.md"})

    hits = search(
        store, embedder, ann_index, "deployment guide", ["researcher"], where={"source": "docs/deploy.md"}
    )
    assert len(hits) == 1
    assert hits[0].metadata["source"] == "docs/deploy.md"


def test_where_with_no_matches_returns_empty(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "deployment guide", metadata={"source": "docs/deploy.md"})

    hits = search(
        store, embedder, ann_index, "deployment guide", ["researcher"], where={"source": "nonexistent.md"}
    )
    assert len(hits) == 0


def test_where_requires_all_keys_to_match(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "release notes", metadata={"source": "docs/a.md", "category": "release"})
    add_chunk("researcher", "release notes", metadata={"source": "docs/a.md", "category": "internal"})

    hits = search(
        store,
        embedder,
        ann_index,
        "release notes",
        ["researcher"],
        where={"source": "docs/a.md", "category": "release"},
    )
    assert len(hits) == 1
    assert hits[0].metadata["category"] == "release"


def test_cached_result_not_reused_across_different_where_filters(rig):
    store, embedder, ann_index, add_chunk = rig
    add_chunk("researcher", "quarterly report content", metadata={"source": "a.md"})

    store_module_kwargs = dict(query_cache=store)
    unfiltered = hybrid_search(
        query="quarterly report",
        namespaces=["researcher"],
        k=5,
        fts_search=store.search_chunks_fts,
        fetch_records=store.get_chunks,
        ann_index=ann_index,
        embedder=embedder,
        **store_module_kwargs,
    )
    assert unfiltered.from_cache is False
    assert len(unfiltered) == 1

    # Same query, but now filtered to a source that doesn't match — must
    # not incorrectly return the cached unfiltered result.
    filtered = hybrid_search(
        query="quarterly report",
        namespaces=["researcher"],
        k=5,
        fts_search=store.search_chunks_fts,
        fetch_records=store.get_chunks,
        ann_index=ann_index,
        embedder=embedder,
        where={"source": "nonexistent.md"},
        **store_module_kwargs,
    )
    assert filtered.from_cache is False
    assert len(filtered) == 0
