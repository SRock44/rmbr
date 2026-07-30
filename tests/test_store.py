import pytest

from rmbr.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def test_insert_and_fetch_document_and_chunks(store):
    doc_id = store.insert_document("researcher", source="notes.md", metadata={"tag": "x"})
    chunk_ids = store.insert_chunks(doc_id, "researcher", ["first chunk", "second chunk"])
    assert len(chunk_ids) == 2

    chunks = store.get_chunks(chunk_ids)
    assert [c.text for c in chunks] == ["first chunk", "second chunk"]
    assert chunks[0].namespace == "researcher"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_delete_document_cascades_to_chunks(store):
    doc_id = store.insert_document("researcher")
    chunk_ids = store.insert_chunks(doc_id, "researcher", ["a", "b"])
    store.delete_document(doc_id)
    assert store.get_chunks(chunk_ids) == []


def test_fts_search_chunks_finds_matching_text(store):
    doc_id = store.insert_document("researcher")
    store.insert_chunks(
        doc_id, "researcher", ["the deployment guide covers docker", "cooking pasta recipes"]
    )
    hits = store.search_chunks_fts("deployment docker", ["researcher"], limit=5)
    assert len(hits) == 1
    assert hits[0].rank == 0


def test_fts_search_respects_namespace_filter(store):
    doc_id = store.insert_document("researcher")
    store.insert_chunks(doc_id, "researcher", ["shared keyword content"])
    other_doc = store.insert_document("coder")
    store.insert_chunks(other_doc, "coder", ["shared keyword content"])

    hits = store.search_chunks_fts("keyword", ["researcher"], limit=5)
    assert len(hits) == 1

    hits_both = store.search_chunks_fts("keyword", ["researcher", "coder"], limit=5)
    assert len(hits_both) == 2


def test_fts_search_handles_special_characters_without_raising(store):
    doc_id = store.insert_document("researcher")
    store.insert_chunks(doc_id, "researcher", ["deploy-to-prod checklist"])
    # Should not raise an FTS5 syntax error on hyphens/punctuation.
    hits = store.search_chunks_fts('deploy-to-prod? "quoted"', ["researcher"], limit=5)
    assert isinstance(hits, list)


def test_memory_crud_and_fts(store):
    mem_id = store.insert_memory("coder", "user prefers dark mode")
    mem = store.get_memory(mem_id)
    assert mem.text == "user prefers dark mode"
    assert mem.namespace == "coder"

    store.insert_memory("coder", "user likes tabs over spaces")
    memories = store.list_memories("coder")
    assert len(memories) == 2

    hits = store.search_memories_fts("dark mode", ["coder"], limit=5)
    assert len(hits) == 1

    store.delete_memory(mem_id)
    assert store.get_memory(mem_id) is None


def test_embed_cache_roundtrip(store):
    vector = b"\x00\x01\x02\x03"
    assert store.get_cached_embedding("hash1", "model-a") is None
    store.set_cached_embedding("hash1", "model-a", vector)
    assert store.get_cached_embedding("hash1", "model-a") == vector
    # different model = different cache entry
    assert store.get_cached_embedding("hash1", "model-b") is None


def test_ann_blob_roundtrip(store):
    assert store.get_ann_blob("chunks") is None
    store.set_ann_blob("chunks", b"fake-index-bytes", dim=384, metric="cosine")
    blob, dim, metric = store.get_ann_blob("chunks")
    assert blob == b"fake-index-bytes"
    assert dim == 384
    assert metric == "cosine"


def test_query_cache_roundtrip_and_purge(store):
    store.set_query_cache("key1", "researcher", b"vec", "[]", created_at=1000.0)
    rows = store.list_query_cache("researcher")
    assert len(rows) == 1

    store.purge_expired_query_cache(ttl_seconds=10, now=2000.0)
    assert store.list_query_cache("researcher") == []


def test_schema_version_rejects_newer_file(tmp_path):
    path = tmp_path / "future.db"
    s = Store(path)
    s.conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    s.conn.commit()
    s.close()

    with pytest.raises(RuntimeError, match="newer version"):
        Store(path)


def test_reopening_existing_file_preserves_data(tmp_path):
    path = tmp_path / "persist.db"
    s1 = Store(path)
    doc_id = s1.insert_document("researcher")
    s1.insert_chunks(doc_id, "researcher", ["persisted chunk"])
    s1.close()

    s2 = Store(path)
    hits = s2.search_chunks_fts("persisted", ["researcher"], limit=5)
    assert len(hits) == 1
    s2.close()
