import asyncio

import pytest

from rmbr.embed import FakeEmbedder
from rmbr.index import Index
from rmbr.policy import Policy


def make_index(path, namespace="default", **kwargs):
    return Index(str(path), namespace=namespace, embedder=FakeEmbedder(dimension=16), **kwargs)


class CallCountingEmbedder:
    """Wraps FakeEmbedder and records how many times .embed() itself was
    called (not how many texts) - proves batching collapsed N per-document
    calls into one call over the whole batch."""

    def __init__(self):
        self.model_name = "call-counting-fake"
        self._inner = FakeEmbedder(dimension=16, model_name=self.model_name)
        self.call_count = 0

    def embed(self, texts):
        self.call_count += 1
        return self._inner.embed(texts)


def test_add_text_and_search_roundtrip(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)
    idx.add_text("the deployment guide covers docker and kubernetes setup", source="deploy.md")

    hits = idx.search("how do I deploy with docker?")
    assert len(hits) >= 1
    assert "docker" in hits[0].text


def test_add_text_rejects_empty_input(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)
    with pytest.raises(ValueError):
        idx.add_text("   ")


def test_add_texts_bulk_ingest_and_search(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)
    document_ids = idx.add_texts(
        ["docker deployment guide", "kubernetes cluster setup", "unrelated gardening tips"]
    )
    assert len(document_ids) == 3

    hits = idx.search("docker deployment")
    assert len(hits) >= 1
    assert "docker" in hits[0].text


def test_add_texts_persists_ann_index_once_not_per_document(tmp_path, monkeypatch):
    db = tmp_path / "agents.db"
    idx = make_index(db)

    save_calls = []
    original_set_ann_blob = idx._store.set_ann_blob

    def counting_set_ann_blob(*args, **kwargs):
        save_calls.append(1)
        return original_set_ann_blob(*args, **kwargs)

    monkeypatch.setattr(idx._store, "set_ann_blob", counting_set_ann_blob)
    idx.add_texts([f"document number {i}" for i in range(10)])

    assert len(save_calls) == 1  # one blob write for the whole batch, not one per document


def test_add_files_persists_ann_index_once_not_per_file(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(5):
        (docs / f"doc{i}.txt").write_text(f"content of document {i}")

    db = tmp_path / "agents.db"
    idx = make_index(db)

    save_calls = []
    original_set_ann_blob = idx._store.set_ann_blob

    def counting_set_ann_blob(*args, **kwargs):
        save_calls.append(1)
        return original_set_ann_blob(*args, **kwargs)

    monkeypatch.setattr(idx._store, "set_ann_blob", counting_set_ann_blob)
    idx.add_files(str(docs))

    assert len(save_calls) == 1


def test_add_texts_empty_list_is_noop(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)
    assert idx.add_texts([]) == []


def test_add_files_indexes_directory(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Setup\n\nInstall the package with pip.")
    (docs / "b.txt").write_text("Troubleshooting network timeouts.")
    (docs / "ignored.png").write_bytes(b"\x89PNG\r\n")

    db = tmp_path / "agents.db"
    idx = make_index(db)
    document_ids = idx.add_files(str(docs))

    assert len(document_ids) == 2  # png skipped
    hits = idx.search("install package pip")
    assert len(hits) >= 1


def test_add_files_single_file(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("a single standalone note file")

    db = tmp_path / "agents.db"
    idx = make_index(db)
    document_ids = idx.add_files(str(doc))
    assert len(document_ids) == 1


def test_delete_removes_document_and_chunks(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)
    doc_id = idx.add_text("some indexed content about rockets")
    idx.delete(doc_id)

    hits = idx.search("rockets")
    assert len(hits) == 0


def test_delete_nonexistent_is_noop(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)
    idx.delete(99999)  # should not raise


def test_search_is_namespace_scoped_by_default(tmp_path):
    db = tmp_path / "agents.db"
    researcher = make_index(db, namespace="researcher")
    coder = make_index(db, namespace="coder")

    researcher.add_text("researcher-only findings about widgets")
    coder.add_text("coder-only findings about widgets")

    hits = coder.search("findings about widgets")
    assert all(h.namespace == "coder" for h in hits)
    assert len(hits) == 1


def test_search_cross_namespace_denied_by_default(tmp_path):
    db = tmp_path / "agents.db"
    coder = make_index(db, namespace="coder")
    coder.add_text("coder doc")

    with pytest.raises(PermissionError):
        coder.search("doc", namespaces="researcher")


def test_search_cross_namespace_allowed_with_grant(tmp_path):
    db = tmp_path / "agents.db"
    researcher = make_index(db, namespace="researcher")
    researcher.add_text("shared knowledge about the release process")

    policy = Policy()
    policy.allow("supervisor", read="*")
    supervisor = make_index(db, namespace="supervisor", policy=policy)

    hits = supervisor.search("release process", namespaces="*")
    assert any(h.namespace == "researcher" for h in hits)


def test_markdown_files_get_header_breadcrumbs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Deployment\n\nRun the deploy script from the repo root.")

    db = tmp_path / "agents.db"
    idx = make_index(db)
    idx.add_files(str(docs))

    hits = idx.search("deploy script")
    assert any("# Deployment" in h.text for h in hits)


def test_index_persists_across_instances(tmp_path):
    db = tmp_path / "agents.db"
    make_index(db).add_text("persisted document content")

    reopened = make_index(db)
    hits = reopened.search("persisted document")
    assert len(hits) == 1


def test_search_where_filters_by_metadata(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)
    idx.add_text("deployment guide for docker", source="docs/deploy.md", metadata={"team": "infra"})
    idx.add_text("deployment guide for testing", source="docs/test.md", metadata={"team": "qa"})

    hits = idx.search("deployment guide", where={"team": "infra"})
    assert len(hits) == 1
    assert hits[0].metadata["team"] == "infra"


def test_add_texts_makes_exactly_one_embed_call_for_the_whole_batch(tmp_path):
    db = tmp_path / "agents.db"
    embedder = CallCountingEmbedder()
    idx = Index(str(db), embedder=embedder)

    idx.add_texts([f"document number {i}" for i in range(10)])

    assert embedder.call_count == 1  # not 10 - true batch ingestion, not a loop


def test_add_texts_returns_ingest_result_with_timings(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)

    result = idx.add_texts(["first document", "second document"])

    assert len(result) == 2
    assert isinstance(result, list)  # still usable as a plain list of ids
    for key in ("chunk_ms", "embed_ms", "store_ms", "ann_ms", "total_ms", "docs_per_second"):
        assert key in result.timings


def test_add_texts_empty_list_returns_empty_result_no_error(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)

    result = idx.add_texts([])

    assert result == []
    assert result.timings == {}


def test_add_files_merges_timings_across_splitter_groups(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Title\n\nmarkdown content here")
    (docs / "b.py").write_text("def foo():\n    return 1\n")

    db = tmp_path / "agents.db"
    idx = make_index(db)
    result = idx.add_files(str(docs))

    assert len(result) == 2
    assert result.timings["docs_per_second"] > 0


def test_splitter_python_auto_detected_for_py_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "server.py").write_text("def start_server():\n    return 'running'\n")

    db = tmp_path / "agents.db"
    idx = make_index(db)
    idx.add_files(str(docs))

    hits = idx.search("start server")
    assert any("# def start_server" in h.text for h in hits)


def test_splitter_explicit_python_on_add_text(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)
    idx.add_text("def handler():\n    return 'ok'\n", splitter="python")

    hits = idx.search("handler")
    assert any("# def handler" in h.text for h in hits)


def test_splitter_accepts_custom_callable(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)

    def shout_splitter(text, chunk_size, chunk_overlap):
        return [text.upper()]

    idx.add_text("hello world", splitter=shout_splitter)
    hits = idx.search("HELLO")
    assert len(hits) == 1
    assert hits[0].text == "HELLO WORLD"


def test_unknown_splitter_name_raises_clear_error(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)
    with pytest.raises(ValueError, match="Unknown splitter"):
        idx.add_text("some text", splitter="not-a-real-splitter")


def test_async_aadd_text_and_asearch(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)

    async def run():
        await idx.aadd_text("deployment guide for docker")
        return await idx.asearch("docker deployment")

    hits = asyncio.run(run())
    assert len(hits) == 1


def test_async_aadd_texts_and_aadd_files(tmp_path):
    db = tmp_path / "agents.db"
    idx = make_index(db)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("some content about testing")

    async def run():
        result1 = await idx.aadd_texts(["one document", "another document"])
        result2 = await idx.aadd_files(str(docs))
        return result1, result2

    result1, result2 = asyncio.run(run())
    assert len(result1) == 2
    assert len(result2) == 1


def test_async_concurrent_calls_on_same_index_do_not_error(tmp_path):
    """The per-instance lock should serialize concurrent async calls safely,
    not deadlock or corrupt state - not a performance claim, a safety one."""
    db = tmp_path / "agents.db"
    idx = make_index(db)

    async def run():
        await asyncio.gather(
            idx.aadd_text("first concurrent document"),
            idx.aadd_text("second concurrent document"),
            idx.aadd_text("third concurrent document"),
        )
        return await idx.asearch("concurrent document", k=10)

    hits = asyncio.run(run())
    assert len(hits) == 3
