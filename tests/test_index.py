import pytest

from rmbr.embed import FakeEmbedder
from rmbr.index import Index
from rmbr.policy import Policy


def make_index(path, namespace="default", **kwargs):
    return Index(str(path), namespace=namespace, embedder=FakeEmbedder(dimension=16), **kwargs)


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
