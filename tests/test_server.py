"""HTTP server tests run in-process against `build_app()` via Starlette's
TestClient — no real network socket, no uvicorn process, so these stay
fast and match the pattern in test_mcp.py.
"""

import pytest

from rmbr.embed import FakeEmbedder
from rmbr.server import build_app

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient  # noqa: E402


def make_client(path, **kwargs):
    app = build_app(str(path), embedder=FakeEmbedder(dimension=16), **kwargs)
    return TestClient(app)


def test_health_reports_namespace_and_version(tmp_path):
    from rmbr import __version__

    client = make_client(tmp_path / "agents.db", namespace="coder")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "namespace": "coder", "version": __version__}


def test_health_does_not_require_auth(tmp_path):
    client = make_client(tmp_path / "agents.db", token="secret")
    assert client.get("/health").status_code == 200


def test_remember_then_get(tmp_path):
    client = make_client(tmp_path / "agents.db")
    r = client.post("/memories", json={"text": "user prefers dark mode"})
    assert r.status_code == 201
    memory_id = r.json()["id"]

    r = client.get(f"/memories/{memory_id}")
    assert r.status_code == 200
    assert r.json()["text"] == "user prefers dark mode"


def test_remember_missing_text_is_400(tmp_path):
    client = make_client(tmp_path / "agents.db")
    r = client.post("/memories", json={})
    assert r.status_code == 400


def test_get_missing_memory_is_404(tmp_path):
    client = make_client(tmp_path / "agents.db")
    r = client.get("/memories/99999")
    assert r.status_code == 404


def test_list_memories(tmp_path):
    client = make_client(tmp_path / "agents.db")
    client.post("/memories", json={"text": "first"})
    client.post("/memories", json={"text": "second"})

    r = client.get("/memories")
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_update_changes_text_same_id(tmp_path):
    client = make_client(tmp_path / "agents.db")
    memory_id = client.post("/memories", json={"text": "old"}).json()["id"]

    r = client.patch(f"/memories/{memory_id}", json={"text": "new"})
    assert r.status_code == 200

    got = client.get(f"/memories/{memory_id}").json()
    assert got["text"] == "new"
    assert got["id"] == memory_id


def test_forget_removes_memory(tmp_path):
    client = make_client(tmp_path / "agents.db")
    memory_id = client.post("/memories", json={"text": "temp"}).json()["id"]

    r = client.delete(f"/memories/{memory_id}")
    assert r.status_code == 204
    assert client.get(f"/memories/{memory_id}").status_code == 404


def test_recall_finds_relevant_memory(tmp_path):
    client = make_client(tmp_path / "agents.db")
    client.post("/memories", json={"text": "user prefers dark mode"})

    r = client.post("/memories/search", json={"query": "dark mode"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 1
    assert "timings" in body


def test_memory_stats(tmp_path):
    client = make_client(tmp_path / "agents.db", namespace="coder")
    client.post("/memories", json={"text": "a memory"})

    r = client.get("/memories/stats")
    assert r.status_code == 200
    assert r.json()["coder"]["count"] == 1


def test_add_document_then_search(tmp_path):
    client = make_client(tmp_path / "agents.db")
    r = client.post("/documents", json={"text": "the deployment guide covers docker"})
    assert r.status_code == 201

    r = client.post("/search", json={"query": "docker deployment"})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1


def test_delete_document(tmp_path):
    client = make_client(tmp_path / "agents.db")
    document_id = client.post("/documents", json={"text": "some content here"}).json()["id"]

    r = client.delete(f"/documents/{document_id}")
    assert r.status_code == 204


def test_document_stats(tmp_path):
    client = make_client(tmp_path / "agents.db", namespace="coder")
    client.post("/documents", json={"text": "some content here"})

    r = client.get("/documents/stats")
    assert r.status_code == 200
    assert r.json()["coder"]["documents"] == 1


def test_read_only_blocks_writes_but_not_reads(tmp_path):
    client = make_client(tmp_path / "agents.db", read_only=True)

    assert client.post("/memories", json={"text": "x"}).status_code == 405
    assert client.post("/documents", json={"text": "x"}).status_code == 405
    assert client.post("/memories/search", json={"query": "x"}).status_code == 200
    assert client.post("/search", json={"query": "x"}).status_code == 200


def test_auth_required_when_token_set(tmp_path):
    client = make_client(tmp_path / "agents.db", token="secret123")

    assert client.post("/memories", json={"text": "x"}).status_code == 401
    r = client.post("/memories", json={"text": "x"}, headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    r = client.post("/memories", json={"text": "x"}, headers={"Authorization": "Bearer secret123"})
    assert r.status_code == 201


def test_no_auth_required_when_token_unset(tmp_path):
    client = make_client(tmp_path / "agents.db")
    assert client.post("/memories", json={"text": "x"}).status_code == 201


def test_no_namespace_field_accepted_anywhere(tmp_path):
    """The same structural guarantee MCP has: a caller can't pass a
    namespace to reach outside the one this server was pinned to."""
    client = make_client(tmp_path / "agents.db", namespace="coder")
    r = client.post("/memories", json={"text": "x", "namespace": "other"})
    assert r.status_code == 201
    record = client.get(f"/memories/{r.json()['id']}").json()
    assert record["namespace"] == "coder"
