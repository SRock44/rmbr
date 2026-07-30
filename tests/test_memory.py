import asyncio

import pytest

from rmbr.embed import FakeEmbedder
from rmbr.memory import Memory
from rmbr.policy import Policy


def make_memory(path, namespace, **kwargs):
    return Memory(str(path), namespace, embedder=FakeEmbedder(dimension=16), **kwargs)


def test_remember_and_recall_roundtrip(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    mem.remember("user prefers dark mode and short answers")

    hits = mem.recall("user preferences")
    assert len(hits) == 1
    assert "dark mode" in hits[0].text


def test_list_returns_most_recent_first(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "coder")
    mem.remember("first memory")
    mem.remember("second memory")

    memories = mem.list()
    assert [m.text for m in memories] == ["second memory", "first memory"]


def test_forget_removes_memory(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "coder")
    memory_id = mem.remember("temporary note")
    mem.forget(memory_id)

    assert mem.list() == []
    hits = mem.recall("temporary note")
    assert len(hits) == 0


def test_forget_nonexistent_id_is_noop(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "coder")
    mem.forget(99999)  # should not raise


def test_recall_is_namespace_scoped_by_default(tmp_path):
    db = tmp_path / "agents.db"
    researcher = make_memory(db, "researcher")
    coder = make_memory(db, "coder")

    researcher.remember("researcher secret preference")
    coder.remember("coder secret preference")

    hits = coder.recall("secret preference")
    assert all(h.namespace == "coder" for h in hits)
    assert len(hits) == 1


def test_recall_cross_namespace_denied_by_default(tmp_path):
    db = tmp_path / "agents.db"
    coder = make_memory(db, "coder")
    coder.remember("coder note")

    with pytest.raises(PermissionError):
        coder.recall("note", namespaces="researcher")


def test_recall_cross_namespace_allowed_with_grant(tmp_path):
    db = tmp_path / "agents.db"
    researcher = make_memory(db, "researcher")
    researcher.remember("researcher shared fact")

    policy = Policy()
    policy.allow("supervisor", read="*")
    supervisor = make_memory(db, "supervisor", policy=policy)

    hits = supervisor.recall("shared fact", namespaces="*")
    assert any(h.namespace == "researcher" for h in hits)


def test_forget_across_namespace_denied_by_default(tmp_path):
    db = tmp_path / "agents.db"
    researcher = make_memory(db, "researcher")
    memory_id = researcher.remember("researcher note")

    coder = make_memory(db, "coder")
    with pytest.raises(PermissionError):
        coder.forget(memory_id)


def test_memory_persists_across_instances(tmp_path):
    db = tmp_path / "agents.db"
    make_memory(db, "researcher").remember("persisted note")

    reopened = make_memory(db, "researcher")
    hits = reopened.recall("persisted note")
    assert len(hits) == 1


def test_context_manager_closes_store(tmp_path):
    db = tmp_path / "agents.db"
    with make_memory(db, "researcher") as mem:
        mem.remember("a note")
    with pytest.raises(Exception):
        mem._store.conn.execute("SELECT 1")


def test_recall_where_filters_by_metadata(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    mem.remember("user prefers dark mode", metadata={"category": "preference"})
    mem.remember("user prefers dark mode too", metadata={"category": "observation"})

    hits = mem.recall("user prefers dark mode", where={"category": "preference"})
    assert len(hits) == 1
    assert hits[0].metadata["category"] == "preference"


def test_async_aremember_and_arecall(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")

    async def run():
        await mem.aremember("user prefers dark mode and short answers")
        return await mem.arecall("user preferences")

    hits = asyncio.run(run())
    assert len(hits) == 1


def test_async_aforget(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")

    async def run():
        memory_id = await mem.aremember("temporary note")
        await mem.aforget(memory_id)
        return await mem.arecall("temporary note")

    hits = asyncio.run(run())
    assert len(hits) == 0


def test_async_concurrent_remember_calls_do_not_error(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")

    async def run():
        await asyncio.gather(
            mem.aremember("first concurrent memory"),
            mem.aremember("second concurrent memory"),
            mem.aremember("third concurrent memory"),
        )
        return mem.list()

    memories = asyncio.run(run())
    assert len(memories) == 3


def test_async_gather_across_namespaces_for_supervisor_pattern(tmp_path):
    """The concrete multi-agent story: a supervisor reading several
    granted namespaces concurrently via asyncio.gather, not sequentially."""
    from rmbr.policy import Policy

    db = tmp_path / "agents.db"
    make_memory(db, "coder").remember("coder note about the api")
    make_memory(db, "researcher").remember("researcher note about the api")

    policy = Policy()
    policy.allow("supervisor", read="*")
    supervisor = Memory(str(db), "supervisor", embedder=FakeEmbedder(dimension=16), policy=policy)

    async def run():
        coder_hits, researcher_hits = await asyncio.gather(
            supervisor.arecall("api", namespaces="coder"),
            supervisor.arecall("api", namespaces="researcher"),
        )
        return coder_hits, researcher_hits

    coder_hits, researcher_hits = asyncio.run(run())
    assert len(coder_hits) == 1
    assert len(researcher_hits) == 1
