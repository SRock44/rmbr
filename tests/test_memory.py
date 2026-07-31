import asyncio
import time

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


def test_remember_without_dedupe_threshold_always_inserts(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    mem.remember("the same note")
    mem.remember("the same note")

    assert len(mem.list()) == 2


def test_remember_with_dedupe_threshold_updates_in_place(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    first_id = mem.remember("the same note", dedupe_threshold=0.9)
    second_id = mem.remember("the same note", metadata={"updated": True}, dedupe_threshold=0.9)

    assert first_id == second_id
    memories = mem.list()
    assert len(memories) == 1
    assert memories[0].metadata == {"updated": True}


def test_remember_dedupe_threshold_can_be_set_at_construction(tmp_path):
    db = tmp_path / "agents.db"
    mem = Memory(str(db), "researcher", embedder=FakeEmbedder(dimension=16), dedupe_threshold=0.9)
    mem.remember("the same note")
    mem.remember("the same note")

    assert len(mem.list()) == 1


def test_remember_dedupe_does_not_merge_across_namespaces(tmp_path):
    db = tmp_path / "agents.db"
    researcher = make_memory(db, "researcher")
    coder = make_memory(db, "coder")

    researcher.remember("the same note", dedupe_threshold=0.9)
    coder.remember("the same note", dedupe_threshold=0.9)

    assert len(researcher.list()) == 1
    assert len(coder.list()) == 1


def test_max_memories_evicts_oldest(tmp_path):
    db = tmp_path / "agents.db"
    mem = Memory(str(db), "researcher", embedder=FakeEmbedder(dimension=16), max_memories=2)
    mem.remember("first")
    mem.remember("second")
    mem.remember("third")

    memories = mem.list()
    assert len(memories) == 2
    assert {m.text for m in memories} == {"second", "third"}


def test_max_memories_never_evicts_pinned_memory(tmp_path):
    db = tmp_path / "agents.db"
    mem = Memory(str(db), "researcher", embedder=FakeEmbedder(dimension=16), max_memories=2)
    mem.remember("critical fact", pinned=True)
    mem.remember("second")
    mem.remember("third")
    mem.remember("fourth")

    memories = mem.list()
    texts = {m.text for m in memories}
    assert "critical fact" in texts  # survives despite being oldest
    # only the most recent 2 *unpinned* memories are kept alongside it
    assert len(memories) == 3
    assert texts == {"critical fact", "third", "fourth"}


def test_pinned_memory_findable_via_where_filter(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    mem.remember("important", pinned=True)
    mem.remember("ordinary")

    pinned = mem.list(where={"_pinned": True})
    assert [m.text for m in pinned] == ["important"]


def test_forget_older_than_deletes_only_stale_memories(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    old_id = mem.remember("stale note")
    fresh_id = mem.remember("fresh note")

    # Backdate the first memory directly - real elapsed time would make
    # this test slow and flaky.
    old_cutoff = "2000-01-01T00:00:00+00:00"
    mem._store.conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (old_cutoff, old_id))
    mem._store.conn.commit()

    deleted_count = mem.forget_older_than(seconds=3600)

    assert deleted_count == 1
    remaining_ids = {m.id for m in mem.list()}
    assert remaining_ids == {fresh_id}


def test_recall_min_similarity_drops_weak_matches(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    mem.remember("user prefers dark mode")

    all_hits = mem.recall("user prefers dark mode")
    real_similarity = all_hits[0].vector_score

    filtered = mem.recall("user prefers dark mode", min_similarity=real_similarity + 0.5)
    assert len(filtered) == 0


def test_recall_recency_weight_favors_newer_memory(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    old_id = mem.remember("status update")
    new_id = mem.remember("status update")

    mem._store.conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?", ("2000-01-01T00:00:00+00:00", old_id)
    )
    mem._store.conn.commit()

    hits = mem.recall("status update", recency_weight=1.0, recency_half_life_seconds=3600)
    assert hits[0].id == new_id


def test_recall_rerank_uses_cross_encoder_score(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    mem.remember("the weak match")
    mem.remember("the strong match")

    class TextKeyedReranker:
        model_name = "fake-reranker"

        def rerank(self, query, documents):
            return [1.0 if doc == "the strong match" else 0.0 for doc in documents]

    mem._reranker = TextKeyedReranker()
    hits = mem.recall("match", rerank=True, k=2)

    assert hits[0].text == "the strong match"
    assert hits[0].score == 1.0


def test_remember_turn_stores_role_and_session_in_metadata_not_text(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "assistant")
    memory_id = mem.remember_turn("user", "I prefer dark mode", session_id="conv-1")

    memories = mem.list()
    assert len(memories) == 1
    assert memories[0].id == memory_id
    assert memories[0].text == "I prefer dark mode"  # not prefixed with "user: "
    assert memories[0].metadata == {"role": "user", "session_id": "conv-1"}


def test_remember_turn_without_session_id_omits_it_from_metadata(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "assistant")
    mem.remember_turn("assistant", "sure, I can help with that")

    assert mem.list()[0].metadata == {"role": "assistant"}


def test_remember_turn_merges_caller_metadata(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "assistant")
    mem.remember_turn("user", "note", session_id="conv-1", metadata={"topic": "settings"})

    assert mem.list()[0].metadata == {"role": "user", "session_id": "conv-1", "topic": "settings"}


def test_list_where_filters_by_session_for_conversation_replay(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "assistant")
    mem.remember_turn("user", "message in conv 1", session_id="conv-1")
    mem.remember_turn("assistant", "reply in conv 1", session_id="conv-1")
    mem.remember_turn("user", "message in conv 2", session_id="conv-2")

    conv1 = mem.list(where={"session_id": "conv-1"})
    assert len(conv1) == 2
    assert all(m.metadata["session_id"] == "conv-1" for m in conv1)
    # most-recent-first, same ordering guarantee as plain list()
    assert conv1[0].text == "reply in conv 1"


def test_list_where_respects_limit_after_filtering(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "assistant")
    for i in range(3):
        mem.remember_turn("user", f"message {i}", session_id="conv-1")
    mem.remember_turn("user", "other session", session_id="conv-2")

    limited = mem.list(where={"session_id": "conv-1"}, limit=2)
    assert len(limited) == 2


def test_async_aremember_turn(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "assistant")

    async def run():
        return await mem.aremember_turn("user", "async note", session_id="conv-async")

    memory_id = asyncio.run(run())
    assert mem.list()[0].id == memory_id
    assert mem.list()[0].metadata["session_id"] == "conv-async"


def test_stats_reports_count_and_time_range_for_own_namespace(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    mem.remember("first")
    mem.remember("second")

    stats = mem.stats()
    assert stats["researcher"]["count"] == 2
    assert stats["researcher"]["oldest"] <= stats["researcher"]["newest"]


def test_stats_empty_namespace_reports_zero_count(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")

    stats = mem.stats()
    assert stats["researcher"]["count"] == 0
    assert stats["researcher"]["oldest"] is None


def test_stats_wildcard_respects_policy(tmp_path):
    db = tmp_path / "agents.db"
    policy = Policy()
    policy.allow("supervisor", read="*")
    make_memory(db, "coder").remember("coder note")
    make_memory(db, "researcher").remember("researcher note")
    supervisor = Memory(str(db), "supervisor", embedder=FakeEmbedder(dimension=16), policy=policy)

    stats = supervisor.stats(namespaces="*")
    assert stats["coder"]["count"] == 1
    assert stats["researcher"]["count"] == 1


def test_stats_explicit_cross_namespace_denied_by_default(tmp_path):
    db = tmp_path / "agents.db"
    make_memory(db, "researcher").remember("note")
    coder = make_memory(db, "coder")

    with pytest.raises(PermissionError):
        coder.stats(namespaces="researcher")


def test_integrity_check_reports_no_problems_on_a_healthy_store(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    mem.remember("first")
    mem.remember("second")

    assert mem.integrity_check() == []


def test_integrity_check_flags_a_row_with_no_vector(tmp_path):
    db = tmp_path / "agents.db"
    mem = make_memory(db, "researcher")
    mem.remember("first")
    # Simulate corruption: a row exists in SQLite with no corresponding
    # vector, bypassing rmbr's own write path entirely.
    mem._store.conn.execute(
        "INSERT INTO memories(namespace, text, metadata, created_at) VALUES (?, ?, ?, ?)",
        ("researcher", "orphaned row", "{}", "2020-01-01T00:00:00+00:00"),
    )
    mem._store.conn.commit()

    problems = mem.integrity_check()
    assert len(problems) == 1
    assert "no vector" in problems[0]
