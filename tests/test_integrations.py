import asyncio

import pytest

from rmbr.embed import FakeEmbedder
from rmbr.index import Index


def make_index(path, **kwargs):
    return Index(str(path), embedder=FakeEmbedder(dimension=16), **kwargs)


def test_as_langchain_retriever_returns_documents(tmp_path):
    pytest.importorskip("langchain_core")  # optional extra - pip install rmbr[langchain]
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("the deployment guide covers docker and kubernetes")

    retriever = idx.as_langchain_retriever(k=3)
    docs = retriever.invoke("docker deployment")

    assert len(docs) == 1
    assert "docker" in docs[0].page_content
    assert docs[0].metadata["rmbr_id"] is not None
    assert docs[0].metadata["namespace"] == "default"


def test_as_langchain_retriever_async(tmp_path):
    pytest.importorskip("langchain_core")
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("the deployment guide covers docker and kubernetes")

    retriever = idx.as_langchain_retriever(k=3)
    docs = asyncio.run(retriever.ainvoke("docker deployment"))
    assert len(docs) == 1


def test_as_langchain_retriever_passes_search_kwargs(tmp_path):
    pytest.importorskip("langchain_core")
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("release notes", metadata={"tier": "internal"})
    idx.add_text("release notes", metadata={"tier": "public"})

    retriever = idx.as_langchain_retriever(k=5, where={"tier": "public"})
    docs = retriever.invoke("release notes")
    assert len(docs) == 1
    assert docs[0].metadata["tier"] == "public"


def test_as_llamaindex_retriever_returns_nodes(tmp_path):
    pytest.importorskip("llama_index.core")  # optional extra - pip install rmbr[llamaindex]
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("the deployment guide covers docker and kubernetes")

    retriever = idx.as_llamaindex_retriever(k=3)
    nodes = retriever.retrieve("docker deployment")

    assert len(nodes) == 1
    assert "docker" in nodes[0].node.text
    assert nodes[0].score is not None


def test_as_llamaindex_retriever_async(tmp_path):
    pytest.importorskip("llama_index.core")
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("the deployment guide covers docker and kubernetes")

    retriever = idx.as_llamaindex_retriever(k=3)
    nodes = asyncio.run(retriever.aretrieve("docker deployment"))
    assert len(nodes) == 1


def test_as_llamaindex_retriever_passes_search_kwargs(tmp_path):
    pytest.importorskip("llama_index.core")
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("release notes", metadata={"tier": "internal"})
    idx.add_text("release notes", metadata={"tier": "public"})

    retriever = idx.as_llamaindex_retriever(k=5, where={"tier": "public"})
    nodes = retriever.retrieve("release notes")
    assert len(nodes) == 1
    assert nodes[0].node.metadata["tier"] == "public"


def make_langgraph_store(path, **kwargs):
    from rmbr.integrations.langgraph import as_store

    return as_store(str(path), embedder=FakeEmbedder(dimension=16), **kwargs)


def test_langgraph_store_put_then_get_round_trips_value(tmp_path):
    pytest.importorskip("langgraph.store.base")
    store = make_langgraph_store(tmp_path / "agents.db")

    store.put(("memories", "user-1"), "pref-1", {"text": "user prefers dark mode", "kind": "preference"})
    item = store.get(("memories", "user-1"), "pref-1")

    assert item is not None
    assert item.value == {"text": "user prefers dark mode", "kind": "preference"}
    assert item.key == "pref-1"
    assert item.namespace == ("memories", "user-1")


def test_langgraph_store_put_on_existing_key_overwrites(tmp_path):
    pytest.importorskip("langgraph.store.base")
    store = make_langgraph_store(tmp_path / "agents.db")

    store.put(("memories", "user-1"), "pref-1", {"text": "old value"})
    store.put(("memories", "user-1"), "pref-1", {"text": "new value"})

    item = store.get(("memories", "user-1"), "pref-1")
    assert item.value == {"text": "new value"}
    assert len(store.search(("memories", "user-1"))) == 1


def test_langgraph_store_delete_removes_the_item(tmp_path):
    pytest.importorskip("langgraph.store.base")
    store = make_langgraph_store(tmp_path / "agents.db")

    store.put(("memories", "user-1"), "pref-1", {"text": "user prefers dark mode"})
    store.delete(("memories", "user-1"), "pref-1")

    assert store.get(("memories", "user-1"), "pref-1") is None


def test_langgraph_store_search_by_query_ranks_by_relevance(tmp_path):
    pytest.importorskip("langgraph.store.base")
    store = make_langgraph_store(tmp_path / "agents.db")

    store.put(("memories", "user-1"), "pref-1", {"text": "user prefers dark mode"})
    store.put(("memories", "user-1"), "pref-2", {"text": "user likes concise answers"})

    results = store.search(("memories", "user-1"), query="dark mode")
    assert results[0].value["text"] == "user prefers dark mode"
    assert results[0].score is not None


def test_langgraph_store_search_by_filter_matches_value_fields(tmp_path):
    pytest.importorskip("langgraph.store.base")
    store = make_langgraph_store(tmp_path / "agents.db")

    store.put(("memories", "user-1"), "a", {"text": "one", "kind": "preference"})
    store.put(("memories", "user-1"), "b", {"text": "two", "kind": "preference"})
    store.put(("memories", "user-1"), "c", {"text": "three", "kind": "fact"})

    results = store.search(("memories", "user-1"), filter={"kind": "preference"})
    assert {r.key for r in results} == {"a", "b"}


def test_langgraph_store_search_by_namespace_prefix_spans_sub_namespaces(tmp_path):
    pytest.importorskip("langgraph.store.base")
    store = make_langgraph_store(tmp_path / "agents.db")

    store.put(("memories", "user-1"), "pref-1", {"text": "user one's memory"})
    store.put(("memories", "user-2"), "pref-1", {"text": "user two's memory"})

    results = store.search(("memories",))
    assert {r.namespace for r in results} == {("memories", "user-1"), ("memories", "user-2")}


def test_langgraph_store_list_namespaces_respects_prefix(tmp_path):
    pytest.importorskip("langgraph.store.base")
    store = make_langgraph_store(tmp_path / "agents.db")

    store.put(("memories", "user-1"), "a", {"text": "x"})
    store.put(("other", "user-1"), "a", {"text": "y"})

    namespaces = store.list_namespaces(prefix=("memories",))
    assert namespaces == [("memories", "user-1")]


def test_langgraph_store_async_get_matches_sync(tmp_path):
    pytest.importorskip("langgraph.store.base")
    store = make_langgraph_store(tmp_path / "agents.db")
    store.put(("memories", "user-1"), "pref-1", {"text": "user prefers dark mode"})

    item = asyncio.run(store.aget(("memories", "user-1"), "pref-1"))
    assert item.value == {"text": "user prefers dark mode"}


def make_mem0_shim(path, **kwargs):
    from rmbr.integrations.mem0_compat import Memory as Mem0Shim

    return Mem0Shim(str(path), embedder=FakeEmbedder(dimension=16), **kwargs)


def test_mem0_shim_add_with_infer_true_raises(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    with pytest.raises(NotImplementedError):
        m.add("hi", user_id="alex", infer=True)


def test_mem0_shim_add_requires_at_least_one_scope_id(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    with pytest.raises(ValueError):
        m.add("hi", infer=False)


def test_mem0_shim_add_then_search_round_trips(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    added = m.add("user prefers dark mode", user_id="alex", infer=False)

    assert added["results"][0]["memory"] == "user prefers dark mode"
    assert added["results"][0]["event"] == "ADD"

    found = m.search("dark mode", filters={"user_id": "alex"})
    assert found["results"][0]["memory"] == "user prefers dark mode"
    assert found["results"][0]["user_id"] == "alex"


def test_mem0_shim_add_list_of_messages_stores_one_per_message(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    result = m.add(
        [{"role": "user", "content": "I like Python"}, {"role": "assistant", "content": "Noted!"}],
        user_id="alex",
        infer=False,
    )
    assert [r["role"] for r in result["results"]] == ["user", "assistant"]

    all_memories = m.get_all(filters={"user_id": "alex"})
    assert len(all_memories["results"]) == 2


def test_mem0_shim_add_skips_system_messages(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    result = m.add(
        [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hi"}],
        user_id="alex",
        infer=False,
    )
    assert len(result["results"]) == 1
    assert result["results"][0]["role"] == "user"


def test_mem0_shim_get_returns_none_for_missing_id(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    assert m.get("99999") is None


def test_mem0_shim_get_all_requires_at_least_one_scope_id(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    with pytest.raises(ValueError):
        m.get_all()


def test_mem0_shim_update_preserves_id_and_changes_text(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    added = m.add("user prefers dark mode", user_id="alex", infer=False)
    memory_id = added["results"][0]["id"]

    m.update(memory_id, text="user prefers light mode")

    record = m.get(memory_id)
    assert record["id"] == memory_id
    assert record["memory"] == "user prefers light mode"


def test_mem0_shim_delete_removes_the_memory(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    added = m.add("temporary note", user_id="alex", infer=False)
    memory_id = added["results"][0]["id"]

    m.delete(memory_id)

    assert m.get(memory_id) is None


def test_mem0_shim_delete_all_clears_only_that_scope(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    m.add("alex's note", user_id="alex", infer=False)
    m.add("bob's note", user_id="bob", infer=False)

    m.delete_all(user_id="alex")

    assert m.get_all(filters={"user_id": "alex"})["results"] == []
    assert len(m.get_all(filters={"user_id": "bob"})["results"]) == 1


def test_mem0_shim_filter_operator_is_translated_to_rmbr_where(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    m.add("high priority", user_id="alex", metadata={"priority": 5}, infer=False)
    m.add("low priority", user_id="alex", metadata={"priority": 1}, infer=False)

    results = m.get_all(filters={"user_id": "alex", "priority": {"gt": 3}})
    assert len(results["results"]) == 1
    assert results["results"][0]["memory"] == "high priority"


def test_mem0_shim_unsupported_logical_combinator_raises(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    m.add("a note", user_id="alex", infer=False)

    with pytest.raises(NotImplementedError):
        m.search("note", filters={"user_id": "alex", "AND": []})


def test_mem0_shim_unsupported_filter_operator_raises(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    m.add("a note", user_id="alex", infer=False)

    with pytest.raises(NotImplementedError):
        m.search("note", filters={"user_id": "alex", "text": {"icontains": "no"}})


def test_mem0_shim_compound_scope_ids_isolate_namespace(tmp_path):
    m = make_mem0_shim(tmp_path / "agents.db")
    m.add("scoped to user+agent", user_id="alex", agent_id="coder", infer=False)
    m.add("scoped to user only", user_id="alex", infer=False)

    scoped = m.get_all(filters={"user_id": "alex", "agent_id": "coder"})
    user_only = m.get_all(filters={"user_id": "alex"})
    assert len(scoped["results"]) == 1
    assert len(user_only["results"]) == 1
