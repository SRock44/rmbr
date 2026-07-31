import pytest

from rmbr.embed import FakeEmbedder
from rmbr.index import Index
from rmbr.memory import Memory
from rmbr.tools import ToolCallError


def make_index(path, **kwargs):
    return Index(str(path), embedder=FakeEmbedder(dimension=16), **kwargs)


def make_memory(path, namespace="default", **kwargs):
    return Memory(str(path), namespace, embedder=FakeEmbedder(dimension=16), **kwargs)


def test_index_as_tool_has_openai_and_anthropic_shapes(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    tool = idx.as_tool()

    openai_shape = tool.to_openai()
    assert openai_shape["type"] == "function"
    assert openai_shape["function"]["name"] == "search"
    assert "query" in openai_shape["function"]["parameters"]["properties"]

    anthropic_shape = tool.to_anthropic()
    assert anthropic_shape["name"] == "search"
    assert "query" in anthropic_shape["input_schema"]["properties"]


def test_index_as_tool_call_dispatches_to_search(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("the deployment guide covers docker")
    tool = idx.as_tool()

    results = tool.call(query="docker deployment")
    assert len(results) == 1
    assert "docker" in results[0]["text"]
    assert "bm25_score" in results[0]
    assert "vector_score" in results[0]


def test_index_as_tool_custom_name(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    tool = idx.as_tool(name="search_docs")
    assert tool.to_openai()["function"]["name"] == "search_docs"


def test_index_as_tool_schema_exposes_where_min_similarity_rerank(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    properties = idx.as_tool().to_openai()["function"]["parameters"]["properties"]
    assert {"where", "min_similarity", "rerank"} <= properties.keys()


def test_index_as_tool_call_forwards_where_to_search(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("release notes", metadata={"tier": "internal"})
    idx.add_text("release notes", metadata={"tier": "public"})
    tool = idx.as_tool()

    results = tool.call(query="release notes", where={"tier": "public"})
    assert len(results) == 1
    assert results[0]["metadata"]["tier"] == "public"


def test_index_as_tool_call_forwards_min_similarity_to_search(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("quarterly revenue report")
    tool = idx.as_tool()

    unfiltered = tool.call(query="quarterly revenue report")
    real_similarity = unfiltered[0]["vector_score"]

    filtered = tool.call(query="quarterly revenue report", min_similarity=real_similarity + 0.5)
    assert len(filtered) == 0


def test_memory_as_tools_includes_remember_by_default(tmp_path):
    mem = make_memory(tmp_path / "agents.db")
    tools = mem.as_tools()
    names = {t.name for t in tools}
    assert names == {"recall", "remember"}


def test_memory_as_tools_read_only_excludes_remember(tmp_path):
    mem = make_memory(tmp_path / "agents.db")
    tools = mem.as_tools(read_only=True)
    names = {t.name for t in tools}
    assert names == {"recall"}


def test_memory_as_tools_remember_call_dispatches(tmp_path):
    mem = make_memory(tmp_path / "agents.db")
    tools = {t.name: t for t in mem.as_tools()}

    memory_id = tools["remember"].call(text="user prefers dark mode")
    assert isinstance(memory_id, int)

    results = tools["recall"].call(query="dark mode")
    assert len(results) == 1
    assert results[0]["id"] == memory_id


def test_memory_as_tools_recall_schema_exposes_where_min_similarity_rerank(tmp_path):
    mem = make_memory(tmp_path / "agents.db")
    recall_tool = next(t for t in mem.as_tools() if t.name == "recall")
    properties = recall_tool.to_anthropic()["input_schema"]["properties"]
    assert {"where", "min_similarity", "rerank"} <= properties.keys()


def test_memory_as_tools_recall_call_forwards_where(tmp_path):
    mem = make_memory(tmp_path / "agents.db")
    mem.remember("dark mode note", metadata={"category": "preference"})
    mem.remember("dark mode note too", metadata={"category": "observation"})
    recall_tool = next(t for t in mem.as_tools() if t.name == "recall")

    results = recall_tool.call(query="dark mode", where={"category": "preference"})
    assert len(results) == 1
    assert results[0]["metadata"]["category"] == "preference"


def test_tool_call_rejects_unexpected_argument_without_crashing(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("some document")
    tool = idx.as_tool()

    with pytest.raises(ToolCallError, match=r"unexpected argument.*'bogus_arg'"):
        tool.call(query="some document", bogus_arg="oops")


def test_tool_call_error_names_valid_arguments(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    tool = idx.as_tool()

    with pytest.raises(ToolCallError, match="query"):
        tool.call(query="x", made_up="y")


def test_tool_call_rejects_missing_required_argument(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    tool = idx.as_tool()

    with pytest.raises(ToolCallError, match="missing required"):
        tool.call()


def test_tool_call_error_is_a_type_error_for_backward_compatible_catching(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    tool = idx.as_tool()

    with pytest.raises(TypeError):
        tool.call(query="x", bogus_arg="y")


def test_built_in_schemas_set_additional_properties_false(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    mem = make_memory(tmp_path / "agents.db")

    for tool in [idx.as_tool(), *mem.as_tools()]:
        assert tool.parameters.get("additionalProperties") is False


def test_to_anthropic_strict_adds_top_level_strict_field(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    tool = idx.as_tool()

    assert "strict" not in tool.to_anthropic()
    assert tool.to_anthropic(strict=True)["strict"] is True


def test_to_openai_strict_adds_strict_field_inside_function(tmp_path):
    idx = make_index(tmp_path / "agents.db")
    tool = idx.as_tool()

    assert "strict" not in tool.to_openai()["function"]
    assert tool.to_openai(strict=True)["function"]["strict"] is True


def test_memory_as_tools_remember_schema_exposes_pinned(tmp_path):
    mem = make_memory(tmp_path / "agents.db")
    remember_tool = next(t for t in mem.as_tools() if t.name == "remember")
    assert "pinned" in remember_tool.to_openai()["function"]["parameters"]["properties"]


def test_memory_as_tools_remember_call_forwards_pinned(tmp_path):
    mem = make_memory(tmp_path / "agents.db", max_memories=1)
    tools = {t.name: t for t in mem.as_tools()}

    tools["remember"].call(text="critical fact", pinned=True)
    tools["remember"].call(text="second fact")
    tools["remember"].call(text="third fact")

    texts = {m.text for m in mem.list()}
    assert "critical fact" in texts
