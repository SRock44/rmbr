from rmbr.embed import FakeEmbedder
from rmbr.index import Index
from rmbr.memory import Memory


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
