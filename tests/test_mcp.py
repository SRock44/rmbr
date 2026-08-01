"""MCP server tests run in-process against `build_mcp_server()` — no subprocess,
no real embedding model download, so these stay fast and offline in CI.
An end-to-end `python -m rmbr` + stdio client test is a reasonable addition
later, but it would need network access for the default embedder's model
download, which the rest of this suite deliberately avoids.
"""

import pytest

from rmbr.embed import FakeEmbedder
from rmbr.mcp_server import build_mcp_server


@pytest.fixture
def anyio_backend():
    return "asyncio"


def build(path, **kwargs):
    return build_mcp_server(str(path), embedder=FakeEmbedder(dimension=16), **kwargs)


def test_tools_are_registered(tmp_path):
    server = build(tmp_path / "agents.db", namespace="coder")
    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert names == {"search", "recall", "remember"}


def test_read_only_hides_remember_tool(tmp_path):
    server = build(tmp_path / "agents.db", namespace="coder", read_only=True)
    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert names == {"search", "recall"}


def test_no_tool_exposes_a_namespace_parameter(tmp_path):
    """The core security property: an external MCP client has no field to
    fill in to reach outside the namespace this server was pinned to."""
    server = build(tmp_path / "agents.db", namespace="coder")
    for tool in server._tool_manager.list_tools():
        properties = tool.parameters.get("properties", {})
        assert "namespace" not in properties, f"{tool.name} exposes a namespace parameter"


def test_every_tool_parameter_has_a_schema_description(tmp_path):
    """A tool-calling model sees the JSON schema, not the docstring - every
    parameter needs its own `description`, not just the tool as a whole."""
    server = build(tmp_path / "agents.db", namespace="coder")
    for tool in server._tool_manager.list_tools():
        properties = tool.parameters.get("properties", {})
        for param_name, schema in properties.items():
            assert schema.get("description"), f"{tool.name}.{param_name} has no schema description"


def test_search_tool_has_accurate_annotations(tmp_path):
    """Real MCP ToolAnnotations, not just prose - read-only, non-destructive,
    idempotent, and closed-world are all true of search() and worth stating
    formally so a client doesn't have to infer them from the description."""
    server = build(tmp_path / "agents.db", namespace="coder")
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
    annotations = tools["search"].annotations
    assert annotations is not None
    assert annotations.read_only_hint is True
    assert annotations.destructive_hint is False
    assert annotations.idempotent_hint is True
    assert annotations.open_world_hint is False


def test_search_tool_description_explains_when_to_use_it(tmp_path):
    """Glama's MCP quality scoring flagged the old one-liner for disclosing
    none of: read-only behavior, result ordering, k's overflow handling, or
    how to choose between search and recall - all should show up here."""
    server = build(tmp_path / "agents.db", namespace="coder")
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
    description = tools["search"].description.lower()
    assert "recall" in description  # usage guidance: when to use the other tool instead
    assert "read-only" in description  # behavioral disclosure
    assert "k" in description  # parameter behavior beyond the schema (overflow handling)


def test_recall_tool_has_accurate_annotations(tmp_path):
    server = build(tmp_path / "agents.db", namespace="coder")
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
    annotations = tools["recall"].annotations
    assert annotations is not None
    assert annotations.read_only_hint is True
    assert annotations.destructive_hint is False
    assert annotations.idempotent_hint is True
    assert annotations.open_world_hint is False


def test_recall_tool_description_explains_when_to_use_it(tmp_path):
    """Glama flagged recall's description for only implying (not stating)
    when to use it over search, and for adding nothing beyond the schema."""
    server = build(tmp_path / "agents.db", namespace="coder")
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
    description = tools["recall"].description.lower()
    assert "search" in description  # usage guidance: when to use the other tool instead
    assert "read-only" in description
    assert "k" in description


def test_remember_tool_has_accurate_annotations(tmp_path):
    server = build(tmp_path / "agents.db", namespace="coder")
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
    annotations = tools["remember"].annotations
    assert annotations is not None
    assert annotations.read_only_hint is False
    assert annotations.destructive_hint is False
    assert annotations.idempotent_hint is False
    assert annotations.open_world_hint is False


def test_remember_tool_description_discloses_eviction_and_pinning(tmp_path):
    """Glama flagged remember's description for omitting eviction behavior
    and what pinned actually does - both disclosed only in the schema
    before this, not in the tool-level description a model actually reads
    for consequences."""
    server = build(tmp_path / "agents.db", namespace="coder")
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
    description = tools["remember"].description.lower()
    assert "evict" in description  # behavioral disclosure: max_memories consequence
    assert "pinned" in description  # parameter interaction beyond the schema


@pytest.mark.anyio
async def test_remember_then_recall_through_the_tool_interface(tmp_path):
    server = build(tmp_path / "agents.db", namespace="coder")

    remember_result = await server.call_tool("remember", {"text": "user prefers dark mode"})
    assert remember_result.is_error is False

    recall_result = await server.call_tool("recall", {"query": "dark mode", "k": 5})
    assert recall_result.is_error is False
    hits = recall_result.structured_content["result"]
    assert len(hits) == 1
    assert "dark mode" in hits[0]["text"]


@pytest.mark.anyio
async def test_search_through_the_tool_interface(tmp_path):
    from rmbr.index import Index

    db = tmp_path / "agents.db"
    Index(str(db), namespace="coder", embedder=FakeEmbedder(dimension=16)).add_text(
        "the deployment guide covers docker setup"
    )

    server = build(db, namespace="coder")
    result = await server.call_tool("search", {"query": "docker deployment", "k": 5})
    assert result.is_error is False
    hits = result.structured_content["result"]
    assert len(hits) == 1
    assert "docker" in hits[0]["text"]


@pytest.mark.anyio
async def test_read_only_server_has_no_remember_tool_to_call(tmp_path):
    server = build(tmp_path / "agents.db", namespace="coder", read_only=True)
    with pytest.raises(Exception):
        await server.call_tool("remember", {"text": "should not work"})


def test_server_reports_rmbr_version(tmp_path):
    from rmbr import __version__

    server = build(tmp_path / "agents.db", namespace="coder")
    assert server.version == __version__


@pytest.mark.anyio
async def test_examples_resource_template_is_registered(tmp_path):
    server = build(tmp_path / "agents.db", namespace="coder")
    templates = await server.list_resource_templates()
    assert [t.uri_template for t in templates] == ["rmbr://examples/{pattern}"]

    resources = await server.list_resources()
    assert [str(r.uri) for r in resources] == ["rmbr://examples"]


@pytest.mark.anyio
async def test_examples_index_lists_every_pattern(tmp_path):
    from rmbr.mcp_server import _EXAMPLES

    server = build(tmp_path / "agents.db", namespace="coder")
    [content] = await server.read_resource("rmbr://examples")
    for pattern in _EXAMPLES:
        assert pattern in content.content


@pytest.mark.anyio
async def test_examples_template_returns_runnable_snippet(tmp_path):
    server = build(tmp_path / "agents.db", namespace="coder")
    [content] = await server.read_resource("rmbr://examples/basic-memory")
    assert "mem.remember(" in content.content
    assert "mem.recall(" in content.content


@pytest.mark.anyio
async def test_examples_template_unknown_pattern_is_helpful_not_an_error(tmp_path):
    server = build(tmp_path / "agents.db", namespace="coder")
    [content] = await server.read_resource("rmbr://examples/not-a-real-pattern")
    assert "Unknown pattern" in content.content
    assert "basic-memory" in content.content  # names a real, valid option
