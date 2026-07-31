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
