"""Real end-to-end MCP test: `python -m rmbr` as an actual subprocess,
talked to over real stdio by the real `mcp` client SDK - not
`build_mcp_server()` called in-process the way tests/test_mcp.py does.

Deliberately kept out of tests/ (see that module's own docstring): this
needs the real default embedder (`fastembed`, downloads its ONNX model on
first use) since `python -m rmbr` has no way to inject `FakeEmbedder` -
there's no CLI flag for it, and there shouldn't be, since rmbr has no CLI
to begin with. That means this test needs network on a cold cache and
takes real wall-clock time to spin up a subprocess and load a model,
neither of which belongs in the fast, always-run suite. Run explicitly:

    pytest tests_integration/ -v
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("mcp.client.stdio")

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402


@pytest.mark.anyio
async def test_python_dash_m_rmbr_serves_real_mcp_over_stdio(tmp_path):
    db_path = tmp_path / "agents.db"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rmbr", str(db_path), "--namespace", "coder"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            assert tool_names == {"search", "recall", "remember"}

            remember_result = await session.call_tool(
                "remember", {"text": "user prefers dark mode and short answers"}
            )
            assert remember_result.is_error is False

            recall_result = await session.call_tool("recall", {"query": "user preferences", "k": 5})
            assert recall_result.is_error is False
            hits = recall_result.structured_content["result"]
            assert len(hits) == 1
            assert "dark mode" in hits[0]["text"]


@pytest.fixture
def anyio_backend():
    return "asyncio"
