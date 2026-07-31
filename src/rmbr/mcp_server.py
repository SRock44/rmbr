"""Expose a namespace-pinned Memory + Index over MCP (stdio) for external agents.

    from rmbr import serve_mcp
    serve_mcp("agents.db", namespace="coder", read_only=True)

This is the only place rmbr talks to the outside world, and only because
an external MCP client (Claude Desktop, another agent's tool-launcher)
needs a protocol to call into your process — rmbr itself never calls an
LLM or makes a network request on its own.

**Namespace-pinned:** the tools below take no `namespace` parameter. An
MCP client can search/remember/recall within the namespace pinned at
startup and nothing else — there's no field for it to fill in even if it
wanted to reach into another agent's memory. Enforcement lives in what
the tool schema exposes, not in trusting the model to behave.
"""

from __future__ import annotations

from typing import Any

from .embed import Embedder
from .index import Index
from .memory import Memory
from .policy import Policy
from .tools import hit_to_dict


def serve_mcp(
    path: str,
    *,
    namespace: str = "default",
    policy: Policy | None = None,
    embedder: Embedder | None = None,
    read_only: bool = False,
    server_name: str = "rmbr",
) -> None:
    """Start an MCP (stdio) server exposing `search`, `recall`, and (unless
    `read_only`) `remember` tools, all scoped to `namespace`.

    Blocks until the client disconnects. This is meant to be your
    process's entire job — see `__main__.py` / `python -m rmbr`, which is
    how an MCP client actually launches this — not something you call
    from inside an app that also wants to keep doing other work.
    """
    build_mcp_server(
        path, namespace=namespace, policy=policy, embedder=embedder, read_only=read_only, server_name=server_name
    ).run(transport="stdio")


def build_mcp_server(
    path: str,
    *,
    namespace: str = "default",
    policy: Policy | None = None,
    embedder: Embedder | None = None,
    read_only: bool = False,
    server_name: str = "rmbr",
) -> Any:
    """Build the configured MCPServer without running it.

    Split out from `serve_mcp` so it can be exercised in-process — by our
    own test suite, or by anyone embedding rmbr's MCP server inside a
    larger process that manages its own event loop instead of calling the
    blocking `serve_mcp()`.
    """
    from mcp.server.mcpserver import MCPServer

    server: Any = MCPServer(name=server_name)
    idx = Index(path, namespace=namespace, policy=policy, embedder=embedder)
    mem = Memory(path, namespace, policy=policy, embedder=embedder)

    @server.tool()
    def search(query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search this agent's indexed documents by keyword and meaning."""
        return [hit_to_dict(h) for h in idx.search(query, k=k)]

    @server.tool()
    def recall(query: str, k: int = 5) -> list[dict[str, Any]]:
        """Recall this agent's own remembered notes by keyword and meaning."""
        return [hit_to_dict(h) for h in mem.recall(query, k=k)]

    if not read_only:

        @server.tool()
        def remember(text: str) -> int:
            """Save a note to this agent's memory. Returns the new memory's id."""
            return mem.remember(text)

    return server
