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

Also exposes an MCP resource template (`rmbr://examples/{pattern}`, plus
`rmbr://examples` as an index of valid `pattern` values) with short,
runnable code examples for common usage patterns — so a connected client
can browse "how do I do X with rmbr" without leaving the MCP session.
"""

from __future__ import annotations

from typing import Any

from . import __version__
from .embed import Embedder
from .index import Index
from .memory import Memory
from .policy import Policy
from .tools import hit_to_dict

# Short, runnable examples for common rmbr usage patterns, exposed as an MCP
# resource template (rmbr://examples/{pattern}) so a connected client can
# browse "how do I do X" without leaving the MCP session. Kept in sync with
# the corresponding README sections - these are excerpts, not a substitute
# for it.
_EXAMPLES: dict[str, str] = {
    "basic-memory": """\
from rmbr import Memory

mem = Memory("agents.db", namespace="assistant")
mem.remember("user prefers dark mode and short answers")
mem.recall("user preferences")
""",
    "document-search": """\
from rmbr import Index

idx = Index("agents.db")
idx.add_files("docs/")                     # .py, .md, and plain text each get an appropriate splitter
hits = idx.search("how do I deploy?", k=5)
hits[0].text, hits[0].score, hits.timings
""",
    "multi-agent-policy": """\
from rmbr import Memory, Policy

policy = Policy()
policy.allow("supervisor", read="*")  # supervisor can read every namespace

mem = Memory("agents.db", namespace="coder", policy=policy)
# coder can only read/write its own namespace unless explicitly granted -
# deny-by-default, enforced on every call, no LLM involved in the decision.
""",
    "conversation-memory": """\
mem.remember_turn("user", "I prefer dark mode")
mem.remember_turn("assistant", "Got it, dark mode from now on", session_id="conv-42")

mem.recall("dark mode", where={"role": "user"})       # who said it
mem.list(where={"session_id": "conv-42"})              # replay one conversation, in order
""",
    "tool-calling": """\
# Raw OpenAI/Anthropic tool-calling - one line to get a ready-made tool
# definition plus a callable, in either API's shape:
tool = idx.as_tool()
response = client.messages.create(..., tools=[tool.to_anthropic()])
result = tool.call(**tool_use_block.input)          # dispatches to idx.search()

recall_tool, remember_tool = mem.as_tools()          # or as_tools(read_only=True) for recall only
""",
    "memory-hygiene": """\
# Update-in-place instead of appending, above a cosine-similarity threshold.
mem = Memory("agents.db", namespace="assistant", dedupe_threshold=0.93)
mem.remember("user prefers dark mode")         # inserts
mem.remember("user really prefers dark mode")  # updates the same row if similarity clears the bar

# Bound growth automatically; pin specific memories to exempt them from eviction:
mem = Memory("agents.db", namespace="assistant", max_memories=5000)
mem.remember("the customer's account was permanently deactivated", pinned=True)

mem.stats()             # {"assistant": {"count": 412, "oldest": "...", "newest": "..."}}
mem.integrity_check()   # [] if healthy; otherwise, what's wrong and which ids
""",
}


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

    server: Any = MCPServer(name=server_name, version=__version__)
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
        def remember(text: str, pinned: bool = False) -> int:
            """Save a note to this agent's memory. Returns the new memory's id.
            Set pinned=true for a fact that should never be automatically
            evicted for being old."""
            return mem.remember(text, pinned=pinned)

    @server.resource("rmbr://examples")
    def examples_index() -> str:
        """List the available rmbr usage-pattern examples (see rmbr://examples/{pattern})."""
        lines = "\n".join(f"- {name}" for name in _EXAMPLES)
        return f"Usage patterns available at rmbr://examples/{{pattern}}:\n{lines}"

    @server.resource("rmbr://examples/{pattern}")
    def example(pattern: str) -> str:
        """A short, runnable code example for one common rmbr usage pattern."""
        if pattern not in _EXAMPLES:
            return f"Unknown pattern {pattern!r}. Available: {', '.join(_EXAMPLES)}"
        return _EXAMPLES[pattern]

    return server
