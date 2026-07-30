"""Tool-calling export — the same tool definitions `serve_mcp()` uses,
factored out so they work in any agent loop, not just MCP.

Most real agent loops aren't MCP: they're a hand-rolled loop around
OpenAI's or Anthropic's tool-calling API, or a framework that wants a
plain JSON Schema. `Index.as_tool()` / `Memory.as_tools()` return
`ToolSpec` objects — a name, description, and JSON Schema (the three
fields every tool-calling API wants), plus a plain callable to dispatch
an actual call. rmbr still never calls a model itself; this only
describes rmbr's own capabilities in a shape a model can be told about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    """One callable tool: enough to both advertise it to a model and run it."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the tool's arguments
    handler: Callable[..., Any]

    def call(self, **kwargs: Any) -> Any:
        """Invoke the underlying rmbr call directly, e.g. with a tool-use block's parsed arguments."""
        return self.handler(**kwargs)

    def to_openai(self) -> dict[str, Any]:
        """OpenAI (Chat Completions and Responses API-compatible) function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        """Anthropic Messages API tool-use definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def hit_to_dict(hit: Any) -> dict[str, Any]:
    return {
        "id": hit.id,
        "text": hit.text,
        "score": hit.score,
        "metadata": hit.metadata,
        "bm25_score": hit.bm25_score,
        "vector_score": hit.vector_score,
    }


# Shared by index_search_tool() and memory_tools()'s recall tool - both
# ultimately call hybrid_search() with the same knobs. `where`'s schema is
# deliberately permissive (no fixed `properties`) since its shape is
# dynamic - either plain equality values or {"$gt": ...}-style operator
# dicts - describing that fully as JSON Schema isn't worth the complexity
# a tool-calling model needs to reason about; the description carries the
# actual contract instead.
_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query"},
        "k": {"type": "integer", "description": "Number of results to return", "default": 5},
        "where": {
            "type": "object",
            "description": (
                'Filter by metadata. A plain value is equality (e.g. {"category": "docs"}); '
                'an operator dict narrows further (e.g. {"price": {"$gt": 10}}) - '
                "$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin are supported. Omit for no filtering."
            ),
            "additionalProperties": True,
        },
        "min_similarity": {
            "type": "number",
            "description": (
                "Drop results below this raw cosine similarity (0-1) - a real confidence "
                "gate, unlike the returned score itself. Omit for no filtering."
            ),
        },
        "rerank": {
            "type": "boolean",
            "description": "Re-score results with a local cross-encoder for higher precision at extra latency.",
            "default": False,
        },
    },
    "required": ["query"],
}

_REMEMBER_PARAMETERS = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "The note to remember"}},
    "required": ["text"],
}


def index_search_tool(index: Any, *, name: str = "search") -> ToolSpec:
    """A `ToolSpec` for `index.search()`. See `Index.as_tool()`."""

    def handler(
        query: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
        min_similarity: float | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        hits = index.search(query, k=k, where=where, min_similarity=min_similarity, rerank=rerank)
        return [hit_to_dict(h) for h in hits]

    return ToolSpec(
        name=name,
        description="Search indexed documents by keyword and meaning.",
        parameters=_SEARCH_PARAMETERS,
        handler=handler,
    )


def memory_tools(memory: Any, *, read_only: bool = False) -> list[ToolSpec]:
    """`ToolSpec`s for `memory.recall()` (and `memory.remember()`, unless `read_only`). See `Memory.as_tools()`."""

    def recall_handler(
        query: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
        min_similarity: float | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        hits = memory.recall(query, k=k, where=where, min_similarity=min_similarity, rerank=rerank)
        return [hit_to_dict(h) for h in hits]

    tools = [
        ToolSpec(
            name="recall",
            description="Recall your own remembered notes by keyword and meaning.",
            parameters=_SEARCH_PARAMETERS,
            handler=recall_handler,
        )
    ]
    if not read_only:

        def remember_handler(text: str) -> int:
            return memory.remember(text)

        tools.append(
            ToolSpec(
                name="remember",
                description="Save a note to memory. Returns the new memory's id.",
                parameters=_REMEMBER_PARAMETERS,
                handler=remember_handler,
            )
        )
    return tools
