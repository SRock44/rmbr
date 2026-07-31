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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolCallError(TypeError):
    """Raised by `ToolSpec.call()` when the given arguments don't match the
    tool's schema - typically because the calling model hallucinated an
    argument. A `TypeError` subclass so existing `except TypeError` handlers
    still catch it, but with a message aimed at being fed straight back to
    the model as a tool-result error rather than a Python traceback.
    """


@dataclass
class ToolSpec:
    """One callable tool: enough to both advertise it to a model and run it."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the tool's arguments
    handler: Callable[..., Any]

    def call(self, **kwargs: Any) -> Any:
        """Invoke the underlying rmbr call directly, e.g. with a tool-use block's parsed arguments.

        Validates `kwargs` against `self.parameters` first - a model can
        hallucinate an argument that isn't in the schema (or omit a
        required one), and without this check that surfaces as a bare
        Python `TypeError` from argument binding deep inside the handler,
        indistinguishable from a real bug. Here it's a `ToolCallError`
        with a message that names the actual problem, safe to return as a
        tool-result error and let the model retry with corrected input.
        """
        properties = self.parameters.get("properties", {})
        if not self.parameters.get("additionalProperties", False):
            unexpected = sorted(set(kwargs) - set(properties))
            if unexpected:
                valid = ", ".join(sorted(properties)) or "(none)"
                raise ToolCallError(
                    f"{self.name}: unexpected argument(s) {unexpected} - valid arguments are: {valid}"
                )
        missing = [r for r in self.parameters.get("required", []) if r not in kwargs]
        if missing:
            raise ToolCallError(f"{self.name}: missing required argument(s) {missing}")
        return self.handler(**kwargs)

    def to_openai(self, *, strict: bool = False) -> dict[str, Any]:
        """OpenAI (Chat Completions and Responses API-compatible) function-calling tool definition.

        `strict=True` adds OpenAI's `strict` field, asking the provider to
        reject a malformed call before it ever reaches you - a complement
        to, not a replacement for, the validation `call()` already does
        (not every OpenAI-compatible provider enforces `strict` as tightly
        as OpenAI itself; `call()` catches what slips through either way).
        """
        function: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if strict:
            function["strict"] = True
        return {"type": "function", "function": function}

    def to_anthropic(self, *, strict: bool = False) -> dict[str, Any]:
        """Anthropic Messages API tool-use definition.

        `strict=True` adds Anthropic's top-level `strict` field, which
        guarantees `tool_use.input` validates exactly against the schema
        before you ever see it. Requires `additionalProperties: false` on
        the schema, which every built-in rmbr tool schema already sets.
        """
        tool: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
        if strict:
            tool["strict"] = True
        return tool


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
    "additionalProperties": False,
}

_REMEMBER_PARAMETERS = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The note to remember"},
        "pinned": {
            "type": "boolean",
            "description": (
                "Protect this memory from automatic eviction under max_memories, "
                "even once it's no longer among the most recent. Use for facts "
                "that stay important regardless of age. Default: not pinned."
            ),
            "default": False,
        },
    },
    "required": ["text"],
    "additionalProperties": False,
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

        def remember_handler(text: str, pinned: bool = False) -> int:
            return memory.remember(text, pinned=pinned)

        tools.append(
            ToolSpec(
                name="remember",
                description=(
                    "Save a note to memory. Returns the new memory's id. Set pinned=true "
                    "for a fact that should never be automatically evicted for being old."
                ),
                parameters=_REMEMBER_PARAMETERS,
                handler=remember_handler,
            )
        )
    return tools
