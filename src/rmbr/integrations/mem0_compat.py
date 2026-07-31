"""A `Memory` class shaped like mem0 OSS's local `Memory`, backed by rmbr —
for switching without rewriting every `add()`/`search()`/`get_all()` call.

This does **not** wrap or import `mem0` — it's a from-scratch
reimplementation of mem0's local (`from mem0 import Memory`) call surface
against rmbr's real storage/search engine. No `mem0ai` dependency, not
even optional.

Verified against `mem0ai==2.0.14`'s local `Memory` (`mem0.memory.main`):
`add()`/`search()`/`get_all()`/`get()`/`update()`/`delete()`/`delete_all()`
signatures and return shapes (`{"results": [...]}` / `{"message": "..."}` /
a single dict or `None`) are matched field-for-field where rmbr has an
equivalent concept.

**The one thing you can't just import-swap:** mem0's `Memory()` takes an
optional config and defaults to a hosted/local vector store path baked
into its `MemoryConfig`; rmbr writes to a file you name. So construction
changes from `Memory()` to `Memory("agents.db")` — everything after that
is the same call shape.

**The one behavior that's a hard no, by design:** mem0's real default is
`infer=True` — an LLM reads your messages and decides what facts to keep.
rmbr never calls an LLM (see the top-level README) and this shim honors
that: `add(..., infer=True)` (or leaving `infer` unset, since that's
mem0's real default too) raises `NotImplementedError` naming exactly
what's not happening, rather than silently storing raw messages under an
argument that claimed something smarter was going on. Pass `infer=False`
explicitly to store `messages` as-is — the same thing rmbr's own
`Memory.remember()`/`remember_turn()` always did anyway.

**Scope mapping:** mem0 scopes memories by `user_id`/`agent_id`/`run_id`
(at least one required, same validation rmbr enforces here); rmbr scopes
by one flat namespace string. The id(s) given to a call are joined
(`"user_id:42|agent_id:coder"`) into one rmbr namespace — every call
touching the same combination of ids lands in the same namespace, exactly
like mem0's own compound scoping.

**Filter translation:** mem0's `filters={"key": {"gt": 10}}` operator
dicts are translated to rmbr's `where={"key": {"$gt": 10}}` for the
operators rmbr actually has (`eq`/`ne`/`gt`/`gte`/`lt`/`lte`/`in`/`nin`).
mem0's `AND`/`OR`/`NOT` logical combinators, `contains`/`icontains`
substring operators, and `"*"` wildcard values have no rmbr equivalent —
rather than silently drop them and return more/fewer results than
expected, these raise `NotImplementedError` naming the unsupported piece.

**Not implemented at all:** `history()` (rmbr keeps no change log),
`memory_type="procedural_memory"`, vision messages, and mem0's own
reranker (rmbr has its own local cross-encoder reranker — `rerank=True`
here routes to `Memory.recall(rerank=True)` instead, not mem0's).

    from rmbr.integrations.mem0_compat import Memory

    m = Memory("agents.db")
    m.add("user prefers dark mode", user_id="alex", infer=False)
    m.search("dark mode", filters={"user_id": "alex"})
"""

from __future__ import annotations

from typing import Any

from ..embed import Embedder
from ..memory import Memory as RmbrMemory
from ..policy import Policy
from ..store import MemoryRecord, Store

_SCOPE_KEYS = ("user_id", "agent_id", "run_id")
_OPS = {"eq": "$eq", "ne": "$ne", "gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte", "in": "$in", "nin": "$nin"}


class Memory:
    """mem0-OSS-shaped `Memory`, backed by an rmbr `.db` file. See module docstring."""

    def __init__(
        self,
        path: str,
        *,
        policy: Policy | None = None,
        embedder: Embedder | None = None,
        dedupe_threshold: float | None = None,
        max_memories: int | None = None,
    ) -> None:
        self._path = path
        self._policy = policy or Policy.open()
        self._embedder = embedder
        self._dedupe_threshold = dedupe_threshold
        self._max_memories = max_memories
        self._memories: dict[str, RmbrMemory] = {}
        self._raw_store = Store(path)

    def _memory_for(self, namespace: str) -> RmbrMemory:
        memory = self._memories.get(namespace)
        if memory is None:
            memory = RmbrMemory(
                self._path,
                namespace,
                policy=self._policy,
                embedder=self._embedder,
                dedupe_threshold=self._dedupe_threshold,
                max_memories=self._max_memories,
            )
            self._memories[namespace] = memory
        return memory

    def _scope(self, user_id: str | None, agent_id: str | None, run_id: str | None) -> tuple[str, dict[str, str]]:
        provided = {
            k: v for k, v in {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}.items() if v is not None
        }
        if not provided:
            raise ValueError("at least one of user_id, agent_id, run_id is required (same as mem0)")
        namespace = "|".join(f"{k}:{v}" for k, v in provided.items())
        return namespace, provided

    def add(
        self,
        messages: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        **_unsupported: Any,
    ) -> dict[str, Any]:
        """Store `messages` as raw memories, one per message. Requires `infer=False`
        (see module docstring for why `infer=True`, mem0's own default, raises)."""
        if infer:
            raise NotImplementedError(
                "rmbr never calls an LLM, so there's no fact-extraction step to run - "
                "pass infer=False explicitly to store messages as-is (see module docstring)."
            )
        namespace, _ = self._scope(user_id, agent_id, run_id)
        memory = self._memory_for(namespace)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]

        results = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role is None or content is None or role == "system":
                continue
            per_message_metadata = {**(metadata or {})}
            memory_id = memory.remember_turn(role, content, metadata=per_message_metadata)
            results.append({"id": str(memory_id), "memory": content, "event": "ADD", "role": role})
        return {"results": results}

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        threshold: float = 0.1,
        rerank: bool = False,
        **_unsupported: Any,
    ) -> dict[str, Any]:
        namespace, scope = self._scope(*_extract_scope(filters))
        memory = self._memory_for(namespace)
        hits = memory.recall(query, k=top_k, where=_translate_filters(filters), min_similarity=threshold, rerank=rerank)
        return {"results": [_hit_to_dict(h, scope) for h in hits]}

    def get(self, memory_id: Any) -> dict[str, Any] | None:
        record = self._raw_store.get_memory(int(memory_id))
        if record is None:
            return None
        return _record_to_dict(record, _scope_from_namespace(record.namespace))

    def get_all(self, *, filters: dict[str, Any] | None = None, top_k: int = 20, **_unsupported: Any) -> dict[str, Any]:
        namespace, scope = self._scope(*_extract_scope(filters))
        memory = self._memory_for(namespace)
        records = memory.list(where=_translate_filters(filters), limit=top_k)
        return {"results": [_record_to_dict(r, scope) for r in records]}

    def update(
        self, memory_id: Any, text: str | None = None, metadata: dict[str, Any] | None = None, **_unsupported: Any
    ) -> dict[str, str]:
        record = self._raw_store.get_memory(int(memory_id))
        if record is None:
            raise ValueError(f"memory {memory_id!r} not found")
        memory = self._memory_for(record.namespace)
        memory.update(record.id, text=text, metadata=metadata)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id: Any) -> dict[str, str]:
        record = self._raw_store.get_memory(int(memory_id))
        if record is not None:
            self._memory_for(record.namespace).forget(record.id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(
        self, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None
    ) -> dict[str, str]:
        namespace, _ = self._scope(user_id, agent_id, run_id)
        memory = self._memory_for(namespace)
        for record in memory.list():
            memory.forget(record.id)
        return {"message": "Memories deleted successfully!"}

    def close(self) -> None:
        for memory in self._memories.values():
            memory.close()
        self._raw_store.close()


def _extract_scope(filters: dict[str, Any] | None) -> tuple[str | None, str | None, str | None]:
    filters = filters or {}
    return filters.get("user_id"), filters.get("agent_id"), filters.get("run_id")


def _translate_filters(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """mem0's `{"key": {"gt": 10}}` -> rmbr's `{"key": {"$gt": 10}}`.

    Raises `NotImplementedError` for mem0 filter features rmbr's `where=`
    has no equivalent for (`AND`/`OR`/`NOT`, `contains`/`icontains`, `"*"`
    wildcards) instead of silently dropping them, which would otherwise
    return more results than the caller asked for without any signal.
    """
    if not filters:
        return None
    where: dict[str, Any] = {}
    for key, value in filters.items():
        if key in _SCOPE_KEYS:
            continue
        if key in ("AND", "OR", "NOT"):
            raise NotImplementedError(
                f"rmbr's mem0 shim doesn't support the {key!r} logical filter combinator "
                "(rmbr's where= is a flat AND of conditions) - flatten your filter manually."
            )
        if value == "*":
            raise NotImplementedError("rmbr's mem0 shim doesn't support '*' wildcard filter values")
        if isinstance(value, dict):
            if len(value) != 1:
                raise NotImplementedError(f"unsupported filter operator shape for {key!r}: {value!r}")
            op, op_value = next(iter(value.items()))
            if op not in _OPS:
                raise NotImplementedError(
                    f"rmbr's mem0 shim doesn't support the {op!r} filter operator "
                    f"(supported: {sorted(_OPS)})"
                )
            where[key] = {_OPS[op]: op_value}
        else:
            where[key] = value
    return where or None


def _scope_from_namespace(namespace: str) -> dict[str, str]:
    scope = {}
    for part in namespace.split("|"):
        key, _, value = part.partition(":")
        if key in _SCOPE_KEYS:
            scope[key] = value
    return scope


def _hit_to_dict(hit: Any, scope: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {"id": str(hit.id), "memory": hit.text, "score": hit.score, **scope}
    metadata = {k: v for k, v in hit.metadata.items() if k != "role"}
    if "role" in hit.metadata:
        result["role"] = hit.metadata["role"]
    if metadata:
        result["metadata"] = metadata
    return result


def _record_to_dict(record: MemoryRecord, scope: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {"id": str(record.id), "memory": record.text, "created_at": record.created_at, **scope}
    metadata = {k: v for k, v in record.metadata.items() if k != "role"}
    if "role" in record.metadata:
        result["role"] = record.metadata["role"]
    if metadata:
        result["metadata"] = metadata
    return result
