"""LangGraph `BaseStore` adapter — thin, lazily-imported, no hard dependency.

Wraps rmbr as a `langgraph.store.base.BaseStore` so it drops into
`StateGraph(...).compile(store=...)` or the Functional API's long-term
memory, giving LangGraph the same hybrid search rmbr already does — no
Postgres/Redis store to stand up.

Verified against `langgraph-checkpoint==4.1.1`'s `BaseStore`: `batch()`/
`abatch()` are its *only* abstract methods — `get`/`put`/`delete`/`search`/
`list_namespaces` (sync and async) are concrete methods built on top of
those two, so implementing `batch`/`abatch` here is enough to satisfy the
whole interface. Ops dispatched: `GetOp(namespace, key, refresh_ttl)`,
`PutOp(namespace, key, value, index, ttl)`, `SearchOp(namespace_prefix,
filter, limit, offset, query, refresh_ttl)`, `ListNamespacesOp
(match_conditions, max_depth, limit, offset)` — returning `Item | None`,
`None`, `list[SearchItem]`, `list[tuple[str, ...]]` respectively. A
`PutOp` with `value=None` is `BaseStore`'s own delete signal.

**Namespace mapping:** a LangGraph namespace is a tuple of strings (e.g.
`("memories", "user-42")`); rmbr namespaces are one flat string. Each
distinct tuple maps 1:1 to an rmbr namespace by joining with
`namespace_separator` (default `"."`) — pick a separator that can't occur
inside one of your own namespace segments. `RmbrStore` opens one internal
`Memory` per distinct tuple it sees, lazily, all sharing the same `.db`
file, policy, and embedder.

**Key mapping:** LangGraph's `key` is caller-chosen and unique per
namespace; rmbr's `remember()` generates its own integer id. This adapter
stores `key` under `metadata["_lg_key"]` and looks records up by it, so
`put()` on an existing key overwrites (forget, then re-remember) rather
than accumulating duplicates — matching `BaseStore`'s upsert contract.

**What's searchable:** LangGraph's `value` is an arbitrary JSON dict, not
text. Every `put()` embeds/indexes `json.dumps(value, sort_keys=True)` in
full, and flattens `value`'s own top-level keys into metadata so
`SearchOp.filter` maps directly onto rmbr's `where=` operators
(`$gt`/`$lt`/`$in`/...). `PutOp.index` (LangGraph's field-path-selective
indexing) is **not** honored — everything in `value` is always
searchable, nothing is field-scoped. This is a deliberate v1
simplification, not an oversight.

**Not supported:** per-item TTL (`PutOp.ttl` / `refresh_ttl`) is accepted
but silently has no effect — rmbr has no per-item expiry, only
`Memory.forget_older_than()` as a manual/periodic sweep. If your graph
relies on LangGraph's TTL semantics, this adapter isn't a drop-in for
that part yet.

    from rmbr.integrations.langgraph import as_store

    store = as_store("agents.db")
    graph = builder.compile(store=store)

No `Memory.as_langgraph_store()` convenience method exists (unlike the
LangChain/LlamaIndex adapters) because a LangGraph store isn't scoped to
one namespace the way a `Memory` handle is — it spans however many
namespaces the graph throws at it, so it's constructed from a raw path
instead. Defaults to `Policy.open()` since `BaseStore` itself has no
per-namespace ACL concept — pass your own `Policy` if you want rmbr's
namespace boundary enforced underneath anyway.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable

from ..embed import Embedder
from ..memory import Memory
from ..policy import Policy
from ..store import MemoryRecord, Store

if TYPE_CHECKING:
    from langgraph.store.base import GetOp, ListNamespacesOp, Op, PutOp, Result, SearchOp

_DEFAULT_SEPARATOR = "."
_RESERVED_PREFIX = "_lg_"


def as_store(
    path: str,
    *,
    policy: Policy | None = None,
    embedder: Embedder | None = None,
    namespace_separator: str = _DEFAULT_SEPARATOR,
) -> Any:
    """Build a `langgraph.store.base.BaseStore` backed by one rmbr `.db` file.

    See module docstring for the namespace/key/search mapping and what
    isn't supported (per-item TTL, field-path-selective indexing).
    """
    from langgraph.store.base import BaseStore, GetOp, ListNamespacesOp, PutOp, SearchOp

    class RmbrStore(BaseStore):
        def __init__(self) -> None:
            self._path = path
            self._policy = policy or Policy.open()
            self._embedder = embedder
            self._separator = namespace_separator
            self._raw_store = Store(path)
            self._memories: dict[str, Memory] = {}

        def _ns_string(self, namespace: tuple[str, ...]) -> str:
            return self._separator.join(namespace)

        def _ns_tuple(self, ns_string: str) -> tuple[str, ...]:
            return tuple(ns_string.split(self._separator))

        def _memory_for(self, namespace: tuple[str, ...]) -> Memory:
            ns_string = self._ns_string(namespace)
            memory = self._memories.get(ns_string)
            if memory is None:
                memory = Memory(self._path, ns_string, policy=self._policy, embedder=self._embedder)
                self._memories[ns_string] = memory
            return memory

        def _known_namespaces(self) -> list[tuple[str, ...]]:
            return [self._ns_tuple(ns) for ns in self._raw_store.list_memory_namespaces()]

        def batch(self, ops: Iterable["Op"]) -> list["Result"]:
            return [self._run_op(op) for op in ops]

        async def abatch(self, ops: Iterable["Op"]) -> list["Result"]:
            ops = list(ops)
            return await asyncio.to_thread(self.batch, ops)

        def _run_op(self, op: "Op") -> Any:
            if isinstance(op, GetOp):
                return self._get(op)
            if isinstance(op, SearchOp):
                return self._search(op)
            if isinstance(op, PutOp):
                return self._put(op)
            if isinstance(op, ListNamespacesOp):
                return self._list_namespaces(op)
            raise TypeError(f"RmbrStore received an unknown op type: {type(op)!r}")

        def _get(self, op: "GetOp") -> Any:
            from langgraph.store.base import Item

            record = _find_by_key(self._memory_for(op.namespace), op.key)
            if record is None:
                return None
            created = _parse_iso(record.created_at)
            return Item(
                namespace=op.namespace,
                key=op.key,
                value=record.metadata.get("_lg_value", {}),
                created_at=created,
                updated_at=created,
            )

        def _put(self, op: "PutOp") -> None:
            memory = self._memory_for(op.namespace)
            existing = _find_by_key(memory, op.key)
            if op.value is None:  # BaseStore's delete signal
                if existing is not None:
                    memory.forget(existing.id)
                return None
            if existing is not None:
                memory.forget(existing.id)
            metadata: dict[str, Any] = {"_lg_key": op.key, "_lg_value": op.value}
            if isinstance(op.value, dict):
                metadata.update({k: v for k, v in op.value.items() if not str(k).startswith(_RESERVED_PREFIX)})
            text = json.dumps(op.value, sort_keys=True, default=str)
            memory.remember(text, metadata=metadata)
            return None

        def _search(self, op: "SearchOp") -> list[Any]:
            from langgraph.store.base import SearchItem

            prefix = op.namespace_prefix
            candidates = [ns for ns in self._known_namespaces() if ns[: len(prefix)] == prefix]

            scored: list[tuple[float, tuple[str, ...], int]] = []
            for ns_tuple in candidates:
                memory = self._memory_for(ns_tuple)
                if op.query:
                    for hit in memory.recall(op.query, k=op.limit + op.offset, where=op.filter):
                        scored.append((hit.score, ns_tuple, hit.id))
                else:
                    for record in memory.list(where=op.filter):
                        scored.append((0.0, ns_tuple, record.id))

            scored.sort(key=lambda row: row[0], reverse=True)
            page = scored[op.offset : op.offset + op.limit]
            if not page:
                return []

            records_by_id = {r.id: r for r in self._raw_store.get_memories([mid for _, _, mid in page])}
            items = []
            for score, ns_tuple, memory_id in page:
                record = records_by_id.get(memory_id)
                if record is None:
                    continue
                created = _parse_iso(record.created_at)
                items.append(
                    SearchItem(
                        namespace=ns_tuple,
                        key=record.metadata.get("_lg_key", str(record.id)),
                        value=record.metadata.get("_lg_value", {}),
                        created_at=created,
                        updated_at=created,
                        score=score or None,
                    )
                )
            return items

        def _list_namespaces(self, op: "ListNamespacesOp") -> list[tuple[str, ...]]:
            namespaces: Iterable[tuple[str, ...]] = self._known_namespaces()
            if op.match_conditions:
                namespaces = [
                    ns for ns in namespaces if all(_matches_condition(ns, c) for c in op.match_conditions)
                ]
            if op.max_depth is not None:
                namespaces = {ns[: op.max_depth] for ns in namespaces}
            namespaces = sorted(set(namespaces))
            return namespaces[op.offset : op.offset + op.limit]

        def close(self) -> None:
            for memory in self._memories.values():
                memory.close()
            self._raw_store.close()

    return RmbrStore()


def _find_by_key(memory: Memory, key: str) -> MemoryRecord | None:
    matches = memory.list(where={"_lg_key": key})
    return matches[0] if matches else None


def _matches_condition(namespace: tuple[str, ...], condition: Any) -> bool:
    path = condition.path
    if len(path) > len(namespace):
        return False
    segment = namespace[-len(path) :] if condition.match_type == "suffix" else namespace[: len(path)]
    return all(p == "*" or p == s for p, s in zip(path, segment))


def _parse_iso(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp)
