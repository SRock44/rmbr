"""The headline API: give an agent durable, namespaced memory.

    from rmbr import Memory

    mem = Memory("agents.db", namespace="researcher")
    mem.remember("user prefers dark mode and short answers")
    mem.recall("user preferences")

A memory is a free-text note an agent chooses to keep — a preference it
learned, a fact worth not re-deriving, a decision it made. That's
different from indexing a pile of documents for RAG (see `Index` in
index.py); both live in the same `.db` file and share the same
embedder/search machinery, so an app that needs both just opens the same
path twice.
"""

from __future__ import annotations

from typing import Any

from ._engine import (
    load_ann_index,
    make_embedder,
    resolve_readable_namespaces,
    resolve_writable_namespace,
    save_ann_index,
)
from .ann import AnnIndex
from .embed import Embedder
from .policy import Policy
from .search import Hits, hybrid_search
from .store import MemoryRecord, Store

_COLLECTION = "memories"


class Memory:
    """A namespace-scoped handle onto one `.db` file's `memories` table."""

    def __init__(
        self,
        path: str,
        namespace: str,
        *,
        policy: Policy | None = None,
        embedder: Embedder | None = None,
    ):
        self.namespace = namespace
        self.policy = policy or Policy.strict()
        self._store = Store(path)
        self._embedder = make_embedder(embedder, self._store)
        self._ann = load_ann_index(self._store, _COLLECTION)

    def remember(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> int:
        """Save a memory. Returns its id (pass it to `forget()` later if needed)."""
        target_namespace = resolve_writable_namespace(self.policy, self.namespace, namespace)
        memory_id = self._store.insert_memory(target_namespace, text, metadata)

        vector = self._embedder.embed_one(text)
        if self._ann is None:
            self._ann = AnnIndex(dim=len(vector))
        self._ann.add([memory_id], [vector])
        save_ann_index(self._store, _COLLECTION, self._ann)
        self._store.clear_query_cache()

        return memory_id

    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        namespaces: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        use_bm25: bool = True,
        use_vectors: bool = True,
        budget_ms: float | None = None,
    ) -> Hits:
        """Search memories by meaning and keyword. Returns Hits — hits[0].text, hits[0].score, hits.timings.

        Hybrid (both signals, the default) is what you want for real
        queries. Set `use_bm25=False` for pure semantic search, or
        `use_vectors=False` for pure keyword search.

        `where` filters to memories whose metadata matches every
        key/value given — equality only in v0.1, no operators.
        """
        readable = resolve_readable_namespaces(
            self.policy, self.namespace, namespaces, self._store.list_memory_namespaces
        )
        return hybrid_search(
            query=query,
            namespaces=readable,
            k=k,
            fts_search=self._store.search_memories_fts,
            fetch_records=self._store.get_memories,
            ann_index=self._ann,
            embedder=self._embedder,
            use_bm25=use_bm25,
            use_vectors=use_vectors,
            where=where,
            budget_ms=budget_ms,
            query_cache=self._store,
        )

    def forget(self, memory_id: int) -> None:
        """Delete a memory by id. No-op if it doesn't exist.

        Only deletes within namespaces this handle can write to — a
        `Memory("agents.db", namespace="coder")` can't forget a
        researcher's memory just by guessing its id, even though ids are
        shared across the whole file.
        """
        record = self._store.get_memory(memory_id)
        if record is None:
            return
        if record.namespace != self.namespace and not self.policy.can_write(
            self.namespace, record.namespace
        ):
            raise PermissionError(
                f"{self.namespace!r} is not allowed to delete from namespace {record.namespace!r}"
            )
        self._store.delete_memory(memory_id)
        if self._ann is not None:
            self._ann.remove([memory_id])
            save_ann_index(self._store, _COLLECTION, self._ann)
        self._store.clear_query_cache()

    def list(self, *, limit: int | None = None) -> list[MemoryRecord]:
        """List this namespace's memories, most recent first."""
        return self._store.list_memories(self.namespace, limit=limit)

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
