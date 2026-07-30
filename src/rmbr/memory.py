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

import asyncio
import time
from datetime import datetime, timezone
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
from .rerank import CrossEncoderReranker
from .search import DEFAULT_RECENCY_HALF_LIFE_SECONDS, Hits, hybrid_search
from .store import MemoryRecord, Store

_COLLECTION = "memories"
_UNSET = object()  # distinguishes "not passed" from "explicitly None" for dedupe_threshold


class Memory:
    """A namespace-scoped handle onto one `.db` file's `memories` table."""

    def __init__(
        self,
        path: str,
        namespace: str,
        *,
        policy: Policy | None = None,
        embedder: Embedder | None = None,
        dedupe_threshold: float | None = None,
        max_memories: int | None = None,
    ):
        self.namespace = namespace
        self.policy = policy or Policy.strict()
        self._store = Store(path)
        self._embedder = make_embedder(embedder, self._store)
        self._ann = load_ann_index(self._store, _COLLECTION)
        self._alock = asyncio.Lock()
        self._default_dedupe_threshold = dedupe_threshold
        self._max_memories = max_memories
        self._reranker: CrossEncoderReranker | None = None

    def remember(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
        dedupe_threshold: float | None = _UNSET,  # type: ignore[assignment]
    ) -> int:
        """Save a memory. Returns its id (pass it to `forget()` later if needed).

        By default every call inserts a new row — `remember()` never
        judges whether two similar-sounding memories are actually the
        same fact, so it doesn't guess. Pass `dedupe_threshold` (here, or
        once at `Memory(...)` construction to apply it to every call) to
        opt in: if an existing memory in the target namespace has cosine
        similarity >= threshold against this text, that memory is
        updated in place (new text, new metadata, bumped timestamp, same
        id) instead of inserting a duplicate. This trades "near-duplicate
        memories can pile up" for "a false-positive match silently
        discards the old memory" — only you know which risk is worse for
        your agent, which is why it's opt-in rather than a default. A
        good starting point is 0.92-0.95; there's no LLM here to judge
        intent, only vector similarity, so keep it conservative.
        """
        target_namespace = resolve_writable_namespace(self.policy, self.namespace, namespace)
        threshold = self._default_dedupe_threshold if dedupe_threshold is _UNSET else dedupe_threshold
        vector = self._embedder.embed_one(text)

        with self._store.transaction():
            existing_id = None
            if threshold is not None and self._ann is not None and len(self._ann) > 0:
                existing_id = self._find_similar_memory(vector, target_namespace, threshold)

            if existing_id is not None:
                self._store.update_memory(existing_id, text, metadata)
                self._ann.replace([existing_id], [vector])
                memory_id = existing_id
            else:
                memory_id = self._store.insert_memory(target_namespace, text, metadata)
                if self._ann is None:
                    self._ann = AnnIndex(dim=len(vector))
                self._ann.add([memory_id], [vector])

            if self._max_memories is not None:
                evicted = self._store.delete_oldest_memories_beyond(target_namespace, self._max_memories)
                if evicted:
                    self._ann.remove(evicted)

            save_ann_index(self._store, _COLLECTION, self._ann)
            self._store.clear_query_cache()

        return memory_id

    def _find_similar_memory(self, vector: Any, namespace: str, threshold: float) -> int | None:
        """Best existing same-namespace match at or above `threshold`, or None.

        The ANN index is shared across every namespace in the file (ids,
        not a per-namespace index), so the nearest neighbor overall might
        belong to a different namespace — oversample and post-filter,
        same pattern `hybrid_search` uses for namespace filtering.
        Matches come back best-first, so the first candidate below
        threshold means nothing further down qualifies either.
        """
        for candidate_id, similarity in self._ann.search(vector, k=10):
            if similarity < threshold:
                break
            record = self._store.get_memory(candidate_id)
            if record is not None and record.namespace == namespace:
                return candidate_id
        return None

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
        min_similarity: float | None = None,
        recency_weight: float = 0.0,
        recency_half_life_seconds: float = DEFAULT_RECENCY_HALF_LIFE_SECONDS,
        rerank: bool = False,
    ) -> Hits:
        """Search memories by meaning and keyword. Returns Hits — hits[0].text, hits[0].score, hits.timings.

        Hybrid (both signals, the default) is what you want for real
        queries. Set `use_bm25=False` for pure semantic search, or
        `use_vectors=False` for pure keyword search.

        `where` filters to memories whose metadata matches every
        key/value given: a plain value is equality, a `{"$gt": ...}`-style
        dict is an operator comparison — see `search.hybrid_search`'s
        docstring for the full operator set.

        `min_similarity` drops hits below that raw cosine similarity
        (requires `use_vectors=True`) — a real confidence gate, unlike
        thresholding on `hit.score` itself (see `search.hybrid_search`).

        `recency_weight` (0.0, off, by default) blends a recency bonus
        into ranking so a fresher memory can outrank an equally-relevant
        older one — `recency_half_life_seconds` controls how fast that
        bonus decays (default: a week).

        `rerank=True` re-scores the candidate pool with a local
        cross-encoder (`rerank.CrossEncoderReranker`, lazily loaded on
        first use) for higher precision at extra latency; `hit.score`
        becomes the cross-encoder's score when this is on.
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
            min_similarity=min_similarity,
            recency_weight=recency_weight,
            recency_half_life_seconds=recency_half_life_seconds,
            record_timestamp=(lambda record: record.created_at) if recency_weight else None,
            reranker=self._get_reranker() if rerank else None,
        )

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

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
        with self._store.transaction():
            self._store.delete_memory(memory_id)
            if self._ann is not None:
                self._ann.remove([memory_id])
                save_ann_index(self._store, _COLLECTION, self._ann)
            self._store.clear_query_cache()

    def forget_older_than(self, seconds: float, *, namespace: str | None = None) -> int:
        """Delete memories older than `seconds` ago. Returns how many were deleted.

        Nothing in `remember()` expires memories on its own — a
        long-running agent accumulates them forever unless something
        prunes. This is the deliberate, ad-hoc way to do that (call it
        periodically, or on startup); `max_memories` at `Memory(...)`
        construction is the automatic, always-on alternative if you'd
        rather cap by count than by age.
        """
        target_namespace = resolve_writable_namespace(self.policy, self.namespace, namespace)
        cutoff = _iso(time.time() - seconds)
        with self._store.transaction():
            deleted_ids = self._store.delete_memories_before(target_namespace, cutoff)
            if deleted_ids and self._ann is not None:
                self._ann.remove(deleted_ids)
                save_ann_index(self._store, _COLLECTION, self._ann)
            self._store.clear_query_cache()
        return len(deleted_ids)

    def list(self, *, limit: int | None = None) -> list[MemoryRecord]:
        """List this namespace's memories, most recent first."""
        return self._store.list_memories(self.namespace, limit=limit)

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- async surface -----------------------------------------------------
    #
    # Thin wrappers around asyncio.to_thread, guarded by a per-instance
    # lock — see Index's async surface in index.py for the full rationale
    # (usearch isn't documented as safe for concurrent mutation from
    # multiple threads, so every async call on one Memory instance is
    # serialized rather than risking that). Open separate Memory instances
    # for true concurrent access.

    async def aremember(self, text: str, **kwargs: Any) -> int:
        async with self._alock:
            return await asyncio.to_thread(self.remember, text, **kwargs)

    async def arecall(self, query: str, **kwargs: Any) -> Hits:
        async with self._alock:
            return await asyncio.to_thread(self.recall, query, **kwargs)

    async def aforget(self, memory_id: int) -> None:
        async with self._alock:
            return await asyncio.to_thread(self.forget, memory_id)

    async def aforget_older_than(self, seconds: float, **kwargs: Any) -> int:
        async with self._alock:
            return await asyncio.to_thread(self.forget_older_than, seconds, **kwargs)


def _iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()
