"""Hybrid retrieval: BM25 + vector search, merged, timed, and budget-aware.

This module is deliberately generic — it doesn't know about "documents" or
"memories", just "things with an id/text/metadata/namespace, searchable by
full-text and optionally by vector". `Index.search()` and `Memory.recall()`
both call `hybrid_search()` with their own storage functions plugged in,
so the fusion/timing/budget/cache logic exists exactly once.

Every result set carries a `.timings` dict (per-stage milliseconds) so you
can see where your latency actually went, and `budget_ms` lets you cap
worst-case latency by skipping the vector stage under time pressure rather
than blowing through a deadline.

**Semantic query cache:** if a `query_cache` Store is given, an incoming
query's embedding is compared against recently-cached query embeddings in
the same namespace scope. A cosine similarity above `cache_threshold`
within `cache_ttl_seconds` returns the cached results directly — BM25,
ANN, and fusion are skipped entirely, not just the embedding step. This is
for the common agent pattern of asking near-identical questions with
slightly different wording in a tight loop (retries, multi-step
reasoning) — exact string caching (which `CachingEmbedder` already does
for the embedding call itself) wouldn't catch that.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .ann import AnnIndex
from .embed import CachingEmbedder
from .store import ScoredId

if TYPE_CHECKING:
    from .store import Store

_RRF_K = 60  # standard reciprocal-rank-fusion constant; dampens the impact of rank 1 vs rank 2
DEFAULT_OVERSAMPLE = 4
DEFAULT_CACHE_TTL_SECONDS = 300.0
DEFAULT_CACHE_THRESHOLD = 0.95


@dataclass
class Hit:
    id: int
    text: str
    score: float
    metadata: dict[str, Any]
    namespace: str


class Hits(list):
    """A list[Hit] with a `.timings` dict attached (per-stage latency in ms).

    Subclassing list means `hits[0]`, `len(hits)`, and `for h in hits` all
    just work — `.timings`, `.degraded`, and `.from_cache` are the only
    additions.
    """

    timings: dict[str, float]
    degraded: bool = False
    from_cache: bool = False


def hybrid_search(
    *,
    query: str,
    namespaces: list[str],
    k: int,
    fts_search: Callable[[str, list[str], int], list[ScoredId]],
    fetch_records: Callable[[list[int]], list[Any]],
    ann_index: AnnIndex | None = None,
    embedder: CachingEmbedder | None = None,
    use_bm25: bool = True,
    use_vectors: bool = True,
    budget_ms: float | None = None,
    oversample: int = DEFAULT_OVERSAMPLE,
    query_cache: "Store | None" = None,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    cache_threshold: float = DEFAULT_CACHE_THRESHOLD,
) -> Hits:
    """Search one collection (chunks or memories) and return ranked Hits.

    ``fts_search`` and ``fetch_records`` are store.py calls the caller
    already has bound to the right table (``store.search_chunks_fts`` /
    ``store.get_chunks``, or the ``memories`` equivalents) — this function
    doesn't touch SQL directly except through ``query_cache``.

    Hybrid (both signals on, the default) is what most callers want.
    ``use_bm25=False`` gives pure vector/semantic search; ``use_vectors=False``
    gives pure keyword search — both are legitimate modes, not just
    intermediate states, and RRF fusion only kicks in when both run.

    If ``budget_ms`` is set and time runs out before the vector stage
    (query embedding + ANN search) would start, that stage is skipped and
    results fall back to BM25-only ranking. ``hits.degraded`` is True when
    this happened, so callers can log or surface it.
    """
    timings: dict[str, float] = {}
    start = time.perf_counter()

    def over_budget() -> bool:
        return budget_ms is not None and (time.perf_counter() - start) * 1000 >= budget_ms

    # The query vector is needed for both the semantic cache check and the
    # ANN stage, so compute it once and reuse it for whichever of those
    # actually run.
    want_vectors = use_vectors and ann_index is not None and embedder is not None and len(ann_index) > 0
    want_cache = want_vectors and query_cache is not None
    query_vector: np.ndarray | None = None
    cache_namespace = ",".join(sorted(namespaces))

    if (want_vectors or want_cache) and not over_budget():
        t0 = time.perf_counter()
        query_vector = embedder.embed_one(query)
        timings["embed_ms"] = _ms_since(t0)

    if want_cache and query_vector is not None and not over_budget():
        t0 = time.perf_counter()
        cached = _check_query_cache(
            query_cache, cache_namespace, query_vector, k, cache_threshold, cache_ttl_seconds
        )
        timings["cache_ms"] = _ms_since(t0)
        if cached is not None:
            timings["total_ms"] = (time.perf_counter() - start) * 1000
            cached.timings = timings
            cached.from_cache = True
            return cached

    bm25_hits: list[ScoredId] = []
    if use_bm25:
        t0 = time.perf_counter()
        bm25_hits = fts_search(query, namespaces, k * oversample)
        timings["bm25_ms"] = _ms_since(t0)

    ann_hits: list[tuple[int, float]] = []
    ran_vector_stage = False
    if want_vectors and query_vector is not None and not over_budget():
        t0 = time.perf_counter()
        ann_hits = ann_index.search(query_vector, k * oversample)
        timings["ann_ms"] = _ms_since(t0)
        ran_vector_stage = True

    t0 = time.perf_counter()
    candidate_ids = {hit.id for hit in bm25_hits} | {id_ for id_, _ in ann_hits}
    records_by_id = {r.id: r for r in fetch_records(list(candidate_ids))}

    # ANN has no notion of namespace, so filter its hits post-hoc against
    # the namespaces we actually fetched for. BM25 hits are already
    # namespace-correct (store.py filters in SQL). A stale ANN id whose
    # row was since deleted also gets dropped here for free.
    ann_hits = [
        (id_, sim)
        for id_, sim in ann_hits
        if id_ in records_by_id and records_by_id[id_].namespace in namespaces
    ]

    fused = _reciprocal_rank_fusion(bm25_hits, ann_hits, k)
    hits = Hits(
        Hit(
            id=id_,
            text=records_by_id[id_].text,
            score=score,
            metadata=records_by_id[id_].metadata,
            namespace=records_by_id[id_].namespace,
        )
        for id_, score in fused
        if id_ in records_by_id
    )
    timings["fusion_ms"] = _ms_since(t0)
    timings["total_ms"] = (time.perf_counter() - start) * 1000
    hits.timings = timings
    hits.degraded = want_vectors and not ran_vector_stage

    if want_cache and query_vector is not None and not hits.degraded:
        _write_query_cache(query_cache, cache_namespace, query, query_vector, k, hits)

    return hits


def _reciprocal_rank_fusion(
    bm25_hits: list[ScoredId], ann_hits: list[tuple[int, float]], k: int
) -> list[tuple[int, float]]:
    """Merge two ranked lists by reciprocal rank, not raw score.

    BM25 scores and cosine similarities live on incomparable scales, so we
    can't just add them. RRF sidesteps that: each list only contributes
    "how highly did you rank this", and the sum across lists rewards items
    both signals agree on.
    """
    scores: dict[int, float] = {}
    for rank, hit in enumerate(bm25_hits):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, (id_, _similarity) in enumerate(ann_hits):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:k]


def _check_query_cache(
    store: "Store",
    cache_namespace: str,
    query_vector: np.ndarray,
    k: int,
    threshold: float,
    ttl_seconds: float,
) -> Hits | None:
    store.purge_expired_query_cache(ttl_seconds, now=time.time())
    best_row, best_similarity = None, -1.0
    for row in store.list_query_cache(cache_namespace):
        cached_vector = np.frombuffer(row["query_vector"], dtype=np.float32)
        similarity = _cosine_similarity(query_vector, cached_vector)
        if similarity > best_similarity:
            best_row, best_similarity = row, similarity

    if best_row is None or best_similarity < threshold:
        return None

    payload = json.loads(best_row["results"])
    if payload["k"] < k:
        return None  # cached for a shallower search than what's being asked now

    return Hits(Hit(**h) for h in payload["hits"][:k])


def _write_query_cache(
    store: "Store", cache_namespace: str, query: str, query_vector: np.ndarray, k: int, hits: Hits
) -> None:
    cache_key = hashlib.sha256(f"{cache_namespace}:{query}".encode("utf-8")).hexdigest()
    payload = json.dumps(
        {
            "k": k,
            "hits": [
                {"id": h.id, "text": h.text, "score": h.score, "metadata": h.metadata, "namespace": h.namespace}
                for h in hits
            ],
        }
    )
    store.set_query_cache(
        cache_key, cache_namespace, query_vector.astype(np.float32).tobytes(), payload, created_at=time.time()
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
