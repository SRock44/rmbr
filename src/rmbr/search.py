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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from .ann import AnnIndex
from .embed import CachingEmbedder
from .rerank import Reranker
from .store import ScoredId

if TYPE_CHECKING:
    from .store import Store

_RRF_K = 60  # standard reciprocal-rank-fusion constant; dampens the impact of rank 1 vs rank 2
DEFAULT_OVERSAMPLE = 4
DEFAULT_CACHE_TTL_SECONDS = 300.0
DEFAULT_CACHE_THRESHOLD = 0.95
DEFAULT_RECENCY_HALF_LIFE_SECONDS = 7 * 24 * 3600.0  # a week: a memory this old carries half the boost of a new one


@dataclass
class Hit:
    id: int
    text: str
    score: float
    metadata: dict[str, Any]
    namespace: str
    # Raw component signals behind `score`, when available — `score` itself
    # is whatever determined final order (RRF-fused by default, or the
    # cross-encoder's relevance score when `rerank=True`), which isn't
    # calibrated to any absolute scale you can threshold on. `vector_score`
    # (raw cosine similarity, 0..1-ish) is: that's what `min_similarity`
    # actually filters on. `bm25_score` is FTS5's raw bm25() value (more
    # negative = more relevant) for whichever hits the keyword stage found.
    # Either is None when that hit didn't come from that stage.
    bm25_score: float | None = None
    vector_score: float | None = None


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
    where: dict[str, Any] | None = None,
    query_cache: Store | None = None,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    cache_threshold: float = DEFAULT_CACHE_THRESHOLD,
    min_similarity: float | None = None,
    recency_weight: float = 0.0,
    recency_half_life_seconds: float = DEFAULT_RECENCY_HALF_LIFE_SECONDS,
    record_timestamp: Callable[[Any], str] | None = None,
    reranker: Reranker | None = None,
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

    ``where`` filters results to records whose metadata matches every
    key/value pair given. A plain value is equality (``{"category": "docs"}``);
    a dict of one or more ``$op`` keys is an operator comparison
    (``{"price": {"$gt": 10}}``, ``{"tag": {"$in": ["a", "b"]}}``) — see
    ``_WHERE_OPERATORS`` for the full set. Applied as a post-filter on the
    same candidate pool BM25/ANN already returned rather than pushed into
    SQL, so it shares the same oversample-then-filter tradeoff as
    namespace filtering below: a filter that's very selective against a
    large candidate pool can return fewer than ``k`` hits.

    ``min_similarity`` drops any hit whose raw cosine similarity
    (``Hit.vector_score``) is below the threshold — including hits that
    only came from BM25 and have no vector_score at all, since there's no
    similarity evidence to vouch for them. Requires ``use_vectors=True``
    (raises ``ValueError`` otherwise, rather than silently returning
    nothing). Deliberately gates on the raw cosine number, not
    ``Hit.score`` — the fused RRF score is a rank-sum with no fixed scale,
    not a confidence you can sensibly threshold on.

    ``recency_weight`` (0.0, off, by default) blends a recency bonus into
    the fusion score: ``0.5 ** (age / recency_half_life_seconds)``, so a
    record right at the half-life is worth half what a brand-new one is.
    Requires ``record_timestamp`` (a ``record -> ISO-8601 string``
    extractor) since not every collection carries a timestamp per record —
    `Memory.recall()` wires this in; `Index.search()` doesn't yet, since
    chunks aren't individually timestamped.

    ``reranker`` (a `rerank.Reranker`, e.g. `rerank.CrossEncoderReranker`),
    if given, re-scores the whole candidate pool (not just the top ``k``
    BM25/ANN already picked) by running a local cross-encoder over
    ``(query, candidate_text)`` pairs, and ``Hit.score`` becomes that
    cross-encoder score rather than the RRF-fused one — it's what actually
    determined the final order.

    If ``budget_ms`` is set and time runs out before the vector stage
    (query embedding + ANN search) would start, that stage is skipped and
    results fall back to BM25-only ranking. ``hits.degraded`` is True when
    this happened, so callers can log or surface it.
    """
    if min_similarity is not None and not use_vectors:
        raise ValueError("min_similarity requires use_vectors=True — there's no similarity to filter on otherwise")
    if recency_weight and record_timestamp is None:
        raise ValueError("recency_weight requires record_timestamp (a record -> created_at extractor)")

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
    # `where`/`min_similarity`/recency/rerank are folded into the cache
    # scope so a cached result computed under different filtering or
    # ranking rules can never be handed back for this call - a correctness
    # requirement, not an optimization.
    cache_scope = (
        ",".join(sorted(namespaces))
        + "|"
        + json.dumps(
            {
                "where": where,
                "min_similarity": min_similarity,
                "recency_weight": recency_weight,
                "recency_half_life_seconds": recency_half_life_seconds if recency_weight else None,
                "reranker": getattr(reranker, "model_name", "custom") if reranker is not None else None,
            },
            sort_keys=True,
        )
    )

    if (want_vectors or want_cache) and not over_budget():
        t0 = time.perf_counter()
        query_vector = embedder.embed_one(query)
        timings["embed_ms"] = _ms_since(t0)

    if want_cache and query_vector is not None and not over_budget():
        t0 = time.perf_counter()
        cached = _check_query_cache(
            query_cache, cache_scope, query_vector, k, cache_threshold, cache_ttl_seconds
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
    if where:
        records_by_id = {
            id_: record for id_, record in records_by_id.items() if matches_where(record.metadata, where)
        }

    # ANN has no notion of namespace or metadata, so filter its hits
    # post-hoc against the namespaces/where-clause we actually fetched
    # for. BM25 hits are already namespace-correct (store.py filters in
    # SQL); the final Hits construction below drops any bm25 id that
    # didn't survive the where-filter the same way. A stale ANN id whose
    # row was since deleted also gets dropped here for free.
    ann_hits = [
        (id_, sim)
        for id_, sim in ann_hits
        if id_ in records_by_id and records_by_id[id_].namespace in namespaces
    ]

    bm25_scores = {hit.id: hit.raw_score for hit in bm25_hits if hit.id in records_by_id}
    vector_scores = {id_: sim for id_, sim in ann_hits}

    recency_scores: dict[int, float] | None = None
    if recency_weight and record_timestamp is not None:
        now = time.time()
        recency_scores = {
            id_: _recency_factor(record_timestamp(record), now, recency_half_life_seconds)
            for id_, record in records_by_id.items()
        }

    # Fuse the *whole* candidate pool, not just the top k - identical to
    # the old behavior when nothing reranks (sorting once and slicing to k
    # is the same result as sorting to k directly), but a reranker needs
    # the full pool to have anything to fix.
    fused_pool = _reciprocal_rank_fusion(
        bm25_hits,
        ann_hits,
        len(candidate_ids) or 1,
        recency_scores=recency_scores,
        recency_weight=recency_weight,
    )
    fused_pool = [(id_, score) for id_, score in fused_pool if id_ in records_by_id]

    if reranker is not None and fused_pool:
        rerank_t0 = time.perf_counter()
        ordered_ids = [id_ for id_, _ in fused_pool]
        rerank_scores = reranker.rerank(query, [records_by_id[id_].text for id_ in ordered_ids])
        fused = sorted(zip(ordered_ids, rerank_scores), key=lambda pair: pair[1], reverse=True)[:k]
        timings["rerank_ms"] = _ms_since(rerank_t0)
    else:
        fused = fused_pool[:k]

    hit_list = [
        Hit(
            id=id_,
            text=records_by_id[id_].text,
            score=score,
            metadata=records_by_id[id_].metadata,
            namespace=records_by_id[id_].namespace,
            bm25_score=bm25_scores.get(id_),
            vector_score=vector_scores.get(id_),
        )
        for id_, score in fused
    ]
    if min_similarity is not None:
        hit_list = [h for h in hit_list if h.vector_score is not None and h.vector_score >= min_similarity]

    hits = Hits(hit_list)
    timings["fusion_ms"] = _ms_since(t0)
    timings["total_ms"] = (time.perf_counter() - start) * 1000
    hits.timings = timings
    hits.degraded = want_vectors and not ran_vector_stage

    if want_cache and query_vector is not None and not hits.degraded:
        _write_query_cache(query_cache, cache_scope, query, query_vector, k, hits)

    return hits


def _reciprocal_rank_fusion(
    bm25_hits: list[ScoredId],
    ann_hits: list[tuple[int, float]],
    k: int,
    *,
    recency_scores: dict[int, float] | None = None,
    recency_weight: float = 0.0,
) -> list[tuple[int, float]]:
    """Merge two ranked lists by reciprocal rank, not raw score.

    BM25 scores and cosine similarities live on incomparable scales, so we
    can't just add them. RRF sidesteps that: each list only contributes
    "how highly did you rank this", and the sum across lists rewards items
    both signals agree on. `recency_weight`/`recency_scores`, when given,
    add a third additive term on top — same trick, just with "how recent"
    standing in for "how highly ranked".
    """
    scores: dict[int, float] = {}
    for rank, hit in enumerate(bm25_hits):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, (id_, _similarity) in enumerate(ann_hits):
        scores[id_] = scores.get(id_, 0.0) + 1.0 / (_RRF_K + rank + 1)
    if recency_weight and recency_scores:
        for id_ in scores:
            scores[id_] += recency_weight * recency_scores.get(id_, 0.0)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:k]


def _check_query_cache(
    store: Store,
    cache_scope: str,
    query_vector: np.ndarray,
    k: int,
    threshold: float,
    ttl_seconds: float,
) -> Hits | None:
    store.purge_expired_query_cache(ttl_seconds, now=time.time())
    best_row, best_similarity = None, -1.0
    for row in store.list_query_cache(cache_scope):
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
    store: Store, cache_scope: str, query: str, query_vector: np.ndarray, k: int, hits: Hits
) -> None:
    cache_key = hashlib.sha256(f"{cache_scope}:{query}".encode()).hexdigest()
    payload = json.dumps(
        {
            "k": k,
            "hits": [
                {
                    "id": h.id,
                    "text": h.text,
                    "score": h.score,
                    "metadata": h.metadata,
                    "namespace": h.namespace,
                    "bm25_score": h.bm25_score,
                    "vector_score": h.vector_score,
                }
                for h in hits
            ],
        }
    )
    store.set_query_cache(
        cache_key, cache_scope, query_vector.astype(np.float32).tobytes(), payload, created_at=time.time()
    )


_WHERE_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "$eq": lambda field, value: field == value,
    "$ne": lambda field, value: field != value,
    "$gt": lambda field, value: field is not None and field > value,
    "$gte": lambda field, value: field is not None and field >= value,
    "$lt": lambda field, value: field is not None and field < value,
    "$lte": lambda field, value: field is not None and field <= value,
    "$in": lambda field, value: field in value,
    "$nin": lambda field, value: field not in value,
}


def matches_where(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    """Equality by default (`{"category": "docs"}`); `$op` dicts for the rest.

    A missing field or a type mismatch (comparing None against a number,
    say) is treated as "doesn't match" rather than raising - a filter that
    can't be evaluated for a given record shouldn't crash the whole search.
    """
    for key, expected in where.items():
        field = metadata.get(key)
        if isinstance(expected, dict) and expected and all(isinstance(k, str) and k.startswith("$") for k in expected):
            for op, operand in expected.items():
                if op not in _WHERE_OPERATORS:
                    raise ValueError(f"Unsupported where operator {op!r}; choose one of {sorted(_WHERE_OPERATORS)}")
                try:
                    if not _WHERE_OPERATORS[op](field, operand):
                        return False
                except TypeError:
                    return False
        elif field != expected:
            return False
    return True


def _recency_factor(created_at: str, now: float, half_life_seconds: float) -> float:
    if half_life_seconds <= 0:
        return 0.0
    age_seconds = max(0.0, now - datetime.fromisoformat(created_at).timestamp())
    return 0.5 ** (age_seconds / half_life_seconds)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
