"""rmbr's bench harness: p50/p95 search latency, ingestion throughput, and
recall@k, measured against chromadb and lancedb on identical vectors.

    python -m venv .venv && .venv/bin/pip install -e ".[bench]"
    python bench/run.py

**Why this exists:** rmbr's claims policy (see docs/PLAN.md) is that no
performance number goes in the README unless this script produced it,
reproducibly, on the same corpus as the competitors it's compared
against. Every engine here is fed the *same* precomputed vectors for the
*same* synthetic documents (see corpus.py) — this measures storage +
ANN engine performance, not embedding-model quality, and it means a
number this script prints is directly comparable across engines.

rmbr is benched twice: once in its default hybrid mode (BM25 + vector,
what `idx.search()` actually does for a real user) and once with
`use_bm25=False` (vector search only, apples-to-apples with chromadb/
lancedb's pure vector query). Reporting only the vector-only number would
flatter rmbr by hiding the BM25 stage's cost; reporting only hybrid would
unfairly penalize it against engines that don't do BM25 at all. Both
numbers are real and both are published.

Numbers from a laptop are directional, not a claim. Publishable numbers
for the README come from running this on the project's pinned Linux
benchmark machine — see docs/PLAN.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corpus import Corpus, generate_corpus, recall_at_k  # noqa: E402

from rmbr.index import Index  # noqa: E402


@dataclass
class EngineResult:
    name: str
    ingest_seconds: float
    docs_per_second: float
    search_p50_ms: float
    search_p95_ms: float
    recall_at_k: float


class PrecomputedEmbedder:
    """Looks vectors up by exact text instead of computing them.

    Lets rmbr be fed the identical vectors used for chromadb/lancedb, so
    the bench measures engine performance, not embedding cost or quality.
    """

    model_name = "precomputed"

    def __init__(self, vectors_by_text: dict[str, np.ndarray]):
        self._vectors = vectors_by_text

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vectors[t] for t in texts]


def run_rmbr(
    corpus: Corpus, query_texts: list[str], k: int, workdir: Path, *, use_bm25: bool, use_vectors: bool
) -> EngineResult:
    vectors_by_text = dict(zip(corpus.doc_texts, corpus.doc_vectors))
    vectors_by_text.update(zip(query_texts, corpus.query_vectors))

    db_path = workdir / f"rmbr_bm25{use_bm25}_vec{use_vectors}.db"
    db_path.unlink(missing_ok=True)
    idx = Index(str(db_path), embedder=PrecomputedEmbedder(vectors_by_text))

    t0 = time.perf_counter()
    document_ids = idx.add_texts(corpus.doc_texts)
    ingest_seconds = time.perf_counter() - t0

    # bench needs the chunk id (what search results are keyed by) for each
    # doc index, which add_texts()'s return value (document ids) doesn't
    # give directly — one chunk per doc here, so this is a direct lookup.
    doc_index_by_chunk_id = {
        idx._store.get_chunk_ids_for_document(doc_id)[0]: i for i, doc_id in enumerate(document_ids)
    }

    latencies_ms = []
    predicted = []
    for query_text in query_texts:
        t0 = time.perf_counter()
        hits = idx.search(query_text, k=k, use_bm25=use_bm25, use_vectors=use_vectors)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        predicted.append([doc_index_by_chunk_id[h.id] for h in hits if h.id in doc_index_by_chunk_id])

    idx.close()
    name = "rmbr (hybrid, default)" if use_bm25 and use_vectors else "rmbr (vector-only)"
    return _summarize(name, ingest_seconds, len(corpus.doc_texts), latencies_ms, predicted, corpus.ground_truth)


def run_competitor(name: str, fn, corpus: Corpus, k: int, workdir: Path) -> EngineResult | None:
    try:
        ingest_seconds, latencies_ms, predicted = fn(corpus, k, workdir)
    except ImportError:
        print(f"  [skip] {name} not installed — `pip install rmbr[bench]` to include it")
        return None
    return _summarize(name, ingest_seconds, len(corpus.doc_texts), latencies_ms, predicted, corpus.ground_truth)


def _summarize(
    name: str,
    ingest_seconds: float,
    n_docs: int,
    latencies_ms: list[float],
    predicted: list[list[int]],
    ground_truth: list[list[int]],
) -> EngineResult:
    sorted_latencies = sorted(latencies_ms)
    return EngineResult(
        name=name,
        ingest_seconds=ingest_seconds,
        docs_per_second=n_docs / ingest_seconds if ingest_seconds > 0 else float("inf"),
        search_p50_ms=statistics.median(sorted_latencies),
        search_p95_ms=sorted_latencies[int(len(sorted_latencies) * 0.95) - 1],
        recall_at_k=recall_at_k(predicted, ground_truth),
    )


def _print_table(results: list[EngineResult]) -> None:
    header = f"{'engine':<26}{'ingest (docs/s)':>18}{'p50 (ms)':>12}{'p95 (ms)':>12}{'recall@k':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<26}{r.docs_per_second:>18.1f}{r.search_p50_ms:>12.3f}"
            f"{r.search_p95_ms:>12.3f}{r.recall_at_k:>12.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-docs", type=int, default=2000)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument(
        "--skip-competitors", action="store_true", help="only bench rmbr (fast local sanity check)"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    workdir = args.out / "workdir"
    workdir.mkdir(exist_ok=True)

    print(f"Generating synthetic corpus: {args.n_docs} docs, {args.n_queries} queries, dim={args.dim}")
    corpus = generate_corpus(args.n_docs, args.n_queries, args.dim, args.k, seed=args.seed)
    query_texts = [f"synthetic benchmark query number {i}" for i in range(args.n_queries)]

    print("\nRunning rmbr...")
    results = [
        run_rmbr(corpus, query_texts, args.k, workdir, use_bm25=True, use_vectors=True),
        run_rmbr(corpus, query_texts, args.k, workdir, use_bm25=False, use_vectors=True),
    ]

    if not args.skip_competitors:
        print("Running competitors...")
        from competitors import run_chromadb, run_lancedb

        for name, fn in [("chromadb", run_chromadb), ("lancedb", run_lancedb)]:
            result = run_competitor(name, fn, corpus, args.k, workdir)
            if result is not None:
                results.append(result)

    print()
    _print_table(results)

    out_path = args.out / f"bench_{int(time.time())}.json"
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "n_docs": args.n_docs,
                    "n_queries": args.n_queries,
                    "dim": args.dim,
                    "k": args.k,
                    "seed": args.seed,
                },
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
