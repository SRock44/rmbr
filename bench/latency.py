"""Single-call latency, with the real default embedder - not a competitor
bake-off, a measurement of what using rmbr as shipped actually costs.

    python bench/latency.py

This is deliberately different from run.py's methodology. run.py isolates
engine performance by feeding every engine identical precomputed vectors -
useful for comparing storage/ANN engines apples-to-apples, but it excludes
embedding time, and embedding time is what an agent actually waits on:
`mem.remember(...)` and `idx.search(...)` both call the real local ONNX
model (`fastembed`, downloaded on first use) here.

The number that matters for rmbr's actual use case - an agent calling
remember()/recall() one at a time in the middle of a reasoning loop, not
bulk-loading a corpus - is single-call latency, not bulk throughput. This
script measures exactly that, at a realistic memory/knowledge-base size,
and reports the same per-stage `timings` breakdown every real call
returns, so the numbers here are exactly what a caller sees in production,
not benchmark-only instrumentation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rmbr.index import Index  # noqa: E402
from rmbr.memory import Memory  # noqa: E402


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[max(0, int(len(values) * p) - 1)]


def summarize(label: str, samples: list[float]) -> dict:
    result = {
        "label": label,
        "n": len(samples),
        "mean_ms": statistics.mean(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
    }
    print(
        f"{label:<32} mean={result['mean_ms']:.2f}ms  p50={result['p50_ms']:.2f}ms  "
        f"p95={result['p95_ms']:.2f}ms  p99={result['p99_ms']:.2f}ms  (n={result['n']})"
    )
    return result


def bench_single_remember(db_path: Path, n_calls: int) -> dict:
    mem = Memory(str(db_path), namespace="agent")
    samples = []
    for i in range(n_calls):
        t0 = time.perf_counter()
        mem.remember(f"the user asked about topic {i} and preferred a concise answer")
        samples.append((time.perf_counter() - t0) * 1000)
    mem.close()
    return summarize("single remember() call", samples)


def bench_single_search(db_path: Path, corpus_size: int, n_queries: int) -> tuple[dict, dict]:
    idx = Index(str(db_path), namespace="agent")
    idx.add_texts(
        [f"document {i}: notes about deployment, testing, and configuration topic {i}" for i in range(corpus_size)]
    )

    samples = []
    embed_stage_samples = []
    for i in range(n_queries):
        # A genuinely unique query text every time, regardless of how
        # n_queries relates to corpus_size - reusing a query string would
        # hit CachingEmbedder's cache and silently understate embed cost.
        t0 = time.perf_counter()
        hits = idx.search(f"how do I configure topic {i} in a way nobody has asked before?", k=5)
        samples.append((time.perf_counter() - t0) * 1000)
        embed_stage_samples.append(hits.timings.get("embed_ms", 0.0))
    idx.close()

    overall = summarize(f"single search() call ({corpus_size} docs)", samples)
    embed_share = summarize("  of which: query embedding", embed_stage_samples)
    return overall, embed_share


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-calls", type=int, default=50, help="samples for the remember() benchmark")
    parser.add_argument("--n-queries", type=int, default=50, help="samples for the search() benchmark")
    parser.add_argument("--corpus-size", type=int, default=500, help="docs pre-indexed before searching")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    workdir = args.out / "latency_workdir"
    # Every run starts from a clean workdir - remember()/search() calls
    # must genuinely embed each sample's (unique) text via the real
    # model, not silently hit CachingEmbedder's on-disk cache from a
    # previous run of this script and report near-zero embed time.
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)

    print("Warming up the local embedding model (downloads on first use)...")
    warm = Memory(str(workdir / "warmup.db"), namespace="warm")
    warm.remember("warmup call to trigger the one-time model load")
    warm.close()

    print("\nSingle-call latency, real default embedder (local ONNX, fastembed):\n")
    remember_result = bench_single_remember(workdir / "remember.db", args.n_calls)
    search_result, embed_share_result = bench_single_search(workdir / "search.db", args.corpus_size, args.n_queries)

    out_path = args.out / f"latency_{int(time.time())}.json"
    out_path.write_text(
        json.dumps(
            {
                "config": {"n_calls": args.n_calls, "n_queries": args.n_queries, "corpus_size": args.corpus_size},
                "results": [remember_result, search_result, embed_share_result],
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
