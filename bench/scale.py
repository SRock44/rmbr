"""Measures the real cost `Memory.bulk()`/`Index.bulk()` eliminates:
without it, every `remember()`/`add_text()` call re-serializes the
*entire* vector index (usearch has no incremental on-disk save - see
ann.py's module docstring), so a single call's cost grows with total
index size, not just what that one call adds. This measures exactly that
growth, with and without `.bulk()`, at several namespace sizes.

    python bench/scale.py

Seeding up to each size ALWAYS uses `.bulk()` (the realistic way to
bulk-load) - this script doesn't pay the O(size^2) cost it exists to show
`.bulk()` eliminates. Uses `FakeEmbedder` (like run.py, unlike
latency.py): this isolates storage/ANN persistence cost from embedding
cost, since that's specifically what's being measured here.
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

from rmbr.embed import FakeEmbedder  # noqa: E402
from rmbr.index import Index  # noqa: E402
from rmbr.memory import Memory  # noqa: E402

_DIM = 384  # matches bge-small-en-v1.5's real output dimension


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
    }
    print(
        f"{label:<52} mean={result['mean_ms']:>9.2f}ms  "
        f"p50={result['p50_ms']:>9.2f}ms  p95={result['p95_ms']:>9.2f}ms  (n={result['n']})"
    )
    return result


def _seed_memory(mem: Memory, n: int) -> None:
    with mem.bulk():
        for i in range(n):
            mem.remember(f"seed memory number {i} about topic {i % 50}")


def _seed_index(idx: Index, n: int) -> None:
    idx.add_texts([f"seed document {i} about topic {i % 50}" for i in range(n)])


def bench_memory_at_size(db_path: Path, size: int, n_samples: int) -> dict:
    mem = Memory(str(db_path), namespace="agent", embedder=FakeEmbedder(dimension=_DIM))
    _seed_memory(mem, size)

    no_bulk_samples = []
    for i in range(n_samples):
        t0 = time.perf_counter()
        mem.remember(f"no-bulk incremental memory {i}")
        no_bulk_samples.append((time.perf_counter() - t0) * 1000)

    bulk_samples = []
    for i in range(n_samples):
        t0 = time.perf_counter()
        with mem.bulk():
            mem.remember(f"bulk incremental memory {i}")
        bulk_samples.append((time.perf_counter() - t0) * 1000)
    mem.close()

    no_bulk = summarize(f"Memory.remember() @ {size} items, no .bulk()", no_bulk_samples)
    bulk = summarize(f"Memory.remember() @ {size} items, with .bulk()", bulk_samples)
    return {"size": size, "no_bulk": no_bulk, "bulk": bulk, "speedup": no_bulk["mean_ms"] / bulk["mean_ms"]}


def bench_index_at_size(db_path: Path, size: int, n_samples: int) -> dict:
    idx = Index(str(db_path), embedder=FakeEmbedder(dimension=_DIM))
    _seed_index(idx, size)

    no_bulk_samples = []
    for i in range(n_samples):
        t0 = time.perf_counter()
        idx.add_text(f"no-bulk incremental document {i}")
        no_bulk_samples.append((time.perf_counter() - t0) * 1000)

    bulk_samples = []
    for i in range(n_samples):
        t0 = time.perf_counter()
        with idx.bulk():
            idx.add_text(f"bulk incremental document {i}")
        bulk_samples.append((time.perf_counter() - t0) * 1000)
    idx.close()

    no_bulk = summarize(f"Index.add_text() @ {size} chunks, no .bulk()", no_bulk_samples)
    bulk = summarize(f"Index.add_text() @ {size} chunks, with .bulk()", bulk_samples)
    return {"size": size, "no_bulk": no_bulk, "bulk": bulk, "speedup": no_bulk["mean_ms"] / bulk["mean_ms"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 20000, 40000])
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    workdir = args.out / "scale_workdir"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)

    print("Memory.bulk() impact, growing namespace sizes:\n")
    memory_results = [bench_memory_at_size(workdir / f"memory_{size}.db", size, args.n_samples) for size in args.sizes]

    print("\nIndex.bulk() impact, growing namespace sizes:\n")
    index_results = [bench_index_at_size(workdir / f"index_{size}.db", size, args.n_samples) for size in args.sizes]

    out_path = args.out / f"scale_{int(time.time())}.json"
    out_path.write_text(
        json.dumps(
            {
                "config": {"sizes": args.sizes, "n_samples": args.n_samples, "dim": _DIM},
                "memory": memory_results,
                "index": index_results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
