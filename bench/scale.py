"""Measures the real cost `Memory.bulk()`/`Index.bulk()` eliminates:
without it, every `remember()`/`add_text()` call re-serializes the
*entire* vector index (usearch has no incremental on-disk save - see
ann.py's module docstring), so a single call's cost grows with total
index size, not just what that one call adds. This measures the cost of
`n_writes` sequential writes into an already-`size`-large namespace, with
and without `.bulk()`, at several namespace sizes.

    python bench/scale.py

**The comparison that matters is N writes without `.bulk()` vs. the same
N writes batched inside ONE `.bulk()` block** - not N writes each
wrapped in their own `.bulk()` block, which would pay the same N
reserializes as not using it at all and show no benefit (`.bulk()` only
amortizes the reserialize cost across everything written *inside one
block*).

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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rmbr.embed import FakeEmbedder  # noqa: E402
from rmbr.index import Index  # noqa: E402
from rmbr.memory import Memory  # noqa: E402

_DIM = 384  # matches bge-small-en-v1.5's real output dimension


def summarize(label: str, size: int, n_writes: int, no_bulk_ms: float, bulk_ms: float) -> dict:
    result = {
        "size": size,
        "n_writes": n_writes,
        "no_bulk_total_ms": no_bulk_ms,
        "no_bulk_per_write_ms": no_bulk_ms / n_writes,
        "bulk_total_ms": bulk_ms,
        "bulk_per_write_ms": bulk_ms / n_writes,
        "speedup": no_bulk_ms / bulk_ms,
    }
    print(
        f"{label} @ {size:>6} items: {n_writes} writes, no .bulk() = {no_bulk_ms:>10.1f}ms "
        f"({result['no_bulk_per_write_ms']:>7.2f}ms/write)  |  with .bulk() = {bulk_ms:>8.1f}ms "
        f"({result['bulk_per_write_ms']:>6.2f}ms/write)  |  {result['speedup']:>6.1f}x faster"
    )
    return result


def _seed_memory(mem: Memory, n: int) -> None:
    with mem.bulk():
        for i in range(n):
            mem.remember(f"seed memory number {i} about topic {i % 50}")


def _seed_index(idx: Index, n: int) -> None:
    idx.add_texts([f"seed document {i} about topic {i % 50}" for i in range(n)])


def bench_memory_at_size(db_path: Path, size: int, n_writes: int) -> dict:
    mem = Memory(str(db_path), namespace="agent", embedder=FakeEmbedder(dimension=_DIM))
    _seed_memory(mem, size)

    t0 = time.perf_counter()
    for i in range(n_writes):
        mem.remember(f"no-bulk incremental memory {i}")
    no_bulk_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with mem.bulk():
        for i in range(n_writes):
            mem.remember(f"bulk incremental memory {i}")
    bulk_ms = (time.perf_counter() - t0) * 1000
    mem.close()

    return summarize("Memory.remember()", size, n_writes, no_bulk_ms, bulk_ms)


def bench_index_at_size(db_path: Path, size: int, n_writes: int) -> dict:
    idx = Index(str(db_path), embedder=FakeEmbedder(dimension=_DIM))
    _seed_index(idx, size)

    t0 = time.perf_counter()
    for i in range(n_writes):
        idx.add_text(f"no-bulk incremental document {i}")
    no_bulk_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with idx.bulk():
        for i in range(n_writes):
            idx.add_text(f"bulk incremental document {i}")
    bulk_ms = (time.perf_counter() - t0) * 1000
    idx.close()

    return summarize("Index.add_text()", size, n_writes, no_bulk_ms, bulk_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 20000, 40000])
    parser.add_argument("--n-writes", type=int, default=50, help="sequential writes measured per size")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    workdir = args.out / "scale_workdir"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)

    print(f"Memory.bulk() impact: cost of {args.n_writes} sequential writes, growing namespace sizes:\n")
    memory_results = [bench_memory_at_size(workdir / f"memory_{size}.db", size, args.n_writes) for size in args.sizes]

    print(f"\nIndex.bulk() impact: cost of {args.n_writes} sequential writes, growing namespace sizes:\n")
    index_results = [bench_index_at_size(workdir / f"index_{size}.db", size, args.n_writes) for size in args.sizes]

    out_path = args.out / f"scale_{int(time.time())}.json"
    out_path.write_text(
        json.dumps(
            {
                "config": {"sizes": args.sizes, "n_writes": args.n_writes, "dim": _DIM},
                "memory": memory_results,
                "index": index_results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
