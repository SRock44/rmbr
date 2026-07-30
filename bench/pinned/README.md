# Pinned bench results

Raw output cited in the main README's Performance section.

- **Hardware:** Intel Core Ultra 9 285K, isolated to 4 cores (`taskset -c 0-3`)
- **OS/Python:** Ubuntu 24.04.4 LTS, Python 3.12.3

## `latency_*.json` — single-call latency, real default embedder

The headline numbers in README.md. `bench/latency.py`, 3 runs, 100 samples/run,
100-call `remember()` benchmark + 100-query `search()` benchmark against a
500-doc index. Uses rmbr's actual default embedder (local ONNX via
`fastembed`) — not precomputed vectors — because this measures what a real
`remember()`/`search()` call actually costs, embedding included.

```bash
pip install -e .
python bench/latency.py --n-calls 100 --n-queries 100 --corpus-size 500
```

## `bench_*.json` — bulk-ingest throughput and recall@k

Disclosed in README.md but not led with — bulk-loading a corpus in one call
isn't rmbr's target workload (see README's "Alternatives" section). Every
engine (rmbr, chromadb, lancedb) is fed identical precomputed vectors for
identical synthetic documents, isolating storage/ANN engine performance from
embedding cost — see `bench/run.py`'s module docstring for the full
methodology. 3 seeds (0, 1, 2), 5,000 synthetic docs, 500 queries, 384-dim
vectors, k=5.

```bash
pip install -e ".[bench]"
python bench/run.py --n-docs 5000 --n-queries 500 --dim 384 --k 5 --seed 0
```
