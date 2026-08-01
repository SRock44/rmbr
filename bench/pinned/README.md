# Pinned bench results

Raw output cited in the main README's Performance section.

- **Hardware:** Intel Core Ultra 9 285K, isolated to 4 cores (`taskset -c 0-3`)
- **OS/Python:** Ubuntu 24.04.4 LTS, Python 3.12.3

## `latency_*.json` — single-call latency, real default embedder

The headline numbers in README.md. `bench/latency.py`, 3 runs, 100 samples/run:
`remember()`, plain `search()`, `search(rerank=True)`, and
`search(recency_weight=0.3)`, each against a 500-doc index. Uses rmbr's
actual default embedder (local ONNX via `fastembed`) — not precomputed
vectors — because this measures what a real call actually costs, embedding
included.

Each scenario runs in its own subprocess (`bench/latency.py --scenario ...`,
orchestrated automatically — you don't need to pass that flag yourself).
Running all scenarios back-to-back in a single process measurably polluted
each other's tail latency (page cache pressure, ONNX runtime thread reuse,
GC pauses from building 4 separate several-hundred-doc indices in one
process) — isolating them is what makes the p95/p99 numbers trustworthy.

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
methodology. `mem0` is the one exception: it has no public "give me a raw
vector" API, so its embedder is swapped out for the same precomputed-vector
lookup *after* construction instead — same vectors, different mechanism (see
`run_mem0()`'s docstring in `bench/competitors.py`). mem0 is benched with its
own default hybrid (dense + BM25 sparse) search left on, matching how anyone
actually gets it by installing it — same reasoning as rmbr's own
"hybrid, default" row. 3 seeds (0, 1, 2), 5,000 synthetic docs, 500 queries,
384-dim vectors, k=5.

```bash
pip install -e ".[bench]"
python bench/run.py --n-docs 5000 --n-queries 500 --dim 384 --k 5 --seed 0
```

## `scale_*.json` — `Memory.bulk()`/`Index.bulk()` impact at growing namespace sizes

Answers a specific question: without `.bulk()`, every write re-serializes
the *entire* vector index (usearch has no incremental on-disk save), so
does rmbr actually hold up once a namespace has tens of thousands of
items in it? Measures the cost of 50 sequential writes with vs. without
`.bulk()`, at namespace sizes from 1,000 to 40,000. `FakeEmbedder`, same
reasoning as `bench_*.json` — isolates storage/ANN persistence cost from
embedding cost. One run (not 3 seeds): this is a deterministic
measurement of a serialization cost, not something embedding/network
jitter affects run to run, so repeat runs add little signal here.

```bash
pip install -e .
python bench/scale.py --sizes 1000 5000 10000 20000 40000 --n-writes 50
```

## `mcp_latency_*.json` / `http_latency_*.json` — real protocol round-trip

What a caller actually experiences through MCP or HTTP, not just the
in-process Python API `latency_*.json` measures: `mcp_latency.py` spawns
a real `python -m rmbr` subprocess and talks to it over real stdio with
the real `mcp` client SDK; `http_latency.py` binds a real uvicorn server
to a real OS socket and hits it with a real `httpx` client. Both use the
real default embedder, 500-doc/memory corpus, 50 samples per call type.

```bash
pip install -e .
python bench/mcp_latency.py --corpus-size 500 --n-calls 50
python bench/http_latency.py --corpus-size 500 --n-calls 50
```
