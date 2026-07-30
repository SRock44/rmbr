# rmbr — v0.1 Plan

## Positioning

> **"Give your agent memory and knowledge. One file, three lines, no server, no API key."**

**rmbr** is an embedded, local-first memory + retrieval engine for agents and LLM apps — to hosted memory services (mem0/Zep/Letta) what SQLite is to Postgres. Memory-first: the `Memory` API and MCP support are the headline; document search/RAG is secondary. Open source (MIT), free, library-only (no CLI, no consumer app, no hosted product).

## Rationale

- Raw vector search is not the RAG bottleneck (sub-ms at scale; LLM generation dominates end-to-end latency). The real latency lives in embedding API round trips (50–300ms each), agents making 5–50 retrieval calls per task, framework bloat, and slow ingestion. rmbr attacks those: local embeddings, caching at every layer, minimal dependencies.
- "Fast RAG library" is a stale, crowded category. **Agent memory is the growth category** — and every incumbent is server- or cloud-shaped. The embedded/local/free lane is open.
- Differentiators, in order:
  1. **Single-file portability** — the index IS one `.db` file
  2. **Time-to-first-search** — published table of install size, dependency count, import time, lines-to-first-result vs incumbents
  3. **Agent-native** — MCP, memory API, policy layer, latency budgets, per-stage timings
  4. **All 3 OSes first-class** (Windows included)
  5. **Local/private by default** — no API key, nothing phones home
- rmbr never calls an LLM → compatible with every model, zero integration bloat.

## Claims policy

No number appears in the README that isn't produced by `bench/run.py` — reproducible, recall pinned, vs named competitors (Chroma + LanceDB) on identical hardware. Until then: only externally citable benchmarks and countable facts.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design. Summary: one SQLite file per index (stdlib, zero added deps) holding documents, chunks, memories, caches, and the serialized HNSW blob; `usearch` in RAM for vectors (switched from the originally-planned `hnswlib`, which has no Windows wheel for Python 3.13 — see ARCHITECTURE.md); FTS5 for BM25; hybrid search via reciprocal rank fusion; local ONNX embeddings by default with a content-hash cache; deny-by-default `Policy` layer over namespaces; namespace-pinned MCP serving.

## Package layout

```
rmbr/
├── pyproject.toml          # hatchling; deps: fastembed, hnswlib, mcp; extras: [openai], [bench]
├── LICENSE                 # MIT
├── README.md
├── docs/ARCHITECTURE.md
├── docs/PLAN.md
├── .github/workflows/ci.yml  # matrix: ubuntu/windows/macos × supported Pythons
├── src/rmbr/
│   ├── __init__.py         # exports Memory, Index, Policy, serve_mcp
│   ├── memory.py           # Memory API (headline)
│   ├── policy.py           # access policy layer
│   ├── index.py            # Index facade
│   ├── store.py            # SQLite: docs, chunks, memories, FTS5, caches, ANN blob
│   ├── ann.py              # hnswlib wrapper + numpy fallback, persistence
│   ├── embed.py            # Embedder protocol, fastembed default, providers, cache
│   ├── search.py           # hybrid + RRF + semantic cache + timings/budget
│   ├── chunk.py            # splitters
│   ├── mcp_server.py       # serve_mcp(), namespace-pinned tools
│   └── __main__.py         # python -m rmbr (MCP launch shim only)
├── bench/
│   ├── run.py              # p50/p95 latency, ingestion throughput, recall@k (pinned)
│   └── competitors.py      # chromadb + lancedb (optional extras)
└── tests/
```

## Implementation order

1. `store.py` + `chunk.py` — schema (memories, ANN blob), FTS5, ingestion
2. `embed.py` — fastembed default + content-hash cache (injectable fake embedder for offline tests)
3. `ann.py` + `search.py` — hybrid RRF, timings, budget
4. `memory.py` + `policy.py` + `index.py`
5. Semantic query cache
6. `mcp_server.py` + `__main__.py`
7. CI (3-OS matrix) — added early to catch platform issues during development
8. `bench/` — vs chromadb + lancedb (Linux server for publishable numbers)
9. README final numbers (bench-produced only)

## Verification

- `pytest` via CI on ubuntu/windows/macos
- Integration: ingest → hybrid search returns expected hits; warm cached query shows ~0 embed time in timings
- Memory: persistence across processes; namespace isolation; policy grant/deny paths
- MCP smoke test: `python -m rmbr` driven by an MCP client script; verify namespace pinning
- Bench: p50/p95 latency, ingestion throughput, recall@k vs Chroma + LanceDB
- Clean-venv wheels-only install on Windows, enforced automatically by CI's `--only-binary :all:` step (this is how the hnswlib→usearch switch above got caught)
