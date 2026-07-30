# rmbr Architecture

## Design principle

One `.db` file per index. Everything — documents, chunks, memories, caches, and the serialized vector index — lives inside a single SQLite file. The file *is* the product: portable, committable, shippable.

```
┌─────────────────────────────────────────────────────┐
│  your process                                       │
│                                                     │
│  Memory ──┐                        ┌── Policy       │
│  Index ───┼── search.py (hybrid) ──┤   (deny-by-    │
│           │                        │    default)    │
│      ┌────┴─────┐                  └───────────────-│
│      │          │                                   │
│  embed.py    ann.py (HNSW, in RAM)                  │
│      │          │                                   │
│      └────┬─────┘                                   │
│        store.py (SQLite)                            │
└───────────┼─────────────────────────────────────────┘
            ▼
       agents.db   ← one file: docs, chunks, memories,
                     FTS5 index, embed cache, query cache,
                     serialized ANN index blob
```

## Components

### store.py — SQLite layer
- Python stdlib `sqlite3` — **zero added dependencies**. SQLite is not in the query hot path; vectors are searched in RAM.
- Tables: `documents`, `chunks`, `memories`, `embed_cache`, `query_cache`, `ann_index` (serialized HNSW blob), plus an FTS5 virtual table over chunk/memory text for BM25.
- WAL mode; ACID gives us crash-safety for free.

### embed.py — embeddings
- `Embedder` protocol (pluggable; tests inject a deterministic fake for offline runs).
- Default: local ONNX model via `fastembed` (`BAAI/bge-small-en-v1.5`) — no network after first model download, no API key.
- Optional extras: OpenAI / Voyage / Cohere (~3–4 providers total, never hundreds).
- **Embedding cache:** SHA-256(content) → vector, stored in SQLite. The same text is never embedded twice, across sessions and processes.

### ann.py — vector search
- HNSW via `usearch`, held in RAM, serialized into the SQLite file on save (preserves the one-file promise). Originally planned around `hnswlib`, but hnswlib ships no prebuilt wheel for Python 3.13 on Windows — building it from source would need a C++ toolchain on every machine that installs rmbr, which fails the "pip install, no fuss" promise. usearch has wheels for all 3 OSes and the same core algorithm; CI's wheels-only install step (`--only-binary :all:`) exists specifically to catch a regression like this early.

### search.py — hybrid retrieval
- BM25 (FTS5) + vector ANN, merged by reciprocal rank fusion. Hybrid is the default.
- **Semantic query cache:** incoming query embedding vs cached query embeddings; cosine ≥ threshold (default ~0.95) within TTL returns cached results without touching the index.
- **Timings:** every result set carries per-stage latency (embed / ann / bm25 / fusion / cache).
- **Budgets:** `search(..., budget_ms=N)` degrades gracefully (skip stages) rather than blowing the budget.

### memory.py — the headline API
- `Memory(path, namespace, policy=None)` → `remember()`, `recall()`, `forget()`, `list()`.
- Namespaced and timestamped. Same engine and file as `Index` — memory and retrieval are one primitive.

### policy.py — access control ("master harness")
- Deny-by-default across namespaces. `Policy.strict()` (no cross-namespace, the default), `Policy.open()`, explicit grants: `policy.allow("coder", read="researcher")`, `read="*"` for supervisors.
- `on_access=(who, verb, namespace) -> bool` callback escape hatch for custom harness logic.
- Deterministic in the hot path — an LLM never gates a recall.
- **Honesty note:** namespaces + policy are organizational boundaries, not cryptographic ones. Hard isolation between untrusted agents = separate files + OS permissions. The docs state this; we don't pretend otherwise.

### mcp_server.py — external agents
- `serve_mcp(path, namespace=..., read_only=...)` on the official `mcp` SDK (stdio). Tools: `search`, `remember`, `recall`.
- **Namespace-pinned:** the tool schemas expose no namespace parameter. An external agent cannot ask for another agent's lane — enforcement lives in the plumbing, not in trust.
- No CLI. `python -m rmbr` is a launch shim for MCP clients only.

## What rmbr deliberately does not do

- **Call LLMs.** It returns text; the caller prompts whatever model they use. Zero LLM integrations, forever.
- **Run a server** (beyond opt-in MCP stdio for external agents).
- **Phone home.** No telemetry, no accounts.
- **Integrate hundreds of providers.** ~4 embedding providers max.

## Performance claims policy

No performance number is published unless produced by a script in `bench/`: reproducible, recall pinned, vs Chroma and LanceDB on identical hardware. Two scripts now, not one: `bench/latency.py` (single-call latency with the real default embedder — the headline number, since it matches how rmbr is actually used) and `bench/run.py` (bulk-ingest throughput/recall on precomputed vectors — disclosed but not led with; that's not the workload rmbr optimizes for). See README.md's Performance and Alternatives sections for the current published numbers and reasoning, and `bench/pinned/` for the raw output. CI runs the test suite on ubuntu-latest, windows-latest, and macos-latest.
