# rmbr

> **Give your agent memory and knowledge. One file, three lines, no server, no API key.**

`rmbr` ("remember", vowels deleted) is an embedded, local-first **memory + retrieval engine for AI agents and LLM apps** — what SQLite is to Postgres, rmbr aims to be to hosted memory services.

> ⚠️ **Status: pre-release.** `Memory`, `Index`, `Policy`, and MCP support (below) are implemented and tested in this repo — see [docs/PLAN.md](docs/PLAN.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design. Not yet published to PyPI as a working release (the `rmbr` package there today is a name-holding stub) — watch this repo for the tagged v0.1.0 release.

## Why

Every agent memory tool on the market assumes you'll run infrastructure for it: a Docker container, a graph database, a hosted API with a key. rmbr assumes the opposite — that memory should ship *inside* whatever you're already building, the same way SQLite ships inside a mobile app instead of requiring a database server. That difference in assumption is what makes these use cases possible:

- **A persistent identity across restarts.** A coding agent that works on your repo over weeks, across dozens of separate sessions, remembers your conventions, past decisions, and what's already failed — not because you re-explain it every time, but because its memory is a file that survives the process exiting.
- **Coordination for a team of agents that don't fully trust each other.** `Policy` gives a supervisor and its specialist agents each their own private memory *and* a shared knowledge base, with cross-namespace access explicitly granted rather than accidental — the substrate for a real multi-agent system, not a single shared vector index everyone can read and write.
- **A memory feature you can ship inside a product**, not bolt onto infrastructure. A desktop app, a CLI tool, an offline-capable mobile backend — anywhere you can't assume a vector database and an API key are available, you can still assume a `.db` file is.
- **A portable, versionable agent brain.** Because it's one file: `git commit` it, diff it, branch it to try something risky, roll it back, attach the exact file to a bug report ("here's what the agent knew when this went wrong"), or check a known-good state into a test fixture for deterministic CI.
- **Fully offline.** Edge devices, field deployments, air-gapped environments — nothing in the core memory/retrieval path needs a network call, ever, by default.

Concretely, rmbr gives you:

- **One file.** Your agent's entire memory and knowledge base is a single `.db` file.
- **Three lines.** `pip install rmbr`, import, remember. No account, no config, no service.
- **No server.** Runs inside your process, like SQLite.
- **No API key.** Embeddings run locally via a small ONNX model by default. Nothing phones home, ever. Cloud embedding providers (OpenAI/Voyage/Cohere) are strictly opt-in.
- **Works with every LLM.** rmbr never calls an LLM — it returns relevant text, you feed it to Claude, GPT, Gemini, or a local model. Zero model lock-in by construction.

## API (v0.1)

```python
from rmbr import Memory, Index, Policy, serve_mcp

# Agent memory — the headline
mem = Memory("agents.db", namespace="researcher")
mem.remember("user prefers dark mode and short answers")
mem.recall("user preferences")                      # relevant memories, fast

# Multi-agent access control — deny-by-default
policy = Policy()
policy.allow("supervisor", read="*")                # supervisor sees all lanes
mem = Memory("agents.db", namespace="coder", policy=policy)

# Knowledge / RAG — same engine
idx = Index("agents.db")
idx.add_files("docs/")
hits = idx.search("how do I deploy?", k=5)          # hybrid BM25 + vector search
hits[0].text, hits[0].score, hits.timings           # per-stage latency, always visible

# Expose memory to external agents (Claude Code, Cursor, any MCP client)
serve_mcp("agents.db", namespace="coder", read_only=True)
```

Library-only by design — no CLI to learn. (`python -m rmbr` exists solely so MCP clients can launch the server.)

### Try it from source

Not on PyPI as a working release yet, but every line above runs today against this repo:

```bash
git clone https://github.com/SRock44/rmbr.git
cd rmbr
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install --only-binary :all: -e .
pytest tests/    # 104 tests, no network or API key required
```

The default embedder (`fastembed`, a local ONNX model) downloads its model weights on first use. Every test in `tests/` instead uses `rmbr.embed.FakeEmbedder` — a deterministic, dependency-free embedder — so the suite runs fully offline; you can inject the same `FakeEmbedder` into your own tests via `Memory(..., embedder=FakeEmbedder())` / `Index(..., embedder=FakeEmbedder())`.

## Multi-agent isolation, honestly stated

- **Namespaces** keep agents' memories separate and are enforced on every call — but they are *organizational*, not cryptographic. Any code with access to the file can open the file. That's true of every embedded database; we say it out loud.
- **Hard isolation** = separate `.db` files per trust boundary, plus OS file permissions.
- **MCP serving is namespace-pinned:** the exposed tools have no namespace parameter, so an external agent structurally cannot query outside its lane — unlike every other MCP memory server we looked at, where the scope is a parameter the calling model supplies (and could be talked into changing).

## Alternatives

Not "competitors" — genuinely different tools for genuinely different jobs. Here's where each one actually fits, including where rmbr *isn't* the right choice.

**If you're evaluating a memory service** (mem0, Zep/Graphiti, Letta): all three are excellent at LLM-mediated memory intelligence — extracting facts from conversation, resolving contradictions, consolidating duplicates. rmbr deliberately does none of that; it never calls an LLM, full stop. That's a real capability gap, not spin — but it's also why rmbr has no API key requirement, no extra LLM cost or latency on every `remember()`, and no risk of a consolidation model quietly rewriting what you actually said. You get the primitives (`remember`/`recall`/`forget`, namespace policy); you decide what, if anything, sits on top.

| | mem0 | Zep / Graphiti | Letta | rmbr |
|---|---|---|---|---|
| Deployment | SDK, but calls a hosted LLM + embedding API by default | Docker + Neo4j/FalkorDB + an LLM API | A server (Docker) + Postgres | Embedded — one file, your process |
| API key required out of the box | Yes (OpenAI) | Yes (LLM for graph extraction) | Yes (LLM) | No |
| Decides what's worth remembering | An LLM (fact extraction) | An LLM (graph edges, contradiction resolution) | An LLM (self-editing memory blocks) | You do — deterministic, no LLM in the write path |
| State is a portable file | No | No | No | Yes |

(GitHub stars as of this writing, for scale: mem0 ~62k, Graphiti ~29k, Letta ~24k. This is a much larger, faster-moving category than rmbr is part of — worth knowing going in.)

**If you're evaluating a vector database** (Chroma, LanceDB, pgvector, Pinecone, ...): these are real peers on "embedded, no API key" — Chroma and LanceDB in particular are just as zero-server as rmbr. The difference is what's built on top of the vector index: with a raw vector database you're still building the memory API, the namespace/access-control layer, the hybrid BM25+vector fusion, the embedding cache, and an MCP server yourself. rmbr ships all of that already assembled, specifically for the agent-memory shape of problem.

Where they legitimately win: **raw bulk-ingestion throughput at large scale.** If you're indexing millions of documents for a dedicated search product, use a purpose-built vector database — that's their job, not rmbr's. rmbr is tuned for what an agent's own memory and knowledge base actually looks like (its own history, a knowledge base in the hundreds-to-low-thousands of chunks), where single-call latency, not bulk-loading speed, is what you actually pay for on every turn. See [Performance](#performance) below for the honest numbers on both.

## Performance

**This README will never contain a performance number that isn't produced by a script in `bench/`** — reproducible by anyone, on disclosed hardware, methodology included.

The number that matters for rmbr's actual usage pattern — an agent calling `remember()`/`search()` one at a time mid-reasoning-loop, not bulk-loading a corpus — is **single-call latency with the real default embedder**, not bulk throughput. That's what's below, run on the project's pinned Ubuntu benchmark machine (Intel Core Ultra 9 285K, 4 cores isolated via `taskset -c 0-3`, Ubuntu 24.04.4 LTS, Python 3.12.3), median of 3 runs, 100 samples/run:

| operation | p50 | p95 | p99 |
|---|---:|---:|---:|
| `mem.remember(text)` | 5.8 ms | 11.1 ms | 11.5 ms |
| `idx.search(query, k=5)` against a 500-doc index | 3.2 ms | 4.1 ms | 4.2 ms |
| — of which, query embedding alone | 2.8 ms | 3.1 ms | 3.3 ms |

Read that last row carefully: **~85-90% of a search call's cost is the embedding model, not rmbr.** rmbr's own storage/retrieval overhead is sub-millisecond. And all of this is imperceptible next to the LLM call that will follow it in any real agent loop — which was rmbr's founding thesis about where RAG latency actually lives (see [docs/PLAN.md](docs/PLAN.md)). Reproduce: `python bench/latency.py`; raw output for all 3 runs is in [`bench/pinned/`](bench/pinned/).

**Bulk-ingest throughput, for full transparency (not a claim we're leading with):** rmbr batches SQLite commits per operation rather than per row — a real, measured ~3x improvement (950 → ~2,750 docs/s on a 5,000-doc synthetic corpus) — but rmbr is still slower at pure bulk loading than both purpose-built alternatives: Chroma ingests ~3x faster (~8,100 docs/s) and LanceDB ~30-70x faster (~80,000-190,000 docs/s, varied across runs), because that's a fundamentally different job (one Arrow batch write, zero per-row relational bookkeeping, in LanceDB's case) than what rmbr is built for. What rmbr does hold its own on: recall@5 (0.96) is competitive with LanceDB's exact search (1.0) and ahead of Chroma's (0.80). Full numbers, all 3 seeds, in [`bench/pinned/`](bench/pinned/) and reproducible via `pip install -e ".[bench]" && python bench/run.py`. We're disclosing this, not hiding it: if bulk document loading at scale is your actual workload, see [Alternatives](#alternatives) above — that's not what rmbr optimizes for.

## Roadmap

- **v0.1 (done in this repo, not yet released to PyPI)** — `Memory` + `Policy` + `Index` (hybrid BM25 + vector search, metadata filtering), embedding + semantic query caches, MCP support (namespace-pinned), 3-OS CI (Linux/Windows/macOS), batched-transaction bulk writes, real single-call and bulk benchmark numbers, PyPI trusted publishing
- **Known gaps** — true batch ingestion (one embedder call + one ANN insert per `add_texts()` batch instead of per-document, to close more of the bulk-throughput gap for anyone who does need it), async API surface, more chunkers, additional embedding providers
- **Next** — cut the v0.1.0 release

## License

[MIT](LICENSE)
