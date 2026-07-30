# rmbr

> **Give your agent memory and knowledge. One file, three lines, no server, no API key.**

`rmbr` ("remember", vowels deleted) is an embedded, local-first **memory + retrieval engine for AI agents and LLM apps** — what SQLite is to Postgres, rmbr aims to be to hosted memory services.

> ⚠️ **Status: pre-release.** `Memory`, `Index`, `Policy`, and MCP support (below) are implemented and tested in this repo — see [docs/PLAN.md](docs/PLAN.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design. Not yet published to PyPI as a working release (the `rmbr` package there today is a name-holding stub) — watch this repo for the tagged v0.1.0 release.

## Why

Agents need two things constantly: **memory** (remember things across sessions) and **retrieval** (look up relevant knowledge mid-task). Today that means signing up for a hosted memory service or assembling a vector database + embedding API + framework glue. rmbr is the embedded alternative:

- **One file.** Your agent's entire memory and knowledge base is a single `.db` file. Commit it to git, ship it in an installer, `scp` it to a server, hand it to a teammate.
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
pytest tests/    # 93 tests, no network or API key required
```

The default embedder (`fastembed`, a local ONNX model) downloads its model weights on first use. Every test in `tests/` instead uses `rmbr.embed.FakeEmbedder` — a deterministic, dependency-free embedder — so the suite runs fully offline; you can inject the same `FakeEmbedder` into your own tests via `Memory(..., embedder=FakeEmbedder())` / `Index(..., embedder=FakeEmbedder())`.

## Multi-agent isolation, honestly stated

- **Namespaces** keep agents' memories separate and are enforced on every call — but they are *organizational*, not cryptographic. Any code with access to the file can open the file. That's true of every embedded database; we say it out loud.
- **Hard isolation** = separate `.db` files per trust boundary, plus OS file permissions.
- **MCP serving is namespace-pinned:** the exposed tools have no namespace parameter, so an external agent structurally cannot query outside its lane.

## Performance claims policy

**This README will never contain a performance number that isn't produced by `bench/run.py`** — reproducible by anyone, recall pinned, against named competitors (Chroma, LanceDB) on identical hardware. The harness exists (`bench/`) and runs end-to-end today, but a number from a laptop is directional, not a claim — publishable numbers come from a run on the project's pinned Linux benchmark machine, not yet done. Until then: no speed claims, only design commitments — local embeddings (no network round trip), a content-hash embedding cache (never embed the same text twice), a semantic query cache, and in-RAM HNSW-family vector search (`usearch`).

Want to run it yourself? `pip install -e ".[bench]" && python bench/run.py` — every engine is fed identical precomputed vectors so the comparison is apples-to-apples (see `bench/run.py`'s module docstring for the full methodology).

## Roadmap

- **v0.1 (done in this repo, not yet released to PyPI)** — `Memory` + `Policy` + `Index` (hybrid BM25 + vector search), embedding + semantic query caches, MCP support (namespace-pinned), 3-OS CI (Linux/Windows/macOS), benchmark harness vs Chroma and LanceDB
- **Before the v0.1.0 tag** — publish real benchmark numbers from the pinned Linux machine, set up PyPI trusted publishing
- **Later** — async API surface, more chunkers, additional embedding providers

## License

[MIT](LICENSE)
