# rmbr

> **Give your agent memory and knowledge. One file, three lines, no server, no API key.**

`rmbr` ("remember", vowels deleted) is an embedded, local-first **memory + retrieval engine for AI agents and LLM apps** — what SQLite is to Postgres, rmbr aims to be to hosted memory services.

> ⚠️ **Status: pre-release, under active development.** The design is finalized (see [docs/PLAN.md](docs/PLAN.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)); implementation of v0.1 is in progress. Nothing below is installable yet.

## Why

Agents need two things constantly: **memory** (remember things across sessions) and **retrieval** (look up relevant knowledge mid-task). Today that means signing up for a hosted memory service or assembling a vector database + embedding API + framework glue. rmbr is the embedded alternative:

- **One file.** Your agent's entire memory and knowledge base is a single `.db` file. Commit it to git, ship it in an installer, `scp` it to a server, hand it to a teammate.
- **Three lines.** `pip install rmbr`, import, remember. No account, no config, no service.
- **No server.** Runs inside your process, like SQLite.
- **No API key.** Embeddings run locally via a small ONNX model by default. Nothing phones home, ever. Cloud embedding providers (OpenAI/Voyage/Cohere) are strictly opt-in.
- **Works with every LLM.** rmbr never calls an LLM — it returns relevant text, you feed it to Claude, GPT, Gemini, or a local model. Zero model lock-in by construction.

## Planned API (v0.1)

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

## Multi-agent isolation, honestly stated

- **Namespaces** keep agents' memories separate and are enforced on every call — but they are *organizational*, not cryptographic. Any code with access to the file can open the file. That's true of every embedded database; we say it out loud.
- **Hard isolation** = separate `.db` files per trust boundary, plus OS file permissions.
- **MCP serving is namespace-pinned:** the exposed tools have no namespace parameter, so an external agent structurally cannot query outside its lane.

## Performance claims policy

**This README will never contain a performance number that isn't produced by `bench/run.py`** — reproducible by anyone, recall pinned, against named competitors on identical hardware. Until those runs are published, we make no speed claims, only design commitments: local embeddings (no network round trip), content-hash embedding cache (never embed the same text twice), semantic query cache, and in-RAM HNSW vector search.

## Roadmap

- **v0.1** — `Memory` + `Policy` + `Index` (hybrid search), embedding + semantic caches, MCP support, 3-OS CI (Linux/Windows/macOS), benchmark harness vs Chroma and LanceDB
- **Later** — async API surface expansion, more chunkers, additional embedding providers

## License

[MIT](LICENSE)
