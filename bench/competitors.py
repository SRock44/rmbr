"""Competitor engines for the bench harness: chromadb, lancedb, and mem0.

All three are optional (`pip install rmbr[bench]`). Each function here
ingests the exact same precomputed vectors and texts that `run.py` feeds
rmbr, and runs the exact same query vectors — see datasets.py for why
precomputed vectors instead of a real embedding model. Import errors are
left to propagate; `run.py` decides whether a missing competitor is fatal
or skippable.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from corpus import Corpus


def run_chromadb(corpus: Corpus, k: int, workdir: Path) -> tuple[float, list[float], list[list[int]]]:
    """Returns (ingest_seconds, per_query_latency_ms, predicted_top_k_doc_indices)."""
    import chromadb

    db_path = workdir / "chromadb"
    if db_path.exists():
        shutil.rmtree(db_path)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.create_collection("bench")

    ids = [str(i) for i in range(len(corpus.doc_texts))]
    t0 = time.perf_counter()
    collection.add(ids=ids, embeddings=corpus.doc_vectors.tolist(), documents=corpus.doc_texts)
    ingest_seconds = time.perf_counter() - t0

    latencies_ms: list[float] = []
    predicted: list[list[int]] = []
    for query_vector in corpus.query_vectors:
        t0 = time.perf_counter()
        result = collection.query(query_embeddings=[query_vector.tolist()], n_results=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        predicted.append([int(doc_id) for doc_id in result["ids"][0]])

    return ingest_seconds, latencies_ms, predicted


def run_lancedb(corpus: Corpus, k: int, workdir: Path) -> tuple[float, list[float], list[list[int]]]:
    """Returns (ingest_seconds, per_query_latency_ms, predicted_top_k_doc_indices)."""
    import lancedb

    db_path = workdir / "lancedb"
    db = lancedb.connect(str(db_path))

    data = [
        {"id": i, "vector": corpus.doc_vectors[i], "text": text} for i, text in enumerate(corpus.doc_texts)
    ]
    t0 = time.perf_counter()
    table = db.create_table("bench", data=data, mode="overwrite")
    ingest_seconds = time.perf_counter() - t0

    latencies_ms: list[float] = []
    predicted: list[list[int]] = []
    for query_vector in corpus.query_vectors:
        t0 = time.perf_counter()
        rows = table.search(query_vector).limit(k).to_list()
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        predicted.append([int(row["id"]) for row in rows])

    return ingest_seconds, latencies_ms, predicted


def run_mem0(corpus: Corpus, k: int, workdir: Path) -> tuple[float, list[float], list[list[int]]]:
    """Returns (ingest_seconds, per_query_latency_ms, predicted_top_k_doc_indices).

    mem0 OSS's local `Memory`, configured to run fully offline: Qdrant's
    embedded/on-disk mode (no server, no network), a fake OpenAI key on the
    LLM slot (constructing an `OpenAI` client never validates or calls out —
    and it's never invoked anyway, since `add(..., infer=False)` is mem0's
    own "skip LLM fact-extraction" mode), and its embedder swapped out
    *after* construction for the same precomputed-vector lookup every other
    engine in this file gets (`Memory.embedding_model` is a plain instance
    attribute mem0 itself reads from on every embed call — not a hack,
    just not exposed as constructor config).

    **Not disabled: mem0's own default hybrid (dense + BM25 sparse) search.**
    Unlike chromadb/lancedb (vector-only databases, benched as such), mem0
    does real hybrid search out of the box — same category of choice rmbr
    itself makes by default. Reporting mem0 with that turned off would
    misrepresent how anyone actually gets it by installing it; this
    benches mem0 as itself, the same principle behind rmbr's own
    "hybrid, default" row.

    The BM25 sparse encoder (a second, separate fastembed model) lazy-loads
    on first use — downloaded outside the timed block below via one
    throwaway warmup call, same reasoning as this project's own
    `rerank=True` being flagged as a one-time cost, not steady-state
    latency (see README's Performance section).
    """
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.embeddings.base import EmbeddingBase
    from mem0.embeddings.configs import EmbedderConfig
    from mem0.llms.configs import LlmConfig
    from mem0.vector_stores.configs import VectorStoreConfig

    dim = corpus.doc_vectors.shape[1]
    db_path = workdir / "mem0"
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True)

    config = MemoryConfig(
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={"collection_name": "bench", "embedding_model_dims": dim, "path": str(db_path / "qdrant")},
        ),
        llm=LlmConfig(provider="openai", config={"api_key": "sk-bench-unused"}),
        embedder=EmbedderConfig(provider="openai", config={"api_key": "sk-bench-unused"}),
        history_db_path=str(db_path / "history.db"),
    )
    memory = Memory(config)

    class _PrecomputedEmbedder(EmbeddingBase):
        def __init__(self, vectors_by_text: dict, dims: int):
            super().__init__(BaseEmbedderConfig(embedding_dims=dims))
            self._vectors = vectors_by_text

        def embed(self, text, memory_action=None):
            return self._vectors[text].tolist()

    query_texts = [f"mem0 bench query {i}" for i in range(len(corpus.query_vectors))]
    vectors_by_text = dict(zip(corpus.doc_texts, corpus.doc_vectors))
    vectors_by_text.update(zip(query_texts, corpus.query_vectors))
    memory.embedding_model = _PrecomputedEmbedder(vectors_by_text, dim)

    memory.vector_store._get_bm25_encoder()  # warm up before timing - see docstring

    messages = [{"role": "user", "content": text} for text in corpus.doc_texts]
    t0 = time.perf_counter()
    add_result = memory.add(messages, user_id="bench", infer=False)
    ingest_seconds = time.perf_counter() - t0

    doc_index_by_mem_id = {r["id"]: i for i, r in enumerate(add_result["results"])}

    latencies_ms: list[float] = []
    predicted: list[list[int]] = []
    for query_text in query_texts:
        t0 = time.perf_counter()
        result = memory.search(query_text, top_k=k, filters={"user_id": "bench"})
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        predicted.append([doc_index_by_mem_id[r["id"]] for r in result["results"] if r["id"] in doc_index_by_mem_id])

    return ingest_seconds, latencies_ms, predicted
