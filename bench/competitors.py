"""Competitor engines for the bench harness: chromadb and lancedb.

Both are optional (`pip install rmbr[bench]`). Each function here ingests
the exact same precomputed vectors and texts that `run.py` feeds rmbr, and
runs the exact same query vectors — see datasets.py for why precomputed
vectors instead of a real embedding model. Import errors are left to
propagate; `run.py` decides whether a missing competitor is fatal or
skippable.
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
