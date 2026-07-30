"""Synthetic corpus + ground truth for the bench harness.

The bench measures *engine* performance (storage + ANN search), not
embedding-model quality — those are separate concerns and conflating them
would make the numbers unreproducible (they'd depend on whichever model
happened to be configured). So every engine under test is fed the exact
same precomputed vectors for the exact same synthetic documents, via
rmbr's own `Embedder` protocol (see `PrecomputedEmbedder` in run.py) for
rmbr, and directly for chromadb/lancedb.

Ground truth for recall@k comes from an exact brute-force numpy cosine
search over the same vectors — that's the "correct" answer every engine's
approximate (HNSW-family) index is compared against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Corpus:
    doc_texts: list[str]
    doc_vectors: np.ndarray  # (n_docs, dim), unit-normalized float32
    query_vectors: np.ndarray  # (n_queries, dim), unit-normalized float32
    ground_truth: list[list[int]]  # ground_truth[q] = true top-k doc indices for query q


def generate_corpus(n_docs: int, n_queries: int, dim: int, k: int, seed: int = 0) -> Corpus:
    rng = np.random.default_rng(seed)

    doc_vectors = _unit_normalize(rng.standard_normal((n_docs, dim)).astype(np.float32))
    doc_texts = [_synthetic_document(i) for i in range(n_docs)]

    # Queries are corpus vectors plus noise: not identical (which every
    # engine would trivially ace) but not arbitrary either, so recall@k
    # measures real approximation quality rather than a coin flip.
    query_source = rng.choice(n_docs, size=n_queries, replace=n_queries > n_docs)
    noise = rng.standard_normal((n_queries, dim)).astype(np.float32) * 0.1
    query_vectors = _unit_normalize(doc_vectors[query_source] + noise)

    ground_truth = _brute_force_top_k(doc_vectors, query_vectors, k)

    return Corpus(doc_texts, doc_vectors, query_vectors, ground_truth)


def recall_at_k(predicted: list[list[int]], ground_truth: list[list[int]]) -> float:
    """Mean fraction of each query's true top-k that appear in its predicted top-k."""
    scores = []
    for pred, truth in zip(predicted, ground_truth):
        if not truth:
            continue
        scores.append(len(set(pred) & set(truth)) / len(truth))
    return sum(scores) / len(scores) if scores else 0.0


def _brute_force_top_k(doc_vectors: np.ndarray, query_vectors: np.ndarray, k: int) -> list[list[int]]:
    similarities = query_vectors @ doc_vectors.T  # vectors are unit-normalized, so dot product == cosine
    top_k = np.argsort(-similarities, axis=1)[:, :k]
    return top_k.tolist()


def _unit_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def _synthetic_document(i: int) -> str:
    # Content only needs to be unique per row (it's the dict key
    # PrecomputedEmbedder looks vectors up by) — the words themselves
    # carry no semantic weight since search uses the precomputed vectors,
    # not a real embedding of this text.
    return f"synthetic benchmark document number {i} filler filler filler"
