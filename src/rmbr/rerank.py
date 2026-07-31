"""Optional cross-encoder reranking — a local, zero-key precision lever.

BM25 and vector search each score a query and a document independently,
then compare; a cross-encoder scores the (query, document) *pair* jointly,
which is slower per item but more accurate at judging "is this document
actually relevant to this query" — the standard second-stage precision fix
in real search systems. `search(..., rerank=True)` costs extra local
compute against an already-fetched candidate pool, not a new vendor: same
`fastembed` dependency rmbr already ships for the default embedder, no new
network call, no API key.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Local ONNX cross-encoder via fastembed.

    Lazy-loaded: the model only downloads/loads on the first actual
    `rerank()` call, not at construction — so `Memory`/`Index` can always
    hand out a reranker instance without cost for callers who never pass
    `rerank=True`.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return list(self._model.rerank(query, documents))
