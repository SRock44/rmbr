"""LlamaIndex retriever adapter — thin, lazily-imported, no hard dependency.

Wraps an `Index` as a `llama_index.core.retrievers.BaseRetriever` so it
drops into any LlamaIndex query engine/agent that expects one, without
hand-writing the glue. `llama_index.core` is only imported inside
`as_retriever()` — importing `rmbr` itself never touches it.

Verified against `llama-index-core==0.14.23`'s `BaseRetriever`
(`_retrieve`/`_aretrieve` returning `list[NodeWithScore]`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..index import Index


def as_retriever(index: Index, *, k: int = 5, **search_kwargs: Any) -> Any:
    """Wrap `index` as a LlamaIndex `BaseRetriever`.

        from rmbr.integrations.llamaindex import as_retriever
        retriever = as_retriever(idx, k=5)
        retriever.retrieve("how do I deploy?")        # -> list[NodeWithScore]
        await retriever.aretrieve("how do I deploy?")  # -> list[NodeWithScore]

    Prefer `idx.as_llamaindex_retriever(...)` — same thing, more discoverable.
    Extra `search_kwargs` (`where=`, `min_similarity=`, `rerank=`, ...) are
    passed straight through to `Index.search()`/`Index.asearch()` on every
    call. Each `Hit` becomes a `TextNode` (id set to the hit's rmbr id,
    `hit.metadata` carried over as-is) wrapped in a `NodeWithScore` with
    `hit.score` as the score.
    """
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    class RmbrRetriever(BaseRetriever):
        def __init__(self, rmbr_index: Index, rmbr_k: int, rmbr_search_kwargs: dict[str, Any]):
            self._rmbr_index = rmbr_index
            self._rmbr_k = rmbr_k
            self._rmbr_search_kwargs = rmbr_search_kwargs
            super().__init__()

        def _retrieve(self, query_bundle: QueryBundle) -> list:
            hits = self._rmbr_index.search(
                query_bundle.query_str, k=self._rmbr_k, **self._rmbr_search_kwargs
            )
            return [_hit_to_node_with_score(h, TextNode, NodeWithScore) for h in hits]

        async def _aretrieve(self, query_bundle: QueryBundle) -> list:
            hits = await self._rmbr_index.asearch(
                query_bundle.query_str, k=self._rmbr_k, **self._rmbr_search_kwargs
            )
            return [_hit_to_node_with_score(h, TextNode, NodeWithScore) for h in hits]

    return RmbrRetriever(rmbr_index=index, rmbr_k=k, rmbr_search_kwargs=search_kwargs)


def _hit_to_node_with_score(hit: Any, text_node_cls: Any, node_with_score_cls: Any) -> Any:
    node = text_node_cls(text=hit.text, id_=str(hit.id), metadata={**hit.metadata, "namespace": hit.namespace})
    return node_with_score_cls(node=node, score=hit.score)
