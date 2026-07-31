"""LangChain retriever adapter — thin, lazily-imported, no hard dependency.

Wraps an `Index` as a `langchain_core.retrievers.BaseRetriever` so it drops
into any LangChain chain/agent that expects one, without hand-writing the
glue. `langchain_core` (or a full LangChain distribution that includes it)
is only imported inside `as_retriever()` — importing `rmbr` itself never
touches it.

Verified against `langchain-core==1.5.3`'s pydantic-v2-based `BaseRetriever`
(`_get_relevant_documents`/`_aget_relevant_documents`); this interface has
been stable since LangChain's 0.3 line, but if you're on something much
older and it breaks, that's why.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..index import Index


def as_retriever(index: Index, *, k: int = 5, **search_kwargs: Any) -> Any:
    """Wrap `index` as a LangChain `BaseRetriever`.

        from rmbr.integrations.langchain import as_retriever
        retriever = as_retriever(idx, k=5)
        retriever.invoke("how do I deploy?")        # -> list[Document]
        await retriever.ainvoke("how do I deploy?")  # -> list[Document]

    Prefer `idx.as_langchain_retriever(...)` — same thing, more discoverable.
    Extra `search_kwargs` (`where=`, `min_similarity=`, `rerank=`, ...) are
    passed straight through to `Index.search()`/`Index.asearch()` on every
    call. Each `Hit` becomes a `Document` with `hit.text` as `page_content`
    and `hit.metadata` plus `rmbr_id`/`rmbr_score`/`namespace` as `metadata`
    — the extra fields are prefixed so they can't silently collide with
    your own metadata keys.
    """
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict

    class RmbrRetriever(BaseRetriever):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        rmbr_index: Any
        rmbr_k: int
        rmbr_search_kwargs: dict

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list:
            hits = self.rmbr_index.search(query, k=self.rmbr_k, **self.rmbr_search_kwargs)
            return [_hit_to_document(h, Document) for h in hits]

        async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list:
            hits = await self.rmbr_index.asearch(query, k=self.rmbr_k, **self.rmbr_search_kwargs)
            return [_hit_to_document(h, Document) for h in hits]

    return RmbrRetriever(rmbr_index=index, rmbr_k=k, rmbr_search_kwargs=search_kwargs)


def _hit_to_document(hit: Any, document_cls: Any) -> Any:
    metadata = {**hit.metadata, "rmbr_id": hit.id, "rmbr_score": hit.score, "namespace": hit.namespace}
    return document_cls(page_content=hit.text, metadata=metadata)
