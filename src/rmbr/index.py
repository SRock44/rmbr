"""Index files or raw text for hybrid (BM25 + vector) search.

    from rmbr import Index

    idx = Index("agents.db")
    idx.add_files("docs/")
    hits = idx.search("how do I deploy?", k=5)
    hits[0].text, hits[0].score, hits.timings

`Index` and `Memory` (memory.py) share the same `.db` file, the same
embedder, and the same `hybrid_search()` — `Index` just operates on
`documents`/`chunks` instead of `memories`. Both can be opened against the
same path at once; SQLite's WAL mode allows concurrent readers with one
writer at a time.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ._engine import (
    check_ann_consistency,
    load_ann_index,
    make_embedder,
    resolve_readable_namespaces,
    resolve_writable_namespace,
    save_ann_index,
)
from .ann import AnnIndex
from .chunk import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_json,
    split_markdown,
    split_python,
    split_rst,
    split_text,
)
from .embed import Embedder
from .policy import Policy
from .rerank import CrossEncoderReranker
from .search import DEFAULT_RECENCY_HALF_LIFE_SECONDS, Hits, hybrid_search
from .store import Store
from .tools import ToolSpec, index_search_tool

_COLLECTION = "chunks"
_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".html", ".css",
}

Splitter = Callable[[str, int, int], list[str]]

_SPLITTERS: dict[str, Splitter] = {
    "text": split_text,
    "markdown": split_markdown,
    "python": split_python,
    "json": split_json,
    "rst": split_rst,
}
_EXTENSION_SPLITTERS: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".py": "python",
    ".json": "json",
    ".rst": "rst",
}


def _resolve_splitter(splitter: str | Splitter) -> Splitter:
    if callable(splitter):
        return splitter
    try:
        return _SPLITTERS[splitter]
    except KeyError:
        raise ValueError(
            f"Unknown splitter {splitter!r}; choose one of {sorted(_SPLITTERS)} or pass a callable"
        ) from None


class IngestResult(list):
    """A list[int] of document ids, with a `.timings` dict attached.

    The same transparency `Hits` gives you for search, applied to
    ingestion: `chunk_ms`/`embed_ms`/`store_ms`/`ann_ms`/`total_ms` show
    where the time actually went, and `docs_per_second` gives you
    throughput. For any real corpus, `embed_ms` dominates — this exists
    so you can see that for yourself instead of taking our word for it.
    """

    timings: dict[str, float]


class Index:
    """A namespace-scoped handle onto one `.db` file's `documents`/`chunks` tables."""

    def __init__(
        self,
        path: str,
        *,
        namespace: str = "default",
        policy: Policy | None = None,
        embedder: Embedder | None = None,
    ):
        self.namespace = namespace
        self.policy = policy or Policy.strict()
        self._store = Store(path)
        self._embedder = make_embedder(embedder, self._store)
        self._ann = load_ann_index(self._store, _COLLECTION)
        self._alock = asyncio.Lock()
        self._reranker: CrossEncoderReranker | None = None
        self._deferred_save = False

    def _save_ann(self) -> None:
        if not self._deferred_save and self._ann is not None:
            save_ann_index(self._store, _COLLECTION, self._ann)

    @contextmanager
    def bulk(self) -> Iterator[None]:
        """Defer the vector-index write until this block exits, instead of
        after every `add_text()`/`add_files()`/`delete()` inside it.

        Same tradeoff and same reasoning as `Memory.bulk()` — see there
        for the full explanation. In short: SQL rows stay immediately
        committed; only the vector index's full re-serialize-and-write is
        deferred to one save at the end, which matters once a namespace's
        index has grown into the tens of thousands and every write would
        otherwise pay to reserialize the entire thing.
        """
        already_deferred = self._deferred_save
        self._deferred_save = True
        try:
            yield
        finally:
            self._deferred_save = already_deferred
            self._save_ann()

    def add_text(
        self,
        text: str,
        *,
        source: str | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        splitter: str | Splitter = "text",
    ) -> int:
        """Index a single piece of text as one document. Returns the document id.

        `metadata` is attached to every chunk this document splits into,
        and is what `search(where=...)` filters against. `splitter` is
        `"text"` (default), `"markdown"`, `"python"`, or your own
        `fn(text, chunk_size, chunk_overlap) -> list[str]`.
        """
        result = self.add_texts(
            [text],
            sources=[source],
            namespace=namespace,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter=splitter,
        )
        return result[0]

    def add_texts(
        self,
        texts: list[str],
        *,
        sources: list[str | None] | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        splitter: str | Splitter = "text",
    ) -> IngestResult:
        """Index multiple standalone texts in one call. Returns document ids
        (an `IngestResult` — a list with a `.timings` breakdown attached).

        True batch ingestion, not a loop: every document is chunked, then
        every chunk across the *whole* batch is embedded in one call to
        the embedder, then every vector is added to the ANN index in one
        call — not once per document. Batching the SQLite transaction
        alone was worth ~3x on its own (see `Store.transaction()`);
        batching the embed and ANN-insert calls on top of that closes the
        rest of the per-document-loop overhead. See `bench/` and
        README.md's Performance section for measured numbers.
        """
        if not texts:
            result = IngestResult()
            result.timings = {}
            return result

        sources = sources or [None] * len(texts)
        split = _resolve_splitter(splitter)
        timings: dict[str, float] = {}
        start = time.perf_counter()

        t0 = time.perf_counter()
        target_namespace = resolve_writable_namespace(self.policy, self.namespace, namespace)
        per_doc_chunks: list[list[str]] = []
        all_chunk_texts: list[str] = []
        for text in texts:
            chunks = split(text, chunk_size, chunk_overlap)
            if not chunks:
                raise ValueError("got no text to index (empty after stripping)")
            per_doc_chunks.append(chunks)
            all_chunk_texts.extend(chunks)
        timings["chunk_ms"] = _ms_since(t0)

        t0 = time.perf_counter()
        all_vectors = self._embedder.embed(all_chunk_texts)
        timings["embed_ms"] = _ms_since(t0)

        document_ids: list[int] = []
        all_chunk_ids: list[int] = []
        with self._store.transaction():
            t0 = time.perf_counter()
            for text, source, chunks in zip(texts, sources, per_doc_chunks):
                document_id = self._store.insert_document(target_namespace, source=source, metadata=metadata)
                # Every chunk inherits the document's metadata (v0.1 has no
                # per-chunk metadata) so `search(where=...)` can actually
                # filter by what the caller passed in.
                chunk_metadatas = [metadata or {} for _ in chunks]
                chunk_ids = self._store.insert_chunks(document_id, target_namespace, chunks, chunk_metadatas)
                document_ids.append(document_id)
                all_chunk_ids.extend(chunk_ids)
            timings["store_ms"] = _ms_since(t0)

            t0 = time.perf_counter()
            if self._ann is None:
                self._ann = AnnIndex(dim=len(all_vectors[0]))
            self._ann.add(all_chunk_ids, all_vectors)
            self._save_ann()
            self._store.clear_query_cache()
            timings["ann_ms"] = _ms_since(t0)

        total_seconds = time.perf_counter() - start
        timings["total_ms"] = total_seconds * 1000
        timings["docs_per_second"] = len(texts) / total_seconds if total_seconds > 0 else 0.0

        result = IngestResult(document_ids)
        result.timings = timings
        return result

    def add_files(
        self,
        path: str,
        *,
        pattern: str = "**/*",
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        encoding: str = "utf-8",
        splitter: str | Splitter = "auto",
    ) -> IngestResult:
        """Index every readable text file under `path` (or a single file).
        Returns document ids (an `IngestResult` — a list with a `.timings`
        breakdown attached, summed across every batch below).

        `splitter="auto"` (the default) picks a splitter per file by
        extension: `.py` gets the AST-aware Python splitter (a chunk never
        cuts a function or class in half), `.md`/`.markdown` get the
        markdown splitter, everything else gets the plain text splitter.
        Pass an explicit splitter name or callable to force the same one
        for every file. Files are grouped by their resolved splitter and
        each group is ingested as one batch via `add_texts()`.

        Only recognized text extensions are read (see `_TEXT_SUFFIXES`);
        anything else is skipped rather than raising, since walking a real
        docs/ directory usually turns up a few images or lockfiles you
        didn't mean to index.
        """
        groups: dict[Any, tuple[list[str], list[str]]] = {}
        for file_path in _iter_text_files(Path(path), pattern):
            text = file_path.read_text(encoding=encoding, errors="ignore")
            if not text.strip():
                continue
            resolved = splitter
            if splitter == "auto":
                resolved = _EXTENSION_SPLITTERS.get(file_path.suffix.lower(), "text")
            group_texts, group_sources = groups.setdefault(resolved, ([], []))
            group_texts.append(text)
            group_sources.append(str(file_path))

        document_ids = IngestResult()
        merged_timings: dict[str, float] = {}
        for resolved_splitter, (group_texts, group_sources) in groups.items():
            batch = self.add_texts(
                group_texts,
                sources=group_sources,
                namespace=namespace,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                splitter=resolved_splitter,
            )
            document_ids.extend(batch)
            for key, value in batch.timings.items():
                if key == "docs_per_second":
                    continue
                merged_timings[key] = merged_timings.get(key, 0.0) + value

        total_ms = merged_timings.get("total_ms", 0.0)
        merged_timings["docs_per_second"] = len(document_ids) / (total_ms / 1000) if total_ms > 0 else 0.0
        document_ids.timings = merged_timings
        return document_ids

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        namespaces: str | list[str] | None = None,
        where: dict[str, Any] | None = None,
        use_bm25: bool = True,
        use_vectors: bool = True,
        budget_ms: float | None = None,
        min_similarity: float | None = None,
        recency_weight: float = 0.0,
        recency_half_life_seconds: float = DEFAULT_RECENCY_HALF_LIFE_SECONDS,
        rerank: bool = False,
    ) -> Hits:
        """Hybrid search over indexed chunks. Returns Hits — hits[0].text, hits[0].score, hits.timings.

        Hybrid (both signals, the default) is what you want for real
        queries. Set `use_bm25=False` for pure semantic search, or
        `use_vectors=False` for pure keyword search.

        `where` filters to chunks whose metadata matches every key/value
        given, e.g. `idx.search(q, where={"source": "docs/deploy.md"})`.
        A plain value is equality; a `{"$gt": ...}`-style dict is an
        operator comparison (`$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte`/`$in`/
        `$nin`) — see `search.hybrid_search`'s docstring for the full set.

        `min_similarity` drops hits below that raw cosine similarity
        (requires `use_vectors=True`) — a real confidence gate, unlike
        thresholding on `hit.score` itself.

        `recency_weight` (0.0, off, by default) blends a recency bonus
        into ranking so a freshly-added chunk can outrank an equally
        relevant older one — `recency_half_life_seconds` controls how
        fast that bonus decays (default: a week). A chunk's "created"
        time is its document's `add_text()`/`add_files()` ingestion time
        (chunks aren't independently re-added later, so there's no
        separate per-chunk timestamp to track).

        `rerank=True` re-scores the candidate pool with a local
        cross-encoder (`rerank.CrossEncoderReranker`, lazily loaded on
        first use) for higher precision at extra latency; `hit.score`
        becomes the cross-encoder's score when this is on.
        """
        readable = resolve_readable_namespaces(
            self.policy, self.namespace, namespaces, self._store.list_chunk_namespaces
        )
        return hybrid_search(
            query=query,
            namespaces=readable,
            k=k,
            fts_search=self._store.search_chunks_fts,
            fetch_records=self._store.get_chunks,
            ann_index=self._ann,
            embedder=self._embedder,
            use_bm25=use_bm25,
            use_vectors=use_vectors,
            where=where,
            budget_ms=budget_ms,
            query_cache=self._store,
            min_similarity=min_similarity,
            recency_weight=recency_weight,
            recency_half_life_seconds=recency_half_life_seconds,
            record_timestamp=(lambda record: record.added_at) if recency_weight else None,
            reranker=self._get_reranker() if rerank else None,
        )

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def as_tool(self, *, name: str = "search") -> ToolSpec:
        """A `ToolSpec` for this Index's `search()` — ready for a hand-rolled
        agent loop that isn't using MCP.

            tool = idx.as_tool()
            response = client.messages.create(..., tools=[tool.to_anthropic()])
            # when the model calls it:
            result = tool.call(**tool_use_block.input)

        Same search this instance already does — namespace/policy scoping
        is whatever this `Index` was constructed with, same as every other
        call on it. `serve_mcp()` exposes the equivalent tool over MCP if
        you want a subprocess-based client instead of wiring this into
        your own loop.
        """
        return index_search_tool(self, name=name)

    def as_langchain_retriever(self, *, k: int = 5, **search_kwargs: Any) -> Any:
        """Wrap this Index as a LangChain `BaseRetriever`. Requires `langchain-core`
        (`pip install langchain-core`, or whatever full LangChain distribution
        you're already using) — imported lazily, not a hard rmbr dependency.

            retriever = idx.as_langchain_retriever(k=5)
            retriever.invoke("how do I deploy?")   # -> list[Document]

        Extra `search_kwargs` (`where=`, `min_similarity=`, `rerank=`, ...)
        are passed straight through to `search()`/`asearch()` on every call.
        """
        from .integrations.langchain import as_retriever

        return as_retriever(self, k=k, **search_kwargs)

    def as_llamaindex_retriever(self, *, k: int = 5, **search_kwargs: Any) -> Any:
        """Wrap this Index as a LlamaIndex `BaseRetriever`. Requires
        `llama-index-core` — imported lazily, not a hard rmbr dependency.

            retriever = idx.as_llamaindex_retriever(k=5)
            retriever.retrieve("how do I deploy?")   # -> list[NodeWithScore]

        Extra `search_kwargs` are passed straight through to `search()`/
        `asearch()` on every call, same as `as_langchain_retriever()`.
        """
        from .integrations.llamaindex import as_retriever

        return as_retriever(self, k=k, **search_kwargs)

    def delete(self, document_id: int) -> None:
        """Delete a document, its chunks, and their vectors. No-op if it doesn't exist."""
        document = self._store.get_document(document_id)
        if document is None:
            return
        if document.namespace != self.namespace and not self.policy.can_write(
            self.namespace, document.namespace
        ):
            raise PermissionError(
                f"{self.namespace!r} is not allowed to delete from namespace {document.namespace!r}"
            )
        chunk_ids = self._store.get_chunk_ids_for_document(document_id)
        with self._store.transaction():
            self._store.delete_document(document_id)
            if self._ann is not None and chunk_ids:
                self._ann.remove(chunk_ids)
                self._save_ann()
            self._store.clear_query_cache()

    def stats(self, namespaces: str | list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Document/chunk counts and time range per namespace —
        `{namespace: {documents, chunks, oldest, newest}}`.

        Same `namespaces=` semantics as `search()` (default: just this
        handle's own namespace; `"*"` for every namespace the policy lets
        it read; an explicit name/list is policy-checked).
        """
        readable = resolve_readable_namespaces(
            self.policy, self.namespace, namespaces, self._store.list_chunk_namespaces
        )
        return {ns: self._store.chunk_stats(ns) for ns in readable}

    def integrity_check(self) -> list[str]:
        """Verify the vector index and the SQLite table agree on which
        chunks exist, across every namespace in the file. Returns a list
        of problems (empty means healthy) — see `_engine.check_ann_consistency`
        for what a mismatch would actually mean and why it'd be surprising.
        """
        return check_ann_consistency(self._ann, self._store.all_chunk_ids())

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- async surface -----------------------------------------------------
    #
    # Thin wrappers: the underlying work (SQLite I/O, ONNX inference) runs
    # in a thread pool via asyncio.to_thread so it doesn't block the event
    # loop, guarded by a per-instance lock. The lock is deliberately
    # coarse — it serializes every async call against this one Index
    # instance, reads included — because usearch's C extension isn't
    # documented as safe for concurrent mutation from multiple threads,
    # and a crash or corrupted index is a much worse failure mode than
    # giving up some read concurrency. Open separate Index instances (they
    # share the underlying file fine under SQLite's WAL mode) if you need
    # true concurrent access.

    async def aadd_text(self, text: str, **kwargs: Any) -> int:
        async with self._alock:
            return await asyncio.to_thread(self.add_text, text, **kwargs)

    async def aadd_texts(self, texts: list[str], **kwargs: Any) -> IngestResult:
        async with self._alock:
            return await asyncio.to_thread(self.add_texts, texts, **kwargs)

    async def aadd_files(self, path: str, **kwargs: Any) -> IngestResult:
        async with self._alock:
            return await asyncio.to_thread(self.add_files, path, **kwargs)

    async def asearch(self, query: str, **kwargs: Any) -> Hits:
        async with self._alock:
            return await asyncio.to_thread(self.search, query, **kwargs)


def _iter_text_files(path: Path, pattern: str) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in _TEXT_SUFFIXES:
            yield path
        return
    for candidate in sorted(path.glob(pattern)):
        if candidate.is_file() and candidate.suffix.lower() in _TEXT_SUFFIXES:
            yield candidate


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
