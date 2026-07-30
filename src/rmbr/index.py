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

from pathlib import Path
from typing import Any, Iterable

from ._engine import (
    load_ann_index,
    make_embedder,
    resolve_readable_namespaces,
    resolve_writable_namespace,
    save_ann_index,
)
from .ann import AnnIndex
from .chunk import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_markdown, split_text
from .embed import Embedder
from .policy import Policy
from .search import Hits, hybrid_search
from .store import Store

_COLLECTION = "chunks"
_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".html", ".css",
}


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

    def add_text(
        self,
        text: str,
        *,
        source: str | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        markdown: bool = False,
    ) -> int:
        """Index a single piece of text as one document. Returns the document id.

        `metadata` is attached to every chunk this document splits into,
        and is what `search(where=...)` filters against.
        """
        with self._store.transaction():
            document_id = self._ingest_one(
                text,
                source=source,
                namespace=namespace,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                markdown=markdown,
            )
            save_ann_index(self._store, _COLLECTION, self._ann)
            self._store.clear_query_cache()
        return document_id

    def add_texts(
        self,
        texts: list[str],
        *,
        sources: list[str | None] | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        markdown: bool = False,
    ) -> list[int]:
        """Index multiple standalone texts in one call. Returns their document ids.

        Prefer this over calling `add_text()` in a loop for anything more
        than a handful of documents: everything below - every chunk/embed
        insert plus the final vector index save - lands in one SQLite
        transaction instead of committing per document. Measured on real
        hardware: that's a ~47x difference in ingest throughput (SQLite's
        per-transaction fsync overhead dominates at one-commit-per-row,
        not SQLite's actual throughput). See bench/ and README.md.
        """
        sources = sources or [None] * len(texts)
        with self._store.transaction():
            document_ids = [
                self._ingest_one(
                    text,
                    source=source,
                    namespace=namespace,
                    metadata=metadata,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    markdown=markdown,
                )
                for text, source in zip(texts, sources)
            ]
            if document_ids:
                save_ann_index(self._store, _COLLECTION, self._ann)
                self._store.clear_query_cache()
        return document_ids

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
    ) -> list[int]:
        """Index every readable text file under `path` (or a single file). Returns document ids.

        Only recognized text extensions are read (see `_TEXT_SUFFIXES`);
        anything else is skipped rather than raising, since walking a real
        docs/ directory usually turns up a few images or lockfiles you
        didn't mean to index.
        """
        document_ids = []
        with self._store.transaction():
            for file_path in _iter_text_files(Path(path), pattern):
                text = file_path.read_text(encoding=encoding, errors="ignore")
                if not text.strip():
                    continue
                document_ids.append(
                    self._ingest_one(
                        text,
                        source=str(file_path),
                        namespace=namespace,
                        metadata=metadata,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        markdown=file_path.suffix.lower() in _MARKDOWN_SUFFIXES,
                    )
                )
            if document_ids:
                save_ann_index(self._store, _COLLECTION, self._ann)
                self._store.clear_query_cache()
        return document_ids

    def _ingest_one(
        self,
        text: str,
        *,
        source: str | None,
        namespace: str | None,
        metadata: dict[str, Any] | None,
        chunk_size: int,
        chunk_overlap: int,
        markdown: bool,
    ) -> int:
        """Chunk, embed, and store one document — everything add_text/add_texts/add_files
        share. Does *not* persist the ANN index; callers save once after their own batch.
        """
        target_namespace = resolve_writable_namespace(self.policy, self.namespace, namespace)
        splitter = split_markdown if markdown else split_text
        chunks = splitter(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError("got no text to index (empty after stripping)")

        document_id = self._store.insert_document(target_namespace, source=source, metadata=metadata)
        # Every chunk inherits the document's metadata (v0.1 has no
        # per-chunk metadata) so `search(where=...)` can actually filter
        # by what the caller passed to add_text/add_texts/add_files.
        chunk_metadatas = [metadata or {} for _ in chunks]
        chunk_ids = self._store.insert_chunks(document_id, target_namespace, chunks, chunk_metadatas)

        vectors = self._embedder.embed(chunks)
        if self._ann is None:
            self._ann = AnnIndex(dim=len(vectors[0]))
        self._ann.add(chunk_ids, vectors)

        return document_id

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
    ) -> Hits:
        """Hybrid search over indexed chunks. Returns Hits — hits[0].text, hits[0].score, hits.timings.

        Hybrid (both signals, the default) is what you want for real
        queries. Set `use_bm25=False` for pure semantic search, or
        `use_vectors=False` for pure keyword search.

        `where` filters to chunks whose metadata matches every key/value
        given, e.g. `idx.search(q, where={"source": "docs/deploy.md"})` —
        equality only in v0.1, no operators.
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
        )

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
                save_ann_index(self._store, _COLLECTION, self._ann)
            self._store.clear_query_cache()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _iter_text_files(path: Path, pattern: str) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in _TEXT_SUFFIXES:
            yield path
        return
    for candidate in sorted(path.glob(pattern)):
        if candidate.is_file() and candidate.suffix.lower() in _TEXT_SUFFIXES:
            yield candidate
