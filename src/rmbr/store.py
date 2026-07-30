"""SQLite persistence layer.

Store is intentionally "dumb": it knows how to save and fetch documents,
chunks, memories, and a couple of caches. It does not embed text, chunk
text, or rank search results — that logic lives in embed.py, chunk.py, and
search.py. Keeping this file free of those concerns is what lets us swap
embedding models or ranking strategies without touching the schema.

Vectors are stored and returned as raw bytes (float32, little-endian, via
numpy's .tobytes()). Store never imports numpy itself, so this file has
zero dependencies beyond the Python standard library — same as the rest of
rmbr's "no server, no API key" promise, just applied to our own code too.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    namespace TEXT NOT NULL,
    source TEXT,
    metadata TEXT NOT NULL DEFAULT '{{}}',
    added_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_namespace ON documents(namespace);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{{}}',
    chunk_index INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_namespace ON chunks(namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{{}}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, content='memories', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (content_hash, model)
);

CREATE TABLE IF NOT EXISTS query_cache (
    cache_key TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    query_vector BLOB NOT NULL,
    results TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_query_cache_namespace ON query_cache(namespace);

CREATE TABLE IF NOT EXISTS ann_index (
    collection TEXT PRIMARY KEY,
    blob BLOB NOT NULL,
    dim INTEGER NOT NULL,
    metric TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DocumentRecord:
    id: int
    namespace: str
    source: str | None
    metadata: dict[str, Any]
    added_at: str


@dataclass
class ChunkRecord:
    id: int
    document_id: int
    namespace: str
    text: str
    metadata: dict[str, Any]
    chunk_index: int


@dataclass
class MemoryRecord:
    id: int
    namespace: str
    text: str
    metadata: dict[str, Any]
    created_at: str


@dataclass
class ScoredId:
    """A row id plus its rank-order position in a single-signal search (FTS or ANN).

    We carry rank rather than raw score because BM25 and cosine distance
    live on incomparable scales — reciprocal rank fusion (see search.py)
    only needs "how many results were ranked ahead of this one".
    """

    id: int
    rank: int
    raw_score: float = field(default=0.0)


class Store:
    """Owns the SQLite file: schema, connection, and CRUD. No ranking, no ML."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        # check_same_thread=False: mcp_server.py's tool calls run on a
        # worker thread supplied by the MCP framework, not the thread that
        # constructed this Store. Python's sqlite3 module defaults to
        # "serialized" thread safety (a single connection may be used from
        # multiple threads, just not concurrently from more than one at
        # once) — check_same_thread only adds a same-thread *assertion* on
        # top of that, which we don't want here.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._auto_commit = True
        self._init_schema()

    def _commit(self) -> None:
        if self._auto_commit:
            self.conn.commit()

    @contextmanager
    def transaction(self):
        """Batch every write made inside this block into a single commit.

        Measured on real hardware: SQLite committing once per statement
        vs. once per batch is a ~47x difference (1,594 vs 75,521
        inserts/sec for the same 5,000-row workload) — that's SQLite's
        per-transaction fsync overhead, not its raw throughput. Bulk
        operations (see index.py's add_texts/add_files) wrap their whole
        batch in this; anything outside a `transaction()` block still
        commits immediately after every call, same as always, so a
        single `remember()` is still durable the instant it returns.
        """
        if not self._auto_commit:
            yield  # already inside an outer transaction; let it own the commit
            return
        self._auto_commit = False
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self._auto_commit = True

    def _init_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._commit()
        else:
            file_version = int(row["value"])
            if file_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.path} was written by a newer version of rmbr "
                    f"(schema v{file_version}, this rmbr supports up to "
                    f"v{SCHEMA_VERSION}). Upgrade rmbr to open this file."
                )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- documents & chunks ------------------------------------------------

    def insert_document(
        self, namespace: str, source: str | None = None, metadata: dict[str, Any] | None = None
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO documents(namespace, source, metadata, added_at) VALUES (?, ?, ?, ?)",
            (namespace, source, json.dumps(metadata or {}), _now()),
        )
        self._commit()
        return cur.lastrowid

    def insert_chunks(
        self,
        document_id: int,
        namespace: str,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Bulk-insert chunks for one document, in order. Returns their new ids."""
        metadatas = metadatas or [{} for _ in texts]
        ids = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            cur = self.conn.execute(
                "INSERT INTO chunks(document_id, namespace, text, metadata, chunk_index) "
                "VALUES (?, ?, ?, ?, ?)",
                (document_id, namespace, text, json.dumps(metadata), i),
            )
            ids.append(cur.lastrowid)
        self._commit()
        return ids

    def get_chunks(self, ids: list[int]) -> list[ChunkRecord]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids
        ).fetchall()
        by_id = {row["id"]: _row_to_chunk(row) for row in rows}
        return [by_id[i] for i in ids if i in by_id]

    def get_document(self, document_id: int) -> DocumentRecord | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return _row_to_document(row) if row else None

    def get_chunk_ids_for_document(self, document_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchall()
        return [row["id"] for row in rows]

    def delete_document(self, document_id: int) -> None:
        self.conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        self._commit()

    def search_chunks_fts(
        self, query: str, namespaces: list[str], limit: int
    ) -> list[ScoredId]:
        return self._search_fts("chunks", query, namespaces, limit)

    def list_chunk_namespaces(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT namespace FROM chunks").fetchall()
        return [row["namespace"] for row in rows]

    # -- memories ------------------------------------------------------------

    def insert_memory(
        self, namespace: str, text: str, metadata: dict[str, Any] | None = None
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO memories(namespace, text, metadata, created_at) VALUES (?, ?, ?, ?)",
            (namespace, text, json.dumps(metadata or {}), _now()),
        )
        self._commit()
        return cur.lastrowid

    def get_memory(self, memory_id: int) -> MemoryRecord | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return _row_to_memory(row) if row else None

    def get_memories(self, ids: list[int]) -> list[MemoryRecord]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
        ).fetchall()
        by_id = {row["id"]: _row_to_memory(row) for row in rows}
        return [by_id[i] for i in ids if i in by_id]

    def list_memories(self, namespace: str, limit: int | None = None) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE namespace = ? ORDER BY created_at DESC"
        params: list[Any] = [namespace]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_memory(row) for row in rows]

    def delete_memory(self, memory_id: int) -> None:
        self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._commit()

    def search_memories_fts(
        self, query: str, namespaces: list[str], limit: int
    ) -> list[ScoredId]:
        return self._search_fts("memories", query, namespaces, limit)

    def list_memory_namespaces(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT namespace FROM memories").fetchall()
        return [row["namespace"] for row in rows]

    def _search_fts(
        self, table: str, query: str, namespaces: list[str], limit: int
    ) -> list[ScoredId]:
        match = _to_fts_match(query)
        if not match or not namespaces:
            return []
        ns_placeholders = ",".join("?" * len(namespaces))
        sql = (
            f"SELECT {table}.id AS id, bm25({table}_fts) AS score "
            f"FROM {table}_fts JOIN {table} ON {table}.id = {table}_fts.rowid "
            f"WHERE {table}_fts MATCH ? AND {table}.namespace IN ({ns_placeholders}) "
            f"ORDER BY score LIMIT ?"
        )
        rows = self.conn.execute(sql, [match, *namespaces, limit]).fetchall()
        # bm25() in FTS5 is *more negative = more relevant*, so row order is
        # already best-first; we just attach the 0-based rank.
        return [
            ScoredId(id=row["id"], rank=i, raw_score=row["score"])
            for i, row in enumerate(rows)
        ]

    # -- embedding cache -------------------------------------------------

    def get_cached_embedding(self, content_hash: str, model: str) -> bytes | None:
        row = self.conn.execute(
            "SELECT vector FROM embed_cache WHERE content_hash = ? AND model = ?",
            (content_hash, model),
        ).fetchone()
        return row["vector"] if row else None

    def set_cached_embedding(self, content_hash: str, model: str, vector: bytes) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO embed_cache(content_hash, model, vector) VALUES (?, ?, ?)",
            (content_hash, model, vector),
        )
        self._commit()

    # -- semantic query cache ---------------------------------------------

    def list_query_cache(self, namespace: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM query_cache WHERE namespace = ?", (namespace,)
        ).fetchall()

    def set_query_cache(
        self, cache_key: str, namespace: str, query_vector: bytes, results: str, created_at: float
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO query_cache "
            "(cache_key, namespace, query_vector, results, created_at) VALUES (?, ?, ?, ?, ?)",
            (cache_key, namespace, query_vector, results, created_at),
        )
        self._commit()

    def purge_expired_query_cache(self, ttl_seconds: float, now: float) -> None:
        self.conn.execute(
            "DELETE FROM query_cache WHERE created_at < ?", (now - ttl_seconds,)
        )
        self._commit()

    def clear_query_cache(self) -> None:
        """Drop all cached search results.

        Called on every write so a cached answer can never mask
        newly-remembered or newly-indexed content. The cache exists to
        speed up read-heavy bursts, not to survive writes — TTL alone
        isn't tight enough for that guarantee.
        """
        self.conn.execute("DELETE FROM query_cache")
        self._commit()

    # -- ANN index blob ------------------------------------------------------

    def get_ann_blob(self, collection: str) -> tuple[bytes, int, str] | None:
        row = self.conn.execute(
            "SELECT blob, dim, metric FROM ann_index WHERE collection = ?", (collection,)
        ).fetchone()
        return (row["blob"], row["dim"], row["metric"]) if row else None

    def set_ann_blob(self, collection: str, blob: bytes, dim: int, metric: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO ann_index(collection, blob, dim, metric, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (collection, blob, dim, metric, _now()),
        )
        self._commit()


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        namespace=row["namespace"],
        source=row["source"],
        metadata=json.loads(row["metadata"]),
        added_at=row["added_at"],
    )


def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        id=row["id"],
        document_id=row["document_id"],
        namespace=row["namespace"],
        text=row["text"],
        metadata=json.loads(row["metadata"]),
        chunk_index=row["chunk_index"],
    )


def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        namespace=row["namespace"],
        text=row["text"],
        metadata=json.loads(row["metadata"]),
        created_at=row["created_at"],
    )


def _to_fts_match(query: str) -> str:
    """Turn free-text user input into a safe FTS5 MATCH expression.

    Quoting each token individually escapes FTS5 syntax characters (like
    ``"``, ``-``, ``*``) so a query such as ``deploy-to-prod?`` can't raise
    a syntax error. Space-separated quoted tokens is FTS5's implicit AND,
    which is the intuitive "all these words" behavior most users expect.
    """
    tokens = query.split()
    if not tokens:
        return ""
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)
