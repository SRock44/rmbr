"""Approximate nearest-neighbor vector index.

Backed by `usearch` (an HNSW implementation). We chose usearch over the
originally-planned `hnswlib` because hnswlib ships no prebuilt wheel for
Python 3.13 on Windows — building it from source would require a C++
toolchain on every contributor's and every user's machine, which fails
the "pip install rmbr, no fuss" promise this whole project is built on.
usearch has wheels for all three OSes we support and the same core
algorithm.

The whole index serializes to a single blob (`to_bytes`/`from_bytes`) that
store.py tucks into the `ann_index` table — that's what makes the
one-file-per-index promise real for vector search, not just text.
"""

from __future__ import annotations

import numpy as np
from usearch.index import Index as _UsearchIndex

DEFAULT_METRIC = "cos"


class AnnIndex:
    """A single collection's vector index (e.g. all chunks, or all memories).

    IDs are the caller's own row ids (SQLite rowids from store.py) — no
    separate id-mapping table needed, usearch keys vectors directly by
    whatever int you give it.
    """

    def __init__(self, dim: int, metric: str = DEFAULT_METRIC):
        self.dim = dim
        self.metric = metric
        self._index = _UsearchIndex(ndim=dim, metric=metric, dtype="f32")

    def __len__(self) -> int:
        return len(self._index)

    def add(self, ids: list[int], vectors: list[np.ndarray]) -> None:
        """Add new vectors. Ids must not already be present — see `replace`."""
        if not ids:
            return
        keys = np.asarray(ids, dtype=np.int64)
        matrix = np.stack([np.asarray(v, dtype=np.float32) for v in vectors])
        self._index.add(keys, matrix)

    def replace(self, ids: list[int], vectors: list[np.ndarray]) -> None:
        """Add vectors, first removing any of these ids that already exist.

        usearch rejects duplicate keys outright, so re-indexing an updated
        chunk means remove-then-add rather than an in-place overwrite.
        """
        if not ids:
            return
        existing = [i for i in ids if i in self._index]
        if existing:
            self.remove(existing)
        self.add(ids, vectors)

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        self._index.remove(np.asarray(ids, dtype=np.int64))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k (id, similarity) pairs, best match first.

        Similarity is ``1 - cosine_distance``, so 1.0 means identical and
        0.0 means unrelated (can go negative for opposite vectors).
        """
        if len(self._index) == 0 or k <= 0:
            return []
        matches = self._index.search(np.asarray(query, dtype=np.float32), min(k, len(self._index)))
        return [(int(key), 1.0 - float(dist)) for key, dist in zip(matches.keys, matches.distances)]

    def to_bytes(self) -> bytes:
        return bytes(self._index.save())

    @classmethod
    def from_bytes(cls, blob: bytes, dim: int, metric: str = DEFAULT_METRIC) -> "AnnIndex":
        instance = cls(dim=dim, metric=metric)
        instance._index.load(blob)
        return instance
