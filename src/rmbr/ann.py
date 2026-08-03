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

# usearch's own defaults (connectivity=16, expansion_add=128,
# expansion_search=64) measurably under-recall once a collection gets
# into the thousands of vectors: benched at recall@5=0.68 on a 5,000-doc
# corpus vs 0.94+ here. expansion_search=256 alone gets recall@5 to 0.94
# at ~0.36ms/query (from usearch's own default ~0.68 at ~0.14ms/query);
# pairing it with expansion_add=256 nudges that to 0.95 for ~30% more
# build time - both still sub-millisecond and sub-second respectively,
# so there's no real reason to keep the leaner defaults. See bench/.
DEFAULT_EXPANSION_ADD = 256
DEFAULT_EXPANSION_SEARCH = 256


class AnnIndex:
    """A single collection's vector index (e.g. all chunks, or all memories).

    IDs are the caller's own row ids (SQLite rowids from store.py) — no
    separate id-mapping table needed, usearch keys vectors directly by
    whatever int you give it.
    """

    def __init__(
        self,
        dim: int,
        metric: str = DEFAULT_METRIC,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
    ):
        self.dim = dim
        self.metric = metric
        self._expansion_add = expansion_add
        self._expansion_search = expansion_search
        self._index = _UsearchIndex(
            ndim=dim,
            metric=metric,
            dtype="f32",
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        self._has_removals = False

    def __len__(self) -> int:
        return len(self._index)

    def ids(self) -> list[int]:
        """Every id currently in the index - for `integrity_check()`."""
        return [int(key) for key in self._index.keys]

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
        self._has_removals = True

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
        self._compact_if_needed()
        return bytes(self._index.save())

    def _compact_if_needed(self) -> None:
        """Rebuild `self._index` from its surviving vectors if anything was
        ever `remove()`-d from it, right before serialization.

        usearch (>=2.9, confirmed through 2.26.0) leaves tombstoned nodes
        in the HNSW graph after `remove()` — fine for a live index, but a
        `save()` → `load()` round trip of an index that has ever had a
        `remove()` (even down to zero vectors) produces one that segfaults
        on its very next `add()` in the new process. Every `Memory`/`Index`
        write path calls `to_bytes()` after `remember()`/`forget()`, and
        `from_bytes()` runs on every subsequent open of that `.db` file, so
        without this, opening a second handle onto a namespace that ever
        had anything removed would native-crash on its first write. See
        https://github.com/SRock44/rmbr/issues/20.
        """
        if not self._has_removals:
            return
        ids = self.ids()
        fresh = _UsearchIndex(
            ndim=self.dim,
            metric=self.metric,
            dtype="f32",
            expansion_add=self._expansion_add,
            expansion_search=self._expansion_search,
        )
        if ids:
            keys = np.asarray(ids, dtype=np.int64)
            got = self._index.get(keys)
            vectors = np.stack(got) if isinstance(got, tuple) else np.asarray(got).reshape(len(ids), self.dim)
            fresh.add(keys, vectors)
        self._index = fresh
        self._has_removals = False

    @classmethod
    def from_bytes(cls, blob: bytes, dim: int, metric: str = DEFAULT_METRIC) -> AnnIndex:
        instance = cls(dim=dim, metric=metric)
        instance._index.load(blob)
        return instance

