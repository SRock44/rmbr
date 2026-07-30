"""Turning text into vectors, without making that a network call by default.

``Embedder`` is a tiny protocol — implement ``model_name`` and ``embed()``
and you can plug in any provider (OpenAI, Voyage, your own model server).
The default, ``FastEmbedEmbedder``, runs a small ONNX model locally via
``fastembed``: no API key, no network after the first model download.

``CachingEmbedder`` wraps any embedder with a SQLite-backed content-hash
cache (see store.py's ``embed_cache`` table), so the same text is never
embedded twice — across calls, across sessions, across restarts. This is
what `Index` and `Memory` actually use; the cache is what makes repeated
`idx.add_files()` runs over mostly-unchanged docs fast.

``FakeEmbedder`` is a deterministic, dependency-free embedder for tests —
both ours and yours. Use it in your own test suite to exercise `Memory`/
`Index` without a model download or a flaky network call.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

from .store import Store

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a batch of strings into a batch of vectors.

    Implement this to plug in a custom provider::

        class MyEmbedder:
            model_name = "my-provider/my-model"

            def embed(self, texts: list[str]) -> list[np.ndarray]:
                return [call_my_api(t) for t in texts]
    """

    model_name: str

    def embed(self, texts: list[str]) -> list[np.ndarray]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed. The default — no API key, no server."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        # Imported lazily so `import rmbr` stays cheap for callers who bring
        # their own embedder (or use FakeEmbedder) and never touch this path.
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(texts)]


class FakeEmbedder:
    """Deterministic, offline embedder for tests. No model, no network, no randomness across runs.

    Each text hashes to a fixed-size vector, so the same input always
    produces the same output within a process and across processes —
    useful for asserting on cache hits, recall behavior, or namespace
    isolation without downloading a real model.
    """

    def __init__(self, dimension: int = 32, model_name: str = "fake-embedder"):
        self.dimension = dimension
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)


class CachingEmbedder:
    """Wraps any Embedder with a persistent content-hash cache in the store's SQLite file."""

    def __init__(self, embedder: Embedder, store: Store):
        self.embedder = embedder
        self.store = store

    @property
    def model_name(self) -> str:
        return self.embedder.model_name

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        hashes = [_content_hash(t) for t in texts]

        # Dedupe by hash before touching the embedder — both against what's
        # already cached in SQLite and against repeats within this same
        # batch (e.g. the same sentence appearing in two chunks).
        by_hash: dict[str, np.ndarray] = {}
        for h in set(hashes):
            cached = self.store.get_cached_embedding(h, self.model_name)
            if cached is not None:
                by_hash[h] = np.frombuffer(cached, dtype=np.float32)

        missing_hashes: list[str] = []
        missing_texts: list[str] = []
        for h, t in zip(hashes, texts):
            if h not in by_hash and h not in missing_hashes:
                missing_hashes.append(h)
                missing_texts.append(t)

        if missing_texts:
            fresh = self.embedder.embed(missing_texts)
            for h, vector in zip(missing_hashes, fresh):
                vector = np.asarray(vector, dtype=np.float32)
                self.store.set_cached_embedding(h, self.model_name, vector.tobytes())
                by_hash[h] = vector

        return [by_hash[h] for h in hashes]

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
