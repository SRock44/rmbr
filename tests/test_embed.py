from unittest.mock import MagicMock

import numpy as np
import pytest

from rmbr.embed import CachingEmbedder, FakeEmbedder
from rmbr.store import Store


@pytest.fixture(autouse=True)
def _clear_shared_fastembed_cache():
    """Every test using get_shared_fastembed_embedder starts from a clean cache."""
    import rmbr.embed as embed_module

    embed_module._shared_fastembed_embedders.clear()
    yield
    embed_module._shared_fastembed_embedders.clear()


class CountingEmbedder:
    """Wraps FakeEmbedder and records every text it was actually asked to embed."""

    def __init__(self):
        self.model_name = "counting-fake"
        self._inner = FakeEmbedder(dimension=16, model_name=self.model_name)
        self.calls: list[str] = []

    def embed(self, texts):
        self.calls.extend(texts)
        return self._inner.embed(texts)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def test_fake_embedder_is_deterministic():
    embedder = FakeEmbedder(dimension=16)
    v1 = embedder.embed(["hello world"])[0]
    v2 = embedder.embed(["hello world"])[0]
    assert np.array_equal(v1, v2)


def test_fake_embedder_different_text_different_vector():
    embedder = FakeEmbedder(dimension=16)
    v1 = embedder.embed(["hello"])[0]
    v2 = embedder.embed(["goodbye"])[0]
    assert not np.array_equal(v1, v2)


def test_fake_embedder_vectors_are_unit_normalized():
    embedder = FakeEmbedder(dimension=16)
    v = embedder.embed(["some text"])[0]
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_caching_embedder_hits_cache_on_repeat_text(store):
    inner = CountingEmbedder()
    cache = CachingEmbedder(inner, store)

    cache.embed(["a", "b", "a"])
    assert sorted(inner.calls) == ["a", "b"]  # "a" embedded once, not twice


def test_caching_embedder_persists_across_instances(store):
    inner1 = CountingEmbedder()
    CachingEmbedder(inner1, store).embed(["persist me"])
    assert inner1.calls == ["persist me"]

    inner2 = CountingEmbedder()
    result = CachingEmbedder(inner2, store).embed(["persist me"])
    assert inner2.calls == []  # cache hit from the first instance's write
    assert result[0].shape == (16,)


def test_caching_embedder_separates_by_model_name(store):
    class ModelA(CountingEmbedder):
        def __init__(self):
            super().__init__()
            self.model_name = "model-a"

    class ModelB(CountingEmbedder):
        def __init__(self):
            super().__init__()
            self.model_name = "model-b"

    CachingEmbedder(ModelA(), store).embed(["shared text"])
    b = ModelB()
    CachingEmbedder(b, store).embed(["shared text"])
    assert b.calls == ["shared text"]  # different model = cache miss, re-embedded


def test_caching_embedder_empty_input_returns_empty(store):
    cache = CachingEmbedder(CountingEmbedder(), store)
    assert cache.embed([]) == []


def test_caching_embedder_embed_one(store):
    cache = CachingEmbedder(CountingEmbedder(), store)
    vec = cache.embed_one("solo text")
    assert vec.shape == (16,)


def test_get_shared_fastembed_embedder_reuses_instance_for_same_model(monkeypatch):
    """The whole point of the fix: same model_name -> same FastEmbedEmbedder,
    so only one onnxruntime InferenceSession ever gets constructed for it."""
    pytest.importorskip("fastembed")
    from rmbr.embed import get_shared_fastembed_embedder

    construction_count = {"n": 0}

    class FakeTextEmbedding:
        def __init__(self, model_name):
            construction_count["n"] += 1

    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)

    e1 = get_shared_fastembed_embedder("some/model")
    e2 = get_shared_fastembed_embedder("some/model")
    e3 = get_shared_fastembed_embedder("some/model")

    assert e1 is e2 is e3
    assert construction_count["n"] == 1


def test_get_shared_fastembed_embedder_separates_by_model_name(monkeypatch):
    pytest.importorskip("fastembed")
    from rmbr.embed import get_shared_fastembed_embedder

    construction_count = {"n": 0}

    class FakeTextEmbedding:
        def __init__(self, model_name):
            construction_count["n"] += 1

    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)

    a = get_shared_fastembed_embedder("model-a")
    b = get_shared_fastembed_embedder("model-b")

    assert a is not b
    assert construction_count["n"] == 2


def test_get_shared_fastembed_embedder_is_thread_safe(monkeypatch):
    """Many threads racing to build the same default embedder must still
    only construct one underlying session, and never crash/raise."""
    pytest.importorskip("fastembed")
    import threading

    from rmbr.embed import get_shared_fastembed_embedder

    construction_count = {"n": 0}
    construction_lock = threading.Lock()

    class FakeTextEmbedding:
        def __init__(self, model_name):
            with construction_lock:
                construction_count["n"] += 1

    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)

    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(get_shared_fastembed_embedder("racing-model"))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 32
    assert all(r is results[0] for r in results)
    assert construction_count["n"] == 1


def test_make_embedder_default_path_shares_fastembed_session_across_instances(monkeypatch, store):
    """Reproduces the crash scenario from the bug report at the make_embedder
    level: multiple Memory/Index-style constructions (one CachingEmbedder per
    "namespace") against the default embedder=None path must not each build
    their own FastEmbedEmbedder/TextEmbedding session."""
    pytest.importorskip("fastembed")
    from rmbr._engine import make_embedder

    construction_count = {"n": 0}

    class FakeTextEmbedding:
        def __init__(self, model_name):
            construction_count["n"] += 1

    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)

    embedders = [make_embedder(None, store) for _ in range(6)]  # e.g. 6 namespaces

    assert construction_count["n"] == 1
    underlying = [ce.embedder for ce in embedders]
    assert all(u is underlying[0] for u in underlying)


def test_openai_embedder_batches_and_sorts_by_index(monkeypatch):
    pytest.importorskip("openai")  # optional extra - pip install rmbr[openai]
    from rmbr.embed import OpenAIEmbedder

    # Deliberately out of request order, to prove the defensive sort works.
    fake_response = MagicMock()
    fake_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3], index=1),
        MagicMock(embedding=[0.4, 0.5, 0.6], index=0),
    ]
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = fake_response
    monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: mock_client)

    embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
    vectors = embedder.embed(["first", "second"])

    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=["first", "second"]
    )
    assert len(vectors) == 2
    assert np.allclose(vectors[0], [0.4, 0.5, 0.6])  # index 0
    assert np.allclose(vectors[1], [0.1, 0.2, 0.3])  # index 1


def test_openai_embedder_empty_input_skips_the_api_call(monkeypatch):
    pytest.importorskip("openai")  # optional extra - pip install rmbr[openai]
    from rmbr.embed import OpenAIEmbedder

    mock_client = MagicMock()
    monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: mock_client)

    embedder = OpenAIEmbedder()
    assert embedder.embed([]) == []
    mock_client.embeddings.create.assert_not_called()


def test_voyage_embedder_embeds_in_request_order(monkeypatch):
    pytest.importorskip("voyageai")  # optional extra - pip install rmbr[voyage]
    from rmbr.embed import VoyageEmbedder

    fake_response = MagicMock()
    fake_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_client = MagicMock()
    mock_client.embed.return_value = fake_response
    monkeypatch.setattr("voyageai.Client", lambda *args, **kwargs: mock_client)

    embedder = VoyageEmbedder(model_name="voyage-3")
    vectors = embedder.embed(["first", "second"])

    mock_client.embed.assert_called_once_with(["first", "second"], model="voyage-3", input_type=None)
    assert len(vectors) == 2
    assert np.allclose(vectors[0], [0.1, 0.2, 0.3])
    assert np.allclose(vectors[1], [0.4, 0.5, 0.6])


def test_voyage_embedder_empty_input_skips_the_api_call(monkeypatch):
    pytest.importorskip("voyageai")  # optional extra - pip install rmbr[voyage]
    from rmbr.embed import VoyageEmbedder

    mock_client = MagicMock()
    monkeypatch.setattr("voyageai.Client", lambda *args, **kwargs: mock_client)

    embedder = VoyageEmbedder()
    assert embedder.embed([]) == []
    mock_client.embed.assert_not_called()


def test_cohere_embedder_embeds_float_vectors(monkeypatch):
    pytest.importorskip("cohere")  # optional extra - pip install rmbr[cohere]
    from rmbr.embed import CohereEmbedder

    fake_response = MagicMock()
    fake_response.embeddings.float_ = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_client = MagicMock()
    mock_client.embed.return_value = fake_response
    monkeypatch.setattr("cohere.ClientV2", lambda *args, **kwargs: mock_client)

    embedder = CohereEmbedder(model_name="embed-english-v3.0")
    vectors = embedder.embed(["first", "second"])

    mock_client.embed.assert_called_once_with(
        model="embed-english-v3.0",
        texts=["first", "second"],
        input_type="search_document",
        embedding_types=["float"],
    )
    assert len(vectors) == 2
    assert np.allclose(vectors[0], [0.1, 0.2, 0.3])
    assert np.allclose(vectors[1], [0.4, 0.5, 0.6])


def test_cohere_embedder_empty_input_skips_the_api_call(monkeypatch):
    pytest.importorskip("cohere")  # optional extra - pip install rmbr[cohere]
    from rmbr.embed import CohereEmbedder

    mock_client = MagicMock()
    monkeypatch.setattr("cohere.ClientV2", lambda *args, **kwargs: mock_client)

    embedder = CohereEmbedder()
    assert embedder.embed([]) == []
    mock_client.embed.assert_not_called()
