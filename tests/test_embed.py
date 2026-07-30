from unittest.mock import MagicMock

import numpy as np
import pytest

from rmbr.embed import CachingEmbedder, FakeEmbedder
from rmbr.store import Store


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
