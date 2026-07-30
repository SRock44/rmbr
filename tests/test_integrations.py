import asyncio

import pytest

from rmbr.embed import FakeEmbedder
from rmbr.index import Index


def make_index(path, **kwargs):
    return Index(str(path), embedder=FakeEmbedder(dimension=16), **kwargs)


def test_as_langchain_retriever_returns_documents(tmp_path):
    pytest.importorskip("langchain_core")  # optional extra - pip install rmbr[langchain]
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("the deployment guide covers docker and kubernetes")

    retriever = idx.as_langchain_retriever(k=3)
    docs = retriever.invoke("docker deployment")

    assert len(docs) == 1
    assert "docker" in docs[0].page_content
    assert docs[0].metadata["rmbr_id"] is not None
    assert docs[0].metadata["namespace"] == "default"


def test_as_langchain_retriever_async(tmp_path):
    pytest.importorskip("langchain_core")
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("the deployment guide covers docker and kubernetes")

    retriever = idx.as_langchain_retriever(k=3)
    docs = asyncio.run(retriever.ainvoke("docker deployment"))
    assert len(docs) == 1


def test_as_langchain_retriever_passes_search_kwargs(tmp_path):
    pytest.importorskip("langchain_core")
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("release notes", metadata={"tier": "internal"})
    idx.add_text("release notes", metadata={"tier": "public"})

    retriever = idx.as_langchain_retriever(k=5, where={"tier": "public"})
    docs = retriever.invoke("release notes")
    assert len(docs) == 1
    assert docs[0].metadata["tier"] == "public"


def test_as_llamaindex_retriever_returns_nodes(tmp_path):
    pytest.importorskip("llama_index.core")  # optional extra - pip install rmbr[llamaindex]
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("the deployment guide covers docker and kubernetes")

    retriever = idx.as_llamaindex_retriever(k=3)
    nodes = retriever.retrieve("docker deployment")

    assert len(nodes) == 1
    assert "docker" in nodes[0].node.text
    assert nodes[0].score is not None


def test_as_llamaindex_retriever_async(tmp_path):
    pytest.importorskip("llama_index.core")
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("the deployment guide covers docker and kubernetes")

    retriever = idx.as_llamaindex_retriever(k=3)
    nodes = asyncio.run(retriever.aretrieve("docker deployment"))
    assert len(nodes) == 1


def test_as_llamaindex_retriever_passes_search_kwargs(tmp_path):
    pytest.importorskip("llama_index.core")
    idx = make_index(tmp_path / "agents.db")
    idx.add_text("release notes", metadata={"tier": "internal"})
    idx.add_text("release notes", metadata={"tier": "public"})

    retriever = idx.as_llamaindex_retriever(k=5, where={"tier": "public"})
    nodes = retriever.retrieve("release notes")
    assert len(nodes) == 1
    assert nodes[0].node.metadata["tier"] == "public"
