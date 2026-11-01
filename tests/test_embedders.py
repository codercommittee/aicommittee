from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from codercommittee.data import Item
from codercommittee.embedders import (
    DIMENSIONS,
    FunctionEmbedder,
    MockEmbedder,
    OpenAICompatibleEmbedder,
    OpenAIEmbedder,
)
from codercommittee.replay import StoredRun


def test_mock_embedder_reproduces_the_study_vectors_in_the_fixture(mock_cycles: dict[str, Any]) -> None:
    stored = StoredRun(mock_cycles["root"] / "c1" / "causalnews")
    payload = stored.embedding(0)
    assert payload is not None
    embedder = MockEmbedder()
    for entry in payload["vectors"].values():
        assert embedder.embed(entry["text"]) == entry["vector"]
    with pytest.raises(ValueError):
        embedder.embed("   ")
    space = embedder.space(Item("a", "t"), ["alpha beta", "gamma"])
    assert len(space) == 2 and space.dimension == DIMENSIONS


class FakeEmbeddingsClient:
    """An OpenAI-shaped embeddings client over the mock embedder, with a configurable returned model."""

    def __init__(self, model: str = "text-embedding-3-large", shuffle: bool = False) -> None:
        self.model = model
        self.shuffle = shuffle
        self.calls: list[dict[str, Any]] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **arguments: Any) -> SimpleNamespace:
        self.calls.append(arguments)
        texts = list(arguments["input"])
        data = [SimpleNamespace(index=i, embedding=MockEmbedder(8).embed(t)) for i, t in enumerate(texts)]
        if self.shuffle:
            data.reverse()
        return SimpleNamespace(model=self.model, data=data)


def test_openai_embedder_requests_once_per_item_and_checks_the_reply() -> None:
    client = FakeEmbeddingsClient()
    embedder = OpenAIEmbedder(dimensions=8, client=client)
    space = embedder.space(Item("a", "t"), ["b text", "a text"])
    assert len(client.calls) == 1
    assert client.calls[0] == {
        "model": "text-embedding-3-large",
        "input": ["b text", "a text"],
        "encoding_format": "float",
        "dimensions": 8,
    }
    assert space.vector("a text") == MockEmbedder(8).embed("a text")
    assert len(OpenAIEmbedder(dimensions=8, client=client).space(Item("a", "t"), [])) == 0 and len(client.calls) == 1
    ordered = OpenAIEmbedder(dimensions=8, client=FakeEmbeddingsClient(shuffle=True)).space(Item("a", "t"), ["x", "y"])
    assert ordered.vector("x") == MockEmbedder(8).embed("x")
    with pytest.raises(RuntimeError, match="model mismatch"):
        OpenAIEmbedder(dimensions=8, client=FakeEmbeddingsClient(model="other")).space(Item("a", "t"), ["x"])
    batching = FakeEmbeddingsClient()
    big = OpenAIEmbedder(dimensions=8, client=batching).space(Item("a", "t"), [f"text number {i}" for i in range(2049)])
    assert len(batching.calls) == 2 and len(big) == 2049 and len(batching.calls[1]["input"]) == 1
    json.dumps(space.vector("a text"))  # vectors are plain floats


def test_compatible_and_function_embedders() -> None:
    client = FakeEmbeddingsClient(model="bge-m3")
    embedder = OpenAICompatibleEmbedder("http://localhost:8000/v1", "bge-m3", client=client)
    space = embedder.space(Item("a", "t"), ["x", "y"])
    assert len(space) == 2 and "dimensions" not in client.calls[0] and client.calls[0]["model"] == "bge-m3"
    sized = OpenAICompatibleEmbedder(
        "http://localhost:8000/v1", "bge-m3", 8, client=FakeEmbeddingsClient(model="bge-m3")
    )
    assert sized.space(Item("a", "t"), ["x"]).dimension == 8
    assert embedder.base_url == "http://localhost:8000/v1" and embedder.api_key == "none"
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [MockEmbedder(4).embed(text) for text in texts]

    function = FunctionEmbedder(embed)
    space = function.space(Item("a", "t"), ["b", "a"])
    assert calls == [["b", "a"]] and space.vector("a") == MockEmbedder(4).embed("a")
    assert len(function.space(Item("a", "t"), [])) == 0 and len(calls) == 1
    with pytest.raises(RuntimeError, match="one vector per text"):
        FunctionEmbedder(lambda texts: []).space(Item("a", "t"), ["x"])


def test_the_mock_embedder_never_collapses_to_the_zero_vector() -> None:
    embedder = MockEmbedder()
    vector = embedder.embed("n@")  # its two 3-grams hash to one index with opposite signs
    assert len(vector) == DIMENSIONS and sum(abs(x) for x in vector) == 1.0 and max(vector) == 1.0
    assert embedder.embed("n@") == vector
    space = embedder.space(Item("a", "n@ causes flood"), ["n@", "flood"])
    assert space.cosine("n@", "n@") == 1.0
