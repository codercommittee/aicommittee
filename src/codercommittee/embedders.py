"""Embedders: the vectors of one item's spans.

The mock embedder hashes character 3-grams into a fixed number of dimensions (the study's
mock, byte for byte, with a one-hot fallback where that mock raised on a cancelled vector) and
needs no network. The OpenAI embedder requests ``text-embedding-3-large``
at 3,072 dimensions; the OpenAI-compatible embedder asks any ``/v1/embeddings`` server (vLLM,
text-embeddings-inference, llama.cpp); the function embedder wraps any callable from texts to
vectors (a sentence-transformers model, for instance). Each returns a Space scoped to the item
asked for.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from typing import Any

from .data import Item
from .similarity import Space

DIMENSIONS = 3072
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_LIMIT = 2048


class MockEmbedder:
    """Deterministic feature hashing of character 3-grams, unit length; no network."""

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """The unit vector of one text."""
        if not text.strip():
            raise ValueError("mock embedding requires non-empty text")
        padded = f" {text.lower()} "
        vector = [0.0] * self.dimensions
        for start in range(len(padded) - 2):
            digest = hashlib.sha256(padded[start : start + 3].encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0 if int(digest[8], 16) % 2 == 0 else -1.0
        norm = math.sqrt(math.fsum(x * x for x in vector))
        if norm == 0.0:
            # The signed contributions cancelled (a two-character text can do it): a one-hot from
            # the whole text. The study's mock raised here, so no stored vector changes.
            index = int(hashlib.sha256(padded.encode("utf-8")).hexdigest()[8:16], 16) % self.dimensions
            vector[index] = 1.0
            return vector
        return [x / norm for x in vector]

    def space(self, item: Item, texts: Sequence[str]) -> Space:
        """The space of the texts, embedded by feature hashing."""
        return Space({text: self.embed(text) for text in texts})


class OpenAIEmbedder:
    """``text-embedding-3-large`` vectors through the OpenAI SDK, one request per item (more above 2,048 spans).

    ``client`` is any object with ``embeddings.create(model=, input=, encoding_format=, dimensions=)``
    returning ``data[i].embedding`` and ``model``; without one the ``openai`` package is imported
    on first use. The returned model must equal the requested one.
    """

    def __init__(
        self, model: str = OPENAI_EMBEDDING_MODEL, dimensions: int | None = DIMENSIONS, *, client: Any = None
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.OpenAI(timeout=120, max_retries=2)
        return self._client

    def space(self, item: Item, texts: Sequence[str]) -> Space:
        """The space of the texts, from one embeddings request."""
        texts = list(texts)
        if not texts:
            return Space({})
        vectors: dict[str, list[float]] = {}
        for start in range(0, len(texts), BATCH_LIMIT):  # the API takes at most BATCH_LIMIT texts per request
            batch = texts[start : start + BATCH_LIMIT]
            arguments: dict[str, Any] = {"model": self.model, "input": batch, "encoding_format": "float"}
            if self.dimensions is not None:
                arguments["dimensions"] = self.dimensions
            reply = self._get_client().embeddings.create(**arguments)
            returned = getattr(reply, "model", None)
            if returned != self.model:
                raise RuntimeError(f"embedding model mismatch: asked {self.model}, got {returned}")
            data = sorted(reply.data, key=lambda entry: entry.index)
            if [entry.index for entry in data] != list(range(len(batch))):
                raise RuntimeError("embedding reply does not cover every text once")
            vectors.update({text: list(entry.embedding) for text, entry in zip(batch, data, strict=True)})
        return Space(vectors)


class OpenAICompatibleEmbedder(OpenAIEmbedder):
    """Any ``/v1/embeddings`` server through the OpenAI SDK: vLLM, text-embeddings-inference, llama.cpp.

    ``base_url`` names the server (``http://localhost:8000/v1`` for vLLM); ``api_key`` is whatever it
    expects. ``dimensions`` is sent only when given, since most servers refuse it.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        dimensions: int | None = None,
        *,
        api_key: str = "none",
        client: Any = None,
    ) -> None:
        super().__init__(model, dimensions, client=client)
        self.base_url = base_url
        self.api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=120, max_retries=2)
        return self._client


class FunctionEmbedder:
    """Any callable from a list of texts to a list of vectors.

    For example a sentence-transformers model::

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = FunctionEmbedder(lambda texts: model.encode(texts).tolist())
    """

    def __init__(self, embed: Callable[[list[str]], Sequence[Sequence[float]]]) -> None:
        self._embed = embed

    def space(self, item: Item, texts: Sequence[str]) -> Space:
        """The space of the texts, from the wrapped callable."""
        texts = list(texts)
        if not texts:
            return Space({})
        vectors = list(self._embed(texts))
        if len(vectors) != len(texts):
            raise RuntimeError("the embedding function must return one vector per text")
        return Space({text: [float(x) for x in vector] for text, vector in zip(texts, vectors, strict=True)})


__all__ = [
    "BATCH_LIMIT",
    "DIMENSIONS",
    "OPENAI_EMBEDDING_MODEL",
    "FunctionEmbedder",
    "MockEmbedder",
    "OpenAICompatibleEmbedder",
    "OpenAIEmbedder",
]
