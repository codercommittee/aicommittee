"""Embedding spaces and the pair similarity of relations.

A :class:`Space` holds the vectors of one item's spans, keyed by the sha256 of the text, and
memoises exact cosines. It is scoped to one item of one run: the same text may have different
vectors elsewhere, and no cache crosses items or runs. A :class:`Similarity` is a task's view of
a space: the mean over the task's similarity fields of the clamped cosine, computed by exact
summation and rounded to twelve places so that the result is identical on any machine. Empty
spans score one against empty and zero against text and are never embedded. Every threshold in
the package is ``>=`` on this rounded value.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

from .data import Relation

DECIMALS = 12
SAFE_MAGNITUDE = (1e-70, 1e70)
"""Vectors whose largest entry lies outside this range are scaled by a power of two before the sums.

The product of two squared norms must stay finite and non-zero: a fourth power of the magnitude.
"""


def text_sha256(text: str) -> str:
    """The sha256 hex digest of a text, the key every vector is stored under."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MissingVectorError(KeyError):
    """A text has no vector in this space."""


def _as_vector(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("embedding vectors hold numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("embedding vectors must be finite")
        out.append(number)
    if not out:
        raise ValueError("embedding vectors must not be empty")
    return out


class Space:
    """The vectors of one item's spans, keyed by sha256(text), with memoised exact cosines."""

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._texts: dict[str, str] = {}
        self._scaled: dict[str, list[float]] = {}
        self._norms: dict[str, float] = {}
        self._cosines: dict[tuple[str, str], float] = {}
        dimension: int | None = None
        for text, vector in vectors.items():
            if not isinstance(text, str):
                raise TypeError("space keys are texts")
            values = _as_vector(vector)
            if dimension is None:
                dimension = len(values)
            elif len(values) != dimension:
                raise ValueError("every vector in a space has the same dimension")
            sha = text_sha256(text)
            self._vectors[sha] = values
            self._texts[sha] = text
        self._dimension = dimension or 0

    @property
    def dimension(self) -> int:
        """The vectors' dimension (0 for an empty space)."""
        return self._dimension

    @property
    def texts(self) -> tuple[str, ...]:
        """The embedded texts in sha256 order."""
        return tuple(self._texts[sha] for sha in sorted(self._texts))

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, text: object) -> bool:
        return isinstance(text, str) and text_sha256(text) in self._vectors

    def vector(self, text: str) -> list[float]:
        """A copy of the text's vector."""
        sha = text_sha256(text)
        if sha not in self._vectors:
            raise MissingVectorError(f"no vector for text {text[:60]!r} (sha256 {sha[:12]})")
        return list(self._vectors[sha])

    def _scaled_vector(self, sha: str) -> list[float]:
        """The vector the sums run on: the original list unless its magnitude would overflow or vanish.

        Scaling by a power of two is exact and the cosine is scale-free, so an in-range vector (every
        embedding model's) is summed as is, bit for bit, and an extreme one gives the same cosine.
        """
        if sha not in self._scaled:
            values = self._vectors[sha]
            peak = max((abs(x) for x in values), default=0.0)
            low, high = SAFE_MAGNITUDE
            if peak == 0.0 or low <= peak <= high:
                self._scaled[sha] = values
            else:
                exponent = math.frexp(peak)[1]
                self._scaled[sha] = [math.ldexp(x, -exponent) for x in values]
        return self._scaled[sha]

    def _squared_norm(self, sha: str) -> float:
        if sha not in self._norms:
            self._norms[sha] = math.fsum(x * x for x in self._scaled_vector(sha))
        return self._norms[sha]

    def cosine(self, text_a: str, text_b: str) -> float:
        """The clamped cosine of two embedded texts, exact summation, twelve places."""
        sha_a, sha_b = text_sha256(text_a), text_sha256(text_b)
        for sha, text in ((sha_a, text_a), (sha_b, text_b)):
            if sha not in self._vectors:
                raise MissingVectorError(f"no vector for text {text[:60]!r} (sha256 {sha[:12]})")
        pair = (sha_a, sha_b) if sha_a <= sha_b else (sha_b, sha_a)
        if pair not in self._cosines:
            denominator = math.sqrt(self._squared_norm(sha_a) * self._squared_norm(sha_b))
            if denominator == 0.0:
                self._cosines[pair] = 0.0
            else:
                a, b = self._scaled_vector(sha_a), self._scaled_vector(sha_b)
                dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
                value = dot / denominator
                if not math.isfinite(value):
                    raise ValueError(f"cosine of {text_a[:40]!r} and {text_b[:40]!r} is not finite")
                self._cosines[pair] = round(max(value, 0.0), DECIMALS)
        return self._cosines[pair]


class Similarity:
    """A task's view of a space: field-wise cosines averaged with equal weights.

    ``fields`` are compared; ``aux`` fields are carried for keys and printing only. ``names`` is
    their concatenation, the order every content key uses.
    """

    def __init__(self, space: Space, fields: Sequence[str] = ("cause", "effect"), aux: Sequence[str] = ("signal",)):
        if not fields:
            raise ValueError("similarity needs at least one field")
        self._space = space
        self.fields = tuple(fields)
        self.aux = tuple(aux)

    @property
    def space(self) -> Space:
        """The space this similarity reads."""
        return self._space

    @property
    def names(self) -> tuple[str, ...]:
        """The field names: the compared fields, then the auxiliary ones."""
        return self.fields + self.aux

    def field(self, left: Relation, right: Relation, name: str) -> float:
        """One field's cosine: empty against empty 1.0, one empty 0.0."""
        a, b = left.get(name, "").strip(), right.get(name, "").strip()
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return self._space.cosine(a, b)

    def per_field(self, left: Relation, right: Relation) -> dict[str, float]:
        """Every compared field's cosine."""
        return {name: self.field(left, right, name) for name in self.fields}

    def pair(self, left: Relation, right: Relation) -> float:
        """The mean over the fields of the clamped cosine, rounded to twelve places."""
        return round(math.fsum(self.per_field(left, right).values()) / len(self.fields), DECIMALS)


__all__ = ["DECIMALS", "MissingVectorError", "Similarity", "Space", "text_sha256"]
