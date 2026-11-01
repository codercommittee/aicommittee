from __future__ import annotations

import math

import pytest

from codercommittee.data import Relation
from codercommittee.similarity import MissingVectorError, Similarity, Space, text_sha256


def _unit(*coordinates: tuple[int, float], size: int = 8) -> list[float]:
    vector = [0.0] * size
    for index, value in coordinates:
        vector[index] = value
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def test_cosine_identity_orthogonal_clamp_and_symmetry() -> None:
    vectors = {"a": _unit((0, 1.0)), "b": _unit((1, 1.0)), "c": [-1.0] + [0.0] * 7, "ab": _unit((0, 1.0), (1, 1.0))}
    space = Space(vectors)
    assert space.cosine("a", "a") == 1.0
    assert space.cosine("a", "b") == 0.0
    assert space.cosine("a", "c") == 0.0  # clamped, not -1
    assert space.cosine("a", "ab") == space.cosine("ab", "a") == round(math.sqrt(0.5), 12)
    assert space.dimension == 8 and len(space) == 4 and "a" in space and "zzz" not in space
    assert space.texts == tuple(sorted(("a", "b", "c", "ab"), key=text_sha256))
    with pytest.raises(MissingVectorError):
        space.cosine("a", "zzz")
    with pytest.raises(MissingVectorError):
        space.vector("zzz")


def test_space_rejects_bad_vectors() -> None:
    with pytest.raises(ValueError):
        Space({"a": [1.0, 2.0], "b": [1.0]})
    with pytest.raises(ValueError):
        Space({"a": [float("nan")]})
    with pytest.raises(ValueError):
        Space({"a": []})
    with pytest.raises(ValueError):
        Space({"a": [True]})
    assert len(Space({})) == 0


def test_pair_similarity_empty_rules_mean_and_rounding() -> None:
    space = Space({"x": _unit((0, 1.0)), "y": _unit((0, 1.0), (1, 1.0)), "z": _unit((2, 1.0))})
    sim = Similarity(space, fields=("cause", "effect"), aux=("signal",))
    assert sim.names == ("cause", "effect", "signal")
    same = Relation(cause="x", effect="z", signal="q")
    assert sim.pair(same, Relation(cause="x", effect="z", signal="other")) == 1.0
    assert sim.pair(Relation(cause="", effect=""), Relation(cause="", effect="")) == 1.0
    assert sim.pair(Relation(cause="x", effect=""), Relation(cause="x", effect="z")) == 0.5
    half = sim.pair(Relation(cause="x", effect="z"), Relation(cause="y", effect="z"))
    assert half == round((round(math.sqrt(0.5), 12) + 1.0) / 2, 12)
    assert sim.per_field(same, Relation(cause="y", effect="z")) == {"cause": round(math.sqrt(0.5), 12), "effect": 1.0}
    assert sim.field(same, Relation(cause="x", effect="z"), "signal") == 0.0  # one empty aux span
    with pytest.raises(ValueError):
        Similarity(space, fields=())


@pytest.mark.parametrize("magnitude", [1.0, 1e100, 1e-100, 1e200, 1e-200, 1e308])
def test_cosine_survives_extreme_magnitudes(magnitude: float) -> None:
    vector = [magnitude, magnitude]
    space = Space({"rain": vector, "flood": vector, "other": [magnitude, 0.0]})
    assert space.cosine("rain", "rain") == 1.0 and space.cosine("rain", "flood") == 1.0
    assert space.cosine("rain", "other") == round(math.sqrt(0.5), 12)
    assert space.vector("rain") == vector  # the stored vector is untouched
    mixed = Space({"big": [magnitude, 0.0], "unit": [1.0, 0.0], "diag": [1.0, 1.0]})
    assert mixed.cosine("big", "unit") == 1.0 and mixed.cosine("big", "diag") == round(math.sqrt(0.5), 12)


def test_in_range_vectors_are_summed_exactly_as_before() -> None:
    a = [0.123456789, -0.987654321, 0.5, 1e-5]
    b = [0.3, 0.2, -0.1, 7e-6]
    space = Space({"a": a, "b": b})
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(math.fsum(x * x for x in a) * math.fsum(y * y for y in b))
    assert space.cosine("a", "b") == round(max(dot / norms, 0.0), 12)
    assert space.cosine("a", "a") == 1.0 and Space({"z": [0.0, 0.0]}).cosine("z", "z") == 0.0
