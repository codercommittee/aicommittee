from __future__ import annotations

import math

from codercommittee.data import Member, Relation
from codercommittee.dedupe import dedupe, medoid, member_key, relation_key
from codercommittee.similarity import Similarity, Space


def _unit(*coordinates: tuple[int, float], size: int = 8) -> list[float]:
    vector = [0.0] * size
    for index, value in coordinates:
        vector[index] = value
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def _chain() -> Similarity:
    """Cause vectors alpha, beta, gamma with cos(alpha, beta) == cos(beta, gamma) and cos(alpha, gamma) == 0."""
    space = Space(
        {
            "alpha": _unit((0, 1.0)),
            "beta": _unit((0, 1.0), (1, 1.0)),
            "gamma": _unit((1, 1.0)),
            "shared effect": _unit((2, 1.0)),
        }
    )
    return Similarity(space)


A = Relation(cause="alpha", effect="shared effect", signal="")
B = Relation(cause="beta", effect="shared effect", signal="")
C = Relation(cause="gamma", effect="shared effect", signal="")


def test_keys_are_casefolded_then_exact_in_name_order() -> None:
    names = ("cause", "effect", "signal")
    assert relation_key(Relation(cause="Rain", effect="Flood"), names) == ("rain", "flood", "", "Rain", "Flood", "")
    assert member_key(Member("m", 2, A, "m"), names) == (relation_key(A, names), 2, "m")


def test_medoid_prefers_the_centre_then_the_key() -> None:
    sim = _chain()
    members = [Member("m", 0, A, "m"), Member("m", 1, B, "m"), Member("m", 2, C, "m")]
    assert medoid(members, sim) is members[1]
    tied = [Member("m", 0, A, "m"), Member("m", 1, A, "m")]
    assert medoid(tied, sim) is tied[0]
    assert medoid([members[2]], sim) is members[2]


def test_dedupe_joins_near_duplicates_and_splits_a_chain_below_the_floor() -> None:
    sim = _chain()
    pair = sim.pair(A, B)
    assert 0.85 <= pair < 0.90  # the chain joins only above the study's threshold: nothing merges here
    assert dedupe([C, A, B], sim) == (A, B, C)
    exact = Space({"x": _unit((0, 1.0)), "y": _unit((1, 1.0))})
    twin = Similarity(exact)
    r1 = Relation(cause="x", effect="y")
    r2 = Relation(cause="x", effect="y", signal="s")
    r3 = Relation(cause="y", effect="x")
    # r1 and r2 are duplicates; the key picks r1 (the empty signal sorts first).
    assert dedupe([r2, r3, r1], twin) == (r1, r3)
    assert dedupe([], twin) == ()
