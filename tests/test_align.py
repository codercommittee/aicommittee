"""Alignment: pairwise matching, the pre-match, the band rule, the weakest-edge rule, units."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import permutations
from typing import Any

import pytest

from codercommittee.align import Prematch, align, prematch
from codercommittee.data import Member, Relation
from codercommittee.embedders import MockEmbedder
from codercommittee.matching import canonical_pair_matching
from codercommittee.referee import BandRule
from codercommittee.similarity import Similarity, Space

RAIN = Relation(cause="Rain", effect="flooding", signal="caused")
STRIKE = Relation(cause="seeking a fare hike", effect="strike was launched", signal="")
BUDGET = Relation(cause="budget cuts", effect="protests", signal="sparked")
RAIN_VARIANT = Relation(cause="heavy Rain", effect="flooding", signal="caused")
NAMES = ("anthropic", "openai", "google")


def members_of(*relation_lists: list[Relation]) -> dict[str, list[Relation]]:
    return {name: list(relations) for name, relations in zip(NAMES, relation_lists, strict=False)}


def similarity(*relations: Relation) -> Similarity:
    """Every cause and effect span mock-embedded, as the study's tests do."""
    embedder = MockEmbedder()
    spans = sorted({relation[field] for relation in relations for field in ("cause", "effect") if relation[field]})
    return Similarity(Space({span: embedder.embed(span) for span in spans}))


def _unit(*coordinates: tuple[int, float], size: int = 8) -> list[float]:
    vector = [0.0] * size
    for index, value in coordinates:
        vector[index] = value
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def chain_similarity() -> Similarity:
    """Cause vectors alpha, beta, gamma with cos(alpha, beta) == cos(beta, gamma) exactly and cos(alpha, gamma) == 0."""
    return Similarity(
        Space(
            {
                "alpha": _unit((0, 1.0)),
                "beta": _unit((0, 1.0), (1, 1.0)),
                "gamma": _unit((1, 1.0)),
                "shared effect": _unit((2, 1.0)),
            }
        )
    )


class Fixed:
    """A referee answering from a mapping (or confirming everything when none is given)."""

    def __init__(self, decisions: Mapping[str, bool] | None = None, accept_at: float = 0.80) -> None:
        self.decisions = decisions
        self.accept_at = accept_at

    def decide(self, pre: Prematch) -> dict[str, bool]:
        if self.decisions is None:
            return dict.fromkeys(pre.candidate_pairs, True)
        return dict(self.decisions)


def kept(alignment: Any, k: int = 2) -> list[tuple[str, str]]:
    return sorted((u.medoid.cause, u.medoid.effect) for u in alignment.units if u.support >= k)


def test_full_triangle_forms_one_unit_of_three() -> None:
    members = members_of([RAIN], [RAIN], [RAIN])
    sim = similarity(RAIN)
    for referee in (BandRule(0.80), Fixed()):
        alignment = align(members, sim, referee)
        assert len(alignment.units) == 1
        unit = alignment.units[0]
        assert unit.support == 3 and unit.coders == 3 and unit.medoid == RAIN
        assert sorted(m.name for m in unit.members) == ["anthropic", "google", "openai"]
        assert alignment.dropped == ()
        assert [edge.band for edge in alignment.prematch.edges] == ["high"] * 3


def test_disjoint_inputs_give_singletons_and_nothing_to_decide() -> None:
    members = members_of([RAIN], [BUDGET], [STRIKE])
    sim = similarity(RAIN, BUDGET, STRIKE)
    alignment = align(members, sim, BandRule(0.80))
    assert alignment.prematch.candidate_pairs == ()
    assert [u.support for u in alignment.units] == [1, 1, 1] and kept(alignment) == []
    assert align(members, sim, Fixed({})).units == alignment.units


def test_support_two_units_survive_and_the_unique_relation_is_a_singleton() -> None:
    members = members_of([RAIN, STRIKE], [RAIN, BUDGET], [STRIKE])
    alignment = align(members, similarity(RAIN, STRIKE, BUDGET), BandRule(0.80))
    assert kept(alignment) == [("Rain", "flooding"), ("seeking a fare hike", "strike was launched")]
    singles = [u for u in alignment.units if u.support == 1]
    assert len(singles) == 1 and singles[0].medoid == BUDGET


def test_member_permutation_invariance() -> None:
    relation_sets = [[RAIN, STRIKE], [RAIN_VARIANT, BUDGET], [STRIKE, RAIN]]
    sim = similarity(RAIN, STRIKE, RAIN_VARIANT, BUDGET)
    for referee in (BandRule(0.80), Fixed()):
        baseline = None
        for ordering in permutations(relation_sets):
            content = kept(align(members_of(*ordering), sim, referee))
            if baseline is None:
                baseline = content
            assert content == baseline


def test_draw_named_members_form_a_self_ensemble_with_one_coder() -> None:
    members = {"1": [RAIN], "2": [RAIN], "3": [RAIN]}
    alignment = align(members, similarity(RAIN), BandRule(0.80), coders=dict.fromkeys(members, "anthropic"))
    unit = alignment.units[0]
    assert unit.support == 3 and unit.coders == 1 and [m.name for m in unit.members] == ["1", "2", "3"]


def test_canonical_matching_is_optimal_and_content_deterministic() -> None:
    sim = similarity(RAIN, STRIKE)
    left = [Member("a", 0, RAIN, "a"), Member("a", 1, STRIKE, "a")]
    right = [Member("b", 0, STRIKE, "b"), Member("b", 1, RAIN, "b")]
    matching, used_fallback = canonical_pair_matching(left, right, 0.60, sim)
    assert used_fallback is False
    assert {(row, col) for row, col, _ in matching} == {(0, 1), (1, 0)}
    assert all(value >= 0.60 for _, _, value in matching)


def test_enumeration_guard_falls_back_deterministically_and_is_recorded() -> None:
    sim = similarity(RAIN)
    many = [RAIN] * 6
    left = [Member("a", index, relation, "a") for index, relation in enumerate(many)]
    right = [Member("b", index, relation, "b") for index, relation in enumerate(many)]
    matching, used_fallback = canonical_pair_matching(left, right, 0.60, sim)
    assert used_fallback is True and len(matching) == 6
    assert len({row for row, _, _ in matching}) == 6 and len({col for _, col, _ in matching}) == 6
    swapped, swapped_fallback = canonical_pair_matching(right, left, 0.60, sim)
    assert swapped_fallback is True and len(swapped) == 6
    dense = {"anthropic": many, "openai": many, "google": [RAIN]}
    pre = prematch(dense, sim, accept_at=0.80)
    assert pre.fallback_pairs == ("anthropic|openai",)
    assert kept(align(dense, sim, BandRule(0.80)))


def test_tied_weakest_edges_drop_symmetrically_under_permutation() -> None:
    x = Relation(cause="alpha", effect="shared effect", signal="")
    y = Relation(cause="beta", effect="shared effect", signal="")
    z = Relation(cause="gamma", effect="shared effect", signal="")
    sim = chain_similarity()
    assert sim.pair(x, y) == sim.pair(y, z) and sim.pair(x, z) == 0.5
    reference = None
    for ordering in permutations([[x], [y], [z]]):
        alignment = align(members_of(*ordering), sim, BandRule(0.80))
        assert alignment.prematch.conflict_components == 1
        # The two edges tie on similarity; the content key orders them, so exactly one drops.
        assert len(alignment.dropped) == 1 and sorted(u.support for u in alignment.units) == [1, 2]
        content = kept(alignment, k=1)
        if reference is None:
            reference = content
        assert content == reference


def test_bands_decisions_and_the_decision_contract() -> None:
    x = Relation(cause="alpha", effect="shared effect", signal="")
    y = Relation(cause="beta", effect="shared effect", signal="")
    sim = chain_similarity()
    members = members_of([x], [y], [])
    pre = prematch(members, sim, accept_at=0.80)
    assert pre.candidate_pairs == ("anthropic:0|openai:0",)
    assert pre.edges[0].band == "high" and pre.edges[0].in_conflict is False
    assert kept(align(members, sim, BandRule(0.80))) == [("alpha", "shared effect")] or kept(
        align(members, sim, BandRule(0.80))
    ) == [("beta", "shared effect")]
    rejected = align(members, sim, Fixed({"anthropic:0|openai:0": False}))
    assert kept(rejected) == [] and rejected.decisions == {"anthropic:0|openai:0": False}
    with pytest.raises(ValueError, match="every candidate pair"):
        align(members, sim, Fixed({}))
    with pytest.raises(ValueError, match="every candidate pair"):
        align(members, sim, Fixed({"anthropic:0|openai:0": True, "made:0|up:0": False}))
    uncertain = prematch(members, sim, accept_at=0.90)
    assert uncertain.edges[0].band == "uncertain"
    assert kept(align(members, sim, BandRule(0.90))) == []
    with pytest.raises(ValueError, match="banded at another threshold"):
        BandRule(0.80).decide(uncertain)


def test_prematch_dictionary_shape_and_hash() -> None:
    members = members_of([RAIN, STRIKE], [RAIN], [STRIKE])
    pre = prematch(members, similarity(RAIN, STRIKE), accept_at=0.80, item="i", label="panel:1", text="t")
    record = pre.as_dict()
    assert list(record) == [
        "low_threshold",
        "high_threshold",
        "nodes",
        "edges",
        "candidate_pairs",
        "pairwise_match_counts",
        "enumeration_fallback_pairs",
        "conflict_components",
    ]
    assert record["low_threshold"] == 0.6 and record["high_threshold"] == 0.8
    assert [node["node"] for node in record["nodes"]] == ["anthropic:0", "anthropic:1", "google:0", "openai:0"]
    assert set(record["edges"][0]) == {
        "pair",
        "a",
        "b",
        "similarity",
        "cause_similarity",
        "effect_similarity",
        "band",
        "in_conflict",
    }
    assert record["candidate_pairs"] == [edge["pair"] for edge in record["edges"]] == sorted(record["candidate_pairs"])
    assert record["pairwise_match_counts"] == {"anthropic|google": 1, "anthropic|openai": 1, "google|openai": 0}
    assert len(pre.sha256) == 64 and pre.sha256 == prematch(members, similarity(RAIN, STRIKE), accept_at=0.80).sha256
    assert pre.item == "i" and pre.label == "panel:1" and pre.text == "t"
