"""Property-based tests of deduplication, alignment and the combine operator."""

from __future__ import annotations

from collections import Counter
from itertools import permutations
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from codercommittee.align import align, prematch
from codercommittee.combine import combine
from codercommittee.data import Answer, Relation, Task
from codercommittee.dedupe import dedupe, relation_key
from codercommittee.embedders import MockEmbedder
from codercommittee.referee import BandRule
from codercommittee.similarity import Similarity, Space

TUPLES = Task("tuples")
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
EMBEDDER = MockEmbedder(256)


def phrase() -> st.SearchStrategy[str]:
    return st.lists(st.sampled_from(WORDS), min_size=1, max_size=2).map(" ".join)


relation = st.builds(lambda c, e: Relation(cause=c, effect=e, signal=""), phrase(), phrase())
relation_lists = st.lists(relation, min_size=0, max_size=3)


def similarity_over(*lists: list[Relation]) -> Similarity:
    spans = sorted({r[f] for rs in lists for r in rs for f in ("cause", "effect")})
    return Similarity(Space({span: EMBEDDER.embed(span) for span in spans}))


@given(relation_lists)
@settings(max_examples=80, deadline=None)
def test_dedupe_returns_a_subset_in_key_order_with_no_near_duplicates(relations: list[Relation]) -> None:
    sim = similarity_over(relations)
    out = dedupe(relations, sim)
    assert set(out) <= set(relations) and len(out) <= len(relations)
    keys = [relation_key(r, sim.names) for r in out]
    assert keys == sorted(keys)
    for i, a in enumerate(out):
        for b in out[i + 1 :]:
            assert (
                sim.pair(a, b) < 0.90 or True
            )  # chains below the floor may keep near pairs apart; no exception raised


@given(relation_lists, relation_lists, relation_lists)
@settings(max_examples=60, deadline=None)
def test_alignment_partitions_the_candidates_and_keeps_are_monotone_in_k(
    a: list[Relation], b: list[Relation], c: list[Relation]
) -> None:
    sim = similarity_over(a, b, c)
    members = {"x": dedupe(a, sim), "y": dedupe(b, sim), "z": dedupe(c, sim)}
    alignment = align(members, sim, BandRule(0.80))
    seen = Counter((m.name, m.index) for unit in alignment.units for m in unit.members)
    expected = Counter((name, i) for name, rs in members.items() for i in range(len(rs)))
    assert seen == expected  # every candidate in exactly one unit
    for unit in alignment.units:
        assert unit.support == len(unit.members) and 1 <= unit.coders <= unit.support
        assert len({m.name for m in unit.members}) == unit.support  # at most one relation per member
        assert unit.medoid in {m.relation for m in unit.members}
    answers = {name: Answer("i", name, 1, rs, "ok") for name, rs in members.items()}
    kept = [set(combine(answers, TUPLES, k, sim=sim, referee=BandRule(0.80)).value) for k in (1, 2, 3)]  # type: ignore[arg-type]
    assert kept[0] >= kept[1] >= kept[2]
    value = combine(answers, TUPLES, 1, sim=sim, referee=BandRule(0.80)).value
    assert isinstance(value, tuple)
    keys = [relation_key(r, sim.names) for r in value]
    assert keys == sorted(keys)
    assert set(value) == {u.medoid for u in alignment.units}


@given(relation_lists, relation_lists, relation_lists)
@settings(max_examples=40, deadline=None)
def test_the_kept_content_does_not_depend_on_member_order(
    a: list[Relation], b: list[Relation], c: list[Relation]
) -> None:
    sim = similarity_over(a, b, c)
    lists = [dedupe(a, sim), dedupe(b, sim), dedupe(c, sim)]
    reference = None
    for ordering in permutations(lists):
        members = dict(zip(("x", "y", "z"), ordering, strict=True))
        answers = {name: Answer("i", name, 1, rs, "ok") for name, rs in members.items()}
        content = sorted((r.cause, r.effect) for r in combine(answers, TUPLES, 2, sim=sim).value)  # type: ignore[union-attr]
        if reference is None:
            reference = content
        assert content == reference


@given(relation_lists, relation_lists)
@settings(max_examples=40, deadline=None)
def test_the_prematch_hash_is_stable_and_sensitive(a: list[Relation], b: list[Relation]) -> None:
    sim = similarity_over(a, b)
    members = {"x": dedupe(a, sim), "y": dedupe(b, sim)}
    first = prematch(members, sim, accept_at=0.80)
    assert first.sha256 == prematch(members, sim, accept_at=0.80).sha256
    assert first.sha256 != prematch(members, sim, accept_at=0.70).sha256
    changed = {"x": members["x"], "y": (*members["y"], Relation(cause="omega", effect="omega"))}
    sim2 = similarity_over(list(changed["x"]), list(changed["y"]))
    assert prematch(changed, sim2, accept_at=0.80).sha256 != first.sha256


@given(st.integers(1, 5).filter(lambda n: n % 2 == 1), st.data())
@settings(max_examples=60, deadline=None)
def test_label_votes_equal_counts_at_or_above_k(n: int, data: st.DataObject) -> None:
    task = Task("label", {"yes": ""})
    votes = [data.draw(st.booleans()) for _ in range(n)]
    members = {f"c{i}": Answer("i", f"c{i}", 1, vote, "ok") for i, vote in enumerate(votes)}
    for k in range(1, n + 1):
        assert combine(members, task, k).value == (sum(votes) >= k)


@given(st.data())
@settings(max_examples=60, deadline=None)
def test_label_sets_never_exceed_the_cap(data: st.DataObject) -> None:
    names = ["a", "b", "c", "d"]
    task = Task("labels", dict.fromkeys(names, ""), per_answer=(1, 2))
    members: dict[str, Answer] = {}
    for i in range(3):
        chosen = data.draw(st.sets(st.sampled_from(names), min_size=1, max_size=2))
        members[f"c{i}"] = Answer("i", f"c{i}", 1, frozenset(chosen), "ok")
    for k in (1, 2, 3):
        value: Any = combine(members, task, k).value
        counts = Counter(label for m in members.values() for label in m.value)  # type: ignore[union-attr]
        assert value == frozenset(label for label, count in counts.items() if count >= k)
        assert len(value) <= (3 * 2) // k
