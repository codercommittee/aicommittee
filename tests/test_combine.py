from __future__ import annotations

import pytest

from codercommittee.combine import combine
from codercommittee.data import Answer, Relation, Task
from codercommittee.embedders import MockEmbedder
from codercommittee.referee import BandRule
from codercommittee.similarity import Similarity, Space

LABEL = Task("label", {"yes": ""})
LABELS = Task("labels", {"a": "", "b": "", "c": ""}, per_answer=(1, 2))
TUPLE = Task("tuple")
TUPLES = Task("tuples")
RAIN = Relation(cause="Rain", effect="flooding", signal="caused")
HEAVY = Relation(cause="heavy Rain", effect="flooding", signal="")
STRIKE = Relation(cause="seeking a fare hike", effect="strike was launched", signal="")
BUDGET = Relation(cause="budget cuts", effect="protests", signal="")


def answers(task: Task, values: dict[str, object], coders: dict[str, str] | None = None) -> dict[str, Answer]:
    return {
        name: Answer("i", (coders or {}).get(name, name), 1, value, "ok")  # type: ignore[arg-type]
        for name, value in values.items()
    }


def sim_of(*relations: Relation) -> Similarity:
    embedder = MockEmbedder()
    spans = sorted({r[f] for r in relations for f in ("cause", "effect") if r[f]})
    return Similarity(Space({span: embedder.embed(span) for span in spans}))


def test_label_majority_and_single() -> None:
    three = answers(LABEL, {"a": True, "b": False, "c": True})
    assert combine(three, LABEL, 2).value is True
    assert combine(answers(LABEL, {"a": True, "b": False, "c": False}), LABEL, 2).value is False
    assert combine(answers(LABEL, {"a": False}), LABEL, 1).value is False
    assert combine(three, LABEL, 3).value is False
    assert combine(three, LABEL, 2).audit["votes"] == {"a": True, "b": False, "c": True}


def test_labels_keep_what_k_members_support_within_the_cap() -> None:
    three = answers(LABELS, {"a": frozenset({"a", "b"}), "b": frozenset({"b", "c"}), "c": frozenset({"a", "c"})})
    assert combine(three, LABELS, 2).value == frozenset({"a", "b", "c"})  # three labels: floor(3 * 2 / 2)
    assert combine(three, LABELS, 3).value == frozenset()
    assert combine(answers(LABELS, {"a": frozenset({"a", "b"})}), LABELS, 1).value == frozenset({"a", "b"})


def test_tuple_majority_then_the_medoid_of_the_positive_members() -> None:
    sim = sim_of(RAIN, HEAVY, STRIKE)
    three = answers(TUPLE, {"a": RAIN, "b": HEAVY, "c": None})
    assert combine(three, TUPLE, 2, sim=sim).value in (RAIN, HEAVY)
    assert combine(answers(TUPLE, {"a": RAIN, "b": None, "c": None}), TUPLE, 2, sim=sim).value is None
    assert combine(answers(TUPLE, {"a": RAIN}), TUPLE, 1, sim=sim).value == RAIN
    assert combine(answers(TUPLE, {"a": None}), TUPLE, 1).value is None
    with pytest.raises(ValueError, match="majority"):
        combine(three, TUPLE, 1, sim=sim)
    with pytest.raises(ValueError, match="similarity"):
        combine(three, TUPLE, 2)
    assert combine(three, TUPLE, 2, sim=sim).audit == {"positive_votes": 2, "negative_votes": 1}


def test_relations_vote_keeps_two_of_three_and_returns_every_unit() -> None:
    sim = sim_of(RAIN, STRIKE, BUDGET)
    three = answers(TUPLES, {"a": (RAIN, STRIKE), "b": (RAIN, BUDGET), "c": (STRIKE,)})
    prediction = combine(three, TUPLES, 2, sim=sim, referee=BandRule(0.80))
    assert prediction.ok and prediction.value == (RAIN, STRIKE)
    assert sorted(u.support for u in prediction.units) == [1, 2, 2]
    assert prediction.audit["kept"] == 2 and prediction.audit["candidate_pairs"] == 2
    assert combine(three, TUPLES, 1, sim=sim).value == (BUDGET, RAIN, STRIKE)  # candidate-key order
    single = combine(answers(TUPLES, {"a": (STRIKE, RAIN)}), TUPLES, 1, sim=sim)
    assert single.value == (RAIN, STRIKE)
    with pytest.raises(ValueError, match="similarity"):
        combine(three, TUPLES, 2)


def test_pool_members_count_support_by_answers_and_coders_by_structure() -> None:
    sim = sim_of(RAIN, STRIKE)
    pool = answers(
        TUPLES,
        {"x:1": (RAIN,), "x:2": (RAIN,), "y:1": (STRIKE,)},
        coders={"x:1": "x", "x:2": "x", "y:1": "y"},
    )
    two_answers = combine(pool, TUPLES, 2, sim=sim, referee=BandRule(0.70))
    assert two_answers.value == (RAIN,)
    two_coders = combine(pool, TUPLES, 2, 2, sim=sim, referee=BandRule(0.70))
    assert two_coders.value == ()
    rain_unit = next(u for u in two_answers.units if u.medoid == RAIN)
    assert rain_unit.support == 2 and rain_unit.coders == 1


def test_an_invalid_member_makes_the_prediction_insufficient() -> None:
    members = answers(LABEL, {"a": True, "b": True, "c": False})
    members["c"] = Answer("i", "c", 1, None, "refused", reason="provider_refusal")
    prediction = combine(members, LABEL, 2)
    assert prediction.status == "insufficient" and prediction.value is None and prediction.audit == {"excluded": ["c"]}
    with pytest.raises(ValueError):
        combine({}, LABEL, 1)
    with pytest.raises(ValueError):
        combine(members, LABEL, 0)


def test_member_names_with_the_pair_separator_are_refused() -> None:
    relation = Relation(cause="rain", effect="flood")
    sim = Similarity(Space({"rain": [1.0, 0.0], "flood": [0.0, 1.0]}))
    members = {name: Answer(11, name, 1, (relation,), "ok") for name in ("a", "b:0|c", "a:0|b", "c")}
    with pytest.raises(ValueError, match="pair labels reserve"):
        combine(members, TUPLES, 4, sim=sim)
    plain = {name: Answer(11, name, 1, (relation,), "ok") for name in ("m0", "m1", "m2", "m3")}
    prediction = combine(plain, TUPLES, 4, sim=sim)
    assert [unit.support for unit in prediction.units] == [4]
