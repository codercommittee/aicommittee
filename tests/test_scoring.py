from __future__ import annotations

import math

import pytest

from codercommittee.data import Item, Prediction, Relation, Task
from codercommittee.scoring import jaccard, macro_f1, score
from codercommittee.similarity import Similarity, Space

LABEL = Task("label", {"is_ai_patent": ""})
LABELS = Task("labels", dict.fromkeys(("gun_rights", "politics", "public_opinion"), ""), per_answer=(1, 2))
TUPLE = Task("tuple")
TUPLES = Task("tuples")


def _unit(*coordinates: tuple[int, float], size: int = 6) -> list[float]:
    vector = [0.0] * size
    for index, value in coordinates:
        vector[index] = value
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def _sim() -> Similarity:
    space = Space(
        {
            "rain": _unit((0, 1.0)),
            "heavy rain": _unit((0, 1.0), (1, 1.0)),
            "flooding": _unit((2, 1.0)),
            "drought": _unit((3, 1.0)),
            "heat": _unit((4, 1.0)),
        }
    )
    return Similarity(space)


def test_label_accuracy_and_the_macro_f1_oracle() -> None:
    pairs = [(True, True), (False, True), (False, False), (False, False)]
    scores = [score(predicted, actual, LABEL) for predicted, actual in pairs]
    assert [s.value for s in scores] == [1.0, 0.0, 1.0, 1.0]
    assert scores[1].confusion == (0, 0, 1, 0, 1, 0)
    assert macro_f1([s.confusion for s in scores if s.confusion]) == pytest.approx(11 / 15)
    assert macro_f1([(1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0)]) == 1.0
    with pytest.raises(TypeError):
        score(1, True, LABEL)  # type: ignore[arg-type]


def test_labels_jaccard_oracle() -> None:
    predicted = frozenset({"gun_rights", "politics", "public_opinion"})
    assert score(predicted, frozenset({"gun_rights", "politics"}), LABELS).value == pytest.approx(2 / 3)
    assert jaccard(frozenset(), frozenset()) == 1.0 and jaccard(frozenset({"a"}), frozenset()) == 0.0


def test_tuple_scores_every_reference_and_gates_on_detection() -> None:
    sim = _sim()
    prediction = Relation(cause="rain", effect="flooding", signal="caused")
    references = (
        Relation(cause="rain", effect="flooding", signal="caused"),
        Relation(cause="heavy rain", effect="flooding", signal="caused"),
    )
    exact, near = (sim.pair(prediction, reference) for reference in references)
    assert exact == 1.0 and 0.0 < near < 1.0
    assert score(prediction, references, TUPLE, sim).value == pytest.approx((exact + near) / 2)
    assert score(None, references, TUPLE, sim).value == 0.0
    assert score(None, (), TUPLE).value == 1.0 and score(prediction, (), TUPLE).value == 0.0
    with pytest.raises(ValueError):
        score(prediction, references, TUPLE)


def test_relations_match_one_to_one_above_the_threshold_and_penalize_counts() -> None:
    sim = _sim()
    rain = Relation(cause="rain", effect="flooding")
    heavy = Relation(cause="heavy rain", effect="flooding")
    heat = Relation(cause="heat", effect="drought")
    result = score((heavy, heat), (rain, heat), TUPLES, sim)
    pair = sim.pair(heavy, rain)
    assert pair >= 0.60
    assert {(m.prediction, m.reference) for m in result.matched} == {(0, 0), (1, 1)}
    cause = (sim.field(heavy, rain, "cause") + 1.0) / 2
    effect = (1.0 + 1.0) / 2
    assert result.value == pytest.approx(0.5 * (cause + effect))
    assert score((rain,), (rain, heat), TUPLES, sim).value == pytest.approx(0.5)
    assert score((rain, heavy, heat), (rain, heat), TUPLES, sim).value == pytest.approx(2 / 3)
    assert score((), (), TUPLES).value == 1.0 and score((rain,), (), TUPLES).value == 0.0
    far = score((Relation(cause="heat", effect="flooding"),), (Relation(cause="rain", effect="drought"),), TUPLES, sim)
    assert far.value == 0.0 and far.matched == ()


def test_predictions_and_missing_references() -> None:
    assert score(Prediction(True, "ok"), True, LABEL).value == 1.0
    with pytest.raises(ValueError):
        score(Prediction(None, "insufficient"), True, LABEL)
    with pytest.raises(ValueError):
        score(True, Item("a", "t").reference, LABEL)
