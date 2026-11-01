from __future__ import annotations

import pytest

from codercommittee.data import Relation, Task
from codercommittee.validation import check_prediction, check_reference, prediction_cap, validate, verbatim

LABEL = Task("label", {"is_ai_patent": ""})
LABELS = Task("labels", {"gun_rights": "", "politics": "", "mental_health": ""}, key="frames", per_answer=(1, 2))
TUPLE = Task("tuple", key="causal")
TUPLES = Task("tuples", key="relations")
TEXT = "Rain caused flooding in the valley and the road closed."


def test_label_requires_the_exact_boolean_object() -> None:
    assert validate(LABEL, {"is_ai_patent": True}, "").value is True
    for bad in ({"is_ai_patent": 1}, {"is_ai_patent": "true"}, {"is_ai_patent": True, "x": 1}, {}, [True], None):
        assert not validate(LABEL, bad, "").ok


def test_labels_are_normalized_unique_and_within_the_codebook() -> None:
    assert validate(LABELS, {"frames": [" Politics ", "gun_rights"]}, "").value == frozenset({"politics", "gun_rights"})
    assert validate(LABELS, {"frames": ["politics", "Politics"]}, "").reason == "labels_must_be_unique"
    assert validate(LABELS, {"frames": ["economy"]}, "").reason == "unknown_labels:economy"
    assert validate(LABELS, {"frames": []}, "").reason == "labels_must_be_array_of_1_to_2"
    assert not validate(LABELS, {"frames": ["politics", "gun_rights", "mental_health"]}, "").ok
    assert not validate(LABELS, {"frames": [1]}, "").ok
    assert not validate(LABELS, {"labels": ["politics"]}, "").ok


def test_tuple_rules_negative_incomplete_and_aux_spans() -> None:
    positive = validate(TUPLE, {"causal": True, "cause": "Rain ", "effect": "flooding", "signal": "caused"}, TEXT)
    assert positive.value == Relation(cause="Rain", effect="flooding", signal="caused")
    assert positive.audit["aux"] == {"signal": "pass"}
    negative = validate(TUPLE, {"causal": False, "cause": "", "effect": "", "signal": "x"}, TEXT)
    assert negative.ok and negative.value is None and negative.audit["aux"] == {"signal": "not_applicable"}
    assert validate(TUPLE, {"causal": False, "cause": "Rain", "effect": "", "signal": ""}, TEXT).reason == (
        "negative_answer_has_spans"
    )
    incomplete = validate(TUPLE, {"causal": True, "cause": "Rain", "effect": "", "signal": "caused"}, TEXT)
    assert incomplete.ok and incomplete.value is None and incomplete.audit["incomplete_read_as_negative"]
    nonverbatim_signal = validate(TUPLE, {"causal": True, "cause": "Rain", "effect": "flooding", "signal": "led"}, TEXT)
    assert nonverbatim_signal.value == Relation(cause="Rain", effect="flooding", signal="")
    assert nonverbatim_signal.audit["aux"] == {"signal": "nonverbatim"}
    assert not validate(TUPLE, {"causal": "yes", "cause": "", "effect": "", "signal": ""}, TEXT).ok
    assert not validate(TUPLE, {"causal": True, "cause": "Rain", "effect": "flooding"}, TEXT).ok


def test_spans_fold_typographic_punctuation_and_keep_the_source_form() -> None:
    text = "The firm\u2019s profit \u2013 after tax \u2013 fell by 5%."
    assert verbatim("The firm's profit", text) == ("The firm\u2019s profit", True)
    assert verbatim("after tax - fell", text) == ("after tax \u2013 fell", True)
    assert verbatim("profit rose", text) == ("profit rose", False)
    assert verbatim("", text) == ("", True)


def test_relations_keep_nonverbatim_spans_and_list_them() -> None:
    answer = {
        "relations": [
            {"cause": "Rain", "effect": "the road closed", "signal": "and"},
            {"cause": "a storm", "effect": "flooding in the valley", "signal": "made"},
        ]
    }
    result = validate(TUPLES, answer, TEXT)
    assert result.ok
    assert result.value == (
        Relation(cause="Rain", effect="the road closed", signal="and"),
        Relation(cause="a storm", effect="flooding in the valley", signal=""),
    )
    assert result.audit["nonverbatim"] == [{"relation": 1, "field": "cause"}]
    assert [entry["status"] for entry in result.audit["aux"]] == ["pass", "nonverbatim"]
    assert validate(TUPLES, {"relations": []}, TEXT).value == ()
    assert validate(TUPLES, {"relations": [{"cause": "", "effect": "x", "signal": ""}]}, TEXT).reason == (
        "relation_0_missing_field"
    )
    assert not validate(TUPLES, {"relations": [{"cause": "a", "effect": "b"}]}, TEXT).ok
    assert not validate(TUPLES, {"relations": [{"cause": "a", "effect": 1, "signal": ""}]}, TEXT).ok
    assert not validate(TUPLES, {"relations": {}}, TEXT).ok
    assert not validate(TUPLES, {"other": []}, TEXT).ok


def test_references_and_prediction_caps() -> None:
    check_reference(LABEL, True)
    check_reference(LABELS, frozenset({"politics"}))
    check_reference(TUPLES, (Relation(cause="a", effect="b"),))
    with pytest.raises(TypeError):
        check_reference(LABEL, frozenset())
    with pytest.raises(ValueError):
        check_reference(LABELS, frozenset({"x"}))
    with pytest.raises(ValueError):
        check_reference(TUPLE, (Relation(cause="a", effect=""),))
    assert prediction_cap(LABELS, 3, 2) == 3 and prediction_cap(LABELS, 1, 1) == 2
    assert prediction_cap(LABEL, 3, 2) is None
    check_prediction(LABELS, frozenset({"politics", "gun_rights", "mental_health"}), 3, 2)
    with pytest.raises(RuntimeError):
        check_prediction(LABELS, frozenset({"politics", "gun_rights", "mental_health"}), 3, 3)
    with pytest.raises(RuntimeError):
        check_prediction(LABEL, None, 3, 2)
    with pytest.raises(RuntimeError):
        check_prediction(TUPLES, (Relation(cause="a", effect=""),), 3, 2)
