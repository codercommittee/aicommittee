from __future__ import annotations

from typing import Any

import pytest

from codercommittee import study
from codercommittee.data import Relation


def test_the_five_tasks_carry_the_study_prompts() -> None:
    assert set(study.TASKS) == set(study.DATASETS) == set(study.BOOTSTRAP_SEED)
    kinds = {name: task.kind for name, task in study.TASKS.items()}
    assert kinds == {
        "miric": "label",
        "gvfc": "labels",
        "politicause": "tuple",
        "fincausal": "tuples",
        "causalnews": "tuples",
    }
    for name, task in study.TASKS.items():
        assert task.prompt == study.prompt(name) and task.prompt.count("{{TEXT}}") == 1
        assert len(task.prompt) > 5000  # the study's own prompt files, not a placeholder
    assert study.TASKS["miric"].key == "is_ai_patent" and study.TASKS["gvfc"].key == "frames"
    assert study.TASKS["gvfc"].per_answer == (1, 2) and set(study.TASKS["gvfc"].codebook) == set(study.GVFC_CODEBOOK)
    assert study.TASKS["politicause"].key == "causal" and study.TASKS["causalnews"].names == (
        "cause",
        "effect",
        "signal",
    )
    assert study.task("miric", "{{TEXT}}").prompt == "{{TEXT}}"
    with pytest.raises(ValueError, match="unknown study dataset"):
        study.prompt("other")


def test_schemas_equal_the_study_pipeline_shapes() -> None:
    assert study.TASKS["miric"].schema()["properties"] == {"is_ai_patent": {"type": "boolean"}}
    frames = study.TASKS["gvfc"].schema()["properties"]["frames"]["items"]["enum"]
    assert frames == list(study.GVFC_CODEBOOK)
    assert set(study.TASKS["politicause"].schema()["required"]) == {"causal", "cause", "effect", "signal"}
    relation = study.TASKS["fincausal"].schema()["properties"]["relations"]["items"]
    assert relation["required"] == ["cause", "effect", "signal"] and relation["additionalProperties"] is False


def test_references_from_the_study_gold_shapes() -> None:
    assert study.reference_from_gold("miric", 1) is True and study.reference_from_gold("miric", 0) is False
    assert study.reference_from_gold("gvfc", ["politics", "gun_rights"]) == frozenset({"politics", "gun_rights"})
    assert study.reference_from_gold("politicause", {"causal": False, "references": []}) == ()
    causal: dict[str, Any] = {
        "causal": True,
        "references": [{"coder_id": "c1", "cause": "a", "effect": "b", "signal": "so"}],
    }
    assert study.reference_from_gold("politicause", causal) == (Relation(cause="a", effect="b", signal="so"),)
    gold = [{"cause": "a", "effect": "b", "signal": "", "signals": []}]
    assert study.reference_from_gold("causalnews", gold) == (Relation(cause="a", effect="b"),)
    items = study.items_from_gold("miric", [{"id": "p1", "text": "t", "gold_label": 1}])
    assert items[0].id == "p1" and items[0].reference is True
