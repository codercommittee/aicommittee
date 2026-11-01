from __future__ import annotations

import pytest

from codercommittee.arms import arm_names, arms
from codercommittee.coders import MockCoder
from codercommittee.collect import collect
from codercommittee.data import Item, Relation, Task
from codercommittee.embedders import MockEmbedder

TUPLES = Task("tuples")
LABEL = Task("label", {"yes": ""})
ITEMS = [
    Item(
        "a",
        "the council cut the subsidy so the buses stopped running at night",
        [Relation(cause="the council cut the subsidy", effect="the buses stopped running")],
    ),
    Item(
        "b",
        "heavy rain filled the river and the bridge washed out",
        [Relation(cause="heavy rain", effect="the bridge washed out")],
    ),
    Item("c", "frost hit the orchard early", [Relation(cause="frost", effect="the orchard")]),
]
CODERS = [MockCoder("x"), MockCoder("y"), MockCoder("z")]


def test_arms_cover_the_four_structures_with_their_realizations() -> None:
    run = collect(ITEMS, TUPLES, CODERS, embedder=MockEmbedder())
    out = arms(run)
    assert (
        list(out)
        == arm_names(run)
        == [
            "single:x",
            "single:y",
            "single:z",
            "self:x",
            "self:y",
            "self:z",
            "vote",
            "pool:1",
            "pool:2",
            "pool:3",
        ]
    )
    for item in run.complete():
        assert len(out["single:x"][item.id]) == 3 and len(out["vote"][item.id]) == 3
        assert len(out["self:x"][item.id]) == 1 and len(out["pool:2"][item.id]) == 1
        for draw in (1, 2, 3):
            assert out["single:x"][item.id][draw - 1].value == run.answer(item.id, "x", draw).value
        kept = [set(out[f"pool:{k}"][item.id][0].value) for k in (1, 2, 3)]  # type: ignore[arg-type]
        assert kept[0] >= kept[1] >= kept[2]
        assert out["pool:1"][item.id][0].units == out["pool:3"][item.id][0].units
        assert len(out["pool:1"][item.id][0].value) == len(out["pool:1"][item.id][0].units)  # type: ignore[arg-type]
    label_run = collect([Item("a", "t", True), Item("b", "u", False)], LABEL, CODERS)
    label_arms = arms(label_run)
    assert list(label_arms) == arm_names(label_run) and "pool:1" not in label_arms
    assert all(isinstance(p.value, bool) for p in label_arms["vote"]["a"])


def test_other_committee_sizes_are_refused_at_the_presets() -> None:
    two = collect([Item("a", "t", True)], LABEL, CODERS[:2])
    with pytest.raises(ValueError, match="three coders"):
        arms(two)
    draws = collect([Item("a", "t", True)], LABEL, CODERS, draws=2)
    with pytest.raises(ValueError, match="draws"):
        arms(draws)
    run = collect([Item("a", "t", True)], LABEL, CODERS)
    with pytest.raises(ValueError, match="pool k"):
        arms(run, pool_k=(0,))


def test_incomplete_items_leave_every_arm() -> None:
    def refuse_b(item: Item, task: Task, draw: int) -> object:
        return {"yes": "no"} if item.id == "b" else {"yes": True}

    coders = [MockCoder("x", refuse_b), MockCoder("y"), MockCoder("z")]
    run = collect([Item("a", "t", True), Item("b", "u", False)], LABEL, coders, attempts=1)
    out = arms(run)
    assert set(out["vote"]) == {"a"} and set(out["single:y"]) == {"a"}
