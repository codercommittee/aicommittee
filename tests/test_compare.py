from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from codercommittee.arms import arms
from codercommittee.coders import MockCoder
from codercommittee.collect import collect
from codercommittee.compare import (
    Scores,
    agreement,
    compare,
    iid_plan,
    paired_interval,
    pick_on_sample,
    stratified_plan,
    tied_at_top,
)
from codercommittee.data import Item, ItemScore, Relation, Task
from codercommittee.embedders import MockEmbedder
from codercommittee.scoring import macro_f1

LABEL = Task("label", {"yes": ""})
TUPLES = Task("tuples")
CODERS = [MockCoder("x"), MockCoder("y"), MockCoder("z")]
LABEL_ITEMS = [Item(f"i{n:02d}", f"text number {n} about things", n % 3 == 0) for n in range(12)]
TEXTS = [
    "the council cut the subsidy so the buses stopped running at night",
    "heavy rain filled the river and the bridge washed out before dawn",
    "frost hit the orchard early and the press stood idle all winter",
    "the mill closed in march so forty workers left the town for good",
]
TUPLE_ITEMS = [
    Item(n, text, [Relation(cause=" ".join(text.split()[:3]), effect=" ".join(text.split()[-3:]))])
    for n, text in enumerate(TEXTS)
]


def label_scores(seed: int) -> Scores:
    """One run whose mock coders answer afresh (the seed stands for another run's sampling)."""
    run = collect(LABEL_ITEMS, LABEL, [MockCoder(c, seed=seed) for c in "xyz"])
    return Scores.of(run, arms(run))


def test_levels_follow_the_kind() -> None:
    run = collect(LABEL_ITEMS, LABEL, CODERS)
    scores = Scores.of(run, arms(run))
    assert scores.arms == ("single:x", "single:y", "single:z", "self:x", "self:y", "self:z", "vote")
    assert scores.singles == ("single:x", "single:y", "single:z")
    assert scores.labels == tuple(int(bool(i.reference)) for i in LABEL_ITEMS)
    by_hand = np.mean([macro_f1(cells) for cells in scores.confusions("single:x")])
    assert scores.level("single:x") == pytest.approx(by_hand) and scores.realizations("single:x") == 3
    assert scores.realizations("self:x") == 1 and scores.level("self:x") == macro_f1(scores.confusions("self:x")[0])
    run2 = collect(TUPLE_ITEMS, TUPLES, CODERS, embedder=MockEmbedder())
    scores2 = Scores.of(run2, arms(run2))
    values = scores2.item_values("vote")
    assert values.shape == (4,) and scores2.level("vote") == pytest.approx(float(np.mean(values)))
    assert scores2.level("pool:1") >= 0.0 and set(scores2.arms) >= {"pool:1", "pool:2", "pool:3"}


def test_best_single_ties_fall_to_seat_order() -> None:
    items = (Item("a", "t", True),)
    same = ((ItemScore(1.0, (1, 0, 0, 0, 0, 0)),),)
    scores = Scores(LABEL, ("x", "y"), items, {"single:y": same, "single:x": same})
    assert scores.best_single() == "single:x"


def test_compare_gives_levels_gaps_intervals_blind_pick_spread_and_premium() -> None:
    runs = [label_scores(1), label_scores(2), label_scores(3)]
    table = compare(runs, draws=40, seed=7)
    assert table.runs == 3 and table.arms == runs[0].arms and table.n_items == 12
    for arm, row in table.rows.items():
        assert row.levels == tuple(s.level(arm) for s in runs)
        assert row.level_mean == pytest.approx(float(np.mean(row.levels)))
        if row.gap is not None:
            assert row.gap.mean == pytest.approx(float(np.mean(row.gap.per_run)))
            assert row.gap.worst == min(row.gap.per_run) and row.gap.n_items == 12
            assert row.gap.interval is not None and row.gap.interval[0] <= row.gap.interval[1]
            # gap = premium - spread, run by run and in the mean
            assert row.gap.mean == pytest.approx(row.premium - table.spread)
    assert all(best.startswith("single:") for best in table.best_single)
    best_rows = [table.rows[best].gap for best in set(table.best_single)]
    if len(set(table.best_single)) == 1:
        assert best_rows == [None]
    pairs = zip(runs, table.best_single, strict=True)
    blind_by_hand = [np.mean([s.level(a) - s.level(b) for a in s.singles]) for s, b in pairs]
    assert table.blind_pick.per_run == pytest.approx(tuple(blind_by_hand))
    assert table.blind_pick.mean == pytest.approx(-table.spread)
    assert compare(runs, draws=40, seed=7).rows["vote"].gap == table.rows["vote"].gap  # deterministic
    one = compare(runs[:1], draws=40, seed=7)
    assert one.rows["vote"].level_sd is None and one.rows["vote"].gap is not None and one.rows["vote"].gap.sd is None
    frame = str(table)
    assert "vote" in frame and "blind pick" in frame


def test_compare_on_relations_uses_item_ids_and_refuses_mismatched_runs() -> None:
    run = collect(TUPLE_ITEMS, TUPLES, CODERS, embedder=MockEmbedder())
    scores = Scores.of(run, arms(run))
    table = compare([scores, scores], draws=30, seed=3)
    assert table.rows["pool:1"].gap is not None and table.rows["pool:1"].gap.sd == 0.0
    other = Scores.of(run, {arm: arms(run)[arm] for arm in scores.arms if arm != "pool:3"})
    with pytest.raises(ValueError, match="share"):
        compare([scores, other], draws=10, seed=1)
    with pytest.raises(ValueError):
        compare([], seed=1)


def test_resample_plans_and_intervals_are_reproducible() -> None:
    plan = iid_plan(10, 5, seed=42)
    assert len(plan) == 5 and all(len(draw) == 10 for draw in plan)
    assert [draw.tolist() for draw in plan] == [draw.tolist() for draw in iid_plan(10, 5, seed=42)]
    labels = [1, 0, 0, 1, 0, 0, 0, 1]
    strata = stratified_plan(labels, 3, seed=1)
    assert all(sorted(labels[i] for i in draw) == sorted(labels) for draw in strata)
    deltas = np.asarray([0.1, -0.2, 0.3, 0.0, 0.05])
    low, high = paired_interval(deltas, 200, seed=5)
    assert low <= float(deltas.mean()) <= high
    assert paired_interval(deltas, 200, seed=5) == (low, high) and paired_interval(deltas, 200, seed=6) != (low, high)


def test_agreement_rates() -> None:
    run = collect(LABEL_ITEMS, LABEL, CODERS)
    result = agreement(run)
    assert result.n_items == 12 and 0.0 <= result.cross <= 1.0
    assert set(result.within) == {"x", "y", "z"} and list(result.pairwise) == ["x|y", "x|z", "y|z"]
    assert result.graded is None
    identical = collect(LABEL_ITEMS, LABEL, [MockCoder(c, lambda item, task, draw: {"yes": True}) for c in "xyz"])
    same = agreement(identical)
    assert same.cross == 0.0 and all(v == 0.0 for v in same.within.values())
    assert all(v == 1.0 for v in same.pairwise.values())
    relations = collect(TUPLE_ITEMS, TUPLES, CODERS, embedder=MockEmbedder())
    graded = agreement(relations).graded
    assert graded is not None and graded["match_threshold"] == 0.6
    keys = {"match_threshold", "within_count_range", "cross_count_range"}
    keys |= {"within_matched_jaccard", "cross_matched_jaccard"}
    assert set(graded) == keys
    assert 0.0 <= graded["cross_matched_jaccard"] <= 1.0


def test_pick_on_sample_prefers_the_dominant_single_and_splits_ties() -> None:
    items = tuple(Item(n, "t", True) for n in range(8))

    def column(values: list[float]) -> tuple[tuple[ItemScore, ...], ...]:
        return tuple((ItemScore(v),) for v in values)

    dominant = Scores(
        Task("labels", {"a": ""}),
        ("x", "y", "z"),
        items,
        {"single:x": column([1.0] * 8), "single:y": column([0.5] * 8), "single:z": column([0.0] * 8)},
    )
    pick = pick_on_sample(dominant, 2, draws=20, seed=3)
    assert pick.pick_share == 1.0 and pick.heldout_gap == 0.0 and pick.blind_gap == pytest.approx(-0.5)
    assert pick.among == ("single:x", "single:y", "single:z") and pick.n == 2 and pick.draws == 20
    tied = Scores(
        Task("labels", {"a": ""}),
        ("x", "y", "z"),
        items,
        {"single:x": column([1.0] * 8), "single:y": column([1.0] * 8), "single:z": column([0.0] * 8)},
    )
    half = pick_on_sample(tied, 2, draws=10, seed=3)
    assert half.pick_share == pytest.approx(0.5) and half.heldout_gap == 0.0
    with pytest.raises(ValueError, match="at most 4"):
        pick_on_sample(dominant, 5, seed=3)
    other = pick_on_sample(dominant, 2, among=("single:z",), draws=5, seed=3)
    assert other.pick_share == 0.0 and other.heldout_gap == pytest.approx(-1.0)
    assert tied_at_top(np.asarray([0.2, 0.2 + 1e-13, 0.1])).tolist() == [0, 1]


def test_an_integer_id_and_a_string_id_with_the_same_text_are_two_items() -> None:
    task = Task("labels", {"yes": "", "no": ""})
    items = [Item(7, "first", {"yes"}), Item("7", "second", {"yes"})]

    def script(name: str) -> Any:
        return lambda item, t, draw: {"labels": ["no" if name == "b" and isinstance(item.id, str) else "yes"]}

    coders = [MockCoder(name, script(name)) for name in "abc"]
    scores = collect(items, task, coders).scores()
    result = compare([scores], seed=917)
    deltas = np.array([float(isinstance(item.id, int)) - 1.0 for item in items])
    rng = np.random.default_rng(917)
    draws = [deltas[rng.integers(0, len(items), size=len(items))].mean() for _ in range(2000)]
    expected = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    gap = result.rows["single:b"].gap
    assert gap is not None and result.n_items == 2 and gap.interval == expected and gap.mean == -0.5
    # The order of the common items is by their text, as before (10 before 9).
    ordered = collect([Item(9, "nine", {"yes"}), Item(10, "ten", {"no"})], task, coders).scores()
    assert list(compare([ordered, ordered], seed=1, draws=5).rows) == list(ordered.arms)
