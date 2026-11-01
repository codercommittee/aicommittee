"""Property-based tests of the comparison table, the files and the store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from codercommittee import Item, MockCoder, MockEmbedder, Relation, Run, Task, arms, collect, compare
from codercommittee.compare import Scores
from codercommittee.data import ItemScore
from codercommittee.io import read_items, write_items

LABELS = Task("labels", {"a": "", "b": ""}, per_answer=(1, 2))
TUPLES = Task("tuples")
ARMS = ("single:x", "single:y", "single:z", "self:x", "vote", "pool:2")


@st.composite
def synthetic_scores(draw: st.DrawFn, n_items: int, ids: tuple[Any, ...]) -> Scores:
    values: dict[str, tuple[tuple[ItemScore, ...], ...]] = {}
    for arm in ARMS:
        realizations = 3 if arm.startswith("single") or arm == "vote" else 1
        rows = tuple(
            tuple(ItemScore(draw(st.floats(0.0, 1.0, allow_nan=False))) for _ in range(realizations))
            for _ in range(n_items)
        )
        values[arm] = rows
    items = tuple(Item(i, f"text {i}", frozenset({"a"})) for i in ids)
    return Scores(LABELS, ("x", "y", "z"), items, values)


@given(st.integers(1, 3), st.integers(2, 6), st.data())
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_compare_identities_hold_for_any_scores(n_runs: int, n_items: int, data: st.DataObject) -> None:
    ids = tuple(range(n_items))
    runs = [data.draw(synthetic_scores(n_items, ids)) for _ in range(n_runs)]
    table = compare(runs, draws=20, seed=5)
    singles = ("single:x", "single:y", "single:z")
    for r, s in enumerate(runs):
        levels = s.levels()
        assert table.best_single[r] == max(singles, key=lambda a: (levels[a], -singles.index(a)))
        assert table.blind_pick.per_run[r] == pytest.approx(
            np.mean([levels[a] - levels[table.best_single[r]] for a in singles])
        )
    for arm, row in table.rows.items():
        assert row.levels == tuple(s.level(arm) for s in runs)
        if row.gap is not None:
            assert row.gap.mean == pytest.approx(row.premium - table.spread)
            assert row.gap.worst == min(row.gap.per_run) and (row.gap.sd is None) == (n_runs == 1)
            assert row.gap.interval is not None and row.gap.interval[0] <= row.gap.interval[1]
        else:
            assert all(arm == best for best in table.best_single)
    assert table.blind_pick.mean == pytest.approx(-table.spread)
    again = compare(runs, draws=20, seed=5)
    assert again.rows == table.rows
    shuffled = [Scores(s.task, s.coders, s.items, dict(reversed(list(s.values.items())))) for s in runs]
    assert compare(shuffled, draws=20, seed=5).rows == table.rows


safe_text = st.text(st.characters(blacklist_categories=("Cs", "Cc")), min_size=0, max_size=30)
ids = st.one_of(
    st.integers(-1000, 1000), st.text(st.characters(blacklist_categories=("Cs", "Cc")), min_size=1, max_size=12)
)


@st.composite
def item_sets(draw: st.DrawFn) -> list[Item]:
    count = draw(st.integers(1, 5))
    chosen: list[Item] = []
    seen: set[Any] = set()
    for _ in range(count):
        item_id = draw(ids)
        if item_id in seen:
            continue
        seen.add(item_id)
        reference = draw(st.none() | st.sets(st.sampled_from(["a", "b"]), min_size=1, max_size=2).map(frozenset))
        chosen.append(Item(item_id, draw(safe_text), reference))
    return chosen


@given(item_sets())
@settings(max_examples=60, deadline=None)
def test_items_round_trip_through_jsonl_and_csv(items: list[Item]) -> None:
    root = Path(__import__("tempfile").mkdtemp())
    write_items(items, root / "items.jsonl", LABELS)
    back = read_items(root / "items.jsonl", LABELS)
    assert [(i.id, i.text, i.reference) for i in back] == [(i.id, i.text, i.reference) for i in items]
    if all(isinstance(i.id, str) and not i.id.lstrip("-").isdigit() for i in items):
        write_items(items, root / "items.csv", LABELS)
        back = read_items(root / "items.csv", LABELS)
        assert [(i.id, i.text, i.reference) for i in back] == [(i.id, i.text, i.reference) for i in items]


@given(
    st.lists(
        st.sampled_from(["rain flood", "frost mill town", "bridge mud rain flood town", "so"]),
        min_size=1,
        max_size=3,
        unique=True,
    )
)
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_saved_runs_reload_with_the_same_arms(texts: list[str]) -> None:
    items = [Item(n, text, [Relation(cause=text.split()[0], effect=text.split()[-1])]) for n, text in enumerate(texts)]
    run = collect(items, TUPLES, [MockCoder("x"), MockCoder("y"), MockCoder("z")], embedder=MockEmbedder(64))
    root = Path(__import__("tempfile").mkdtemp()) / "run"
    run.save(root)
    loaded = Run.load(root)
    assert arms(loaded) == arms(run)
    manifest = json.loads((root / "run.json").read_text())
    assert [record["id"] for record in manifest["items"]] == [i.id for i in items]
