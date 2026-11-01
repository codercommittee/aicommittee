from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from codercommittee import (
    BandRule,
    Committee,
    Item,
    MockCoder,
    MockEmbedder,
    Relation,
    Scores,
    Task,
    agreement,
    arms,
    compare,
)

ROOT = Path(__file__).resolve().parents[1]
TUPLES = Task("tuples")
ITEMS = [
    Item(
        1,
        "heavy rain filled the river so the bridge washed out",
        [Relation(cause="heavy rain", effect="the bridge washed out")],
    ),
    Item(
        2,
        "the council cut the subsidy and the buses stopped running",
        [Relation(cause="the council cut the subsidy", effect="the buses stopped running")],
    ),
    Item(3, "frost hit the orchard early and the press stood idle"),
]


def test_the_facade_reaches_every_core_call() -> None:
    committee = Committee([MockCoder("a"), MockCoder("b"), MockCoder("c")], MockEmbedder(), BandRule(0.80))
    run = committee.code(ITEMS, TUPLES)
    assert run.coders == ("a", "b", "c") and run.referee is committee.referee
    assert list(run.arms()) == list(arms(run))
    assert run.predictions("vote") == arms(run)["vote"]
    assert set(run.units("pool:2")) == {item.id for item in run.complete()}
    assert all(len(realizations) == 1 for realizations in run.units("pool:2").values())
    assert all(len(realizations) == 3 for realizations in run.units("vote").values())
    scores = run.scores()
    assert isinstance(scores, Scores) and [item.id for item in scores.items] == [1, 2]  # item 3 has no reference
    assert run.agreement() == agreement(run)
    table = compare([scores, run.scores()], seed=1)
    assert table.runs == 2
    with pytest.raises(KeyError, match="no arm"):
        run.predictions("nothing")
    with pytest.raises(ValueError):
        Committee([])
    assert "Committee([a, b, c]" in repr(committee)


def test_frames_need_pandas_only_when_asked() -> None:
    pandas = pytest.importorskip("pandas")
    run = Committee([MockCoder("a"), MockCoder("b"), MockCoder("c")]).code(
        [Item("x", "t", True)], Task("label", {"y": ""})
    )
    frame = run.to_frame()
    assert (
        isinstance(frame, pandas.DataFrame)
        and len(frame) == 9
        and set(frame.columns) >= {"item", "coder", "draw", "status"}
    )
    scores = run.scores()
    assert len(scores.to_frame()) == 3 * 3 + 3 + 3  # singles x 3 draws, selfs, vote x 3
    table = compare([scores], seed=1).to_frame()
    assert list(table.index) == list(scores.arms)


def test_quickstart_runs_end_to_end_in_under_a_minute() -> None:
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-W", "error", str(ROOT / "quickstart.py")], capture_output=True, text=True
    )
    elapsed = time.perf_counter() - started
    assert result.returncode == 0, result.stderr
    assert elapsed < 60
    out = result.stdout
    assert "levels of run 1" in out and "two runs compared" in out and "pool:2" in out and "blind pick" in out
