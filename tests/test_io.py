from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from codercommittee import (
    Item,
    MockCoder,
    MockEmbedder,
    Relation,
    Task,
    arms,
    collect,
    compare,
    read_items,
    write_comparison,
    write_items,
    write_predictions,
    write_scores,
)
from codercommittee.compare import Scores, agreement, pick_on_sample
from codercommittee.io import read_jsonl, reference_from_json, to_json, write_agreement, write_levels, write_pick

LABEL = Task("label", {"yes": ""})
LABELS = Task("labels", {"a": "", "b": "", "c": ""}, per_answer=(1, 2))
TUPLE = Task("tuple")
TUPLES = Task("tuples")


def items_of(task: Task) -> list[Item]:
    if task.kind == "label":
        return [Item("p1", "a text", True), Item("p2", "another", False), Item("p3", "no reference")]
    if task.kind == "labels":
        return [Item(1, "head", {"a", "b"}), Item(2, "line", {"c"}), Item(3, "none")]
    if task.kind == "tuple":
        return [
            Item(1, "rain caused floods", [Relation(cause="rain", effect="floods", signal="caused")]),
            Item(2, "quiet", []),
        ]
    return [
        Item(
            "r1",
            "rain caused floods and mud",
            [Relation(cause="rain", effect="floods"), Relation(cause="rain", effect="mud")],
        )
    ]


@pytest.mark.parametrize("suffix", [".jsonl", ".csv", ".json"])
@pytest.mark.parametrize("task", [LABEL, LABELS, TUPLE, TUPLES], ids=lambda t: t.kind)
def test_items_round_trip(tmp_path: Path, task: Task, suffix: str) -> None:
    items = items_of(task)
    path = tmp_path / f"items{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps([json.loads(line) for line in _jsonl(items, task, tmp_path)]))
    else:
        write_items(items, path, task)
    back = read_items(path, task, int_ids=(suffix == ".csv"))
    assert [(i.id, i.text, i.reference) for i in back] == [(i.id, i.text, i.reference) for i in items]


def _jsonl(items: list[Item], task: Task, tmp_path: Path) -> list[str]:
    path = tmp_path / "tmp.jsonl"
    write_items(items, path, task)
    return path.read_text().splitlines()


def test_csv_holds_json_references_and_keeps_string_ids(tmp_path: Path) -> None:
    path = tmp_path / "items.csv"
    write_items(items_of(TUPLES), path, TUPLES)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["id"] == "r1" and json.loads(rows[0]["reference"])[0] == {
        "cause": "rain",
        "effect": "floods",
        "signal": "",
    }
    assert read_items(path, TUPLES)[0].id == "r1"
    numeric = tmp_path / "numeric.csv"
    numeric.write_text("id,text,reference\n7,seven,\n")
    assert read_items(numeric, LABEL)[0].id == "7" and read_items(numeric, LABEL, int_ids=True)[0].id == 7
    assert read_items(numeric, LABEL)[0].reference is None


def test_errors_name_the_line() -> None:
    with pytest.raises(ValueError, match="label reference is a JSON boolean"):
        reference_from_json("yes", LABEL)
    with pytest.raises(ValueError, match="outside the task"):
        reference_from_json([{"cause": "a", "effect": "b", "extra": "c"}], TUPLES)
    with pytest.raises(ValueError, match="outside the codebook"):
        reference_from_json(["zzz"], LABELS)


def test_bad_files_fail_with_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text('{"id": "a", "text": "t"}\n{"id": "b"}\n')
    with pytest.raises(ValueError, match=r"items.jsonl:2: an item needs an id and a text"):
        read_items(path, LABEL)
    path.write_text('{"id": "a", "text": "t"}\n{"id": "a", "text": "u"}\n')
    with pytest.raises(ValueError, match="duplicate item id"):
        read_items(path, LABEL)
    path.write_text('{"id": "a", "text": "t", "reference": 1}\n')
    with pytest.raises(ValueError, match=r"items\.jsonl:1"):
        read_items(path, LABEL)
    path.write_text("")
    with pytest.raises(ValueError, match="no items"):
        read_items(path, LABEL)
    path.write_text("{bad\n")
    with pytest.raises(ValueError, match="not strict JSON"):
        read_items(path, LABEL)
    (tmp_path / "items.json").write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="JSON array"):
        read_items(tmp_path / "items.json", LABEL)


def test_result_writers_are_readable_back(tmp_path: Path) -> None:
    items = [
        Item(n, f"rain caused floods and mud number {n}", [Relation(cause="rain", effect="floods")]) for n in range(6)
    ]
    run = collect(items, TUPLES, [MockCoder("a"), MockCoder("b"), MockCoder("c")], embedder=MockEmbedder())
    predictions = arms(run)
    write_predictions(predictions, TUPLES, tmp_path / "predictions.jsonl")
    rows = read_jsonl(tmp_path / "predictions.jsonl")
    assert {row["arm"] for row in rows} == set(predictions)
    assert all(isinstance(row["value"], list) for row in rows if row["status"] == "ok")
    scores = Scores.of(run, predictions)
    write_scores(scores, tmp_path / "scores.csv")
    with (tmp_path / "scores.csv").open(newline="") as handle:
        lines = list(csv.DictReader(handle))
    assert len(lines) == sum(len(r) for by_item in scores.values.values() for r in by_item)
    write_scores(scores, tmp_path / "scores.json")
    assert json.loads((tmp_path / "scores.json").read_text())["levels"] == pytest.approx(scores.levels())
    write_levels(scores, tmp_path / "levels.json")
    assert json.loads((tmp_path / "levels.json").read_text())["best_single"] == scores.best_single()
    table = compare([scores, scores], draws=20, seed=1)
    write_comparison(table, tmp_path / "table.csv")
    with (tmp_path / "table.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["arm"] for row in csv_rows] == list(table.arms) and "gap_low" in csv_rows[0]
    write_comparison(table, tmp_path / "table.json")
    loaded = json.loads((tmp_path / "table.json").read_text())
    assert (
        loaded["runs"] == 2
        and set(loaded["rows"]) == set(table.arms)
        and loaded["blind_pick"]["mean"] == table.blind_pick.mean
    )
    write_agreement(agreement(run), tmp_path / "agreement.json")
    assert "cross" in json.loads((tmp_path / "agreement.json").read_text())
    write_pick(pick_on_sample(scores, 2, draws=5, seed=1), tmp_path / "pick.json")
    assert json.loads((tmp_path / "pick.json").read_text())["n"] == 2
    assert to_json({"x": frozenset({"b", "a"}), "r": Relation(cause="c", effect="e")}) == {
        "x": ["a", "b"],
        "r": {"cause": "c", "effect": "e"},
    }
