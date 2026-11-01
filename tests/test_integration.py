"""Integration under failure: crashes and resumes, mixed statuses, every kind through the CLI, logged referees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codercommittee import Item, MockCoder, MockEmbedder, Relation, Run, RunStore, Task, arms, collect, compare
from codercommittee.cli import main
from codercommittee.compare import Scores
from codercommittee.data import Answer
from codercommittee.io import write_items
from codercommittee.protocols import Coder, Completion
from codercommittee.referee import LLMReferee, decision_schema, parse_decisions
from codercommittee.store import LoggedReferee, StoreError, StoreReferee

TUPLES = Task("tuples")
LABEL = Task("label", {"yes": ""})
LABELS = Task("labels", {"a": "", "b": "", "c": ""}, per_answer=(1, 2))
TUPLE = Task("tuple")
TEXTS = [
    "heavy rain filled the river so the bridge washed out",
    "the council cut the subsidy and the buses stopped running",
    "frost hit the orchard early and the press stood idle",
    "the mill closed in march so forty workers left town",
    "a new tariff raised prices and the bakery closed its shop",
]
ITEMS = [
    Item(n, t, [Relation(cause=" ".join(t.split()[:3]), effect=" ".join(t.split()[-3:]))]) for n, t in enumerate(TEXTS)
]


class Crashing:
    """A mock coder that raises after a number of calls, as a lost connection would."""

    def __init__(self, name: str, after: int) -> None:
        self.inner = MockCoder(name)
        self.name = name
        self.after = after
        self.calls = 0

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        if self.calls >= self.after:
            raise ConnectionError("lost")
        self.calls += 1
        return self.inner.answer(item, task, draw)


def test_a_crashed_run_resumes_from_its_store_and_equals_an_uninterrupted_one(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    crashing = Crashing("a", 7)
    coders: list[Coder] = [crashing, MockCoder("b"), MockCoder("c")]
    with pytest.raises(ConnectionError):
        collect(ITEMS, TUPLES, coders, embedder=MockEmbedder(64), store=store)
    stored = len((tmp_path / "run" / "answers.jsonl").read_text().splitlines())
    assert stored == 7 + 2 * 3 * 2  # items 0 and 1 fully by b and c, a's seven answers before the crash
    revived = Crashing("a", 10_000)
    fresh: list[Coder] = [revived, MockCoder("b"), MockCoder("c")]
    resumed = collect(ITEMS, TUPLES, fresh, embedder=MockEmbedder(64), store=RunStore(tmp_path / "run"))
    assert revived.calls == 15 - 7
    uninterrupted = collect(ITEMS, TUPLES, [MockCoder("a"), MockCoder("b"), MockCoder("c")], embedder=MockEmbedder(64))
    assert arms(resumed) == arms(uninterrupted)
    assert len((tmp_path / "run" / "answers.jsonl").read_text().splitlines()) == 45


class Scripted:
    """A coder whose behaviour per item is scripted: refuse, fail, recover after two invalid replies, or answer."""

    def __init__(self, name: str, plan: dict[Any, str]) -> None:
        self.inner = MockCoder(name)
        self.name = name
        self.plan = plan
        self.seen: dict[tuple[Any, int], int] = {}

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        kind = self.plan.get(item.id, "answer")
        count = self.seen[(item.id, draw)] = self.seen.get((item.id, draw), 0) + 1
        if kind == "refuse":
            return Answer(item.id, self.name, draw, None, "refused", reason="provider_refusal")
        if kind == "fail" or (kind == "recover" and count < 3):
            return Answer(item.id, self.name, draw, None, "invalid", reason="relations_must_be_array", raw="{}")
        return self.inner.answer(item, task, draw)


def test_mixed_statuses_flow_through_the_store_the_arms_and_the_comparison(tmp_path: Path) -> None:
    coders = [Scripted("a", {4: "fail"}), Scripted("b", {1: "refuse"}), Scripted("c", {2: "recover"})]
    store = RunStore(tmp_path / "run")
    run = collect(ITEMS, TUPLES, coders, embedder=MockEmbedder(64), store=store)
    assert [i.id for i in run.complete()] == [0, 2, 3]
    assert run.answer(1, "b", 1).status == "refused" and run.answer(4, "a", 2).status == "invalid"
    assert run.answer(2, "c", 1).ok and run.answer(2, "c", 1).audit["attempts"] == 3
    assert coders[2].seen[(2, 1)] == 3 and coders[1].seen[(1, 1)] == 3  # the refusal spent its budget
    loaded = Run.load(tmp_path / "run")
    assert [i.id for i in loaded.complete()] == [0, 2, 3]
    assert loaded.answer(1, "b", 1).status == "refused" and loaded.answer(4, "a", 3).status == "invalid"
    predictions = arms(loaded)
    assert set(predictions["vote"]) == {0, 2, 3}
    table = compare([Scores.of(loaded, predictions), Scores.of(run, arms(run))], draws=20, seed=1)
    assert table.n_items == 3
    again = collect(ITEMS, TUPLES, coders, embedder=MockEmbedder(64), store=RunStore(tmp_path / "run"))
    assert coders[1].seen[(1, 1)] == 3  # the refusal spent its budget: final, not asked again
    assert coders[0].seen[(4, 1)] == 6  # the invalid answer was asked again, a full block more
    assert [i.id for i in again.complete()] == [0, 2, 3]


def run_cli(*argv: str) -> int:
    return main(list(argv))


@pytest.mark.parametrize(
    ("task", "suffix", "items"),
    [
        (LABEL, ".csv", [Item(n, f"patent {n}", n % 2 == 0) for n in range(4)]),
        (LABELS, ".csv", [Item(n, f"headline {n}", {"a", "b"} if n % 2 else {"c"}) for n in range(4)]),
        (
            TUPLE,
            ".jsonl",
            [
                Item(n, t, [Relation(cause=t.split()[0], effect=t.split()[-1])] if n % 2 else [])
                for n, t in enumerate(TEXTS)
            ],
        ),
        (TUPLES, ".json", ITEMS),
    ],
    ids=["label", "labels", "tuple", "tuples"],
)
def test_every_kind_runs_through_the_command_line(
    tmp_path: Path, task: Task, suffix: str, items: list[Item], capsys: pytest.CaptureFixture[str]
) -> None:
    items_path = tmp_path / f"items{suffix}"
    if suffix == ".json":
        records = []
        for item in items:
            assert isinstance(item.reference, tuple)
            records.append({"id": item.id, "text": item.text, "reference": [r.as_dict() for r in item.reference]})
        items_path.write_text(json.dumps(records))
    else:
        write_items(items, items_path, task)
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps(
            {"kind": task.kind, "codebook": dict(task.codebook), "key": task.key, "per_answer": list(task.per_answer)}
        )
    )
    argv = [
        "code",
        "--task",
        str(task_path),
        "--items",
        str(items_path),
        "--out",
        str(tmp_path / "run"),
        "--int-ids",
        "--coder",
        "a=mock",
        "--coder",
        "b=mock:1",
        "--coder",
        "c=mock:2",
    ]
    if task.kind in ("tuple", "tuples"):
        argv += ["--embedder", "mock:64"]
    assert run_cli(*argv) == 0
    out = capsys.readouterr().out
    assert "items complete" in out and (tmp_path / "run" / "levels.json").is_file()
    loaded = Run.load(tmp_path / "run")
    assert loaded.task.kind == task.kind and [i.id for i in loaded.items] == [i.id for i in items]
    assert run_cli("score", str(tmp_path / "run")) == 0
    assert run_cli("agreement", str(tmp_path / "run")) == 0
    assert run_cli("compare", str(tmp_path / "run"), str(tmp_path / "run"), "--seed", "1", "--draws", "10") == 0
    if len(loaded.scored()) >= 2:
        assert run_cli("pick", str(tmp_path / "run"), "--n", "1", "--seed", "1", "--draws", "5") == 0


def test_tasks_written_by_the_cli_load_back_as_tasks(tmp_path: Path) -> None:
    assert run_cli("tasks", "gvfc", "--out", str(tmp_path / "gvfc.json")) == 0
    items = tmp_path / "items.jsonl"
    write_items([Item("h1", "a headline", frozenset({"politics"}))], items, Task("labels", {"politics": ""}))
    assert (
        run_cli(
            "code",
            "--task",
            str(tmp_path / "gvfc.json"),
            "--items",
            str(items),
            "--out",
            str(tmp_path / "run"),
            "--coder",
            "a=mock",
            "--coder",
            "b=mock",
            "--coder",
            "c=mock",
        )
        == 0
    )
    assert Run.load(tmp_path / "run").task.key == "frames"


class FakeModel:
    """Answers every candidate pair by its band, as a model following the prompt would."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, schema: Any, *, system: str) -> Completion:
        self.calls += 1
        pairs = schema["properties"]["decisions"]["items"]["properties"]["pair"]["enum"]
        decisions = [
            {
                "pair": pair,
                "decision": "confirm" if "high band" in prompt.split(pair, 1)[1].split("\n")[0] else "reject",
            }
            for pair in pairs
        ]
        return Completion(json.dumps({"decisions": decisions}), {"requested_model": "fake"})


def test_an_llm_referee_logged_into_the_store_replays_exactly(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    run = collect(
        ITEMS, TUPLES, [MockCoder("a"), MockCoder("b"), MockCoder("c")], embedder=MockEmbedder(64), store=store
    )
    model = FakeModel()
    logged = LoggedReferee(LLMReferee(model, template="{{PAIRS}}"), store)
    live = arms(run, logged)
    assert model.calls > 0 and store.has_decisions
    assert arms(Run.load(tmp_path / "run")) == live
    assert arms(run, StoreReferee(store)) == live
    records = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    records[0]["prematch_sha256"] = "0" * 64
    (tmp_path / "run" / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(StoreError, match="differs from the recomputed"):
        arms(Run.load(tmp_path / "run"))
    schema = decision_schema
    assert callable(schema) and callable(parse_decisions)
