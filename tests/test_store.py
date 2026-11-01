"""Saved, resumable and replayable runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codercommittee import (
    BandRule,
    Item,
    LLMReferee,
    MockCoder,
    MockEmbedder,
    Relation,
    Run,
    RunStore,
    StoreError,
    Task,
    arms,
    collect,
    write_items,
)
from codercommittee.align import prematch
from codercommittee.data import Answer
from codercommittee.protocols import Completion
from codercommittee.store import LoggedReferee, StoreReferee, answer_from_record, task_to_json

TUPLES = Task("tuples")
LABEL = Task("label", {"yes": ""})
ITEMS = [
    Item(n, text, [Relation(cause=" ".join(text.split()[:2]), effect=" ".join(text.split()[-2:]))])
    for n, text in enumerate(
        [
            "heavy rain filled the river so the bridge washed out",
            "the council cut the subsidy and the buses stopped running",
            "frost hit the orchard early and the press stood idle",
            "the mill closed in march so forty workers left town",
        ]
    )
]


class Counting:
    """A mock coder that counts how often it is asked."""

    def __init__(self, name: str, script: Any = None) -> None:
        self.inner = MockCoder(name, script)
        self.name = name
        self.calls = 0

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        self.calls += 1
        return self.inner.answer(item, task, draw)


def relations_of(answer: Answer) -> tuple[Relation, ...]:
    """The relations of an ok relation-list answer, typed."""
    assert isinstance(answer.value, tuple)
    return answer.value


def test_save_then_load_gives_the_same_run_and_arms(tmp_path: Path) -> None:
    run = collect(ITEMS, TUPLES, [MockCoder("a"), MockCoder("b"), MockCoder("c")], embedder=MockEmbedder())
    run.save(tmp_path / "run")
    assert (tmp_path / "run" / "run.json").is_file() and len(list((tmp_path / "run" / "spaces").iterdir())) == 4
    back = Run.load(tmp_path / "run")
    assert back.task == run.task and back.coders == run.coders and back.draws == run.draws
    assert [i.id for i in back.items] == [i.id for i in run.items] and back.items[0].reference == run.items[0].reference
    for key, answer in run.answers.items():
        assert back.answers[key].status == answer.status and back.answers[key].value == answer.value
    assert set(back.spaces[0].texts) == set(run.spaces[0].texts)
    assert arms(back) == arms(run)
    run.save(tmp_path / "run")  # saving again is idempotent
    assert len((tmp_path / "run" / "answers.jsonl").read_text().splitlines()) == 36


def test_collect_with_a_store_resumes_and_never_reasks_final_answers(tmp_path: Path) -> None:
    coders = [Counting("a"), Counting("b"), Counting("c")]
    store = RunStore(tmp_path / "run")
    first = collect(ITEMS, TUPLES, coders, embedder=MockEmbedder(), store=store)
    assert [c.calls for c in coders] == [12, 12, 12] and store.answered() == 36
    again = collect(ITEMS, TUPLES, coders, embedder=MockEmbedder(), store=RunStore(tmp_path / "run"))
    assert [c.calls for c in coders] == [12, 12, 12]
    assert arms(again) == arms(first)
    # An invalid stored answer is asked again; a refusal that spent its budget is not; one that did not is.
    lines = (tmp_path / "run" / "answers.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    records[0].update(status="invalid", reason="x", raw=None, value=None)
    records[1].update(status="refused", reason="provider_refusal", raw=None, value=None, audit={"attempts": 3})
    records[2].update(status="refused", reason="provider_refusal", raw=None, value=None, audit={"attempts": 1})
    (tmp_path / "run" / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    third = collect(ITEMS, TUPLES, coders, embedder=MockEmbedder(), store=RunStore(tmp_path / "run"))
    assert [c.calls for c in coders] == [14, 12, 12]
    assert third.answer(records[1]["item"], records[1]["coder"], records[1]["draw"]).status == "refused"
    assert third.answer(records[0]["item"], records[0]["coder"], records[0]["draw"]).ok
    assert third.answer(records[2]["item"], records[2]["coder"], records[2]["draw"]).ok
    with pytest.raises(StoreError, match="another run"):
        collect(ITEMS[:2], TUPLES, coders, embedder=MockEmbedder(), store=RunStore(tmp_path / "run"))
    with pytest.raises(StoreError, match="holds no run"):
        Run.load(tmp_path / "nowhere")


def test_stored_values_are_never_authoritative(tmp_path: Path) -> None:
    item = Item("x", "t", True)
    record = {
        "item": "x",
        "coder": "a",
        "draw": 1,
        "status": "ok",
        "raw": '{"yes": false}',
        "value": {"yes": True},
        "identity": {},
    }
    answer = answer_from_record(record, item, LABEL)
    assert answer.ok and answer.value is False  # the raw reply decides
    assert answer_from_record({**record, "raw": "{bad"}, item, LABEL).reason == "strict_json_parse_failed"
    assert answer_from_record({**record, "raw": None}, item, LABEL).status == "invalid"
    assert answer_from_record({**record, "status": "refused", "reason": "r"}, item, LABEL).status == "refused"


def test_logged_decisions_replay_through_the_store_referee(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    run = collect(ITEMS, TUPLES, [MockCoder("a"), MockCoder("b"), MockCoder("c")], embedder=MockEmbedder(), store=store)
    logged = LoggedReferee(BandRule(0.80), store)
    live = arms(run, logged)
    assert store.has_decisions
    replayed = arms(Run.load(tmp_path / "run"))  # the loaded run carries a StoreReferee
    assert replayed == live
    item = run.items[0]
    sim = run.similarity(item.id)
    assert sim is not None
    members = {c: relations_of(run.answer(item.id, c, 1)) for c in run.coders}
    pre = prematch(members, sim, accept_at=0.80, item=item.id, label="panel:1", text=item.text)
    referee = StoreReferee(store)
    if pre.candidate_pairs:
        assert referee.decide(pre) == BandRule(0.80).decide(pre)
        record = store.decision(item.id, "panel:1")
        assert record is not None and record["prematch_sha256"] == pre.sha256
    with pytest.raises(StoreError, match="no stored decision"):
        other = prematch(members, sim, accept_at=0.80, item=item.id, label="panel:9", text=item.text)
        if not other.candidate_pairs:
            raise StoreError("no stored decision (nothing to decide)")
        referee.decide(other)
    with pytest.raises(ValueError, match="banded at another threshold"):
        referee.decide(prematch(members, sim, accept_at=0.70, item=item.id, label="panel:1"))


class FakeRefereeModel:
    """Confirms or rejects every candidate pair it is shown, counting its calls."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls = 0

    def complete(self, prompt: str, schema: Any, **kwargs: Any) -> Completion:
        self.calls += 1
        pairs = schema["properties"]["decisions"]["items"]["properties"]["pair"]["enum"]
        decisions = [{"pair": pair, "decision": self.verdict} for pair in pairs]
        return Completion(json.dumps({"decisions": decisions}), {"requested_model": "fake"})


def _coders() -> list[MockCoder]:
    return [MockCoder("a"), MockCoder("b"), MockCoder("c")]


def test_a_resumed_run_serves_its_recorded_decisions_without_a_referee_call(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    first = FakeRefereeModel("confirm")
    logged = LoggedReferee(LLMReferee(first), store)
    run = collect(ITEMS, TUPLES, _coders(), embedder=MockEmbedder(64), store=store, referee=logged)
    live = run.arms()
    assert first.calls > 0
    second = FakeRefereeModel("reject")
    again_store = RunStore(tmp_path / "run")
    logged = LoggedReferee(LLMReferee(second), again_store)
    again = collect(ITEMS, TUPLES, _coders(), embedder=MockEmbedder(64), store=again_store, referee=logged)
    assert again.arms() == live and second.calls == 0
    assert len((tmp_path / "run" / "decisions.jsonl").read_text().splitlines()) == len(again_store.decisions())


def test_the_referee_configuration_is_part_of_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from codercommittee import cli

    (tmp_path / "task.json").write_text(json.dumps(task_to_json(TUPLES)))
    write_items(ITEMS, tmp_path / "items.jsonl", TUPLES)
    base = ["code", "--task", str(tmp_path / "task.json"), "--items", str(tmp_path / "items.jsonl")]
    base += ["--coder", "a=mock", "--coder", "b=mock:1", "--coder", "c=mock:2", "--embedder", "mock:64"]
    assert cli.main([*base, "--out", str(tmp_path / "band"), "--referee", "band:0.7"]) == 0
    levels = json.loads((tmp_path / "band" / "levels.json").read_text())["levels"]
    loaded = Run.load(tmp_path / "band")
    assert isinstance(loaded.referee, BandRule) and loaded.referee.accept_at == 0.7
    assert loaded.scores().level("vote") == levels["vote"]
    assert loaded.provenance["referee"] == {"kind": "band", "accept_at": 0.7}
    seats = [MockCoder("a"), MockCoder("b", seed=1), MockCoder("c", seed=2)]
    with pytest.raises(StoreError, match="another referee"):
        collect(
            loaded.items,
            TUPLES,
            seats,
            embedder=MockEmbedder(64),
            store=RunStore(tmp_path / "band"),
            referee=BandRule(0.8),
        )
    # A manifest written before the record existed accepts a referee and gains the record.
    manifest_path = tmp_path / "band" / "run.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["provenance"]
    manifest_path.write_text(json.dumps(manifest))
    assert Run.load(tmp_path / "band").referee is None
    collect(
        loaded.items, TUPLES, seats, embedder=MockEmbedder(64), store=RunStore(tmp_path / "band"), referee=BandRule(0.7)
    )
    assert json.loads(manifest_path.read_text())["provenance"]["referee"] == {"kind": "band", "accept_at": 0.7}
    # The LLM referee through the command line: the second invocation makes no call and the levels stand.
    models = [FakeRefereeModel("confirm"), FakeRefereeModel("reject")]
    monkeypatch.setattr(cli, "parse_model", lambda spec, settings=None: models.pop(0))
    llm = [*base, "--out", str(tmp_path / "llm"), "--referee", "llm:openai:fake"]
    assert cli.main(llm) == 0
    first_levels = json.loads((tmp_path / "llm" / "levels.json").read_text())["levels"]
    assert cli.main(llm) == 0
    assert json.loads((tmp_path / "llm" / "levels.json").read_text())["levels"] == first_levels
    assert not models
    loaded = Run.load(tmp_path / "llm")
    assert isinstance(loaded.referee, StoreReferee) and loaded.scores().level("vote") == first_levels["vote"]
    assert loaded.provenance["referee"]["kind"] == "llm"


def test_saving_a_loaded_run_keeps_its_decisions_and_an_unlogged_llm_referee_is_refused(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "a")
    model = FakeRefereeModel("confirm")
    logged = LoggedReferee(LLMReferee(model), store)
    run = collect(ITEMS, TUPLES, _coders(), embedder=MockEmbedder(64), store=store, referee=logged)
    live = run.arms()
    made = model.calls
    Run.load(tmp_path / "a").save(tmp_path / "b")
    assert RunStore(tmp_path / "b").decisions() == RunStore(tmp_path / "a").decisions()
    copied = Run.load(tmp_path / "b")
    assert copied.arms() == live and model.calls == made
    assert copied.provenance["referee"]["kind"] == "llm" and copied.provenance["referee"]["model"] is None
    copied.save(tmp_path / "b")  # idempotent
    unlogged = collect(
        ITEMS, TUPLES, _coders(), embedder=MockEmbedder(64), referee=LLMReferee(FakeRefereeModel("confirm"))
    )
    with pytest.raises(StoreError, match="LoggedReferee"):
        unlogged.save(tmp_path / "c")


def _cli_base(tmp_path: Path, out: str) -> list[str]:
    (tmp_path / "task.json").write_text(json.dumps(task_to_json(TUPLES)))
    write_items(ITEMS, tmp_path / "items.jsonl", TUPLES)
    base = ["code", "--task", str(tmp_path / "task.json"), "--items", str(tmp_path / "items.jsonl")]
    base += ["--coder", "a=mock", "--coder", "b=mock:1", "--coder", "c=mock:2", "--embedder", "mock:64"]
    return [*base, "--out", str(tmp_path / out)]


def test_a_resume_without_the_referee_flag_keeps_the_stored_referee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from codercommittee import cli

    models = [FakeRefereeModel("confirm")]
    monkeypatch.setattr(cli, "parse_model", lambda spec, settings=None: models.pop(0))
    base = _cli_base(tmp_path, "run")
    assert cli.main([*base, "--referee", "llm:openai:fake"]) == 0
    levels_path = tmp_path / "run" / "levels.json"
    first = json.loads(levels_path.read_text())["levels"]
    assert cli.main(base) == 0  # no flag: the stored referee serves, nothing is asked, the levels stand
    assert json.loads(levels_path.read_text())["levels"] == first
    assert Run.load(tmp_path / "run").scores().level("vote") == first["vote"]
    second = FakeRefereeModel("reject")
    models.append(second)
    assert cli.main(["score", str(tmp_path / "run"), "--referee", "llm:openai:fake"]) == 0
    assert second.calls == 0 and not models  # logged into the store, served from it
    assert cli.main(["score", str(tmp_path / "run"), "--referee", "band:0.7"]) == 0  # the band rule overrides freely
    band = _cli_base(tmp_path, "band")
    assert cli.main([*band, "--referee", "band:0.7"]) == 0
    models.append(FakeRefereeModel("confirm"))
    assert cli.main(["score", str(tmp_path / "band"), "--referee", "llm:openai:fake"]) == 2
    assert "another referee" in capsys.readouterr().err


def test_a_refused_save_leaves_the_target_manifest_alone(tmp_path: Path) -> None:
    store_c = RunStore(tmp_path / "c")
    run_c = collect(
        ITEMS,
        TUPLES,
        _coders(),
        embedder=MockEmbedder(64),
        store=store_c,
        referee=LoggedReferee(BandRule(0.8), store_c),
    )
    run_c.arms()
    manifest_path = tmp_path / "c" / "run.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["provenance"]["referee"]
    manifest_path.write_text(json.dumps(manifest))
    store_a = RunStore(tmp_path / "a")
    logged = LoggedReferee(LLMReferee(FakeRefereeModel("reject")), store_a)
    run_a = collect(ITEMS, TUPLES, _coders(), embedder=MockEmbedder(64), store=store_a, referee=logged)
    run_a.arms()
    with pytest.raises(StoreError, match="different decision"):
        Run.load(tmp_path / "a").save(tmp_path / "c")
    assert "referee" not in json.loads(manifest_path.read_text())["provenance"]


def test_a_logged_referee_must_agree_with_the_manifest(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    logged = LoggedReferee(LLMReferee(FakeRefereeModel("confirm")), store)
    run = collect(ITEMS, TUPLES, _coders(), embedder=MockEmbedder(64), store=store, referee=logged)
    run.arms()
    with pytest.raises(StoreError, match="another referee"):
        LoggedReferee(BandRule(0.7), RunStore(tmp_path / "run"))
    fresh = LoggedReferee(BandRule(0.8), RunStore(tmp_path / "later" / "deep"))  # created on the first decision
    arms(run, fresh)
    assert (tmp_path / "later" / "deep" / "decisions.jsonl").is_file()


def test_malformed_referee_records_and_missing_decisions_are_store_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from codercommittee import cli

    store = RunStore(tmp_path / "run")
    logged = LoggedReferee(LLMReferee(FakeRefereeModel("confirm")), store)
    run = collect(ITEMS, TUPLES, _coders(), embedder=MockEmbedder(64), store=store, referee=logged)
    run.arms()
    (tmp_path / "run" / "decisions.jsonl").unlink()
    with pytest.raises(StoreError, match="no stored decision"):
        Run.load(tmp_path / "run").scores()
    assert cli.main(["score", str(tmp_path / "run")]) == 2 and "no stored decision" in capsys.readouterr().err
    manifest_path = tmp_path / "run" / "run.json"
    manifest = json.loads(manifest_path.read_text())
    for bad in (None, [0.8], 0, 1.5, "x", True):
        manifest["provenance"]["referee"] = {"kind": "llm", "accept_at": bad}
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(StoreError, match="threshold"):
            Run.load(tmp_path / "run")
    manifest["provenance"]["referee"] = {"kind": "oracle", "accept_at": 0.8}
    manifest_path.write_text(json.dumps(manifest))
    assert isinstance(Run.load(tmp_path / "run").referee, StoreReferee)  # served from its records
    assert cli.main(["score", str(tmp_path / "run")]) == 2 and "no stored decision" in capsys.readouterr().err


def test_decision_identities_do_not_count_and_thresholds_are_floats(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "a")
    run = collect(
        ITEMS, TUPLES, _coders(), embedder=MockEmbedder(64), store=store, referee=LoggedReferee(BandRule(1), store)
    )
    run.arms()
    assert run.referee is not None and run.referee.accept_at == 1.0
    assert json.loads((tmp_path / "a" / "run.json").read_text())["provenance"]["referee"] == {
        "kind": "band",
        "accept_at": 1.0,
    }
    Run.load(tmp_path / "a").save(tmp_path / "a")
    store = RunStore(tmp_path / "llm")
    logged = LoggedReferee(LLMReferee(FakeRefereeModel("confirm")), store)
    collect(ITEMS, TUPLES, _coders(), embedder=MockEmbedder(64), store=store, referee=logged).arms()
    Run.load(tmp_path / "llm").save(tmp_path / "b")
    path = tmp_path / "b" / "decisions.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        record["identity"] = {"provider_response_id": "other"}
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    Run.load(tmp_path / "llm").save(tmp_path / "b")  # identity differences are not conflicts
    assert len(path.read_text().splitlines()) == len(records)


def test_a_seat_keeps_its_model_across_a_resume(tmp_path: Path) -> None:
    class Interrupted(MockCoder):
        def answer(self, item: Item, task: Task, draw: int) -> Answer:
            if draw == 2:
                raise RuntimeError("simulated interruption")
            return super().answer(item, task, draw)

    items = [Item(11, "a legal text", True)]
    with pytest.raises(RuntimeError):
        collect(items, LABEL, [Interrupted("a", model="model-one")], store=RunStore(tmp_path / "run"))
    manifest = json.loads((tmp_path / "run" / "run.json").read_text())
    assert manifest["provenance"]["coder_identities"] == {"a": {"kind": "mock", "model": "model-one", "seed": 0}}
    with pytest.raises(StoreError, match="coder 'a' was"):
        collect(items, LABEL, [MockCoder("a", model="model-two")], store=RunStore(tmp_path / "run"))
    resumed = collect(items, LABEL, [MockCoder("a", model="model-one")], store=RunStore(tmp_path / "run"))
    assert {a.identity["requested_model"] for a in resumed.answers.values()} == {"model-one"}
    # A manifest from before the record existed: the stored answers' own model guards the seat.
    del manifest["provenance"]["coder_identities"]
    (tmp_path / "run" / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(StoreError, match="from model 'model-one'"):
        collect(items, LABEL, [MockCoder("a", model="model-three")], store=RunStore(tmp_path / "run"))
    collect(items, LABEL, [MockCoder("a", model="model-one")], store=RunStore(tmp_path / "run"))
    manifest = json.loads((tmp_path / "run" / "run.json").read_text())
    assert manifest["provenance"]["coder_identities"]["a"]["model"] == "model-one"
    # A coder without describe() is recorded by its class; a loaded run keeps the stored identities.
    collect(items, LABEL, [Counting("b")], store=RunStore(tmp_path / "other"))
    loaded = Run.load(tmp_path / "other")
    assert loaded.provenance["coder_identities"] == {"b": {"kind": "Counting"}}
    loaded.save(tmp_path / "copy")
    assert RunStore(tmp_path / "copy").provenance()["coder_identities"] == {"b": {"kind": "Counting"}}
