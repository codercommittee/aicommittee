"""The re-audit of 0.2.2: credentials in error text, an interrupted save, the referee's identity, a retry
after a failed write, and the referee record of a hand-built run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import codercommittee.store as store_module
from codercommittee import (
    BandRule,
    Item,
    LLMReferee,
    MockCoder,
    MockEmbedder,
    ModelCoder,
    Relation,
    Run,
    RunStore,
    StoreError,
    Task,
    collect,
)
from codercommittee.data import Answer
from codercommittee.providers import OpenAICompatibleModel
from codercommittee.providers.base import scrub_credentials
from codercommittee.store import LoggedReferee, StoreReferee

LABEL = Task("label", {"yes": ""})
TUPLES = Task("tuples")
ITEMS = [
    Item(n, text, [Relation(cause=" ".join(text.split()[:2]), effect=" ".join(text.split()[-2:]))])
    for n, text in enumerate(
        [
            "heavy rain filled the river so the bridge washed out",
            "the council cut the subsidy and the buses stopped running",
            "frost hit the orchard early and the press stood idle",
        ]
    )
]
SECRET = "SYNTHETIC_PASSWORD"
URL = f"https://audit:{SECRET}@gateway.invalid/v1"


class Raising:
    """A chat-completions client whose every call raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0
        self.chat = type("C", (), {})()
        self.chat.completions = type("D", (), {})()
        self.chat.completions.create = self._create

    def _create(self, **arguments: Any) -> Any:
        self.calls += 1
        raise self.exc


def _referee_model(verdict: str, base_url: str) -> OpenAICompatibleModel:
    class Deciding:
        def __init__(self) -> None:
            self.calls = 0
            self.chat = type("C", (), {})()
            self.chat.completions = type("D", (), {})()
            self.chat.completions.create = self._create

        def _create(self, **arguments: Any) -> Any:
            self.calls += 1
            schema = arguments["response_format"]["json_schema"]["schema"]
            pairs = schema["properties"]["decisions"]["items"]["properties"]["pair"]["enum"]
            raw = json.dumps({"decisions": [{"pair": p, "decision": verdict} for p in pairs]})
            message = type("M", (), {"content": raw, "refusal": None})()
            choice = type("Ch", (), {"message": message, "finish_reason": "stop"})()
            return type("R", (), {"choices": [choice], "model": "m", "id": "r", "usage": None})()

    return OpenAICompatibleModel("m", base_url=base_url, client=Deciding())


def test_credentials_never_reach_an_answers_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import codercommittee.providers.base as base

    monkeypatch.setattr(base, "_sleep", lambda seconds: None)
    client = Raising(TimeoutError(f"Cannot connect to {URL}"))
    model = OpenAICompatibleModel("m", base_url=URL, client=client)
    run = collect([Item(1, "text", True)], LABEL, [ModelCoder("a", model)], draws=1, store=RunStore(tmp_path / "run"))
    answer = run.answer(1, "a", 1)
    assert answer.status == "invalid" and client.calls == 3
    assert SECRET not in (answer.reason or "") and SECRET not in (tmp_path / "run" / "answers.jsonl").read_text()
    assert "gateway.invalid/v1" in (answer.reason or "")
    assert scrub_credentials("x https://u:p@h/v1?k=a@b y") == "x https://h/v1?k=a@b y"
    assert scrub_credentials("no url here") == "no url here"


def test_an_interrupted_save_leaves_the_seats_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = collect([Item(1, "ordinary text", True)], LABEL, [MockCoder("a", model="model-one")])
    original = RunStore.put_answer
    calls = 0

    def interrupted(self: RunStore, answer: Answer, task: Task) -> None:
        nonlocal calls
        calls += 1
        original(self, answer, task)
        if calls == 1:
            raise OSError("simulated crash after the first answer")

    monkeypatch.setattr(RunStore, "put_answer", interrupted)
    with pytest.raises(OSError, match="simulated"):
        run.save(tmp_path / "run")
    monkeypatch.setattr(RunStore, "put_answer", original)
    manifest = json.loads((tmp_path / "run" / "run.json").read_text())
    assert manifest["provenance"]["coder_identities"]["a"]["model"] == "model-one"
    with pytest.raises(StoreError, match="coder 'a' was"):
        collect(
            [Item(1, "ordinary text", True)],
            LABEL,
            [MockCoder("a", model="model-two")],
            store=RunStore(tmp_path / "run"),
        )
    # A manifest from before the record existed: the stored answers' own model guards the seat.
    del manifest["provenance"]["coder_identities"]
    (tmp_path / "run" / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(StoreError, match="from model 'model-one', not 'model-two'"):
        collect(
            [Item(1, "ordinary text", True)],
            LABEL,
            [MockCoder("a", model="model-two")],
            store=RunStore(tmp_path / "run"),
        )
    resumed = collect(
        [Item(1, "ordinary text", True)], LABEL, [MockCoder("a", model="model-one")], store=RunStore(tmp_path / "run")
    )
    assert {a.identity["requested_model"] for a in resumed.answers.values()} == {"model-one"}


def test_the_referee_record_binds_the_endpoint_and_the_prompts(tmp_path: Path) -> None:
    coders = [MockCoder("a"), MockCoder("b"), MockCoder("c")]
    first = _referee_model("confirm", "https://first.invalid/v1")
    store = RunStore(tmp_path / "run")
    run = collect(
        ITEMS, TUPLES, coders, embedder=MockEmbedder(64), store=store, referee=LoggedReferee(LLMReferee(first), store)
    )
    live = run.arms()
    assert first._client.calls > 0
    second = _referee_model("reject", "https://second.invalid/v1")
    with pytest.raises(StoreError, match="another referee"):
        LoggedReferee(LLMReferee(second), RunStore(tmp_path / "run"))
    same = _referee_model("reject", "https://first.invalid/v1")
    served = LoggedReferee(LLMReferee(same), RunStore(tmp_path / "run"))
    assert (
        collect(
            ITEMS, TUPLES, coders, embedder=MockEmbedder(64), store=RunStore(tmp_path / "run"), referee=served
        ).arms()
        == live
    )
    assert same._client.calls == 0
    with pytest.raises(StoreError, match="another referee"):
        LoggedReferee(LLMReferee(first, template="{{PAIRS}}"), RunStore(tmp_path / "run"))
    with pytest.raises(StoreError, match="another referee"):
        LoggedReferee(LLMReferee(first, system="Decide at random."), RunStore(tmp_path / "run"))
    record = LLMReferee(first).describe()
    assert record["base_url"] == "https://first.invalid/v1" and len(record["template_sha256"]) == 16
    # A record written by 0.2.2 (provider, model and prompt version only) is upgraded, not refused.
    manifest_path = tmp_path / "run" / "run.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["referee"] = {
        "kind": "llm",
        "accept_at": 0.8,
        "provider": "openai_compatible",
        "model": "m",
        "prompt_version": record["prompt_version"],
    }
    manifest_path.write_text(json.dumps(manifest))
    collect(
        ITEMS,
        TUPLES,
        coders,
        embedder=MockEmbedder(64),
        store=RunStore(tmp_path / "run"),
        referee=LoggedReferee(LLMReferee(first), RunStore(tmp_path / "run")),
    )
    assert (
        json.loads(manifest_path.read_text())["provenance"]["referee"]["template_sha256"] == record["template_sha256"]
    )


def test_a_retry_after_a_failed_write_leaves_a_readable_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = store_module._append
    count = 0

    def failing(path: Path, record: Any) -> None:
        nonlocal count
        count += 1
        if count == 2:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record)[:50])  # the write stops half way
            raise OSError("simulated interrupted write")
        original(path, record)

    monkeypatch.setattr(store_module, "_append", failing)
    store = RunStore(tmp_path / "run")
    coders = [MockCoder("a"), MockCoder("b"), MockCoder("c")]
    with pytest.raises(OSError, match="simulated"):
        collect([Item(1, "text", True)], LABEL, coders, store=store)
    resumed = collect([Item(1, "text", True)], LABEL, coders, store=store)  # the same store object
    assert len(resumed.complete()) == 1
    assert len(Run.load(tmp_path / "run").complete()) == 1
    lines = (tmp_path / "run" / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 9 and all(isinstance(json.loads(line), dict) for line in lines)


def test_a_hand_built_run_records_its_referee_and_a_custom_logged_referee_is_served(tmp_path: Path) -> None:
    base = collect(
        ITEMS, TUPLES, [MockCoder("a"), MockCoder("b", seed=1), MockCoder("c", seed=2)], embedder=MockEmbedder(64)
    )
    manual = Run(base.task, base.items, base.coders, base.draws, base.answers, base.spaces, BandRule(0.7))
    manual.save(tmp_path / "manual")
    loaded = Run.load(tmp_path / "manual")
    assert isinstance(loaded.referee, BandRule) and loaded.referee.accept_at == 0.7
    assert loaded.scores().level("vote") == manual.scores().level("vote")

    class RejectAll:
        accept_at = 0.8

        def decide(self, prematch: Any) -> dict[str, bool]:
            return dict.fromkeys(prematch.candidate_pairs, False)

    store = RunStore(tmp_path / "custom")
    run = collect(
        ITEMS,
        TUPLES,
        [MockCoder("a"), MockCoder("b"), MockCoder("c")],
        embedder=MockEmbedder(64),
        store=store,
        referee=LoggedReferee(RejectAll(), store),
    )
    before = run.scores().level("vote")
    restored = Run.load(tmp_path / "custom")
    assert isinstance(restored.referee, StoreReferee) and restored.scores().level("vote") == before
    (tmp_path / "custom" / "decisions.jsonl").unlink()
    with pytest.raises(StoreError, match="no stored decision"):
        Run.load(tmp_path / "custom").scores()
    label_run = collect([Item(1, "t", True)], LABEL, [MockCoder("a"), MockCoder("b"), MockCoder("c")])
    Run(label_run.task, label_run.items, label_run.coders, label_run.draws, label_run.answers, {}, RejectAll()).save(
        tmp_path / "label"
    )
    assert Run.load(tmp_path / "label").scores().level("vote") == label_run.scores().level("vote")
    # An emptier stored seat record does not bypass the answers check.
    store_dir = tmp_path / "seat"
    collect([Item(1, "t", True)], LABEL, [MockCoder("a", model="model-one")], store=RunStore(store_dir))
    manifest = json.loads((store_dir / "run.json").read_text())
    manifest["provenance"]["coder_identities"]["a"] = {"kind": "mock"}
    (store_dir / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(StoreError, match="from model 'model-one'"):
        collect([Item(1, "t", True)], LABEL, [MockCoder("a", model="model-two")], store=RunStore(store_dir))
    assert scrub_credentials("https://u:p@ss@h/v1 and https://h?k=a@b") == "https://h/v1 and https://h?k=a@b"


def test_the_referee_object_wins_over_a_stale_record_and_a_failed_append_is_not_asked_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    run = collect(
        ITEMS,
        TUPLES,
        [MockCoder("a"), MockCoder("b"), MockCoder("c")],
        embedder=MockEmbedder(64),
        referee=BandRule(0.8),
    )
    manual = replace(run, referee=BandRule(0.7))
    manual.save(tmp_path / "run")
    loaded = Run.load(tmp_path / "run")
    assert isinstance(loaded.referee, BandRule) and loaded.referee.accept_at == 0.7
    assert loaded.scores().level("vote") == manual.scores().level("vote")
    # The bytes landed but the append raised: the same store object re-reads the file, nothing is asked twice.
    original = store_module._append
    count = 0

    def landed_then_raised(path: Path, record: Any) -> None:
        nonlocal count
        count += 1
        original(path, record)
        if count == 2:
            raise OSError("simulated failure after the write")

    monkeypatch.setattr(store_module, "_append", landed_then_raised)
    store = RunStore(tmp_path / "landed")
    coders = [MockCoder("a"), MockCoder("b"), MockCoder("c")]
    with pytest.raises(OSError, match="simulated"):
        collect([Item(1, "text", True)], LABEL, coders, store=store)
    collect([Item(1, "text", True)], LABEL, coders, store=store)
    lines = (tmp_path / "landed" / "answers.jsonl").read_text().splitlines()
    assert len(lines) == 9 and store.answered() == 9


def test_secrets_in_any_form_are_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import codercommittee.providers.base as base

    monkeypatch.setattr(base, "_sleep", lambda seconds: None)
    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "sk-SYNTHETIC_KEY_VALUE")
    assert scrub_credentials("https://audit:SE@CR@PW@gateway.invalid/v1 x") == "https://gateway.invalid/v1 x"
    assert scrub_credentials("https:\\/\\/audit:PW@gateway.invalid\\/v1") == "https:\\/\\/gateway.invalid\\/v1"
    assert scrub_credentials("key sk-SYNTHETIC_KEY_VALUE rejected") == "key [redacted] rejected"
    url = "https://audit:SE@CR@SYNTHETIC_PASSWORD@gateway.invalid/v1"
    model = OpenAICompatibleModel(
        "m",
        base_url=url,
        api_key="sk-literal-key",
        client=Raising(TimeoutError(f"{url} and bare SE@CR@SYNTHETIC_PASSWORD and sk-literal-key")),
    )
    answer = collect(
        [Item(1, "text", True)], LABEL, [ModelCoder("a", model)], draws=1, store=RunStore(tmp_path / "run")
    ).answer(1, "a", 1)
    saved = (tmp_path / "run" / "answers.jsonl").read_text()
    for secret in ("SYNTHETIC_PASSWORD", "sk-literal-key"):
        assert secret not in (answer.reason or "") and secret not in saved

    # A seat is also guarded by the endpoint its stored answers name.
    class Replying:
        def __init__(self) -> None:
            self.chat = type("C", (), {})()
            self.chat.completions = type("D", (), {})()
            self.chat.completions.create = self._create

        def _create(self, **arguments: Any) -> Any:
            message = type("M", (), {"content": '{"yes": true}', "refusal": None})()
            choice = type("Ch", (), {"message": message, "finish_reason": "stop"})()
            return type("R", (), {"choices": [choice], "model": "m", "id": "r", "usage": None})()

    first = OpenAICompatibleModel("m", base_url="https://gateway.invalid/v1", client=Replying())
    collect([Item(1, "text", True)], LABEL, [ModelCoder("a", first)], draws=1, store=RunStore(tmp_path / "seat"))
    manifest_path = tmp_path / "seat" / "run.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["provenance"]["coder_identities"]
    manifest_path.write_text(json.dumps(manifest))
    assert '"base_url": "https://gateway.invalid/v1"' in (tmp_path / "seat" / "answers.jsonl").read_text()
    other = OpenAICompatibleModel("m", base_url="https://other.invalid/v1", client=Replying())
    with pytest.raises(StoreError, match="from base_url"):
        collect([Item(1, "text", True)], LABEL, [ModelCoder("a", other)], draws=1, store=RunStore(tmp_path / "seat"))
