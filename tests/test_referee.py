"""The LLM referee under a fake model, and the replay of stored decisions."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from codercommittee.align import align, prematch
from codercommittee.data import Answer, Relation
from codercommittee.embedders import MockEmbedder
from codercommittee.protocols import Completion
from codercommittee.referee import (
    PROMPT_VERSION,
    LLMReferee,
    RefereeError,
    ReplayReferee,
    decision_schema,
    identity_acceptable,
    load_prompt,
    parse_decisions,
    render_prompt,
)
from codercommittee.replay import ReplayError, StoredRun, load_run
from codercommittee.similarity import Similarity, Space
from codercommittee.study import items_from_gold as items_for
from codercommittee.study import task as study_task

RAIN = Relation(cause="Rain", effect="flooding", signal="caused")
RAIN_VARIANT = Relation(cause="heavy Rain", effect="flooding", signal="caused")
STRIKE = Relation(cause="seeking a fare hike", effect="strike was launched", signal="")


def similarity(*relations: Relation) -> Similarity:
    embedder = MockEmbedder()
    spans = sorted({r[f] for r in relations for f in ("cause", "effect") if r[f]})
    return Similarity(Space({span: embedder.embed(span) for span in spans}))


class FakeModel:
    """Answers from a queue of raw replies; records every prompt and schema it was given."""

    def __init__(self, replies: list[Completion]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def complete(self, prompt: str, schema: Mapping[str, Any], *, system: str) -> Completion:
        self.calls.append((prompt, schema))
        return self.replies.pop(0)


def confirm_all(pre: Any) -> str:
    return json.dumps({"decisions": [{"pair": pair, "decision": "confirm"} for pair in pre.candidate_pairs]})


def relations_of(answer: Answer) -> tuple[Relation, ...]:
    """The relations of an ok relation-list answer, typed."""
    assert isinstance(answer.value, tuple)
    return answer.value


def test_prompt_schema_and_decisions_are_bound_to_the_candidate_pairs() -> None:
    members = {"anthropic": [RAIN, STRIKE], "openai": [RAIN_VARIANT], "google": [STRIKE]}
    sim = similarity(RAIN, STRIKE, RAIN_VARIANT)
    pre = prematch(members, sim, accept_at=0.80, item="i", label="panel:1", text="Rain caused flooding.")
    assert len(pre.candidate_pairs) == 2
    template = load_prompt()
    assert PROMPT_VERSION in template.splitlines()[0]
    prompt = render_prompt(template, pre)
    assert "Rain caused flooding." in prompt
    assert '- anthropic:0: cause = "Rain"; effect = "flooding"' in prompt
    for edge in pre.edges:
        assert f"- {edge.pair}: {edge.a} with {edge.b} ({edge.band} band, similarity {edge.similarity:.3f})" in prompt
    schema = decision_schema(pre)
    assert schema["properties"]["decisions"]["items"]["properties"]["pair"]["enum"] == list(pre.candidate_pairs)
    decisions = parse_decisions(confirm_all(pre), pre)
    assert decisions == dict.fromkeys(pre.candidate_pairs, True)
    first = pre.candidate_pairs[0]
    with pytest.raises(ValueError, match="cover the candidate pairs"):
        parse_decisions(json.dumps({"decisions": [{"pair": first, "decision": "reject"}]}), pre)
    with pytest.raises(ValueError, match="twice"):
        parse_decisions(json.dumps({"decisions": [{"pair": first, "decision": "reject"}] * 2}), pre)
    with pytest.raises(ValueError, match="strict JSON"):
        parse_decisions("{bad", pre)
    with pytest.raises(ValueError, match="decisions array"):
        parse_decisions(json.dumps({"other": []}), pre)
    with pytest.raises(ValueError, match="nothing to decide"):
        decision_schema(prematch({"a": [RAIN], "b": [STRIKE]}, sim, accept_at=0.80))


def test_llm_referee_retries_inside_its_block_and_never_calls_without_pairs() -> None:
    members = {"anthropic": [RAIN], "openai": [RAIN_VARIANT], "google": []}
    sim = similarity(RAIN, RAIN_VARIANT)
    pre = prematch(members, sim, accept_at=0.80, text="t")
    pair = pre.candidate_pairs[0]
    model = FakeModel(
        [
            Completion("", error="timeout"),
            Completion('{"decisions": []}'),
            Completion(json.dumps({"decisions": [{"pair": pair, "decision": "reject"}]})),
        ]
    )
    referee = LLMReferee(model, template="{{TEXT}}|{{RELATIONS}}|{{PAIRS}}")
    assert referee.decide(pre) == {pair: False} and referee.calls == 3
    assert model.calls[0][1] == decision_schema(pre)
    empty = prematch({"anthropic": [RAIN], "openai": [STRIKE], "google": []}, similarity(RAIN, STRIKE), accept_at=0.80)
    assert referee.decide(empty) == {} and referee.calls == 3
    failing = LLMReferee(FakeModel([Completion("", refused=True)] * 3), template="{{TEXT}}", attempts=3)
    with pytest.raises(RefereeError, match="after 3 attempts"):
        failing.decide(pre)
    alignment = align(members, sim, LLMReferee(FakeModel([Completion(confirm_all(pre))]), template="{{PAIRS}}"))
    assert [u.support for u in alignment.units] == [2]


def test_identity_rule_for_the_referee() -> None:
    assert identity_acceptable("claude-opus-5-5", "claude-opus-5-5")
    assert identity_acceptable("claude-opus-5-5", "claude-opus-5-5-20261001")
    assert not identity_acceptable("claude-opus-5-5", "claude-opus-5-5-latest")
    assert not identity_acceptable("claude-opus-5-5", "") and not identity_acceptable("claude-opus-5-5", None)


def test_replay_referee_serves_stored_decisions_and_refuses_other_evidence(
    mock_cycles: dict[str, Any], tmp_path: Path
) -> None:
    source = mock_cycles["root"] / "c1" / "causalnews"
    run_dir = tmp_path / "causalnews"
    shutil.copytree(source, run_dir)
    task = study_task("causalnews", "{{TEXT}}")
    items = items_for("causalnews", mock_cycles["items"]["causalnews"])
    stored = StoredRun(run_dir)
    referee = ReplayReferee(stored)
    run = load_run(run_dir, task, items, referee=referee)
    item = run.items[0]
    sim = run.similarity(item.id)
    assert sim is not None
    members = {coder: relations_of(run.answer(item.id, coder, 1)) for coder in run.coders}
    pre = prematch(members, sim, accept_at=0.80, item=item.id, label="panel:1", text=item.text)
    record = referee.record(pre)
    assert record is not None and record["prematch_sha256"] == pre.sha256
    decisions = referee.decide(pre)
    assert set(decisions) == set(pre.candidate_pairs)
    raw = json.loads(record["response"]["raw_response"])
    stored_decisions = {d["pair"]: d["decision"] == "confirm" for d in raw["decisions"]}
    assert decisions == stored_decisions
    with pytest.raises(ValueError, match="banded at another threshold"):
        referee.decide(prematch(members, sim, accept_at=0.70, item=item.id, label="panel:1"))
    with pytest.raises(ReplayError, match="needs the pre-match's item and label"):
        referee.decide(prematch(members, sim, accept_at=0.80))
    alone = prematch({"a": [RAIN]}, similarity(RAIN), accept_at=0.80, item=item.id, label="single:x")
    assert referee.decide(alone) == {}
    path = run_dir / "checkpoints" / "agent" / "000000_panel_1.json"
    original = json.loads(path.read_text())
    tampered = dict(original, prematch_sha256="0" * 64)
    path.write_text(json.dumps(tampered))
    with pytest.raises(ReplayError, match="differs from the recomputed"):
        referee.decide(pre)
    tampered = json.loads(json.dumps(original))
    tampered["response"]["returned_model_id"] = "other"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ReplayError, match="unacceptable identity"):
        referee.decide(pre)
    path.unlink()
    with pytest.raises(ReplayError, match="record missing"):
        referee.decide(pre)


def test_the_referee_spends_the_same_transport_budget_as_a_coder() -> None:
    from codercommittee.align import prematch
    from codercommittee.data import Relation
    from codercommittee.protocols import Completion
    from codercommittee.referee import LLMReferee, RefereeError
    from codercommittee.similarity import Similarity, Space

    class Exhausting:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, schema: Any, *, system: str) -> Completion:
            self.calls += 1
            return Completion("", {"attempts": 3}, "APITimeoutError: slow")

    relation = Relation(cause="rain", effect="flood")
    sim = Similarity(Space({"rain": [1.0, 0.0], "flood": [0.0, 1.0]}))
    pre = prematch({"a": (relation,), "b": (relation,)}, sim, accept_at=0.8, item=1, label="panel:1")
    model = Exhausting()
    with pytest.raises(RefereeError):
        LLMReferee(model, template="{{PAIRS}}").decide(pre)
    assert model.calls == 1
