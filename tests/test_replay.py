"""The replay loader against the portable fixture: every stored answer reproduced, identity
checks that make answers invalid, absent evidence that raises."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from codercommittee.replay import ReplayCoder, ReplayEmbedder, ReplayError, StoredRun, load_run, stored_value
from codercommittee.study import items_from_gold as items_for
from codercommittee.study import task as study_task

DATASETS = ("causalnews", "miric", "gvfc", "politicause")


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "per_item.jsonl").read_text().splitlines() if line.strip()]


def test_replay_reproduces_every_stored_answer(mock_cycles: dict[str, Any]) -> None:
    for cycle in ("c1", "c2", "c3"):
        for dataset in DATASETS:
            run_dir = mock_cycles["root"] / cycle / dataset
            task = study_task(dataset, "{{TEXT}}")
            items = items_for(dataset, mock_cycles["items"][dataset])
            run = load_run(run_dir, task, items)
            rows = _rows(run_dir)
            assert [item.id for item in run.complete()] == [row["item_id"] for row in rows if row["exclusion"] is None]
            for row, item in zip(rows, items, strict=True):
                for coder in run.coders:
                    for draw in range(1, run.draws + 1):
                        record = row["model_responses"][coder][str(draw)]
                        answer = run.answer(item.id, coder, draw)
                        where = (dataset, item.id, coder, draw, answer.reason)
                        assert answer.ok == record["validation"]["valid"], where
                        if not answer.ok:
                            continue
                        if dataset == "causalnews":
                            assert stored_value(answer, task) == row["prepared_predictions"][coder][str(draw)]
                        else:
                            assert stored_value(answer, task) == record["validated_prediction"]
                        assert answer.identity["request_fingerprint"] == record["request_fingerprint"]


def test_identity_failures_make_answers_invalid_and_absent_evidence_raises(
    mock_cycles: dict[str, Any], tmp_path: Path
) -> None:
    source = mock_cycles["root"] / "c1" / "causalnews"
    run_dir = tmp_path / "causalnews"
    shutil.copytree(source, run_dir)
    task = study_task("causalnews", "{{TEXT}}")
    items = items_for("causalnews", mock_cycles["items"]["causalnews"])
    stored = StoredRun(run_dir)
    coder = ReplayCoder(stored, stored.coders[0])
    path = run_dir / "checkpoints" / "responses" / f"000000_00_r1_{coder.name}.json"
    original = json.loads(path.read_text())

    def tampered(**changes: Any) -> str | None:
        record = json.loads(json.dumps(original))
        record["response"].update(changes)
        path.write_text(json.dumps(record))
        return coder.answer(items[0], task, 1).reason

    assert coder.answer(items[0], task, 1).ok
    assert tampered(returned_model_id="other-model") == "returned_model_mismatch"
    assert tampered(requested_model_id="other-model") == "requested_model_mismatch"
    assert tampered(cache_namespace="x") == "cache_namespace_mismatch"
    assert tampered(prompt_version="x") == "prompt_version_mismatch"
    assert tampered(attempts=4) == "invalid_transport_attempt_count"
    assert tampered(request_fingerprint="0" * 64) == "request_fingerprint_mismatch"
    assert tampered(raw_response="{not json") == "strict_json_parse_failed"
    assert tampered(parsed_response={"relations": []}) == "parsed_response_does_not_match_raw"
    broken = {"relations": [{"cause": "", "effect": "x", "signal": ""}]}
    assert tampered(raw_response=json.dumps(broken), parsed_response=broken) == "relation_0_missing_field"
    refusal = tampered(provider_error="provider_incomplete_finish:refusal")
    assert refusal == "provider_incomplete_finish:refusal" and coder.answer(items[0], task, 1).status == "refused"
    assert tampered(provider_error="transport: timeout") == "transport: timeout"
    assert coder.answer(items[0], task, 1).status == "invalid"
    discard = {**original["response"], "attempts": 1}
    assert tampered(content_retry_discarded=[discard], attempts=2) == "content_retry_discarded_a_valid_response"
    bad_discard = {**discard, "raw_response": "{bad"}
    assert tampered(content_retry_discarded=[bad_discard], attempts=2) is None
    assert tampered(content_retry_discarded=[bad_discard], attempts=1) == "content_retry_attempt_accounting_mismatch"
    path.write_text(json.dumps(original))

    run = load_run(run_dir, task, items)
    assert run.complete()
    path.unlink()
    with pytest.raises(ReplayError, match="no stored answer"):
        load_run(run_dir, task, items)
    with pytest.raises(ReplayError, match="differ from the stored run"):
        load_run(source, task, list(reversed(items)))
    with pytest.raises(ReplayError):
        ReplayCoder(stored, "nobody")
    with pytest.raises(ReplayError):
        StoredRun(tmp_path / "nowhere")


def test_replay_embedder_requires_the_exact_stored_span_set(mock_cycles: dict[str, Any]) -> None:
    run_dir = mock_cycles["root"] / "c1" / "causalnews"
    stored = StoredRun(run_dir)
    items = items_for("causalnews", mock_cycles["items"]["causalnews"])
    payload = stored.embedding(0)
    assert payload is not None
    texts = [entry["text"] for entry in payload["vectors"].values()]
    space = ReplayEmbedder(stored).space(items[0], texts)
    assert len(space) == len(texts) and space.dimension == payload["dimensions"]
    with pytest.raises(ReplayError, match="differ from the spans asked for"):
        ReplayEmbedder(stored).space(items[0], texts[:-1])
    with pytest.raises(ReplayError, match="differ from the spans asked for"):
        ReplayEmbedder(stored).space(items[0], [*texts, "a span never embedded"])


def test_the_stored_decisions_are_the_default_referee_of_a_relation_run(
    mock_cycles: dict[str, Any], tmp_path: Path
) -> None:
    from codercommittee.arms import arms
    from codercommittee.referee import BandRule, ReplayReferee

    source = mock_cycles["root"] / "c1" / "causalnews"
    run_dir = tmp_path / "causalnews"
    shutil.copytree(source, run_dir)
    task = study_task("causalnews", "{{TEXT}}")
    items = items_for("causalnews", mock_cycles["items"]["causalnews"])
    run = load_run(run_dir, task, items)
    assert isinstance(run.referee, ReplayReferee) and run.provenance["referee"]["kind"] == "replay"
    before = run.arms()["vote"][items[0].id]
    band = arms(run, BandRule(0.80))
    politicause = load_run(
        mock_cycles["root"] / "c1" / "politicause",
        study_task("politicause", "{{TEXT}}"),
        items_for("politicause", mock_cycles["items"]["politicause"]),
    )
    assert politicause.referee is None
    checkpoint = run_dir / "checkpoints" / "agent" / "000000_panel_1.json"
    record = json.loads(checkpoint.read_text())
    raw = json.loads(record["response"]["raw_response"])
    for entry in raw["decisions"]:
        entry["decision"] = "reject"
    record["response"].update(raw_response=json.dumps(raw), parsed_response=raw)
    checkpoint.write_text(json.dumps(record))
    altered = load_run(run_dir, task, items)
    assert altered.arms()["vote"][items[0].id] != before
    assert arms(altered, BandRule(0.80)) == band
    checkpoint.unlink()
    with pytest.raises(ReplayError, match="referee record missing"):
        load_run(run_dir, task, items).arms()


def test_a_saved_replay_keeps_the_study_decisions(mock_cycles: dict[str, Any], tmp_path: Path) -> None:
    from codercommittee.data import Run
    from codercommittee.store import StoreReferee

    task = study_task("causalnews", "{{TEXT}}")
    items = items_for("causalnews", mock_cycles["items"]["causalnews"])
    run = load_run(mock_cycles["root"] / "c1" / "causalnews", task, items)
    run.save(tmp_path / "saved")
    assert (tmp_path / "saved" / "decisions.jsonl").is_file()
    saved = Run.load(tmp_path / "saved")
    assert isinstance(saved.referee, StoreReferee) and saved.provenance["referee"]["kind"] == "replay"
    assert saved.arms() == run.arms()
    saved.save(tmp_path / "saved")  # idempotent
