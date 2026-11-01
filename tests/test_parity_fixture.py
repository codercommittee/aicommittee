"""Parity against the portable fixture: the library's predictions and numbers equal what the
study pipeline derived from the same stored evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codercommittee.arms import arms
from codercommittee.combine import combine
from codercommittee.compare import Scores, agreement, compare, pick_on_sample
from codercommittee.data import Run
from codercommittee.referee import BandRule, ReplayReferee
from codercommittee.replay import StoredRun, load_run, value_as_stored
from codercommittee.scoring import score
from codercommittee.study import BOOTSTRAP_SEED
from codercommittee.study import items_from_gold as items_for
from codercommittee.study import task as study_task

DATASETS = ("causalnews", "miric", "gvfc", "politicause")
CYCLES = ("c1", "c2", "c3")
METRIC = {
    "causalnews": "cardinality_penalized_matched_embedding_score",
    "miric": "accuracy",
    "gvfc": "jaccard",
    "politicause": "detection_gated_mean_coder_embedding_score",
}
CONFUSION = ("tp_pos", "fp_pos", "fn_pos", "tp_neg", "fp_neg", "fn_neg")


def rows_of(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "per_item.jsonl").read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def runs(mock_cycles: dict[str, Any]) -> dict[tuple[str, str], tuple[Run, list[dict[str, Any]]]]:
    out = {}
    for cycle in CYCLES:
        for dataset in DATASETS:
            run_dir = mock_cycles["root"] / cycle / dataset
            items = items_for(dataset, mock_cycles["items"][dataset])
            out[(cycle, dataset)] = (load_run(run_dir, study_task(dataset, "{{TEXT}}"), items), rows_of(run_dir))
    return out


def test_single_answer_scores_equal_the_stored_rows(
    runs: dict[tuple[str, str], tuple[Run, list[dict[str, Any]]]],
) -> None:
    checked = 0
    for (_cycle, dataset), (run, rows) in runs.items():
        metric = METRIC[dataset]
        for row, item in zip(rows, run.items, strict=True):
            sim = run.similarity(item.id)
            for coder in run.coders:
                for draw in range(1, run.draws + 1):
                    stored = row["scores"]["models"][coder][str(draw)]
                    answer = run.answer(item.id, coder, draw)
                    if stored is None:
                        assert not answer.ok
                        continue
                    mine = score(answer.value, item.reference, run.task, sim)
                    assert mine.value == pytest.approx(stored[metric], abs=1e-12), (dataset, item.id, coder, draw)
                    if dataset == "miric":
                        assert mine.confusion == tuple(stored[name] for name in CONFUSION)
                    if dataset == "causalnews":
                        expected = {(a["pred_index"], a["gold_index"]) for a in stored.get("assignments", [])}
                        assert {(m.prediction, m.reference) for m in mine.matched} == expected
                        assert len(mine.matched) == stored["n_pairs_matched"]
                    checked += 1
    assert checked > 500


def check_structures(
    runs: dict[tuple[str, str], tuple[Run, list[dict[str, Any]]]], referees: Any, construction: str
) -> int:
    """Vote realizations and self-ensembles under a referee, against the rows' construction."""
    checked = 0
    for (cycle, dataset), (run, rows) in runs.items():
        referee = referees(cycle, dataset) if callable(referees) else referees
        task, metric = run.task, METRIC[dataset]
        relation = dataset == "causalnews"
        for row, item in zip(rows, run.items, strict=True):
            if row["exclusion"] is not None:
                continue
            sim = run.similarity(item.id)
            realizations = []
            for draw in range(1, run.draws + 1):
                members = {coder: run.answer(item.id, coder, draw) for coder in run.coders}
                stored = row["panel"]["realizations"][str(draw)]
                stored_score = row["scores"]["panel_realizations"][str(draw)]
                if relation:
                    stored, stored_score = stored[construction], stored_score[construction]
                label = f"panel:{draw}"
                realizations.append((members, 2, label, stored["prediction"], stored_score["panel"][metric]))
            for coder in run.coders:
                members = {str(draw): run.answer(item.id, coder, draw) for draw in range(1, run.draws + 1)}
                stored = row["self_ensembles"][coder]
                stored_score = row["scores"]["self_ensembles"][coder]
                if relation:
                    stored, stored_score = stored[construction], stored_score[construction]
                realizations.append((members, 2, f"self:{coder}", stored["prediction"], stored_score[metric]))
            for members, k, label, expected_value, expected_score in realizations:
                prediction = combine(
                    members, task, k, sim=sim, referee=referee, item=item.id, label=label, text=item.text
                )
                assert prediction.ok, (dataset, item.id, label)
                assert value_as_stored(prediction.value, task) == expected_value, (dataset, item.id, label)
                mine = score(prediction, item.reference, task, sim)
                assert mine.value == pytest.approx(expected_score, abs=1e-12), (dataset, item.id, label)
                checked += 1
    return checked


def test_vote_and_self_ensemble_under_the_band_rule_equal_the_stored_rows(
    runs: dict[tuple[str, str], tuple[Run, list[dict[str, Any]]]],
) -> None:
    assert check_structures(runs, BandRule(0.80), "deterministic") > 300


def test_the_pool_at_every_k_equals_the_pipeline_on_the_fixture(
    runs: dict[tuple[str, str], tuple[Run, list[dict[str, Any]]]], mock_cycles: dict[str, Any]
) -> None:
    checked = 0
    for cycle in CYCLES:
        run, _rows = runs[(cycle, "causalnews")]
        expected = mock_cycles["expected"]["cycles"][cycle]["causalnews"]["pool_items"]
        out = arms(run, BandRule(0.80), pool_k=range(1, 10))
        for item in run.complete():
            sim = run.similarity(item.id)
            for k in range(1, 10):
                prediction = out[f"pool:{k}"][item.id][0]
                stored = expected[str(item.id)][str(k)]
                # The pipeline lists consensus units before singletons; the library sorts by candidate key.
                got = sorted(json.dumps(r, sort_keys=True) for r in value_as_stored(prediction.value, run.task))
                assert got == sorted(json.dumps(r, sort_keys=True) for r in stored["prediction"]), (cycle, item.id, k)
                mine = score(prediction, item.reference, run.task, sim)
                assert mine.value == pytest.approx(stored["score"], abs=1e-12), (cycle, item.id, k)
                checked += 1
    assert checked == 3 * 6 * 9


def test_vote_and_self_ensemble_with_the_stored_referee_decisions_equal_the_rows(
    runs: dict[tuple[str, str], tuple[Run, list[dict[str, Any]]]], mock_cycles: dict[str, Any]
) -> None:
    def referee_for(cycle: str, dataset: str) -> Any:
        if dataset != "causalnews":
            return BandRule()
        return ReplayReferee(StoredRun(mock_cycles["root"] / cycle / dataset))

    assert check_structures(runs, referee_for, "alignment") > 300


def to_mine(arm: str) -> str:
    """The derived build's arm key as the library's."""
    if arm == "panel:agent":
        return "vote"
    if arm == "panel:det":
        return "vote:band"
    if arm.startswith("pool:k"):
        return "pool:" + arm[len("pool:k") :]
    return arm


def composed_scores(run: Run, run_dir: Path, dataset: str) -> Scores:
    """The study's table: the vote and the self-ensembles with the stored referee, the band-rule vote
    beside them and the pool at every k on the relation dataset; the band rule elsewhere."""
    if dataset == "causalnews":
        predictions = dict(arms(run, ReplayReferee(StoredRun(run_dir)), pool_k=range(1, 10)))
        predictions["vote:band"] = arms(run, BandRule(0.80), pool_k=())["vote"]
    else:
        predictions = dict(arms(run, BandRule(0.80)))
    return Scores.of(run, predictions)


@pytest.fixture(scope="module")
def tables(
    runs: dict[tuple[str, str], tuple[Run, list[dict[str, Any]]]], mock_cycles: dict[str, Any]
) -> dict[str, dict[str, Scores]]:
    return {
        dataset: {
            cycle: composed_scores(runs[(cycle, dataset)][0], mock_cycles["root"] / cycle / dataset, dataset)
            for cycle in CYCLES
        }
        for dataset in DATASETS
    }


def test_levels_gaps_intervals_blind_pick_and_composition_equal_the_derived_values(
    tables: dict[str, dict[str, Scores]], mock_cycles: dict[str, Any]
) -> None:
    close = pytest.approx
    for dataset in DATASETS:
        cross = mock_cycles["expected"]["cross"][dataset]
        by_cycle = tables[dataset]
        for arm, levels in cross["levels"].items():
            for cycle, level in levels.items():
                assert by_cycle[cycle].level(to_mine(arm)) == close(level, abs=1e-9), (dataset, arm, cycle)
        table = compare([by_cycle[c] for c in CYCLES], seed=BOOTSTRAP_SEED[dataset])
        assert table.best_single == tuple(f"single:{cross['best_single_by_cycle'][c]}" for c in CYCLES)
        regret = cross["regret"]
        # The derived table lists the pool at k = 1, 2, 3; the library's table carries every k asked for.
        with_gap = {arm for arm, row in table.rows.items() if row.gap is not None and not arm.startswith("pool:")}
        assert with_gap == {to_mine(arm) for arm in regret["cycle_mean"] if not arm.startswith("pool:")}, dataset
        for arm, record in regret["cycle_mean"].items():
            gap = table.rows[to_mine(arm)].gap
            assert gap is not None
            assert gap.mean == close(record["regret_mean"], abs=1e-9), (dataset, arm)
            assert gap.sd == close(record["regret_sd"], abs=1e-9) and gap.worst == close(record["worst"], abs=1e-9)
            assert gap.n_items == record["n_items"], (dataset, arm)
            assert gap.interval is not None
            assert gap.interval == close(tuple(record["interval_95"]), abs=1e-9), (dataset, arm)
        for arm, per_cycle in regret["per_cycle"].items():
            row = table.rows[to_mine(arm)]
            gaps = row.gap.per_run if row.gap else tuple(0.0 for _ in CYCLES)
            assert gaps == close(tuple(per_cycle[c] for c in CYCLES), abs=1e-9), (dataset, arm)
        blind = regret["blind_single_pick"]
        assert table.blind_pick.mean == close(blind["mean"], abs=1e-9)
        assert table.blind_pick.worst == close(blind["worst"], abs=1e-9)
        assert table.blind_pick.per_run == close(tuple(blind["per_cycle"][c] for c in CYCLES), abs=1e-9)
        composition = mock_cycles["expected"]["composition"][dataset]
        assert table.mean_member == close(composition["mean_member"], abs=1e-9)
        assert table.best_level == close(composition["best_level"], abs=1e-9)
        assert table.spread == close(composition["best_level"] - composition["mean_member"], abs=1e-9)
        for arm, premium in composition["premiums"].items():
            assert table.rows[to_mine(arm)].premium == close(premium, abs=1e-9), (dataset, arm)
        for arm, level in cross["level_summary"].items():
            row = table.rows[to_mine(arm)]
            assert row.level_mean == close(level["mean"], abs=1e-9) and row.level_sd == close(level["sd"], abs=1e-9)
        if dataset == "causalnews":
            for k, bar in cross["pools"]["keep_bar"]["bars"].items():
                row = table.rows[f"pool:{k}"]
                assert row.levels == close(tuple(bar["per_cycle_level"]), abs=1e-9), k
                assert row.gap is not None and row.gap.mean == close(bar["minus_best_single"]["estimate"], abs=1e-9)
                assert row.gap.interval == close(tuple(bar["minus_best_single"]["interval_95"]), abs=1e-9), k


def test_agreement_equals_the_derived_disagreement(
    runs: dict[tuple[str, str], tuple[Run, list[dict[str, Any]]]], mock_cycles: dict[str, Any]
) -> None:
    close = pytest.approx
    for (cycle, dataset), (run, _rows) in runs.items():
        expected = mock_cycles["expected"]["cycles"][cycle][dataset]["disagreement"]
        mine = agreement(run)
        assert mine.cross == close(expected["cross_model"], abs=1e-12), (cycle, dataset)
        assert mine.within == close(expected["within_model"], abs=1e-12), (cycle, dataset)
        assert mine.pairwise == close(expected["pairwise_agreement"], abs=1e-12), (cycle, dataset)
        graded = expected["relation_graded"]
        if graded is None:
            assert mine.graded is None
            continue
        assert mine.graded is not None
        pairs = {
            "within_count_range": "within_model_relation_count_range_mean",
            "cross_count_range": "cross_model_relation_count_range_mean",
            "within_matched_jaccard": "within_model_mean_pairwise_matched_jaccard",
            "cross_matched_jaccard": "cross_model_mean_pairwise_matched_jaccard",
        }
        for mine_key, theirs_key in pairs.items():
            assert mine.graded[mine_key] == close(graded[theirs_key], abs=1e-12), (cycle, mine_key)


def test_the_labelled_sample_pick_equals_the_derived_validation(
    tables: dict[str, dict[str, Scores]], mock_cycles: dict[str, Any]
) -> None:
    checked = 0
    for dataset in DATASETS:
        validation = mock_cycles["expected"]["cross"][dataset]["validation"]["by_size"]
        for size, record in validation.items():
            for cycle, expected in record["by_cycle"].items():
                pick = pick_on_sample(tables[dataset][cycle], int(size), seed=BOOTSTRAP_SEED[dataset])
                assert pick.pick_share == pytest.approx(expected["pick_best_share"], abs=1e-12), (dataset, size, cycle)
                assert pick.heldout_gap == pytest.approx(expected["heldout_gap_mean"], abs=1e-12)
                assert pick.blind_gap == pytest.approx(expected["blind_heldout_gap_mean"], abs=1e-12)
                checked += 1
    assert checked >= 6
