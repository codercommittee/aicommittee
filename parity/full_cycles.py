"""Parity of the library against the study's stored cycles, item by item.

Run inside the study's virtual environment, from this repository::

    python parity/full_cycles.py --study /path/to/multi-model-llm-qualitative-coding

The command reads the main panel's cycles from ``code/figures/evidence.json``, the items
through the study's own dataset loaders, and every stored answer, vector and referee decision
through the library's replay adapters (nothing is re-requested). It then checks that the
library reproduces the pipeline's rows and the derived numbers. It fails with a clear message
when the study checkout or its run directories are absent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from codercommittee.arms import arms  # noqa: E402
from codercommittee.combine import combine  # noqa: E402
from codercommittee.compare import Scores, agreement, compare, pick_on_sample  # noqa: E402
from codercommittee.data import Item, Run, Task  # noqa: E402
from codercommittee.referee import BandRule, ReplayReferee, load_prompt, render_prompt  # noqa: E402
from codercommittee.replay import ReplayError, StoredRun, load_run, value_as_stored  # noqa: E402
from codercommittee.scoring import score  # noqa: E402
from codercommittee.study import BOOTSTRAP_SEED, TASKS  # noqa: E402
from codercommittee.study import items_from_gold as items_for  # noqa: E402

METRIC = {
    "causalnews": "cardinality_penalized_matched_embedding_score",
    "fincausal": "cardinality_penalized_matched_embedding_score",
    "miric": "accuracy",
    "gvfc": "jaccard",
    "politicause": "detection_gated_mean_coder_embedding_score",
}
CONFUSION = ("tp_pos", "fp_pos", "fn_pos", "tp_neg", "fp_neg", "fn_neg")
RELATION = ("causalnews", "fincausal")


class Report:
    """Counts checks and keeps the first failures of each kind."""

    def __init__(self) -> None:
        self.checks = 0
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, message: str) -> None:
        self.checks += 1
        if not ok and len(self.failures) < 200:
            self.failures.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def fail(message: str) -> None:
    print(f"parity: {message}", file=sys.stderr)
    sys.exit(2)


def study_items(study: Path, dataset: str) -> list[Item]:
    """The dataset's items through the study's loaders (texts and gold), in the study's order."""
    code = study / "code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from pipeline import run_study
    from pipeline.dataset_inputs import resolve_dataset_inputs

    run_study.bootstrap_env()
    inputs = resolve_dataset_inputs(dataset)
    seed = int(run_study.STUDY_SPEC["datasets"][dataset]["seed"])
    rows = run_study._load_bound_items(dataset, n=None, seed=seed, inputs=inputs)
    return items_for(dataset, rows)


def stored_rows(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "per_item.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def check_answers(run: Run, rows: list[dict[str, Any]], task: Task, dataset: str, report: Report) -> None:
    """Every stored answer: validity, value and the single-answer score, item by item."""
    from codercommittee.replay import stored_value

    metric = METRIC[dataset]
    complete = [row["item_id"] for row in rows if row["exclusion"] is None]
    report.check([item.id for item in run.complete()] == complete, f"{dataset}: complete items differ from the rows")
    for row, item in zip(rows, run.items, strict=True):
        sim = run.similarity(item.id)
        for coder in run.coders:
            for draw in range(1, run.draws + 1):
                record = row["model_responses"][coder][str(draw)]
                answer = run.answer(item.id, coder, draw)
                where = f"{dataset} {item.id} {coder} r{draw}"
                valid = bool(record["validation"]["valid"])
                report.check(answer.ok == valid, f"{where}: validity {answer.status} ({answer.reason})")
                if not answer.ok:
                    continue
                if dataset in RELATION:
                    expected = row["prepared_predictions"][coder][str(draw)]
                else:
                    expected = record["validated_prediction"]
                report.check(stored_value(answer, task) == expected, f"{where}: value differs")
                stored = row["scores"]["models"][coder][str(draw)]
                mine = score(answer.value, item.reference, task, sim)
                close = abs(mine.value - stored[metric]) <= 1e-9
                report.check(close, f"{where}: score {mine.value} vs {stored[metric]}")
                if dataset == "miric":
                    confusion = tuple(stored[name] for name in CONFUSION)
                    report.check(mine.confusion == confusion, f"{where}: confusion differs")
                if dataset in RELATION:
                    pairs = {(a["pred_index"], a["gold_index"]) for a in stored.get("assignments", [])}
                    mine_pairs = {(m.prediction, m.reference) for m in mine.matched}
                    if mine_pairs != pairs:
                        report.note(f"{where}: tied assignment, value equal")
                    report.check(len(mine.matched) == stored["n_pairs_matched"], f"{where}: match count differs")


def check_prompts(run: Run, run_dir: Path, task: Task, dataset: str, report: Report) -> None:
    """The packaged study prompt, rendered for each item, equals the prompt every stored request sent."""
    stored = StoredRun(run_dir)
    for index, item in enumerate(run.items):
        expected = task.render(item.text)
        for coder in run.coders:
            for draw in range(1, run.draws + 1):
                checkpoint = stored.response(index, coder, draw)
                sent = ((checkpoint.get("execution_binding") or {}).get("request_envelope") or {}).get("prompt")
                report.check(sent == expected, f"{dataset} {item.id} {coder} r{draw}: the study prompt differs")


def check_structures(
    run: Run, rows: list[dict[str, Any]], task: Task, dataset: str, report: Report, referee: Any, construction: str
) -> None:
    """The vote's three realizations and the three self-ensembles under a referee, against the rows."""
    metric = METRIC[dataset]
    relation = dataset in RELATION
    for row, item in zip(rows, run.items, strict=True):
        if row["exclusion"] is not None:
            continue
        sim = run.similarity(item.id)
        cases = []
        for draw in range(1, run.draws + 1):
            members = {coder: run.answer(item.id, coder, draw) for coder in run.coders}
            stored = row["panel"]["realizations"][str(draw)]
            stored_score = row["scores"]["panel_realizations"][str(draw)]
            if relation:
                stored, stored_score = stored[construction], stored_score[construction]
            cases.append((members, f"panel:{draw}", stored["prediction"], stored_score["panel"]))
        for coder in run.coders:
            members = {str(draw): run.answer(item.id, coder, draw) for draw in range(1, run.draws + 1)}
            stored = row["self_ensembles"][coder]
            stored_score = row["scores"]["self_ensembles"][coder]
            if relation:
                stored, stored_score = stored[construction], stored_score[construction]
            cases.append((members, f"self:{coder}", stored["prediction"], stored_score))
        for members, label, expected_value, expected_score in cases:
            where = f"{dataset} {item.id} {label} ({construction})"
            prediction = combine(members, task, 2, sim=sim, referee=referee, item=item.id, label=label, text=item.text)
            report.check(prediction.ok, f"{where}: prediction insufficient")
            if not prediction.ok:
                continue
            report.check(value_as_stored(prediction.value, task) == expected_value, f"{where}: prediction differs")
            mine = score(prediction, item.reference, task, sim)
            close = abs(mine.value - expected_score[metric]) <= 1e-9
            report.check(close, f"{where}: score {mine.value} vs {expected_score[metric]}")
            if relation:
                pairs = {(a["pred_index"], a["gold_index"]) for a in expected_score.get("assignments", [])}
                if {(m.prediction, m.reference) for m in mine.matched} != pairs:
                    report.note(f"{where}: tied assignment, value equal")
                report.check(len(mine.matched) == expected_score["n_pairs_matched"], f"{where}: match count differs")


def canon(relation: dict[str, Any]) -> str:
    return json.dumps(relation, sort_keys=True, ensure_ascii=False)


def check_pool(run: Run, run_dir: Path, rows: list[dict[str, Any]], task: Task, dataset: str, report: Report) -> None:
    """The pool at every k against the pipeline's own one-step construction over the stored answers and vectors."""
    from extension import construction_study as cs
    from figures import pool_analysis
    from pipeline.run_study import _item_key, row_similarities
    from pipeline.scoring import score_relations

    spec = cs.load_spec()
    sims = row_similarities(run_dir)
    metric = METRIC[dataset]
    out = arms(run, BandRule(0.80), pool_k=range(1, 10))
    for row, item in zip(rows, run.items, strict=True):
        if row["exclusion"] is not None:
            continue
        sim = run.similarity(item.id)
        pipeline_sim = sims[_item_key(row["item_id"])]
        members = {f"{m}:{r}": row["prepared_predictions"][m][r] for m in run.coders for r in ("1", "2", "3")}
        units = pool_analysis._flat_units(members, pipeline_sim, spec)
        for k in range(1, 10):
            where = f"{dataset} {item.id} pool:{k}"
            expected = [dict(u["relation"]) for u in units if u["total_support"] >= k]
            prediction = out[f"pool:{k}"][item.id][0]
            got = value_as_stored(prediction.value, task)
            # The pipeline lists consensus units before singletons; the library sorts by candidate key.
            report.check(sorted(map(canon, got)) == sorted(map(canon, expected)), f"{where}: prediction differs")
            stored = score_relations(
                expected, row["gold_references"], similarity=pipeline_sim, assignment="hungarian", match_threshold=0.6
            )
            mine = score(prediction, item.reference, task, sim)
            report.check(abs(mine.value - stored[metric]) <= 1e-9, f"{where}: score {mine.value} vs {stored[metric]}")
            pairs = {(canon(expected[a["pred_index"]]), a["gold_index"]) for a in stored.get("assignments", [])}
            if {(canon(got[m.prediction]), m.reference) for m in mine.matched} != pairs:
                report.note(f"{where}: tied assignment, value equal")
            report.check(len(mine.matched) == stored["n_pairs_matched"], f"{where}: match count differs")


class RecordingReferee:
    """Wraps the replay referee and keeps every pre-match it decided, for the prompt check."""

    def __init__(self, inner: ReplayReferee) -> None:
        self.inner = inner
        self.accept_at = inner.accept_at
        self.seen: list[Any] = []

    def decide(self, prematch: Any) -> dict[str, bool]:
        self.seen.append(prematch)
        return self.inner.decide(prematch)


def check_referee_prompts(recording: RecordingReferee, dataset: str, report: Report) -> None:
    """The library renders the referee prompt the study sent, byte for byte, for every stored decision."""
    template = load_prompt()
    for pre in recording.seen:
        if not pre.candidate_pairs:
            continue
        record = recording.inner.record(pre)
        stored = (((record or {}).get("execution_binding") or {}).get("request_envelope") or {}).get("prompt")
        if stored is None:
            report.note(f"{dataset} {pre.item} {pre.label}: stored record carries no prompt")
            continue
        same = render_prompt(template, pre) == stored
        report.check(same, f"{dataset} {pre.item} {pre.label}: referee prompt differs")


MEMBER = {"anthropic": "Claude", "openai": "Gpt", "google": "Gemini"}
PREFIX = {"miric": "MI", "gvfc": "GV", "politicause": "PC", "causalnews": "CN", "fincausal": "FC"}
WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
SIZE_WORD = {"10": "Ten", "20": "Twenty", "40": "Forty", "80": "Eighty"}


def _words(text: str) -> str:
    out = []
    for char in text:
        if char.isdigit():
            out.append(WORD.get(int(char), "Zero") if char != "0" else "Zero")
        elif char.isalpha():
            out.append(char)
    return "".join(out)


def arm_key(arm: str) -> str:
    """The macro stem of an arm, as the study's numbers.tex names it."""
    family, _, member = arm.partition(":")
    if family == "single":
        return f"Single{MEMBER[member]}"
    if family == "self":
        return f"Self{MEMBER[member]}"
    if family == "pool":
        return "Pool" + _words(member.title())
    return {"panel:agent": "Panel", "panel:det": "PanelDet"}[arm]


def cycle_word(cycle_id: str) -> str:
    return "C" + WORD[int(cycle_id[-1])]


def fmt_level(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def fmt_signed(value: float | None) -> str:
    return "--" if value is None else f"${value:+.3f}$"


def fmt_ci(estimate: float, interval: tuple[float, float]) -> str:
    return f"{fmt_signed(estimate)} [{fmt_signed(interval[0])}, {fmt_signed(interval[1])}]"


def fmt_pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.0f}\\%"


def parse_numbers(path: Path) -> dict[str, str]:
    numbers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", line)
        if match:
            numbers[match.group(1)] = match.group(2)
    return numbers


def composed_scores(run: Run, run_dir: Path, dataset: str) -> Scores:
    """The study's table under the derived build's arm names: the vote and the self-ensembles with the
    stored referee, the band-rule vote as panel:det, the pool at every k on the relation datasets."""
    if dataset in RELATION:
        agent = arms(run, ReplayReferee(StoredRun(run_dir)), pool_k=range(1, 10))
        band = arms(run, BandRule(0.80), pool_k=())
        kept = [arm for arm in agent if not arm.startswith("pool:") and arm != "vote"]
        predictions: dict[str, Any] = {arm: agent[arm] for arm in kept}
        predictions["panel:agent"] = agent["vote"]
        predictions["panel:det"] = band["vote"]
        for k in range(1, 10):
            predictions[f"pool:k{k}"] = agent[f"pool:{k}"]
    else:
        band = arms(run, BandRule(0.80))
        predictions = {arm: band[arm] for arm in band if arm != "vote"}
        predictions["panel:agent"] = band["vote"]
    return Scores.of(run, predictions)


def check_numbers(
    study: Path, dataset: str, scores_by_cycle: dict[str, Scores], runs_by_cycle: dict[str, Run], report: Report
) -> None:
    """Levels, gaps, intervals, blind pick, spread, premium, agreement and the pick against derived.json,
    and the printed macros against numbers.tex, re-rendered with the study's own formats."""
    from figures import panels

    derived = json.loads((study / "paper" / "generated" / "derived.json").read_text(encoding="utf-8"))
    numbers = parse_numbers(study / "paper" / "generated" / "numbers.tex")
    entry = derived["cross"][dataset]
    cycles = list(entry["cycles"])
    report.check(cycles == list(scores_by_cycle), f"{dataset}: cycle ids differ from derived.json")
    p = PREFIX[dataset]
    seed = BOOTSTRAP_SEED[dataset]

    def near(mine: float | None, theirs: float | None, where: str) -> None:
        if mine is None or theirs is None:
            report.check(mine is None and theirs is None, f"{where}: {mine} vs {theirs}")
        else:
            report.check(abs(mine - theirs) <= 1e-9, f"{where}: {mine} vs {theirs}")

    def printed(name: str, mine: str) -> None:
        if name not in numbers:
            report.note(f"{dataset}: numbers.tex has no macro {name}")
            return
        report.check(numbers[name] == mine, f"{dataset}: \\{name} prints {numbers[name]!r}, library gives {mine!r}")

    for arm, by_cycle in entry["levels"].items():
        for cycle, value in by_cycle.items():
            mine = scores_by_cycle[cycle].level(arm)
            near(mine, value, f"{dataset} {cycle} level {arm}")
            printed(f"{p}{arm_key(arm)}{cycle_word(cycle)}", fmt_level(mine))
    table = compare([scores_by_cycle[c] for c in cycles], seed=seed)
    for arm, record in entry["level_summary"].items():
        row = table.rows[arm]
        near(row.level_mean, record["mean"], f"{dataset} level mean {arm}")
        near(row.level_sd, record["sd"], f"{dataset} level sd {arm}")
        printed(f"{p}{arm_key(arm)}LevelMean", fmt_level(row.level_mean))
        printed(f"{p}{arm_key(arm)}LevelSd", fmt_level(row.level_sd))
    best = entry["best_single_by_cycle"]
    report.check(table.best_single == tuple(f"single:{best[c]}" for c in cycles), f"{dataset}: best single differs")
    regret = entry["regret"]
    # The derived table lists the pool at k = 1, 2, 3; the library's table carries every k asked for.
    with_gap = {arm for arm, row in table.rows.items() if row.gap is not None and not arm.startswith("pool:")}
    listed = {arm for arm in regret["cycle_mean"] if not arm.startswith("pool:")}
    report.check(with_gap == listed, f"{dataset}: arms with a gap differ")
    for arm, record in regret["cycle_mean"].items():
        gap = table.rows[arm].gap
        if gap is None:
            continue
        near(gap.mean, record["regret_mean"], f"{dataset} gap mean {arm}")
        near(gap.sd, record["regret_sd"], f"{dataset} gap sd {arm}")
        near(gap.worst, record["worst"], f"{dataset} gap worst {arm}")
        report.check(gap.n_items == record["n_items"], f"{dataset} gap n_items {arm}")
        if gap.interval is None or "interval_95" not in record:
            report.check(gap.interval is None and "interval_95" not in record, f"{dataset} interval presence {arm}")
        else:
            near(gap.interval[0], record["interval_95"][0], f"{dataset} interval low {arm}")
            near(gap.interval[1], record["interval_95"][1], f"{dataset} interval high {arm}")
            printed(f"{p}{arm_key(arm)}RegretMeanCI", fmt_ci(gap.mean, gap.interval))
        printed(f"{p}{arm_key(arm)}RegretMean", fmt_signed(gap.mean))
        printed(f"{p}{arm_key(arm)}RegretSd", fmt_level(gap.sd))
        printed(f"{p}{arm_key(arm)}RegretWorst", fmt_signed(gap.worst))
    for arm, per_cycle in regret["per_cycle"].items():
        row = table.rows[arm]
        gaps = row.gap.per_run if row.gap else tuple(0.0 for _ in cycles)
        for cycle, value, mine in zip(cycles, (per_cycle[c] for c in cycles), gaps, strict=True):
            near(mine, value, f"{dataset} {cycle} gap {arm}")
            printed(f"{p}{arm_key(arm)}Regret{cycle_word(cycle)}", fmt_signed(mine))
    blind = regret["blind_single_pick"]
    near(table.blind_pick.mean, blind["mean"], f"{dataset} blind pick mean")
    near(table.blind_pick.worst, blind["worst"], f"{dataset} blind pick worst")
    printed(f"{p}BlindSingleRegretMean", fmt_signed(table.blind_pick.mean))
    printed(f"{p}BlindSingleRegretWorst", fmt_signed(table.blind_pick.worst))
    for cycle, mine in zip(cycles, table.blind_pick.per_run, strict=True):
        near(mine, blind["per_cycle"][cycle], f"{dataset} {cycle} blind pick")
        printed(f"{p}BlindSingleRegret{cycle_word(cycle)}", fmt_signed(mine))
    composition = panels._composition(entry)
    near(table.mean_member, composition["mean_member"], f"{dataset} mean member")
    near(table.best_level, panels._best_level(entry), f"{dataset} best level")
    for arm, premium in composition["premiums"].items():
        near(table.rows[arm].premium, premium, f"{dataset} premium {arm}")
    if dataset in RELATION:
        for k, bar in entry["pools"]["keep_bar"]["bars"].items():
            row = table.rows[f"pool:k{k}"]
            for cycle, value, mine in zip(cycles, bar["per_cycle_level"], row.levels, strict=True):
                near(mine, value, f"{dataset} {cycle} pool:k{k} level")
            gap = row.gap
            report.check(gap is not None and gap.interval is not None, f"{dataset} pool:k{k} has no gap")
            if gap is not None and gap.interval is not None:
                near(gap.mean, bar["minus_best_single"]["estimate"], f"{dataset} pool:k{k} gap")
                near(gap.interval[0], bar["minus_best_single"]["interval_95"][0], f"{dataset} pool:k{k} interval low")
                near(gap.interval[1], bar["minus_best_single"]["interval_95"][1], f"{dataset} pool:k{k} interval high")
    for cycle in cycles:
        expected = entry["disagreement"][cycle]
        mine_agreement = agreement(runs_by_cycle[cycle])
        near(mine_agreement.cross, expected["cross_model"], f"{dataset} {cycle} cross disagreement")
        printed(f"{p}CrossDisagreement{cycle_word(cycle)}", fmt_pct(mine_agreement.cross))
        within = list(mine_agreement.within.values())
        printed(f"{p}WithinDisagreementMean{cycle_word(cycle)}", fmt_pct(sum(within) / len(within)))
        for coder, value in expected["within_model"].items():
            near(mine_agreement.within[coder], value, f"{dataset} {cycle} within disagreement {coder}")
        for pair, value in expected["pairwise_agreement"].items():
            near(mine_agreement.pairwise.get(pair), value, f"{dataset} {cycle} pairwise agreement {pair}")
        graded = expected.get("relation_graded")
        if graded:
            mine_graded = mine_agreement.graded or {}
            near(
                mine_graded.get("cross_count_range"),
                graded["cross_model_relation_count_range_mean"],
                f"{dataset} {cycle} cross count range",
            )
            near(
                mine_graded.get("cross_matched_jaccard"),
                graded["cross_model_mean_pairwise_matched_jaccard"],
                f"{dataset} {cycle} cross matched jaccard",
            )
            for coder in runs_by_cycle[cycle].coders:
                near(
                    mine_graded["within_count_range"][coder],
                    graded["within_model_relation_count_range_mean"][coder],
                    f"{dataset} {cycle} within count range {coder}",
                )
                near(
                    mine_graded["within_matched_jaccard"][coder],
                    graded["within_model_mean_pairwise_matched_jaccard"][coder],
                    f"{dataset} {cycle} within matched jaccard {coder}",
                )
    for size, record in entry["validation"]["by_size"].items():
        pick_shares: list[float] = []
        heldout_gaps: list[float] = []
        blind_gaps: list[float] = []
        for cycle, expected in record["by_cycle"].items():
            pick = pick_on_sample(scores_by_cycle[cycle], int(size), seed=seed)
            near(pick.pick_share, expected["pick_best_share"], f"{dataset} {cycle} pick share n={size}")
            near(pick.heldout_gap, expected["heldout_gap_mean"], f"{dataset} {cycle} held-out gap n={size}")
            near(pick.blind_gap, expected["blind_heldout_gap_mean"], f"{dataset} {cycle} blind held-out gap n={size}")
            pick_shares.append(pick.pick_share)
            heldout_gaps.append(pick.heldout_gap)
            blind_gaps.append(pick.blind_gap)
        stem = f"{p}ValidationN{SIZE_WORD.get(size, _words(size))}"
        printed(f"{stem}PickBest", fmt_pct(float(np.mean(pick_shares))))
        printed(f"{stem}Gap", fmt_signed(float(np.mean(heldout_gaps))))
        printed(f"{stem}BlindGap", fmt_signed(float(np.mean(blind_gaps))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", required=True, type=Path, help="path to the study repository checkout")
    parser.add_argument("--datasets", nargs="*", default=list(METRIC), help="datasets to check (default: all five)")
    parser.add_argument("--cycles", nargs="*", default=None, help="cycle ids to check (default: the evidence manifest)")
    args = parser.parse_args()
    study = args.study.resolve()
    manifest = study / "code" / "figures" / "evidence.json"
    if not manifest.is_file():
        fail(f"{manifest} is absent: pass --study with the study checkout that holds the stored cycles")
    evidence = json.loads(manifest.read_text(encoding="utf-8"))
    cycles = [c for c in evidence["cycles"] if args.cycles is None or c["id"] in args.cycles]
    if not cycles:
        fail("no cycle selected")
    for cycle in cycles:
        for dataset in args.datasets:
            run_dir = study / cycle["runs"][dataset]
            if not (run_dir / "per_item.jsonl").is_file():
                fail(f"run directory absent: {run_dir} (the stored cycles live on the study machine only)")
    report = Report()
    started = time.perf_counter()
    for dataset in args.datasets:
        task = TASKS[dataset]
        items = study_items(study, dataset)
        scores_by_cycle: dict[str, Scores] = {}
        runs_by_cycle: dict[str, Run] = {}
        for cycle in cycles:
            run_dir = study / cycle["runs"][dataset]
            rows = stored_rows(run_dir)
            try:
                run = load_run(run_dir, task, items)
            except ReplayError as exc:
                fail(f"{dataset} {cycle['id']}: {exc}")
            check_answers(run, rows, task, dataset, report)
            check_prompts(run, run_dir, task, dataset, report)
            check_structures(run, rows, task, dataset, report, BandRule(0.80), "deterministic")
            if dataset in RELATION:
                recording = RecordingReferee(ReplayReferee(StoredRun(run_dir)))
                check_structures(run, rows, task, dataset, report, recording, "alignment")
                check_referee_prompts(recording, dataset, report)
                check_pool(run, run_dir, rows, task, dataset, report)
            scores_by_cycle[cycle["id"]] = composed_scores(run, run_dir, dataset)
            runs_by_cycle[cycle["id"]] = run
            print(f"[{time.perf_counter() - started:6.1f}s] {dataset} {cycle['id']}: {report.checks} checks so far")
        if args.cycles is None:
            check_numbers(study, dataset, scores_by_cycle, runs_by_cycle, report)
            print(f"[{time.perf_counter() - started:6.1f}s] {dataset} numbers: {report.checks} checks so far")
    for note in report.notes:
        print(f"note: {note}")
    if report.failures:
        print(f"parity FAILED: {len(report.failures)} of {report.checks} checks")
        for failure in report.failures[:50]:
            print(f"  {failure}")
        sys.exit(1)
    print(f"parity passed: {report.checks} checks, {len(report.notes)} notes")


if __name__ == "__main__":
    main()
