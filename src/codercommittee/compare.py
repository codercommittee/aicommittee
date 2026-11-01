"""Scores, levels, the comparison table, agreement and the labelled-sample pick.

A :class:`Scores` holds one run's item x arm x realization scores. Its level is the mean over
items per realization, then over realizations; for the binary label it is the macro-F1 per
realization, then the mean. :func:`compare` takes several runs' Scores over the same items and
returns, in one call, every arm's level per run and its mean and sd, its gap to the best single
coder of the same run with the item-paired bootstrap interval on run-averaged item differences,
the blind pick, the spread and each arm's premium. :func:`agreement` gives the cross- and
within-coder disagreement rates. :func:`pick_on_sample` asks what a labelled sample of n items
would pick and what that pick costs on the remaining items.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt

from .arms import ArmPredictions
from .data import Confusion, Item, ItemId, ItemScore, Run, Task
from .scoring import MATCH_THRESHOLD, macro_f1, score

REPLICATES = 2000
PICK_DRAWS = 500
PERCENTILES = (2.5, 97.5)
TIE_TOLERANCE = 1e-12
SINGLE = "single:"
Array = npt.NDArray[np.float64]
Indices = npt.NDArray[np.int64]


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _sd(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


# --------------------------------------------------------------------------
# Scores of one run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scores:
    """One run's scores: per arm, per scored item (run order), one ItemScore per realization."""

    task: Task
    coders: tuple[str, ...]
    items: tuple[Item, ...]
    values: Mapping[str, tuple[tuple[ItemScore, ...], ...]]

    @classmethod
    def of(cls, run: Run, predictions: ArmPredictions) -> Scores:
        """Score every arm's predictions on the run's scored items (complete, with a usable reference)."""
        items = run.scored()
        values: dict[str, tuple[tuple[ItemScore, ...], ...]] = {}
        for arm, by_item in predictions.items():
            rows: list[tuple[ItemScore, ...]] = []
            for item in items:
                sim = run.similarity(item.id)
                realizations = by_item.get(item.id)
                if not realizations:
                    raise ValueError(f"arm {arm!r} has no prediction for item {item.id!r}")
                rows.append(tuple(score(prediction, item.reference, run.task, sim) for prediction in realizations))
            values[arm] = tuple(rows)
        return cls(run.task, run.coders, items, values)

    @property
    def arms(self) -> tuple[str, ...]:
        """The arm keys, in the order they were given."""
        return tuple(self.values)

    @property
    def singles(self) -> tuple[str, ...]:
        """The one-answer arms in seat order."""
        return tuple(f"{SINGLE}{coder}" for coder in self.coders if f"{SINGLE}{coder}" in self.values)

    @property
    def item_ids(self) -> tuple[ItemId, ...]:
        """The scored items' ids in run order."""
        return tuple(item.id for item in self.items)

    @property
    def labels(self) -> tuple[int, ...]:
        """The binary references as strata (1 or 0); zeros for the other kinds."""
        if self.task.kind != "label":
            return tuple(0 for _ in self.items)
        return tuple(int(bool(item.reference)) for item in self.items)

    def realizations(self, arm: str) -> int:
        """The number of realizations an arm has."""
        rows = self.values[arm]
        return len(rows[0]) if rows else 0

    def item_values(self, arm: str) -> Array:
        """Per item, the mean of the realizations' values, in run order."""
        return np.asarray([_mean([s.value for s in realizations]) for realizations in self.values[arm]], dtype=float)

    def confusions(self, arm: str) -> list[list[Confusion]]:
        """Per realization, the items' confusion cells (binary label only)."""
        out: list[list[Confusion]] = []
        for index in range(self.realizations(arm)):
            cells: list[Confusion] = []
            for realizations in self.values[arm]:
                cell = realizations[index].confusion
                if cell is None:
                    raise ValueError("confusion cells exist for the binary label only")
                cells.append(cell)
            out.append(cells)
        return out

    def level(self, arm: str) -> float:
        """The arm's level: mean over items per realization, then over realizations; macro-F1 for the label."""
        if not self.items:
            return float("nan")
        if self.task.kind == "label":
            return _mean([macro_f1(cells) for cells in self.confusions(arm)])
        per_realization = [
            _mean([realizations[index].value for realizations in self.values[arm]])
            for index in range(self.realizations(arm))
        ]
        return _mean(per_realization)

    def levels(self) -> dict[str, float]:
        """Every arm's level."""
        return {arm: self.level(arm) for arm in self.arms}

    def best_single(self) -> str:
        """The single arm with the highest level; ties fall to the first in seat order."""
        singles = self.singles
        if not singles:
            raise ValueError("no single arm in these scores")
        levels = self.levels()
        return max(singles, key=lambda arm: levels[arm]) if len({levels[a] for a in singles}) > 1 else singles[0]

    def to_frame(self) -> Any:
        """A pandas DataFrame with one row per item, arm and realization (pandas required)."""
        import pandas as pd

        rows = [
            {"item": item.id, "arm": arm, "realization": index + 1, "value": s.value}
            for arm, by_item in self.values.items()
            for item, realizations in zip(self.items, by_item, strict=True)
            for index, s in enumerate(realizations)
        ]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Comparison across runs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Gap:
    """An arm's gap to the best single of the same run, summarised across runs."""

    per_run: tuple[float, ...]
    mean: float
    sd: float | None
    worst: float
    interval: tuple[float, float] | None
    n_items: int | None


@dataclass(frozen=True)
class ArmRow:
    """One arm's row of the comparison table."""

    arm: str
    levels: tuple[float, ...]
    level_mean: float
    level_sd: float | None
    gap: Gap | None
    premium: float
    premium_worst: float


@dataclass(frozen=True)
class Comparison:
    """Every arm against the best single coder of the same run, with intervals, in one table."""

    runs: int
    rows: Mapping[str, ArmRow]
    best_single: tuple[str, ...]
    blind_pick: Gap
    best_level: float
    mean_member: float
    spread: float
    draws: int
    seed: int
    n_items: int

    @property
    def arms(self) -> tuple[str, ...]:
        """The arm keys of the table."""
        return tuple(self.rows)

    def to_frame(self) -> Any:
        """A pandas DataFrame with one row per arm (pandas required)."""
        import pandas as pd

        return pd.DataFrame([_row_record(row) for row in self.rows.values()]).set_index("arm")

    def __str__(self) -> str:
        head = f"{'arm':<22}{'level':>9}{'sd':>8}{'gap':>9}{'interval':>22}{'worst':>9}{'premium':>10}"
        lines = [head, "-" * len(head)]
        for row in self.rows.values():
            gap = row.gap
            interval = f"[{gap.interval[0]:+.3f}, {gap.interval[1]:+.3f}]" if gap and gap.interval else ""
            lines.append(
                f"{row.arm:<22}{row.level_mean:>9.3f}{_fmt(row.level_sd):>8}"
                f"{_fmt(gap.mean if gap else None, signed=True):>9}{interval:>22}"
                f"{_fmt(gap.worst if gap else None, signed=True):>9}{row.premium:>+10.3f}"
            )
        lines.append(
            f"best single per run: {', '.join(self.best_single)}; blind pick {self.blind_pick.mean:+.3f} "
            f"(worst {self.blind_pick.worst:+.3f}); spread {self.spread:+.3f}; "
            f"{self.runs} run(s), {self.n_items} items, {self.draws} bootstrap draws, seed {self.seed}"
        )
        return "\n".join(lines)


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return ""
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _row_record(row: ArmRow) -> dict[str, Any]:
    gap = row.gap
    return {
        "arm": row.arm,
        "level": row.level_mean,
        "level_sd": row.level_sd,
        "gap": gap.mean if gap else None,
        "gap_sd": gap.sd if gap else None,
        "gap_low": gap.interval[0] if gap and gap.interval else None,
        "gap_high": gap.interval[1] if gap and gap.interval else None,
        "gap_worst": gap.worst if gap else None,
        "premium": row.premium,
        "premium_worst": row.premium_worst,
        **{f"level_run{index + 1}": level for index, level in enumerate(row.levels)},
    }


def iid_plan(n_items: int, draws: int, seed: int) -> list[Indices]:
    """The item-paired resample plan: one fresh generator, n indices with replacement per draw."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_items, size=n_items) for _ in range(draws)]


def stratified_plan(labels: Sequence[int], draws: int, seed: int) -> list[Indices]:
    """The binary label's plan: resample inside each stratum (sorted label order), concatenated."""
    strata = [
        np.asarray([i for i, label in enumerate(labels) if label == value], dtype=int) for value in sorted(set(labels))
    ]
    rng = np.random.default_rng(seed)
    plan: list[Indices] = []
    for _ in range(draws):
        sampled = [rng.choice(indices, size=indices.size, replace=True) for indices in strata if indices.size]
        plan.append(np.concatenate(sampled))
    return plan


def paired_interval(deltas: Array, draws: int, seed: int) -> tuple[float, float]:
    """Percentile interval of the mean of item differences under the item-paired plan."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = [float(deltas[rng.integers(0, n, n)].mean()) for _ in range(draws)]
    low, high = PERCENTILES
    return float(np.percentile(means, low)), float(np.percentile(means, high))


def compare(scores: Sequence[Scores], *, draws: int = REPLICATES, seed: int) -> Comparison:
    """Every arm's level, gap, interval, blind pick, spread and premium across runs, in one call.

    The runs must share their arms and their coders (seat order). Items are paired by typed id
    across runs (``7`` and ``"7"`` are two items); the interval runs over the items common to
    every run, ordered by ``str(id)`` (the binary label: the first run's items in run order,
    which every run must share).
    """
    if not scores:
        raise ValueError("compare needs at least one run's scores")
    first = scores[0]
    arms = first.arms
    for other in scores[1:]:
        if other.arms != arms or other.coders != first.coders or other.task.kind != first.task.kind:
            raise ValueError("every run must share its arms, coders and task kind")
    singles = first.singles
    if not singles:
        raise ValueError("the comparison needs the single arms to name a best single coder")
    levels = [s.levels() for s in scores]
    best = tuple(s.best_single() for s in scores)
    binary = first.task.kind == "label"

    # Item-level columns for the interval: run-averaged item differences on the common items.
    if binary:
        ids = [s.item_ids for s in scores]
        if any(other != ids[0] for other in ids[1:]):
            raise ValueError("the binary label's plan pairs items by position: every run must hold the same items")
        plan = stratified_plan(first.labels, draws, seed)
        cells = [{arm: s.confusions(arm) for arm in arms} for s in scores]
        n_common = len(first.items)

        def macro(run_index: int, arm: str, indices: Indices) -> float:
            return _mean([macro_f1([cells_r[i] for i in indices]) for cells_r in cells[run_index][arm]])

    else:
        shared = set.intersection(*(set(s.item_ids) for s in scores))
        common = sorted(shared, key=lambda item_id: (str(item_id), type(item_id).__name__))
        n_common = len(common)
        columns: dict[str, Array] = {}
        for arm in arms:
            columns[arm] = np.mean([columns_run(scores, r, arm, common) for r in range(len(scores))], axis=0)
        base = np.mean([columns_run(scores, r, best[r], common) for r in range(len(scores))], axis=0)

    rows: dict[str, ArmRow] = {}
    mean_member_per_run = [sum(levels[r][arm] for arm in singles) / len(singles) for r in range(len(scores))]
    best_per_run = [levels[r][best[r]] for r in range(len(scores))]
    mean_member = sum(mean_member_per_run) / len(scores)
    best_level = sum(best_per_run) / len(scores)
    for arm in arms:
        arm_levels = tuple(levels[r][arm] for r in range(len(scores)))
        gap: Gap | None = None
        if any(arm != best[r] for r in range(len(scores))):
            gaps_per_run = tuple(levels[r][arm] - best_per_run[r] for r in range(len(scores)))
            interval: tuple[float, float] | None
            if binary:
                deltas = [
                    _mean([macro(r, arm, indices) - macro(r, best[r], indices) for r in range(len(scores))])
                    for indices in plan
                ]
                interval = (float(np.percentile(deltas, PERCENTILES[0])), float(np.percentile(deltas, PERCENTILES[1])))
            else:
                interval = paired_interval(columns[arm] - base, draws, seed) if n_common else None
            gap = Gap(
                gaps_per_run, _mean(gaps_per_run), _sd(gaps_per_run), min(gaps_per_run), interval, n_common or None
            )
        premium_per_run = [levels[r][arm] - mean_member_per_run[r] for r in range(len(scores))]
        rows[arm] = ArmRow(
            arm,
            arm_levels,
            _mean(arm_levels),
            _sd(arm_levels),
            gap,
            sum(arm_levels) / len(scores) - mean_member,
            min(premium_per_run),
        )
    blind_per_run = tuple(_mean([levels[r][arm] - best_per_run[r] for arm in singles]) for r in range(len(scores)))
    blind_worst = min(min(levels[r][arm] - best_per_run[r] for arm in singles) for r in range(len(scores)))
    blind = Gap(blind_per_run, _mean(blind_per_run), _sd(blind_per_run), blind_worst, None, None)
    return Comparison(
        len(scores), rows, best, blind, best_level, mean_member, best_level - mean_member, draws, seed, n_common
    )


def columns_run(scores: Sequence[Scores], run_index: int, arm: str, common: Sequence[ItemId]) -> Array:
    """One run's item values for an arm over the common items, in their order."""
    s = scores[run_index]
    index = {item_id: position for position, item_id in enumerate(s.item_ids)}
    values = s.item_values(arm)
    return np.asarray([values[index[item_id]] for item_id in common], dtype=float)


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Agreement:
    """Disagreement rates over the complete items: identity of the coders' normalized answers."""

    n_items: int
    cross: float
    within: Mapping[str, float]
    pairwise: Mapping[str, float]
    graded: Mapping[str, Any] | None = None


def agreement(run: Run) -> Agreement:
    """Disagreement rates over the complete items.

    Cross-coder disagreement is the share of item realizations whose coders' answers are not all
    identical; within-coder disagreement the share of items whose draws differ; pairwise the share
    of identical answers per coder pair. Relation lists also get the graded block: matched Jaccard
    at the match threshold and count ranges.
    """
    complete = run.complete()
    coders = run.coders
    draws = range(1, run.draws + 1)

    def value(item: Item, coder: str, draw: int) -> Any:
        return run.answer(item.id, coder, draw).value

    def differ(values: list[Any]) -> float:
        return float(any(v != values[0] for v in values[1:]))

    within = {
        coder: _mean([differ([value(item, coder, draw) for draw in draws]) for item in complete]) for coder in coders
    }
    cross = _mean([differ([value(item, coder, draw) for coder in coders]) for item in complete for draw in draws])
    pairwise = {
        f"{left}|{right}": _mean(
            [1.0 - differ([value(item, left, draw), value(item, right, draw)]) for item in complete for draw in draws]
        )
        for left, right in combinations(coders, 2)
    }
    graded: dict[str, Any] | None = None
    if run.task.kind == "tuples":

        def matched_jaccard(item: Item, left: Any, right: Any) -> float:
            if not left and not right:
                return 1.0
            matched = len(score(left, right, run.task, run.similarity(item.id)).matched)
            return matched / (len(left) + len(right) - matched)

        def count_range(lists: list[Any]) -> float:
            sizes = [len(relations) for relations in lists]
            return float(max(sizes) - min(sizes))

        def mean_pairwise(item: Item, lists: list[Any]) -> float:
            return float(np.mean([matched_jaccard(item, a, b) for a, b in combinations(lists, 2)]))

        graded = {
            "match_threshold": MATCH_THRESHOLD,
            "within_count_range": {
                coder: _mean([count_range([value(item, coder, d) for d in draws]) for item in complete])
                for coder in coders
            },
            "cross_count_range": _mean(
                [count_range([value(item, coder, draw) for coder in coders]) for item in complete for draw in draws]
            ),
            "within_matched_jaccard": {
                coder: _mean([mean_pairwise(item, [value(item, coder, d) for d in draws]) for item in complete])
                for coder in coders
            },
            "cross_matched_jaccard": _mean(
                [mean_pairwise(item, [value(item, c, d) for c in coders]) for item in complete for d in draws]
            ),
        }
    return Agreement(len(complete), cross, within, pairwise, graded)


# --------------------------------------------------------------------------
# The labelled-sample pick
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pick:
    """What a labelled sample of n items picks, and what the pick costs on the remaining items."""

    n: int
    draws: int
    heldout_gap: float
    pick_share: float
    blind_gap: float
    among: tuple[str, ...] = field(default_factory=tuple)


def tied_at_top(values: Array) -> Indices:
    """Indices of every value at the top, within the tie tolerance."""
    values = np.asarray(values, dtype=float)
    top = float(np.max(values))
    return np.flatnonzero(values >= top - TIE_TOLERANCE)


def pick_on_sample(
    scores: Scores, n: int, among: Sequence[str] | None = None, *, draws: int = PICK_DRAWS, seed: int
) -> Pick:
    """What a labelled sample of n items picks, over ``draws`` random samples.

    Each draw picks the option with the highest mean item score on the sample (a tie split evenly,
    no further random number) and records the pick's gap to the run's best single on the remaining
    items, whether the best single was picked, and the blind pick's gap on the same remaining
    items. Refused when n exceeds half the items.
    """
    options = tuple(among) if among is not None else scores.singles
    if not options:
        raise ValueError("nothing to pick among")
    total = len(scores.items)
    if n < 1 or n > total // 2:
        raise ValueError(f"a sample of {n} items leaves too few of {total} held out (n must be at most {total // 2})")
    matrix = np.asarray([scores.item_values(arm) for arm in options])
    singles = np.asarray([scores.item_values(arm) for arm in scores.singles])
    best = scores.best_single()
    best_values = scores.item_values(best)
    best_index = options.index(best) if best in options else None
    rng = np.random.default_rng(seed + n)
    picked_best = 0.0
    gaps: list[float] = []
    blind: list[float] = []
    for _ in range(draws):
        sample = rng.choice(total, size=n, replace=False)
        mask = np.ones(total, dtype=bool)
        mask[sample] = False
        tied = tied_at_top(matrix[:, sample].mean(axis=1))
        heldout = matrix[:, mask].mean(axis=1)
        heldout_best = float(best_values[mask].mean())
        gaps.append(float(heldout[tied].mean() - heldout_best))
        blind.append(float(singles[:, mask].mean(axis=1).mean() - heldout_best))
        if best_index is not None:
            picked_best += float(best_index in tied) / len(tied)
    return Pick(n, draws, _mean(gaps), picked_best / draws, _mean(blind), options)


__all__ = [
    "PERCENTILES",
    "PICK_DRAWS",
    "REPLICATES",
    "TIE_TOLERANCE",
    "Agreement",
    "ArmRow",
    "Comparison",
    "Gap",
    "Pick",
    "Scores",
    "agreement",
    "compare",
    "iid_plan",
    "paired_interval",
    "pick_on_sample",
    "stratified_plan",
    "tied_at_top",
]
