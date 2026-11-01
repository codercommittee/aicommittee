"""Scores of one prediction against one reference, per answer kind.

label: accuracy with the confusion cell (the level is a macro-F1 over pooled cells).
labels: Jaccard similarity, shared labels over all labels named.
tuple: detection-gated similarity: zero when the prediction is negative on a positive
reference, else the mean pair similarity over every retained reference tuple.
tuples: a count-penalized matched similarity: predictions and references are matched one to
one by an exact assignment on the pair similarity, pairs below 0.60 never match, each matched
pair contributes its per-field cosines, and the matched mass is divided by the larger of the
two counts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .assignment import maximize
from .data import Confusion, ItemScore, Match, Prediction, Reference, Relation, Task, Value
from .similarity import Similarity

MATCH_THRESHOLD = 0.60


def macro_f1(confusions: Sequence[Confusion]) -> float:
    """Binary macro-F1 from pooled confusion cells (positive class, then negative class)."""
    totals = [float(sum(cell[index] for cell in confusions)) for index in range(6)]

    def f1(tp: float, fp: float, fn: float) -> float:
        denominator = 2.0 * tp + fp + fn
        return 2.0 * tp / denominator if denominator else 0.0

    return 0.5 * (f1(*totals[:3]) + f1(*totals[3:]))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Shared labels over all labels named; both empty one, one empty zero."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def score(
    prediction: Prediction | Value | None,
    reference: Reference | None,
    task: Task,
    sim: Similarity | None = None,
) -> ItemScore:
    """Score one prediction (or a bare value) against one reference on the task's scale."""
    if isinstance(prediction, Prediction):
        if not prediction.ok:
            raise ValueError("an insufficient prediction has no score; its item leaves the arm")
        value = prediction.value
    else:
        value = prediction
    if reference is None:
        raise ValueError("an item without a reference is unscored")
    if task.kind == "label":
        return _label(value, reference)
    if task.kind == "labels":
        return _labels(value, reference)
    if task.kind == "tuple":
        return _tuple(value, reference, sim)
    return _tuples(value, reference, task, sim)


def _label(value: Value | None, reference: Reference) -> ItemScore:
    if not isinstance(value, bool) or not isinstance(reference, bool):
        raise TypeError("a label score needs a boolean prediction and reference")
    if value and reference:
        confusion: Confusion = (1, 0, 0, 0, 0, 0)
    elif value and not reference:
        confusion = (0, 1, 0, 0, 0, 1)
    elif not value and reference:
        confusion = (0, 0, 1, 0, 1, 0)
    else:
        confusion = (0, 0, 0, 1, 0, 0)
    return ItemScore(float(value == reference), confusion)


def _labels(value: Value | None, reference: Reference) -> ItemScore:
    if not isinstance(value, frozenset) or not isinstance(reference, frozenset):
        raise TypeError("a labels score needs label-set prediction and reference")
    return ItemScore(jaccard(value, reference))


def _tuple(value: Value | None, reference: Reference, sim: Similarity | None) -> ItemScore:
    if not isinstance(reference, tuple):
        raise TypeError("a tuple score needs a sequence of reference relations")
    if value is not None and not isinstance(value, Relation):
        raise TypeError("a tuple prediction is a Relation or None")
    if not reference:
        return ItemScore(1.0 if value is None else 0.0)
    if value is None:
        return ItemScore(0.0)
    if sim is None:
        raise ValueError("a tuple score needs the item's similarity")
    return ItemScore(float(np.mean([sim.pair(value, relation) for relation in reference])))


def _tuples(value: Value | None, reference: Reference, task: Task, sim: Similarity | None) -> ItemScore:
    if not isinstance(value, tuple) or not isinstance(reference, tuple):
        raise TypeError("a relation score needs sequences of predicted and reference relations")
    if not value or not reference:
        return ItemScore(float(not value and not reference))
    if sim is None:
        raise ValueError("a relation score needs the item's similarity")
    matrix = [[sim.pair(predicted, actual) for actual in reference] for predicted in value]
    thresholded = [[cell if cell >= MATCH_THRESHOLD else 0.0 for cell in row] for row in matrix]
    matches = [a for a in maximize(thresholded) if a.value >= MATCH_THRESHOLD]
    denominator = max(len(value), len(reference))
    per_field = [sim.per_field(value[m.row], reference[m.col]) for m in matches]
    field_means = [sum(fields[name] for fields in per_field) / denominator for name in task.fields]
    total = sum(field_means) / len(task.fields)
    matched = tuple(Match(m.row, m.col, matrix[m.row][m.col]) for m in matches)
    return ItemScore(total, matched=matched)


__all__ = ["MATCH_THRESHOLD", "jaccard", "macro_f1", "score"]
