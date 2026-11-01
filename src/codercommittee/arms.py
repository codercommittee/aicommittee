"""The four structures as presets over one run.

``single:<coder>``: each of the coder's draws alone, three realizations. ``self:<coder>``: the
coder's three draws combined with the vote's own rules, members named by draw. ``vote``: draw
r of every coder, members named by coder, three realizations. ``pool:<k>``: all nine answers
aligned in one step with the band rule at 0.70, members named ``coder:draw``, a relation kept
when at least k answers contain it (relation lists only). The population is the run's complete
items. This version supports what the study ran, three coders and three draws; other counts
are refused here, not in the operator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .align import align
from .combine import combine, keep_units
from .data import Answer, Item, ItemId, Prediction, Relation, Run
from .protocols import Referee
from .referee import BandRule
from .similarity import Similarity

COMMITTEE_SIZE = 3
DRAWS = 3
POOL_ACCEPT_AT = 0.70
POOL_KS = (1, 2, 3)
VOTE_K = 2

ArmPredictions = Mapping[str, Mapping[ItemId, tuple[Prediction, ...]]]


def arm_names(run: Run, pool_k: Sequence[int] = POOL_KS) -> list[str]:
    """The arm keys in the study's order: singles, self-ensembles, the vote, then the pools."""
    names = [f"single:{coder}" for coder in run.coders] + [f"self:{coder}" for coder in run.coders] + ["vote"]
    if run.task.kind == "tuples":
        names.extend(f"pool:{k}" for k in pool_k)
    return names


def arms(
    run: Run, referee: Referee | None = None, pool_k: Sequence[int] = POOL_KS
) -> dict[str, dict[ItemId, tuple[Prediction, ...]]]:
    """Every structure's predictions, per arm and item, one Prediction per realization.

    ``referee`` decides the vote's and the self-ensembles' candidate pairs (default: the run's
    referee, else the band rule at 0.80); the pool always uses the band rule at 0.70.
    """
    if len(run.coders) != COMMITTEE_SIZE or run.draws != DRAWS:
        raise ValueError(
            "this version supports three coders and three draws per item, as the study ran; "
            f"the run has {len(run.coders)} coders and {run.draws} draws"
        )
    if any(k < 1 or k > COMMITTEE_SIZE * DRAWS for k in pool_k):
        raise ValueError(f"pool k must lie between 1 and {COMMITTEE_SIZE * DRAWS}")
    rule = referee if referee is not None else (run.referee if run.referee is not None else BandRule())
    task = run.task
    out: dict[str, dict[ItemId, tuple[Prediction, ...]]] = {name: {} for name in arm_names(run, pool_k)}
    draws = range(1, DRAWS + 1)
    for item in run.complete():
        sim = run.similarity(item.id)

        def one(
            members: dict[str, Answer], k: int, label: str, item: Item = item, sim: Similarity | None = sim
        ) -> Prediction:
            return combine(members, task, k, sim=sim, referee=rule, item=item.id, label=label, text=item.text)

        for coder in run.coders:
            out[f"single:{coder}"][item.id] = tuple(
                one({coder: run.answer(item.id, coder, draw)}, 1, f"single:{coder}:{draw}") for draw in draws
            )
            members = {str(draw): run.answer(item.id, coder, draw) for draw in draws}
            out[f"self:{coder}"][item.id] = (one(members, VOTE_K, f"self:{coder}"),)
        out["vote"][item.id] = tuple(
            one({coder: run.answer(item.id, coder, draw) for coder in run.coders}, VOTE_K, f"panel:{draw}")
            for draw in draws
        )
        if task.kind == "tuples":
            _pool(run, item, pool_k, out)
    return out


def _pool(run: Run, item: Item, pool_k: Sequence[int], out: dict[str, dict[ItemId, tuple[Prediction, ...]]]) -> None:
    """All nine answers aligned once; every k applies its keep rule to the same units."""
    sim = run.similarity(item.id)
    assert sim is not None
    draws = range(1, DRAWS + 1)
    answers = {f"{coder}:{draw}": run.answer(item.id, coder, draw) for coder in run.coders for draw in draws}
    relations: dict[str, tuple[Relation, ...]] = {}
    for name, answer in answers.items():
        relations[name] = tuple(answer.value) if isinstance(answer.value, tuple) else ()
    coders = {name: answer.coder for name, answer in answers.items()}
    rule = BandRule(POOL_ACCEPT_AT)
    alignment = align(relations, sim, rule, coders=coders, item=item.id, label="pool", text=item.text)
    for k in pool_k:
        out[f"pool:{k}"][item.id] = (keep_units(alignment, run.task, len(answers), k, 1, sim),)


__all__ = ["COMMITTEE_SIZE", "DRAWS", "POOL_ACCEPT_AT", "POOL_KS", "VOTE_K", "ArmPredictions", "arm_names", "arms"]
