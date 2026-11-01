"""The one operator behind the four structures.

``combine`` takes the selected members' answers, aligns them, keeps what at least ``k`` members
from at least ``p`` coders support, represents each kept unit by its medoid, and returns a
Prediction. Labels align by exact equality and need no similarity; a positive tuple is the
majority's, represented by the medoid of the positive members; relation lists are aligned by
the referee.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .align import Alignment, align
from .data import Answer, ItemId, Member, Prediction, Relation, Task
from .dedupe import medoid, member_key
from .protocols import Referee
from .referee import BandRule
from .similarity import Similarity
from .validation import check_prediction

VOTE_ACCEPT_AT = 0.80


def combine(
    members: Mapping[str, Answer],
    task: Task,
    k: int,
    p: int = 1,
    *,
    sim: Similarity | None = None,
    referee: Referee | None = None,
    item: ItemId | None = None,
    label: str | None = None,
    text: str = "",
) -> Prediction:
    """Combine the members' answers: keep what at least ``k`` members from ``p`` coders support."""
    if k < 1 or p < 1:
        raise ValueError("k and p are at least one")
    if not members:
        raise ValueError("combine needs at least one member")
    excluded = sorted(name for name, answer in members.items() if not answer.ok)
    if excluded:
        return Prediction(None, "insufficient", audit={"excluded": excluded})
    n = len(members)
    names = sorted(members)
    if task.kind == "label":
        votes = {name: bool(members[name].value) for name in names}
        value: Any = sum(votes.values()) >= k
        audit: dict[str, Any] = {"votes": votes, "k": k}
    elif task.kind == "labels":
        counts = Counter(label_ for name in names for label_ in members[name].value)  # type: ignore[union-attr]
        value = frozenset(label_ for label_, count in counts.items() if count >= k)
        audit = {"counts": dict(sorted(counts.items())), "k": k}
    elif task.kind == "tuple":
        quorum = n // 2 + 1
        if k != quorum:
            raise ValueError(f"the tuple kind decides by majority: with {n} members pass k = {quorum}")
        positive: list[Member] = []
        for name in names:
            relation = members[name].value
            if isinstance(relation, Relation):
                positive.append(Member(name, 0, relation, members[name].coder))
        audit = {"positive_votes": len(positive), "negative_votes": n - len(positive)}
        if len(positive) > n / 2:
            if sim is None:
                raise ValueError("a positive tuple needs the item's similarity to choose its medoid")
            value = medoid(positive, sim).relation
        else:
            value = None
    else:
        if sim is None:
            raise ValueError("relations need the item's similarity to align")
        rule = referee if referee is not None else BandRule(VOTE_ACCEPT_AT)
        relations: dict[str, tuple[Relation, ...]] = {}
        for name in names:
            listed = members[name].value
            relations[name] = tuple(listed) if isinstance(listed, tuple) else ()
        coders = {name: members[name].coder for name in names}
        alignment = align(relations, sim, rule, coders=coders, item=item, label=label, text=text)
        return keep_units(alignment, task, n, k, p, sim)
    check_prediction(task, value, n, k)
    return Prediction(value, "ok", audit=audit)


def keep_units(alignment: Alignment, task: Task, members: int, k: int, p: int, sim: Similarity) -> Prediction:
    """The prediction an alignment gives under one keep rule: units with support >= k from >= p coders."""
    kept = [unit for unit in alignment.units if unit.support >= k and unit.coders >= p]
    kept.sort(key=lambda unit: member_key(unit.representative, sim.names))
    value = tuple(unit.medoid for unit in kept)
    check_prediction(task, value, members, k)
    return Prediction(
        value,
        "ok",
        units=alignment.units,
        audit={
            "k": k,
            "p": p,
            "kept": len(kept),
            "decisions": dict(alignment.decisions),
            "dropped_edges": list(alignment.dropped),
            "prematch_sha256": alignment.prematch.sha256,
            "candidate_pairs": len(alignment.prematch.candidate_pairs),
            "fallback_pairs": list(alignment.prematch.fallback_pairs),
        },
    )


__all__ = ["VOTE_ACCEPT_AT", "combine", "keep_units"]
