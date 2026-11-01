"""Deduplication of one answer's relations, content keys and the medoid.

Every relation answer is deduplicated exactly once before any use: Union-Find at pair
similarity 0.90 with a 0.85 complete-linkage floor (a cluster of three or more whose weakest
pair falls below the floor splits back into singletons, so chaining through an intermediary
cannot merge relations that are not themselves near-duplicates), each cluster represented by
its medoid. The output order is the candidate key, which fixes the indices every later step
names.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

from .data import Member, Relation
from .similarity import Similarity

DEDUP_THRESHOLD = 0.90
LINKAGE_FLOOR = 0.85


def relation_key(relation: Relation, names: Sequence[str]) -> tuple[str, ...]:
    """A coder-neutral content ordering: casefolded spans, then the exact spans, in name order."""
    exact = tuple(relation.get(name, "") for name in names)
    return tuple(value.casefold() for value in exact) + exact


def member_key(member: Member, names: Sequence[str]) -> tuple[Any, ...]:
    """The candidate key of a member: its relation key, its index and its name."""
    return (relation_key(member.relation, names), member.index, member.name)


def medoid(members: Sequence[Member], sim: Similarity) -> Member:
    """The member closest on average to the others; ties by the candidate key."""
    if not members:
        raise ValueError("cannot choose a medoid from an empty cluster")
    ranked: list[tuple[float, tuple[Any, ...], Member]] = []
    for member in members:
        others = [other for other in members if other is not member]
        mean = sum(sim.pair(member.relation, other.relation) for other in others) / len(others) if others else 1.0
        ranked.append((-mean, member_key(member, sim.names), member))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return ranked[0][2]


def dedupe(relations: Sequence[Relation], sim: Similarity) -> tuple[Relation, ...]:
    """One answer's relations without near-duplicates, each cluster's medoid, in key order."""
    members = [Member("", index, relation, "") for index, relation in enumerate(relations)]
    count = len(members)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pair_similarity: dict[tuple[int, int], float] = {}
    for left, right in itertools.combinations(range(count), 2):
        value = sim.pair(members[left].relation, members[right].relation)
        pair_similarity[(left, right)] = value
        if value >= DEDUP_THRESHOLD:
            parent[find(left)] = find(right)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    clusters: list[list[int]] = []
    for indices in groups.values():
        indices.sort()
        weakest = (
            min(pair_similarity[pair] for pair in itertools.combinations(indices, 2)) if len(indices) > 1 else None
        )
        if len(indices) >= 3 and weakest is not None and weakest < LINKAGE_FLOOR:
            clusters.extend([index] for index in indices)
        else:
            clusters.append(indices)
    names = sim.names
    clusters.sort(key=lambda indices: tuple(member_key(members[i], names) for i in indices))
    medoids = [medoid([members[i] for i in indices], sim) for indices in clusters]
    return tuple(member.relation for member in sorted(medoids, key=lambda m: member_key(m, names)))


__all__ = ["DEDUP_THRESHOLD", "LINKAGE_FLOOR", "dedupe", "medoid", "member_key", "relation_key"]
