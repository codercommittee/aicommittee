"""Pairwise matching of two members' relations: the pre-match's edges.

For a pair of members, the maximum-weight one-to-one matching over the candidate edges (pair
similarity at or above the candidate threshold) is found by branch-and-bound enumeration with
a 1e-9 value tolerance; among matchings of equal value the one whose sorted content edge list
is lexicographically smallest is chosen, so the result is invariant to member relabelling and
input order. Above the enumeration guard of 24 candidate edges the exact enumeration is
intractable and a deterministic content-lexicographic greedy selection is used instead, which
the pre-match records. Ported from the study pipeline as is.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .data import Member
from .dedupe import relation_key
from .similarity import Similarity

ENUMERATION_GUARD = 24
VALUE_TOLERANCE = 1e-9

Edge = tuple[int, int, float]


def content_greedy_matching(
    left: Sequence[Member], right: Sequence[Member], edges: Sequence[Edge], names: Sequence[str]
) -> list[Edge]:
    """Edges taken in (similarity descending, unordered content-key pair ascending) order, skipping conflicts."""

    def edge_key(edge: Edge) -> tuple[Any, ...]:
        left_key = relation_key(left[edge[0]].relation, names)
        right_key = relation_key(right[edge[1]].relation, names)
        low, high = sorted((left_key, right_key))
        return (-edge[2], low, high)

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    chosen: list[Edge] = []
    for row, col, value in sorted(edges, key=edge_key):
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        chosen.append((row, col, value))
    return chosen


def canonical_pair_matching(
    left: Sequence[Member], right: Sequence[Member], threshold: float, sim: Similarity
) -> tuple[list[Edge], bool]:
    """The content-canonical maximum matching of two members' relations, and whether the guard fell back."""
    names = sim.names
    edges: list[Edge] = []
    for row, left_member in enumerate(left):
        for col, right_member in enumerate(right):
            value = sim.pair(left_member.relation, right_member.relation)
            if value >= threshold:
                edges.append((row, col, value))
    if len(edges) > ENUMERATION_GUARD:
        return content_greedy_matching(left, right, edges, names), True
    edges.sort(
        key=lambda edge: (
            relation_key(left[edge[0]].relation, names),
            relation_key(right[edge[1]].relation, names),
            edge[0],
            edge[1],
        )
    )
    best_value = -1.0
    best_key: tuple[Any, ...] | None = None
    best: list[Edge] = []

    def content_key(chosen: list[Edge]) -> tuple[Any, ...]:
        return tuple(
            sorted(
                (relation_key(left[row].relation, names), relation_key(right[col].relation, names))
                for row, col, _ in chosen
            )
        )

    remaining = [0.0] * (len(edges) + 1)
    for index in range(len(edges) - 1, -1, -1):
        remaining[index] = remaining[index + 1] + edges[index][2]

    def recurse(
        index: int, used_rows: frozenset[int], used_cols: frozenset[int], chosen: list[Edge], value: float
    ) -> None:
        nonlocal best_value, best_key, best
        if value + remaining[index] < best_value - VALUE_TOLERANCE:
            return
        if index == len(edges):
            key = content_key(chosen)
            if value > best_value + VALUE_TOLERANCE or (
                abs(value - best_value) <= VALUE_TOLERANCE and (best_key is None or key < best_key)
            ):
                best_value, best_key, best = value, key, list(chosen)
            return
        row, col, edge_value = edges[index]
        if row not in used_rows and col not in used_cols:
            taken = [*chosen, (row, col, edge_value)]
            recurse(index + 1, used_rows | {row}, used_cols | {col}, taken, value + edge_value)
        recurse(index + 1, used_rows, used_cols, chosen, value)

    recurse(0, frozenset(), frozenset(), [], 0.0)
    return best, False


__all__ = ["ENUMERATION_GUARD", "VALUE_TOLERANCE", "Edge", "canonical_pair_matching", "content_greedy_matching"]
