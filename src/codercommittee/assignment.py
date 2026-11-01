"""Exact maximum-weight one-to-one assignment, written here so scoring needs no scipy.

The matrix is padded to a square with zeros and the Hungarian method (Kuhn-Munkres with
potentials, O(n^3)) minimises ``max - value``; padded cells are dropped. Among equal optima
the method's own scan order decides, which is why parity against a run scored with another
solver compares the matched index sets and, where they differ, the values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Assignment:
    """One assigned cell: its row, its column and the value there."""

    row: int
    col: int
    value: float


def _hungarian(cost: list[list[float]]) -> list[int]:
    """The column assigned to each row of a square cost matrix, minimising the total cost."""
    n = len(cost)
    infinity = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [infinity] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = infinity
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def maximize(matrix: Sequence[Sequence[float]]) -> list[Assignment]:
    """The assignment of rows to columns with the largest total value, by value descending.

    An empty matrix gives no assignment. Rows and columns beyond the smaller dimension stay
    unassigned.
    """
    rows = [[float(value) for value in row] for row in matrix]
    if not rows or not rows[0]:
        return []
    n_rows, n_cols = len(rows), len(rows[0])
    if any(len(row) != n_cols for row in rows):
        raise ValueError("the matrix must be rectangular")
    n = max(n_rows, n_cols)
    padded = [[rows[i][j] if i < n_rows and j < n_cols else 0.0 for j in range(n)] for i in range(n)]
    top = max(max(row) for row in padded)
    cost = [[top - value for value in row] for row in padded]
    columns = _hungarian(cost)
    out = [Assignment(i, columns[i], rows[i][columns[i]]) for i in range(n_rows) if columns[i] < n_cols]
    out.sort(key=lambda assignment: assignment.value, reverse=True)
    return out


__all__ = ["Assignment", "maximize"]
