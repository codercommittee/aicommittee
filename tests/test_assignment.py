from __future__ import annotations

import itertools

import numpy as np
import pytest

from codercommittee.assignment import maximize


def test_optimal_beats_greedy_on_the_study_example() -> None:
    matrix = [[0.85, 0.70], [0.83, 0.40]]
    assert sum(a.value for a in maximize(matrix)) == pytest.approx(1.53)
    thresholded = [[0.85, 0.70], [0.83, 0.0]]
    score = sum(a.value for a in maximize(thresholded) if a.value >= 0.60) / 2
    assert score == pytest.approx(0.765)
    assert [(a.row, a.col) for a in maximize(thresholded)] == [(1, 0), (0, 1)]


def test_matches_brute_force_on_square_and_rectangular_matrices() -> None:
    rng = np.random.default_rng(42)
    for n_rows in range(1, 6):
        for n_cols in range(1, 6):
            for _ in range(5):
                matrix = rng.random((n_rows, n_cols))
                if rng.random() < 0.5:
                    matrix[matrix < 0.4] = 0.0  # zeros, as a thresholded score matrix has
                k = min(n_rows, n_cols)
                expected = max(
                    sum(matrix[row, col] for row, col in zip(rows, cols, strict=True))
                    for rows in itertools.permutations(range(n_rows), k)
                    for cols in itertools.combinations(range(n_cols), k)
                )
                found = maximize(matrix.tolist())
                assert len(found) == k
                assert len({a.row for a in found}) == k and len({a.col for a in found}) == k
                assert sum(a.value for a in found) == pytest.approx(expected)
                assert [a.value for a in found] == sorted((a.value for a in found), reverse=True)


def test_empty_and_malformed_matrices() -> None:
    assert maximize([]) == []
    assert maximize([[]]) == []
    with pytest.raises(ValueError):
        maximize([[1.0, 2.0], [1.0]])
