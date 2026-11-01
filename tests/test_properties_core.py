"""Property-based tests of the core: validation, similarity, assignment, strict JSON."""

from __future__ import annotations

import itertools
import math
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from codercommittee.assignment import maximize
from codercommittee.data import Relation, Task
from codercommittee.replay import value_as_stored
from codercommittee.similarity import Similarity, Space
from codercommittee.strictjson import StrictJSONError, canonical_json, strict_json_loads
from codercommittee.validation import Validation, validate, verbatim

LABEL = Task("label", {"yes": ""})
LABELS = Task("labels", {"a": "", "b": "", "c": ""}, per_answer=(1, 2))
TUPLE = Task("tuple")
TUPLES = Task("tuples")
TASKS = [LABEL, LABELS, TUPLE, TUPLES]

json_scalars = (
    st.none()
    | st.booleans()
    | st.integers(-10, 10)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=8)
)
json_values = st.recursive(
    json_scalars,
    lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=6), children, max_size=4),
    max_leaves=12,
)
words = st.lists(st.sampled_from(["rain", "flood", "mud", "bridge", "town", "frost", "mill"]), min_size=1, max_size=4)


@given(json_values, st.sampled_from(TASKS))
@settings(max_examples=150, deadline=None)
def test_validate_never_raises_and_ok_values_fit_the_kind(value: Any, task: Task) -> None:
    result = validate(task, value, "rain flood mud bridge town")
    assert isinstance(result, Validation)
    if not result.ok:
        assert isinstance(result.reason, str) and result.reason and result.value is None
        return
    if task.kind == "label":
        assert isinstance(result.value, bool)
    elif task.kind == "labels":
        assert isinstance(result.value, frozenset) and result.value <= set(task.codebook)
    elif task.kind == "tuple":
        assert result.value is None or isinstance(result.value, Relation)
    else:
        assert isinstance(result.value, tuple) and all(isinstance(r, Relation) for r in result.value)


@given(st.text(max_size=20), st.text(max_size=60))
@settings(max_examples=200, deadline=None)
def test_verbatim_keeps_the_length_and_returns_text_when_it_reports_a_copy(span: str, text: str) -> None:
    out, is_copy = verbatim(span, text)
    assert len(out) == len(span)
    if is_copy:
        assert out == "" or out in text
    else:
        assert out == span


def _text_and_relations(draw: st.DrawFn) -> tuple[str, list[dict[str, str]]]:
    tokens = draw(
        st.lists(
            st.sampled_from(["rain", "flood", "mud", "bridge", "town", "frost", "mill", "so"]), min_size=4, max_size=10
        )
    )
    text = " ".join(tokens)
    relations = []
    for _ in range(draw(st.integers(0, 3))):
        i = draw(st.integers(0, len(tokens) - 2))
        j = draw(st.integers(i + 1, len(tokens) - 1))
        k = draw(st.integers(j, len(tokens) - 1))
        relations.append({"cause": " ".join(tokens[i:j]), "effect": " ".join(tokens[j : k + 1]), "signal": ""})
    return text, relations


@st.composite
def relation_answers(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    text, relations = _text_and_relations(draw)
    return text, {"relations": relations}


@given(relation_answers())
@settings(max_examples=100, deadline=None)
def test_a_valid_relation_answer_round_trips_through_its_stored_shape(case: tuple[str, dict[str, Any]]) -> None:
    text, answer = case
    first = validate(TUPLES, answer, text)
    assert first.ok and isinstance(first.value, tuple)
    again = validate(TUPLES, {"relations": value_as_stored(first.value, TUPLES)}, text)
    assert again.ok and again.value == first.value


vectors = st.lists(st.floats(-1.0, 1.0, allow_nan=False), min_size=4, max_size=4).filter(
    lambda v: any(abs(x) > 1e-6 for x in v)
)


@given(vectors, vectors)
@settings(max_examples=150, deadline=None)
def test_cosine_is_symmetric_clamped_rounded_and_one_on_itself(a: list[float], b: list[float]) -> None:
    space = Space({"a": a, "b": b})
    ab, ba = space.cosine("a", "b"), space.cosine("b", "a")
    assert ab == ba and 0.0 <= ab <= 1.0 and ab == round(ab, 12)
    assert space.cosine("a", "a") == 1.0 and space.cosine("b", "b") == 1.0
    assert space.cosine("a", "b") == ab  # memoised: the same float again
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(math.fsum(x * x for x in a) * math.fsum(y * y for y in b))
    assert ab == round(max(dot / norms, 0.0), 12)
    sim = Similarity(space)
    left, right = Relation(cause="a", effect="b"), Relation(cause="b", effect="a")
    assert sim.pair(left, right) == round(math.fsum([ab, ab]) / 2, 12)
    assert sim.pair(left, left) == 1.0


@given(st.integers(1, 5), st.integers(1, 5), st.data())
@settings(max_examples=100, deadline=None)
def test_assignment_is_optimal_and_a_matching(n_rows: int, n_cols: int, data: st.DataObject) -> None:
    cell = st.one_of(st.just(0.0), st.floats(0.0, 1.0, allow_nan=False))
    matrix = [[data.draw(cell) for _ in range(n_cols)] for _ in range(n_rows)]
    found = maximize(matrix)
    k = min(n_rows, n_cols)
    assert len(found) == k
    assert len({a.row for a in found}) == k and len({a.col for a in found}) == k
    assert [a.value for a in found] == sorted((a.value for a in found), reverse=True)
    expected = max(
        sum(matrix[row][col] for row, col in zip(rows, cols, strict=True))
        for rows in itertools.permutations(range(n_rows), k)
        for cols in itertools.combinations(range(n_cols), k)
    )
    assert abs(sum(a.value for a in found) - expected) < 1e-9


@given(json_values)
@settings(max_examples=150, deadline=None)
def test_strict_json_round_trips_canonical_text(value: Any) -> None:
    assert strict_json_loads(canonical_json(value)) == value


@given(st.text(min_size=1, max_size=6))
@settings(max_examples=50, deadline=None)
def test_strict_json_rejects_duplicate_keys_and_nan(key: str) -> None:
    quoted = canonical_json(key)
    try:
        strict_json_loads("{" + quoted + ": 1, " + quoted + ": 2}")
    except StrictJSONError:
        pass
    else:
        raise AssertionError("duplicate keys were accepted")
    for bad in ("NaN", "Infinity", "-Infinity", "[1, NaN]"):
        try:
            strict_json_loads(bad)
        except StrictJSONError:
            continue
        raise AssertionError(f"{bad} was accepted")
