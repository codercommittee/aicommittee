"""Strict JSON: a garbage reply is an invalid answer, never an exception that ends the run."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from codercommittee import Item, MockCoder, ModelCoder, Task, collect
from codercommittee.protocols import Coder, Completion
from codercommittee.strictjson import StrictJSONError, strict_json_loads

LABEL = Task("label", {"yes": ""})
GARBAGE = (
    '{"yes": ' + "9" * (sys.get_int_max_str_digits() + 1) + "}",
    "[" * 2000 + "0" + "]" * 2000,
    "{" * 2000,
    '{"yes": NaN}',
    '{"yes": true, "yes": false}',
    '"\\ud800"',
    "",
    "{",
)


@pytest.mark.parametrize(
    "text", GARBAGE, ids=["digits", "deep-array", "deep-objects", "nan", "duplicate", "surrogate", "empty", "cut"]
)
def test_garbage_is_a_strict_json_error(text: str) -> None:
    with pytest.raises(StrictJSONError):
        strict_json_loads(text)


@pytest.mark.parametrize("text", GARBAGE[:3], ids=["digits", "deep-array", "deep-objects"])
def test_a_garbage_reply_is_retried_like_any_malformed_one(text: str) -> None:
    class Fake:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, schema: Any, *, system: str) -> Completion:
            self.calls += 1
            return Completion(text if self.calls == 1 else '{"yes": true}')

    fake = Fake()
    coders: list[Coder] = [ModelCoder("a", fake), MockCoder("b"), MockCoder("c")]
    run = collect([Item(11, "text", True)], LABEL, coders, draws=1)
    assert len(run.complete()) == 1 and fake.calls == 2
    assert run.answer(11, "a", 1).ok
