from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from codercommittee.coders import MockCoder, ModelCoder
from codercommittee.collect import collect, required_texts
from codercommittee.data import Answer, Item, Relation, Task
from codercommittee.protocols import Completion
from codercommittee.similarity import Space, text_sha256

TUPLES = Task("tuples")
LABEL = Task("label", {"yes": ""})


class CountingEmbedder:
    def __init__(self) -> None:
        self.requests: list[tuple[Any, list[str]]] = []

    def space(self, item: Item, texts: Sequence[str]) -> Space:
        self.requests.append((item.id, list(texts)))
        return Space({text: [float(1 + index), 1.0] for index, text in enumerate(texts)})


def test_collect_asks_every_coder_every_draw_and_embeds_once_per_item() -> None:
    first = Item("a", "one two three four five six", [Relation(cause="one two", effect="three")])
    items = [first, Item("b", "x y z w")]
    coders = [MockCoder("p"), MockCoder("q")]
    embedder = CountingEmbedder()
    run = collect(items, TUPLES, coders, draws=2, embedder=embedder)
    assert run.coders == ("p", "q") and run.draws == 2 and len(run.answers) == 8
    assert [item_id for item_id, _ in embedder.requests] == ["a", "b"]
    assert all(answer.ok for answer in run.answers.values())
    assert run.similarity("a") is not None and run.similarity("zzz") is None
    with pytest.raises(ValueError):
        collect(items, TUPLES, coders)  # no embedder for a span task
    with pytest.raises(ValueError):
        collect(items, LABEL, [MockCoder("p"), MockCoder("p")])


def test_required_texts_are_the_references_and_valid_answers_spans_in_sha_order() -> None:
    item = Item("a", "t", [Relation(cause="r1", effect="r2")])
    answers = [
        Answer("a", "p", 1, (Relation(cause="a1", effect="r2"),), "ok"),
        Answer("a", "p", 2, (Relation(cause="bad", effect="bad"),), "invalid"),
    ]
    texts = required_texts(TUPLES, item, answers)
    assert set(texts) == {"r1", "r2", "a1"} and texts == sorted(texts, key=text_sha256)
    assert required_texts(LABEL, Item("a", "t", True), []) == []
    one = Task("tuple")
    assert required_texts(one, Item("a", "t", []), [Answer("a", "p", 1, None, "ok")]) == []
    assert required_texts(one, Item("a", "t", []), [Answer("a", "p", 1, Relation(cause="c", effect="e"), "ok")]) == (
        sorted(["c", "e"], key=text_sha256)
    )


def test_invalid_and_refused_answers_are_asked_again_inside_one_budget() -> None:
    calls: list[int] = []

    def flaky(item: Item, task: Task, draw: int) -> Any:
        calls.append(draw)
        return {"yes": True} if len(calls) % 3 == 0 else {"yes": "no"}

    run = collect([Item("a", "t", True)], LABEL, [MockCoder("p", flaky)], draws=1, attempts=3)
    answer = run.answer("a", "p", 1)
    assert answer.ok and answer.audit["attempts"] == 3 and len(calls) == 3

    calls.clear()
    run = collect([Item("a", "t", True)], LABEL, [MockCoder("p", flaky)], draws=1, attempts=2)
    assert run.answer("a", "p", 1).status == "invalid" and run.complete() == ()

    class Refusing:
        name = "r"

        def __init__(self) -> None:
            self.calls = 0

        def answer(self, item: Item, task: Task, draw: int) -> Answer:
            self.calls += 1
            return Answer(item.id, "r", draw, None, "refused", reason="provider_refusal")

    coder = Refusing()
    run = collect([Item("a", "t", True)], LABEL, [coder], draws=1, attempts=3)
    answer = run.answer("a", "r", 1)
    assert coder.calls == 3 and answer.status == "refused" and answer.audit["attempts"] == 3


def test_mock_coder_answers_every_kind_validly() -> None:
    item = Item("a", "the council cut the subsidy so the buses stopped running at night")
    labels = Task("labels", {"a": "", "b": "", "c": ""}, per_answer=(1, 2))
    for task in (LABEL, labels, Task("tuple"), TUPLES):
        for draw in (1, 2, 3):
            answer = MockCoder("m").answer(item, task, draw)
            assert answer.ok, (task.kind, answer.reason)
    assert MockCoder("m").answer(Item("b", "two words"), TUPLES, 1).value == ()


class Replies:
    """A model whose completions are scripted: a reply text with the transport attempts it took."""

    def __init__(self, *completions: Completion) -> None:
        self.queue = list(completions)
        self.calls = 0

    def complete(self, prompt: str, schema: Any, *, system: str) -> Completion:
        self.calls += 1
        return self.queue.pop(0)


def test_a_refusal_followed_by_a_valid_reply_is_an_ok_answer_as_the_study_counted_it() -> None:
    model = Replies(
        Completion("", {"attempts": 1}, "provider_incomplete_finish:refusal", True),
        Completion('{"yes": true}', {"attempts": 1}),
    )
    run = collect([Item("a", "t", True)], LABEL, [ModelCoder("m", model)], draws=1, attempts=3)
    answer = run.answer("a", "m", 1)
    assert answer.ok and answer.audit["attempts"] == 2 and model.calls == 2 and run.scored() == run.items


def test_transport_attempts_count_toward_the_budget() -> None:
    exhausted = Replies(Completion("", {"attempts": 3}, "APITimeoutError: slow"))
    run = collect([Item("a", "t", True)], LABEL, [ModelCoder("m", exhausted)], draws=1, attempts=3)
    answer = run.answer("a", "m", 1)
    assert answer.status == "invalid" and answer.audit["attempts"] == 3 and exhausted.calls == 1
    partial = Replies(Completion("{", {"attempts": 2}), Completion('{"yes": false}', {"attempts": 1}))
    run = collect([Item("a", "t", True)], LABEL, [ModelCoder("m", partial)], draws=1, attempts=3)
    answer = run.answer("a", "m", 1)
    assert answer.ok and answer.value is False and answer.audit["attempts"] == 3 and partial.calls == 2
    over = Replies(Completion("{", {"attempts": 2}), Completion("{", {"attempts": 2}))
    run = collect([Item("a", "t", True)], LABEL, [ModelCoder("m", over)], draws=1, attempts=3)
    assert run.answer("a", "m", 1).status == "invalid" and over.calls == 2


def test_a_call_never_spends_more_transport_attempts_than_the_budget_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    import codercommittee.providers.base as base
    from codercommittee.providers import OpenAIModel

    monkeypatch.setattr(base, "_sleep", lambda seconds: None)

    class Failing:
        def __init__(self) -> None:
            self.calls = 0
            self.responses = type("R", (), {})()
            self.responses.create = self._create

        def _create(self, **arguments: Any) -> Any:
            self.calls += 1
            raise TimeoutError("slow")

    client = Failing()
    run = collect([Item("a", "t", True)], LABEL, [ModelCoder("m", OpenAIModel("gpt-x", client=client))], draws=1)
    answer = run.answer("a", "m", 1)
    assert client.calls == 3 and answer.status == "invalid" and answer.audit["attempts"] == 3
    # A round that reports more than the remainder is capped in the record.
    greedy = Replies(Completion("{", {"attempts": 2}), Completion("{", {"attempts": 3}))
    run = collect([Item("a", "t", True)], LABEL, [ModelCoder("m", greedy)], draws=1)
    assert run.answer("a", "m", 1).audit["attempts"] == 3
    assert base.TRANSPORT_BUDGET.get() is None  # reset after every call


def test_discarded_attempts_are_recorded_in_the_audit() -> None:
    model = Replies(
        Completion("{", {"attempts": 1, "provider_response_id": "r1"}), Completion('{"yes": true}', {"attempts": 1})
    )
    run = collect([Item("a", "t", True)], LABEL, [ModelCoder("m", model)], draws=1)
    audit = run.answer("a", "m", 1).audit
    assert audit["attempts"] == 2 and len(audit["discarded"]) == 1
    assert (
        audit["discarded"][0]["status"] == "invalid"
        and audit["discarded"][0]["identity"]["provider_response_id"] == "r1"
    )
