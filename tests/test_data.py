from __future__ import annotations

import pytest

from codercommittee.data import Answer, Item, Relation, Run, Task


def test_relation_is_an_immutable_mapping_with_attribute_access() -> None:
    relation = Relation(cause=" rain ", effect="flooding", signal="")
    assert relation.cause == "rain" and relation["effect"] == "flooding"
    assert relation.get("missing") == "" and "cause" in relation and len(relation) == 3
    assert relation.as_dict() == {"cause": "rain", "effect": "flooding", "signal": ""}
    assert relation == Relation(cause="rain", effect="flooding", signal="") and hash(relation) == hash(relation)
    with pytest.raises(AttributeError):
        relation.cause = "x"
    with pytest.raises(AttributeError):
        _ = relation.nothing
    with pytest.raises(TypeError):
        Relation(cause=3)  # type: ignore[arg-type]


def test_task_defaults_keys_and_schemas() -> None:
    label = Task("label", {"is_ai_patent": "AI-related"})
    assert label.key == "is_ai_patent" and label.schema()["properties"] == {"is_ai_patent": {"type": "boolean"}}
    labels = Task("labels", {"a": "", "b": ""}, key="frames", per_answer=(1, 2))
    assert labels.schema()["properties"]["frames"]["items"]["enum"] == ["a", "b"]
    tuples = Task("tuples")
    assert tuples.key == "relations" and tuples.names == ("cause", "effect", "signal")
    assert tuples.schema()["properties"]["relations"]["items"]["required"] == ["cause", "effect", "signal"]
    one = Task("tuple")
    assert one.key == "causal" and set(one.schema()["required"]) == {"causal", "cause", "effect", "signal"}
    assert one.render("the text") == "the text"


def test_task_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        Task("label", {})
    with pytest.raises(ValueError):
        Task("label", {"a": "", "b": ""})
    with pytest.raises(ValueError):
        Task("labels", {})
    with pytest.raises(ValueError):
        Task("labels", {"a": ""}, per_answer=(2, 1))
    with pytest.raises(ValueError):
        Task("tuples", fields=("cause", "cause"))
    with pytest.raises(ValueError):
        Task("tuple", key="cause")
    with pytest.raises(ValueError):
        Task("label", {"x": ""}, prompt="no placeholder")
    with pytest.raises(ValueError):
        Task("other", {"x": ""})  # type: ignore[arg-type]


def test_item_normalizes_references() -> None:
    assert Item("a", "t", True).reference is True
    assert Item(1, "t", []).reference == ()
    assert Item("a", "t", {"b", "a"}).reference == frozenset({"a", "b"})
    relations = [Relation(cause="a", effect="b")]
    assert Item("a", "t", relations).reference == tuple(relations)
    with pytest.raises(TypeError):
        Item("a", "t", ["x"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        Item(True, "t")


def _answer(item: str, coder: str, draw: int, status: str = "ok") -> Answer:
    return Answer(item, coder, draw, True if status == "ok" else None, status)  # type: ignore[arg-type]


def test_run_complete_and_scored_populations() -> None:
    task = Task("label", {"yes": ""})
    items = (Item("a", "t", True), Item("b", "t", None), Item("c", "t", False))
    answers = {}
    for item in items:
        for coder in ("x", "y"):
            for draw in (1, 2):
                status = "refused" if (item.id, coder, draw) == ("c", "y", 2) else "ok"
                answers[(item.id, coder, draw)] = _answer(item.id, coder, draw, status)  # type: ignore[arg-type]
    run = Run(task, items, ("x", "y"), 2, answers)
    assert [item.id for item in run.complete()] == ["a", "b"]
    assert [item.id for item in run.scored()] == ["a"]
    assert run.answer("zzz", "x", 1).status == "missing"
    with pytest.raises(ValueError):
        Run(task, items, ("x", "x"), 2, answers)


def test_tuple_kind_scores_causal_references_only() -> None:
    task = Task("tuple")
    items = (Item(1, "t", []), Item(2, "t", [Relation(cause="a", effect="b")]))
    answers = {(i.id, "x", 1): Answer(i.id, "x", 1, None, "ok") for i in items}
    run = Run(task, items, ("x",), 1, answers)
    assert [item.id for item in run.complete()] == [1, 2]
    assert [item.id for item in run.scored()] == [2]


def test_coder_names_and_codebook_names_cannot_carry_what_the_labels_reserve() -> None:
    task = Task("label", {"yes": ""})
    items = (Item("a", "t", True),)
    for name in ("b|c", "", " a", "a "):
        with pytest.raises(ValueError, match="coder name"):
            Run(task, items, (name,), 1, {})
    assert Run(task, items, ("claude-sonnet-4-5", "qwen3:32b"), 1, {}).coders == ("claude-sonnet-4-5", "qwen3:32b")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        Task("labels", {"x": "", " x": ""})
    with pytest.raises(ValueError, match="surrounding whitespace"):
        Task("label", {"yes ": ""})


def test_every_answer_of_a_run_agrees_with_its_key() -> None:
    task = Task("label", {"yes": ""})
    items = (Item(0, "t", True),)
    good: dict[tuple[str | int, str, int], Answer] = {(0, "a", 1): Answer(0, "a", 1, True, "ok")}
    assert Run(task, items, ("a",), 1, good).answers == good
    for key, answer in (
        ((0, "a", 1), Answer(0, "b", 1, True, "ok")),
        ((0, "a", 1), Answer(1, "a", 1, True, "ok")),
        ((0, "a", 1), Answer(0, "a", 2, True, "ok")),
        ((0, "z", 1), Answer(0, "z", 1, True, "ok")),
        ((9, "a", 1), Answer(9, "a", 1, True, "ok")),
        ((0, "a", 2), Answer(0, "a", 2, True, "ok")),
    ):
        with pytest.raises(ValueError, match="answer"):
            Run(task, items, ("a",), 1, {key: answer})
    for name in ("a\nb", "a\tb", "a\u2028b"):
        with pytest.raises(ValueError, match="line break"):
            Run(task, items, (name,), 1, {})
    with pytest.raises(ValueError, match="line breaks"):
        Task("labels", {"a\nb": "", "c": ""})
