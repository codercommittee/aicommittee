"""Store and file integrity under the audit's cases: separators inside strings, torn appends, atomic
snapshots, conflicting saves, explicit empty task keys, stored booleans, raw-less answers, attempt counts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from codercommittee import (
    Item,
    MockCoder,
    MockEmbedder,
    ModelCoder,
    Relation,
    Run,
    RunStore,
    StoreError,
    Task,
    collect,
    read_items,
    write_items,
)
from codercommittee.align import prematch
from codercommittee.data import Answer
from codercommittee.protocols import Completion
from codercommittee.similarity import Similarity, Space
from codercommittee.store import StoreReferee, task_from_json, task_to_json

LABEL = Task("label", {"yes": ""})
LABELS = Task("labels", {"a": "", "b": ""})
TUPLE = Task("tuple")
TUPLES = Task("tuples")
SEPARATORS = ("\u0085", "\u2028", "\u2029", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e")


def constant(name: str, value: bool = True) -> MockCoder:
    return MockCoder(name, lambda item, task, draw: {task.key: value})


def relation_run(
    task: Task = TUPLES,
    text: str = "rain causes flood",
    cause: str = "rain",
    embedder: Any = None,
    store: RunStore | None = None,
) -> Run:
    relation = Relation(cause=cause, effect="flood")

    def script(item: Item, t: Task, draw: int) -> Any:
        return {t.key: [{"cause": relation.cause, "effect": relation.effect, **dict.fromkeys(t.aux, "")}]}

    coders = [MockCoder(name, script) for name in "abc"]
    return collect([Item(11, text, [relation])], task, coders, embedder=embedder or MockEmbedder(), store=store)


class Counting:
    def __init__(self, name: str) -> None:
        self.inner = MockCoder(name)
        self.name = name
        self.calls = 0

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        self.calls += 1
        return self.inner.answer(item, task, draw)


class CountingEmbedder(MockEmbedder):
    def __init__(self) -> None:
        super().__init__(32)
        self.calls = 0

    def space(self, item: Item, texts: Any) -> Space:
        self.calls += 1
        return super().space(item, texts)


@pytest.mark.parametrize("separator", SEPARATORS, ids=[f"U+{ord(c):04X}" for c in SEPARATORS])
def test_line_separators_inside_strings_round_trip_through_files_and_the_store(tmp_path: Path, separator: str) -> None:
    text = f"prefix{separator}suffix"
    path = tmp_path / "items.jsonl"
    write_items([Item(11, text, True)], path, LABEL)
    assert [item.text for item in read_items(path, LABEL)] == [text]
    run = relation_run(text=f"heavy{separator}rain causes flood", cause=f"heavy{separator}rain")
    run.save(tmp_path / "run")
    back = Run.load(tmp_path / "run")
    assert len(back.complete()) == 1 and back.items[0].text == run.items[0].text
    assert back.answer(11, "a", 1).value == run.answer(11, "a", 1).value


def test_carriage_return_line_endings_read(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    path.write_bytes(b'{"id": 1, "text": "one", "reference": true}\r\n{"id": 2, "text": "two"}\r\n')
    assert [item.id for item in read_items(path, LABEL)] == [1, 2]


def test_a_torn_last_line_is_dropped_and_its_slot_asked_again(tmp_path: Path) -> None:
    items = [Item(1, "one text", True), Item(2, "two text", False)]
    coders = [Counting("a"), Counting("b"), Counting("c")]
    reference = collect(items, LABEL, [MockCoder("a"), MockCoder("b"), MockCoder("c")])
    collect(items, LABEL, coders, store=RunStore(tmp_path / "run"))
    path = tmp_path / "run" / "answers.jsonl"
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:-25], encoding="utf-8")  # the crash cut the last append short
    resumed = collect(items, LABEL, coders, store=RunStore(tmp_path / "run"))
    assert [c.calls for c in coders] == [6, 6, 7]
    assert {k: a.value for k, a in resumed.answers.items()} == {k: a.value for k, a in reference.answers.items()}
    repaired = path.read_text(encoding="utf-8")
    assert repaired.endswith("\n") and len(repaired.splitlines()) == 18
    assert all(json.loads(line) for line in repaired.splitlines())
    store = RunStore(tmp_path / "run")
    store.answered()
    assert not store.torn_tail
    # A complete line that is not JSON is still an error, torn tail or not.
    lines = repaired.splitlines()
    path.write_text("\n".join([lines[0], "{bad", *lines[2:]]) + "\n", encoding="utf-8")
    with pytest.raises(StoreError, match=r"answers\.jsonl:2: not strict JSON"):
        RunStore(tmp_path / "run").answered()


def test_snapshots_are_written_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "run"
    embedder = CountingEmbedder()
    run = relation_run(store=RunStore(directory), embedder=embedder)
    assert embedder.calls == 1 and not list(directory.rglob("*.tmp"))
    # A crash between the temporary file and the replace leaves no half-written space.
    space_path = directory / "spaces" / "000000.json"
    space_path.unlink()
    (directory / "spaces" / "000000.json.tmp").write_text("{truncated", encoding="utf-8")
    resumed = relation_run(store=RunStore(directory), embedder=embedder)
    assert embedder.calls == 2 and len(resumed.complete()) == 1 and space_path.is_file()
    assert resumed.spaces[11].texts == run.spaces[11].texts
    # The manifest: a replace that fails leaves no run.json at all.
    original = os.replace

    def crash(src: Any, dst: Any) -> None:
        if str(dst).endswith("run.json"):
            raise OSError("simulated crash")
        original(src, dst)

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="simulated"):
        RunStore(tmp_path / "other").create(LABEL, ["a"], 1, [Item(1, "t", True)])
    assert not (tmp_path / "other" / "run.json").exists() and not RunStore(tmp_path / "other").exists


def test_saving_a_different_run_into_the_same_directory_is_refused(tmp_path: Path) -> None:
    item = [Item(11, "a legal text", True)]
    old = collect(item, LABEL, [constant(n, True) for n in "abc"])
    new = collect(item, LABEL, [constant(n, False) for n in "abc"])
    old.save(tmp_path / "run")
    old.save(tmp_path / "run")  # the same run again is idempotent
    with pytest.raises(StoreError, match="different answer for item 11, coder a, draw 1"):
        new.save(tmp_path / "run")
    assert Run.load(tmp_path / "run").answer(11, "a", 1).value is True
    narrow = relation_run()
    wide = collect(
        narrow.items, TUPLES, [MockCoder(n, lambda i, t, d: {t.key: []}) for n in "abc"], embedder=MockEmbedder()
    )
    narrow.save(tmp_path / "spaces")
    with pytest.raises(StoreError, match="different answer"):
        wide.save(tmp_path / "spaces")
    other_vectors = relation_run(embedder=MockEmbedder(16))
    with pytest.raises(StoreError, match="different vectors"):
        other_vectors.save(tmp_path / "spaces")


def test_an_explicit_empty_aux_survives_the_manifest(tmp_path: Path) -> None:
    task = Task("tuples", aux=())
    run = relation_run(task=task)
    run.save(tmp_path / "run")
    assert RunStore(tmp_path / "run").task().aux == () and len(Run.load(tmp_path / "run").complete()) == 1
    for original in (
        task,
        LABELS,
        Task("labels", {"a": ""}, per_answer=(0, 1)),
        Task("tuples", fields=("x", "y"), aux=()),
    ):
        assert task_from_json(task_to_json(original)) == original
    assert task_from_json({"kind": "tuples"}) == TUPLES


def test_stored_decisions_must_be_json_booleans(tmp_path: Path) -> None:
    relation = Relation(cause="rain", effect="flood")
    sim = Similarity(Space({"rain": [1.0, 0.0], "flood": [0.0, 1.0]}))
    pre = prematch({"a": (relation,), "b": (relation,)}, sim, accept_at=0.8, item=11, label="panel:1")
    store = RunStore(tmp_path / "run")
    store.create(TUPLES, ["a", "b"], 1, [Item(11, "rain causes flood", [relation])])
    store.put_decision(11, "panel:1", pre.sha256, dict.fromkeys(pre.candidate_pairs, False), {})
    assert set(StoreReferee(RunStore(tmp_path / "run")).decide(pre).values()) == {False}
    path = tmp_path / "run" / "decisions.jsonl"
    record = json.loads(path.read_text())
    record["decisions"] = dict.fromkeys(pre.candidate_pairs, "false")
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(StoreError, match="JSON booleans"):
        StoreReferee(RunStore(tmp_path / "run")).decide(pre)


class Handmade:
    """A coder that builds its Answer itself, without a raw reply."""

    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        return Answer(item.id, self.name, draw, self.value, "ok")


@pytest.mark.parametrize(
    ("task", "item", "value"),
    [
        (LABEL, Item(11, "a legal text", True), True),
        (LABELS, Item(11, "a legal text", {"a"}), frozenset({"b"})),
        (
            TUPLE,
            Item(11, "rain causes flood", [Relation(cause="rain", effect="flood")]),
            Relation(cause="rain", effect="flood"),
        ),
        (TUPLE, Item(12, "quiet", []), None),
        (
            TUPLES,
            Item(11, "rain causes flood", [Relation(cause="rain", effect="flood")]),
            (Relation(cause="rain", effect="flood"),),
        ),
    ],
    ids=["label", "labels", "tuple", "tuple-negative", "tuples"],
)
def test_answers_without_a_raw_reply_reload_as_ok(tmp_path: Path, task: Task, item: Item, value: Any) -> None:
    embedder = MockEmbedder() if task.kind in ("tuple", "tuples") else None
    run = collect([item], task, [Handmade(n, value) for n in "abc"], embedder=embedder)
    run.save(tmp_path / "run")
    stored = RunStore(tmp_path / "run").answer(item.id, "a", 1)
    assert stored is not None and stored["audit"]["raw_synthesized"] is True and isinstance(stored["raw"], str)
    back = Run.load(tmp_path / "run")
    restored = back.answer(item.id, "a", 1)
    assert restored.status == "ok" and restored.value == value
    assert len(back.complete()) == 1 and len(back.scored()) == len(run.scored())


def test_loaded_attempt_counts_are_the_original_ones(tmp_path: Path) -> None:
    class ReplyAfterRetries:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *args: Any, **kwargs: Any) -> Completion:
            self.calls += 1
            return Completion("{" if self.calls < 3 else '{"yes": true}')

    model = ReplyAfterRetries()
    run = collect([Item(11, "a legal text", True)], LABEL, [ModelCoder("a", model), constant("b"), constant("c")])
    assert run.answer(11, "a", 1).audit["attempts"] == 3
    run.save(tmp_path / "run")
    assert Run.load(tmp_path / "run").answer(11, "a", 1).audit["attempts"] == 3
    assert Run.load(tmp_path / "run").answer(11, "b", 1).audit["attempts"] == 1


def test_an_unterminated_complete_last_line_is_never_glued_to_the_next_append(tmp_path: Path) -> None:
    items = [Item(1, "one text", True)]
    coders = [Counting("a"), Counting("b"), Counting("c")]
    collect(items, LABEL, coders, store=RunStore(tmp_path / "run"))
    path = tmp_path / "run" / "answers.jsonl"
    data = path.read_bytes()
    assert data.endswith(b"\n")
    path.write_bytes(data[:-1])  # the crash cut exactly the final newline
    store = RunStore(tmp_path / "run")
    assert store.answered() == 9 and not store.torn_tail
    store.put_answer(Answer(1, "a", 1, False, "ok", raw='{"yes": false}'), LABEL)
    again = RunStore(tmp_path / "run")
    record = again.answer(1, "a", 1)
    assert again.answered() == 9 and record is not None and record["value"] == {"yes": False}
    assert len(path.read_text(encoding="utf-8").splitlines()) == 10


def test_a_tail_torn_inside_a_multibyte_character_is_dropped(tmp_path: Path) -> None:
    run = relation_run(text="la pluie dévastatrice provoqua la flood", cause="pluie dévastatrice")
    run.save(tmp_path / "run")
    path = tmp_path / "run" / "answers.jsonl"
    data = path.read_bytes()
    last_line_start = data.rstrip(b"\n").rfind(b"\n") + 1
    glyph = "é".encode()
    position = data.find(glyph, last_line_start)
    assert position > 0
    path.write_bytes(data[: position + 1])  # cut after the first byte of é
    store = RunStore(tmp_path / "run")
    assert store.answered() == 8 and store.torn_tail
    assert path.read_bytes().endswith(b"\n") and len(path.read_text(encoding="utf-8").splitlines()) == 8


def test_saving_the_same_span_run_again_is_idempotent_even_with_reordered_or_duplicate_relations(
    tmp_path: Path,
) -> None:
    text = "heavy rain and snow filled the river so the bridge washed out"
    relations = [
        {"cause": "snow", "effect": "the bridge washed out", "signal": ""},
        {"cause": "heavy rain", "effect": "the bridge washed out", "signal": ""},
    ]

    def script(item: Item, task: Task, draw: int) -> Any:
        return {task.key: [*relations, relations[0]]}  # out of key order, with a duplicate

    coders = [MockCoder(name, script) for name in "abc"]
    item = Item(13, text, [Relation(cause="heavy rain", effect="the bridge washed out")])
    run = collect([item], TUPLES, coders, embedder=MockEmbedder(), store=RunStore(tmp_path / "run"))
    assert len(run.answer(13, "a", 1).value) == 2  # type: ignore[arg-type]
    run.save(tmp_path / "run")
    loaded = Run.load(tmp_path / "run")
    loaded.save(tmp_path / "run")
    loaded.save(tmp_path / "copy")
    assert Run.load(tmp_path / "copy").answer(13, "a", 1).value == run.answer(13, "a", 1).value
    assert len((tmp_path / "run" / "answers.jsonl").read_text().splitlines()) == 9


def test_a_refused_save_writes_nothing(tmp_path: Path) -> None:
    item = [Item(11, "a legal text", True)]
    first = collect(item, LABEL, [constant(n, True) for n in "abc"])
    first.save(tmp_path / "run")
    path = tmp_path / "run" / "answers.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line for line in lines if '"coder": "a"' not in line) + "\n")
    assert RunStore(tmp_path / "run").answered() == 6
    mixed = collect(item, LABEL, [constant("a", True), constant("b", False), constant("c", True)])
    with pytest.raises(StoreError, match="coder b"):
        mixed.save(tmp_path / "run")
    assert RunStore(tmp_path / "run").answered() == 6


def test_the_writer_refuses_non_boolean_decisions(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.create(TUPLES, ["a", "b"], 1, [Item(11, "rain causes flood", [Relation(cause="rain", effect="flood")])])
    with pytest.raises(StoreError, match="not booleans"):
        store.put_decision(11, "panel:1", "0" * 64, {"a:0|b:0": 1}, {})  # type: ignore[dict-item]
    assert not store.has_decisions


def test_items_files_may_start_with_a_bom(tmp_path: Path) -> None:
    csv_path = tmp_path / "items.csv"
    csv_path.write_bytes(b"\xef\xbb\xbfid,text,reference\r\n1,one,true\r\n")
    assert [(i.id, i.reference) for i in read_items(csv_path, LABEL, int_ids=True)] == [(1, True)]
    jsonl_path = tmp_path / "items.jsonl"
    jsonl_path.write_bytes(b'\xef\xbb\xbf{"id": 1, "text": "one"}\n')
    assert read_items(jsonl_path, LABEL)[0].id == 1


def test_a_manifest_missing_a_key_is_a_store_error(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "run.json").write_text(
        json.dumps({"version": "codercommittee-run-v1", "task": task_to_json(LABEL)})
    )
    with pytest.raises(StoreError, match="lacks coders, draws, items"):
        Run.load(tmp_path / "run")
    assert task_from_json({"kind": "tuples", "aux": None, "fields": None, "codebook": None}) == TUPLES


def test_snapshots_are_ordinary_files_and_malformed_manifests_are_store_errors(tmp_path: Path) -> None:
    import stat

    mask = os.umask(0)
    os.umask(mask)
    store = RunStore(tmp_path / "run")
    collect([Item(1, "t", True)], LABEL, [constant("a")], store=store)
    assert stat.S_IMODE((tmp_path / "run" / "run.json").stat().st_mode) == 0o666 & ~mask
    manifest_path = tmp_path / "run" / "run.json"
    good = json.loads(manifest_path.read_text())
    for key, bad in (("provenance", 5), ("task", {}), ("coders", 5), ("draws", None), ("items", 5), ("draws", True)):
        manifest_path.write_text(json.dumps({**good, key: bad}))
        with pytest.raises(StoreError, match=r"run\.json"):
            Run.load(tmp_path / "run")
    manifest_path.write_text(json.dumps({**good, "provenance": {"coder_identities": "x"}}))
    with pytest.raises(StoreError, match="coder_identities"):
        Run.load(tmp_path / "run")
    manifest_path.write_text(json.dumps(good))
    answers = tmp_path / "run" / "answers.jsonl"
    lines = answers.read_text().splitlines()
    answers.write_text('{"item": 1}\n' + "\n".join(lines[1:]) + "\n")
    with pytest.raises(StoreError, match=r"answers\.jsonl:1"):
        Run.load(tmp_path / "run")
    record = json.loads(lines[0])
    answers.write_text(json.dumps({**record, "audit": [3]}) + "\n" + "\n".join(lines[1:]) + "\n")
    with pytest.raises(StoreError, match="audit is not an object"):
        Run.load(tmp_path / "run")
    with pytest.raises(ValueError, match="task record"):
        task_from_json({"kind": "tuples", "per_answer": [1]})


def test_a_run_whose_provenance_is_not_json_is_refused_before_any_write(tmp_path: Path) -> None:
    class Weird:
        def __init__(self, record: Any) -> None:
            self.name = "a"
            self.record = record
            self.inner = MockCoder("a")

        def describe(self) -> Any:
            return self.record

        def answer(self, item: Item, task: Task, draw: int) -> Answer:
            return self.inner.answer(item, task, draw)

    item = [Item(1, "t", True)]
    for record, error in (({"kind": "x", "t": float("nan")}, ValueError), ({"kind": "x", "s": {1, 2}}, TypeError)):
        run = collect(item, LABEL, [Weird(record)], draws=1)
        with pytest.raises(error):
            run.save(tmp_path / "saved")
        assert not (tmp_path / "saved").exists()
        with pytest.raises(error):
            collect(item, LABEL, [Weird(record)], draws=1, store=RunStore(tmp_path / "stored"))
        assert not (tmp_path / "stored" / "run.json").exists()
    with pytest.raises(TypeError, match="mapping or None"):
        collect(item, LABEL, [Weird("claude-x")], draws=1)
