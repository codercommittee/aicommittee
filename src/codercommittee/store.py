"""Saved, resumable and replayable runs in the package's own layout.

A run directory holds::

    run.json            the task, the coders in seat order, the draws, the items with their references,
                        and the run's provenance (the referee's configuration, the coders' identities)
    answers.jsonl       one line per answer as it arrives (append-only; the last line of a slot wins)
    spaces/NNNNNN.json  the vectors of one item's spans, NNNNNN the item's position in run.json
    decisions.jsonl     one line per referee decision, bound to the pre-match by its sha256

``collect(..., store=RunStore(dir))`` reuses every stored answer with status ``ok``, every stored
refusal that spent the budget it is asked under, and every stored space whose texts are the spans
asked for, asks only for the rest, and appends each new answer before the next call, so a crash
loses nothing and the same command resumes.
A crash inside the final append leaves a line without its newline: that line is dropped and
repaired at the next load (``RunStore.torn_tail``) or before the next append, and its slot is
asked again; the manifest and the spaces are written atomically, so a crash never leaves a
half-written snapshot behind.
``Run.load(dir)`` rebuilds a run through the same ``collect`` path from :class:`StoreCoder` and
:class:`StoreEmbedder`: stored raw replies are re-validated and relation answers re-deduplicated,
and stored parsed values are never authoritative. :class:`LoggedReferee` serves a recorded
decision when the store holds one for the same pre-match and asks its referee only otherwise;
:class:`StoreReferee` serves the records alone. The referee a run was made with is part of its
manifest, so a loaded run scores as the run that wrote it did.
"""

from __future__ import annotations

import json
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .align import Prematch
from .data import Answer, AnswerStatus, Item, ItemId, Run, Task
from .protocols import Referee, describe_referee
from .referee import BandRule, LLMReferee, ReplayReferee
from .similarity import Space, text_sha256
from .strictjson import StrictJSONError, canonical_json, jsonl_lines, strict_json_loads
from .validation import validate

if TYPE_CHECKING:
    from .protocols import Coder, Embedder

STORE_VERSION = "codercommittee-run-v1"
FINAL_STATUSES = frozenset({"ok", "refused"})
REFEREE_KINDS = frozenset({"band", "llm", "replay"})


class StoreError(RuntimeError):
    """The store lacks what was asked for, or holds another run."""


def _canonical_id(item: ItemId) -> str:
    return canonical_json(item)


def _read_json(path: Path) -> Any:
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"))
    except StrictJSONError as exc:
        raise StoreError(f"{path} is not strict JSON: {exc}") from exc


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """The records of an append-only file, and whether an unterminated last line was dropped.

    Only a final line without its newline may fail to parse (a crash inside the append); a
    complete line that is not a strict JSON object raises.
    """
    if not path.is_file():
        return [], False
    data = path.read_bytes()
    unterminated = not data.endswith(b"\n")
    cut_tail = False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        # The crash cut a multi-byte character: everything up to the last newline must decode.
        try:
            text = data[: data.rfind(b"\n") + 1].decode("utf-8")
        except UnicodeDecodeError:
            raise StoreError(f"{path} is not UTF-8: {exc}") from exc
        cut_tail = True
    lines = list(jsonl_lines(text))
    out: list[dict[str, Any]] = []
    for position, (number, line) in enumerate(lines):
        torn = unterminated and not cut_tail and position == len(lines) - 1
        try:
            record = strict_json_loads(line)
        except StrictJSONError as exc:
            if torn:
                return out, True
            raise StoreError(f"{path}:{number}: not strict JSON: {exc}") from exc
        if not isinstance(record, dict):
            if torn:
                return out, True
            raise StoreError(f"{path}:{number}: not an object")
        out.append(record)
    return out, cut_tail


def _same_or_older(stored: Any, given: Any) -> bool:
    """Equal records, or a stored record from an older version that lacks keys the given one adds."""
    if canonical_json(stored) == canonical_json(given):
        return True
    if isinstance(stored, dict) and isinstance(given, dict) and set(stored) < set(given):
        return all(canonical_json(stored[key]) == canonical_json(given[key]) for key in stored)
    return False


def _check_manifest_shape(directory: Path, manifest: Mapping[str, Any]) -> None:
    """The manifest's values have the shapes the store wrote, else StoreError (never a bare TypeError)."""
    task = manifest["task"]
    if not isinstance(task, dict) or not isinstance(task.get("kind"), str):
        raise StoreError(f"{directory}/run.json: task is not a task record")
    coders = manifest["coders"]
    if not isinstance(coders, list) or any(not isinstance(name, str) for name in coders):
        raise StoreError(f"{directory}/run.json: coders is not a list of names")
    draws = manifest["draws"]
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise StoreError(f"{directory}/run.json: draws is not a positive integer")
    items = manifest["items"]
    if not isinstance(items, list) or any(not isinstance(record, dict) for record in items):
        raise StoreError(f"{directory}/run.json: items is not a list of item records")
    provenance = manifest.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise StoreError(f"{directory}/run.json: provenance is not an object")
    identities = (provenance or {}).get("coder_identities")
    if identities is not None and not isinstance(identities, dict):
        raise StoreError(f"{directory}/run.json: coder_identities is not an object")


def _check_answer_record(path: Path, number: int, record: Mapping[str, Any]) -> None:
    """An answer line names its item, coder and draw, and its identity and audit are objects."""
    if "item" not in record or not isinstance(record.get("coder"), str):
        raise StoreError(f"{path}:{number}: an answer record names its item and coder")
    draw = record.get("draw")
    if isinstance(draw, bool) or not isinstance(draw, int):
        raise StoreError(f"{path}:{number}: draw is not an integer")
    for key in ("identity", "audit"):
        if record.get(key) is not None and not isinstance(record[key], dict):
            raise StoreError(f"{path}:{number}: {key} is not an object")


def _truncate_torn_tail(path: Path) -> None:
    """Cut an append-only file back to its last complete line, atomically."""
    data = path.read_bytes()
    _write_atomic(path, data[: data.rfind(b"\n") + 1])


def _write_atomic(path: Path, content: str | bytes) -> None:
    """Write a snapshot through a temporary file in the same directory, fsync, and an atomic replace.

    The temporary name is unique per writer, so two processes writing the same snapshot never share
    it; the last replace wins whole. A new file gets the mode an ordinary open would; a rewritten
    one keeps the mode it had.
    """
    data = content.encode("utf-8") if isinstance(content, str) else content
    name = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o666
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)  # the umask applies as usual
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        name.unlink(missing_ok=True)
        raise


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            return True
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def _repair_tail(path: Path) -> None:
    """Make an append-only file end with a newline: a complete last record gets one, a torn one is cut."""
    if not path.is_file() or _ends_with_newline(path):
        return
    data = path.read_bytes()
    tail = data[data.rfind(b"\n") + 1 :]
    try:
        complete = isinstance(strict_json_loads(tail.decode("utf-8")), dict)
    except (UnicodeDecodeError, StrictJSONError):
        complete = False
    if complete:
        with path.open("ab") as handle:
            handle.write(b"\n")
    else:
        _truncate_torn_tail(path)


def _append(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record as one line, after repairing a tail a failed write left behind."""
    from .io import to_json

    _repair_tail(path)
    line = json.dumps(to_json(record), ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def task_to_json(task: Task) -> dict[str, Any]:
    """A task as a JSON record."""
    return {
        "kind": task.kind,
        "codebook": dict(task.codebook),
        "prompt": task.prompt,
        "key": task.key,
        "per_answer": list(task.per_answer),
        "fields": list(task.fields),
        "aux": list(task.aux),
    }


def task_from_json(record: Mapping[str, Any]) -> Task:
    """A task from its JSON record; an empty list or object counts, a null or absent key takes the default.

    A record of the wrong shape raises ValueError naming the key.
    """

    def given(key: str, default: Any) -> Any:
        value = record.get(key)
        return default if value is None else value

    if "kind" not in record:
        raise ValueError("a task record names its kind")
    try:
        per_answer = given("per_answer", (1, 1))
        return Task(
            record["kind"],
            dict(given("codebook", {})),
            str(given("prompt", "{{TEXT}}")),
            key=str(record.get("key") or ""),
            per_answer=(int(per_answer[0]), int(per_answer[1])),
            fields=tuple(given("fields", ("cause", "effect"))),
            aux=tuple(given("aux", ("signal",))),
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"task record: {type(exc).__name__}: {exc}") from exc


class RunStore:
    """One run directory: its manifest, answers, spaces and referee decisions."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.torn_tail = False
        self._manifest: dict[str, Any] | None = None
        self._answers: dict[tuple[str, str, int], dict[str, Any]] | None = None
        self._decisions: dict[tuple[str, str], dict[str, Any]] | None = None

    # -- manifest ------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """Whether the directory holds a run manifest."""
        return (self.dir / "run.json").is_file()

    def manifest(self) -> dict[str, Any]:
        """The run manifest, run.json."""
        if self._manifest is None:
            if not self.exists:
                raise StoreError(f"{self.dir} holds no run (no run.json)")
            manifest = _read_json(self.dir / "run.json")
            if not isinstance(manifest, dict) or manifest.get("version") != STORE_VERSION:
                raise StoreError(f"{self.dir}/run.json is not a {STORE_VERSION} manifest")
            missing = [key for key in ("task", "coders", "draws", "items") if key not in manifest]
            if missing:
                raise StoreError(f"{self.dir}/run.json lacks {', '.join(missing)}")
            _check_manifest_shape(self.dir, manifest)
            self._manifest = manifest
        return self._manifest

    def create(
        self,
        task: Task,
        coders: Sequence[str],
        draws: int,
        items: Sequence[Item],
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Write the manifest, or check that the existing one describes the same run.

        ``provenance`` (the referee's configuration, the coders' identities) must agree with what
        an existing manifest records; a key the manifest lacks is added to it.
        """
        from .io import item_to_json

        provenance = dict(provenance or {})
        canonical_json(provenance)  # a record that is not JSON fails here, before anything is written
        wanted = {
            "version": STORE_VERSION,
            "task": task_to_json(task),
            "coders": list(coders),
            "draws": draws,
            "items": [item_to_json(item, task) for item in items],
        }
        if self.exists:
            found = {key: self.manifest()[key] for key in wanted}
            if canonical_json(found) != canonical_json(wanted):
                raise StoreError(f"{self.dir} holds another run (task, coders, draws or items differ)")
            self.merge_provenance(provenance)
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "spaces").mkdir(exist_ok=True)
        from . import __version__

        manifest = {
            **wanted,
            "provenance": provenance,
            "created": datetime.now(UTC).isoformat(),
            "codercommittee": __version__,
        }
        self._write_manifest(manifest)

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        text = json.dumps(manifest, ensure_ascii=False, indent=1, allow_nan=False)
        _write_atomic(self.dir / "run.json", text + "\n")
        self._manifest = manifest

    def merge_provenance(self, given: Mapping[str, Any], *, write: bool = True) -> None:
        """Add provenance keys the manifest lacks; a key it holds must agree, else StoreError.

        A stored record that lacks keys the given one adds (written by an older version) agrees
        when its own keys do, and is replaced by the fuller one. A seat the manifest never recorded,
        or recorded without its model, is checked against the stored answers' own model first.
        With ``write`` false only the comparison runs: nothing is written.
        """
        manifest = self.manifest()
        existing = dict(manifest.get("provenance") or {})
        merged = dict(existing)
        for key, value in given.items():
            if key == "coder_identities" and isinstance(value, dict):
                names = dict(existing.get(key) or {})
                for name, identity in value.items():
                    was = names.get(name)
                    if was is None or not isinstance(was, dict) or "model" not in was:
                        self._check_seat_against_answers(name, identity)
                    if was is not None and not _same_or_older(was, identity):
                        raise StoreError(
                            f"{self.dir} holds a run whose coder {name!r} was {was}, not {identity}; "
                            "a seat keeps its model across a resume"
                        )
                    names[name] = identity
                merged[key] = names
                continue
            if key in existing and not _same_or_older(existing[key], value):
                what = key.replace("_", " ")
                raise StoreError(
                    f"{self.dir} holds a run made with another {what}: stored {existing[key]}, given {value}"
                )
            merged[key] = value
        if write and merged != existing:
            self._write_manifest({**manifest, "provenance": merged})

    def _check_seat_against_answers(self, name: str, identity: Mapping[str, Any]) -> None:
        """A seat without a recorded identity must not already hold answers from another model."""
        pairs = (("model", "requested_model"), ("base_url", "base_url"))
        for record in self._load_answers().values():
            if record.get("coder") != name:
                continue
            recorded = record.get("identity") or {}
            for own, theirs in pairs:
                wanted, found = identity.get(own), recorded.get(theirs)
                if isinstance(wanted, str) and isinstance(found, str) and found != wanted:
                    raise StoreError(
                        f"{self.dir} holds answers of coder {name!r} from {own} {found!r}, not {wanted!r}; "
                        "a seat keeps its model across a resume"
                    )

    def provenance(self) -> dict[str, Any]:
        """The stored provenance: the referee's configuration and the coders' identities, when recorded."""
        return dict(self.manifest().get("provenance") or {})

    def task(self) -> Task:
        """The stored task."""
        return task_from_json(self.manifest()["task"])

    def coders(self) -> tuple[str, ...]:
        """The stored coders in seat order."""
        return tuple(self.manifest()["coders"])

    def draws(self) -> int:
        """The stored draws per coder and item."""
        return int(self.manifest()["draws"])

    def items(self) -> list[Item]:
        """The stored items with their references."""
        from .io import item_from_json

        task = self.task()
        return [item_from_json(record, task) for record in self.manifest()["items"]]

    def index_of(self, item: ItemId) -> int:
        """The position of an item in the stored run."""
        key = _canonical_id(item)
        for index, record in enumerate(self.manifest()["items"]):
            if _canonical_id(record["id"]) == key:
                return index
        raise StoreError(f"item {item!r} is not in the stored run")

    # -- answers -------------------------------------------------------------

    def _records(self, name: str) -> list[dict[str, Any]]:
        path = self.dir / name
        records, torn = _read_jsonl(path)
        if torn:
            _truncate_torn_tail(path)
            self.torn_tail = True
        return records

    def _load_answers(self) -> dict[tuple[str, str, int], dict[str, Any]]:
        if self._answers is None:
            self._answers = {}
            for number, record in enumerate(self._records("answers.jsonl"), 1):
                _check_answer_record(self.dir / "answers.jsonl", number, record)
                self._answers[(_canonical_id(record["item"]), record["coder"], record["draw"])] = record
        return self._answers

    def answer(self, item: ItemId, coder: str, draw: int) -> dict[str, Any] | None:
        """The last stored record of one answer slot, or None."""
        return self._load_answers().get((_canonical_id(item), coder, draw))

    def put_answer(self, answer: Answer, task: Task) -> None:
        """Append an answer to the store."""
        record = answer_record(answer, task)
        try:
            _append(self.dir / "answers.jsonl", record)
        except BaseException:
            self._answers = None  # whether the bytes landed is unknown: the next access reads the file
            raise
        self._load_answers()[(_canonical_id(answer.item), answer.coder, answer.draw)] = record

    def answered(self) -> int:
        """The number of answer slots with a stored record."""
        return len(self._load_answers())

    # -- spaces --------------------------------------------------------------

    def _space_path(self, item: ItemId) -> Path:
        return self.dir / "spaces" / f"{self.index_of(item):06d}.json"

    def space(self, item: ItemId) -> Space | None:
        """The stored space of an item, or None."""
        path = self._space_path(item)
        if not path.is_file():
            return None
        payload = _read_json(path)
        if _canonical_id(payload.get("item")) != _canonical_id(item):
            raise StoreError(f"{path} holds another item's vectors")
        vectors = payload.get("vectors")
        if not isinstance(vectors, dict):
            raise StoreError(f"{path} holds no vectors")
        entries: dict[str, list[float]] = {}
        for sha, entry in vectors.items():
            if not isinstance(entry, dict) or sha != text_sha256(str(entry.get("text"))):
                raise StoreError(f"{path}: entry {sha[:12]} is not keyed by its text")
            entries[str(entry["text"])] = list(entry["vector"])
        return Space(entries)

    def put_space(self, item: ItemId, space: Space) -> None:
        """Write an item's space."""
        payload = {
            "item": item,
            "dimension": space.dimension,
            "vectors": {text_sha256(text): {"text": text, "vector": space.vector(text)} for text in space.texts},
        }
        (self.dir / "spaces").mkdir(exist_ok=True)
        _write_atomic(self._space_path(item), json.dumps(payload, ensure_ascii=False) + "\n")

    # -- decisions -----------------------------------------------------------

    def _load_decisions(self) -> dict[tuple[str, str], dict[str, Any]]:
        if self._decisions is None:
            self._decisions = {}
            for number, record in enumerate(self._records("decisions.jsonl"), 1):
                if "item" not in record or not isinstance(record.get("label"), str):
                    raise StoreError(f"{self.dir / 'decisions.jsonl'}:{number}: a decision names its item and label")
                self._decisions[(_canonical_id(record["item"]), record["label"])] = record
        return self._decisions

    @property
    def has_decisions(self) -> bool:
        """Whether any referee decision is stored."""
        return bool(self._load_decisions())

    def decision(self, item: ItemId, label: str) -> dict[str, Any] | None:
        """The stored decision of one realization, or None."""
        return self._load_decisions().get((_canonical_id(item), label))

    def decisions(self) -> list[dict[str, Any]]:
        """Every stored decision record, the last line of a realization winning, in file order."""
        return list(self._load_decisions().values())

    def put_decision(
        self, item: ItemId, label: str, prematch_sha256: str, decisions: Mapping[str, bool], identity: Mapping[str, Any]
    ) -> None:
        """Append a referee decision bound to its pre-match hash; decisions are booleans."""
        if any(not isinstance(value, bool) for value in decisions.values()):
            raise StoreError(f"decisions of item {item!r}, {label} are not booleans")
        record = {
            "item": item,
            "label": label,
            "prematch_sha256": prematch_sha256,
            "decisions": dict(decisions),
            "identity": dict(identity),
        }
        try:
            _append(self.dir / "decisions.jsonl", record)
        except BaseException:
            self._decisions = None
            raise
        self._load_decisions()[(_canonical_id(item), label)] = record


def reply_shape(value: Any, task: Task) -> Any:
    """A value in the shape of a coder's reply, the object the task's schema describes."""
    from .replay import value_as_stored

    if task.kind == "tuple" and value is not None:
        return {task.key: True, **{name: value.get(name, "") for name in task.names}}
    if task.kind == "tuples":
        return {task.key: [{name: relation.get(name, "") for name in task.names} for relation in value]}
    return value_as_stored(value, task)


def answer_record(answer: Answer, task: Task) -> dict[str, Any]:
    """An answer as a stored line.

    An ``ok`` answer without a raw reply (a custom coder that built the Answer itself) is stored
    with the canonical JSON of its value as the raw text, marked ``raw_synthesized`` in the audit,
    so the load re-validates it like any other.
    """
    from .replay import value_as_stored

    raw = answer.raw
    audit = dict(answer.audit)
    if answer.ok and raw is None:
        raw = canonical_json(reply_shape(answer.value, task))
        audit["raw_synthesized"] = True
    return {
        "item": answer.item,
        "coder": answer.coder,
        "draw": answer.draw,
        "status": answer.status,
        "reason": answer.reason,
        "raw": raw,
        "value": value_as_stored(answer.value, task) if answer.ok else None,
        "identity": dict(answer.identity),
        "audit": audit,
    }


class StoreCoder:
    """Serves one coder's stored answers; a stored ``ok`` reply is re-validated from its raw text."""

    def __init__(self, store: RunStore, coder: str) -> None:
        self.store = store
        self.name = coder

    def describe(self) -> dict[str, Any] | None:
        """The identity the stored run recorded for this seat, when it did; else None."""
        identities = self.store.provenance().get("coder_identities") if self.store.exists else None
        if isinstance(identities, dict) and isinstance(identities.get(self.name), dict):
            return dict(identities[self.name])
        return None

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        """The stored answer of this coder's draw on the item, re-validated."""
        record = self.store.answer(item.id, self.name, draw)
        if record is None:
            raise StoreError(f"no stored answer for item {item.id!r}, coder {self.name}, draw {draw}")
        return answer_from_record(record, item, task)


def answer_from_record(record: Mapping[str, Any], item: Item, task: Task) -> Answer:
    """An Answer from a stored line; the raw reply decides the value, the stored value never does."""
    status = str(record.get("status"))
    identity = dict(record.get("identity") or {})
    audit = dict(record.get("audit") or {})
    raw = record.get("raw")
    coder = str(record["coder"])
    draw = int(record["draw"])
    if status in ("refused", "invalid", "missing"):
        final: AnswerStatus = "refused" if status == "refused" else "invalid" if status == "invalid" else "missing"
        reason = record.get("reason")
        return Answer(item.id, coder, draw, None, final, reason=reason, raw=raw, identity=identity, audit=audit)
    if status != "ok":
        return Answer(
            item.id, coder, draw, None, "invalid", reason=f"unknown stored status {status!r}", identity=identity
        )
    if not isinstance(raw, str):
        return Answer(
            item.id, coder, draw, None, "invalid", reason="stored answer lacks its raw reply", identity=identity
        )
    try:
        parsed = strict_json_loads(raw)
    except StrictJSONError:
        return Answer(
            item.id, coder, draw, None, "invalid", reason="strict_json_parse_failed", raw=raw, identity=identity
        )
    validation = validate(task, parsed, item.text)
    if not validation.ok:
        return Answer(item.id, coder, draw, None, "invalid", reason=validation.reason, raw=raw, identity=identity)
    return Answer(
        item.id, coder, draw, validation.value, "ok", raw=raw, identity=identity, audit={**audit, **validation.audit}
    )


class StoreEmbedder:
    """Serves one item's stored vectors; the stored texts must be the spans asked for."""

    def __init__(self, store: RunStore) -> None:
        self.store = store

    def space(self, item: Item, texts: Sequence[str]) -> Space:
        """The stored vectors of the item; the stored texts must be the spans asked for."""
        space = self.store.space(item.id)
        if space is None:
            if not texts:
                return Space({})
            raise StoreError(f"no stored vectors for item {item.id!r}")
        if set(space.texts) != set(texts):
            raise StoreError(f"stored vectors of item {item.id!r} differ from the spans asked for")
        return space


def _recorded(record: Mapping[str, Any] | None, prematch: Prematch) -> dict[str, bool] | None:
    """The stored decisions when the record binds to this pre-match and covers its pairs with booleans."""
    if record is None or record.get("prematch_sha256") != prematch.sha256:
        return None
    decisions = record.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != set(prematch.candidate_pairs):
        return None
    if any(not isinstance(value, bool) for value in decisions.values()):
        return None
    return dict(decisions)


class LoggedReferee:
    """Serves the store's recorded decision for a pre-match, else decides with ``inner`` and records it.

    A record counts when it binds to the pre-match's sha256 and covers its candidate pairs; a
    resumed or repeated run therefore never asks the referee twice for the same question.
    """

    def __init__(self, inner: Referee, store: RunStore) -> None:
        self.inner = inner
        self.store = store
        self.accept_at = inner.accept_at
        record = describe_referee(inner)
        existing = store.provenance().get("referee") if store.exists else None
        if record is not None and isinstance(existing, dict) and not _same_or_older(existing, record):
            raise StoreError(f"{store.dir} holds a run made with another referee: stored {existing}, given {record}")

    def describe(self) -> dict[str, Any] | None:
        """The inner referee's record."""
        return describe_referee(self.inner)

    def decide(self, prematch: Prematch) -> Mapping[str, bool]:
        """A recorded decision for this pre-match, else the inner referee's, recorded."""
        item, label = prematch.item, prematch.label
        if not prematch.candidate_pairs or item is None or label is None:
            return self.inner.decide(prematch)
        recorded = _recorded(self.store.decision(item, label), prematch)
        if recorded is not None:
            return recorded
        self.store.dir.mkdir(parents=True, exist_ok=True)
        decisions = self.inner.decide(prematch)
        identity = dict(getattr(self.inner, "last_identity", {}) or {})
        self.store.put_decision(item, label, prematch.sha256, decisions, identity)
        return decisions


class StoreReferee:
    """Serves the store's recorded decisions; raises when none exists or the pre-match differs."""

    def __init__(self, store: RunStore, *, accept_at: float = 0.80) -> None:
        self.store = store
        self.accept_at = accept_at

    def describe(self) -> dict[str, Any] | None:
        """The record of the referee that made the stored decisions, when the manifest holds one; else None."""
        record = self.store.provenance().get("referee") if self.store.exists else None
        if isinstance(record, dict) and record.get("accept_at") == self.accept_at:
            return dict(record)
        return None

    def decide(self, prematch: Prematch) -> dict[str, bool]:
        """The stored decisions of this pre-match, bound by its hash."""
        if prematch.high != self.accept_at:
            raise ValueError("the pre-match was banded at another threshold than the stored decisions")
        if not prematch.candidate_pairs:
            return {}
        if prematch.item is None or prematch.label is None:
            raise StoreError("serving stored decisions needs the pre-match's item and label")
        record = self.store.decision(prematch.item, prematch.label)
        where = f"item {prematch.item!r}, {prematch.label}"
        if record is None:
            raise StoreError(f"no stored decision for {where}")
        if record.get("prematch_sha256") != prematch.sha256:
            raise StoreError(f"the stored pre-match of {where} differs from the recomputed one")
        decisions = record.get("decisions")
        if not isinstance(decisions, dict) or set(decisions) != set(prematch.candidate_pairs):
            raise StoreError(f"stored decisions of {where} do not cover the candidate pairs exactly")
        if any(not isinstance(value, bool) for value in decisions.values()):
            raise StoreError(f"stored decisions of {where} are not JSON booleans")
        return dict(decisions)


def _same_answer(stored: Mapping[str, Any], answer: Answer, task: Task) -> bool:
    """Same status and raw reply; the stored value is derived (and pre-deduplication) so it does not count."""
    fresh = answer_record(answer, task)
    return bool(stored.get("status") == fresh["status"] and stored.get("raw") == fresh["raw"])


def _same_space(stored: Space, space: Space) -> bool:
    if set(stored.texts) != set(space.texts):
        return False
    return all(stored.vector(text) == space.vector(text) for text in space.texts)


def _source_records(referee: Referee | None, destination: RunStore) -> list[dict[str, Any]]:
    """The decision records a run's referee serves, to copy into ``destination`` (none for its own store)."""
    if isinstance(referee, StoreReferee | LoggedReferee):
        if referee.store.dir.resolve() == destination.dir.resolve():
            return []
        return referee.store.decisions()
    if isinstance(referee, ReplayReferee):
        return referee.records()
    return []


def _same_decision(stored: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    keys = ("prematch_sha256", "decisions")
    return canonical_json({k: stored.get(k) for k in keys}) == canonical_json({k: record.get(k) for k in keys})


def save_run(run: Run, directory: str | Path) -> RunStore:
    """Write a run in memory into a store: answers, spaces, provenance and recorded decisions.

    Saving into a directory that already holds the same run is idempotent (an identical answer is
    not rewritten: the first record's evidence, attempt counts and response ids, stands); one that
    holds different answers, vectors or decisions for the same slots is refused, never merged.
    Every check runs before the first write, then the seats and the referee are bound in the
    manifest, then the answers follow, so an interrupted save never leaves answers without
    their seats' identities. The recorded decisions a loaded or replayed run's referee serves
    are copied with it. A ``tuples`` run whose referee is an LLM referee not wrapped in
    :class:`LoggedReferee` is refused: its decisions were never recorded and could not be saved.
    """
    if run.task.kind == "tuples" and isinstance(run.referee, LLMReferee):
        raise StoreError("an LLM referee's decisions are saved only through LoggedReferee; wrap it before collecting")
    provenance = dict(run.provenance)
    record = describe_referee(run.referee)  # the referee the run scores with, whatever its provenance says
    if record is not None:
        provenance["referee"] = record
    canonical_json(provenance)  # a record that is not JSON (a set, NaN) fails here, before any write
    store = RunStore(directory)
    store.create(run.task, run.coders, run.draws, run.items)
    store.merge_provenance(provenance, write=False)  # a conflict raises here, before any write
    answers_to_write: list[Answer] = []
    spaces_to_write: list[tuple[ItemId, Space]] = []
    for item in run.items:
        for coder in run.coders:
            for draw in range(1, run.draws + 1):
                answer = run.answer(item.id, coder, draw)
                stored = store.answer(item.id, coder, draw)
                if stored is None:
                    answers_to_write.append(answer)
                elif not _same_answer(stored, answer, run.task):
                    raise StoreError(
                        f"{store.dir} already holds a different answer for item {item.id!r}, coder {coder}, "
                        f"draw {draw}; a run is saved into an empty directory or over its own copy"
                    )
        space = run.spaces.get(item.id)
        if space is not None:
            existing = store.space(item.id)
            if existing is None:
                spaces_to_write.append((item.id, space))
            elif not _same_space(existing, space):
                raise StoreError(f"{store.dir} already holds different vectors for item {item.id!r}")
    decisions_to_write: list[dict[str, Any]] = []
    for record in _source_records(run.referee, store):
        item_id, label = record["item"], str(record["label"])
        existing_record = store.decision(item_id, label)
        if existing_record is None:
            decisions_to_write.append(record)
        elif not _same_decision(existing_record, record):
            raise StoreError(f"{store.dir} already holds a different decision for item {item_id!r}, {label}")
    store.merge_provenance(provenance)  # every check passed: bind the seats and the referee before the answers
    for answer in answers_to_write:
        store.put_answer(answer, run.task)
    for item_id, space in spaces_to_write:
        store.put_space(item_id, space)
    for record in decisions_to_write:
        store.put_decision(
            record["item"],
            str(record["label"]),
            str(record["prematch_sha256"]),
            record["decisions"],
            record.get("identity") or {},
        )
    return store


def stored_referee(store: RunStore) -> Referee | None:
    """The referee a stored run scores with: the band rule it recorded, else its recorded decisions.

    A manifest without a referee record (written before the record existed) falls back to the
    recorded decisions when there are any, else to no referee at all. Pass a referee to
    ``Run.load`` to score a run under another one.
    """
    record = store.provenance().get("referee")
    if record is None:
        return StoreReferee(store) if store.has_decisions else None
    if not isinstance(record, dict):
        raise StoreError(f"{store.dir}/run.json records a referee of no known shape: {record!r}")
    accept_at = record.get("accept_at", 0.80)
    if isinstance(accept_at, bool) or not isinstance(accept_at, int | float) or not 0.0 < accept_at <= 1.0:
        raise StoreError(f"{store.dir}/run.json records a referee threshold outside (0, 1]: {accept_at!r}")
    if record.get("kind") == "band":
        return BandRule(float(accept_at))
    # An LLM or replayed referee, or any custom one: its recorded decisions are what the run scored
    # with; a realization it never decided is named by the StoreReferee when scoring asks for it.
    return StoreReferee(store, accept_at=float(accept_at))


def load_run(directory: str | Path, referee: Referee | None = None) -> Run:
    """Rebuild a stored run through collect: answers re-validated, spaces served, decisions replayed."""
    from .collect import collect
    from .data import SPAN_KINDS

    store = RunStore(directory)
    task = store.task()
    coders: list[Coder] = [StoreCoder(store, name) for name in store.coders()]
    embedder: Embedder | None = StoreEmbedder(store) if task.kind in SPAN_KINDS else None
    chosen = referee if referee is not None else stored_referee(store)
    run = collect(store.items(), task, coders, draws=store.draws(), attempts=1, embedder=embedder, referee=chosen)
    return replace(run, referee=chosen, provenance={**store.provenance(), **run.provenance})


__all__ = [
    "FINAL_STATUSES",
    "STORE_VERSION",
    "LoggedReferee",
    "RunStore",
    "StoreCoder",
    "StoreEmbedder",
    "StoreError",
    "StoreReferee",
    "answer_from_record",
    "answer_record",
    "load_run",
    "reply_shape",
    "save_run",
    "stored_referee",
    "task_from_json",
    "task_to_json",
]
