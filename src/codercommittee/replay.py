"""Replay of a stored run: answers, vectors and referee decisions served from its checkpoints.

A run directory holds ``per_item.jsonl`` (one row per item, in run order; the item index is
its position), ``summary.json``, and under ``checkpoints/`` the response records
``responses/{index:06d}_{slot:02d}_r{draw}_{coder}.json``, the embedding records
``embeddings/{index:06d}.json`` and the referee records ``agent/{index:06d}_{label}.json``
(``panel:1`` written as ``panel_1``). The replay adapters raise on absent evidence and are
never retried; a stored refusal is final. Every identity check is theirs: a failed check makes
the answer invalid.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .collect import collect
from .data import SPAN_KINDS, Answer, Item, ItemId, Run, Task
from .protocols import Referee
from .similarity import Space, text_sha256
from .strictjson import StrictJSONError, canonical_json, jsonl_lines, strict_json_loads
from .validation import validate

EMBEDDING_SCHEMA_VERSION = "study-embedding-checkpoint-v1"
MAX_ATTEMPTS = 3
REFUSAL_REASONS = frozenset({"provider_refusal", "provider_incomplete_finish:refusal"})


class ReplayError(RuntimeError):
    """Evidence is absent or does not describe what was asked for."""


def is_refusal(reason: object) -> bool:
    """A provider declined on policy grounds: asking again could not change the answer."""
    return reason in REFUSAL_REASONS or (isinstance(reason, str) and reason.endswith(":content_filter"))


def canonical_id(item: object) -> str:
    """The canonical JSON of an item id, the key stored rows are matched by."""
    return canonical_json(item)


def _read_json(path: Path) -> Any:
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"))
    except StrictJSONError as exc:
        raise ReplayError(f"{path} is not strict JSON: {exc}") from exc


class StoredRun:
    """A run directory: its rows, coders, identities and checkpoint files."""

    def __init__(self, run_dir: str | Path) -> None:
        self.dir = Path(run_dir)
        rows_path = self.dir / "per_item.jsonl"
        summary_path = self.dir / "summary.json"
        if not rows_path.is_file() or not summary_path.is_file():
            raise ReplayError(f"{self.dir} is not a run directory (per_item.jsonl and summary.json required)")
        self.summary: dict[str, Any] = _read_json(summary_path)
        self.rows: list[dict[str, Any]] = [
            strict_json_loads(line) for _, line in jsonl_lines(rows_path.read_text(encoding="utf-8"))
        ]
        if not self.rows:
            raise ReplayError(f"{rows_path} holds no rows")
        self.ids: list[Any] = [row["item_id"] for row in self.rows]
        self._index = {canonical_id(item_id): index for index, item_id in enumerate(self.ids)}
        if len(self._index) != len(self.ids):
            raise ReplayError("stored item ids are not unique")
        self.coders: list[str] = list(self.summary["model_ids"])
        self.requested: dict[str, str] = {m["model_id"]: str(m["provider_model"]) for m in self.summary["models"]}
        pin_path = self.dir / "canary_pin.json"
        pinned = _read_json(pin_path).get("models", {}) if pin_path.is_file() else {}
        self.returned: dict[str, str] = {coder: str(pinned.get(coder, self.requested[coder])) for coder in self.coders}
        first = self.rows[0]
        self.cache_namespace: str = str(first["cache_namespace"])
        self.prompt_version: str = str(first["prompt_version"])
        self.draws: int = int(first["responses_per_model"])

    def index_of(self, item: ItemId) -> int:
        """The position of an item in the stored run."""
        key = canonical_id(item)
        if key not in self._index:
            raise ReplayError(f"item {item!r} is not in the stored run")
        return self._index[key]

    def row(self, index: int) -> dict[str, Any]:
        """The stored row at an index."""
        return self.rows[index]

    def response(self, index: int, coder: str, draw: int) -> dict[str, Any]:
        """The stored response checkpoint of one coder's draw on one item."""
        slot = self.coders.index(coder)
        path = self.dir / "checkpoints" / "responses" / f"{index:06d}_{slot:02d}_r{draw}_{coder}.json"
        if not path.is_file():
            raise ReplayError(
                f"no stored answer for item {self.ids[index]!r}, coder {coder}, draw {draw} ({path.name})"
            )
        return _read_json(path)  # type: ignore[no-any-return]

    def embedding(self, index: int) -> dict[str, Any] | None:
        """The stored embedding checkpoint of an item, or None."""
        path = self.dir / "checkpoints" / "embeddings" / f"{index:06d}.json"
        return _read_json(path) if path.is_file() else None

    def agent(self, index: int, label: str) -> dict[str, Any] | None:
        """The stored referee checkpoint of an item's realization, or None."""
        path = self.dir / "checkpoints" / "agent" / f"{index:06d}_{label.replace(':', '_')}.json"
        return _read_json(path) if path.is_file() else None


def _fingerprint_failure(record: dict[str, Any], binding: dict[str, Any]) -> str | None:
    fingerprint = record.get("request_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        return "request_fingerprint_missing"
    if binding.get("request_fingerprint") not in (None, fingerprint):
        return "request_fingerprint_mismatch"
    envelope = binding.get("request_envelope")
    if envelope is not None:
        recomputed = hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()
        if recomputed != fingerprint:
            return "request_fingerprint_mismatch"
    return None


class ReplayCoder:
    """Serves one coder's stored answers, checking every identity the record carries."""

    def __init__(self, stored: StoredRun | str | Path, coder: str) -> None:
        self.stored = stored if isinstance(stored, StoredRun) else StoredRun(stored)
        if coder not in self.stored.coders:
            raise ReplayError(f"coder {coder!r} is not one of the stored run's coders {self.stored.coders}")
        self.name = coder

    def describe(self) -> dict[str, Any]:
        """The seat's identity in the stored run: the requested and the pinned returned model."""
        return {
            "kind": "replay",
            "model": self.stored.requested[self.name],
            "returned": self.stored.returned[self.name],
        }

    def _identity_failure(self, record: dict[str, Any], binding: dict[str, Any]) -> str | None:
        stored = self.stored
        if record.get("requested_model_id") != stored.requested[self.name]:
            return "requested_model_mismatch"
        if record.get("cache_namespace") != stored.cache_namespace:
            return "cache_namespace_mismatch"
        if record.get("prompt_version") != stored.prompt_version:
            return "prompt_version_mismatch"
        attempts = record.get("attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= MAX_ATTEMPTS:
            return "invalid_transport_attempt_count"
        returned = record.get("returned_model_id")
        if not isinstance(returned, str) or not returned.strip():
            return "missing_returned_model_id"
        if returned != stored.returned[self.name]:
            return "returned_model_mismatch"
        return _fingerprint_failure(record, binding)

    def _content(self, record: dict[str, Any], task: Task, text: str) -> tuple[Any, str | None]:
        """The validated value, or the reason it is invalid; stored parsed values are never trusted."""
        raw = record.get("raw_response")
        if not isinstance(raw, str):
            return None, "raw_response_missing"
        try:
            parsed = strict_json_loads(raw)
        except StrictJSONError:
            return None, "strict_json_parse_failed"
        if "parsed_response" in record and record["parsed_response"] != parsed:
            return None, "parsed_response_does_not_match_raw"
        validation = validate(task, parsed, text)
        return validation, validation.reason

    def _discard_failure(self, record: dict[str, Any], task: Task, text: str) -> str | None:
        """The reason the content-retry discards are unacceptable, or None.

        Discards are paid replies the runner rejected; each must share the final record's request
        identity, fit inside the attempt budget, and fail validation when replayed.
        """
        if "content_retry_discarded" not in record:
            return None
        discards = record["content_retry_discarded"]
        if not isinstance(discards, list) or not discards or any(not isinstance(d, dict) for d in discards):
            return "content_retry_discards_malformed"
        if len(discards) > MAX_ATTEMPTS - 1:
            return "content_retry_discards_exceed_budget"
        discarded_attempts = 0
        for entry in discards:
            attempts = entry.get("attempts")
            if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
                return "content_retry_discard_attempts_invalid"
            discarded_attempts += attempts
            if (
                entry.get("requested_model_id") != self.stored.requested[self.name]
                or entry.get("cache_namespace") != self.stored.cache_namespace
                or entry.get("prompt_version") != self.stored.prompt_version
                or entry.get("request_fingerprint") != record.get("request_fingerprint")
            ):
                return "content_retry_discard_identity_mismatch"
            probe = {key: value for key, value in entry.items() if key != "parsed_response"}
            if not probe.get("provider_error"):
                _, reason = self._content(probe, task, text)
                if reason is None:
                    return "content_retry_discarded_a_valid_response"
        final_attempts = record.get("attempts")
        if not isinstance(final_attempts, int) or not discarded_attempts < final_attempts <= MAX_ATTEMPTS:
            return "content_retry_attempt_accounting_mismatch"
        return None

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        """The stored answer, re-validated, with every identity check."""
        stored = self.stored
        index = stored.index_of(item.id)
        checkpoint = stored.response(index, self.name, draw)
        if (
            canonical_id(checkpoint.get("item_id")) != canonical_id(stored.ids[index])
            or checkpoint.get("model_id") != self.name
            or checkpoint.get("repeat_index") != draw
        ):
            raise ReplayError(f"stored answer {index}/{self.name}/{draw} names another item, coder or draw")
        record = checkpoint.get("response")
        if not isinstance(record, dict):
            raise ReplayError(f"stored answer {index}/{self.name}/{draw} carries no response record")
        binding = checkpoint.get("execution_binding") or {}
        identity = {
            "coder": self.name,
            "requested_model": record.get("requested_model_id"),
            "returned_model": record.get("returned_model_id"),
            "request_fingerprint": record.get("request_fingerprint"),
            "attempts": record.get("attempts"),
            "provider_response_id": record.get("provider_response_id"),
        }

        def failed(status: str, reason: str) -> Answer:
            return Answer(item.id, self.name, draw, None, status, reason=reason, identity=identity)  # type: ignore[arg-type]

        reason = self._identity_failure(record, binding)
        if reason is not None:
            return failed("invalid", reason)
        error = record.get("provider_error")
        if error:
            return failed("refused" if is_refusal(error) else "invalid", str(error))
        validation, reason = self._content(record, task, item.text)
        if reason is not None:
            return failed("invalid", reason)
        reason = self._discard_failure(record, task, item.text)
        if reason is not None:
            return failed("invalid", reason)
        return Answer(
            item.id,
            self.name,
            draw,
            validation.value,
            "ok",
            raw=record["raw_response"],
            identity=identity,
            audit=validation.audit,
        )


class ReplayEmbedder:
    """Serves one item's stored vectors; the stored set must equal the texts asked for."""

    def __init__(self, stored: StoredRun | str | Path) -> None:
        self.stored = stored if isinstance(stored, StoredRun) else StoredRun(stored)

    def space(self, item: Item, texts: Sequence[str]) -> Space:
        """The stored vectors of the item; the stored set must equal the texts asked for."""
        index = self.stored.index_of(item.id)
        payload = self.stored.embedding(index)
        required = {text_sha256(text): text for text in texts}
        if payload is None:
            if not required:
                return Space({})
            raise ReplayError(f"no stored vectors for item {item.id!r}")
        if payload.get("schema_version") != EMBEDDING_SCHEMA_VERSION:
            raise ReplayError(f"embedding record of item {item.id!r} has another schema version")
        if canonical_id(payload.get("item_id")) != canonical_id(self.stored.ids[index]):
            raise ReplayError(f"embedding record {index} names another item")
        vectors = payload.get("vectors")
        dimensions = payload.get("dimensions")
        if not isinstance(vectors, dict) or not isinstance(dimensions, int):
            raise ReplayError(f"embedding record of item {item.id!r} lacks vectors or dimensions")
        entries: dict[str, list[float]] = {}
        for key, entry in vectors.items():
            text = entry.get("text") if isinstance(entry, dict) else None
            vector = entry.get("vector") if isinstance(entry, dict) else None
            if not isinstance(text, str) or key != text_sha256(text) or not isinstance(vector, list):
                raise ReplayError(f"embedding record of item {item.id!r}: entry {key[:12]} is not keyed by its text")
            if len(vector) != dimensions:
                raise ReplayError(f"embedding record of item {item.id!r}: a vector has not {dimensions} entries")
            entries[text] = vector
        for request in payload.get("requests") or []:
            shas = request.get("input_sha256s") if isinstance(request, dict) else None
            if not isinstance(shas, list) or any(sha not in vectors for sha in shas):
                raise ReplayError(f"embedding record of item {item.id!r} names vectors it does not hold")
        stored_set = set(vectors)
        if stored_set != set(required):
            missing = [required[sha][:40] for sha in sorted(set(required) - stored_set)][:5]
            extra = [vectors[sha]["text"][:40] for sha in sorted(stored_set - set(required))][:5]
            raise ReplayError(
                f"stored vectors of item {item.id!r} differ from the spans asked for: missing={missing} extra={extra}"
            )
        return Space(entries)


def load_run(run_dir: str | Path, task: Task, items: Sequence[Item], *, referee: Referee | None = None) -> Run:
    """The stored run as a Run: every answer, vector and decision served from the checkpoints.

    ``items`` carry the texts and references the stored rows do not hold; their ids and order
    must be the stored run's. For the ``tuples`` kind the stored referee decisions are the
    run's referee unless another is given (``BandRule()`` recomputes the vote under the rule).
    """
    stored = StoredRun(run_dir)
    given = [canonical_id(item.id) for item in items]
    expected = [canonical_id(item_id) for item_id in stored.ids]
    if given != expected:
        raise ReplayError("the items differ from the stored run (ids or order)")
    coders = [ReplayCoder(stored, name) for name in stored.coders]
    embedder = ReplayEmbedder(stored) if task.kind in SPAN_KINDS else None
    if referee is None and task.kind == "tuples":
        from .referee import ReplayReferee

        referee = ReplayReferee(stored)
    return collect(items, task, coders, draws=stored.draws, attempts=1, embedder=embedder, referee=referee)


def value_as_stored(value: Any, task: Task) -> Any:
    """A value in the shape the study's rows store (``validated_prediction`` / ``prediction``)."""
    if task.kind == "label":
        return {task.key: value}
    if task.kind == "labels":
        return {task.key: sorted(value)}
    if task.kind == "tuple":
        if value is None:
            return {task.key: False, **dict.fromkeys(task.names, "")}
        return {task.key: True, **value.as_dict()}
    return [relation.as_dict() for relation in value]


def stored_value(answer: Answer, task: Task) -> Any:
    """An answer's value in the stored row's shape, or None when the answer is not ok."""
    return value_as_stored(answer.value, task) if answer.ok else None


__all__ = [
    "EMBEDDING_SCHEMA_VERSION",
    "ReplayCoder",
    "ReplayEmbedder",
    "ReplayError",
    "StoredRun",
    "canonical_id",
    "is_refusal",
    "load_run",
    "stored_value",
    "value_as_stored",
]
