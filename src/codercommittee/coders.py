"""Coders: the mock coder, and the coder that answers through a model adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from .data import Answer, Item, Task
from .protocols import Model
from .providers.base import public_url, scrub_credentials
from .strictjson import StrictJSONError, strict_json_loads
from .validation import validate

SYSTEM_PROMPT = "You are a careful research annotator. Return only schema-valid JSON."

Script = Callable[[Item, Task, int], Any]


def _seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12], 16)


def default_script(name: str, seed: int = 0) -> Script:
    """A deterministic, schema-valid answer for any task, varying by coder, item, draw and seed.

    Labels come from the coder's hash of the item; spans are word windows copied from the text,
    so a span answer is always verbatim. Draws differ, so the mock has within-coder variety; a
    different ``seed`` gives the same coder fresh answers, as a second run would.
    """

    def script(item: Item, task: Task, draw: int) -> Any:
        seed_value = _seed(name, seed, item.id, draw)
        words = item.text.split()
        if task.kind == "label":
            return {task.key: bool(seed_value % 2)}
        if task.kind == "labels":
            names = list(task.codebook)
            low, high = task.per_answer
            count = max(low, min(high, 1 + seed_value % 2))
            start = seed_value % len(names)
            return {task.key: [names[(start + offset) % len(names)] for offset in range(count)]}
        if len(words) < 4:
            if task.kind == "tuple":
                return {task.key: False, **dict.fromkeys(task.names, "")}
            return {task.key: []}
        cut = 1 + (seed_value % (len(words) - 2))
        cause = " ".join(words[: cut + (seed_value // 7) % 2])
        effect = " ".join(words[cut:])
        aux = dict.fromkeys(task.aux, "")

        def relation(first: str, rest: str) -> dict[str, str]:
            return {task.fields[0]: first, **dict.fromkeys(task.fields[1:], rest), **aux}

        if task.kind == "tuple":
            if seed_value % 5 == 0:
                return {task.key: False, **dict.fromkeys(task.names, "")}
            return {task.key: True, **relation(cause, effect)}
        relations = [relation(cause, effect)]
        if seed_value % 3 == 0 and len(words) >= 6:
            relations.append(relation(" ".join(words[-3:-1]), words[-1]))
        return {task.key: relations}

    return script


class MockCoder:
    """A coder that answers from a script without any network.

    The script is a function of item, task and draw returning the parsed JSON object; the default
    script gives schema-valid, verbatim answers.
    """

    def __init__(self, name: str, script: Script | None = None, *, model: str = "mock", seed: int = 0) -> None:
        self.name = name
        self.model = model
        self.seed = seed
        self._script = script or default_script(name, seed)

    def describe(self) -> dict[str, Any]:
        """The seat's identity for a run's provenance: the mock's model name and seed."""
        return {"kind": "mock", "model": self.model, "seed": self.seed}

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        """Answer one item on one draw from the script."""
        parsed = self._script(item, task, draw)
        raw = json.dumps(parsed, ensure_ascii=False)
        validation = validate(task, parsed, item.text)
        identity = {"coder": self.name, "requested_model": self.model, "returned_model": self.model}
        return Answer(
            item.id,
            self.name,
            draw,
            validation.value,
            "ok" if validation.ok else "invalid",
            reason=validation.reason,
            raw=raw,
            identity=identity,
            audit=validation.audit,
        )


class ModelCoder:
    """A coder over a :class:`~codercommittee.protocols.Model`.

    Renders the task's prompt, asks for the task's schema, parses strictly and validates. A refusal
    is an answer with status ``refused``; a transport failure or malformed reply is ``invalid``.
    """

    def __init__(self, name: str, model: Model, *, system: str = SYSTEM_PROMPT) -> None:
        self.name = name
        self.model = model
        self.system = system

    def describe(self) -> dict[str, Any]:
        """The seat's identity for a run's provenance: the adapter's settings and the system prompt's hash."""
        describe = getattr(self.model, "describe", None)
        record = describe() if callable(describe) else None
        settings: dict[str, Any]
        if record is not None:
            settings = dict(record)
        else:
            settings = {
                "provider": getattr(self.model, "provider", None),
                "model": getattr(self.model, "model", None),
                "base_url": public_url(getattr(self.model, "base_url", None)),
            }
        system_sha256 = hashlib.sha256(self.system.encode("utf-8")).hexdigest()[:16]
        return {"kind": "model", **settings, "system_sha256": system_sha256}

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        """Ask the model for the task's schema on one item and validate the reply."""
        completion = self.model.complete(task.render(item.text), task.schema(), system=self.system)
        identity = {"coder": self.name, "draw": draw, **completion.identity}
        if completion.refused:
            reason = scrub_credentials(completion.error or "refused")
            return Answer(item.id, self.name, draw, None, "refused", reason=reason, identity=identity)
        if completion.error:
            reason = scrub_credentials(completion.error)
            return Answer(item.id, self.name, draw, None, "invalid", reason=reason, identity=identity)
        try:
            parsed = strict_json_loads(completion.raw)
        except StrictJSONError as exc:
            reason = f"strict_json:{exc}"
            raw = completion.raw
            return Answer(item.id, self.name, draw, None, "invalid", reason=reason, raw=raw, identity=identity)
        validation = validate(task, parsed, item.text)
        return Answer(
            item.id,
            self.name,
            draw,
            validation.value,
            "ok" if validation.ok else "invalid",
            reason=validation.reason,
            raw=completion.raw,
            identity=identity,
            audit=validation.audit,
        )


__all__ = ["SYSTEM_PROMPT", "MockCoder", "ModelCoder", "Script", "default_script"]
