"""Referees: who confirms or rejects the candidate pairs of a pre-match.

The band rule accepts every pair at or above its threshold and rejects the rest. The LLM
referee reads the passage and the listed relations and decides every candidate pair, the high
band included; it can decide only the pairs the pre-match listed (its schema enumerates them),
and the computed similarities are never replaced by what it reports. The replay referee serves
stored decisions bound to the pre-match's sha256. A referee asked about a pre-match without
candidate pairs has nothing to decide and answers with an empty mapping, without any call.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .align import Prematch
from .protocols import Model
from .providers.base import TRANSPORT_BUDGET
from .replay import MAX_ATTEMPTS, ReplayError, StoredRun, _read_json
from .strictjson import StrictJSONError, strict_json_loads

PROMPT_PATH = Path(__file__).with_name("referee_prompt.md")
PROMPT_VERSION = "alignment-agent-v1"
SYSTEM = "You are a careful research annotator. Return only schema-valid JSON."
DECISIONS = ("confirm", "reject")


class RefereeError(RuntimeError):
    """The referee gave no usable decision inside its block of attempts."""


class BandRule:
    """Accept every candidate pair whose similarity is at or above ``accept_at``; reject the rest."""

    def __init__(self, accept_at: float = 0.80) -> None:
        if not 0.0 < accept_at <= 1.0:
            raise ValueError("accept_at lies in (0, 1]")
        self.accept_at = float(accept_at)

    def decide(self, prematch: Prematch) -> dict[str, bool]:
        """Accept the high band and reject the uncertain band."""
        if prematch.high != self.accept_at:
            raise ValueError("the pre-match was banded at another threshold than this rule accepts at")
        return {edge.pair: edge.band == "high" for edge in prematch.edges}

    def describe(self) -> dict[str, Any]:
        """The rule's record for a run's provenance."""
        return {"kind": "band", "accept_at": self.accept_at}

    def __repr__(self) -> str:
        return f"BandRule(accept_at={self.accept_at})"


def load_prompt(path: Path = PROMPT_PATH) -> str:
    """The referee's prompt template; its first line names the prompt version."""
    template = path.read_text(encoding="utf-8")
    if f"— {PROMPT_VERSION}" not in template.splitlines()[0]:
        raise ValueError("the referee prompt does not carry its version in the first line")
    return template


def render_prompt(template: str, prematch: Prematch, fields: tuple[str, ...] = ("cause", "effect")) -> str:
    """The prompt for one pre-match: the passage, the relations by node, the candidate pairs with bands."""
    relations = "\n".join(
        f"- {member.name}:{member.index}: "
        + "; ".join(f"{name} = {json.dumps(member.relation.get(name, ''), ensure_ascii=False)}" for name in fields)
        for member in prematch.nodes
    )
    pairs = "\n".join(
        f"- {edge.pair}: {edge.a} with {edge.b} ({edge.band} band, similarity {edge.similarity:.3f}"
        + (", inside an inconsistent group" if edge.in_conflict else "")
        + ")"
        for edge in prematch.edges
    )
    return template.replace("{{TEXT}}", prematch.text).replace("{{RELATIONS}}", relations).replace("{{PAIRS}}", pairs)


def decision_schema(prematch: Prematch) -> dict[str, Any]:
    """A portable schema whose pair enum is exactly the candidate pairs."""
    pairs = list(prematch.candidate_pairs)
    if not pairs:
        raise ValueError("the referee has nothing to decide without candidate pairs")
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pair": {"type": "string", "enum": pairs},
                        "decision": {"type": "string", "enum": list(DECISIONS)},
                    },
                    "required": ["pair", "decision"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def decisions_in(raw: str) -> dict[str, bool]:
    """The decisions a raw reply names, each pair once; coverage of the candidate pairs is not checked here."""
    try:
        parsed = strict_json_loads(raw)
    except StrictJSONError as exc:
        raise ValueError(f"referee reply is not strict JSON: {exc}") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"decisions"} or not isinstance(parsed["decisions"], list):
        raise ValueError("referee reply must be an object with a decisions array")
    decisions: dict[str, bool] = {}
    for entry in parsed["decisions"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"pair", "decision"}
            or not isinstance(entry["pair"], str)
            or entry["decision"] not in DECISIONS
            or entry["pair"] in decisions
        ):
            raise ValueError("referee reply names a pair twice, an unknown decision, or a malformed entry")
        decisions[entry["pair"]] = entry["decision"] == "confirm"
    return decisions


def parse_decisions(raw: str, prematch: Prematch) -> dict[str, bool]:
    """The decisions in a raw reply: every candidate pair exactly once and nothing else, else ValueError."""
    decisions = decisions_in(raw)
    if set(decisions) != set(prematch.candidate_pairs):
        raise ValueError("referee decisions do not cover the candidate pairs exactly")
    return decisions


class LLMReferee:
    """A model confirms or rejects every candidate pair of the pre-match it is shown.

    The pre-match is banded at ``accept_at`` for the prompt; the model decides the high band too.
    An unusable reply (refusal, transport failure, malformed or incomplete decisions) is asked
    again inside one budget of ``attempts``, the transport attempts counting; after the budget a
    :class:`RefereeError` is raised.
    """

    def __init__(
        self,
        model: Model,
        *,
        accept_at: float = 0.80,
        template: str | None = None,
        system: str = SYSTEM,
        attempts: int = MAX_ATTEMPTS,
        fields: tuple[str, ...] = ("cause", "effect"),
    ) -> None:
        self.model = model
        self.accept_at = accept_at
        self.template = template if template is not None else load_prompt()
        self.system = system
        self.attempts = attempts
        self.fields = fields
        self.calls = 0
        self.last_identity: dict[str, Any] = {}

    def describe(self) -> dict[str, Any]:
        """The referee's record for a run's provenance: the adapter's settings, the template and the system prompt."""
        describe = getattr(self.model, "describe", None)
        record = describe() if callable(describe) else None
        settings: dict[str, Any]
        if record is not None:
            settings = dict(record)
        else:
            settings = {"provider": getattr(self.model, "provider", None), "model": getattr(self.model, "model", None)}
        return {
            "kind": "llm",
            "accept_at": self.accept_at,
            **settings,
            "prompt_version": PROMPT_VERSION,
            "template_sha256": hashlib.sha256(self.template.encode("utf-8")).hexdigest()[:16],
            "system_sha256": hashlib.sha256(self.system.encode("utf-8")).hexdigest()[:16],
            "fields": list(self.fields),
        }

    def decide(self, prematch: Prematch) -> dict[str, bool]:
        """Ask the model to decide every candidate pair; nothing to decide gives an empty mapping."""
        if not prematch.candidate_pairs:
            return {}
        prompt = render_prompt(self.template, prematch, self.fields)
        schema = decision_schema(prematch)
        reasons: list[str] = []
        made = 0
        while made < self.attempts:
            self.calls += 1
            token = TRANSPORT_BUDGET.set(self.attempts - made)
            try:
                completion = self.model.complete(prompt, schema, system=self.system)
            finally:
                TRANSPORT_BUDGET.reset(token)
            used = completion.identity.get("attempts")
            made += used if isinstance(used, int) and not isinstance(used, bool) and used >= 1 else 1
            if completion.refused or completion.error:
                reasons.append(completion.error or "refused")
                continue
            try:
                decisions = parse_decisions(completion.raw, prematch)
            except ValueError as exc:
                reasons.append(str(exc))
                continue
            self.last_identity = dict(completion.identity)
            return decisions
        where = prematch.label or "the pre-match"
        raise RefereeError(f"no usable decision for {where} after {self.attempts} attempts: {reasons}")


def identity_acceptable(requested: str, returned: object) -> bool:
    """The referee has no pinned identity: the server may return the requested id or a dated release of it."""
    if not isinstance(returned, str) or not returned.strip():
        return False
    return returned == requested or re.fullmatch(re.escape(requested) + r"-\d{8}", returned) is not None


class ReplayReferee:
    """Serves the stored decisions of a run, bound to the pre-match by its sha256.

    A record must exist iff the pre-match has candidate pairs; decisions are re-parsed from the
    stored raw reply and must name every candidate pair exactly once.
    """

    def __init__(self, stored: StoredRun | str | Path, *, accept_at: float = 0.80) -> None:
        self.stored = stored if isinstance(stored, StoredRun) else StoredRun(stored)
        self.accept_at = accept_at
        service = (self.stored.summary.get("services") or {}).get("alignment_agent") or {}
        self.requested: str | None = service.get("requested_model_id")
        self.prompt_version: str | None = service.get("prompt_version")

    def describe(self) -> dict[str, Any]:
        """The referee's record for a run's provenance: the stored run's referee model and prompt."""
        return {
            "kind": "replay",
            "accept_at": self.accept_at,
            "model": self.requested,
            "prompt_version": self.prompt_version,
        }

    def records(self) -> list[dict[str, Any]]:
        """Every stored decision as a record of the package's own store (item, label, hash, decisions, identity).

        ``Run.save`` copies these into ``decisions.jsonl`` so a saved replay scores as the study did.
        """
        out: list[dict[str, Any]] = []
        for path in sorted((self.stored.dir / "checkpoints" / "agent").glob("*.json")):
            checkpoint = _read_json(path)
            if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("response"), dict):
                raise ReplayError(f"{path} is not a referee checkpoint")
            label, sha = checkpoint.get("label"), checkpoint.get("prematch_sha256")
            if "item_id" not in checkpoint or not isinstance(label, str) or not isinstance(sha, str):
                raise ReplayError(f"{path} lacks its item, label or pre-match hash")
            raw = self._checked_raw(checkpoint["response"], f"{path.name}")
            try:
                decisions = decisions_in(raw)
            except ValueError as exc:
                raise ReplayError(f"{path}: {exc}") from exc
            response = checkpoint["response"]
            identity = {
                key: response.get(key)
                for key in ("requested_model_id", "returned_model_id", "request_fingerprint", "provider_response_id")
            }
            identity["prompt_version"] = response.get("prompt_version")
            out.append(
                {
                    "item": checkpoint["item_id"],
                    "label": label,
                    "prematch_sha256": sha,
                    "decisions": decisions,
                    "identity": identity,
                }
            )
        return out

    def _checked_raw(self, response: Mapping[str, Any], where: str) -> str:
        """The raw reply of a stored referee record, after every identity check the record must pass."""
        requested = response.get("requested_model_id")
        if self.requested is not None and requested != self.requested:
            raise ReplayError(f"referee record of {where} asked another model ({requested})")
        if not identity_acceptable(str(requested), response.get("returned_model_id")):
            raise ReplayError(f"referee record of {where} returned an unacceptable identity")
        if self.prompt_version is not None and response.get("prompt_version") != self.prompt_version:
            raise ReplayError(f"referee record of {where} used another prompt version")
        attempts = response.get("attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= MAX_ATTEMPTS:
            raise ReplayError(f"referee record of {where} has an invalid attempt count")
        raw = response.get("raw_response")
        if not isinstance(raw, str):
            raise ReplayError(f"referee record of {where} lacks a raw response")
        return raw

    def record(self, prematch: Prematch) -> dict[str, Any] | None:
        """The stored record of this pre-match's realization, or None when none was written."""
        if prematch.item is None or prematch.label is None:
            raise ReplayError("replaying a referee needs the pre-match's item and label")
        return self.stored.agent(self.stored.index_of(prematch.item), prematch.label)

    def decide(self, prematch: Prematch) -> dict[str, bool]:
        """The stored decisions of this pre-match, re-parsed and bound by its hash."""
        if prematch.high != self.accept_at:
            raise ValueError("the pre-match was banded at another threshold than the stored decisions")
        record = self.record(prematch)
        where = f"item {prematch.item!r}, {prematch.label}"
        if not prematch.candidate_pairs:
            if record is not None:
                raise ReplayError(f"a referee record exists for {where} although there was nothing to decide")
            return {}
        if record is None:
            raise ReplayError(f"referee record missing for {where}")
        if record.get("prematch_sha256") != prematch.sha256:
            raise ReplayError(f"the stored pre-match of {where} differs from the recomputed one")
        response = record.get("response")
        if not isinstance(response, dict):
            raise ReplayError(f"referee record of {where} carries no response")
        raw = self._checked_raw(response, where)
        try:
            return parse_decisions(raw, prematch)
        except ValueError as exc:
            raise ReplayError(f"referee record of {where}: {exc}") from exc


__all__ = [
    "DECISIONS",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "SYSTEM",
    "BandRule",
    "LLMReferee",
    "RefereeError",
    "ReplayReferee",
    "decision_schema",
    "decisions_in",
    "identity_acceptable",
    "load_prompt",
    "parse_decisions",
    "render_prompt",
]
