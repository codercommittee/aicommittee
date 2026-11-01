"""The four protocols a committee is built from: coders, models, embedders and referees."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .align import Prematch
from .data import Answer, Item, Task
from .similarity import Space


@runtime_checkable
class Coder(Protocol):
    """Answers one item on one draw. ``name`` is the coder's id inside a run."""

    name: str

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        """Answer one item on one draw."""
        ...


@dataclass(frozen=True)
class Completion:
    """One raw model reply with the identity of the request that produced it."""

    raw: str
    identity: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    refused: bool = False


@runtime_checkable
class Model(Protocol):
    """A structured-output model call: the provider adapters, used by coders and the referee."""

    def complete(self, prompt: str, schema: Mapping[str, Any], *, system: str) -> Completion:
        """Ask the model for a reply matching the schema."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Returns the vectors of one item's spans; the space is scoped to that item of that run."""

    def space(self, item: Item, texts: Sequence[str]) -> Space:
        """The vectors of one item's spans, as a Space scoped to that item."""
        ...


@runtime_checkable
class Referee(Protocol):
    """Confirms or rejects every candidate pair of a pre-match whose bands split at ``accept_at``."""

    accept_at: float

    def decide(self, prematch: Prematch) -> Mapping[str, bool]:
        """Confirm (True) or reject (False) every candidate pair of the pre-match."""
        ...


def describe_referee(referee: Referee | None) -> dict[str, Any] | None:
    """A referee's configuration as a JSON record: its ``kind``, ``accept_at`` and what else it knows.

    A referee with a ``describe()`` method answers for itself (``None`` meaning it has no record to
    give); any other is recorded by its class name and threshold.
    """
    if referee is None:
        return None
    describe = getattr(referee, "describe", None)
    if callable(describe):
        record = describe()
        if record is None:
            return None
        if not isinstance(record, Mapping):
            raise TypeError(
                f"{type(referee).__name__}.describe() returns a mapping or None, not {type(record).__name__}"
            )
        return {str(key): value for key, value in record.items()}
    accept_at = getattr(referee, "accept_at", None)
    fallback: dict[str, Any] = {"kind": type(referee).__name__}
    if isinstance(accept_at, int | float) and not isinstance(accept_at, bool):
        fallback["accept_at"] = float(accept_at)
    return fallback


def describe_coder(coder: Coder) -> dict[str, Any] | None:
    """A coder's identity as a JSON record: what sits in the seat, for a run's provenance.

    A coder with a ``describe()`` method answers for itself (``None`` meaning it has no record to
    give); any other is recorded by its class name, so a resume with another class in the seat is
    refused. A record's ``model`` names the same thing as the answers' ``identity["requested_model"]``:
    a store checks one against the other when it adopts a seat it never recorded.
    """
    describe = getattr(coder, "describe", None)
    if callable(describe):
        record = describe()
        if record is None:
            return None
        if not isinstance(record, Mapping):
            raise TypeError(f"{type(coder).__name__}.describe() returns a mapping or None, not {type(record).__name__}")
        return {str(key): value for key, value in record.items()}
    return {"kind": type(coder).__name__}


__all__ = ["Coder", "Completion", "Embedder", "Model", "Referee", "describe_coder", "describe_referee"]
