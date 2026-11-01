"""The data model: tasks, items, relations, answers, units, predictions, runs and scores.

Every public type is immutable. A :class:`Task` says what an answer is (its *kind*), an
:class:`Item` is one text with an optional human reference, an :class:`Answer` is one coder's
validated reply to one item on one draw, a :class:`Prediction` is what a committee structure
says about an item, and a :class:`Run` holds every answer and every embedding space of one
cycle over the items.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .compare import Agreement, Scores
    from .protocols import Referee
    from .similarity import Similarity, Space

Kind = Literal["label", "labels", "tuple", "tuples"]
KINDS: tuple[Kind, ...] = ("label", "labels", "tuple", "tuples")
SPAN_KINDS: tuple[Kind, ...] = ("tuple", "tuples")
AnswerStatus = Literal["ok", "invalid", "refused", "missing"]
PredictionStatus = Literal["ok", "insufficient"]
ItemId = str | int


class Relation:
    """The one structured atom: an immutable mapping of field name to a stripped span.

    Field names come from the :class:`Task` (the study's ``cause``, ``effect`` and ``signal`` are
    defaults, not types). Fields are read as attributes or items::

        r = Relation(cause="rain", effect="flooding", signal="caused")
        r.cause == r["cause"] == "rain"
    """

    __slots__ = ("_spans",)
    _spans: dict[str, str]

    def __init__(self, **spans: str) -> None:
        clean: dict[str, str] = {}
        for name, value in spans.items():
            if not isinstance(value, str):
                raise TypeError(f"relation field {name!r} must be a string, got {type(value).__name__}")
            clean[name] = value.strip()
        object.__setattr__(self, "_spans", clean)

    def __getattr__(self, name: str) -> str:
        if name.startswith("__"):
            raise AttributeError(name)
        spans: dict[str, str] = object.__getattribute__(self, "_spans")
        try:
            return spans[name]
        except KeyError:
            raise AttributeError(f"relation has no field {name!r}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Relation is immutable")

    def __getitem__(self, name: str) -> str:
        return self._spans[name]

    def __contains__(self, name: object) -> bool:
        return name in self._spans

    def __iter__(self) -> Iterator[str]:
        return iter(self._spans)

    def __len__(self) -> int:
        return len(self._spans)

    def get(self, name: str, default: str = "") -> str:
        """The field's span, or ``default`` when the relation has no such field."""
        return self._spans.get(name, default)

    def keys(self) -> tuple[str, ...]:
        """The field names in field order."""
        return tuple(self._spans)

    def items(self) -> tuple[tuple[str, str], ...]:
        """The (field, span) pairs in field order."""
        return tuple(self._spans.items())

    def as_dict(self) -> dict[str, str]:
        """A plain dict, field by field, in field order."""
        return dict(self._spans)

    def _filled(self) -> tuple[tuple[str, str], ...]:
        """The non-empty spans: a field left out and a field given as "" are the same thing."""
        return tuple(sorted((name, value) for name, value in self._spans.items() if value))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Relation) and self._filled() == other._filled()

    def __hash__(self) -> int:
        return hash(self._filled())

    def __repr__(self) -> str:
        inner = ", ".join(f"{name}={value!r}" for name, value in self._spans.items())
        return f"Relation({inner})"


Reference = bool | frozenset[str] | tuple[Relation, ...]
"""A stored reference: a boolean, a frozenset of label names, or a tuple of relations."""
ReferenceInput = bool | AbstractSet[str] | Sequence[Relation] | None
"""What ``Item`` accepts: a boolean, any set of names, any sequence of relations, or None; stored as ``Reference``."""
Value = bool | frozenset[str] | Relation | tuple[Relation, ...]

_DEFAULT_KEY: dict[str, str] = {"labels": "labels", "tuples": "relations", "tuple": "causal"}


@dataclass(frozen=True)
class Task:
    """What one answer is.

    ``kind`` is one of ``label`` (one boolean), ``labels`` (a set of names from the codebook),
    ``tuple`` (a flag plus one relation) or ``tuples`` (a list of relations). ``codebook`` maps
    label names to descriptions: exactly one entry for the ``label`` kind (the boolean's name),
    the vocabulary for ``labels``, empty for the span kinds. ``key`` is the answer's JSON key
    (the label's name, ``labels``, ``relations``, or the flag key ``causal``). ``per_answer``
    bounds the label count of one ``labels`` answer. ``fields`` are the span fields similarity
    reads; ``aux`` are span fields that are carried, keyed and printed but never compared.
    """

    kind: Kind
    codebook: Mapping[str, str] = field(default_factory=dict)
    prompt: str = "{{TEXT}}"
    key: str = ""
    per_answer: tuple[int, int] = (1, 1)
    fields: tuple[str, ...] = ("cause", "effect")
    aux: tuple[str, ...] = ("signal",)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        codebook = dict(self.codebook)
        if any(not isinstance(name, str) or not name for name in codebook):
            raise ValueError("codebook names must be non-empty strings")
        if any(name != name.strip() for name in codebook):
            raise ValueError("codebook names carry no surrounding whitespace (answers are stripped before lookup)")
        if any(character.isspace() and character != " " for name in codebook for character in name):
            raise ValueError("codebook names hold no line breaks (they are printed into the prompt)")
        object.__setattr__(self, "codebook", codebook)
        if self.kind == "label":
            if len(codebook) != 1:
                raise ValueError("a label task's codebook names exactly one label")
            key = self.key or next(iter(codebook))
        else:
            key = self.key or _DEFAULT_KEY[self.kind]
        object.__setattr__(self, "key", key)
        if self.kind == "labels":
            if not codebook:
                raise ValueError("a labels task needs a codebook")
            low, high = self.per_answer
            if not (isinstance(low, int) and isinstance(high, int) and 0 <= low <= high and high >= 1):
                raise ValueError("per_answer must be (low, high) with 0 <= low <= high and high >= 1")
            folded = {name.casefold() for name in codebook}
            if len(folded) != len(codebook):
                raise ValueError("codebook names must be distinct ignoring case")
        if self.kind in SPAN_KINDS:
            names = tuple(self.fields) + tuple(self.aux)
            if not self.fields:
                raise ValueError("a span task needs at least one similarity field")
            if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
                raise ValueError("fields and aux must be distinct non-empty names")
            if self.kind == "tuple" and key in names:
                raise ValueError("the flag key must differ from the span fields")
            object.__setattr__(self, "fields", tuple(self.fields))
            object.__setattr__(self, "aux", tuple(self.aux))
        if "{{TEXT}}" not in self.prompt:
            raise ValueError("the prompt must contain the {{TEXT}} placeholder")

    @property
    def names(self) -> tuple[str, ...]:
        """Every span field in order: the similarity fields, then the auxiliary ones."""
        return tuple(self.fields) + tuple(self.aux)

    def render(self, text: str) -> str:
        """The prompt for one text; ``{{CODEBOOK}}`` expands to the codebook, one line per label."""
        codebook = "\n".join(f"- {name}: {description}" for name, description in self.codebook.items())
        return self.prompt.replace("{{TEXT}}", text).replace("{{CODEBOOK}}", codebook)

    def schema(self) -> dict[str, Any]:
        """The portable JSON schema of one answer (object, array, string, boolean and enum only)."""
        if self.kind == "label":
            return _object({self.key: {"type": "boolean"}})
        if self.kind == "labels":
            return _object({self.key: {"type": "array", "items": {"type": "string", "enum": list(self.codebook)}}})
        spans = {name: {"type": "string"} for name in self.names}
        if self.kind == "tuple":
            return _object({self.key: {"type": "boolean"}, **spans})
        return _object({self.key: {"type": "array", "items": _object(spans)}})


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class Item:
    """One text to code, with its human reference when there is one.

    The reference is a boolean (``label``), a set of label names (``labels``) or a sequence of
    relations (``tuple``: every retained coder tuple, an empty sequence meaning non-causal;
    ``tuples``: the reference relations). ``None`` means no reference: the item is coded and
    compared across coders but never scored.
    """

    id: ItemId
    text: str
    reference: Reference | None = None

    def __init__(self, id: ItemId, text: str, reference: ReferenceInput = None) -> None:
        if isinstance(id, bool) or not isinstance(id, str | int):
            raise TypeError("an item id is a string or an integer")
        if not isinstance(text, str):
            raise TypeError("item text must be a string")
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "reference", normalize_reference(reference))


def check_coder_name(name: object) -> None:
    """A coder name is a non-empty string without surrounding whitespace or ``|``.

    ``|`` joins the two members of a pair label, so a name holding it could spell another pair.
    ``:`` (as in a model tag like ``qwen3:32b``) is fine: a node label is the name followed by
    ``:`` and an integer, which no other name can reproduce.
    """
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError(f"coder name {name!r}: a non-empty string without surrounding whitespace")
    if any(character.isspace() and character != " " for character in name):
        raise ValueError(f"coder name {name!r} holds a line break or another whitespace character than a space")
    if "|" in name:
        raise ValueError(f"coder name {name!r} holds '|', which the pair labels reserve")


def normalize_reference(reference: object) -> Reference | None:
    """Coerce a reference into its stored form: bool, frozenset of names, or tuple of relations."""
    if reference is None or isinstance(reference, bool):
        return reference
    if isinstance(reference, set | frozenset):
        if any(not isinstance(name, str) for name in reference):
            raise TypeError("a label-set reference holds strings")
        return frozenset(reference)
    if isinstance(reference, list | tuple):
        if any(not isinstance(relation, Relation) for relation in reference):
            raise TypeError("a relation reference holds Relation values")
        return tuple(reference)
    raise TypeError(f"unsupported reference type {type(reference).__name__}")


@dataclass(frozen=True)
class Answer:
    """One coder's validated reply to one item on one draw.

    ``value`` is the normalized answer: a boolean, a frozenset of label names, a
    :class:`Relation` or ``None`` for the ``tuple`` kind (``None`` with status ``ok`` is the
    non-causal answer), or a tuple of relations, deduplicated once, for ``tuples``. Any
    status other than ``ok`` removes the item from every arm of its run. ``identity`` is an
    opaque record of where the answer came from (model ids, request fingerprint); the replay
    coder checks it, live adapters fill it.
    """

    item: ItemId
    coder: str
    draw: int
    value: Value | None
    status: AnswerStatus
    reason: str | None = None
    raw: str | None = None
    identity: Mapping[str, Any] = field(default_factory=dict)
    audit: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the answer's status is ``ok``."""
        return self.status == "ok"


@dataclass(frozen=True)
class Member:
    """One member's candidate relation inside an alignment: who said it, and which of theirs."""

    name: str
    index: int
    relation: Relation
    coder: str


@dataclass(frozen=True)
class Unit:
    """An aligned unit: the relations the committee's members constructed for one thing.

    ``support`` counts the members in the unit; ``coders`` counts the distinct coders among them
    (from the structured members, never by parsing a name). ``representative`` is the member
    whose relation is the unit's medoid, and ``medoid`` that relation.
    """

    members: tuple[Member, ...]
    representative: Member

    @property
    def medoid(self) -> Relation:
        """The unit's representative relation."""
        return self.representative.relation

    @property
    def support(self) -> int:
        """The number of members in the unit."""
        return len(self.members)

    @property
    def coders(self) -> int:
        """The number of distinct coders among the members."""
        return len({member.coder for member in self.members})


@dataclass(frozen=True)
class Prediction:
    """What one structure says about one item: a value with a status, and the units behind it."""

    value: Value | None
    status: PredictionStatus
    units: tuple[Unit, ...] = ()
    audit: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the prediction's status is ``ok``."""
        return self.status == "ok"


@dataclass(frozen=True)
class Match:
    """One matched prediction-reference pair of a relation score."""

    prediction: int
    reference: int
    similarity: float


Confusion = tuple[int, int, int, int, int, int]
"""(tp_pos, fp_pos, fn_pos, tp_neg, fp_neg, fn_neg) of one binary label."""


@dataclass(frozen=True)
class ItemScore:
    """One prediction's score against one reference, with what the level needs later."""

    value: float
    confusion: Confusion | None = None
    matched: tuple[Match, ...] = ()


@dataclass(frozen=True)
class Run:
    """One cycle: every coder's answers on every draw over the items, and each item's space.

    ``items`` keep their run order; ``coders`` is the seat order (the order the coders were given
    in), which only best-single ties use. ``complete()`` are the items with every coder x draw
    answered with status ``ok``; ``scored()`` are the complete items with a reference the level
    uses (for the ``tuple`` kind a causal one). ``provenance`` records how the run was made (the
    referee's configuration, each coder's identity); a store writes it into ``run.json``.
    """

    task: Task
    items: tuple[Item, ...]
    coders: tuple[str, ...]
    draws: int
    answers: Mapping[tuple[ItemId, str, int], Answer]
    spaces: Mapping[ItemId, Space] = field(default_factory=dict)
    referee: Referee | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("item ids must be unique within a run")
        if len(set(self.coders)) != len(self.coders) or not self.coders:
            raise ValueError("coders must be distinct and at least one")
        for name in self.coders:
            check_coder_name(name)
        known_items, seats = set(ids), set(self.coders)
        for (item_id, coder, draw), answer in self.answers.items():
            if (answer.item, answer.coder, answer.draw) != (item_id, coder, draw):
                raise ValueError(
                    f"answer keyed {(item_id, coder, draw)!r} names {(answer.item, answer.coder, answer.draw)!r}"
                )
            if item_id not in known_items or coder not in seats or not 1 <= draw <= self.draws:
                raise ValueError(f"answer {(item_id, coder, draw)!r} names no item, seat or draw of this run")

    def answer(self, item: ItemId, coder: str, draw: int) -> Answer:
        """The stored answer, or a ``missing`` one when none was recorded."""
        found = self.answers.get((item, coder, draw))
        if found is None:
            return Answer(item, coder, draw, None, "missing", reason="no answer recorded")
        return found

    def is_complete(self, item: Item) -> bool:
        """Whether every coder answered the item on every draw with status ``ok``."""
        return all(self.answer(item.id, coder, draw).ok for coder in self.coders for draw in range(1, self.draws + 1))

    def is_scored(self, item: Item) -> bool:
        """Whether the item is complete and carries a reference the level uses."""
        if item.reference is None or not self.is_complete(item):
            return False
        if self.task.kind == "tuple":
            return len(item.reference) > 0  # type: ignore[arg-type]
        return True

    def complete(self) -> tuple[Item, ...]:
        """The complete items, in run order."""
        return tuple(item for item in self.items if self.is_complete(item))

    def scored(self) -> tuple[Item, ...]:
        """The scored items, in run order."""
        return tuple(item for item in self.items if self.is_scored(item))

    def similarity(self, item: ItemId) -> Similarity | None:
        """The task's similarity over the item's space, or ``None`` when the item has no space."""
        from .similarity import Similarity

        space = self.spaces.get(item)
        if space is None:
            return None
        return Similarity(space, fields=self.task.fields, aux=self.task.aux)

    def item(self, item: ItemId) -> Item:
        """The item with this id."""
        for candidate in self.items:
            if candidate.id == item:
                return candidate
        raise KeyError(f"no item {item!r} in the run")

    # The facade: each method is one call into the core.

    def arms(
        self, referee: Referee | None = None, pool_k: Sequence[int] | None = None
    ) -> dict[str, dict[ItemId, tuple[Prediction, ...]]]:
        """Every structure's predictions per arm and item, one per realization (see :mod:`codercommittee.arms`)."""
        from .arms import POOL_KS, arms

        return arms(self, referee, POOL_KS if pool_k is None else pool_k)

    def predictions(self, arm: str, referee: Referee | None = None) -> dict[ItemId, tuple[Prediction, ...]]:
        """One arm's predictions per item, one per realization."""
        built = self.arms(referee)
        if arm not in built:
            raise KeyError(f"no arm {arm!r}; the arms are {list(built)}")
        return built[arm]

    def units(self, arm: str, referee: Referee | None = None) -> dict[ItemId, tuple[tuple[Unit, ...], ...]]:
        """One arm's aligned units per item, one tuple per realization (empty for the label kinds)."""
        return {
            item: tuple(p.units for p in realizations) for item, realizations in self.predictions(arm, referee).items()
        }

    def scores(self, referee: Referee | None = None, pool_k: Sequence[int] | None = None) -> Scores:
        """Every arm scored on the scored items (see :class:`codercommittee.compare.Scores`)."""
        from .compare import Scores

        return Scores.of(self, self.arms(referee, pool_k))

    def agreement(self) -> Agreement:
        """Cross- and within-coder disagreement over the complete items."""
        from .compare import agreement

        return agreement(self)

    def save(self, directory: str | Path) -> None:
        """Write the run into a directory in the package's layout (see :mod:`codercommittee.store`)."""
        from .store import save_run

        save_run(self, directory)

    @staticmethod
    def load(directory: str | Path, referee: Referee | None = None) -> Run:
        """Rebuild a saved run: stored replies re-validated, vectors served, recorded decisions replayed."""
        from .store import load_run

        return load_run(directory, referee)

    def to_frame(self) -> Any:
        """A pandas DataFrame with one row per answer (pandas required)."""
        import pandas as pd

        from .replay import value_as_stored

        rows = [
            {
                "item": answer.item,
                "coder": answer.coder,
                "draw": answer.draw,
                "status": answer.status,
                "reason": answer.reason,
                "value": value_as_stored(answer.value, self.task) if answer.ok else None,
            }
            for answer in self.answers.values()
        ]
        return pd.DataFrame(rows)


__all__ = [
    "KINDS",
    "SPAN_KINDS",
    "Answer",
    "AnswerStatus",
    "Confusion",
    "Item",
    "ItemId",
    "ItemScore",
    "Kind",
    "Match",
    "Member",
    "Prediction",
    "PredictionStatus",
    "Reference",
    "ReferenceInput",
    "Relation",
    "Run",
    "Task",
    "Unit",
    "Value",
    "check_coder_name",
    "normalize_reference",
]
