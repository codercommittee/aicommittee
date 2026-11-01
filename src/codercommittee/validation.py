"""Validation and normalization of one parsed answer against its task.

Nothing reaches aggregation unless it is valid for the item it answers. A span that is not a
copy of the item text is kept as written and listed in the audit; a span that matches the
text after a narrow six-character punctuation fold (curly quotes, apostrophes, dashes) counts
as a copy and takes the text's own characters. Auxiliary spans (the study's ``signal``) are
kept only when they are copies of the text, else emptied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .data import SPAN_KINDS, Reference, Relation, Task, Value

PUNCTUATION_FOLD = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
)


@dataclass(frozen=True)
class Validation:
    """The normalized value with an optional failure reason and an audit of the normalization."""

    value: Value | None
    reason: str | None = None
    audit: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when validation found no fault."""
        return self.reason is None


def verbatim(span: str, text: str) -> tuple[str, bool]:
    """``span`` as it appears in ``text`` and True, or ``span`` as written and False."""
    if not span or span in text:
        return span, True
    start = text.translate(PUNCTUATION_FOLD).find(span.translate(PUNCTUATION_FOLD))
    return (text[start : start + len(span)], True) if start >= 0 else (span, False)


def validate(task: Task, parsed: Any, text: str) -> Validation:
    """Validate and normalize a parsed answer for one item."""
    if task.kind == "label":
        return _label(task, parsed)
    if task.kind == "labels":
        return _labels(task, parsed)
    if task.kind == "tuple":
        return _tuple(task, parsed, text)
    return _tuples(task, parsed, text)


def _bad(reason: str) -> Validation:
    return Validation(None, reason)


def _label(task: Task, parsed: Any) -> Validation:
    if not isinstance(parsed, dict) or set(parsed) != {task.key}:
        return _bad("expected_exact_label_object")
    value = parsed[task.key]
    if not isinstance(value, bool):
        return _bad("label_must_be_boolean")
    return Validation(value)


def _labels(task: Task, parsed: Any) -> Validation:
    if not isinstance(parsed, dict) or set(parsed) != {task.key}:
        return _bad("expected_exact_labels_object")
    labels = parsed[task.key]
    low, high = task.per_answer
    if not isinstance(labels, list) or not low <= len(labels) <= high:
        return _bad(f"labels_must_be_array_of_{low}_to_{high}")
    if any(not isinstance(label, str) for label in labels):
        return _bad("label_names_must_be_strings")
    lookup = {name.casefold(): name for name in task.codebook}
    normalized = [lookup.get(label.strip().casefold()) for label in labels]
    if len(set(normalized)) != len(normalized):
        return _bad("labels_must_be_unique")
    unknown = sorted({label.strip() for label, name in zip(labels, normalized, strict=True) if name is None})
    if unknown:
        return _bad("unknown_labels:" + ",".join(unknown))
    return Validation(frozenset(name for name in normalized if name is not None))


def _tuple(task: Task, parsed: Any, text: str) -> Validation:
    required = {task.key, *task.names}
    if not isinstance(parsed, dict) or set(parsed) != required:
        return _bad("expected_exact_tuple_object")
    flag = parsed[task.key]
    if not isinstance(flag, bool):
        return _bad("flag_must_be_boolean")
    if any(not isinstance(parsed[name], str) for name in task.names):
        return _bad("spans_must_be_strings")
    spans = {name: str(parsed[name]).strip() for name in task.names}
    if not flag and any(spans[name] for name in task.fields):
        return _bad("negative_answer_has_spans")
    # A positive answer without every field states no complete relation: it is read as the
    # negative answer, for every coder, instead of rejected.
    incomplete = flag and any(not spans[name] for name in task.fields)
    audit: dict[str, Any] = {"incomplete_read_as_negative": incomplete}
    if incomplete:
        flag = False
        spans = dict.fromkeys(task.names, "")
    nonverbatim: list[str] = []
    for name in task.fields:
        spans[name], is_copy = verbatim(spans[name], text)
        if not is_copy:
            nonverbatim.append(name)
    audit["aux"] = {}
    for name in task.aux:
        supplied = bool(spans[name])
        valid = supplied and spans[name] in text
        if not flag or not valid:
            spans[name] = ""
        audit["aux"][name] = (
            "not_applicable" if not flag else "pass" if valid else "absent" if not supplied else "nonverbatim"
        )
    if nonverbatim:
        audit["nonverbatim"] = nonverbatim
    return Validation(Relation(**spans) if flag else None, audit=audit)


def _tuples(task: Task, parsed: Any, text: str) -> Validation:
    if not isinstance(parsed, dict) or set(parsed) != {task.key}:
        return _bad("expected_exact_relations_object")
    relations = parsed[task.key]
    if not isinstance(relations, list):
        return _bad("relations_must_be_array")
    required = set(task.names)
    out: list[Relation] = []
    nonverbatim: list[dict[str, Any]] = []
    aux_audit: list[dict[str, Any]] = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict) or set(relation) != required:
            return _bad(f"relation_{index}_has_wrong_shape")
        if any(not isinstance(relation[name], str) for name in task.names):
            return _bad(f"relation_{index}_has_nonstring_span")
        spans = {name: str(relation[name]).strip() for name in task.names}
        if any(not spans[name] for name in task.fields):
            return _bad(f"relation_{index}_missing_field")
        for name in task.fields:
            spans[name], is_copy = verbatim(spans[name], text)
            if not is_copy:
                nonverbatim.append({"relation": index, "field": name})
        for name in task.aux:
            supplied = bool(spans[name])
            valid = supplied and spans[name] in text
            status = "pass" if valid else "absent" if not supplied else "nonverbatim"
            aux_audit.append({"relation": index, "field": name, "status": status})
            if not valid:
                spans[name] = ""
        out.append(Relation(**spans))
    audit: dict[str, Any] = {"aux": aux_audit}
    if nonverbatim:
        audit["nonverbatim"] = nonverbatim
    return Validation(tuple(out), audit=audit)


def check_reference(task: Task, reference: Reference | None) -> None:
    """Raise when a reference does not fit the task's kind."""
    if reference is None:
        return
    if task.kind == "label":
        if not isinstance(reference, bool):
            raise TypeError("a label reference is a boolean")
    elif task.kind == "labels":
        if not isinstance(reference, frozenset):
            raise TypeError("a labels reference is a set of label names")
        unknown = sorted(reference - set(task.codebook))
        if unknown:
            raise ValueError(f"reference labels outside the codebook: {unknown}")
    else:
        if not isinstance(reference, tuple):
            raise TypeError("a relation reference is a sequence of Relation values")
        for relation in reference:
            missing = [name for name in task.fields if not relation.get(name)]
            if missing:
                raise ValueError(f"reference relation {relation!r} lacks {missing}")


def prediction_cap(task: Task, members: int, k: int) -> int | None:
    """The most labels an aggregated ``labels`` prediction can hold: floor(n * m / k)."""
    if task.kind != "labels":
        return None
    return (members * task.per_answer[1]) // k


def check_prediction(task: Task, value: Value | None, members: int, k: int) -> None:
    """The boundary between aggregation and scoring: an aggregation bug must not reach a score."""
    if task.kind == "label":
        if not isinstance(value, bool):
            raise RuntimeError("a label prediction must be a boolean")
    elif task.kind == "labels":
        cap = prediction_cap(task, members, k)
        if not isinstance(value, frozenset) or (cap is not None and len(value) > cap):
            raise RuntimeError(f"a labels prediction holds at most {cap} labels")
        if not value <= set(task.codebook):
            raise RuntimeError("a labels prediction names labels outside the codebook")
    elif task.kind == "tuple":
        if value is not None and (not isinstance(value, Relation) or any(not value.get(n) for n in task.fields)):
            raise RuntimeError("a positive tuple prediction needs every field")
    else:
        if not isinstance(value, tuple) or any(any(not r.get(n) for n in task.fields) for r in value):
            raise RuntimeError("every predicted relation needs every field")


__all__ = [
    "PUNCTUATION_FOLD",
    "SPAN_KINDS",
    "Validation",
    "check_prediction",
    "check_reference",
    "prediction_cap",
    "validate",
    "verbatim",
]
