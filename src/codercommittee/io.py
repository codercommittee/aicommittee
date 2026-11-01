"""Items and results as files: JSON Lines and CSV in, JSON Lines, CSV and JSON out.

An items file holds one record per item with ``id``, ``text`` and an optional ``reference``
encoded by the task's kind: a JSON boolean for ``label``; a JSON list of names for ``labels``;
a JSON list of objects keyed by the task's span fields for ``tuple`` (an empty list is the
negative reference) and ``tuples``. In CSV the ``reference`` cell holds that JSON, and an empty
cell means no reference. Every writer here produces plain files a spreadsheet or another
program reads back; :func:`to_json` is the one encoder for the package's dataclasses.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .arms import ArmPredictions
from .compare import Agreement, Comparison, Pick, Scores
from .data import Item, ItemId, Reference, Relation, Task
from .replay import value_as_stored
from .strictjson import StrictJSONError, jsonl_lines, strict_json_loads
from .validation import check_reference

ITEM_KEYS = ("id", "text", "reference")


def to_json(value: Any) -> Any:
    """A JSON-ready shape for the package's values: dataclasses, relations, sets, numpy numbers."""
    if isinstance(value, Relation):
        return value.as_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_json(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, frozenset | set):
        return sorted(to_json(v) for v in value)
    if isinstance(value, Mapping):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_json(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def dump(value: Any, path: str | Path) -> None:
    """Write a value as indented JSON."""
    Path(path).write_text(json.dumps(to_json(value), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def reference_to_json(reference: Reference | None, task: Task) -> Any:
    """A reference in its file encoding; None stays None."""
    if reference is None:
        return None
    if task.kind == "label":
        return bool(reference)
    if task.kind == "labels":
        return sorted(reference)  # type: ignore[arg-type]
    return [{name: relation.get(name, "") for name in task.names} for relation in reference]  # type: ignore[union-attr]


def reference_from_json(value: Any, task: Task) -> Reference | None:
    """A reference from its file encoding, checked against the task's kind."""
    if value is None:
        return None
    if task.kind == "label":
        if not isinstance(value, bool):
            raise ValueError("a label reference is a JSON boolean")
        return value
    if task.kind == "labels":
        if not isinstance(value, list) or any(not isinstance(name, str) for name in value):
            raise ValueError("a labels reference is a JSON list of label names")
        reference: Reference = frozenset(value)
    else:
        if not isinstance(value, list) or any(not isinstance(entry, dict) for entry in value):
            raise ValueError("a relation reference is a JSON list of objects keyed by the task's span fields")
        relations = []
        for entry in value:
            unknown = sorted(set(entry) - set(task.names))
            if unknown:
                raise ValueError(f"reference relation has fields outside the task's: {unknown}")
            if any(not isinstance(span, str) for span in entry.values()):
                raise ValueError("reference spans are strings")
            relations.append(Relation(**{name: entry.get(name, "") for name in task.names}))
        reference = tuple(relations)
    check_reference(task, reference)
    return reference


def item_to_json(item: Item, task: Task) -> dict[str, Any]:
    """An item as a file record."""
    return {"id": item.id, "text": item.text, "reference": reference_to_json(item.reference, task)}


def item_from_json(record: Any, task: Task, *, int_ids: bool = False) -> Item:
    """An item from a file record, its reference checked against the task."""
    if not isinstance(record, dict):
        raise ValueError("an item record is an object with id, text and an optional reference")
    unknown = sorted(set(record) - set(ITEM_KEYS))
    if unknown:
        raise ValueError(f"unknown item keys {unknown}; the keys are {list(ITEM_KEYS)}")
    if "id" not in record or "text" not in record:
        raise ValueError("an item needs an id and a text")
    item_id: Any = record["id"]
    if int_ids and isinstance(item_id, str) and item_id.lstrip("-").isdigit():
        item_id = int(item_id)
    if isinstance(item_id, bool) or not isinstance(item_id, str | int):
        raise ValueError("an item id is a string or an integer")
    if not isinstance(record["text"], str):
        raise ValueError("an item text is a string")
    return Item(item_id, record["text"], reference_from_json(record.get("reference"), task))


def _records_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {key: value for key, value in row.items() if key in ITEM_KEYS and value is not None}
        reference = record.get("reference")
        if reference is None or reference == "":
            record.pop("reference", None)
        else:
            try:
                record["reference"] = strict_json_loads(reference)
            except StrictJSONError as exc:
                raise ValueError(f"reference cell is not JSON: {exc}") from exc
        records.append(record)
    return records


def read_items(path: str | Path, task: Task, *, int_ids: bool = False) -> list[Item]:
    """Items from a ``.jsonl``, ``.json`` (array) or ``.csv`` file (UTF-8, a BOM allowed); errors name the line."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        records = _records_from_csv(path)
        int_ids = True if int_ids else int_ids
    elif suffix == ".json":
        loaded = strict_json_loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, list):
            raise ValueError(f"{path}: a .json items file holds a JSON array of item objects")
        records = loaded
    else:
        records = []
        for number, line in jsonl_lines(path.read_text(encoding="utf-8-sig")):
            try:
                records.append(strict_json_loads(line))
            except StrictJSONError as exc:
                raise ValueError(f"{path}:{number}: not strict JSON: {exc}") from exc
    items: list[Item] = []
    seen: set[Any] = set()
    for number, record in enumerate(records, 1):
        try:
            item = item_from_json(record, task, int_ids=int_ids)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if item.id in seen:
            raise ValueError(f"{path}:{number}: duplicate item id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise ValueError(f"{path}: no items")
    return items


def write_items(items: Iterable[Item], path: str | Path, task: Task) -> None:
    """Items to ``.jsonl`` or ``.csv`` (the reference cell holds JSON)."""
    path = Path(path)
    records = [item_to_json(item, task) for item in items]
    if path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ITEM_KEYS))
            writer.writeheader()
            for record in records:
                reference = record["reference"]
                writer.writerow(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "reference": "" if reference is None else json.dumps(reference, ensure_ascii=False),
                    }
                )
        return
    _write_jsonl(records, path)


def _write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_json(record), ensure_ascii=False) + "\n")


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path, fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _cell(row.get(field)) for field in fields})


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list | dict | tuple):
        return json.dumps(to_json(value), ensure_ascii=False)
    return value


def write_predictions(predictions: ArmPredictions, task: Task, path: str | Path) -> None:
    """Every arm's predictions as JSON Lines: item, arm, realization, status, value (the stored shape)."""
    records = [
        {
            "item": item,
            "arm": arm,
            "realization": index + 1,
            "status": prediction.status,
            "value": value_as_stored(prediction.value, task) if prediction.ok else None,
        }
        for arm, by_item in predictions.items()
        for item, realizations in by_item.items()
        for index, prediction in enumerate(realizations)
    ]
    _write_jsonl(records, Path(path))


def write_scores(scores: Scores, path: str | Path) -> None:
    """Per-item scores as CSV (item, arm, realization, value) or JSON (with the levels)."""
    path = Path(path)
    rows = [
        {"item": item.id, "arm": arm, "realization": index + 1, "value": s.value}
        for arm, by_item in scores.values.items()
        for item, realizations in zip(scores.items, by_item, strict=True)
        for index, s in enumerate(realizations)
    ]
    if path.suffix.lower() == ".csv":
        _write_csv(rows, path, ("item", "arm", "realization", "value"))
        return
    dump({"levels": scores.levels(), "n_items": len(scores.items), "scores": rows}, path)


def write_levels(scores: Scores, path: str | Path) -> None:
    """The levels and the best single coder as JSON."""
    dump({"levels": scores.levels(), "best_single": scores.best_single(), "n_items": len(scores.items)}, path)


def comparison_rows(table: Comparison) -> list[dict[str, Any]]:
    """The comparison table as one record per arm."""
    rows = []
    for row in table.rows.values():
        gap = row.gap
        rows.append(
            {
                "arm": row.arm,
                "level": row.level_mean,
                "level_sd": row.level_sd,
                "gap": gap.mean if gap else None,
                "gap_sd": gap.sd if gap else None,
                "gap_low": gap.interval[0] if gap and gap.interval else None,
                "gap_high": gap.interval[1] if gap and gap.interval else None,
                "gap_worst": gap.worst if gap else None,
                "premium": row.premium,
                "premium_worst": row.premium_worst,
                **{f"level_run{index + 1}": level for index, level in enumerate(row.levels)},
            }
        )
    return rows


def write_comparison(table: Comparison, path: str | Path) -> None:
    """The comparison table as CSV (one row per arm) or JSON (the whole record, intervals included)."""
    path = Path(path)
    rows = comparison_rows(table)
    if path.suffix.lower() == ".csv":
        _write_csv(rows, path, list(rows[0]) if rows else ["arm"])
        return
    dump(table, path)


def write_agreement(agreement: Agreement, path: str | Path) -> None:
    """The agreement as JSON."""
    dump(agreement, path)


def write_pick(pick: Pick, path: str | Path) -> None:
    """The pick as JSON."""
    dump(pick, path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Strict JSON Lines, for reading the package's own files back."""
    out = []
    for number, line in jsonl_lines(Path(path).read_text(encoding="utf-8")):
        try:
            out.append(strict_json_loads(line))
        except StrictJSONError as exc:
            raise ValueError(f"{path}:{number}: not strict JSON: {exc}") from exc
    return out


ItemIdLike = ItemId

__all__ = [
    "ITEM_KEYS",
    "comparison_rows",
    "dump",
    "item_from_json",
    "item_to_json",
    "read_items",
    "read_jsonl",
    "reference_from_json",
    "reference_to_json",
    "to_json",
    "write_agreement",
    "write_comparison",
    "write_items",
    "write_levels",
    "write_pick",
    "write_predictions",
    "write_scores",
]
