"""Strict JSON: no duplicate keys, no NaN or Infinity, no lone UTF-16 surrogates.

Every coder's raw reply and every stored record is read through this parser, so a stored
parsed value is never authoritative.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


class StrictJSONError(ValueError):
    """The text is not strict JSON."""


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_surrogates(value: Any, path: str = "$") -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise StrictJSONError(f"unpaired UTF-16 surrogate is forbidden at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _reject_surrogates(key, f"{path}.<key>")
            _reject_surrogates(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_surrogates(child, f"{path}[{index}]")


def strict_json_loads(text: str) -> Any:
    """Parse strict JSON or raise :class:`StrictJSONError`."""
    if not isinstance(text, str) or not text.strip():
        raise StrictJSONError("JSON text is empty")
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
        _reject_surrogates(value)
    except StrictJSONError:
        raise
    except (ValueError, RecursionError) as exc:
        # JSONDecodeError, the integer digit limit, nesting beyond the recursion limit.
        raise StrictJSONError(f"{type(exc).__name__}: {exc}") from exc
    return value


def jsonl_lines(text: str) -> Iterator[tuple[int, str]]:
    """The non-blank lines of a JSON Lines text with their 1-based numbers, split on newlines alone.

    ``str.splitlines`` would also split on U+0085, U+2028, U+2029 and the other separators a JSON
    string may hold raw; records are separated by the newline character, a trailing carriage
    return is dropped.
    """
    for number, line in enumerate(text.split("\n"), 1):
        if line.endswith("\r"):
            line = line[:-1]
        if line.strip():
            yield number, line


def canonical_json(value: Any) -> str:
    """The one serialization every hash uses: sorted keys, no spaces, raw unicode, no NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


__all__ = ["StrictJSONError", "canonical_json", "jsonl_lines", "strict_json_loads"]
