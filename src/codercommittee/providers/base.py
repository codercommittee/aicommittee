"""What every provider adapter shares: the transport block, retry classification, identities.

An adapter makes at most three transport attempts per request. Retryability is read from typed
or status metadata, never from message text: HTTP 408, 409, 425, 429, 500, 502, 503, 504 and 529,
timeouts, connection errors and the SDKs' transient exception classes. After the block the
adapter returns a Completion whose ``error`` names the last failure; it never raises for
transport. A refusal on policy grounds is a Completion with ``refused`` set.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..protocols import Completion
from ..strictjson import canonical_json

MAX_ATTEMPTS = 3
TRANSPORT_BUDGET: ContextVar[int | None] = ContextVar("codercommittee_transport_budget", default=None)
"""How many transport attempts the current coder call may still make; ``collect`` sets it per call.

A context variable: it reaches the calling thread only. A coder that runs its model in a worker
thread copies the context (``contextvars.copy_context().run``) or spends up to three attempts there.
"""
TIMEOUT_SECONDS = 120
MAX_OUTPUT_TOKENS = 4096
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
TRANSIENT_CLASSES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "ServiceUnavailable",
        "ServiceUnavailableError",
        "TooManyRequests",
        "DeadlineExceeded",
        "TimeoutException",
        "TransportError",
    }
)


def status_code_of(exc: Exception) -> int | None:
    """An HTTP status code carried by the exception or its response, when there is one."""
    candidates: list[Any] = [
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ]
    for candidate in candidates:
        if callable(candidate):
            try:
                candidate = candidate()
            except TypeError:
                continue
        value: Any = getattr(candidate, "value", candidate)
        if value is None:
            continue
        try:
            code = int(value)
        except (TypeError, ValueError):
            continue
        if 100 <= code <= 599:
            return code
    return None


def retryable(exc: Exception) -> bool:
    """Transport retryability from typed or status metadata, never from message text."""
    code = status_code_of(exc)
    if code is not None:
        return code in RETRYABLE_STATUS
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    return bool({kind.__name__ for kind in type(exc).mro()} & TRANSIENT_CLASSES)


def retry_after_seconds(exc: Exception) -> float | None:
    """A Retry-After header's delay in seconds (at most 60), when the failure carries one."""
    headers: Any = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    value: Any = headers.get("retry-after") if hasattr(headers, "get") else None
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(value))
            if target.tzinfo is None:
                return None
            seconds = (target - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if 0.0 <= seconds <= 60.0 else None


def public_url(url: object) -> str | None:
    """The URL without its userinfo: a gateway password never reaches an identity or a saved file."""
    if url is None:
        return None
    url = str(url)
    if "@" not in url:
        return url
    parts = urlsplit(url)
    if not parts.netloc:
        # No scheme: ``user:pw@host/v1``; drop everything up to the authority's last ``@``.
        return re.sub(r"^[^/?#]*@", "", url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, parts.query, parts.fragment))


_USERINFO = re.compile(r"(://|:\\/\\/)[^/\\\s?#]+@")


def known_secrets() -> tuple[str, ...]:
    """The API keys the environment holds, so an error that echoes one never reaches a saved file."""
    return tuple(value for name, value in os.environ.items() if name.upper().endswith("API_KEY") and len(value) >= 8)


def scrub_credentials(text: str, secrets: Sequence[str] = ()) -> str:
    """The text with every URL's ``user:password@`` removed and every given or environment secret redacted."""
    text = _USERINFO.sub(r"\1", text)
    for secret in (*secrets, *known_secrets()):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def url_password(url: object) -> tuple[str, ...]:
    """The password a URL's userinfo carries, if any, so an error echoing it bare can be redacted."""
    if url is None:
        return ()
    netloc = urlsplit(str(url)).netloc
    if "@" not in netloc:
        return ()
    userinfo = netloc.rpartition("@")[0]
    password = userinfo.partition(":")[2]
    return (password,) if password else ()


def fingerprint(provider: str, arguments: Mapping[str, Any]) -> str:
    """The sha256 of the canonical request: provider and every argument sent."""
    return hashlib.sha256(canonical_json({"provider": provider, "arguments": arguments}).encode("utf-8")).hexdigest()


_sleep: Callable[[float], None] = time.sleep


def with_retries(
    call: Callable[[], Completion], *, attempts: int = MAX_ATTEMPTS, secrets: Sequence[str] = ()
) -> Completion:
    """Run ``call`` inside the transport block; the last failure becomes the Completion's error.

    The block makes at most ``attempts`` attempts, and never more than the budget ``collect`` left
    for the current call (``TRANSPORT_BUDGET``), as the study's runner capped every call by the
    attempts the item's earlier rounds had spent.
    """
    budget = TRANSPORT_BUDGET.get()
    if budget is not None:
        attempts = max(1, min(attempts, budget))
    last: Exception | None = None
    made = 0
    for attempt in range(1, attempts + 1):
        made = attempt
        try:
            completion = call()
        except Exception as exc:
            last = exc
            if attempt == attempts or not retryable(exc):
                break
            delay = retry_after_seconds(exc)
            _sleep((2 ** (attempt - 1)) * (0.75 + 0.5 * random.random()) if delay is None else delay)
            continue
        identity = {**completion.identity, "attempts": attempt}
        return Completion(completion.raw, identity, completion.error, completion.refused)
    error = f"{type(last).__name__}: {scrub_credentials(str(last), secrets)}"
    return Completion("", {"attempts": made}, error, False)


def usage_int(value: Any) -> int | None:
    """A non-negative integer usage count, or None."""
    return value if not isinstance(value, bool) and isinstance(value, int) and value >= 0 else None


def enum_name(value: Any) -> str | None:
    """The name of an enum-like value, else its last dotted part."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    rendered = str(value)
    return rendered.rsplit(".", 1)[-1] if rendered else None


__all__ = [
    "MAX_ATTEMPTS",
    "MAX_OUTPUT_TOKENS",
    "RETRYABLE_STATUS",
    "TIMEOUT_SECONDS",
    "TRANSPORT_BUDGET",
    "enum_name",
    "fingerprint",
    "known_secrets",
    "public_url",
    "retry_after_seconds",
    "retryable",
    "scrub_credentials",
    "status_code_of",
    "url_password",
    "usage_int",
    "with_retries",
]
