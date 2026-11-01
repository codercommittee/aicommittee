"""Google Gemini through the ``google-genai`` SDK with a JSON response schema."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ..protocols import Completion
from .base import MAX_OUTPUT_TOKENS, TIMEOUT_SECONDS, enum_name, fingerprint, usage_int, with_retries

REFUSAL_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"})


class GoogleModel:
    """A Gemini model as a :class:`~codercommittee.protocols.Model`.

    ``client`` is any object with ``models.generate_content(**arguments)``; without one the
    ``google-genai`` package is imported on first use with ``GEMINI_API_KEY`` from the environment
    (``pip install "codercommittee[google]"``). Thinking runs at the ``low`` level, the lowest a
    Gemini model accepts.
    """

    provider = "google"

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        thinking_level: str = "low",
        temperature: float | None = None,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self.temperature = temperature
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"),
                http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
            )
        return self._client

    def describe(self) -> dict[str, Any]:
        """The settings that make this adapter's requests, for a run's provenance."""
        return {
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "thinking_level": self.thinking_level,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }

    def arguments(self, prompt: str, schema: Mapping[str, Any], system: str) -> dict[str, Any]:
        """The request arguments, as sent (``temperature`` only when one was given)."""
        config: dict[str, Any] = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_json_schema": dict(schema),
            "max_output_tokens": self.max_output_tokens,
            "thinking_config": {"thinking_level": self.thinking_level},
        }
        if self.temperature is not None:
            config["temperature"] = self.temperature
        return {"model": self.model, "contents": prompt, "config": config}

    def complete(self, prompt: str, schema: Mapping[str, Any], *, system: str) -> Completion:
        """One request inside the transport block."""
        arguments = self.arguments(prompt, schema, system)
        identity = {
            "provider": self.provider,
            "requested_model": self.model,
            "request_fingerprint": fingerprint(self.provider, arguments),
        }

        def once() -> Completion:
            result = self._get_client().models.generate_content(**arguments)
            candidates = list(getattr(result, "candidates", None) or [])
            candidate = candidates[0] if candidates else None
            finish = enum_name(getattr(candidate, "finish_reason", None)) if candidate else None
            parts = list(getattr(getattr(candidate, "content", None), "parts", None) or []) if candidate else []
            text = "".join(
                str(getattr(part, "text", None) or "") for part in parts if not getattr(part, "thought", False)
            )
            usage = getattr(result, "usage_metadata", None)
            found = {
                **identity,
                "returned_model": getattr(result, "model_version", None),
                "provider_response_id": getattr(result, "response_id", None),
                "finish_reason": finish,
                "tokens_in": usage_int(getattr(usage, "prompt_token_count", None)),
                "tokens_out": usage_int(getattr(usage, "candidates_token_count", None)),
            }
            blocked = enum_name(getattr(getattr(result, "prompt_feedback", None), "block_reason", None))
            if blocked or finish in REFUSAL_FINISH_REASONS:
                return Completion(text, found, "provider_refusal", True)
            if finish != "STOP":
                return Completion(text, found, f"provider_incomplete_finish:{finish or 'missing'}")
            return Completion(text, found)

        return with_retries(once)


__all__ = ["REFUSAL_FINISH_REASONS", "GoogleModel"]
