"""OpenAI Responses API with a strict JSON-schema text format."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..protocols import Completion
from .base import MAX_OUTPUT_TOKENS, TIMEOUT_SECONDS, enum_name, fingerprint, usage_int, with_retries


class OpenAIModel:
    """An OpenAI model as a :class:`~codercommittee.protocols.Model`.

    ``client`` is any object with ``responses.create(**arguments)``; without one the ``openai``
    package is imported on first use (``pip install "codercommittee[openai]"``). Nothing is stored
    on the provider's side (``store=False``).
    """

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float | None = None,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(timeout=self.timeout, max_retries=0)
        return self._client

    def describe(self) -> dict[str, Any]:
        """The settings that make this adapter's requests, for a run's provenance."""
        return {
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }

    def arguments(self, prompt: str, schema: Mapping[str, Any], system: str) -> dict[str, Any]:
        """The request arguments, as sent (``temperature`` only when one was given)."""
        arguments: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": prompt,
            "text": {"format": {"type": "json_schema", "name": "annotation", "strict": True, "schema": dict(schema)}},
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.temperature is not None:
            arguments["temperature"] = self.temperature
        return arguments

    def complete(self, prompt: str, schema: Mapping[str, Any], *, system: str) -> Completion:
        """One request inside the transport block."""
        arguments = self.arguments(prompt, schema, system)
        identity = {
            "provider": self.provider,
            "requested_model": self.model,
            "request_fingerprint": fingerprint(self.provider, arguments),
        }

        def once() -> Completion:
            result = self._get_client().responses.create(**arguments)
            text = getattr(result, "output_text", None) or ""
            refusal = next(
                (
                    getattr(content, "refusal", None)
                    for output in getattr(result, "output", None) or []
                    for content in getattr(output, "content", None) or []
                    if getattr(content, "type", None) == "refusal"
                ),
                None,
            )
            status = enum_name(getattr(result, "status", None))
            usage = getattr(result, "usage", None)
            found = {
                **identity,
                "returned_model": getattr(result, "model", None),
                "provider_response_id": getattr(result, "id", None),
                "finish_reason": status,
                "tokens_in": usage_int(getattr(usage, "input_tokens", None)),
                "tokens_out": usage_int(getattr(usage, "output_tokens", None)),
            }
            if refusal:
                return Completion(text, found, "provider_refusal", True)
            incomplete = getattr(result, "incomplete_details", None)
            if (status or "").lower() != "completed" or incomplete:
                reason = enum_name(getattr(incomplete, "reason", None))
                return Completion(
                    text,
                    found,
                    f"provider_incomplete_finish:{status or 'missing'}" + (f":{reason}" if reason else ""),
                    reason == "content_filter",
                )
            return Completion(text, found)

        return with_retries(once)


__all__ = ["OpenAIModel"]
