"""Anthropic Messages API with a JSON-schema output format and thinking pinned off."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..protocols import Completion
from .base import MAX_OUTPUT_TOKENS, TIMEOUT_SECONDS, enum_name, fingerprint, usage_int, with_retries


class AnthropicModel:
    """A Claude model as a :class:`~codercommittee.protocols.Model`.

    ``client`` is any object with ``messages.create(**arguments)``; without one the ``anthropic``
    package is imported on first use (``pip install "codercommittee[anthropic]"``). ``thinking``
    is sent as given (default: disabled); models that cannot switch thinking off take their own
    closest setting, for example ``{"type": "adaptive"}`` with ``effort="low"``.
    """

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        thinking: Mapping[str, Any] | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.thinking = dict(thinking) if thinking is not None else {"type": "disabled"}
        self.effort = effort
        self.temperature = temperature
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=self.timeout, max_retries=0)
        return self._client

    def describe(self) -> dict[str, Any]:
        """The settings that make this adapter's requests, for a run's provenance."""
        return {
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "thinking": dict(self.thinking),
            "effort": self.effort,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }

    def arguments(self, prompt: str, schema: Mapping[str, Any], system: str) -> dict[str, Any]:
        """The request arguments, as sent (``temperature`` only when one was given)."""
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": dict(schema)}}
        if self.effort:
            output_config["effort"] = self.effort
        arguments: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": dict(self.thinking),
            "output_config": output_config,
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
            result = self._get_client().messages.create(**arguments)
            content = list(getattr(result, "content", None) or [])
            text = "".join(getattr(block, "text", "") for block in content if getattr(block, "type", None) == "text")
            refused = any(getattr(block, "type", None) == "refusal" for block in content)
            stop = enum_name(getattr(result, "stop_reason", None))
            usage = getattr(result, "usage", None)
            found = {
                **identity,
                "returned_model": getattr(result, "model", None),
                "provider_response_id": getattr(result, "id", None),
                "finish_reason": stop,
                "tokens_in": usage_int(getattr(usage, "input_tokens", None)),
                "tokens_out": usage_int(getattr(usage, "output_tokens", None)),
            }
            if refused:
                return Completion(text, found, "provider_refusal", True)
            if stop != "end_turn":
                return Completion(text, found, f"provider_incomplete_finish:{stop or 'missing'}", stop == "refusal")
            return Completion(text, found)

        return with_retries(once)


__all__ = ["AnthropicModel"]
