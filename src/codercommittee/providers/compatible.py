"""OpenAI-compatible chat endpoints (vLLM, llama.cpp servers, gateways) with a JSON schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..protocols import Completion
from .base import (
    MAX_OUTPUT_TOKENS,
    TIMEOUT_SECONDS,
    enum_name,
    fingerprint,
    public_url,
    url_password,
    usage_int,
    with_retries,
)


class OpenAICompatibleModel:
    """A model behind an OpenAI-compatible ``/v1/chat/completions`` endpoint.

    ``base_url`` names the server (for example ``http://localhost:8000/v1`` for vLLM); ``api_key``
    is whatever the server expects (vLLM accepts any string). ``client`` is any object with
    ``chat.completions.create(**arguments)``; without one the ``openai`` package is imported on
    first use. The schema is sent as a strict ``json_schema`` response format, which vLLM and
    most gateways honour; servers that ignore it still have their replies validated here.
    """

    provider = "openai_compatible"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str = "none",
        client: Any = None,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float | None = None,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=0)
        return self._client

    def describe(self) -> dict[str, Any]:
        """The settings that make this adapter's requests, for a run's provenance (the URL without credentials)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": public_url(self.base_url),
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }

    def arguments(self, prompt: str, schema: Mapping[str, Any], system: str) -> dict[str, Any]:
        """The request arguments, as sent."""
        arguments: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "annotation", "strict": True, "schema": dict(schema)},
            },
            "max_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            arguments["temperature"] = self.temperature
        return arguments

    def complete(self, prompt: str, schema: Mapping[str, Any], *, system: str) -> Completion:
        """One request inside the transport block."""
        arguments = self.arguments(prompt, schema, system)
        base_url = public_url(self.base_url)
        identity = {
            "provider": self.provider,
            "base_url": base_url,
            "requested_model": self.model,
            "request_fingerprint": fingerprint(self.provider, {**arguments, "base_url": base_url}),
        }

        def once() -> Completion:
            result = self._get_client().chat.completions.create(**arguments)
            choices = list(getattr(result, "choices", None) or [])
            choice = choices[0] if choices else None
            message = getattr(choice, "message", None)
            text = getattr(message, "content", None) or ""
            refusal = getattr(message, "refusal", None)
            finish = enum_name(getattr(choice, "finish_reason", None)) if choice else None
            usage = getattr(result, "usage", None)
            found = {
                **identity,
                "returned_model": getattr(result, "model", None),
                "provider_response_id": getattr(result, "id", None),
                "finish_reason": finish,
                "tokens_in": usage_int(getattr(usage, "prompt_tokens", None)),
                "tokens_out": usage_int(getattr(usage, "completion_tokens", None)),
            }
            if refusal:
                return Completion(text, found, "provider_refusal", True)
            if finish != "stop":
                return Completion(
                    text, found, f"provider_incomplete_finish:{finish or 'missing'}", finish == "content_filter"
                )
            return Completion(text, found)

        return with_retries(once, secrets=(self.api_key, *url_password(self.base_url)))


__all__ = ["OpenAICompatibleModel"]
