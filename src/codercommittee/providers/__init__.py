"""Thin model adapters: Anthropic, OpenAI, Google, and OpenAI-compatible endpoints such as vLLM.

Each is a :class:`~codercommittee.protocols.Model`: ``complete(prompt, schema, system=...)`` returns
a :class:`~codercommittee.protocols.Completion`. The SDKs are optional extras and are imported on
first use; every adapter also takes an injected ``client`` so that tests never touch a network.
"""

from .anthropic import AnthropicModel
from .compatible import OpenAICompatibleModel
from .google import GoogleModel
from .openai import OpenAIModel

__all__ = ["AnthropicModel", "GoogleModel", "OpenAICompatibleModel", "OpenAIModel"]
