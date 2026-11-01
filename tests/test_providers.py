"""The provider adapters under fake SDK clients: request shape, refusal mapping, the transport block."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from codercommittee.coders import ModelCoder
from codercommittee.data import Item, Task
from codercommittee.protocols import Completion
from codercommittee.providers import AnthropicModel, GoogleModel, OpenAICompatibleModel, OpenAIModel
from codercommittee.providers.base import fingerprint, retryable, with_retries

TASK = Task("label", {"yes": "the text says yes"})
ITEM = Item("a", "yes indeed")
SCHEMA = TASK.schema()


class Fake:
    """A client whose one method returns queued replies or raises queued exceptions."""

    def __init__(self, path: str, replies: list[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies = list(replies)
        node: Any = self
        parts = path.split(".")
        for part in parts[:-1]:
            child = SimpleNamespace()
            setattr(node, part, child)
            node = child
        setattr(node, parts[-1], self._call)

    def _call(self, **arguments: Any) -> Any:
        self.calls.append(arguments)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"http {status_code}")
        self.status_code = status_code


def anthropic_reply(text: str = '{"yes": true}', stop: str = "end_turn", refusal: bool = False) -> SimpleNamespace:
    content = [SimpleNamespace(type="text", text=text)]
    if refusal:
        content.append(SimpleNamespace(type="refusal", text="no"))
    return SimpleNamespace(
        content=content,
        stop_reason=stop,
        model="claude-x",
        id="msg_1",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def openai_reply(text: str = '{"yes": true}', status: str = "completed", refusal: str | None = None) -> SimpleNamespace:
    content = [SimpleNamespace(type="refusal", refusal=refusal)] if refusal else []
    return SimpleNamespace(
        output_text=text,
        output=[SimpleNamespace(content=content)],
        status=status,
        model="gpt-x",
        id="resp_1",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        incomplete_details=None,
    )


def google_reply(text: str = '{"yes": true}', finish: str = "STOP", blocked: str | None = None) -> SimpleNamespace:
    candidate = SimpleNamespace(
        finish_reason=finish, content=SimpleNamespace(parts=[SimpleNamespace(text=text, thought=False)])
    )
    return SimpleNamespace(
        candidates=[candidate],
        model_version="gemini-x",
        response_id="r1",
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        prompt_feedback=SimpleNamespace(block_reason=blocked),
    )


def compatible_reply(text: str = '{"yes": true}', finish: str = "stop", refusal: str | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=text, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        model="llama-x",
        id="c1",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def test_anthropic_request_shape_and_outcomes() -> None:
    client = Fake(
        "messages.create", [anthropic_reply(), anthropic_reply(refusal=True), anthropic_reply(stop="max_tokens")]
    )
    model = AnthropicModel("claude-x", client=client, max_output_tokens=99)
    done = model.complete("p", SCHEMA, system="s")
    assert done.raw == '{"yes": true}' and done.error is None and not done.refused
    assert done.identity["returned_model"] == "claude-x" and done.identity["attempts"] == 1
    assert client.calls[0] == {
        "model": "claude-x",
        "max_tokens": 99,
        "system": "s",
        "messages": [{"role": "user", "content": "p"}],
        "thinking": {"type": "disabled"},
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
    }
    assert model.complete("p", SCHEMA, system="s").refused
    incomplete = model.complete("p", SCHEMA, system="s")
    assert incomplete.error == "provider_incomplete_finish:max_tokens" and not incomplete.refused
    adaptive = AnthropicModel("claude-y", client=client, thinking={"type": "adaptive"}, effort="low")
    assert adaptive.arguments("p", SCHEMA, "s")["output_config"]["effort"] == "low"


def test_openai_request_shape_and_outcomes() -> None:
    client = Fake("responses.create", [openai_reply(), openai_reply(refusal="no"), openai_reply(status="incomplete")])
    model = OpenAIModel("gpt-x", client=client)
    done = model.complete("p", SCHEMA, system="s")
    assert done.raw == '{"yes": true}' and done.error is None
    assert client.calls[0]["text"]["format"] == {
        "type": "json_schema",
        "name": "annotation",
        "strict": True,
        "schema": SCHEMA,
    }
    assert (
        client.calls[0]["store"] is False and client.calls[0]["instructions"] == "s" and client.calls[0]["input"] == "p"
    )
    assert model.complete("p", SCHEMA, system="s").refused
    assert model.complete("p", SCHEMA, system="s").error == "provider_incomplete_finish:incomplete"


def test_google_request_shape_and_outcomes() -> None:
    client = Fake(
        "models.generate_content",
        [
            google_reply(),
            google_reply(finish="SAFETY"),
            google_reply(finish="MAX_TOKENS"),
            google_reply(blocked="BLOCKLIST"),
        ],
    )
    model = GoogleModel("gemini-x", client=client)
    done = model.complete("p", SCHEMA, system="s")
    assert done.raw == '{"yes": true}' and done.error is None and done.identity["returned_model"] == "gemini-x"
    config = client.calls[0]["config"]
    assert config["response_json_schema"] == SCHEMA and config["thinking_config"] == {"thinking_level": "low"}
    assert config["system_instruction"] == "s" and client.calls[0]["contents"] == "p"
    assert model.complete("p", SCHEMA, system="s").refused
    assert model.complete("p", SCHEMA, system="s").error == "provider_incomplete_finish:MAX_TOKENS"
    assert model.complete("p", SCHEMA, system="s").refused


def test_compatible_request_shape_and_outcomes() -> None:
    client = Fake(
        "chat.completions.create",
        [compatible_reply(), compatible_reply(refusal="no"), compatible_reply(finish="length")],
    )
    model = OpenAICompatibleModel("llama-x", base_url="http://localhost:8000/v1", client=client, temperature=0.0)
    done = model.complete("p", SCHEMA, system="s")
    assert (
        done.raw == '{"yes": true}' and done.error is None and done.identity["base_url"] == "http://localhost:8000/v1"
    )
    sent = client.calls[0]
    assert sent["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}]
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA and sent["temperature"] == 0.0
    assert model.complete("p", SCHEMA, system="s").refused
    assert model.complete("p", SCHEMA, system="s").error == "provider_incomplete_finish:length"


def test_transport_block_retries_transient_failures_and_stops_on_fatal_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    import codercommittee.providers.base as base

    slept: list[float] = []
    monkeypatch.setattr(base, "_sleep", slept.append)
    client = Fake("responses.create", [HttpError(429), HttpError(503), openai_reply()])
    done = OpenAIModel("gpt-x", client=client).complete("p", SCHEMA, system="s")
    assert done.error is None and done.identity["attempts"] == 3 and len(slept) == 2
    fatal = Fake("responses.create", [HttpError(400), openai_reply()])
    failed = OpenAIModel("gpt-x", client=fatal).complete("p", SCHEMA, system="s")
    assert failed.error == "HttpError: http 400" and failed.identity["attempts"] == 1 and len(fatal.replies) == 1
    exhausted = Fake("responses.create", [HttpError(500)] * 3)
    assert OpenAIModel("gpt-x", client=exhausted).complete("p", SCHEMA, system="s").identity["attempts"] == 3
    assert retryable(HttpError(429)) and not retryable(HttpError(404)) and retryable(TimeoutError())
    assert retryable(type("RateLimitError", (Exception,), {})()) and not retryable(ValueError("x"))
    assert with_retries(lambda: Completion("ok")).raw == "ok"
    assert fingerprint("p", {"a": 1}) == fingerprint("p", {"a": 1}) != fingerprint("p", {"a": 2})


def test_model_coder_turns_completions_into_answers() -> None:
    client = Fake(
        "responses.create",
        [
            openai_reply(),
            openai_reply(refusal="no"),
            openai_reply(text="{bad"),
            openai_reply(text='{"yes": "maybe"}'),
            HttpError(400),
        ],
    )
    coder = ModelCoder("gpt", OpenAIModel("gpt-x", client=client))
    ok = coder.answer(ITEM, TASK, 1)
    assert ok.ok and ok.value is True and ok.identity["coder"] == "gpt" and ok.identity["returned_model"] == "gpt-x"
    assert coder.answer(ITEM, TASK, 1).status == "refused"
    assert coder.answer(ITEM, TASK, 1).reason.startswith("strict_json:")  # type: ignore[union-attr]
    assert coder.answer(ITEM, TASK, 1).reason == "label_must_be_boolean"
    failed = coder.answer(ITEM, TASK, 1)
    assert failed.status == "invalid" and failed.reason == "HttpError: http 400"
    sent = client.calls[0]
    assert json.loads(json.dumps(sent["text"]["format"]["schema"])) == SCHEMA and sent["input"] == "yes indeed"


def test_no_real_sdk_client_is_constructed_in_tests() -> None:
    """The conftest guard replaces every SDK constructor (without the SDK the import itself fails); the
    transport block turns that into a Completion error, never a call."""
    models = (AnthropicModel("m"), OpenAIModel("m"), GoogleModel("m"), OpenAICompatibleModel("m", base_url="http://x"))
    for model in models:
        completion = model.complete("p", SCHEMA, system="s")
        assert completion.error is not None
        assert "no provider client in tests" in completion.error or "ModuleNotFoundError" in completion.error


def test_gateway_credentials_never_reach_the_identity() -> None:
    from codercommittee.providers.base import public_url

    secret = "SYNTHETIC_PASSWORD"
    with_secret = Fake("chat.completions.create", [compatible_reply()])
    without = Fake("chat.completions.create", [compatible_reply()])
    guarded = OpenAICompatibleModel("m", base_url=f"https://audit:{secret}@gateway.invalid:8443/v1", client=with_secret)
    plain = OpenAICompatibleModel("m", base_url="https://gateway.invalid:8443/v1", client=without)
    done = guarded.complete("p", SCHEMA, system="s")
    assert done.identity["base_url"] == "https://gateway.invalid:8443/v1"
    assert secret not in json.dumps(done.identity)
    assert (
        done.identity["request_fingerprint"] == plain.complete("p", SCHEMA, system="s").identity["request_fingerprint"]
    )
    assert public_url(None) is None and public_url("http://localhost:8000/v1") == "http://localhost:8000/v1"
    assert public_url("http://user:pw@[::1]:8000/v1?x=1") == "http://[::1]:8000/v1?x=1"
    assert public_url("http://user@host/v1") == "http://host/v1"
    assert public_url("http://u:p@Host.Example:abc/v1") == "http://Host.Example:abc/v1"
    assert public_url("user:pw@host/v1") == "host/v1"
    assert public_url("http://host/v1?token=a@b") == "http://host/v1?token=a@b"


def test_policy_finishes_are_refusals_as_the_study_classified_them() -> None:
    anthropic = AnthropicModel("claude-x", client=Fake("messages.create", [anthropic_reply(stop="refusal")]))
    refused = anthropic.complete("p", SCHEMA, system="s")
    assert refused.refused and refused.error == "provider_incomplete_finish:refusal"
    filtered = openai_reply(status="incomplete")
    filtered.incomplete_details = SimpleNamespace(reason="content_filter")
    openai = OpenAIModel("gpt-x", client=Fake("responses.create", [filtered, openai_reply(status="incomplete")]))
    refused = openai.complete("p", SCHEMA, system="s")
    assert refused.refused and refused.error == "provider_incomplete_finish:incomplete:content_filter"
    assert not openai.complete("p", SCHEMA, system="s").refused
    compatible = OpenAICompatibleModel(
        "llama-x",
        base_url="http://localhost:8000/v1",
        client=Fake("chat.completions.create", [compatible_reply(finish="content_filter")]),
    )
    refused = compatible.complete("p", SCHEMA, system="s")
    assert refused.refused and refused.error == "provider_incomplete_finish:content_filter"


def test_adapters_and_the_coder_describe_their_settings() -> None:
    anthropic = AnthropicModel("claude-x", thinking={"type": "adaptive"}, effort="low", max_output_tokens=99)
    assert anthropic.describe() == {
        "provider": "anthropic",
        "model": "claude-x",
        "max_output_tokens": 99,
        "thinking": {"type": "adaptive"},
        "effort": "low",
        "temperature": None,
        "timeout": 120,
    }
    assert OpenAIModel("gpt-x").describe()["max_output_tokens"] == 4096
    assert GoogleModel("gemini-x").describe()["thinking_level"] == "low"
    compatible = OpenAICompatibleModel("m", base_url="https://u:pw@gateway.invalid/v1", temperature=0.5)
    assert (
        compatible.describe()["base_url"] == "https://gateway.invalid/v1"
        and compatible.describe()["temperature"] == 0.5
    )
    coder = ModelCoder("gpt", OpenAIModel("gpt-x"), system="Answer at random.")
    record = coder.describe()
    assert record["kind"] == "model" and record["model"] == "gpt-x" and len(record["system_sha256"]) == 16
    assert record["system_sha256"] != ModelCoder("gpt", OpenAIModel("gpt-x")).describe()["system_sha256"]

    class UrlModel:
        provider = "custom"
        model = "m"

        class Url:
            def __str__(self) -> str:
                return "https://u:pw@h/v1"

        base_url = Url()

        def complete(self, prompt: str, schema: Any, *, system: str) -> Completion:
            return Completion('{"yes": true}')

    assert ModelCoder("c", UrlModel()).describe()["base_url"] == "https://h/v1"
