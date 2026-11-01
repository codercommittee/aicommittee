# Providers

A provider adapter is a `Model`: one method, `complete(prompt, schema, *, system)`, that returns a `Completion` with the raw reply, the identity of the request, an error when there was one, and whether the model refused. `ModelCoder` renders the task's prompt and passes the task's schema; `LLMReferee` passes its own prompt and the decision schema. The adapters are thin on purpose: structured output, three transport attempts, a request fingerprint, nothing else.

## The four adapters

| adapter | API | extra | key |
|---|---|---|---|
| `AnthropicModel(model)` | Messages API with a JSON-schema output format | `anthropic` | `ANTHROPIC_API_KEY` |
| `OpenAIModel(model)` | Responses API with a strict JSON-schema text format; nothing is stored on the provider's side | `openai` | `OPENAI_API_KEY` |
| `GoogleModel(model)` | `google-genai` with a JSON response schema | `google` | `GEMINI_API_KEY` |
| `OpenAICompatibleModel(model, base_url=...)` | any `/v1/chat/completions` server with a `json_schema` response format: vLLM, llama.cpp, Ollama, gateways | `openai` | `api_key`, default `"none"` |

Every adapter takes the same three settings: `max_output_tokens` (default 4,096), `temperature` (default none, so the provider's own; sent only when given) and `timeout` in seconds (default 120). `AnthropicModel` also takes `thinking` (disabled by default; a model that cannot switch thinking off takes its closest setting, for example `thinking={"type": "adaptive"}, effort="low"`) and `GoogleModel` a `thinking_level` (default `low`). `describe()` returns these settings as a record; a run stores it, and a resume under other settings is refused.

The SDKs are optional extras, imported on first use. Building an adapter calls nothing:

```python
from codercommittee import Committee, LLMReferee, ModelCoder, OpenAIEmbedder
from codercommittee.providers import AnthropicModel, GoogleModel, OpenAIModel

coders = [
    ModelCoder("claude", AnthropicModel("claude-sonnet-5")),
    ModelCoder("gpt", OpenAIModel("gpt-5.6-terra", temperature=0.0)),
    ModelCoder("gemini", GoogleModel("gemini-3.8-flash")),
]
committee = Committee(coders, OpenAIEmbedder(), LLMReferee(AnthropicModel("claude-opus-5-5")))
record = coders[1].describe()
print({key: record[key] for key in ("kind", "provider", "model", "max_output_tokens", "temperature", "timeout")})
```

```text
{'kind': 'model', 'provider': 'openai', 'model': 'gpt-5.6-terra', 'max_output_tokens': 4096, 'temperature': 0.0, 'timeout': 120}
```

`committee.code(items, task)` would call the three providers. Keys come from the environment as the SDKs expect them; `GoogleModel` reads `GEMINI_API_KEY` itself. Every adapter also takes a `client=` argument, any object with the SDK's method (`messages.create`, `responses.create`, `models.generate_content`, `chat.completions.create`), which is how the package's own tests run without a network.

On the command line a coder is `NAME=anthropic:MODEL`, `NAME=openai:MODEL`, `NAME=google:MODEL` or `NAME=compatible:MODEL@BASE_URL`, and `--max-output-tokens`, `--temperature` and `--timeout` apply to every model built from a spec, the referee included.

## OpenAI-compatible servers: vLLM and Ollama

`OpenAICompatibleModel` speaks to any server with the OpenAI chat completions shape. `base_url` names the server; `api_key` is whatever it expects (vLLM and Ollama accept any string). The schema is sent as a strict `json_schema` response format, which vLLM and Ollama honour; a server that ignores it still has its replies validated here, so a malformed reply is an invalid answer, not a crash.

```python
from codercommittee import ModelCoder, OpenAICompatibleEmbedder
from codercommittee.providers import OpenAICompatibleModel

vllm = ModelCoder(
    "llama", OpenAICompatibleModel("meta-llama/Llama-3.3-70B-Instruct", base_url="http://localhost:8000/v1")
)
ollama = ModelCoder(
    "qwen", OpenAICompatibleModel("qwen3:30b-a3b", base_url="http://localhost:11434/v1", max_output_tokens=16384)
)
embedder = OpenAICompatibleEmbedder("http://localhost:11434/v1", "nomic-embed-text")
print(ollama.describe()["provider"], ollama.describe()["base_url"])
```

```text
openai_compatible http://localhost:11434/v1
```

A model that thinks before it answers spends output tokens on the thinking. The default budget of 4,096 tokens is enough for an answer, often not for the referee's prompt with its listed pairs; give such a model `max_output_tokens=16384` (`--max-output-tokens 16384` on the command line). The [Ollama guide](ollama.md) runs a whole committee this way.

A password inside the URL (`http://user:secret@gateway/v1`) never reaches a saved file or an error text: identities carry the URL without its credentials, and error messages are scrubbed of URL credentials and of any API key the environment holds.

## The transport block

Every request runs inside one block of at most three transport attempts. Whether a failure is retried is read from typed or status metadata, never from the message text: HTTP 408, 409, 425, 429, 500, 502, 503, 504 and 529, timeouts, connection errors, and the SDKs' transient exception classes. Between attempts the adapter waits with an exponential backoff and some jitter, or for the `Retry-After` the server named (up to 60 seconds). After the block the adapter returns a `Completion` whose `error` names the last failure; it never raises for transport.

The block shares one budget with the content retries. `collect` gives each logical answer `attempts` (default 3), and an invalid or refused answer is asked again until the budget is spent, with every transport attempt counting toward it. So one answer costs at most three requests, whatever mix of transport failures and bad replies it meets. The remaining budget travels in a context variable (`codercommittee.providers.base.TRANSPORT_BUDGET`); a coder that calls its model in a worker thread copies the context, or the adapter spends up to three attempts there.

Each completion's identity records the provider, the requested and the returned model, the provider's response id, the finish reason, token counts and the request fingerprint, a sha256 of the canonical request. The answer keeps it, and a store writes it, so a stored run says exactly what was asked of whom.

## Refusals and incomplete replies

A refusal on policy grounds is a `Completion` with `refused` set, and the answer gets status `refused`:

- Anthropic: a `refusal` content block, or a `refusal` stop reason.
- OpenAI: a `refusal` output, or an incomplete response whose reason is `content_filter`.
- Google: a blocked prompt, or a finish reason among `SAFETY`, `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII` and `RECITATION`.
- OpenAI-compatible: a `refusal` in the message, or a `content_filter` finish reason.

A refusal is asked again inside the budget, since a second sample may answer; one that survives the budget is final, is stored as final, and removes its item from every structure of the run. Any other incomplete finish (`max_tokens`, `length`, `MAX_TOKENS`) is an error, so the answer is `invalid` and asked again inside the same budget. The replies a budget threw away are kept in the answer's audit.
