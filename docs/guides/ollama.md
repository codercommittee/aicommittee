# Local models with Ollama

Ollama serves open-weight models on your own machine behind an OpenAI-compatible endpoint, so a whole committee can run without an API key or a bill. This page runs three local models as the coders, a local embedding model, and one of the coders as the referee, on five passages. Every number below comes from that run.

## Before you start

Install Ollama and pull the models. The three coders are a mixture-of-experts model that thinks before it answers and two small models; the embedder is a small text embedding model.

```bash
ollama pull qwen3:30b-a3b
ollama pull llama3.2:3b
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

Ollama listens on `http://localhost:11434`; its OpenAI-compatible routes are under `/v1`. The package talks to them through the OpenAI SDK, so install that extra:

```bash
pip install "codercommittee[openai]"
```

## The task and the items

A small relation task and five passages with their references:

```json title="task.json"
{
 "kind": "tuples",
 "prompt": "Read the passage and list every cause-effect relation it states. Copy the cause and the effect verbatim from the passage. Put the connective that marks the relation in signal, or an empty string when there is none.\n\nPassage: {{TEXT}}"
}
```

```json title="items.jsonl"
{"id": 1, "text": "heavy rain filled the river so the bridge washed out before dawn", "reference": [{"cause": "heavy rain filled the river", "effect": "the bridge washed out", "signal": "so"}]}
{"id": 2, "text": "the council cut the bus subsidy and evening services stopped across the valley", "reference": [{"cause": "the council cut the bus subsidy", "effect": "evening services stopped", "signal": ""}]}
{"id": 3, "text": "frost hit the orchard early and the cider press stood idle all winter", "reference": [{"cause": "frost hit the orchard early", "effect": "the cider press stood idle", "signal": ""}]}
{"id": 4, "text": "the mill closed in march so forty workers left the town for good", "reference": [{"cause": "the mill closed in march", "effect": "forty workers left the town", "signal": "so"}]}
{"id": 5, "text": "a new tariff raised import prices which pushed the bakery to close its second shop", "reference": [{"cause": "a new tariff raised import prices", "effect": "the bakery to close its second shop", "signal": "which pushed"}]}
```

## Coding

Every model is a `compatible:MODEL@BASE_URL` spec. The referee is the same spec behind `llm:`.

```bash
codercommittee code --task task.json --items items.jsonl --out runs/local \
    --coder qwen=compatible:qwen3:30b-a3b@http://localhost:11434/v1 \
    --coder llama=compatible:llama3.2:3b@http://localhost:11434/v1 \
    --coder gemma=compatible:gemma3:4b@http://localhost:11434/v1 \
    --embedder compatible:nomic-embed-text@http://localhost:11434/v1 \
    --referee llm:compatible:qwen3:30b-a3b@http://localhost:11434/v1 \
    --max-output-tokens 16384 --timeout 600
```

Two settings matter for a model that thinks before it answers. Its thinking counts against the output budget, and the referee's prompt with its listed pairs needs more of it than an answer does: in this run the referee's replies used up to 7,603 output tokens, well over the default budget of 4,096, so `--max-output-tokens 16384`. And a long think takes long: `--timeout 600` gives each request ten minutes before it counts as a transport failure.

```text
5 of 5 items complete; 45 answers stored in runs/local
predictions written to runs/local/predictions.jsonl
arm                       level
single:qwen               0.884
single:llama              0.736
single:gemma              0.746
self:qwen                 0.854
self:llama                0.802
self:gemma                0.756
vote                      0.756
pool:1                    0.578
pool:2                    0.756
pool:3                    0.756
best single: single:qwen; 5 scored items
```

The run took 33 minutes on a laptop: 45 answers and 30 referee decisions, almost all of the time spent in the thinking model. Every answer was valid on its first attempt. Two of the small model's spans were not exact copies of the passage; they were kept as written and listed in the answers' audit. The referee confirmed 95 of the 108 candidate pairs it was shown.

On the first passage the three coders constructed different units, which is what the alignment is for:

```text
qwen   [{"cause": "heavy rain filled the river", "effect": "the bridge washed out before dawn", "signal": "so"}]
llama  [{"cause": "heavy rain", "effect": "filled the river", "signal": ""}, {"cause": "the bridge", "effect": "washed out", "signal": "so"}]
gemma  [{"cause": "heavy rain", "effect": "filled the river", "signal": "so"}, {"cause": "filled the river", "effect": "the bridge washed out", "signal": "so"}]
vote   [{"cause": "filled the river", "effect": "the bridge washed out", "signal": "so"}, {"cause": "heavy rain", "effect": "filled the river", "signal": ""}]
```

Running the same command again asks for nothing: the answers, the vectors and the referee's decisions are stored, and the run resumes from its directory. Interrupt it half way and the next run picks up where it stopped.

## The numbers

```bash
codercommittee score runs/local
codercommittee agreement runs/local
codercommittee pick runs/local --n 2 --seed 42
```

```text
items 5; cross-coder disagreement 0.867
  within qwen         0.600
  within llama        0.800
  within gemma        0.400
  identical qwen|llama       0.133
  identical qwen|gemma       0.267
  identical llama|gemma      0.333
  matched Jaccard across coders 0.874
a labelled sample of 2 items picks the best single coder in 85% of 500 draws; held-out gap -0.035; blind pick -0.093
```

Five items say nothing about the models. They show the mechanics: the disagreement rates, the alignment, the levels of the four structures, the gap to the best single coder, and a run that costs nothing and resumes.

A second and a third run into `runs/local2` and `runs/local3` (the same command with another `--out`) give `compare` its runs:

```bash
codercommittee compare runs/local runs/local2 runs/local3 --seed 42 --out table.csv
```

## The same in Python

```python title="needs a running Ollama server"
from codercommittee import (
    Item,
    LLMReferee,
    LoggedReferee,
    ModelCoder,
    OpenAICompatibleEmbedder,
    Relation,
    RunStore,
    Task,
    collect,
)
from codercommittee.providers import OpenAICompatibleModel

OLLAMA = "http://localhost:11434/v1"


def local(model: str, **settings) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(model, base_url=OLLAMA, timeout=600, **settings)


task = Task(
    "tuples",
    prompt="Read the passage and list every cause-effect relation it states. Copy the cause and the effect verbatim "
    "from the passage. Put the connective that marks the relation in signal, or an empty string when there is none."
    "\n\nPassage: {{TEXT}}",
)
items = [
    Item(
        1,
        "heavy rain filled the river so the bridge washed out before dawn",
        [Relation(cause="heavy rain filled the river", effect="the bridge washed out", signal="so")],
    ),
    Item(
        2,
        "the council cut the bus subsidy and evening services stopped across the valley",
        [Relation(cause="the council cut the bus subsidy", effect="evening services stopped")],
    ),
    Item(
        3,
        "frost hit the orchard early and the cider press stood idle all winter",
        [Relation(cause="frost hit the orchard early", effect="the cider press stood idle")],
    ),
    Item(
        4,
        "the mill closed in march so forty workers left the town for good",
        [Relation(cause="the mill closed in march", effect="forty workers left the town", signal="so")],
    ),
    Item(
        5,
        "a new tariff raised import prices which pushed the bakery to close its second shop",
        [
            Relation(
                cause="a new tariff raised import prices",
                effect="the bakery to close its second shop",
                signal="which pushed",
            )
        ],
    ),
]
coders = [
    ModelCoder("qwen", local("qwen3:30b-a3b", max_output_tokens=16384)),
    ModelCoder("llama", local("llama3.2:3b")),
    ModelCoder("gemma", local("gemma3:4b")),
]
store = RunStore("runs/local")
referee = LoggedReferee(LLMReferee(local("qwen3:30b-a3b", max_output_tokens=16384)), store)
run = collect(
    items, task, coders, embedder=OpenAICompatibleEmbedder(OLLAMA, "nomic-embed-text"), referee=referee, store=store
)
for arm, level in run.scores().levels().items():
    print(f"{arm:<13} {level:.3f}")
```

`LoggedReferee` records every decision in the store, so a repeated run never asks the referee the same question twice, and a saved run scores as this one did. Without it a `tuples` run with an LLM referee cannot be saved.

## What to expect from small models

A small model more often returns a reply that does not fit the schema, or spans that are not copies of the passage. A reply that does not fit is an invalid answer and is asked again inside the budget of three attempts; an item whose budget runs out for any coder leaves every structure of that run. A span that is not a copy is kept and listed in the audit. Both show up in `answers.jsonl`, so a run tells you what its coders did before you read its levels.
