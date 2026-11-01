# codercommittee

A committee of coders. `codercommittee` takes several coders' answers to the same text (large language models, people, or anything that answers a prompt), combines them under four aggregation structures, aligns the units the coders constructed themselves, scores every structure against human references, and reports each structure's gap to the committee's best member, with its interval, in one table.

It is to [GABRIEL](https://github.com/openai/gabriel) what a committee is to a single coder: GABRIEL asks one model to measure a text; this package asks several and tells you whether combining them was worth it. The algorithms are those of the study *Three Models Walk Into a Sentence*, re-implemented as a typed, documented library and proven against that study's stored runs item by item (see [DESIGN.md](DESIGN.md) and `parity/`).

```python
from codercommittee import BandRule, Committee, Item, MockCoder, MockEmbedder, Relation, Task, compare

task = Task("tuples", prompt="List every cause-effect relation in the passage as verbatim spans.\n\n{{TEXT}}")
items = [
    Item(
        1,
        "heavy rain filled the river so the bridge washed out",
        [Relation(cause="heavy rain filled the river", effect="the bridge washed out")],
    )
]

committee = Committee([MockCoder("alpha"), MockCoder("beta"), MockCoder("gamma")], MockEmbedder(), BandRule(0.80))
run = committee.code(items, task)  # three draws per coder and item: nine answers per item
run.arms()  # single:alpha ... self:alpha ... vote, pool:1, pool:2, pool:3
scores = run.scores()  # item x arm x realization scores; scores.levels()
print(compare([scores, run.scores()], seed=42))  # levels, gaps, intervals, blind pick, spread, premium
```

`python quickstart.py` runs the whole thing on mock coders and a mock embedder in a few seconds.

Documentation: <https://codercommittee.github.io/aicommittee/> (getting started, concepts, guides, the API reference, and the whole site as [one PDF](https://codercommittee.github.io/aicommittee/pdf/codercommittee.pdf)).

## Install

```
pip install codercommittee                 # numpy only
pip install "codercommittee[anthropic,openai,google,pandas]"   # the provider SDKs and DataFrames
```

Python 3.11 or later. From a checkout: `pip install -e ".[dev]"` adds pytest, ruff, mypy and the documentation tools.

## The four structures

Every coder answers every item three times, so each item has nine answers: three coders times three draws. The four structures are defined on these nine answers, and all four are one operator: select members, align what they said, keep what at least *k* members support, represent each kept unit by its medoid, score.

- **One answer (delegation).** A coder's item score is the mean of its three single-answer scores. One arm per coder; the best of them in a run is that run's best single coder. Arms `single:<coder>`.
- **One coder asked three times (repetition).** One coder's three answers combined with the vote's own rules, the three answers standing in for three members: majority on labels, and on relation lists the same alignment, referee, support and medoid rules. One arm per coder. Arms `self:<coder>`.
- **The vote (majority of three coders).** One answer per coder per draw, so three realizations per item; the item score is the mean over the three. A label: an unweighted majority. A label set: every label supported by at least two of the three answers. A flag with one relation: a majority on the flag and, when positive, the observed relation that is the embedding medoid of the positive members. A relation list: alignment and vote (below). Arm `vote`.
- **The pool at *k* of nine.** Relation lists only: all nine answers aligned in one step, and a relation kept when at least *k* of the nine answers contain it, from any coders; a coder's own repeated draws count. *k* is a family of arms, not a chosen rule: `pool:1` is the widest net, `pool:2` a second answer from any member, larger *k* runs toward a hierarchy. Arms `pool:<k>`.

**Alignment.** Wherever two spans are compared, the comparison is the cosine similarity of their embedding vectors, never token overlap. A relation pair scores the mean over the similarity fields of the clamped cosine, computed by exact summation and rounded to twelve places; empty spans score one against empty and zero against text. Every relation answer is deduplicated once (Union-Find at 0.90 with a 0.85 complete-linkage floor, each cluster represented by its medoid). For the vote, every pair of members is matched one to one over pairs at or above 0.60; each candidate pair is banded high (at or above 0.80) or uncertain, and a **referee** confirms or rejects every candidate pair: the band rule (`BandRule`) accepts the high band and rejects the rest, or a language model (`LLMReferee`) reads the passage and decides each pair, the high band included. Confirmed edges form units; a unit stands only when every within-unit pair is confirmed and it holds at most one relation per member, else the weakest edge drops; a unit needs support from at least two members and is represented by its medoid. The pool aligns all nine answers with the band rule at 0.70.

**Scores.** Every score runs from 0 to 1 against the human reference: accuracy with a macro-F1 level for a label; Jaccard for a label set; a detection-gated mean similarity over every reference tuple for a flag with one relation; and for relation lists a count-penalized matched similarity: predictions and references are matched one to one by an exact assignment on the pair similarity, pairs below 0.60 never match, and the matched mass is divided by the larger of the two counts.

**The gap.** The headline comparison is the gap: an arm's level minus the level of the best single coder of the same run, ties by seat order. Several runs are summarised by the run-mean gap with an item-paired percentile bootstrap on the run-averaged item differences (2,000 draws, percentiles 2.5 and 97.5), the between-run standard deviation and the worst run. The intervals are not adjusted for the choice of the best coder or for the many comparisons: they describe the spread of each estimate and are not tests. Beside the gap: the blind pick (the mean gap of the single coders, what picking a coder without a labelled sample costs), the spread (best member minus mean member) and each arm's premium over the mean member, so that gap = premium − spread.

## Tasks, items and answers

A `Task` says what one answer is:

| kind | one answer | reference | example |
|---|---|---|---|
| `label` | one boolean under the codebook's single name | `bool` | is this patent about AI? |
| `labels` | one to `per_answer[1]` names from the codebook | `frozenset[str]` | the one or two frames of a headline |
| `tuple` | a flag (`key`, default `causal`) with one relation over `fields` + `aux` | sequence of `Relation`; empty = negative | causal? cause, effect, signal |
| `tuples` | a list of relations under `key` (default `relations`) | sequence of `Relation` | every cause-effect relation of a passage |

`fields` are the span fields similarity compares (default `cause`, `effect`); `aux` fields (default `signal`) are carried and printed but never compared. The prompt must hold `{{TEXT}}`; `{{CODEBOOK}}` expands to the codebook. `Task.schema()` is the JSON schema every provider receives.

An `Item` is one text with an optional reference; an item without a reference is coded and compared across coders but never scored. A `Run` holds every `Answer` of one cycle (its `status` is `ok`, `invalid`, `refused` or `missing`; any status other than `ok` removes the item from every arm of that run) and each item's embedding `Space`, scoped to that item of that run.

## Coders, embedders, referees

- **Coders** answer one item on one draw. `MockCoder(name)` answers from a deterministic script without a network. `ModelCoder(name, model)` renders the task's prompt, asks a model for the task's schema, parses strictly and validates; an invalid or refused answer is asked again inside one budget of three attempts (the transport attempts counting), and a refusal that survives the budget is final.
- **Models** (`codercommittee.providers`): `AnthropicModel`, `OpenAIModel`, `GoogleModel` and `OpenAICompatibleModel` (vLLM and other `/v1/chat/completions` servers). Each is a thin adapter with structured outputs, three transport attempts and request fingerprints; the SDKs are optional extras imported on first use, and every adapter takes an injected client for tests. The same adapters serve the `LLMReferee`.
- **Embedders** return the vectors of one item's spans: `MockEmbedder()` (feature-hashed character 3-grams) or `OpenAIEmbedder()` (`text-embedding-3-large`, 3,072 dimensions).
- **Referees**: `BandRule(accept_at=0.80)`, `LLMReferee(model)`, `ReplayReferee(run_dir)`.

Keys come from the environment as the SDKs expect them (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`).

```python
from codercommittee import Committee, LLMReferee, ModelCoder, OpenAIEmbedder
from codercommittee.providers import AnthropicModel, GoogleModel, OpenAIModel

coders = [
    ModelCoder("claude", AnthropicModel("claude-sonnet-5")),
    ModelCoder("gpt", OpenAIModel("gpt-5.6-terra")),
    ModelCoder("gemini", GoogleModel("gemini-3.8-flash")),
]
committee = Committee(coders, OpenAIEmbedder(), LLMReferee(AnthropicModel("claude-opus-5-5")))
```

## The functional core

The facade is thin; every method is one call into these functions, which are importable on their own:

| call | what it does |
|---|---|
| `collect(items, task, coders, draws=3, attempts=3, embedder=None, referee=None, store=None)` | the only loop that calls a coder; an invalid or refused answer asked again inside one budget of attempts; embeds each item's spans once; deduplicates relation answers once |
| `dedupe(relations, sim)` | Union-Find at 0.90, floor 0.85, medoids, in candidate-key order |
| `prematch(members, sim, accept_at=...)`, `align(members, sim, referee)` | the candidate pairs with bands and their sha256; every unit with its support and distinct coders, singletons included |
| `combine(members, task, k, p=1, sim=..., referee=...)` | the one operator: keep what at least *k* members from *p* coders support |
| `arms(run, referee=None, pool_k=(1, 2, 3))` | the four structures as presets over the run's complete items |
| `score(prediction, reference, task, sim=None)` | one item's score on the task's scale |
| `Scores.of(run, predictions)`, `compare(scores, seed=...)` | levels; the comparison table across runs |
| `agreement(run)`, `pick_on_sample(scores, n, seed=...)` | disagreement rates; what a labelled sample of *n* items would pick |

Version one supports what the study ran: three coders and three draws per item. Other counts are refused at the presets with a clear error, not in the operator: the majority rule equals "two of three" only for three members, and no tie or quorum rule is invented.

`to_frame()` on a `Run`, `Scores` or `Comparison` returns a pandas DataFrame when pandas is installed; pandas is not a dependency.

## Files in and out

Items come from JSON Lines, JSON or CSV: `id`, `text`, and a `reference` encoded by the task's kind (a boolean; a list of label names; a list of objects keyed by the span fields, an empty list being the negative reference for the `tuple` kind). `read_items(path, task)` names the line of any error; `write_items` writes either format. Predictions, scores, levels, the comparison table, agreement and the pick each have a writer (`write_predictions`, `write_scores`, `write_comparison`, ...) to JSON Lines, CSV or JSON, so results open in a spreadsheet or another program.

## Saving, resuming and replaying a run

`collect(..., store=RunStore("runs/one"))` writes every answer to `answers.jsonl` as it arrives, every item's vectors to `spaces/`, and the manifest to `run.json` (the task, the items, the coders with each seat's model, and the referee); run again with the same task, items, coders and referee, it asks only for what is missing, and a run with another model under a seat or another referee is refused by name. A crash inside an append loses that one line, which the next load (or the next append) drops and asks again. `Run.save(dir)` writes a run already in memory and refuses a directory that holds different answers for the same slots; `Run.load(dir)` rebuilds it through the same path (stored replies re-validated, relation answers re-deduplicated) and scores with the referee the run was made with, so a saved run is a replayable run. `LoggedReferee(LLMReferee(model), store)` serves a decision the store already holds for the same pre-match (bound by its hash) and asks the model only for the rest, so a repeated run never pays twice; `StoreReferee(store)` serves the records alone.

## Command line

```
codercommittee code --task study:causalnews --items items.jsonl --out runs/one \
    --coder claude=anthropic:claude-sonnet-5 --coder gpt=openai:gpt-5.6-terra --coder gemini=google:gemini-3.8-flash \
    --embedder openai --referee llm:anthropic:claude-opus-5-5
codercommittee score runs/one
codercommittee compare runs/one runs/two runs/three --seed 42 --out table.csv
codercommittee agreement runs/one
codercommittee pick runs/one --n 20 --seed 42
codercommittee tasks causalnews --out task.json
```

`--max-output-tokens`, `--temperature` and `--timeout` on `code` and `score` apply to every model built from a spec, the LLM referee included; a model that thinks before it answers (qwen3 through Ollama, for instance) needs a larger output budget than the default 4,096 tokens on the alignment prompt.

A coder is `NAME=mock[:SEED]`, `NAME=anthropic:MODEL`, `NAME=openai:MODEL`, `NAME=google:MODEL` or `NAME=compatible:MODEL@BASE_URL` (vLLM and other OpenAI-compatible servers); an embedder is `mock`, `openai[:MODEL[:DIMS]]` or `compatible:MODEL@BASE_URL`; a referee is `band[:ACCEPT_AT]` or `llm:PROVIDER:MODEL`. A task is `study:NAME` or a JSON file with the keys of `Task` (`prompt` inline or `prompt_file`). `code` resumes a run directory that already exists.

## Embedders

`MockEmbedder()` for tests and demonstrations; `OpenAIEmbedder()` for `text-embedding-3-large` at 3,072 dimensions (the study's); `OpenAICompatibleEmbedder(base_url, model)` for any `/v1/embeddings` server (vLLM, text-embeddings-inference, llama.cpp); and `FunctionEmbedder(fn)` for any callable from texts to vectors, for example a local sentence-transformers model:

```python
from sentence_transformers import SentenceTransformer

from codercommittee import FunctionEmbedder

model = SentenceTransformer("all-MiniLM-L6-v2")
embedder = FunctionEmbedder(lambda texts: model.encode(texts).tolist())
```

## Reproducing the study

The package ships the study's five tasks with the prompts the study sent (`codercommittee tasks`, `codercommittee.study.TASKS`); [docs/study.md](docs/study.md) is the recipe: fetch a dataset from its publisher, export its items with `parity/export_items.py` from the study checkout, code them with the study's committee, and compare the cycles with the study's seeds.

## Replaying stored runs

`ReplayCoder`, `ReplayEmbedder` and `ReplayReferee` serve the answers, vectors and referee decisions of one of the study's run directories (its checkpoint layout) with every identity check; `load_run(run_dir, task, items)` builds the Run from them. Nothing is re-requested; absent evidence raises. `parity/full_cycles.py` replays the study's three stored cycles on five datasets through the library and checks every stored answer, prediction and score, the prompt of every stored request against the packaged task, every arm's level, the gaps with their intervals, the blind pick, spread and premium, the disagreement rates, the labelled-sample pick and the printed numbers of the paper. `parity/build_fixture.py` builds the portable fixture that `tests/test_parity_fixture.py` checks the same way anywhere.

## Development

```
pip install -e ".[dev]"
python -W error -m pytest           # no test calls a provider; a forgotten fake fails loudly
ruff check . && ruff format --check . && mypy
mkdocs build --strict               # the documentation site and its PDF, into site/
```

## Licence

MIT. Copyright (c) 2026 the codercommittee authors.
