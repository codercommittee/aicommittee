# Getting started

## Install

```bash
pip install codercommittee
```

That installs the package and numpy, nothing else. The provider SDKs and pandas are extras:

```bash
pip install "codercommittee[anthropic]"    # the Anthropic SDK
pip install "codercommittee[openai]"       # the OpenAI SDK, also used for OpenAI-compatible servers and embeddings
pip install "codercommittee[google]"       # the Google GenAI SDK
pip install "codercommittee[pandas]"       # to_frame() on runs, scores and comparison tables
pip install "codercommittee[anthropic,openai,google,pandas]"
```

Python 3.11 or later. From a checkout of the repository, `pip install -e ".[dev]"` adds the test and documentation tools.

## The quickstart in Python

Three mock coders, four passages with their references, one run. Nothing on this page calls a network.

```python
from codercommittee import BandRule, Committee, Item, MockCoder, MockEmbedder, Relation, Task

task = Task(
    "tuples",
    prompt="Read the passage and list every cause-effect relation it states, as verbatim spans.\n\nPassage: {{TEXT}}",
)
passages = [
    (
        "heavy rain filled the river so the bridge washed out before dawn",
        "heavy rain filled the river",
        "the bridge washed out",
    ),
    (
        "the council cut the bus subsidy and evening services stopped across the valley",
        "the council cut the bus subsidy",
        "evening services stopped",
    ),
    (
        "frost hit the orchard early and the cider press stood idle all winter",
        "frost hit the orchard early",
        "the cider press stood idle",
    ),
    (
        "the mill closed in march so forty workers left the town for good",
        "the mill closed in march",
        "forty workers left the town",
    ),
]
items = [Item(n, text, [Relation(cause=cause, effect=effect)]) for n, (text, cause, effect) in enumerate(passages, 1)]

committee = Committee([MockCoder("alpha"), MockCoder("beta"), MockCoder("gamma")], MockEmbedder(), BandRule(0.80))
run = committee.code(items, task)
print(len(run.complete()), "items complete,", len(run.answers), "answers")
```

```text
4 items complete, 36 answers
```

`MockCoder` answers from a script: labels from a hash of the item, spans as word windows copied from the text, so every answer is schema-valid and verbatim. `MockEmbedder` hashes character 3-grams into a unit vector. Both stand in for a model; neither measures anything.

The run holds every answer. `run.arms()` builds the four structures over the complete items, one prediction per realization:

```python
predictions = run.arms()
print(list(predictions))
first = run.items[0]
for arm in ("single:alpha", "vote", "pool:2"):
    for prediction in predictions[arm][first.id]:
        print(arm, [(r.cause, r.effect) for r in prediction.value])
```

```text
['single:alpha', 'single:beta', 'single:gamma', 'self:alpha', 'self:beta', 'self:gamma', 'vote', 'pool:1', 'pool:2', 'pool:3']
single:alpha [('heavy rain', 'rain filled the river so the bridge washed out before dawn'), ('out before', 'dawn')]
single:alpha [('heavy rain filled the river', 'river so the bridge washed out before dawn')]
single:alpha [('heavy rain filled the', 'river so the bridge washed out before dawn')]
vote [('out before', 'dawn')]
vote [('heavy rain filled the river', 'river so the bridge washed out before dawn')]
vote [('heavy rain filled', 'the river so the bridge washed out before dawn')]
pool:2 [('heavy rain filled', 'the river so the bridge washed out before dawn'), ('heavy rain filled the', 'river so the bridge washed out before dawn'), ('out before', 'dawn')]
```

`single:alpha` and `vote` have three realizations per item, one per draw; `pool:2` has one. Scoring every structure against the references gives one level per arm:

```python
scores = run.scores()
for arm, level in scores.levels().items():
    print(f"{arm:<13} {level:.3f}")
print("best single:", scores.best_single())
```

```text
single:alpha  0.625
single:beta   0.570
single:gamma  0.538
self:alpha    0.649
self:beta     0.615
self:gamma    0.505
vote          0.558
pool:1        0.177
pool:2        0.387
pool:3        0.429
best single: single:alpha
```

A second run, here the same mock coders with another seed, and `compare` puts both into one table:

```python
from codercommittee import compare

second = Committee(
    [MockCoder("alpha", seed=1), MockCoder("beta", seed=1), MockCoder("gamma", seed=1)], MockEmbedder(), BandRule(0.80)
)
table = compare([scores, second.code(items, task).scores()], seed=42)
print(table)
```

```text
arm                       level      sd      gap              interval    worst   premium
-----------------------------------------------------------------------------------------
single:alpha              0.568   0.080   -0.065      [-0.131, +0.012]   -0.130    -0.009
single:beta               0.606   0.050   -0.027      [-0.093, +0.047]   -0.055    +0.029
single:gamma              0.556   0.025   -0.077      [-0.381, +0.110]   -0.087    -0.020
self:alpha                0.601   0.068   -0.032      [-0.169, +0.058]   -0.088    +0.025
self:beta                 0.606   0.012   -0.027      [-0.267, +0.121]   -0.044    +0.029
self:gamma                0.615   0.155   -0.018      [-0.265, +0.244]   -0.120    +0.038
vote                      0.619   0.086   -0.014      [-0.134, +0.073]   -0.067    +0.042
pool:1                    0.185   0.012   -0.448      [-0.601, -0.270]   -0.448    -0.391
pool:2                    0.372   0.021   -0.261      [-0.434, -0.088]   -0.284    -0.204
pool:3                    0.420   0.011   -0.213      [-0.478, -0.035]   -0.229    -0.156
best single per run: single:alpha, single:beta; blind pick -0.056 (worst -0.130); spread +0.056; 2 run(s), 4 items, 2000 bootstrap draws, seed 42
```

Two more questions the package answers: how much the coders disagree, and what a small labelled sample would pick.

```python
from codercommittee import pick_on_sample

agreement = run.agreement()
print(f"cross-coder disagreement {agreement.cross:.2f}; within-coder {agreement.within}")
pick = pick_on_sample(scores, 2, seed=42)
print(f"a labelled sample of {pick.n} items picks the best coder in {pick.pick_share:.0%} of {pick.draws} draws")
```

```text
cross-coder disagreement 1.00; within-coder {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0}
a labelled sample of 2 items picks the best coder in 69% of 500 draws
```

To run this on real coders, replace the mock coders with `ModelCoder` over a provider adapter and the mock embedder with `OpenAIEmbedder` or an OpenAI-compatible one. The [providers guide](guides/providers.md) shows how; the [Ollama guide](guides/ollama.md) does it with local models.

## The same run on the command line

The command line works on files. A task is a JSON file with the keys of `Task` (`codercommittee tasks causalnews --out task.json` writes one of the study's tasks to start from), and items are one JSON object per line:

```json title="task.json"
{
 "kind": "tuples",
 "prompt": "Read the passage and list every cause-effect relation it states, as verbatim spans.\n\nPassage: {{TEXT}}"
}
```

```json title="items.jsonl"
{"id": 1, "text": "heavy rain filled the river so the bridge washed out before dawn", "reference": [{"cause": "heavy rain filled the river", "effect": "the bridge washed out", "signal": "so"}]}
{"id": 2, "text": "the council cut the bus subsidy and evening services stopped across the valley", "reference": [{"cause": "the council cut the bus subsidy", "effect": "evening services stopped", "signal": ""}]}
{"id": 3, "text": "frost hit the orchard early and the cider press stood idle all winter", "reference": [{"cause": "frost hit the orchard early", "effect": "the cider press stood idle", "signal": ""}]}
{"id": 4, "text": "the mill closed in march so forty workers left the town for good", "reference": [{"cause": "the mill closed in march", "effect": "forty workers left the town", "signal": "so"}]}
```

`code` asks the coders and writes a run directory. A coder is `NAME=SPEC`; `mock` and `mock:SEED` are the mock coder, the real specs are in the [command line guide](guides/command-line.md).

```bash
codercommittee code --task task.json --items items.jsonl --out runs/one \
    --coder alpha=mock --coder beta=mock --coder gamma=mock --embedder mock --referee band
```

```text
4 of 4 items complete; 36 answers stored in runs/one
predictions written to runs/one/predictions.jsonl
arm                       level
single:alpha              0.625
single:beta               0.570
single:gamma              0.538
self:alpha                0.649
self:beta                 0.615
self:gamma                0.505
vote                      0.558
pool:1                    0.177
pool:2                    0.387
pool:3                    0.429
best single: single:alpha; 4 scored items
```

Running the same command again asks for nothing: every answer is stored, and the run resumes from its directory. A second run with other seeds, then the table across both, the agreement and the pick:

```bash
codercommittee code --task task.json --items items.jsonl --out runs/two \
    --coder alpha=mock:1 --coder beta=mock:1 --coder gamma=mock:1 --embedder mock --referee band
codercommittee score runs/one
codercommittee compare runs/one runs/two --seed 42 --out table.csv
codercommittee agreement runs/one
codercommittee pick runs/one --n 2 --seed 42
```

```text
a labelled sample of 2 items picks the best single coder in 69% of 500 draws; held-out gap -0.109; blind pick -0.042
```

## Reading the results

**The levels table** (`code` and `score`) has one row per arm and the arm's level: the mean over the scored items of the item score, then over the realizations; for the `label` kind a macro-F1. The last line names the best single coder, the `single:` arm with the highest level, and how many items were scored.

**The comparison table** (`compare`) has one row per arm:

| column | meaning |
|---|---|
| `level` | the arm's level, averaged over the runs |
| `sd` | the standard deviation of the level across runs; empty with one run |
| `gap` | the arm's level minus the best single coder's level of the same run, averaged over runs |
| `interval` | the 2.5 and 97.5 percentiles of an item-paired bootstrap on the run-averaged item differences |
| `worst` | the gap in the run where it was lowest |
| `premium` | the arm's level minus the mean single coder's level |

An arm that is the best single coder in every run has no gap. Below the rows: the best single coder of each run, the blind pick (the mean gap of the single coders, with its worst case), the spread (best single minus mean single) and the settings. The gap equals the premium minus the spread. The interval describes the spread of each estimate; it is not a test.

**The run directory** holds what `code` collected and built:

| file | contents |
|---|---|
| `run.json` | the task, the coders in seat order, the draws, the items with their references, and how the run was made |
| `answers.jsonl` | one line per answer as it arrived |
| `spaces/` | one file per item with the vectors of its spans |
| `decisions.jsonl` | one line per referee decision, when a language model refereed |
| `predictions.jsonl` | every arm's predictions: item, arm, realization, status, value |
| `scores.csv` | one row per item, arm and realization |
| `levels.json` | the levels and the best single coder |

`compare --out table.csv` writes the comparison table as CSV with one row per arm; `--out table.json` writes the whole record, intervals included. The [files guide](guides/files.md) lists every format.
