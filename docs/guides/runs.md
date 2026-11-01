# Saving, resuming and replaying runs

A run can be written to a directory as it is collected, saved after the fact, loaded back, resumed, and scored again. The package has one layout for its own runs; it also reads the layout of the study's stored runs.

## The run directory

| file | contents |
|---|---|
| `run.json` | the manifest: the task, the coders in seat order, the draws, the items with their references, and the provenance (the referee's configuration and each seat's identity) |
| `answers.jsonl` | one line per answer as it arrived; the last line of a slot wins |
| `spaces/000000.json`, ... | the vectors of one item's spans, named by the item's position in the manifest |
| `decisions.jsonl` | one line per referee decision, bound to its pre-match by a sha256 |

An answer line holds the item, coder and draw, the status and reason, the raw reply, the normalized value, the identity the adapter reported (requested and returned model, request fingerprint, attempts) and the audit. The raw reply is what counts: a load re-validates it and never trusts the stored value.

## Writing a run as it is collected

`collect(..., store=RunStore(directory))` writes the manifest before the first call, appends each answer before the next call, writes each item's space once its answers are in, and, with a `LoggedReferee`, each referee decision as it is made. The command line's `code` works this way.

```python
from codercommittee import Item, MockCoder, MockEmbedder, Relation, RunStore, Task, collect

task = Task("tuples", prompt="List every cause-effect relation.\n\n{{TEXT}}")
items = [
    Item(
        1,
        "heavy rain filled the river so the bridge washed out",
        [Relation(cause="heavy rain filled the river", effect="the bridge washed out")],
    ),
    Item(
        2,
        "the mill closed in march so forty workers left the town",
        [Relation(cause="the mill closed in march", effect="forty workers left the town")],
    ),
    Item(
        3,
        "frost hit the orchard early and the cider press stood idle",
        [Relation(cause="frost hit the orchard early", effect="the cider press stood idle")],
    ),
]


class Counting:
    """A coder that counts how often it is asked; otherwise the mock coder."""

    def __init__(self, name: str) -> None:
        self.inner = MockCoder(name)
        self.name = name
        self.calls = 0

    def describe(self) -> dict:
        return self.inner.describe()

    def answer(self, item, task, draw):
        self.calls += 1
        return self.inner.answer(item, task, draw)


coders = [Counting("a"), Counting("b"), Counting("c")]
run = collect(items, task, coders, embedder=MockEmbedder(64), store=RunStore("runs/one"))
print([coder.calls for coder in coders], RunStore("runs/one").answered(), "answers stored")
```

```text
[9, 9, 9] 27 answers stored
```

## Resuming

Run `collect` again with the same task, items, coders and referee into the same directory, and it asks only for what is missing:

```python
again = collect(items, task, coders, embedder=MockEmbedder(64), store=RunStore("runs/one"))
print([coder.calls for coder in coders], again.arms() == run.arms())
```

```text
[9, 9, 9] True
```

What a resume reuses:

- every stored answer with status `ok`, re-validated from its raw reply;
- every stored refusal that spent the budget it is now asked under (a refusal stored under a smaller budget is asked again);
- every stored space whose texts are exactly the spans the item needs now;
- every recorded referee decision bound to the same pre-match (see [referees](referees.md)).

What it asks again: an `invalid` answer, a refusal with a smaller spent budget, and the slot of a torn final line. A crash inside an append leaves a line without its newline; the next load drops it, repairs the file and asks for that slot again. The manifest and the spaces are written atomically, so a crash never leaves a half-written snapshot.

What it refuses, by name:

- a directory that holds another run, when the task, the coders, the draws or the items differ;
- another model under a seat: each seat's identity (`describe()` on the coder) is in the manifest, and a seat keeps its model across a resume;
- another referee: its configuration is in the manifest too.

```python
from codercommittee import StoreError

try:
    collect(
        items,
        task,
        [MockCoder("a", seed=5), MockCoder("b"), MockCoder("c")],
        embedder=MockEmbedder(64),
        store=RunStore("runs/one"),
    )
except StoreError as error:
    print(error)
```

```text
runs/one holds a run whose coder 'a' was {'kind': 'mock', 'model': 'mock', 'seed': 0}, not {'kind': 'mock', 'model': 'mock', 'seed': 5}; a seat keeps its model across a resume
```

On the command line the same rules apply to `code`: a run directory that exists is resumed, a coder spec with another model is refused, and without `--referee` the resume keeps the referee the run was made with.

## Saving a run already in memory

`Run.save(directory)` writes a run that was collected without a store: the answers, the spaces, the provenance and the decisions its referee served. Saving a run over its own copy is idempotent; an identical answer is not rewritten, so the first record's evidence stands. Saving into a directory that holds different answers, vectors or decisions for the same slots is refused, never merged.

```python
from pathlib import Path

second = collect(
    items, task, [MockCoder("a", seed=1), MockCoder("b", seed=1), MockCoder("c", seed=1)], embedder=MockEmbedder(64)
)
second.save("runs/two")
print(sorted(path.name for path in Path("runs/two").iterdir()))
```

```text
['answers.jsonl', 'run.json', 'spaces']
```

One case is refused outright: a `tuples` run whose referee is an `LLMReferee` not wrapped in a `LoggedReferee`. Its decisions were never recorded, so a saved copy could not score as the run did. Wrap the referee before collecting.

## Loading

`Run.load(directory)` rebuilds a run through the same `collect` path, with the store standing in for the coders and the embedder. Stored replies are re-validated and relation answers re-deduplicated. The run scores with the referee it was made with: the band rule at its recorded threshold, or the recorded decisions of a language-model referee. Pass another referee to score under it instead.

```python
from codercommittee import BandRule, Run

loaded = Run.load("runs/one")
print(loaded.coders, len(loaded.complete()), "complete items")
print(loaded.arms() == run.arms())
stricter = Run.load("runs/one", referee=BandRule(0.90))
print(stricter.referee)
```

```text
('a', 'b', 'c') 3 complete items
True
BandRule(accept_at=0.9)
```

A run made with a language-model referee and loaded without one serves its recorded decisions, and a realization the recorded referee never decided is named when scoring asks for it. `codercommittee score` and `Run.load` therefore agree with what `code` printed.

## Replaying the study's stored runs

The study's own run directories have another layout: `per_item.jsonl` (one row per item, in run order), `summary.json`, and under `checkpoints/` one file per response, per item's embeddings and per referee decision. `codercommittee.load_run` builds a `Run` from one of them without re-requesting anything. Every identity the records carry is checked: the requested and returned model, the request fingerprint, the prompt version, strict JSON, validation of the stored reply, and the replies a retry discarded. A failed check makes that answer invalid; absent evidence raises `ReplayError`; a stored refusal is final.

```python title="needs one of the study's run directories"
from codercommittee import BandRule, load_run, read_items, study

task = study.TASKS["causalnews"]
items = read_items("causalnews.jsonl", task)  # the items the stored run coded, in its order
run = load_run("study_runs/causalnews/cycle1", task, items)  # answers, vectors and the stored referee decisions
print(run.scores().levels()["vote"])
under_the_band_rule = load_run("study_runs/causalnews/cycle1", task, items, referee=BandRule())
run.save("runs/causalnews_cycle1")  # the package's layout, the stored decisions copied along
```

For the `tuples` kind the stored referee decisions are the run's referee unless another is given. [Reproducing the study](../study.md) says where the items come from and how the replay is checked against the study's numbers.
