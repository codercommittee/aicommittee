# Files in and out

Items come in from JSON Lines, JSON or CSV. Results go out as JSON Lines, CSV or JSON, so they open in a spreadsheet or another program. Every writer lives in `codercommittee.io`; the most used ones are importable from the package.

## Items

An items file holds one record per item with three keys:

| key | value |
|---|---|
| `id` | a string or an integer; unique within the file |
| `text` | the text to code |
| `reference` | optional; the human reference, encoded by the task's kind |

The reference's encoding follows the kind:

| kind | `reference` | example |
|---|---|---|
| `label` | a JSON boolean | `true` |
| `labels` | a JSON list of label names | `["politics", "gun_control"]` |
| `tuple` | a JSON list of objects keyed by the task's span fields; an empty list is the negative reference | `[{"cause": "rain", "effect": "floods", "signal": "caused"}]` or `[]` |
| `tuples` | a JSON list of objects keyed by the task's span fields | `[{"cause": "rain", "effect": "floods", "signal": ""}]` |

A relation object may leave out a span field; a field given as an empty string and a field left out are the same thing. A field outside the task's `fields` and `aux` is an error.

Three file formats:

- **JSON Lines** (`.jsonl`): one object per line. Records are split on the newline character alone, so a text may hold any other separator character.
- **JSON** (`.json`): one array of item objects.
- **CSV** (`.csv`): columns `id`, `text`, `reference`; the reference cell holds the JSON above, and an empty cell means no reference. Ids read from CSV are strings; `int_ids=True` (`--int-ids` on the command line) reads a numeric id as an integer.

Files are UTF-8; a byte order mark is accepted. Every error names the file and the line.

```python
from pathlib import Path

from codercommittee import Item, Relation, Task, read_items, write_items

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
    Item(3, "frost hit the orchard early and the press stood idle"),
]
write_items(items, "items.jsonl", task)
write_items(items, "items.csv", task)
print(Path("items.jsonl").read_text().splitlines()[0])
print(Path("items.csv").read_text().splitlines()[1])

back = read_items("items.csv", task, int_ids=True)
print([item.id for item in back], back[2].reference)
```

```text
{"id": 1, "text": "heavy rain filled the river so the bridge washed out", "reference": [{"cause": "heavy rain filled the river", "effect": "the bridge washed out", "signal": ""}]}
1,heavy rain filled the river so the bridge washed out,"[{""cause"": ""heavy rain filled the river"", ""effect"": ""the bridge washed out"", ""signal"": """"}]"
[1, 2, 3] None
```

A bad line is named:

```python
Path("bad.jsonl").write_text('{"id": "a", "text": "t"}\n{"id": "b"}\n')
try:
    read_items("bad.jsonl", task)
except ValueError as error:
    print(error)
```

```text
bad.jsonl:2: an item needs an id and a text
```

## Tasks as files

The command line reads a task from a JSON file with the keys of `Task`: `kind`, `codebook`, `prompt` (or `prompt_file`, a path relative to the task file), `key`, `per_answer`, `fields`, `aux`. A missing or null key takes the default. `codercommittee tasks NAME --out task.json` writes one of the study's tasks in this shape, and `codercommittee.store.task_to_json` and `task_from_json` convert in Python.

## Results

Each result has a writer. The suffix of the path picks the format where more than one is possible.

| writer | what | formats |
|---|---|---|
| `write_predictions(predictions, task, path)` | every arm's predictions: `item`, `arm`, `realization`, `status`, `value` | JSON Lines |
| `write_scores(scores, path)` | per item, arm and realization: `item`, `arm`, `realization`, `value` | CSV; or JSON with the levels and the item count |
| `write_levels(scores, path)` | the levels, the best single coder, the item count | JSON |
| `write_comparison(table, path)` | one row per arm: `arm`, `level`, `level_sd`, `gap`, `gap_sd`, `gap_low`, `gap_high`, `gap_worst`, `premium`, `premium_worst`, `level_run1`, `level_run2`, ... | CSV; or JSON with the whole record |
| `write_agreement(agreement, path)` | the agreement record | JSON |
| `write_pick(pick, path)` | the pick record | JSON |

A prediction's `value` is written in the shape a coder's reply has: a boolean under the task's key for `label`, a sorted list of names for `labels`, the flag and the spans for `tuple`, a list of relation objects for `tuples`. An insufficient prediction (one of its members was not `ok`) has `status` `insufficient` and a null value.

```python
from codercommittee import (
    MockCoder,
    MockEmbedder,
    arms,
    collect,
    compare,
    write_comparison,
    write_predictions,
    write_scores,
)
from codercommittee.compare import Scores, agreement, pick_on_sample
from codercommittee.io import read_jsonl, write_agreement, write_levels, write_pick

run = collect(items, task, [MockCoder("a"), MockCoder("b"), MockCoder("c")], embedder=MockEmbedder())
predictions = arms(run)
write_predictions(predictions, task, "predictions.jsonl")
print(read_jsonl("predictions.jsonl")[0])

scores = Scores.of(run, predictions)
write_scores(scores, "scores.csv")
write_scores(scores, "scores.json")
write_levels(scores, "levels.json")
print(Path("scores.csv").read_text().splitlines()[:2])

table = compare([scores], seed=1)
write_comparison(table, "table.csv")
write_comparison(table, "table.json")
print(Path("table.csv").read_text().splitlines()[0])

write_agreement(agreement(run), "agreement.json")
write_pick(pick_on_sample(scores, 1, draws=20, seed=1), "pick.json")
```

```text
{'item': 1, 'arm': 'single:a', 'realization': 1, 'status': 'ok', 'value': [{'cause': 'heavy rain filled the river so', 'effect': 'the bridge washed out', 'signal': ''}]}
['item,arm,realization,value', '1,single:a,1,0.9743416490255']
arm,level,level_sd,gap,gap_sd,gap_low,gap_high,gap_worst,premium,premium_worst,level_run1
```

`read_jsonl(path)` reads any of the package's JSON Lines files back as a list of records. `to_json(value)` turns any of the package's values (dataclasses, relations, sets, numpy numbers) into a JSON-ready shape, and `dump(value, path)` writes one as indented JSON.

## DataFrames

With pandas installed, `Run.to_frame()` gives one row per answer, `Scores.to_frame()` one row per item, arm and realization, and `Comparison.to_frame()` one row per arm. pandas is an extra (`pip install "codercommittee[pandas]"`), not a dependency; without it these three methods raise `ImportError` and nothing else changes.
