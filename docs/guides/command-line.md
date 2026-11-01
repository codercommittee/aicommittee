# Command line

`pip install codercommittee` installs the `codercommittee` command. It codes items into a run directory, then scores, compares, measures agreement and picks from saved runs. Everything it does is one call into the library, so the Python API can do the same.

```text
usage: codercommittee [-h] [--version]
                      {code,score,compare,agreement,pick,tasks} ...

positional arguments:
  {code,score,compare,agreement,pick,tasks}
    code                code items into a run directory (resumes if it exists)
    score               levels and per-item scores of a saved run
    compare             the comparison table across saved runs
    agreement           cross- and within-coder disagreement of a saved run
    pick                what a labelled sample of n items would pick
    tasks               list the study tasks, or write one as a JSON file to
                        start from
```

A usage error (an unknown task, a missing file, a coder spec that does not parse, a run directory made with another model) is printed on standard error with the prefix `codercommittee:` and exits with status 2. Success exits with 0.

## Specs

| what | spec | examples |
|---|---|---|
| task | `study:NAME`, or the path of a JSON file with the keys of `Task` (`prompt` inline or `prompt_file`) | `study:causalnews`, `task.json` |
| coder | `NAME=mock`, `NAME=mock:SEED`, `NAME=anthropic:MODEL`, `NAME=openai:MODEL`, `NAME=google:MODEL`, `NAME=compatible:MODEL@BASE_URL` | `claude=anthropic:claude-sonnet-5`, `qwen=compatible:qwen3:30b-a3b@http://localhost:11434/v1` |
| embedder | `mock`, `mock:DIMS`, `openai`, `openai:MODEL`, `openai:MODEL:DIMS`, `compatible:MODEL@BASE_URL` | `openai`, `compatible:nomic-embed-text@http://localhost:11434/v1` |
| referee | `band`, `band:ACCEPT_AT`, `llm:PROVIDER:MODEL`, `llm:compatible:MODEL@BASE_URL` | `band`, `llm:anthropic:claude-opus-5-5` |

Coder names are distinct and hold no `|`. The study's task names are `miric`, `gvfc`, `politicause`, `fincausal` and `causalnews`.

Three options apply to every model built from a spec, the coders and the LLM referee alike, on `code` and `score`:

| option | meaning | default |
|---|---|---|
| `--max-output-tokens N` | output budget per request | 4096 |
| `--temperature T` | sampling temperature; sent only when given | the provider's own |
| `--timeout S` | seconds per request before a transport failure | 120 |

## `code`

Asks every coder every item three times and writes a run directory. For three coders and three draws it then builds the four structures, writes the predictions, and, when any item carries a reference, the scores and the levels.

```text
usage: codercommittee code [-h] --task TASK --items ITEMS --out OUT --coder
                           CODER [--embedder EMBEDDER] [--referee REFEREE]
                           [--draws DRAWS] [--attempts ATTEMPTS] [--int-ids]
                           [--max-output-tokens MAX_OUTPUT_TOKENS]
                           [--temperature TEMPERATURE] [--timeout TIMEOUT]
```

| option | meaning |
|---|---|
| `--task TASK` | the task spec |
| `--items ITEMS` | the items file: `.jsonl`, `.json` or `.csv` |
| `--out OUT` | the run directory; resumed when it exists |
| `--coder NAME=SPEC` | one coder; repeat for each seat, in seat order |
| `--embedder SPEC` | required for the `tuple` and `tuples` kinds |
| `--referee SPEC` | the vote's and the self-ensembles' referee; default the band rule at 0.80, or, on a resume, the referee the run was made with |
| `--draws N` | answers per coder and item; default 3 |
| `--attempts N` | the budget of attempts per answer; default 3 |
| `--int-ids` | read numeric ids as integers (CSV ids are strings otherwise) |

```bash
codercommittee code --task study:causalnews --items causalnews.jsonl --out runs/cycle1 \
    --coder claude=anthropic:claude-sonnet-5 --coder gpt=openai:gpt-5.6-terra --coder gemini=google:gemini-3.8-flash \
    --embedder openai --referee llm:anthropic:claude-opus-5-5
```

```text
120 of 120 items complete; 1080 answers stored in runs/cycle1
predictions written to runs/cycle1/predictions.jsonl
arm                       level
single:claude             0.612
...
best single: single:claude; 120 scored items
```

Every answer is appended to `runs/cycle1/answers.jsonl` as it arrives, every item's vectors to `spaces/`, and with an LLM referee every decision to `decisions.jsonl`. Running the same command again asks only for what is missing. A spec that puts another model under a seat, or another referee behind the run, is refused by name. With other coder or draw counts the answers are stored and nothing else is built: the four structures need three coders and three draws.

When an LLM referee runs out of output tokens on the alignment prompt, `code` stops after storing the answers and says what to do: run again with a larger `--max-output-tokens`, and the recorded decisions are served while only the missing ones are asked.

## `score`

Prints the levels and the best single coder of a saved run; `--out` writes the per-item scores.

```text
usage: codercommittee score [-h] [--referee REFEREE] [--out OUT]
                            [--max-output-tokens MAX_OUTPUT_TOKENS]
                            [--temperature TEMPERATURE] [--timeout TIMEOUT]
                            run
```

| option | meaning |
|---|---|
| `run` | the run directory |
| `--referee SPEC` | score under another referee; default the one the run was made with |
| `--out PATH` | the scores as `.csv` (item, arm, realization, value) or `.json` (with the levels) |

```bash
codercommittee score runs/cycle1 --out scores.csv
codercommittee score runs/cycle1 --referee band
```

A run without any reference has nothing to score and exits with status 2.

## `compare`

The comparison table across saved runs over the same items: every arm's level per run with its mean and sd, the gap to the best single coder of the same run with its item-paired bootstrap interval and worst run, the premium, and beneath the rows the blind pick and the spread.

```text
usage: codercommittee compare [-h] --seed SEED [--draws DRAWS] [--out OUT]
                              runs [runs ...]
```

| option | meaning |
|---|---|
| `runs` | one or more run directories |
| `--seed N` | the bootstrap seed; required, so a table can be reproduced |
| `--draws N` | bootstrap draws; default 2000 |
| `--out PATH` | the table as `.csv` (one row per arm) or `.json` (the whole record) |

```bash
codercommittee compare runs/cycle1 runs/cycle2 runs/cycle3 --seed 42017 --out table.csv
```

The runs must share their arms, their coders in seat order and their task kind. Items are paired across runs by their typed id; the interval runs over the items common to every run.

## `agreement`

Cross- and within-coder disagreement over the complete items of a run.

```text
usage: codercommittee agreement [-h] [--out OUT] run
```

```bash
codercommittee agreement runs/cycle1 --out agreement.json
```

```text
items 120; cross-coder disagreement 0.842
  within claude       0.517
  within gpt          0.608
  within gemini       0.692
  identical claude|gpt       0.119
  identical claude|gemini    0.094
  identical gpt|gemini       0.081
  matched Jaccard across coders 0.531
```

The last line appears for the `tuples` kind only. `--out` writes the whole record, the graded block with its count ranges included.

## `pick`

What a labelled sample of *n* items would pick among the single coders, and what the pick costs on the remaining items.

```text
usage: codercommittee pick [-h] --n N --seed SEED [--draws DRAWS]
                           [--among [AMONG ...]] [--out OUT]
                           run
```

| option | meaning |
|---|---|
| `run` | the run directory |
| `--n N` | the sample size; at most half the scored items |
| `--seed N` | the seed; required |
| `--draws N` | random samples; default 500 |
| `--among ARM ...` | the arms to pick among; default the single coders |
| `--out PATH` | the record as JSON |

```bash
codercommittee pick runs/cycle1 --n 20 --seed 42017
codercommittee pick runs/cycle1 --n 20 --seed 42017 --among single:claude vote pool:2
```

```text
a labelled sample of 20 items picks the best single coder in 71% of 500 draws; held-out gap -0.012; blind pick -0.031
```

## `tasks`

Lists the study's five tasks, or writes one as a JSON file to start a task of your own from.

```text
usage: codercommittee tasks [-h] [--out OUT] [name]
```

```bash
codercommittee tasks
codercommittee tasks causalnews --out task.json
```

```text
miric        label   key=is_ai_patent
gvfc         labels  key=frames
politicause  tuple   key=causal
fincausal    tuples  key=relations
causalnews   tuples  key=relations
```

Without `--out` the task is printed as JSON. Edit the prompt, the codebook or the key, keep `{{TEXT}}` in the prompt, and pass the file to `code --task`.
