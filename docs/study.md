# Reproducing the study with codercommittee

The study *Three Models Walk Into a Sentence* coded five public datasets with three commercial models, three times each, and compared four aggregation structures. This page is the recipe for running the same design with this package, on the study's datasets or on your own. The package ships the study's five tasks (prompts, codebooks, answer keys, span fields); it does not ship the datasets, whose licences stay with their publishers.

## 1. The study's tasks

```
codercommittee tasks                 # miric, gvfc, politicause, fincausal, causalnews
codercommittee tasks causalnews      # the task as JSON: kind, key, codebook, fields, aux, prompt
```

In Python: `from codercommittee import study; study.TASKS["causalnews"]`. Each prompt is the file the study sent, verbatim, with the one placeholder `{{TEXT}}`; `parity/full_cycles.py` checks that the rendered prompt equals the prompt of every stored request of the study's three cycles.

## 2. The items

An items file holds one record per item: `id`, `text`, and a `reference` encoded by the task's kind (a boolean; a list of label names; a list of `{cause, effect, signal}` objects). Write your own in JSON Lines or CSV, or export a study dataset:

1. Fetch the dataset from its publisher and set the path variables the study's `code/datasets/DATASET_FETCH.md` and `.env.example` name (`CAUSALNEWS_DATA_ROOT`, `FINCAUSAL_DATA_ROOT`, `POLITICAUSE_DATA_ROOT`, `MIRIC_CSV_PATH`, `GVFC_XLSX_PATH` with its codebook and review manifest).
2. From a checkout of the study repository, in its virtual environment:

```
python parity/export_items.py --study /path/to/multi-model-llm-qualitative-coding --dataset causalnews --out causalnews.jsonl
```

The file holds the items the study coded, in the study's order, with their human references.

## 3. Coding

The study's main committee, three draws per coder and item, the referee on the relation datasets, into a run directory that resumes if interrupted:

```
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...
codercommittee code --task study:causalnews --items causalnews.jsonl --out runs/cycle1 \
    --coder anthropic=anthropic:claude-sonnet-5 \
    --coder openai=openai:gpt-5.6-terra \
    --coder google=google:gemini-3.8-flash \
    --embedder openai --referee llm:anthropic:claude-opus-5-5
```

Every answer is appended to `runs/cycle1/answers.jsonl` as it arrives, every item's vectors to `spaces/`, every referee decision to `decisions.jsonl`; running the same command again asks only for what is missing. Label and label-set tasks (Miric, GVFC) need no embedder or referee. Open-weight coders run through any OpenAI-compatible server: `--coder llama=compatible:meta-llama/Llama-3.3-70B-Instruct@http://localhost:8000/v1 --embedder compatible:BAAI/bge-m3@http://localhost:8001/v1`.

A second and third cycle are the same command into `runs/cycle2` and `runs/cycle3`: the study's cycles differ only in the providers' sampling.

## 4. The numbers

```
codercommittee score runs/cycle1                                   # every arm's level, the best single coder
codercommittee compare runs/cycle1 runs/cycle2 runs/cycle3 --seed 42017 --out table.csv
codercommittee agreement runs/cycle1
codercommittee pick runs/cycle1 --n 20 --seed 42017
```

`compare` prints the study's table: per arm the level per run and its mean and sd, the gap to the best single coder of the same run, its item-paired bootstrap interval on run-averaged item differences, the worst run, and the premium over the mean member; beneath it the blind pick and the spread. The study's seeds are 42 for Miric, GVFC and PolitiCAUSE, 42019 for FinCausal and 42017 for CausalNews (`study.BOOTSTRAP_SEED`).

## 5. Checking against the study's own runs

On a machine that holds the study's stored cycles, `python parity/full_cycles.py --study /path/to/study` replays them through the package and checks every answer, prediction, score, level, gap, interval and printed number against the study's outputs. The portable fixture in `tests/fixtures/` runs the same checks on three mock cycles anywhere. `load_run(run_dir, task, items)` replays a stored relation run with its stored referee decisions; pass `referee=BandRule()` to recompute the vote under the band rule instead.

## 6. What the study fixed

Three coders and three draws per item; candidate pairs from similarity 0.60, the vote's band at 0.80 and the pool's at 0.70; deduplication at 0.90 with a 0.85 floor; a match threshold of 0.60; 2,000 bootstrap draws and 500 pick draws. These are constants of the package, not options, so a run here is comparable to the study's. Committees of other sizes are refused at the presets; the `combine` operator itself takes any number of members.
