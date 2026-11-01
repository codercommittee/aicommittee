# codercommittee

A committee of coders. `codercommittee` takes several coders' answers to the same text (language models, people, or anything that answers a prompt), combines them under four aggregation structures, aligns the units the coders constructed themselves, scores every structure against human references, and reports each structure's gap to the committee's best member, with an interval, in one table.

```bash
pip install codercommittee
```

The package lives on [GitHub](https://github.com/codercommittee/aicommittee) and on [PyPI](https://pypi.org/project/codercommittee/). This site is also available as [one PDF](https://codercommittee.github.io/aicommittee/pdf/codercommittee.pdf).

## A first run

Three mock coders, two passages, one table. Nothing here calls a network.

```python
from codercommittee import BandRule, Committee, Item, MockCoder, MockEmbedder, Relation, Task, compare

task = Task("tuples", prompt="List every cause-effect relation in the passage as verbatim spans.\n\n{{TEXT}}")
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
]
committee = Committee([MockCoder("alpha"), MockCoder("beta"), MockCoder("gamma")], MockEmbedder(), BandRule(0.80))
run = committee.code(items, task)  # three draws per coder and item: nine answers per item
scores = run.scores()  # one score per item, arm and realization
print(compare([scores], seed=42))  # levels, gaps, intervals, blind pick, spread, premium
```

```text
arm                       level      sd      gap              interval    worst   premium
-----------------------------------------------------------------------------------------
single:alpha              0.619           -0.084      [-0.185, +0.017]   -0.084    +0.020
single:beta               0.475           -0.228      [-0.281, -0.176]   -0.228    -0.124
single:gamma              0.703                                                    +0.104
self:alpha                0.704           +0.001      [-0.205, +0.207]   +0.001    +0.105
self:beta                 0.214           -0.489      [-0.638, -0.340]   -0.489    -0.385
self:gamma                0.876           +0.173      [+0.135, +0.211]   +0.173    +0.277
vote                      0.599           -0.104      [-0.205, -0.002]   -0.104    +0.000
pool:1                    0.189           -0.514      [-0.605, -0.422]   -0.514    -0.410
pool:2                    0.206           -0.497      [-0.573, -0.422]   -0.497    -0.393
pool:3                    0.217           -0.487      [-0.768, -0.205]   -0.487    -0.382
best single per run: single:gamma; blind pick -0.104 (worst -0.228); spread +0.104; 1 run(s), 2 items, 2000 bootstrap draws, seed 42
```

The mock coders answer from a script, so the numbers mean nothing. The columns do: [Reading the results](getting-started.md#reading-the-results) explains them.

## What is in the package

- **Four structures, one operator.** One coder's answer; one coder asked three times; a majority vote of three coders with alignment of the units they constructed; a pool at *k* of nine. All four select members, align what they said, keep what at least *k* members support, and represent each kept unit by its medoid.
- **Alignment with a referee.** Spans are compared by the cosine of their embeddings, never by token overlap. Candidate pairs are banded, and a referee confirms or rejects each: a band rule, or a language model that reads the passage.
- **Four answer kinds.** A boolean label, a label set, one relation with a flag, a list of relations, each with its own score against the human reference.
- **One comparison call.** Every structure's level, its gap to the committee's best member with an item-paired bootstrap interval, the blind pick, the spread and the premium.
- **A facade and a functional core.** `Committee` for the common path; `collect`, `combine`, `align`, `score`, `compare` and the rest importable on their own.
- **Offline by default.** Mock coders and a mock embedder for tests and demonstrations. Thin adapters for Anthropic, OpenAI, Google and any OpenAI-compatible server (vLLM, Ollama) when you want real coders.
- **Saved, resumable runs.** Every answer is written as it arrives; the same command resumes; a resume under another model or referee is refused.
- **The study's tasks.** The five tasks of the study the package re-implements, with their prompts, as package data, and a replay of the study's stored runs that checks every answer, prediction and number item by item.
- **A command line.** `code`, `score`, `compare`, `agreement`, `pick` and `tasks`.

## Where to go next

- [Getting started](getting-started.md): install, the quickstart in Python and on the command line, how to read the table.
- [Concepts](concepts.md): tasks, items, coders, the four structures, alignment, scores, the comparison, the constants.
- Guides: the [command line](guides/command-line.md), [files](guides/files.md), [runs](guides/runs.md), [providers](guides/providers.md), [embedders](guides/embedders.md), [referees](guides/referees.md), [custom components](guides/custom.md), [reproducing the study](study.md), [local models with Ollama](guides/ollama.md).
- [API reference](reference/codercommittee.md), generated from the code.
- [Design](design.md): the data model, the operations, the constants and the tie rules.
- [Changelog](changelog.md) and [releasing](release.md).

## Licence

MIT. The [licence file](https://github.com/codercommittee/aicommittee/blob/main/LICENSE) is in the repository.
