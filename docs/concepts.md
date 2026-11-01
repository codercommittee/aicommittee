# Concepts

The terms this site uses are the ones the code uses. This page defines them in the order a run meets them.

## Tasks and the four answer kinds

A `Task` says what one answer is. Its `kind` is one of four:

| kind | one answer | reference | example |
|---|---|---|---|
| `label` | one boolean under the codebook's single name | `bool` | is this patent about AI? |
| `labels` | one to `per_answer[1]` names from the codebook | a set of names | the one or two frames of a headline |
| `tuple` | a flag (`key`, default `causal`) with one relation over `fields` and `aux` | a sequence of relations; empty means negative | causal? cause, effect, signal |
| `tuples` | a list of relations under `key` (default `relations`) | a sequence of relations | every cause-effect relation of a passage |

A relation is a `Relation`: an immutable mapping from field name to a span of text. `fields` are the span fields that similarity compares (default `cause` and `effect`); `aux` fields (default `signal`) are carried and printed but never compared. The `codebook` maps label names to descriptions: exactly one entry for `label`, the vocabulary for `labels`, empty for the span kinds.

The prompt must hold `{{TEXT}}`. `{{CODEBOOK}}` expands to one line per label. `Task.schema()` is the JSON schema every model receives with the prompt: an object with exactly the task's keys and no extra properties.

```python
from codercommittee import Task

patent = Task(
    "label", {"is_ai_patent": "the patent is related to AI technologies"}, "Is this patent about AI?\n\n{{TEXT}}"
)
frames = Task(
    "labels",
    {"politics": "campaigns, parties, politicians", "gun_control": "regulating guns through legislation"},
    "Pick one or two frames.\n{{CODEBOOK}}\n\n{{TEXT}}",
    key="frames",
    per_answer=(1, 2),
)
relations = Task("tuples", prompt="List every cause-effect relation as verbatim spans.\n\n{{TEXT}}")

print(patent.schema()["properties"])
print(frames.render("Senators split over new gun bill").splitlines()[1])
print(relations.names)
```

```text
{'is_ai_patent': {'type': 'boolean'}}
- politics: campaigns, parties, politicians
('cause', 'effect', 'signal')
```

## Items and references

An `Item` is one text with an id and an optional human reference. The reference takes the kind's shape: a boolean, a set of label names, or a sequence of relations. For the `tuple` kind an empty sequence is the negative reference: the text states no relation.

An item without a reference is coded and compared across coders but never scored.

```python
from codercommittee import Item, Relation

scored = Item(
    "p1",
    "heavy rain filled the river so the bridge washed out",
    [Relation(cause="heavy rain filled the river", effect="the bridge washed out")],
)
negative = Item("p2", "the river was quiet", [])
unscored = Item("p3", "frost hit the orchard early and the press stood idle")
print(scored.reference, negative.reference, unscored.reference)
```

```text
(Relation(cause='heavy rain filled the river', effect='the bridge washed out'),) () None
```

## Coders, draws and seats

A coder answers one item on one draw. Every coder answers every item three times. A draw is one of those answers, numbered 1 to 3; three coders times three draws give nine answers per item.

The seat order is the order the coders were given in. It matters in one place: when two single coders tie for the best level, the earlier seat wins.

A coder's name is any string without `|` and without surrounding whitespace. A colon is fine, so a model tag such as `qwen3:30b-a3b` can be a name.

## Answers and statuses

An `Answer` carries a status: `ok`, `invalid`, `refused` or `missing`. A model's reply is parsed as strict JSON and validated against the task. A span that is not a copy of the text is kept as written and listed in the answer's audit; a span that matches after folding curly quotes, apostrophes and dashes counts as a copy and takes the text's own characters.

An invalid or refused answer is asked again inside one budget of three attempts. The transport attempts an adapter made count toward the budget, so one logical answer never costs more than three requests. A refusal that survives the budget is final.

Any status other than `ok` removes the item from every structure of that run. `Run.complete()` lists the items every coder answered on every draw with status `ok`. `Run.scored()` lists the complete items with a reference the level uses; for the `tuple` kind that is a causal one, so items with a negative reference are complete but not scored.

## The four structures

All four structures are one operator: select members, align what they said, keep what at least *k* members support, represent each kept unit by its medoid, score. The structures are presets of that operator. Their names, the arm keys, are the keys of every table.

| arm | members | k | referee | realizations per item |
|---|---|---|---|---|
| `single:<coder>` | one draw of one coder | 1 | none | three, one per draw |
| `self:<coder>` | the coder's three draws | 2 | the run's | one |
| `vote` | draw *r* of every coder | 2 | the run's | three, one per draw |
| `pool:<k>` | all nine answers | *k* | the band rule at 0.70 | one |

A realization is one application of the operator. An arm's item value is the mean over its realizations. What "support" means depends on the kind:

- `label`: the majority. The prediction is true when at least *k* members say true.
- `labels`: every label that at least *k* members named.
- `tuple`: a majority on the flag and, when positive, the relation that is the embedding medoid of the positive members.
- `tuples`: alignment under the referee, then the units with support at least *k*, each represented by its medoid.

The pool exists for the `tuples` kind only. `pool:1` keeps every unit, `pool:2` what a second answer from any member supports, and so on. A coder's own repeated draws count, so *k* is a family of arms rather than a chosen rule.

This version supports three coders and three draws per item, what the study ran. The presets refuse other counts with a clear error; `combine` itself takes any number of members.

## Alignment and the referee

Wherever two spans are compared, the comparison is the cosine similarity of their embedding vectors. A relation pair's similarity is the mean over the compared fields of the clamped cosine, computed by exact summation and rounded to twelve places. An empty span scores one against another empty span and zero against text. Every threshold below is "at or above" on this rounded value.

Before any structure runs, each answer's relation list is deduplicated once: relations at pair similarity 0.90 or above join one cluster, a cluster of three or more whose weakest pair falls below 0.85 splits back into singletons, and each cluster is represented by its medoid.

Alignment then runs over the members of one realization:

1. **The pre-match.** Every two members are matched one to one over pairs at or above 0.60. Each candidate pair is banded `high` (at or above the referee's `accept_at`) or `uncertain`, and flagged when it sits inside an inconsistent group.
2. **The referee.** It confirms or rejects every candidate pair and nothing else. `BandRule` accepts the high band and rejects the rest. `LLMReferee` reads the passage and the listed relations and decides each pair, the high band included.
3. **Units.** Confirmed pairs form components. A component stands as a unit only when every pair inside it is confirmed and it holds at most one relation per member; while that fails, the weakest edge inside it drops. Every component is a unit, singletons included, with its support (the number of members) and its medoid.
4. **The keep rule.** The units with support at least *k* are kept, their medoids in content order.

The vote and the self-ensembles band at 0.80 and use the run's referee. The pool bands at 0.70 and always uses the band rule.

The pre-match is hashed (a sha256 over the exact shape the study stored), so a recorded referee decision stays bound to the pre-match it answered. The [referees guide](guides/referees.md) shows the prompt and the records.

## Scores and levels

Every score runs from 0 to 1 against the human reference.

| kind | item score | level |
|---|---|---|
| `label` | accuracy, 1 or 0 | macro-F1 over the pooled confusion counts per realization, then the mean over realizations |
| `labels` | Jaccard: shared labels over all labels named; both empty scores one, one side empty scores zero | mean over items per realization, then over realizations |
| `tuple` | 0 when the prediction is negative on a positive reference, else the mean pair similarity over the reference relations | as above |
| `tuples` | a count-penalized matched similarity: predictions and references matched one to one by an exact assignment, pairs below 0.60 never match, the matched similarity divided by the larger of the two counts | as above |

A `Scores` holds one run's item by arm by realization scores. `levels()` gives one level per arm; `best_single()` names the single coder with the highest level.

## The comparison

`compare([scores, ...], seed=...)` takes the scores of one or more runs over the same items and returns one table:

- **Level.** Per arm, the level per run, its mean and its standard deviation across runs.
- **Best single.** Per run, the `single:` arm with the highest level, ties by seat order.
- **Gap.** An arm's level minus the best single's level of the same run: the run-mean gap, its standard deviation and the worst run. An arm that is the best single in every run has no gap.
- **Interval.** The 2.5 and 97.5 percentiles of an item-paired bootstrap on the run-averaged item differences: 2,000 draws by default, one fresh generator per interval from the seed, the same draws for every arm. The interval describes the spread of the estimate. It is not adjusted for the choice of the best coder or for the many comparisons, and it is not a test.
- **Blind pick.** The mean gap of the single coders, with its worst case: what picking a coder without a labelled sample costs.
- **Spread.** The best single's level minus the mean member's level, both averaged over runs.
- **Premium.** An arm's mean level minus the mean member's. The gap is the premium minus the spread.

## Agreement and the labelled-sample pick

`agreement(run)` reports disagreement over the complete items: the share of item realizations whose coders' answers differ (cross), the share of items whose draws differ per coder (within), and the share of identical answers per coder pair. Relation lists also get a graded block: the mean matched Jaccard at 0.60 within and across coders, and the ranges of the relation counts.

`pick_on_sample(scores, n, seed=...)` asks what a labelled sample of *n* items would pick. Over 500 random samples by default, it picks the single coder with the highest sample mean (a tie split evenly), and reports the pick's gap to the run's best single on the remaining items, the share of samples that pick the best single, and the blind pick's gap on the same remaining items. *n* is at most half the items.

## The constants

A parameter with one value in the study is a constant of the package, not an option, so a run here is comparable to the study's.

| constant | value | name in the code |
|---|---|---|
| deduplication threshold | 0.90 | `codercommittee.dedupe.DEDUP_THRESHOLD` |
| complete-linkage floor of a cluster | 0.85 | `codercommittee.dedupe.LINKAGE_FLOOR` |
| candidate pairs from | 0.60 | `codercommittee.align.LOW_THRESHOLD` |
| the vote's band | 0.80 | `codercommittee.combine.VOTE_ACCEPT_AT` |
| the pool's band | 0.70 | `codercommittee.arms.POOL_ACCEPT_AT` |
| match threshold in scoring | 0.60 | `codercommittee.scoring.MATCH_THRESHOLD` |
| rounding of every similarity | 12 places | `codercommittee.similarity.DECIMALS` |
| enumeration guard of the pairwise matching | 24 edges | `codercommittee.matching.ENUMERATION_GUARD` |
| value tolerance of the matching | 1e-9 | `codercommittee.matching.VALUE_TOLERANCE` |
| tie tolerance of the pick | 1e-12 | `codercommittee.compare.TIE_TOLERANCE` |
| coders and draws at the presets | 3 and 3 | `codercommittee.arms.COMMITTEE_SIZE`, `codercommittee.arms.DRAWS` |
| interval percentiles | 2.5 and 97.5 | `codercommittee.compare.PERCENTILES` |

Two counts are keyword defaults so that tests can run smaller: the bootstrap draws (2,000, `codercommittee.compare.REPLICATES`) and the pick draws (500, `codercommittee.compare.PICK_DRAWS`).
