# Referees

For the relation kinds, the vote and the self-ensembles align the members' relations before they count support. Alignment proposes candidate pairs; a referee confirms or rejects each one. The referee is the one place where a judgement enters the aggregation, so the package records exactly what it decided.

## What a referee decides

Alignment starts with a pre-match: every two members are matched one to one over pairs at similarity 0.60 or above, and each candidate pair is banded `high` (at or above the referee's `accept_at`) or `uncertain`. The referee receives the `Prematch`, with the passage, the relations by node, and the candidate pairs with their bands and similarities, and returns one boolean per candidate pair: exactly the listed pairs, nothing else. A pre-match without candidate pairs has nothing to decide, and the referee answers with an empty mapping without any call.

The vote and the self-ensembles band at 0.80 and use the run's referee. The pool bands at 0.70 and always uses the band rule. A single answer is never aligned.

Three referees ship with the package: `BandRule`, `LLMReferee` and `ReplayReferee`. A referee is also a small protocol, so [a custom one](custom.md#a-referee) is a class with an `accept_at` and a `decide` method.

## The band rule

`BandRule(accept_at=0.80)` confirms the high band and rejects the uncertain band. It is deterministic and free. It refuses a pre-match banded at another threshold than its own, so the vote's referee and the pool's rule cannot be confused.

```python
from codercommittee import BandRule, Item, MockCoder, MockEmbedder, Relation, Task, collect

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
]
coders = [MockCoder("a"), MockCoder("b"), MockCoder("c")]
run = collect(items, task, coders, embedder=MockEmbedder(64), referee=BandRule(0.80))
vote = run.arms()["vote"][1][0]  # item 1, first realization
print(vote.audit["candidate_pairs"], "candidate pairs;", vote.audit["decisions"])
print([(unit.support, unit.medoid.cause) for unit in vote.units])
```

```text
4 candidate pairs; {'a:0|b:1': True, 'a:0|c:1': False, 'b:0|c:0': True, 'b:1|c:1': False}
[(2, 'heavy rain filled the river'), (2, 'bridge washed'), (1, 'heavy rain filled the river so the bridge washed')]
```

Every prediction of the vote and the self-ensembles keeps the decisions, the dropped edges and the pre-match's hash in its audit, and its units with their support.

## The LLM referee

`LLMReferee(model)` shows a language model the passage, the relations and the candidate pairs, and asks for one decision per pair. The model decides the high band too. The pre-match is banded at `accept_at` (default 0.80) for the prompt; the computed similarities are shown but never replaced by what the model says.

The prompt is the package's own, with three placeholders:

```text
--8<-- "src/codercommittee/referee_prompt.md"
```

`{{TEXT}}` is the passage, `{{RELATIONS}}` one line per node (`member:index: cause = "..."; effect = "..."`) and `{{PAIRS}}` one line per candidate pair with its band and similarity. The decision schema enumerates exactly the candidate pairs, so the model can neither add a pair nor name one twice. A reply that is a refusal, a transport failure, malformed, or does not cover the pairs exactly is asked again inside one budget of three attempts, the transport attempts counting; after the budget a `RefereeError` is raised. The referee's `calls` counts its requests and `last_identity` holds the identity of its last usable reply.

Any `Model` serves, the provider adapters included. Here a stand-in model confirms every pair, so the example runs without a network:

```python
import json

from codercommittee import LLMReferee
from codercommittee.protocols import Completion


class ConfirmEverything:
    """A stand-in model: it confirms every candidate pair it is shown."""

    def complete(self, prompt, schema, *, system):
        pairs = schema["properties"]["decisions"]["items"]["properties"]["pair"]["enum"]
        decisions = [{"pair": pair, "decision": "confirm"} for pair in pairs]
        return Completion(json.dumps({"decisions": decisions}), {"requested_model": "stand-in"})


referee = LLMReferee(ConfirmEverything())
run = collect(items, task, coders, embedder=MockEmbedder(64), referee=referee)
predictions = run.arms()
print(referee.calls, "referee calls")
print(predictions["vote"][1][0].audit["decisions"])
print(sorted(referee.describe()))
```

```text
12 referee calls
{'a:0|b:1': True, 'a:0|c:1': True, 'b:0|c:0': True, 'b:1|c:1': True}
['accept_at', 'fields', 'kind', 'model', 'prompt_version', 'provider', 'system_sha256', 'template_sha256']
```

`describe()` is the referee's record for a run's provenance: its kind, its threshold, the adapter's settings, the prompt version, and hashes of the template and the system prompt. A run made with one configuration refuses a resume with another.

On the command line the referee is `--referee llm:PROVIDER:MODEL` (`llm:compatible:MODEL@BASE_URL` for a local server); `--max-output-tokens`, `--temperature` and `--timeout` apply to it as to the coders. A model that thinks before it answers needs a larger output budget than the default on this prompt, and the error says so when it runs out.

## Logged and stored decisions

A referee call costs money and time, and the same question should never be paid for twice. `LoggedReferee(inner, store)` wraps any referee: before asking, it looks in the store for a decision of the same item and label bound to the same pre-match hash and covering the same pairs, and serves it; otherwise it asks `inner` and appends the decision to `decisions.jsonl`. `StoreReferee(store)` serves the records alone and raises when one is missing or the pre-match differs.

```python
from codercommittee import LoggedReferee, RunStore, StoreReferee

store = RunStore("runs/refereed")
model = ConfirmEverything()
logged = LoggedReferee(LLMReferee(model), store)
first = collect(items, task, coders, embedder=MockEmbedder(64), referee=logged, store=store)
live = first.arms()
calls_after_first = logged.inner.calls

again = collect(
    items,
    task,
    coders,
    embedder=MockEmbedder(64),
    referee=LoggedReferee(LLMReferee(model), RunStore("runs/refereed")),
    store=RunStore("runs/refereed"),
)
print(again.arms() == live, "and the second pass made", logged.inner.calls - calls_after_first, "new calls")
print(first.arms(StoreReferee(store)) == live, store.has_decisions)
```

```text
True and the second pass made 0 new calls
True True
```

A record of `decisions.jsonl` holds the item, the label of the realization (`panel:1`, `panel:2`, `panel:3` for the vote's three draws, `self:<coder>` for a self-ensemble), the pre-match's sha256, the decisions and the identity of the reply. The pre-match is hashed over the exact shape the study stored, so a decision stays bound to the pair list it answered: change a span, a threshold or a member and the hash changes, and the decision is asked again rather than reused.

A loaded run (`Run.load`) scores with the referee it was made with: the band rule at its recorded threshold, or a `StoreReferee` over its recorded decisions. The command line's `score` and `compare` do the same.

## The replay referee

`ReplayReferee(run_dir)` serves the decisions stored in one of the study's run directories, bound by the same hash, with the model identity and prompt version checked against the run's summary. `codercommittee.load_run` uses it by default for the `tuples` kind. See [replaying the study's stored runs](runs.md#replaying-the-studys-stored-runs).
