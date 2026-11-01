# Custom coders, embedders and referees

The four building blocks of a committee are protocols, not base classes. Anything with the right methods fits; `isinstance` checks work at run time. This page says what each must provide, and what a store expects from it.

## A coder

A coder has a `name` and answers one item on one draw:

```python
from codercommittee import Answer, Item, Task
from codercommittee.validation import validate


class KeywordCoder:
    """Says the text is about AI when it mentions a keyword. A rule, not a model."""

    def __init__(self, name: str, keywords: tuple[str, ...]) -> None:
        self.name = name
        self.keywords = keywords

    def describe(self) -> dict:
        return {"kind": "keyword", "model": "keywords:" + ",".join(self.keywords)}

    def answer(self, item: Item, task: Task, draw: int) -> Answer:
        parsed = {task.key: any(word in item.text.lower() for word in self.keywords)}
        checked = validate(task, parsed, item.text)
        identity = {"coder": self.name, "requested_model": self.describe()["model"]}
        return Answer(
            item.id,
            self.name,
            draw,
            checked.value,
            "ok" if checked.ok else "invalid",
            reason=checked.reason,
            identity=identity,
            audit=checked.audit,
        )
```

What the `Answer` must say:

- `item`, `coder` and `draw` equal to what was asked; `collect` refuses an answer for another slot.
- `status`: `ok`, `invalid` or `refused`. An invalid or refused answer is asked again inside the budget of attempts; a refusal that survives it is final.
- `value` in the kind's shape: a boolean, a frozenset of codebook names, a `Relation` or `None` for the `tuple` kind, a tuple of relations for `tuples`. The easiest way to get it right is to build the reply as the JSON object the task's schema describes and pass it through `validate(task, parsed, item.text)`, which returns the normalized value, a reason when the reply is invalid, and the audit.
- `raw`: the reply as text. A store writes it and re-validates it on load; an `ok` answer without a raw reply is stored with the canonical JSON of its value instead, marked as synthesized.
- `identity`: a record of where the answer came from. With a `requested_model` key, a store can check that a seat without a recorded identity holds answers from the same model.

`describe()` is optional. It returns a JSON record for the run's provenance, and a resume whose record differs for a seat is refused. Its `model` key names the same thing as the answers' `requested_model`. A coder without `describe()` is recorded by its class name, so a resume with another class in the seat is refused too.

```python
from codercommittee import Committee

task = Task("label", {"is_ai_patent": "the patent is about AI"}, "Is this patent about AI?\n\n{{TEXT}}")
items = [
    Item("p1", "A neural network for classifying images", True),
    Item("p2", "A hinge for a cupboard door", False),
    Item("p3", "A machine learning method for routing", True),
]
coders = [
    KeywordCoder("neural", ("neural", "network")),
    KeywordCoder("learning", ("learning", "model")),
    KeywordCoder("either", ("neural", "learning", "machine")),
]
run = Committee(coders).code(items, task)
print(run.scores().levels()["vote"], run.answer("p1", "neural", 1).value)
```

```text
1.0 True
```

## A model

Writing a `Model` is often the better path: `ModelCoder` then renders the prompt, parses the reply as strict JSON, validates it, fills the identity and handles refusals, and the same object can serve the `LLMReferee`. A model has one method:

```python
import json

from codercommittee import ModelCoder, Relation
from codercommittee.protocols import Completion


class FirstAndLast:
    """A stand-in model for the relation kinds: the first three words cause the last three."""

    provider = "stand-in"
    model = "first-and-last"

    def complete(self, prompt: str, schema: dict, *, system: str) -> Completion:
        text = prompt.rsplit("\n\n", 1)[-1]  # this stand-in knows the prompt ends with the passage
        words = text.split()
        reply = {"relations": [{"cause": " ".join(words[:3]), "effect": " ".join(words[-3:]), "signal": ""}]}
        return Completion(json.dumps(reply), {"requested_model": self.model, "returned_model": self.model})


relations = Task("tuples", prompt="List every cause-effect relation.\n\n{{TEXT}}")
coder = ModelCoder("edges", FirstAndLast())
answer = coder.answer(Item(1, "heavy rain filled the river so the bridge washed out"), relations, 1)
print(answer.status, answer.value)
print(coder.describe()["provider"], coder.describe()["model"])
```

```text
ok (Relation(cause='heavy rain filled', effect='bridge washed out', signal=''),)
stand-in first-and-last
```

A `Completion` carries `raw` (the reply text; strict JSON matching the schema), `identity` (any record; `requested_model`, `returned_model`, `request_fingerprint` and `attempts` are the keys the package reads), `error` (a text when the request failed; the answer becomes `invalid`) and `refused` (true when the model declined on policy grounds; the answer becomes `refused`). Transport retries are the model's own business; `codercommittee.providers.base.TRANSPORT_BUDGET` says how many attempts the current call may still make, and `with_retries` in the same module is the block the shipped adapters use. With a `describe()` method the model's record becomes the seat's; without one, the coder records the model's `provider`, `model` and `base_url` attributes.

## An embedder

An embedder has one method, `space(item, texts)`, returning a `Space` with one vector per text. `FunctionEmbedder` covers any callable from texts to vectors, so a class is needed only when the embedder keeps state or needs the item:

```python
from codercommittee import Space


class WordOverlapEmbedder:
    """A bag of words per span. Cosine then measures shared words."""

    def space(self, item, texts):
        vocabulary = sorted({word for text in texts for word in text.lower().split()})
        index = {word: position for position, word in enumerate(vocabulary)}
        vectors = {}
        for text in texts:
            vector = [0.0] * len(vocabulary)
            for word in text.lower().split():
                vector[index[word]] += 1.0
            vectors[text] = vector
        return Space(vectors)


space = WordOverlapEmbedder().space(Item(1, "t"), ["heavy rain", "heavy rain filled the river"])
print(round(space.cosine("heavy rain", "heavy rain filled the river"), 3))
```

```text
0.632
```

Every vector in a space has the same dimension; the values are finite numbers. A space is scoped to one item of one run, so an embedder may use the item's text as context and need not cache anything.

## A referee

A referee has an `accept_at` threshold, the band boundary of the pre-matches it decides, and a `decide(prematch)` method returning one boolean per candidate pair, exactly the pairs in `prematch.candidate_pairs`:

```python
from codercommittee import MockCoder, MockEmbedder, collect


class SharedWordReferee:
    """Confirms a candidate pair when the two causes share a word. A rule for the example, not a judgement; here it confirms every pair, since the mock spans are windows of one text."""

    accept_at = 0.80

    def describe(self) -> dict:
        return {"kind": "shared-word", "accept_at": self.accept_at}

    def decide(self, prematch) -> dict:
        by_label = {f"{member.name}:{member.index}": member for member in prematch.nodes}
        decisions = {}
        for edge in prematch.edges:
            left, right = by_label[edge.a].relation, by_label[edge.b].relation
            decisions[edge.pair] = bool(set(left.cause.split()) & set(right.cause.split()))
        return decisions


items = [
    Item(
        1,
        "heavy rain filled the river so the bridge washed out",
        [Relation(cause="heavy rain filled the river", effect="the bridge washed out")],
    )
]
run = collect(
    items,
    relations,
    [MockCoder("a"), MockCoder("b"), MockCoder("c")],
    embedder=MockEmbedder(64),
    referee=SharedWordReferee(),
)
print(run.arms()["vote"][1][0].audit["decisions"])
print(run.provenance["referee"])
```

```text
{'a:0|b:1': True, 'a:0|c:1': True, 'b:0|c:0': True, 'b:1|c:1': True}
{'kind': 'shared-word', 'accept_at': 0.8}
```

A `Prematch` gives the referee `text` (the passage), `nodes` (the members with their relations), `edges` (each with `pair`, `a`, `b`, `similarity`, `fields`, `band` and `in_conflict`), `candidate_pairs`, and `sha256`. A referee that returns a pair the pre-match did not list, or leaves one out, is an error.

`describe()` returns the record a run stores; `kind` and `accept_at` are the keys the package reads. A custom referee can be logged like any other with `LoggedReferee`, and a saved run whose referee the package cannot rebuild scores from its recorded decisions.

## Checking a component

The protocols are runtime-checkable:

```python
from codercommittee import Coder, Embedder, Model, Referee

print(isinstance(KeywordCoder("k", ("ai",)), Coder), isinstance(FirstAndLast(), Model))
print(isinstance(WordOverlapEmbedder(), Embedder), isinstance(SharedWordReferee(), Referee))
```

```text
True True
True True
```
