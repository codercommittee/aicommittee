# Embedders

An embedder returns the vectors of one item's spans. Wherever the package compares two spans, it compares these vectors by cosine, never by token overlap, so the embedder decides what "the same relation" means for alignment and for scoring.

## What is embedded, and when

The label kinds need no embedder. For the span kinds, once an item's answers are in, `collect` makes one embedding request for the item: every span the item needs, that is the reference's spans and every valid answer's spans over the compared fields (for the `tuple` kind, the causal answers'), each once, in sha256 order. The vectors go into a `Space` scoped to that item of that run. The same text may have other vectors in another item or run; no cache crosses either.

A `Similarity` is a task's view of a space: for two relations, the cosine per compared field, then the mean over the fields, rounded to twelve places. Cosines are clamped at zero and computed by exact summation, so a value is the same on any machine. An empty span scores one against another empty span and zero against text.

```python
from codercommittee import Item, MockEmbedder, Relation, Similarity

item = Item("x", "heavy rain filled the river so the bridge washed out")
space = MockEmbedder().space(item, ["heavy rain", "heavy rain filled the river", "the bridge washed out"])
print(space.dimension, len(space))
print(space.cosine("heavy rain", "heavy rain filled the river"))

sim = Similarity(space, fields=("cause", "effect"))
a = Relation(cause="heavy rain", effect="the bridge washed out")
b = Relation(cause="heavy rain filled the river", effect="the bridge washed out")
print(sim.per_field(a, b), sim.pair(a, b))
```

```text
3072 3
0.60858061945
{'cause': 0.60858061945, 'effect': 1.0} 0.804290309725
```

## The four embedders

**`MockEmbedder(dimensions=3072)`** hashes the character 3-grams of a lowercased text into a unit vector. It is deterministic and needs no network; `MockEmbedder(64)` is enough for a test. It is a stand-in for an embedding model, not a measurement: two spans that share letters look similar to it, two paraphrases do not.

**`OpenAIEmbedder(model="text-embedding-3-large", dimensions=3072)`** asks the OpenAI embeddings API, the study's embedder at the study's dimensions. One request per item, more when an item has over 2,048 spans. The returned model must equal the requested one. Needs the `openai` extra and `OPENAI_API_KEY`.

**`OpenAICompatibleEmbedder(base_url, model, dimensions=None)`** asks any `/v1/embeddings` server through the same SDK: vLLM, text-embeddings-inference, llama.cpp, Ollama. `dimensions` is sent only when given, since most servers refuse it. `api_key` defaults to `"none"`.

**`FunctionEmbedder(embed)`** wraps any callable from a list of texts to a list of vectors, for example a local sentence-transformers model:

```python
from sentence_transformers import SentenceTransformer

from codercommittee import FunctionEmbedder

model = SentenceTransformer("all-MiniLM-L6-v2")
embedder = FunctionEmbedder(lambda texts: model.encode(texts).tolist())
```

The callable must return one vector per text, every vector of the same length, finite numbers only.

On the command line an embedder is `mock`, `mock:DIMS`, `openai`, `openai:MODEL`, `openai:MODEL:DIMS` or `compatible:MODEL@BASE_URL`. A `tuple` or `tuples` task without `--embedder` is a usage error.

Embedders in Python are built, not called, until `collect` runs:

```python
from codercommittee import OpenAICompatibleEmbedder, OpenAIEmbedder

study_embedder = OpenAIEmbedder()
local = OpenAICompatibleEmbedder("http://localhost:11434/v1", "nomic-embed-text")
print(study_embedder.model, study_embedder.dimensions, local.model, local.dimensions)
```

```text
text-embedding-3-large 3072 nomic-embed-text None
```

Changing the embedder changes every similarity, so the levels of a run are comparable only with runs that used the same one. A stored run keeps its vectors in `spaces/`, and a resume reuses them.
