# Package

::: codercommittee
    options:
      members: false

## What the package exports

`from codercommittee import ...` reaches every name below. Each links to the module that defines it; the submodules hold more.

### [Committee](committee.md)

| Name | What it is |
|---|---|
| [Committee][codercommittee.committee.Committee] | Several coders asked the same items; the embedder and the referee the span kinds need |

### [Data model](data.md)

| Name | What it is |
|---|---|
| [Answer][codercommittee.data.Answer] | One coder's validated reply to one item on one draw |
| [Item][codercommittee.data.Item] | One text to code, with its human reference when there is one |
| [ItemScore][codercommittee.data.ItemScore] | One prediction's score against one reference, with what the level needs later |
| [Match][codercommittee.data.Match] | One matched prediction-reference pair of a relation score |
| [Member][codercommittee.data.Member] | One member's candidate relation inside an alignment: who said it, and which of theirs |
| [Prediction][codercommittee.data.Prediction] | What one structure says about one item: a value with a status, and the units behind it |
| [Relation][codercommittee.data.Relation] | The one structured atom: an immutable mapping of field name to a stripped span |
| [Run][codercommittee.data.Run] | One cycle: every coder's answers on every draw over the items, and each item's space |
| [Task][codercommittee.data.Task] | What one answer is |
| [Unit][codercommittee.data.Unit] | An aligned unit: the relations the committee's members constructed for one thing |

### [Protocols](protocols.md)

| Name | What it is |
|---|---|
| [Coder][codercommittee.protocols.Coder] | Answers one item on one draw. ``name`` is the coder's id inside a run |
| [Completion][codercommittee.protocols.Completion] | One raw model reply with the identity of the request that produced it |
| [Embedder][codercommittee.protocols.Embedder] | Returns the vectors of one item's spans; the space is scoped to that item of that run |
| [Model][codercommittee.protocols.Model] | A structured-output model call: the provider adapters, used by coders and the referee |
| [Referee][codercommittee.protocols.Referee] | Confirms or rejects every candidate pair of a pre-match whose bands split at ``accept_at`` |

### [Collect](collect.md)

| Name | What it is |
|---|---|
| [collect][codercommittee.collect.collect] | Ask every coder every item on every draw and return the Run |

### [Combine](combine.md)

| Name | What it is |
|---|---|
| [combine][codercommittee.combine.combine] | Combine the members' answers: keep what at least ``k`` members from ``p`` coders support |

### [Arms](arms.md)

| Name | What it is |
|---|---|
| [POOL_KS][codercommittee.arms.POOL_KS] | The pool's `k` values, (1, 2, 3) |
| [arm_names][codercommittee.arms.arm_names] | The arm keys in the study's order: singles, self-ensembles, the vote, then the pools |
| [arms][codercommittee.arms.arms] | Every structure's predictions, per arm and item, one Prediction per realization |

### [Align](align.md)

| Name | What it is |
|---|---|
| [Alignment][codercommittee.align.Alignment] | What alignment produced: every unit, the pre-match, the decisions and the dropped edges |
| [Prematch][codercommittee.align.Prematch] | The quantitative pre-match: nodes, candidate edges, bands, and what the matching did |
| [align][codercommittee.align.align] | Align the members' relations into units under the referee's decisions |
| [prematch][codercommittee.align.prematch] | The candidate pairs across members, banded at ``accept_at`` and flagged inside inconsistent groups |

### [Dedupe](dedupe.md)

| Name | What it is |
|---|---|
| [dedupe][codercommittee.dedupe.dedupe] | One answer's relations without near-duplicates, each cluster's medoid, in key order |
| [medoid][codercommittee.dedupe.medoid] | The member closest on average to the others; ties by the candidate key |

### [Similarity](similarity.md)

| Name | What it is |
|---|---|
| [Similarity][codercommittee.similarity.Similarity] | A task's view of a space: field-wise cosines averaged with equal weights |
| [Space][codercommittee.similarity.Space] | The vectors of one item's spans, keyed by sha256(text), with memoised exact cosines |

### [Validation](validation.md)

| Name | What it is |
|---|---|
| [Validation][codercommittee.validation.Validation] | The normalized value with an optional failure reason and an audit of the normalization |
| [validate][codercommittee.validation.validate] | Validate and normalize a parsed answer for one item |

### [Scoring](scoring.md)

| Name | What it is |
|---|---|
| [macro_f1][codercommittee.scoring.macro_f1] | Binary macro-F1 from pooled confusion cells (positive class, then negative class) |
| [score][codercommittee.scoring.score] | Score one prediction (or a bare value) against one reference on the task's scale |

### [Compare](compare.md)

| Name | What it is |
|---|---|
| [Agreement][codercommittee.compare.Agreement] | Disagreement rates over the complete items: identity of the coders' normalized answers |
| [ArmRow][codercommittee.compare.ArmRow] | One arm's row of the comparison table |
| [Comparison][codercommittee.compare.Comparison] | Every arm against the best single coder of the same run, with intervals, in one table |
| [Gap][codercommittee.compare.Gap] | An arm's gap to the best single of the same run, summarised across runs |
| [Pick][codercommittee.compare.Pick] | What a labelled sample of n items picks, and what the pick costs on the remaining items |
| [Scores][codercommittee.compare.Scores] | One run's scores: per arm, per scored item (run order), one ItemScore per realization |
| [agreement][codercommittee.compare.agreement] | Disagreement rates over the complete items |
| [compare][codercommittee.compare.compare] | Every arm's level, gap, interval, blind pick, spread and premium across runs, in one call |
| [pick_on_sample][codercommittee.compare.pick_on_sample] | What a labelled sample of n items picks, over ``draws`` random samples |

### [Referee](referee.md)

| Name | What it is |
|---|---|
| [BandRule][codercommittee.referee.BandRule] | Accept every candidate pair whose similarity is at or above ``accept_at``; reject the rest |
| [LLMReferee][codercommittee.referee.LLMReferee] | A model confirms or rejects every candidate pair of the pre-match it is shown |
| [RefereeError][codercommittee.referee.RefereeError] | The referee gave no usable decision inside its block of attempts |
| [ReplayReferee][codercommittee.referee.ReplayReferee] | Serves the stored decisions of a run, bound to the pre-match by its sha256 |

### [Coders](coders.md)

| Name | What it is |
|---|---|
| [MockCoder][codercommittee.coders.MockCoder] | A coder that answers from a script without any network |
| [ModelCoder][codercommittee.coders.ModelCoder] | A coder over a :class:`~codercommittee.protocols.Model` |

### [Embedders](embedders.md)

| Name | What it is |
|---|---|
| [FunctionEmbedder][codercommittee.embedders.FunctionEmbedder] | Any callable from a list of texts to a list of vectors |
| [MockEmbedder][codercommittee.embedders.MockEmbedder] | Deterministic feature hashing of character 3-grams, unit length; no network |
| [OpenAICompatibleEmbedder][codercommittee.embedders.OpenAICompatibleEmbedder] | Any ``/v1/embeddings`` server through the OpenAI SDK: vLLM, text-embeddings-inference, llama.cpp |
| [OpenAIEmbedder][codercommittee.embedders.OpenAIEmbedder] | ``text-embedding-3-large`` vectors through the OpenAI SDK, one request per item (more above 2,048 spans) |

### [Store](store.md)

| Name | What it is |
|---|---|
| [LoggedReferee][codercommittee.store.LoggedReferee] | Serves the store's recorded decision for a pre-match, else decides with ``inner`` and records it |
| [RunStore][codercommittee.store.RunStore] | One run directory: its manifest, answers, spaces and referee decisions |
| [StoreError][codercommittee.store.StoreError] | The store lacks what was asked for, or holds another run |
| [StoreReferee][codercommittee.store.StoreReferee] | Serves the store's recorded decisions; raises when none exists or the pre-match differs |

### [Replay](replay.md)

| Name | What it is |
|---|---|
| [ReplayCoder][codercommittee.replay.ReplayCoder] | Serves one coder's stored answers, checking every identity the record carries |
| [ReplayEmbedder][codercommittee.replay.ReplayEmbedder] | Serves one item's stored vectors; the stored set must equal the texts asked for |
| [ReplayError][codercommittee.replay.ReplayError] | Evidence is absent or does not describe what was asked for |
| [StoredRun][codercommittee.replay.StoredRun] | A run directory: its rows, coders, identities and checkpoint files |
| [load_run][codercommittee.replay.load_run] | The stored run as a Run: every answer, vector and decision served from the checkpoints |

### [Files](io.md)

| Name | What it is |
|---|---|
| [read_items][codercommittee.io.read_items] | Items from a ``.jsonl``, ``.json`` (array) or ``.csv`` file (UTF-8, a BOM allowed); errors name the line |
| [write_comparison][codercommittee.io.write_comparison] | The comparison table as CSV (one row per arm) or JSON (the whole record, intervals included) |
| [write_items][codercommittee.io.write_items] | Items to ``.jsonl`` or ``.csv`` (the reference cell holds JSON) |
| [write_predictions][codercommittee.io.write_predictions] | Every arm's predictions as JSON Lines: item, arm, realization, status, value (the stored shape) |
| [write_scores][codercommittee.io.write_scores] | Per-item scores as CSV (item, arm, realization, value) or JSON (with the levels) |

`codercommittee.__version__` is the installed version.
