# Design

`codercommittee` re-implements the aggregation algorithms of the study *Three Models Walk Into a Sentence* as a reusable, task-agnostic library. The study pipeline is the reference implementation and its stored runs are the oracle: the library proves itself by reproducing the stored predictions and numbers item by item (see `parity/`). This document is the protocol: the data model, the operations, their constants, and the rules that decide ties, missing answers and numerical identity.

## 1. Six design rules

Each is the pipeline's own factoring, stated once.

1. **One operator.** The four structures (one answer; one coder asked three times; the majority vote with alignment; the pool at k of nine) are one `combine` operator: select members, align, keep what at least k members support, represent each kept unit by its medoid, score. The four names are presets and the keys of every table.
2. **Alignment returns every unit**, singletons included, with its support. The keep rule is applied after, never inside unit construction.
3. **Vectors are scoped to one item of one run.** The same text may have different vectors elsewhere; no cache crosses items or runs.
4. **Typed status.** Answers and predictions carry a status. A missing answer (invalid or refused after its budget of attempts, or absent) removes the item from every arm of that run. A missing reference leaves the item unscored, never zero.
5. **A parameter with one value in the study is a constant.** Deduplication at 0.90 with the 0.85 floor; candidate pairs from 0.60; the vote's band at 0.80; the pool's band at 0.70; the match threshold 0.60; twelve-decimal rounding; the enumeration guard of 24 edges; value tolerance 1e-9; tie tolerance 1e-12; three coders and three draws at the presets; percentiles 2.5 and 97.5. The bootstrap draws (2000) and the pick draws (500) are keyword defaults so tests can run smaller.
6. **An estimate never travels without its interval.** Level, gap, interval, blind pick, spread and premium come from one `compare` call.

## 2. Data model (`codercommittee.data`)

```
Relation(**spans)                  the one structured atom: an immutable mapping field -> stripped span;
                                   field names come from the Task; attribute access (r.cause)
Task(kind, codebook, prompt, key, per_answer=(1, 1), fields=("cause", "effect"), aux=("signal",))
    kind        "label" | "labels" | "tuple" | "tuples"
    codebook    {label name: description}; one entry for "label" (the boolean's name), the vocabulary
                for "labels", empty for the span kinds
    key         the answer's JSON key: the label's name, "labels", "relations", or the flag key "causal"
    per_answer  (low, high) labels per answer for "labels"; the cap on an aggregated prediction is
                floor(n * high / k), never a parameter
    fields      span fields similarity reads; aux: span fields carried, keyed and printed, never compared
    schema()    the portable JSON schema (object, array, string, boolean, enum; no extra properties)
Item(id, text, reference=None)     reference: bool | frozenset[str] | tuple[Relation, ...]; for the "tuple"
                                   kind an empty tuple means non-causal and a causal reference holds every
                                   retained coder tuple; None = no reference (coded, compared, never scored)
Answer(item, coder, draw, value, status, reason, raw, identity, audit)
    status      "ok" | "invalid" | "refused" | "missing"
    value       bool | frozenset[str] | Relation or None ("tuple": None with status ok is the negative
                answer) | tuple[Relation, ...] ("tuples": deduplicated once, in candidate-key order)
    identity    opaque: requested and returned model, request fingerprint, ...; the replay coder checks
                it, live adapters fill it
Member(name, index, relation, coder)   one member's candidate inside an alignment
Unit(members, representative)     medoid = representative.relation; support = len(members); coders = distinct
                                   coders among the members
Prediction(value, status, units, audit)   status "ok" | "insufficient"
Run(task, items, coders, draws, answers, spaces, referee, provenance)
    items in run order; coders in seat order (the order given to collect); answers keyed by
    (item id, coder, draw); spaces keyed by item id; provenance = the referee's record and each
    seat's identity (what `describe()` gives), written into run.json by a store
    complete()  items with every coder x draw at status ok
    scored()    complete items with a reference the level uses ("tuple": a causal one)
ItemScore(value, confusion, matched)   confusion kept for the binary label, matched pairs for relations
```

## 3. Protocols (`codercommittee.protocols`)

```
Coder.answer(item, task, draw) -> Answer      MockCoder | ReplayCoder(run_dir, coder) | ModelCoder(name, model)
Model.complete(prompt, schema, *, system) -> Completion(raw, identity, error, refused)
                                              the provider adapters (Anthropic, OpenAI, Google, OpenAI-
                                              compatible endpoints such as vLLM); ModelCoder and the LLM
                                              referee call them
Embedder.space(item, texts) -> Space          MockEmbedder | OpenAIEmbedder | ReplayEmbedder(run_dir)
Similarity(space, fields, aux).pair(a, b)     mean over fields of the clamped cosine; fsum; 12 places;
                                              both empty 1.0, one empty 0.0; every threshold is >=
Referee.decide(prematch) -> {pair: bool}      BandRule(accept_at=0.80) | LLMReferee(model) | ReplayReferee(run_dir);
                                              accept_at is the band boundary of the pre-match it decides
```

Replay adapters raise on absent evidence and are never retried; a stored refusal is final.

## 4. Core operations

| Function | Rule | Re-implements (study pipeline) |
|---|---|---|
| `collect(items, task, coders, draws=3, attempts=3, embedder, referee, store)` | the only loop that calls a coder; an invalid or refused answer is asked again inside one budget of `attempts`, the adapter's transport attempts counting toward it and each call capped at the budget's remainder; a refusal that survives the budget is final (`is_provider_refusal` after `MAX_TRANSPORT_ATTEMPTS`); the discarded replies recorded in the audit; with a store, final answers and matching spaces reused, the referee's record (adapter settings, endpoint, template and system prompt hashes) and each seat's identity checked against the manifest and, for a seat the manifest never recorded, against the stored answers' own model and endpoint; one embedding request per item over the reference's spans and every valid answer's spans ("tuple": causal only), unique, in sha256 order; "tuples" answers deduplicated once | `run_study` item loop, `embeddings.required_span_texts` |
| `dedupe(relations, sim)` | Union-Find at 0.90; a cluster of three or more whose weakest pair is below 0.85 splits into singletons; each cluster's medoid; output in candidate-key order | `panel_prediction._deduplicate_relations`, `prepare_relation_predictions` |
| `prematch(members, sim, accept_at, ...)` | for every member pair, a maximum-weight one-to-one matching over edges >= 0.60: branch-and-bound with 1e-9 tolerance and the lexicographically smallest content edge list among optima; above 24 candidate edges the content-lexicographic greedy, recorded; each edge banded high (>= accept_at) or uncertain and flagged inside an inconsistent group | `_canonical_pair_matching`, `_content_greedy_matching`, `_alignment_edge_graph`, `alignment_prematch` |
| `align(members, sim, referee, ...)` | the referee decides every candidate pair; kept edges form components; while a component holds two candidates of one member or lacks a within-pair edge, the weakest edge by (similarity, sorted content keys of its ends) drops, every exact tie with it; every component becomes a Unit, singletons included, represented by its medoid | `_alignment_panel`, `_resolve_units`, `_component_is_unit`; `construction_study.units_with_support` |
| `medoid(members, sim)` | rank by (-(plain sum of pair similarities to the others, node order) / len(others), candidate key); alone: itself | `_medoid` |
| `combine(members, task, k, p=1, sim, referee, ...)` | any member not ok: insufficient. label: True iff count(True) >= k. labels: labels with count >= k, sorted. tuple: positive iff positive votes > n/2 (k must equal n//2 + 1; no other quorum rule exists), then the medoid of the positive members. tuples: align, keep units with support >= k and coders >= p, medoids in candidate-key order | `build_panel_prediction`, `_majority`, `_politicause_panel`, `pool_analysis._flat_pool` |
| `arms(run, referee, pool_k=(1, 2, 3))` | the presets of section 5 over `run.complete()`; refuses coder or draw counts other than three | `item_derivation.derive_item_result`, `pool_analysis.flat_pool_scores` |
| `score(prediction, reference, task, sim)` | section 6 | `scoring.score_*` |
| `Scores`, `compare`, `agreement`, `pick_on_sample` | section 7 | `derive._cycle_gaps`, `_regret`, `_validation_simulation`, `analysis._disagreement`, `panels._composition` |

The candidate key of a member is (casefolded spans over fields then aux, the exact spans, index, member name); it orders deduplicated relations, fixes node indices and pair names, and breaks medoid ties.

## 5. Arms

| Arm key | Members (name -> answer) | k, p | Referee | Realizations |
|---|---|---|---|---|
| `single:<c>` | `{c: answer(c, r)}` | 1, 1 | none | three (r = 1, 2, 3) |
| `self:<c>` | `{"1": a(c, 1), "2": a(c, 2), "3": a(c, 3)}` | 2, 1 | the run's; label `self:<c>` | one |
| `vote` | `{c: answer(c, r)}` for every coder | 2, 1 | the run's; label `panel:<r>` | three |
| `pool:<k>` | `{"<c>:<r>": answer(c, r)}`, all nine | k, 1 | `BandRule(0.70)`, a constant | one |

Task-by-arm support:

| Kind | single | self | vote | pool | alignment | similarity needed |
|---|---|---|---|---|---|---|
| label | yes | yes (majority of draws) | yes | no | exact equality | no |
| labels | yes | yes (count >= 2) | yes | no | exact equality | no |
| tuple | yes | yes (majority positive, medoid) | yes | no | none | medoid and scoring |
| tuples | yes | yes | yes | yes, k = 1, 2, 3 | pairwise matching, units | dedupe, align, medoid, scoring |

Member names: the coder id in the vote, the draw string in the self-ensemble, `coder:draw` in the pool. Inside alignment, members iterate in sorted-name order; the seat order (the coder order given to `collect`) is a different order, used only for best-single ties. A draw is one answer of one coder (1 to 3); a realization is one application of `combine`; an arm's item value is the mean over its realizations; a Run is one cycle; `compare` takes several Runs over the same items. The referee is a property of the `arms` call: the vote with the LLM referee and the vote with the band rule (the study's `panel:agent` and `panel:det`) are two calls.

## 6. Scoring

| Kind | Reference | `ItemScore.value` | Kept |
|---|---|---|---|
| label | bool | accuracy (1 or 0) | the confusion cell, for macro-F1 |
| labels | frozenset[str] | Jaccard: shared over named; both empty 1, one empty 0 | |
| tuple | tuple[Relation, ...]; () = negative | causal reference: 0 when the prediction is negative, else `np.mean` of the pair similarity over the references; negative reference: 1 when the prediction is negative, else 0 (outside the level) | |
| tuples | tuple[Relation, ...] | matched mass over max(n_pred, n_ref): an exact maximum assignment on the thresholded (>= 0.60, else 0) zero-padded square matrix, matches >= 0.60 kept; per field the sum of the unrounded cosines over the denominator, then the mean over fields; empty against empty 1, one side empty 0 | the matched pairs |

The exact assignment is a Hungarian solver written here (the pipeline calls scipy on the cost matrix `max - value`). scipy's choice among equal optima is unspecified, so parity asserts the matched index set and the match count and, where the sets differ, equal values. The greedy evaluator of the study's sensitivity check is out of scope.

## 7. Level, compare, agreement, pick (`codercommittee.compare`)

- `Scores`: item x arm x realization per Run, items in run order, with the references' labels for the binary kind. `level(arm)`: `np.mean` over items per realization, then over realizations; binary label: macro-F1 per realization from pooled confusion counts, then the mean.
- `compare([scores...], draws=2000, seed)`, one call: per arm the level per run, its mean and sd (ddof = 1; None with one run); the best single per run is the first maximal `single:` arm in seat order; the gap per run is the level minus the best single's; the gap's mean, sd and worst run; the interval is the 2.5 and 97.5 percentile of the item-paired bootstrap on run-averaged item differences over the items common to every run in `sorted(str(id))` order: `np.random.default_rng(seed)`, `rng.integers(0, n, size=n)` per draw, one fresh generator per interval, the same draws for every arm; binary label: strata in `sorted(set(labels))` order, `rng.choice(indices, size, replace=True)` per stratum, concatenated, the plan built from the first run's labels in run order and reused after asserting identical item lists, macro-F1 recomputed per resample from each run's per-realization confusion. The blind pick is the mean gap of the singles per run, its mean and its worst. The spread is the mean over runs of the best single level minus the mean over runs of the mean member; an arm's premium is its mean level minus the mean member, with its worst run; gap = premium - spread. An arm that is the best single in every run has no gap.
- `agreement(run)` over `run.complete()`: cross (share of item realizations whose members' values differ), within per coder (share of items whose draws differ), pairwise (share identical per coder pair); for "tuples" also the graded block: mean pairwise matched Jaccard at 0.60 within and across coders, and relation-count ranges.
- `pick_on_sample(scores, n, among, draws=500, seed)`: `default_rng(seed + n)`, `rng.choice(total, size=n, replace=False)`, items in run order, item values the realization means; the option with the highest sample mean is picked, ties within 1e-12 split evenly with no extra random number; the held-out gap to the run's best single, the share of draws that pick it, and the blind gap on the same held-out items; refused when n > total // 2.

## 8. Missing answers, ties, support

- An answer with status other than ok removes its item from every arm of that run. A reference of None keeps the item in `complete()` (agreement) and out of `scored()`.
- Ties: best single by seat order; medoid by candidate key; weakest edges with equal keys drop together; the pick splits sample ties evenly; levels are compared exactly and reported, never broken.
- `Unit.support` counts members, `Unit.coders` distinct coders; keep when support >= k and coders >= p. The presets support three coders and three draws only and refuse other counts with a clear error; `combine` accepts any member count.

## 9. Numerical identity

Cosine: `math.fsum` on the dot product and both squared norms, `round(max(cos, 0), 12)`, memoised by the ordered pair of text hashes; a vector whose largest entry lies outside [1e-70, 1e70] is scaled by a power of two before the sums (exact, and never the case for an embedding model's vectors, so the study's sums run on the original lists); a non-finite cosine raises. Pair similarity: `round(fsum(field cosines) / n_fields, 12)` on stripped spans. Dedupe output order and medoid ranking as in section 4. Matching ported verbatim with its guard and tolerance. Relation score: per field `sum / denominator`, then the mean over fields (the pipeline's `0.5 * (cause/d + effect/d)`). `np.mean` for the tuple kind's reference mean and for every level; `np.percentile` (linear) for intervals. The pre-match sha256 is taken over `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` of the dictionary the pipeline stores (float thresholds; nodes with node, member_id, index, relation; edges with pair, a, b, similarity, `<field>_similarity`, band, in_conflict, sorted by pair; candidate_pairs; pairwise_match_counts keyed `a|b`; enumeration_fallback_pairs; conflict_components). Item orders: run order for the pick and the binary plan; for cross-run gaps the items common to every run by typed id, ordered by `str(id)` (the pipeline's `sorted(str(id))`, which coincides with it whenever no two ids share a text); the pipeline's canonical-JSON order for in-run intervals coincides with it on ids of one type.

## 10. Replay (`codercommittee.replay`)

A run directory holds `per_item.jsonl` (run order; the item index is the position), `summary.json`, `canary_pin.json` when the run pinned returned identities, and `checkpoints/responses/{index:06d}_{slot:02d}_r{draw}_{coder}.json`, `checkpoints/embeddings/{index:06d}.json`, `checkpoints/agent/{index:06d}_{label}.json` (`panel:1` as `panel_1`). `load_run(run_dir, task, items)` builds the Run through `ReplayCoder`, `ReplayEmbedder` and `ReplayReferee` with `collect(attempts=1)`.

`ReplayCoder` makes an answer invalid when: the requested model differs from the run's model for the slot; the returned model is missing or differs from the pinned identity; `request_fingerprint` differs from the execution binding's or from the sha256 of its canonical envelope; cache namespace or prompt version differ from the run's; attempts are outside 1..3; `raw_response` is not strict JSON (no duplicate keys, no NaN, no lone surrogates) or `parsed_response` differs from it; validation fails; content-retry discards do not share the final record's fingerprint, namespace, prompt version and model, do not fit inside the attempt budget, or would have validated. A `provider_error` that is a refusal (`provider_refusal`, `provider_incomplete_finish:refusal`, `*:content_filter`) gives status refused; any other provider error gives invalid. `ReplayEmbedder` requires the stored vector set to equal the spans asked for. `ReplayReferee` requires a record iff the pre-match has candidate pairs, the stored `prematch_sha256` to equal the recomputed hash, and decisions re-parsed from `raw_response` that name every candidate pair exactly once and nothing else.

## 11. Module map

| Module | Contents | Study reference |
|---|---|---|
| `data.py` | Relation, Task, Item, Answer, Member, Unit, Prediction, Match, ItemScore, Run | `validation.ValidationResult`, `panel_prediction.PanelPredictionResult`, per-item rows |
| `validation.py` | `validate`, `verbatim` (the six-character fold), `check_reference`, `prediction_cap`, `check_prediction` | `validation.validate_response`, `validate_panel_prediction` |
| `similarity.py` | `text_sha256`, `Space`, `Similarity` | `embeddings.EmbeddingStore`, `SpanSimilarity` |
| `dedupe.py` | `relation_key`, `member_key`, `medoid`, `dedupe` | `panel_prediction._deduplicate_relations`, `_medoid`, `_candidate_key` |
| `matching.py` | branch-and-bound pairwise matching, greedy fallback | `_canonical_pair_matching`, `_content_greedy_matching` |
| `align.py` | `Prematch` with its sha256, `prematch`, `align` | `alignment_prematch`, `_alignment_panel`, `_resolve_units` |
| `referee.py` | `BandRule`, `LLMReferee`, `ReplayReferee` | `alignment_agent` |
| `combine.py` | the operator | `build_panel_prediction` |
| `arms.py` | the presets | `item_derivation.derive_item_result`, `pool_analysis._flat_pool` |
| `assignment.py` | exact maximum assignment | `alignment.hungarian_maximize` (scipy) |
| `scoring.py` | `score`, `macro_f1` | `scoring` |
| `compare.py` | `Scores`, `compare`, `agreement`, `pick_on_sample` | `derive`, `analysis`, `panels` |
| `collect.py` | `collect`, `required_texts` | `run_study`, `embeddings.required_span_texts` |
| `coders.py` | `MockCoder`, `ModelCoder`; each describes its seat (model, seed or settings, the system prompt's hash) for the run's provenance | `models.MockClient`, `call_model` |
| `providers/` | the four `Model` adapters | `models._call_*` |
| `embedders.py` | `MockEmbedder` (the study's mock byte for byte, with a one-hot fallback where that mock raised on a cancelled vector), `OpenAIEmbedder` (batched above 2,048 spans), `OpenAICompatibleEmbedder`, `FunctionEmbedder` | `embeddings.MockEmbeddingClient`, `embed_texts` |
| `replay.py` | `StoredRun`, `ReplayCoder`, `ReplayEmbedder`, `load_run` | `run_study.row_similarities`, `item_derivation._validated_response_record` |
| `strictjson.py` | `strict_json_loads`, `canonical_json` | `models.strict_json_loads` |
| `committee.py` | `Committee`; the Run's `arms`, `predictions`, `units`, `scores`, `agreement`, `save`, `load`, `to_frame` live on `Run` | |
| `io.py` | items from JSON Lines, JSON and CSV; writers for predictions, scores, levels, the comparison table, agreement, the pick; `to_json` | |
| `store.py` | the package's run layout (`run.json` with the task, items, coders and provenance; `answers.jsonl`; `spaces/`; `decisions.jsonl`); `RunStore`, `StoreCoder`, `StoreEmbedder`, `LoggedReferee` (serves a recorded decision bound to the pre-match's sha256 before asking), `StoreReferee`, `save_run` (refuses a directory holding other answers, vectors or decisions; copies recorded decisions), `load_run` and `stored_referee` (the recorded band rule at its threshold, else the recorded decisions); `collect(store=...)` resumes, drops and repairs a torn final line, writes snapshots atomically, refuses another referee or another model under a seat | the study's checkpoints, in the package's own shape |
| `study.py` + `study_prompts/` | the five study tasks with their prompts verbatim, seeds, references from the study's gold shapes | `code/datasets/<d>/prompt.md`, `study_spec.json` |
| `cli.py` | `codercommittee code / score / compare / agreement / pick / tasks` | |
| `providers/base.py` | the transport block, retry classification, request fingerprints | `models._retry_transport`, `classify_retry` |

Outside the package: `parity/export_items.py` (a study dataset's items through the study's loaders, into the package's items file), `parity/build_fixture.py` (the portable fixture from the study's mock runs), `parity/full_cycles.py` (parity against the stored cycles, run from the study checkout: answers, the packaged task prompt against every stored request, structures under the band rule and under the stored decisions, the referee prompt byte for byte, the pool at every k, and every derived number and printed macro), `tests/`, `quickstart.py`.

## 12. Where the appendix and the pipeline differ

The pipeline is the truth and the library follows it:

1. Appendix section 4.1 describes the PolitiCAUSE vote as "the observed tuple that is the embedding medoid of the three"; the pipeline takes the medoid of the positive members only.
2. Section 4.2 calls the pairwise pre-match "Hungarian assignment"; the pipeline enumerates with branch-and-bound and content-canonical ties and falls back to a content-lexicographic greedy above 24 candidate edges. The value is the same; the choice among tied optima is not.
3. Section 4.2 says a chained cluster whose weakest pair falls below the floor splits; the pipeline splits clusters of three or more only (a pair that joined at 0.90 cannot fall below 0.85, so the two agree).
4. Section 4.3 reports signal spans by token F1 as an audit; the library drops the audit.
5. The pipeline lists a pool prediction's consensus units before its singletons (the order `units_with_support` builds); the library sorts every prediction by candidate key, as the vote already does. The parity checks compare the pool as a multiset; units, supports, medoids and scores agree.

In the study code itself (reported, not fixed; the library follows the first-named behaviour): best-single ties fall to seat order in `analysis.py` and to alphabetical order in `construction_study.load_base_run`; scoring uses scipy's assignment with unspecified tie choice while alignment uses the canonical branch-and-bound; `units_with_support` counts coders by splitting a member name on ":"; the keep rule is fused into unit construction and the pool rebuilds singletons from the audit.

## 13. Out of scope in this version

Budgets, authorizations, canary pins, the offline verifier of the study's layout, the figures build, benchmark loaders (the study's items are exported through the study's own loaders), concurrency, weighted voting, the composition analysis (swaps, insurance, shared errors), the greedy evaluator and threshold sweeps, the win table, costs. Runs are saved and resumed in the package's own layout (section 11, `store.py`).

The mock embedder lowercases its text with the interpreter's Unicode tables, as the study's mock does, so a character whose case mapping a newer Unicode version added (two in Unicode 16) embeds differently on Python 3.14 than on 3.11. It is a stand-in for an embedding model, not a measurement, and it is left as it is.
