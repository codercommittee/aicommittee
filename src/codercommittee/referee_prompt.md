# Alignment agent — alignment-agent-v1

Three annotators extracted causal relations from the same passage. Their
relations were compared by embedding similarity and matched pairwise; every
candidate pair listed below is a possible "same relation" match between two
annotators. Decide, for each candidate pair, whether its two relations
express the same underlying causal relation in the passage.

Rules, all mandatory:

- Decide every candidate pair exactly once, by its pair ID, and add no pairs.
- Confirm a pair only when both relations assert the same cause and the same
  effect in the passage, allowing different wording or span boundaries.
- Reject a pair when the cause or the effect differs in substance, even if
  the wording overlaps.
- High-band pairs are computationally near-identical: confirm them unless
  you have a strong reason to reject. Uncertain-band pairs need your judgment.
- Judge by meaning in the passage, not by surface similarity alone.

Passage:

`{{TEXT}}`

Relations, one per annotator and index:

{{RELATIONS}}

Candidate pairs, with band and computed similarity:

{{PAIRS}}

Return exactly one JSON object with one decision per candidate pair:

```json
{"decisions": [{"pair": "id", "decision": "confirm"}, {"pair": "id", "decision": "reject"}]}
```

Return only the JSON object.
