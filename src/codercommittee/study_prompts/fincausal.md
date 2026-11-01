# Financial multi-relation extraction — v7

Read the supplied financial-news passage (one to three sentences) and
extract every cause–effect relation that the reference annotators would
mark, following their annotation scheme. For each relation:

- `cause` is the verbatim cause span;
- `effect` is the verbatim effect span;
- `signal` is a required audit key: the verbatim connective when one is
  present, otherwise `""`.

What counts as a relation:

- A relation links two facts stated in the passage, one triggering the
  other. A fact is an acknowledged past event, a concrete process, a state
  of affairs, a forward-looking statement, or a projection based on facts.
- The effect is always a quantified fact: it has a direct connection to an
  explicit measure, a quantity or number that can be identified in the
  passage (an amount, a count, a percentage, a price, a share count), and
  it is at least a state of affairs. The cause may be quantified or not.
- The link may be implicit: the passage itself must present the second
  fact as following from the first (an announcement, a transaction, a
  decision, a policy change, a market move, and the figures that follow
  from it). Do not require "because", "due to" or "led to", and do not
  treat their absence as evidence against a relation; a word whose main
  role is temporal ("after", "when") can still carry the relation. Do not
  supply a cause the passage does not mention.
- A lexical causative alone is not a relation: "The insurer trimmed its
  reserves by 3%" is one fact, because the effect is inside the verb.
- A calculation is not a relation: when the second figure is merely
  computed from the first (a fund with 50 holdings and $2 billion under
  management, "or $40 million per holding"), nothing is extracted.
- The cause and the effect are at most one untagged sentence apart: in a
  three-sentence passage a fact in the first sentence may be linked to a
  fact in the third, never further.
- One cause may have several effects, and several causes one effect: emit
  one relation per cause–effect pair, repeating the shared span. Two facts
  of the same type in the same sentence that relate to the same partner
  form one span, not two; this joining applies only to facts in the same
  sentence and on the same side of the partner. When the partner sits
  between two same-type facts, they stay two separate relations.
- In a causal chain inside one sentence, isolate the rightmost fact and tag
  it by its nature; gather every remaining fact into one span of the other
  type.
- When a pronoun (not a definite noun phrase) stands for a fact, extract
  the antecedent it refers to.
- When the role of a clause is unclear between cause, means and goal, treat
  it as the cause. When only one of two facts is quantified, it is the
  effect; when both are, decide by reformulating ("X, therefore Y") and
  keep the order that reads naturally.

Span boundaries, the scheme's priority rules:

- A sentence that contains only one of the two facts is taken whole, even
  when it also carries noise or a connective.
- Linking two whole sentences is preferred to subdividing one sentence.
- A sentence that contains both the cause and the effect is split into the
  two spans, each extended to cover its side of the sentence; only a
  connective standing between the two spans is left out. A connective that
  opens the sentence or sits inside a fact stays in its span.
- No stretch of text belongs to two different spans across the relations
  of a passage: when a sentence linked whole to another sentence also
  contains an internal cause–effect pair, the internal pair is not
  emitted. The same span may be repeated in the same role across
  relations.
- A fact inserted in the middle of the other fact is a relation that is
  not extracted at all.
- A whole-sentence span keeps its final punctuation; a span cut from inside
  a sentence stops before it.
- A date that belongs to a fact and sits next to it is part of the span.

Every span must be copied from the text exactly as written, as one unbroken
stretch: keep every word between its first and last word, never shorten it
with an ellipsis ("...") or skip words, and never alter spelling,
punctuation or quotation marks. A span that is not an exact copy is
rejected.

Return exactly one JSON object:

```json
{
  "relations": [
    {"cause": "...", "effect": "...", "signal": "..."}
  ]
}
```

Use `{"relations": []}` only when no fact in the passage triggers a
quantified fact stated in the passage, for example a lone figure, a
notice, or a calculation. Every relation object must contain `cause`,
`effect`, and `signal`.

The examples below are synthetic: they are deliberately fabricated for this
prompt and are not adapted from, paraphrased from, or drawn from any benchmark
split.

Example — implicit relations across sentences, one cause with two effects
in different sentences, whole sentences kept with their punctuation

Passage: `Helios Marine Holdings declared a quarterly dividend of 12 cents
per share on Tuesday. The payout will total about $18 million. The new rate
lifts the annual payout to 48 cents per share.`

```json
{
  "relations": [
    {"cause": "Helios Marine Holdings declared a quarterly dividend of 12 cents per share on Tuesday.", "effect": "The payout will total about $18 million.", "signal": ""},
    {"cause": "Helios Marine Holdings declared a quarterly dividend of 12 cents per share on Tuesday.", "effect": "The new rate lifts the annual payout to 48 cents per share.", "signal": ""}
  ]
}
```

Example — connectives inside sentences: a connective between the spans is
left out, a sentence-opening connective stays in its span, and a span cut
from inside a sentence stops before the period

Passage: `Quarterly revenue fell 12 percent after the retailer closed forty
stores. Because of the closures, the supplier base shrank by 8 percent.`

```json
{
  "relations": [
    {"cause": "the retailer closed forty stores", "effect": "Quarterly revenue fell 12 percent", "signal": "after"},
    {"cause": "Because of the closures", "effect": "the supplier base shrank by 8 percent", "signal": "Because of"}
  ]
}
```

Example — two effects in one sentence from one cause form one span

Passage: `The port authority lost its main tenant in June. Container volumes
dropped 15 percent and 120 dock jobs were cut.`

```json
{
  "relations": [
    {"cause": "The port authority lost its main tenant in June.", "effect": "Container volumes dropped 15 percent and 120 dock jobs were cut.", "signal": ""}
  ]
}
```

Example — no relation

Passage: `The annual shareholder meeting is scheduled for the second week of
May in Frankfurt.`

```json
{"relations": []}
```

Passage to annotate:

`{{TEXT}}`

Return only the JSON object.
