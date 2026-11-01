# News multi-relation extraction — v7

Read the supplied news sentence and extract every causal relation that the
reference annotators would mark, following their annotation manual. For
each relation:

- `cause` is the verbatim cause span;
- `effect` is the verbatim effect span;
- `signal` is a required audit key: the verbatim causal connective when one
  is present, otherwise `""`.

What counts as a relation:

- One argument provides the reason, explanation or justification for the
  situation described by the other. Four senses count: cause ("because",
  "since", "as", "due to", or no marker at all); purpose, where the goal is
  the cause and the action taken to reach it is the effect ("in order to",
  "so that", and "to" when it can be rewritten "in order to"); condition,
  where an unrealised situation would lead to the other if it occurred
  ("if", "as long as"); and negative condition or prevention, where a
  situation that does not occur, or that prevents or ends the other, is
  the cause ("unless", "till", "prevented"). In this kind of news the link
  is often carried by "over", "with", "by" (the means counts as the
  cause), "demanding", "seeking", "in protest", "in support of" and "in
  connection with": the demand or grievance behind a protest is its cause.
- Both arguments must be inside this one sentence. Never link to another
  sentence, and never link a span inside a quotation to one outside it;
  normally leave the reporting frame ("X said") out of the span.
- Both arguments must contain an event: something that happens or occurs,
  or a state that holds. A person, organisation or pronoun alone ("the
  police", "John", "they") is not an event; the name of an event, a policy
  or an Act ("the protest", "the housing act") is. A sentence-initial
  "This" or "That" used as an argument is accepted as a span even when
  what it refers to is not in the sentence.
- Implicit relations count. Many references have no connective: a reported
  event followed by its consequence is a relation. Do not require a marker,
  and do not treat its absence as evidence against a relation.
- Apply the manual's five tests on the words of the sentence alone: "Why
  <effect>?" is answered by "Because <cause>" (for purpose: "In order to
  <cause>"; for condition: "Why could <effect> occur?"; for prevention:
  "Why was <effect> prevented?"); the cause does not follow the effect in
  time; the effect would be less likely without the cause; the two cannot
  simply be swapped; and the pair can be rephrased as "<cause> causes
  <effect>" or "Due to <cause>, <effect>".
- Not a relation: a sequence of events in time with no causal link; a
  correlation with no direction ("coffee is linked to longevity"); a clause
  that gives the writer's or speaker's grounds for asserting or believing
  something ("let's talk later because I have a call"; "it rained
  overnight, because the ground is wet"), whereas an event that leads a
  person in the sentence to decide, believe or act is a relation; and a
  sentence that denies that the effect happened at all ("no shop was
  damaged in the march"), which carries no relation. Negation inside a
  cause or effect ("if it does not act", "unless the demands are met")
  does not block a relation.

Span rules of the manual:

- A span is the continuous stretch of words sufficient to interpret the
  relation: the event with the agent, time, location and modifiers that
  belong to it. Take the phrase that names the event, not just its head
  word. A comma ends a span only when nothing after it belongs to the same
  argument; if the words that make up one argument are separated by other
  material, take the whole continuous stretch from its first to its last
  word; if the only continuous span would overlap the other argument, do
  not report the relation.
- Within one relation the cause and the effect never overlap. Across
  relations they may: the effect of one relation is often the cause of the
  next, and the same event should then keep the same wording.
- Outcomes that happen together and are joined by "and", "or", a comma or
  a semicolon form one effect span, not several relations.
- When one outcome causes the next (A causes B, B causes C), list A→B and
  B→C as separate relations, never A→C and never AB→C; mere succession in
  time is not a chain. An infinitive that only says what someone did,
  decided, threatened or continued to do is part of that event's span,
  not a separate relation.
- The connective is reported as `signal` and left out of both spans, with
  these exceptions: a causal verb stays inside the span when it heads an
  effect that lists several outcomes ("injuring six and closing the
  market" is one effect), when it follows "which" or "that" at the start
  of the span, or when it means more than "cause" ("infuriated tenants",
  "demanding a higher wage"); a purpose preposition stays at the start of
  the goal span ("to stop the leak spreading" is the whole cause); "for"
  is purpose only when it can be rewritten "in order to get", otherwise it
  is an ordinary connective and stays out of both spans. Words of time such
  as "after", "when", "following", "during" and "in" often introduce an
  implicit cause: extract the relation, keep the word out of both spans,
  and leave `signal` as `""` or report the word; either is accepted. A
  discontinuous signal ("if ... then") may be reported by its first part.
- Punctuation at the start or end of a span is left out; punctuation inside
  a span stays.
- List every relation the sentence contains, one relation per cause–effect
  pair; a sentence may have several.

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

Use `{"relations": []}` when no pair of events in the sentence passes the
tests. Every relation object must contain `cause`, `effect`, and `signal`.

The examples below are synthetic: they are deliberately fabricated for this
prompt and are not adapted from, paraphrased from, or drawn from any benchmark
split.

Example — implicit relation, a time word that is not in any span, two
independent outcomes in one span

Sentence: `After the tunnel flooded, officials closed the road and two shops were damaged.`

```json
{
  "relations": [
    {"cause": "the tunnel flooded", "effect": "officials closed the road and two shops were damaged", "signal": ""}
  ]
}
```

Example — purpose (the goal is the cause and keeps its "to"), then a chain
listed as adjacent relations

Sentence: `Because the council cut the bus subsidy to balance its budget, fares rose within a month.`

```json
{
  "relations": [
    {"cause": "to balance its budget", "effect": "the council cut the bus subsidy", "signal": "to"},
    {"cause": "the council cut the bus subsidy", "effect": "fares rose within a month", "signal": "Because"}
  ]
}
```

Example — an event noun, a causal verb, and several outcomes kept in one
effect span with the verb inside it

Sentence: `The blockade triggered fuel shortages across the valley and forced two clinics to close.`

```json
{
  "relations": [
    {"cause": "The blockade", "effect": "triggered fuel shortages across the valley and forced two clinics to close", "signal": "triggered"}
  ]
}
```

Example — explicit connective, context kept inside the spans, reporting
frame left out

Sentence: `A union official said exports had stopped on Monday because the port had no replacement crews.`

```json
{
  "relations": [
    {"cause": "the port had no replacement crews", "effect": "exports had stopped on Monday", "signal": "because"}
  ]
}
```

Example — two events with no dependency

Sentence: `The delegates arrived at noon and the weather stayed dry all afternoon.`

```json
{"relations": []}
```

Sentence to annotate:

`{{TEXT}}`

Return only the JSON object.
