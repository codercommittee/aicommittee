# AI-patent classification — v7

## Task

You are given the abstract of a US utility patent. Decide whether the patent
is related to AI technologies under the definition below, which is the one
the reference labels follow.

The task: identify AI-related technologies. An AI technology is any system
that perceives its environment and predicts an action that maximizes its
chance of achieving its goals; a learned classifier, detector or predictor
that outputs a judgment rather than an action counts in the same way. AI
technologies are those employing some form of statistical learning: methods
that learn patterns or relationships from data in order to predict,
classify, recognise, recommend, or decide from data. Most AI-related
patents are prediction tasks.

The rules:

- If a patent has **some element** of AI technology, it is AI-related. The
  test is the presence of such an element, not whether it is the main
  novelty: an autonomous vehicle that uses a reinforcement-learning
  approach, a greenhouse controller that trains a network to predict
  moisture, a recommendation system that learns from user behaviour, or a
  traffic forecast fitted to historical flow data is AI-related.
- Recognise learning even when no AI keyword appears. Many such patents
  never say "artificial intelligence" or "deep learning"; they describe a
  system that learns from data, is trained, fits a model, or improves its
  predictions with experience. Infer from the description of the method
  whether it relies on any such technology.
- The methods that count: machine learning, neural networks, deep
  learning, reinforcement learning, supervised and unsupervised learning,
  Bayesian belief networks, statistical prediction, machine intelligence;
  pattern recognition, image matching and machine vision; natural language
  processing and speech processing. Prediction or recommendation from
  historical or behavioural data counts even when the abstract describes
  it as profiles, frequencies, scores or fitting rather than training.
  Evolutionary or genetic algorithms and fuzzy inference count only when
  they fit or learn from data.
- Not AI-related: a fixed rule-based or traditional control system with no
  learned or predictive component (a cruise control that adjusts speed by
  a fixed rule), knowledge processing or expert systems that manipulate
  stored rules without learning from data, fixed formulas or descriptive
  tallies that predict or recommend nothing, and a term used in a non-AI
  sense (a "neural" graft in biology, a "classifier" screen that sorts
  particles by size, a biological "pattern recognition receptor"). A
  keyword alone does not make a patent AI-related.
- Decide from the abstract alone. Infer learning from what the method does
  (fits, adapts, updates, profiles or predicts from accumulated data), not
  from the field of application: a vehicle or camera patent whose method
  does none of this is not AI-related. Every patent belongs to exactly one
  class; a described learning or predictive component makes the patent
  AI-related however small its share, and "the more appropriate class"
  decides only when it is unclear whether any component learns at all.

Return the binary judgment, with no explanation.

## Output format

Return exactly one JSON object with one key:

```json
{"is_ai_patent": true}
```

`is_ai_patent` must be a JSON boolean (`true` or `false`).

## Synthetic format examples

The abstracts below are deliberately fabricated for this prompt. They are not
adapted from, paraphrased from, or drawn from any patent or benchmark split.

### Example 1 — AI stated by name

**Abstract:**
> A greenhouse controller trains a recurrent neural network on temperature,
> humidity, and valve histories, then uses the trained network to predict the
> next-hour moisture level and select an irrigation command.

**Output:**
```json
{"is_ai_patent": true}
```

### Example 2 — AI as one element, described without AI words

**Abstract:**
> A delivery scheduling system assigns vans to routes. Expected travel times
> for each road segment are estimated from a model fitted to the fleet's
> recorded trips and updated as new trips are completed, and routes are
> assigned to minimise the predicted total time.

**Output:**
```json
{"is_ai_patent": true}
```

### Example 3 — AI described as profiles and frequencies, with no learning word

**Abstract:**
> A content service records which categories of items each account opens
> over a period, computes the frequency of each category for that account,
> keeps the categories whose frequency exceeds a threshold, and recommends
> new items whose categories match the account's retained profile.

**Output:**
```json
{"is_ai_patent": true}
```

### Example 4 — not AI

**Abstract:**
> A folding crate includes four hinged walls, a locking floor plate, and
> spring-loaded corner pins arranged to hold the walls upright during use and
> release them for storage.

**Output:**
```json
{"is_ai_patent": false}
```

### Example 5 — fixed rules and a term used mechanically

**Abstract:**
> A fluid separator routes particles through a classifier screen whose fixed
> apertures divide material by diameter, and a controller opens a bypass
> valve whenever the measured pressure exceeds a preset threshold.

**Output:**
```json
{"is_ai_patent": false}
```

## Now classify

Abstract:
> {{TEXT}}

Respond with only the JSON object.
