"""codercommittee quickstart: the four structures end to end on mock coders, in seconds.

Three mock coders extract cause-effect relations from eight short passages, three times each.
The committee combines the nine answers per passage as one answer, one coder asked three times,
the majority vote with alignment, and the pool at k of nine; scores every structure against the
human reference; and compares two runs in one table. Swap the mock coders for AnthropicModel,
OpenAIModel, GoogleModel or an OpenAI-compatible endpoint, and the mock embedder for
OpenAIEmbedder, to run it on real coders.

    python quickstart.py
"""

from __future__ import annotations

import time

from codercommittee import (
    BandRule,
    Committee,
    Item,
    MockCoder,
    MockEmbedder,
    Relation,
    Task,
    compare,
    pick_on_sample,
)

TASK = Task(
    "tuples",
    prompt=(
        "Read the passage and list every cause-effect relation it states, as verbatim spans.\n\n"
        'Passage: {{TEXT}}\n\nReturn JSON: {"relations": [{"cause": ..., "effect": ..., "signal": ...}]}'
    ),
)

PASSAGES = [
    (
        "heavy rain filled the river so the bridge washed out before dawn",
        ("heavy rain filled the river", "the bridge washed out"),
    ),
    (
        "the council cut the bus subsidy and evening services stopped across the valley",
        ("the council cut the bus subsidy", "evening services stopped"),
    ),
    (
        "frost hit the orchard early and the cider press stood idle all winter",
        ("frost hit the orchard early", "the cider press stood idle"),
    ),
    (
        "the mill closed in march so forty workers left the town for good",
        ("the mill closed in march", "forty workers left the town"),
    ),
    (
        "a new tariff raised import prices which pushed the bakery to close its second shop",
        ("a new tariff raised import prices", "the bakery to close its second shop"),
    ),
    (
        "early warnings can prevent a harvest collapse when the frost arrives in april",
        ("early warnings", "a harvest collapse"),
    ),
    (
        "the harbour was widened so the larger ferries can dock in winter",
        ("the harbour was widened", "the larger ferries can dock"),
    ),
    ("the strike over a fare hike stopped the trams for a week", ("the strike over a fare hike", "stopped the trams")),
]
ITEMS = [Item(n, text, [Relation(cause=cause, effect=effect)]) for n, (text, (cause, effect)) in enumerate(PASSAGES)]


def main() -> None:
    started = time.perf_counter()
    first = Committee([MockCoder("alpha"), MockCoder("beta"), MockCoder("gamma")], MockEmbedder(), BandRule(0.80))
    run = first.code(ITEMS, TASK)
    print(f"run 1: {len(run.complete())} of {len(run.items)} items complete, {len(run.answers)} answers\n")

    item = run.items[0]
    print(f"item {item.id}: {item.text!r}")
    print(f"  reference : {[r.as_dict() for r in item.reference]}")  # type: ignore[union-attr]
    for arm, by_item in run.arms().items():
        for index, prediction in enumerate(by_item[item.id], start=1):
            relations = [(r.cause, r.effect) for r in prediction.value]  # type: ignore[union-attr]
            print(f"  {arm:<13} realization {index}: {relations}")
    print()

    scores = run.scores()
    print("levels of run 1:")
    for arm, level in scores.levels().items():
        print(f"  {arm:<13} {level:.3f}")
    print()

    second = Committee(
        [MockCoder("alpha", seed=1), MockCoder("beta", seed=1), MockCoder("gamma", seed=1)],
        MockEmbedder(),
        BandRule(0.80),
    )
    other = second.code(ITEMS, TASK).scores()
    table = compare([scores, other], seed=42)
    print("two runs compared (gap = level minus the best single coder of the same run):")
    print(table)
    print()

    agreement = run.agreement()
    print(f"cross-coder disagreement {agreement.cross:.2f}; within-coder {agreement.within}")
    pick = pick_on_sample(scores, 4, seed=42)
    print(
        f"a labelled sample of {pick.n} items picks the best coder in {pick.pick_share:.0%} of {pick.draws} draws; "
        f"held-out gap {pick.heldout_gap:+.3f}, blind pick {pick.blind_gap:+.3f}"
    )
    print(f"\ndone in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
