"""The facade: a committee of coders codes items and returns a Run; the Run's methods reach the core.

committee = Committee([coder_a, coder_b, coder_c], embedder=MockEmbedder(), referee=BandRule())
run = committee.code(items, task)            # three draws per coder and item
run.arms()                                   # every structure's predictions
run.scores()                                 # item x arm x realization scores, with levels
compare([run.scores(), other.scores()], seed=42)
"""

from __future__ import annotations

from collections.abc import Sequence

from .collect import collect
from .data import Item, Run, Task
from .protocols import Coder, Embedder, Referee


class Committee:
    """Several coders asked the same items; the embedder and the referee the span kinds need."""

    def __init__(
        self, coders: Sequence[Coder], embedder: Embedder | None = None, referee: Referee | None = None
    ) -> None:
        if not coders:
            raise ValueError("a committee needs at least one coder")
        self.coders = tuple(coders)
        self.embedder = embedder
        self.referee = referee

    def code(self, items: Sequence[Item], task: Task, *, draws: int = 3, attempts: int = 3) -> Run:
        """Ask every coder every item ``draws`` times and return the Run (one cycle)."""
        return collect(
            items, task, self.coders, draws=draws, attempts=attempts, embedder=self.embedder, referee=self.referee
        )

    def __repr__(self) -> str:
        names = ", ".join(coder.name for coder in self.coders)
        return f"Committee([{names}], embedder={self.embedder!r}, referee={self.referee!r})"


__all__ = ["Committee"]
