"""Collecting a run: the only loop that calls a coder.

Every coder answers every item on every draw. An invalid or refused answer is asked again
inside one budget of ``attempts``, the transport attempts an adapter reports counting toward
it, as the study's runner did; the last answer's status stands, so a refusal that survives the
budget is final. Once an item is answered, the spans the item needs (the reference's and every
valid answer's) are embedded in one request, and each ``tuples`` answer is deduplicated once
with the item's similarity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .data import SPAN_KINDS, Answer, Item, Relation, Run, Task, check_coder_name
from .dedupe import dedupe
from .protocols import Coder, Embedder, Referee, describe_coder, describe_referee
from .providers.base import TRANSPORT_BUDGET
from .similarity import Similarity, Space, text_sha256
from .validation import check_reference

if TYPE_CHECKING:
    from .store import RunStore

FINAL_STATUSES = frozenset({"ok", "refused"})
RETRIED_STATUSES = frozenset({"invalid", "refused"})


def required_texts(task: Task, item: Item, answers: Sequence[Answer]) -> list[str]:
    """Every span the item's similarity must cover, unique, in sha256 order (the request order)."""
    spans: set[str] = set()
    if task.kind not in SPAN_KINDS:
        return []

    def add(relation: Relation) -> None:
        for name in task.fields:
            span = relation.get(name, "").strip()
            if span:
                spans.add(span)

    if isinstance(item.reference, tuple):
        for relation in item.reference:
            add(relation)
    for answer in answers:
        if not answer.ok:
            continue
        if isinstance(answer.value, Relation):
            add(answer.value)
        elif isinstance(answer.value, tuple):
            for relation in answer.value:
                add(relation)
    return sorted(spans, key=text_sha256)


def _transport_attempts(answer: Answer) -> int:
    """The transport attempts the adapter reports for this reply (at least one)."""
    used = answer.identity.get("attempts")
    return used if isinstance(used, int) and not isinstance(used, bool) and used >= 1 else 1


def _ask(coder: Coder, item: Item, task: Task, draw: int, attempts: int) -> Answer:
    """One budget of attempts: an invalid or refused answer is asked again until the budget is spent.

    Each call may spend at most the budget's remainder on transport attempts (the adapters read it
    from ``TRANSPORT_BUDGET``), so a logical call never costs more than ``attempts`` requests.
    """
    made = 0
    discarded: list[dict[str, Any]] = []
    while True:
        token = TRANSPORT_BUDGET.set(attempts - made)
        try:
            answer = coder.answer(item, task, draw)
        finally:
            TRANSPORT_BUDGET.reset(token)
        made = min(attempts, made + _transport_attempts(answer))
        if answer.status not in RETRIED_STATUSES or made >= attempts:
            # A served answer (store, replay) carries the count of the run that produced it.
            audit: dict[str, Any] = {**answer.audit, "attempts": answer.audit.get("attempts", made)}
            if discarded:
                audit["discarded"] = discarded  # the paid replies the budget threw away, as the study kept them
            return replace(answer, audit=audit)
        discarded.append({"status": answer.status, "reason": answer.reason, "identity": dict(answer.identity)})


def _stored_answer(
    store: RunStore | None, item: Item, coder: str, task: Task, draw: int, attempts: int
) -> Answer | None:
    """A final stored answer, re-validated from its raw reply; else None.

    ``ok`` is final; ``refused`` is final when it spent the budget it is now asked under.
    """
    if store is None:
        return None
    record = store.answer(item.id, coder, draw)
    if record is None or record.get("status") not in FINAL_STATUSES:
        return None
    if record.get("status") == "refused":
        spent = (record.get("audit") or {}).get("attempts")
        if not isinstance(spent, int) or isinstance(spent, bool) or spent < attempts:
            return None
    from .store import answer_from_record

    return answer_from_record(record, item, task)


def _stored_space(store: RunStore | None, item: Item, texts: Sequence[str]) -> Space | None:
    if store is None:
        return None
    space = store.space(item.id)
    if space is None or set(space.texts) != set(texts):
        return None
    return space


def collect(
    items: Sequence[Item],
    task: Task,
    coders: Sequence[Coder],
    *,
    draws: int = 3,
    attempts: int = 3,
    embedder: Embedder | None = None,
    referee: Referee | None = None,
    store: RunStore | None = None,
) -> Run:
    """Ask every coder every item on every draw and return the Run.

    With a ``store``, every stored answer with status ``ok``, every stored refusal that spent its
    budget, and every stored space whose texts are the spans asked for is reused and never
    re-requested; each new answer is appended before the next call, so an interrupted run resumes
    from where it stopped. The run's provenance records the referee and each seat's identity, and
    a resume whose seats hold other models is refused.
    """
    if draws < 1 or attempts < 1:
        raise ValueError("draws and attempts are at least one")
    names = tuple(coder.name for coder in coders)
    if not names or len(set(names)) != len(names):
        raise ValueError("coders need distinct names")
    for name in names:
        check_coder_name(name)
    if task.kind in SPAN_KINDS and embedder is None:
        raise ValueError(f"a {task.kind} task needs an embedder for its spans")
    for item in items:
        check_reference(task, item.reference)
    identities = {coder.name: record for coder in coders if (record := describe_coder(coder)) is not None}
    provenance: dict[str, Any] = {"coder_identities": identities} if identities else {}
    referee_record = describe_referee(referee)
    if referee_record is not None:
        provenance["referee"] = referee_record
    if store is not None:
        store.create(task, names, draws, items, provenance)
    answers: dict[tuple[str | int, str, int], Answer] = {}
    spaces: dict[str | int, Space] = {}
    for item in items:
        item_answers: list[Answer] = []
        for coder in coders:
            for draw in range(1, draws + 1):
                answer = _stored_answer(store, item, coder.name, task, draw, attempts)
                if answer is None:
                    answer = _ask(coder, item, task, draw, attempts)
                    if answer.item != item.id or answer.coder != coder.name or answer.draw != draw:
                        raise ValueError(f"coder {coder.name!r} answered for another item, coder or draw")
                    if store is not None:
                        store.put_answer(answer, task)
                item_answers.append(answer)
        if embedder is not None:
            texts = required_texts(task, item, item_answers)
            space = _stored_space(store, item, texts)
            if space is None:
                space = embedder.space(item, texts)
                if store is not None:
                    store.put_space(item.id, space)
            spaces[item.id] = space
            if task.kind == "tuples":
                sim = Similarity(space, fields=task.fields, aux=task.aux)
                item_answers = [
                    replace(answer, value=dedupe(answer.value, sim))
                    if answer.ok and isinstance(answer.value, tuple)
                    else answer
                    for answer in item_answers
                ]
        for answer in item_answers:
            answers[(item.id, answer.coder, answer.draw)] = answer
    return Run(task, tuple(items), names, draws, answers, spaces, referee, provenance)


__all__ = ["collect", "required_texts"]
