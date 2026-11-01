"""The study's five tasks, with the prompts the study sent, as package data.

``TASKS`` holds one :class:`~codercommittee.data.Task` per dataset of *Three Models Walk Into a
Sentence*: Miric AI patents (one boolean), GVFC headlines (one or two of nine frames),
PolitiCAUSE sentences (a causal flag with cause, effect and signal), FinCausal sections and
CausalNews sentences (every cause-effect relation). Each prompt is the study's own file,
verbatim, with the one placeholder ``{{TEXT}}``; the committee sends it as the user turn under
the package's constant system line. The datasets themselves are not bundled (their licences
stay with their publishers); see ``docs/study.md`` for how to fetch them and export their items.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .data import Item, Reference, Relation, Task

PROMPTS_DIR = Path(__file__).with_name("study_prompts")
DATASETS = ("miric", "gvfc", "politicause", "fincausal", "causalnews")
# The study's bootstrap seed per dataset.
BOOTSTRAP_SEED = {"miric": 42, "gvfc": 42, "politicause": 42, "fincausal": 42019, "causalnews": 42017}
GVFC_CODEBOOK = {
    "gun_rights": "the Constitution, the second amendment, gun ownership as an individual right",
    "gun_control": "regulating guns through legislation and other institutional measures",
    "politics": "the political issues around guns and shootings: campaigns, parties, politicians, lobbies",
    "mental_health": "mental health as a cause of, or a remedy for, gun violence",
    "school_public_space_safety": "the safety of schools and other public spaces",
    "race_ethnicity": "race and ethnicity in gun violence",
    "public_opinion": "public opinion, polls, protests and movements",
    "society_culture": "society and culture, including gun culture and entertainment",
    "economic_consequences": "economic consequences of gun violence or of gun policy",
}


def prompt(dataset: str) -> str:
    """The study's prompt template of a dataset, read from the package."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown study dataset {dataset!r}; the datasets are {DATASETS}")
    return (PROMPTS_DIR / f"{dataset}.md").read_text(encoding="utf-8")


def task(dataset: str, prompt_text: str | None = None) -> Task:
    """The study task of a dataset: its kind, key, codebook, span fields and prompt."""
    text = prompt(dataset) if prompt_text is None else prompt_text
    if dataset == "miric":
        return Task("label", {"is_ai_patent": "the patent is related to AI technologies"}, text)
    if dataset == "gvfc":
        return Task("labels", GVFC_CODEBOOK, text, key="frames", per_answer=(1, 2))
    if dataset == "politicause":
        return Task("tuple", prompt=text, key="causal")
    return Task("tuples", prompt=text, key="relations")


TASKS: dict[str, Task] = {dataset: task(dataset) for dataset in DATASETS}


def reference_from_gold(dataset: str, gold: Any) -> Reference:
    """A reference from the study's ``gold_label`` shape (what its dataset loaders give)."""
    if dataset == "miric":
        return bool(gold)
    if dataset == "gvfc":
        return frozenset(str(frame) for frame in gold)
    if dataset == "politicause":
        if not gold["causal"]:
            return ()
        references = gold["references"]
        return tuple(Relation(cause=r["cause"], effect=r["effect"], signal=r.get("signal", "")) for r in references)
    return tuple(Relation(cause=r["cause"], effect=r["effect"], signal=r.get("signal") or "") for r in gold)


def items_from_gold(dataset: str, rows: Sequence[dict[str, Any]]) -> list[Item]:
    """Items from the study loaders' rows (``id``, ``text``, ``gold_label``)."""
    return [Item(row["id"], str(row["text"]), reference_from_gold(dataset, row["gold_label"])) for row in rows]


__all__ = [
    "BOOTSTRAP_SEED",
    "DATASETS",
    "GVFC_CODEBOOK",
    "PROMPTS_DIR",
    "TASKS",
    "items_from_gold",
    "prompt",
    "reference_from_gold",
    "task",
]
