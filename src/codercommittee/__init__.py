"""codercommittee: a committee of coders.

Take several coders' answers to the same text, combine them under four aggregation structures
(one answer; one coder asked three times; the majority vote with alignment; the pool at k of
nine), align the units the coders constructed themselves, score every structure against human
references, and report each structure's gap to the committee's best member, with its interval.
"""

from .align import Alignment, Prematch, align, prematch
from .arms import POOL_KS, arm_names, arms
from .coders import MockCoder, ModelCoder
from .collect import collect
from .combine import combine
from .committee import Committee
from .compare import Agreement, ArmRow, Comparison, Gap, Pick, Scores, agreement, compare, pick_on_sample
from .data import Answer, Item, ItemScore, Match, Member, Prediction, Relation, Run, Task, Unit
from .dedupe import dedupe, medoid
from .embedders import FunctionEmbedder, MockEmbedder, OpenAICompatibleEmbedder, OpenAIEmbedder
from .io import read_items, write_comparison, write_items, write_predictions, write_scores
from .protocols import Coder, Completion, Embedder, Model, Referee
from .referee import BandRule, LLMReferee, RefereeError, ReplayReferee
from .replay import ReplayCoder, ReplayEmbedder, ReplayError, StoredRun, load_run
from .scoring import macro_f1, score
from .similarity import Similarity, Space
from .store import LoggedReferee, RunStore, StoreError, StoreReferee
from .validation import Validation, validate

__version__ = "0.2.4"

__all__ = [
    "POOL_KS",
    "Agreement",
    "Alignment",
    "Answer",
    "ArmRow",
    "BandRule",
    "Coder",
    "Committee",
    "Comparison",
    "Completion",
    "Embedder",
    "FunctionEmbedder",
    "Gap",
    "Item",
    "ItemScore",
    "LLMReferee",
    "LoggedReferee",
    "Match",
    "Member",
    "MockCoder",
    "MockEmbedder",
    "Model",
    "ModelCoder",
    "OpenAICompatibleEmbedder",
    "OpenAIEmbedder",
    "Pick",
    "Prediction",
    "Prematch",
    "Referee",
    "RefereeError",
    "Relation",
    "ReplayCoder",
    "ReplayEmbedder",
    "ReplayError",
    "ReplayReferee",
    "Run",
    "RunStore",
    "Scores",
    "Similarity",
    "Space",
    "StoreError",
    "StoreReferee",
    "StoredRun",
    "Task",
    "Unit",
    "Validation",
    "__version__",
    "agreement",
    "align",
    "arm_names",
    "arms",
    "collect",
    "combine",
    "compare",
    "dedupe",
    "load_run",
    "macro_f1",
    "medoid",
    "pick_on_sample",
    "prematch",
    "read_items",
    "score",
    "validate",
    "write_comparison",
    "write_items",
    "write_predictions",
    "write_scores",
]
