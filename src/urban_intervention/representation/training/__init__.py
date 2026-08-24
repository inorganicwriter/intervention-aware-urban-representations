"""Modular candidate implementation of the representation trainer.

The production entry point remains ``representation.trainer`` until this
candidate has been accepted.  This package keeps the same callable surface
while separating training responsibilities into independently testable units.
"""

from .batching import _collate_fn
from .epoch import _evaluate_retrieval, _run_epoch
from .evaluation import build_evaluation_report
from .pools import _collect_pool
from .runner import train_representation
from .tracking import _append_run_record
from .types import Pool
from .visualization import _visualize_embeddings

__all__ = [
    "Pool",
    "_append_run_record",
    "_collate_fn",
    "_collect_pool",
    "_evaluate_retrieval",
    "_run_epoch",
    "_visualize_embeddings",
    "build_evaluation_report",
    "train_representation",
]
