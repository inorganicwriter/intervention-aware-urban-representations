"""Compatibility facade for the modular representation trainer candidate.

Existing callers continue to use :mod:`urban_intervention.representation.trainer`.
This module is an opt-in replacement used for equivalence validation before any
production cutover.
"""

from .training import (
    Pool,
    _append_run_record,
    _collate_fn,
    _collect_pool,
    _evaluate_retrieval,
    _run_epoch,
    _visualize_embeddings,
    build_evaluation_report,
    train_representation,
)

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
