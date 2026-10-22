"""Shared types for the modular representation trainer."""

from __future__ import annotations

import typing

import torch


class Pool(typing.TypedDict, total=False):
    embeddings: torch.Tensor
    features: torch.Tensor
    responses: torch.Tensor
    masks: torch.Tensor
    city_keys: list[str]
    quality_grades: list[str]
    predictions: dict[str, torch.Tensor]
