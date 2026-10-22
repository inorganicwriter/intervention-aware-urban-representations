"""Independent task validity and active-task normalization for shared STC."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .loss import response_alignment_loss

TASK_DIMS = {"viirs": 6, "poi": 12}
TRAINING_TASKS = {"joint": ("viirs", "poi"), "viirs_only": ("viirs",), "poi_only": ("poi",)}


def joint_causal_loss(
    embeddings: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    response_offsets: dict,
    response_weight: torch.Tensor | None = None,
    response_se: torch.Tensor | None = None,
    training_mode: str = "joint",
    anchor_weight: float = 0.85,
    align_weight: float = 0.15,
    delta: float = 1.0,
    temperature: float = 0.07,
) -> dict[str, object]:
    if training_mode not in TRAINING_TASKS:
        raise ValueError(f"Unknown joint training_mode: {training_mode}")
    if delta <= 0 or temperature <= 0 or anchor_weight < 0 or align_weight < 0:
        raise ValueError("Invalid joint loss hyperparameters")
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("Non-finite shared state embedding")
    zero = embeddings.sum() * 0.0
    anchors, alignments, tasks = [], [], {}
    for family, dim in TASK_DIMS.items():
        if family not in response_offsets:
            raise ValueError(f"Joint data must declare {family} response offsets")
        start, end, *_ = response_offsets[family]
        if end - start != dim or end > responses.shape[1]:
            raise ValueError(f"Invalid response layout for {family}")
        target = responses[:, start:end]
        weight = (
            torch.ones_like(target) if response_weight is None else response_weight[:, start:end]
        )
        mask = (
            response_mask[:, start:end].bool()
            & torch.isfinite(target)
            & torch.isfinite(weight)
            & (weight > 0)
        )
        enabled = family in TRAINING_TASKS[training_mode]
        if not enabled:
            mask = torch.zeros_like(mask)
        pred = predictions[family]
        if pred.shape != target.shape:
            raise ValueError(f"Prediction shape mismatch for {family}")
        active = bool(mask.any().item())
        anchor = zero
        if active:
            if not torch.isfinite(pred[mask]).all():
                raise FloatingPointError(f"Non-finite prediction at valid {family} labels")
            w = weight[mask]
            anchor = (
                w * functional.huber_loss(pred[mask], target[mask], reduction="none", delta=delta)
            ).sum() / w.sum()
            anchors.append(anchor)
        alignment = {"loss": zero, "active": False, "valid_anchors": 0, "valid_pairs": 0}
        if enabled and align_weight > 0:
            alignment = response_alignment_loss(
                embeddings,
                target,
                mask,
                temperature=temperature,
                response_weight=torch.where(mask, weight, 0.0),
                response_se=response_se[:, start:end] if response_se is not None else None,
                return_details=True,
            )
            if alignment["active"]:
                alignments.append(alignment["loss"])
        tasks[family] = {
            "anchor": anchor,
            "align": alignment["loss"],
            "anchor_active": active,
            "align_active": alignment["active"],
            "valid_cells": int(mask.sum().item()),
            "valid_pairs": alignment["valid_pairs"],
            "valid_anchors": alignment["valid_anchors"],
            "diagnostic_loss": anchor_weight * anchor + align_weight * alignment["loss"],
        }
    anchor = torch.stack(anchors).mean() if anchors else zero
    align = torch.stack(alignments).mean() if alignments else zero
    for task in tasks.values():
        task["anchor_coefficient"] = anchor_weight / len(anchors) if task["anchor_active"] else 0.0
        task["align_coefficient"] = align_weight / len(alignments) if task["align_active"] else 0.0
        task["diagnostic_loss"] = (
            task["anchor_coefficient"] * task["anchor"] + task["align_coefficient"] * task["align"]
        )
    total = anchor_weight * anchor + align_weight * align
    if not torch.isfinite(total):
        raise FloatingPointError("Joint causal loss is not finite")
    return {
        "total": total,
        "anchor": anchor,
        "align": align,
        "tasks": tasks,
        "active_anchor_tasks": len(anchors),
        "active_align_tasks": len(alignments),
        "update_active": bool((anchors and anchor_weight > 0) or (alignments and align_weight > 0)),
    }
