"""Loss functions for response-aligned representation learning."""

from __future__ import annotations

import torch

from .dataset import RESPONSE_OFFSETS
from .evaluation import (
    response_similarity,
    response_similarity_between,
    response_similarity_with_validity,
)

_pairwise_response_similarity = response_similarity


def _unit_reliability(response_se: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Return one reliability score per response row."""
    valid = response_mask & torch.isfinite(response_se) & (response_se > 0)
    se = torch.where(valid, response_se, torch.ones_like(response_se))
    reliability = torch.where(valid, 1.0 / (1.0 + se**2), torch.ones_like(se))
    counts = response_mask.float().sum(dim=1).clamp(min=1e-8)
    return (reliability * response_mask.float()).sum(dim=1) / counts


def _pair_reliability(
    response_se: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """Per-pair shrinkage factor derived from label standard errors.

    Each observed cell contributes reliability ``1 / (1 + SE²)``; cells whose
    SE is missing, NaN, non-positive or infinite are neutral (reliability 1).
    A unit's reliability is the mean over its observed cells; the pair factor
    is the geometric mean ``sqrt(r_i * r_j)``.  Multiplying response
    similarity by this factor attenuates pairs whose response labels are
    noisy, which mirrors the classic measurement-error attenuation applied
    per pair.  With a constant SE across all cells the factor is a uniform
    scalar: the alignment loss is then invariant (positive weights are
    row-normalized) and the distance loss simply rescales its targets.
    """
    unit = _unit_reliability(response_se, response_mask)
    return (unit.unsqueeze(1) * unit.unsqueeze(0)).sqrt()


def response_alignment_loss(
    embeddings: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float | torch.Tensor = 0.07,
    min_similarity: float = 0.0,
    response_se: torch.Tensor | None = None,
    queue_embeddings: torch.Tensor | None = None,
    queue_responses: torch.Tensor | None = None,
    queue_response_mask: torch.Tensor | None = None,
    queue_response_se: torch.Tensor | None = None,
    anchor_ids: torch.Tensor | None = None,
    queue_ids: torch.Tensor | None = None,
    response_weight: torch.Tensor | None = None,
    queue_response_weight: torch.Tensor | None = None,
    return_details: bool = False,
) -> torch.Tensor | dict[str, object]:
    """Soft-weighted InfoNCE where positive weights = response similarity.

    Similar pairs (high response correlation) are pulled together;
    dissimilar pairs are pulled apart via the softmax denominator.

    With ``response_se`` the response similarities are shrunk toward zero by
    label reliability (see ``_pair_reliability``), so noisy pairs exert less
    pull.  With ``response_weight`` (per-label design reliability) each pair's
    common cells carry weight ``sqrt(w_i * w_j)`` and pairs without common
    reliable support are dropped from the denominator instead of being treated
    as zero similarity.  A labeled queue is response-aware: comparable positive
    and negative pairs participate, incomparable and duplicate-unit pairs do
    not.
    """
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("Non-finite embedding in response alignment")
    response_mask = response_mask.bool() & torch.isfinite(responses)
    if response_weight is not None:
        response_mask = response_mask & torch.isfinite(response_weight) & (response_weight > 0)
        response_weight = torch.where(response_mask, response_weight, 0.0)
    responses = torch.where(response_mask, responses, 0.0)
    if queue_responses is not None and queue_response_mask is not None:
        queue_response_mask = queue_response_mask.bool() & torch.isfinite(queue_responses)
        if queue_response_weight is not None:
            queue_response_mask = (
                queue_response_mask & torch.isfinite(queue_response_weight)
                & (queue_response_weight > 0)
            )
            queue_response_weight = torch.where(queue_response_mask, queue_response_weight, 0.0)
        queue_responses = torch.where(queue_response_mask, queue_responses, 0.0)
    batch_size = embeddings.shape[0]
    emb_sim = embeddings @ embeddings.T / temperature
    resp_sim, valid_pairs = response_similarity_with_validity(
        responses, response_mask, weights=response_weight
    )
    if response_se is not None:
        resp_sim = resp_sim * _pair_reliability(response_se, response_mask)

    off_diagonal = ~torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    valid_pairs = valid_pairs & off_diagonal
    logits = emb_sim

    has_labeled_queue = (
        queue_embeddings is not None
        and queue_embeddings.shape[0] > 0
        and queue_responses is not None
        and queue_response_mask is not None
    )
    if queue_embeddings is not None and queue_embeddings.shape[0] > 0:
        queue = queue_embeddings.to(embeddings.device)
        queue_logits = embeddings @ queue.T / temperature
        logits = torch.cat([logits, queue_logits], dim=1)
        if has_labeled_queue:
            assert queue_responses is not None
            assert queue_response_mask is not None
            queue_resp = queue_responses.to(responses.device)
            queue_mask = queue_response_mask.to(response_mask.device)
            queue_weight = (
                queue_response_weight.to(responses.device)
                if queue_response_weight is not None
                else None
            )
            queue_sim, queue_valid = response_similarity_between(
                responses,
                response_mask,
                queue_resp,
                queue_mask,
                weights_a=response_weight,
                weights_b=queue_weight,
            )
            if response_se is not None and queue_response_se is not None:
                current_unit = _unit_reliability(response_se, response_mask)
                queue_unit = _unit_reliability(
                    queue_response_se.to(response_se.device), queue_mask
                )
                queue_sim = queue_sim * (
                    current_unit.unsqueeze(1) * queue_unit.unsqueeze(0)
                ).sqrt()
            if anchor_ids is not None and queue_ids is not None:
                different_unit = anchor_ids.to(embeddings.device).unsqueeze(1) != queue_ids.to(
                    embeddings.device
                ).unsqueeze(0)
                queue_valid = queue_valid & different_unit
            resp_sim = torch.cat([resp_sim, queue_sim], dim=1)
            valid_pairs = torch.cat([valid_pairs, queue_valid], dim=1)
        else:
            # Backward compatibility for callers that provide an unlabeled
            # queue: its entries can only act as negatives.
            resp_sim = torch.cat(
                [resp_sim, resp_sim.new_zeros(batch_size, queue.shape[0])], dim=1
            )
            valid_pairs = torch.cat(
                [
                    valid_pairs,
                    torch.ones(batch_size, queue.shape[0], dtype=torch.bool, device=embeddings.device),
                ],
                dim=1,
            )

    positive_weights = torch.where(
        valid_pairs & (resp_sim > min_similarity),
        resp_sim.clamp_min(0.0),
        torch.zeros_like(resp_sim),
    )
    positive_sum = positive_weights.sum(dim=1)
    valid_anchors = positive_sum > 0
    if not valid_anchors.any():
        loss = embeddings.sum() * 0.0
        if return_details:
            return {"loss": loss, "active": False, "valid_anchors": 0,
                    "valid_pairs": int(valid_pairs.sum().item())}
        return loss

    positive_weights = positive_weights[valid_anchors] / positive_sum[valid_anchors].unsqueeze(1)
    anchor_logits = logits[valid_anchors].masked_fill(~valid_pairs[valid_anchors], -torch.inf)
    log_softmax = anchor_logits - anchor_logits.logsumexp(dim=1, keepdim=True)
    log_softmax = log_softmax.masked_fill(~valid_pairs[valid_anchors], 0.0)
    loss = -(positive_weights * log_softmax).sum(dim=1).mean()
    if return_details:
        return {"loss": loss, "active": True,
                "valid_anchors": int(valid_anchors.sum().item()),
                "valid_pairs": int(valid_pairs.sum().item())}
    return loss


def embedding_distance_loss(
    embeddings: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    response_se: torch.Tensor | None = None,
    response_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE between embedding cosine distance and response similarity.

    L = mean_{i != j} (cos_dist(emb_i, emb_j) - (1 - sim_resp(i,j)))²
    which equals mean_{i != j} (1 - cos(emb_i, emb_j) - (1 - sim_resp(i,j)))²
           = mean_{i != j} (cos(emb_i, emb_j) - sim_resp(i,j))²

    With ``response_se`` the target similarities are shrunk toward zero by
    label reliability, so uncertain pairs are not forced to be close or far.
    ``response_weight`` uses the same pairwise design weighting as the
    alignment loss.
    """
    emb_cos = embeddings @ embeddings.T
    resp_sim, valid_pairs = response_similarity_with_validity(
        responses, response_mask, weights=response_weight
    )
    if response_se is not None:
        resp_sim = resp_sim * _pair_reliability(response_se, response_mask)

    diag_mask = ~torch.eye(emb_cos.shape[0], device=emb_cos.device, dtype=torch.bool)
    valid_pairs = valid_pairs & diag_mask
    if not valid_pairs.any():
        return embeddings.sum() * 0.0

    diff = emb_cos[valid_pairs] - resp_sim[valid_pairs]
    return (diff**2).mean()


def combined_representation_loss(
    embeddings: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float | torch.Tensor = 0.07,
    alpha: float = 0.5,
    response_se: torch.Tensor | None = None,
    queue_embeddings: torch.Tensor | None = None,
    queue_responses: torch.Tensor | None = None,
    queue_response_mask: torch.Tensor | None = None,
    queue_response_se: torch.Tensor | None = None,
    anchor_ids: torch.Tensor | None = None,
    queue_ids: torch.Tensor | None = None,
    response_weight: torch.Tensor | None = None,
    queue_response_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    l_alignment = response_alignment_loss(
        embeddings,
        responses,
        response_mask,
        temperature,
        response_se=response_se,
        queue_embeddings=queue_embeddings,
        queue_responses=queue_responses,
        queue_response_mask=queue_response_mask,
        queue_response_se=queue_response_se,
        anchor_ids=anchor_ids,
        queue_ids=queue_ids,
        response_weight=response_weight,
        queue_response_weight=queue_response_weight,
    )
    l_distance = embedding_distance_loss(
        embeddings, responses, response_mask, response_se, response_weight
    )
    return alpha * l_alignment + (1 - alpha) * l_distance


def _precision_weights(se_values: torch.Tensor) -> torch.Tensor:
    """Return 1/SE² precision weights with missing SE imputed at the median.

    Matched labels carry no standard error by design.  A literal weight of 1.0
    would give them ~1/SE² (hundreds of times) less influence than GSC/MC
    cells; using the family's median observed precision keeps the weighting
    scale-consistent across methods.
    """
    valid = torch.isfinite(se_values) & (se_values > 0)
    if valid.any():
        reference = se_values[valid].median()
        se_values = torch.where(valid, se_values, reference)
    else:
        se_values = torch.ones_like(se_values)
    return 1.0 / (se_values**2 + 1e-8)


def prediction_loss(
    predictions: dict[str, torch.Tensor],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    response_se: torch.Tensor | None,
    response_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-family mean squared error, optionally weighted per response cell.

    Without ``response_weight`` the historical precision weighting is kept
    (``1 / SE²`` with the family median imputation).  With ``response_weight``
    the design-only reliability is used instead: each family averages
    ``sum(m * w * diff²) / sum(m * w)`` and families whose observable cells all
    carry zero weight are skipped.  The statistical uncertainty of the labels
    stays in the representation loss through ``response_se``; the prediction
    head is intentionally seeded with the design signal only.
    """
    total: torch.Tensor = torch.zeros((), device=responses.device)
    count = 0
    for family, (start, end, _) in RESPONSE_OFFSETS.items():
        if family not in predictions:
            continue
        pred = predictions[family]
        if responses.shape[1] == pred.shape[1] and len(predictions) == 1:
            target = responses
            mask = response_mask
            se = response_se if response_se is not None else None
            weight_slice = response_weight if response_weight is not None else None
        else:
            target = responses[:, start:end]
            mask = response_mask[:, start:end]
            se = response_se[:, start:end] if response_se is not None else None
            weight_slice = response_weight[:, start:end] if response_weight is not None else None

        if mask.sum() == 0:
            continue

        diff = pred[mask] - target[mask]
        if response_weight is not None and weight_slice is not None:
            design = torch.nan_to_num(
                weight_slice, nan=0.0
            ).clamp_min(0.0)
            weight = design[mask]
            weight_sum = weight.sum()
            if weight_sum <= 0:
                continue
            total += (weight * diff**2).sum() / weight_sum
        else:
            weight = torch.ones_like(diff) if se is None else _precision_weights(se[mask])
            total += (weight * diff**2).sum() / weight.sum()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=responses.device, requires_grad=True)
    return total / count


def total_loss(
    embeddings: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    response_se: torch.Tensor | None,
    temperature: float | torch.Tensor = 0.07,
    rep_alpha: float = 0.5,
    pred_weight: float = 0.5,
    se_shrinkage: bool = True,
    queue_embeddings: torch.Tensor | None = None,
    queue_responses: torch.Tensor | None = None,
    queue_response_mask: torch.Tensor | None = None,
    queue_response_se: torch.Tensor | None = None,
    anchor_ids: torch.Tensor | None = None,
    queue_ids: torch.Tensor | None = None,
    log_var_rep: torch.Tensor | None = None,
    log_var_pred: torch.Tensor | None = None,
    response_weight: torch.Tensor | None = None,
    queue_response_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    l_rep = combined_representation_loss(
        embeddings,
        responses,
        response_mask,
        temperature,
        rep_alpha,
        response_se=response_se if se_shrinkage else None,
        queue_embeddings=queue_embeddings,
        queue_responses=queue_responses,
        queue_response_mask=queue_response_mask,
        queue_response_se=queue_response_se,
        anchor_ids=anchor_ids,
        queue_ids=queue_ids,
        response_weight=response_weight,
        queue_response_weight=queue_response_weight,
    )
    l_pred = prediction_loss(predictions, responses, response_mask, response_se, response_weight)
    if log_var_rep is not None and log_var_pred is not None:
        # Uncertainty weighting (Kendall et al. 2018): the task weights are
        # learned as exp(-log_var) and the log variances regularize against
        # collapsing to zero.  With both log vars initialised at log(2)
        # the starting weights are exactly the default 0.5/0.5 balance.
        weight_rep = torch.exp(-log_var_rep)
        weight_pred = torch.exp(-log_var_pred)
        l_total = weight_rep * l_rep + weight_pred * l_pred + log_var_rep + log_var_pred
        return {
            "total": l_total,
            "representation": l_rep,
            "prediction": l_pred,
            "rep_weight": weight_rep.detach(),
            "pred_weight": weight_pred.detach(),
        }
    l_total = (1 - pred_weight) * l_rep + pred_weight * l_pred
    return {"total": l_total, "representation": l_rep, "prediction": l_pred}


def soft_causal_anchoring_loss(
    predictions: dict[str, torch.Tensor],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    response_weight: torch.Tensor | None = None,
    delta: float = 1.0,
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
) -> torch.Tensor:
    """Design-weighted Huber loss on causal response trajectory predictions.

    Primary driver (~85%) in multimodal intervention-conditioned representation learning.
    Huber loss provides outlier resistance against extreme spatial interventions,
    and weighting by W_design prioritizes high-confidence econometric counterfactuals.
    """
    total: torch.Tensor = torch.zeros((), device=responses.device)
    count = 0
    if response_offsets is not None:
        offsets_map = response_offsets
    elif responses.shape[1] == 27:
        offsets_map = RESPONSE_OFFSETS
    elif len(predictions) == 1:
        fam = next(iter(predictions.keys()))
        offsets_map = {fam: (0, responses.shape[1], [])}
    else:
        offsets_map = {}
        curr = 0
        for fam, pred in predictions.items():
            dim = pred.shape[1]
            offsets_map[fam] = (curr, curr + dim, [])
            curr += dim

    for family, (start, end, *_) in offsets_map.items():
        if family in predictions:
            pred = predictions[family]
        elif len(predictions) == 1:
            only = next(iter(predictions.values()))
            pred = (
                only[:, start:end]
                if only.shape[1] == responses.shape[1]
                else None
            )
        else:
            pred = None
        if pred is None:
            continue
        target = responses[:, start:end]
        mask = response_mask[:, start:end]
        weight_slice = response_weight[:, start:end] if response_weight is not None else None

        if mask.sum() == 0:
            continue

        diff = pred[mask] - target[mask]
        abs_diff = torch.abs(diff)
        huber_element = torch.where(
            abs_diff <= delta,
            0.5 * (diff**2),
            delta * (abs_diff - 0.5 * delta),
        )

        if response_weight is not None and weight_slice is not None:
            design = torch.nan_to_num(
                weight_slice, nan=0.0
            ).clamp_min(0.0)
            weight = design[mask]
            weight_sum = weight.sum()
            if weight_sum <= 0:
                continue
            total += (weight * huber_element).sum() / weight_sum
        else:
            total += huber_element.mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=responses.device, requires_grad=True)
    return total / count


def multimodal_causal_loss(
    embeddings: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    response_weight: torch.Tensor | None = None,
    response_se: torch.Tensor | None = None,
    anchor_weight: float = 0.85,
    align_weight: float = 0.15,
    temperature: float | torch.Tensor = 0.07,
    delta: float = 1.0,
    log_var_rep: torch.Tensor | None = None,
    log_var_pred: torch.Tensor | None = None,
    response_offsets: dict[str, tuple[int, int, list[str]]] | None = None,
    queue_embeddings: torch.Tensor | None = None,
    queue_responses: torch.Tensor | None = None,
    queue_response_mask: torch.Tensor | None = None,
    queue_response_se: torch.Tensor | None = None,
    anchor_ids: torch.Tensor | None = None,
    queue_ids: torch.Tensor | None = None,
    queue_response_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Unified loss function for multimodal intervention-conditioned representations.

    L_total = 0.85 * L_anchor (Huber) + 0.15 * L_align (Soft-InfoNCE)
    Optionally supports Kendall et al. (2018) uncertainty weighting.

    ``response_se`` is used only by the representation alignment term.  The
    prediction anchor deliberately keeps the design-only ``response_weight`` so
    sampling precision does not determine which causal trajectories the
    supervised head learns.
    """
    l_anchor = soft_causal_anchoring_loss(
        predictions=predictions,
        responses=responses,
        response_mask=response_mask,
        response_weight=response_weight,
        delta=delta,
        response_offsets=response_offsets,
    )
    l_align = response_alignment_loss(
        embeddings=embeddings,
        responses=responses,
        response_mask=response_mask,
        temperature=temperature,
        response_se=response_se,
        response_weight=response_weight,
        queue_embeddings=queue_embeddings,
        queue_responses=queue_responses,
        queue_response_mask=queue_response_mask,
        queue_response_se=queue_response_se,
        anchor_ids=anchor_ids,
        queue_ids=queue_ids,
        queue_response_weight=queue_response_weight,
    )

    if log_var_rep is not None and log_var_pred is not None:
        prec_align = 0.5 * torch.exp(-log_var_rep)
        prec_anchor = 0.5 * torch.exp(-log_var_pred)
        l_total = prec_align * l_align + prec_anchor * l_anchor + 0.5 * (log_var_rep + log_var_pred)
        return {
            "total": l_total,
            "anchor_loss": l_anchor,
            "align_loss": l_align,
            "anchor_weight": prec_anchor.detach(),
            "align_weight": prec_align.detach(),
            "rep_weight": prec_align.detach(),
            "pred_weight": prec_anchor.detach(),
        }

    l_total = anchor_weight * l_anchor + align_weight * l_align
    return {
        "total": l_total,
        "anchor_loss": l_anchor,
        "align_loss": l_align,
        "anchor_weight": torch.tensor(anchor_weight, device=embeddings.device),
        "align_weight": torch.tensor(align_weight, device=embeddings.device),
    }
