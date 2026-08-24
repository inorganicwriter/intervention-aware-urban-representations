"""Single-epoch optimization and validation."""

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..dataset import RepresentationDataset
from ..evaluation import retrieval_metrics
from ..loss import total_loss
from ..model import ResponseEmbeddingModel
from ..queue import EmbeddingQueue


def _evaluate_retrieval(
    embeddings: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    k: int = 5,
) -> dict[str, float]:
    metrics = retrieval_metrics(embeddings, responses, response_mask, k=k)
    overall = metrics["overall"]
    nn_val = overall.get("nn_corr@k", None)
    base_val = overall.get("baseline_corr", None)
    nn_corr = float(nn_val) if isinstance(nn_val, (int, float)) else 0.0
    baseline = float(base_val) if isinstance(base_val, (int, float)) else 0.0
    return {
        "mean_nn_corr@k": round(nn_corr, 4),
        "baseline_corr": round(baseline, 4),
    }


def _run_epoch(
    model: ResponseEmbeddingModel,
    loader: DataLoader,
    ds: RepresentationDataset,
    device: torch.device,
    temperature: float,
    rep_alpha: float,
    pred_weight: float,
    use_images: bool,
    is_training: bool,
    optimizer: AdamW | None = None,
    queue: EmbeddingQueue | None = None,
    se_shrinkage: bool = True,
    learnable_temperature: bool = False,
    eval_k: int = 5,
) -> dict[str, float]:
    if is_training:
        model.train()
    else:
        model.eval()

    total_sum = rep_sum = pred_sum = 0.0
    rep_weight_sum = pred_weight_sum = 0.0
    all_embeddings: list[torch.Tensor] = []
    all_responses: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_response_se: list[torch.Tensor] = []
    all_predictions: dict[str, list[torch.Tensor]] = {}
    n_batches = 0
    n_examples = 0
    losses: dict[str, torch.Tensor] = {}

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            if batch["features"].shape[0] < 2 and is_training:
                continue
            features = batch["features"].to(device)
            responses = batch["responses"].to(device)
            response_mask = batch["response_mask"].to(device)
            response_se = batch["response_se"].to(device)
            treatment_orders = batch["treatment_order"].to(device)
            images = batch.get("images")
            image_mask = batch.get("image_mask")
            conditioning_tokens = batch.get("conditioning_tokens")
            station_tokens = batch.get("station_tokens")
            if images is not None:
                images = images.to(device)
            if image_mask is not None:
                image_mask = image_mask.to(device)
            if conditioning_tokens is not None:
                conditioning_tokens = conditioning_tokens.to(device)

            embedding, predictions = model(
                features, images, image_mask,
                conditioning_tokens=conditioning_tokens,
                station_tokens=station_tokens,
            )
            temperature_t = model.temperature() if learnable_temperature else temperature
            queue_state = queue.labeled_state() if queue is not None else None
            losses = total_loss(
                embedding,
                predictions,
                responses,
                response_mask,
                response_se,
                temperature=temperature_t,
                rep_alpha=rep_alpha,
                pred_weight=pred_weight,
                se_shrinkage=se_shrinkage,
                queue_embeddings=queue_state["embeddings"] if queue_state else None,
                queue_responses=queue_state["responses"] if queue_state else None,
                queue_response_mask=queue_state["response_mask"] if queue_state else None,
                queue_response_se=queue_state["response_se"] if queue_state else None,
                anchor_ids=treatment_orders,
                queue_ids=queue_state["ids"] if queue_state else None,
                log_var_rep=model.log_var_rep if model.uncertainty_weighted else None,
                log_var_pred=model.log_var_pred if model.uncertainty_weighted else None,
            )
            loss = losses["total"]

            if is_training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if queue is not None:
                    queue.enqueue(
                        embedding,
                        responses,
                        response_mask,
                        response_se,
                        treatment_orders,
                    )

            batch_size = int(features.shape[0])
            total_sum += loss.item() * batch_size
            rep_sum += losses["representation"].item() * batch_size
            pred_sum += losses["prediction"].item() * batch_size
            if "rep_weight" in losses:
                rep_weight_sum += float(losses["rep_weight"].item()) * batch_size
                pred_weight_sum += float(losses["pred_weight"].item()) * batch_size
            n_batches += 1
            n_examples += batch_size

            if not is_training:
                all_embeddings.append(embedding.cpu())
                all_responses.append(responses.cpu())
                all_masks.append(response_mask.cpu())
                all_response_se.append(response_se.cpu())
                for family, prediction in predictions.items():
                    all_predictions.setdefault(family, []).append(prediction.cpu())

    result = {
        "total": total_sum / max(n_examples, 1),
        "representation": rep_sum / max(n_examples, 1),
        "prediction": pred_sum / max(n_examples, 1),
    }
    if is_training:
        result["steps"] = n_batches
        if "rep_weight" in losses:
            result["rep_weight"] = rep_weight_sum / max(n_examples, 1)
            result["pred_weight"] = pred_weight_sum / max(n_examples, 1)

    if not is_training and all_embeddings:
        val_emb = torch.cat(all_embeddings, dim=0)
        val_resp = torch.cat(all_responses, dim=0)
        val_mask = torch.cat(all_masks, dim=0)
        val_se = torch.cat(all_response_se, dim=0)
        val_predictions = {
            family: torch.cat(parts, dim=0) for family, parts in all_predictions.items()
        }
        full_temperature = model.temperature() if learnable_temperature else temperature
        if isinstance(full_temperature, torch.Tensor):
            full_temperature = full_temperature.detach().cpu()
        eval_log_var_rep = (
            model.log_var_rep.detach().cpu() if model.log_var_rep is not None else None
        )
        eval_log_var_pred = (
            model.log_var_pred.detach().cpu() if model.log_var_pred is not None else None
        )
        full_losses = total_loss(
            val_emb,
            val_predictions,
            val_resp,
            val_mask,
            val_se,
            temperature=full_temperature,
            rep_alpha=rep_alpha,
            pred_weight=pred_weight,
            se_shrinkage=se_shrinkage,
            log_var_rep=eval_log_var_rep if model.uncertainty_weighted else None,
            log_var_pred=eval_log_var_pred if model.uncertainty_weighted else None,
        )
        result.update(
            {
                "total": float(full_losses["total"].item()),
                "representation": float(full_losses["representation"].item()),
                "prediction": float(full_losses["prediction"].item()),
            }
        )
        result.update(_evaluate_retrieval(val_emb, val_resp, val_mask, k=eval_k))

    return result
