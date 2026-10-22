"""Training loop for response-aligned urban representation learning."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import typing
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .baselines import (
    appearance_autoencoder_baseline,
)
from .baselines import (
    run_baselines as compute_baselines,
)
from .dataset import RepresentationDataset, collate_samples
from .evaluation import (
    bootstrap_ci,
    cosine_similarity,
    permutation_test,
    planning_semantic_classification_metrics,
    probe_rmse,
    response_similarity_with_validity,
    retrieval_metrics,
    trajectory_prediction_metrics,
)
from .loss import multimodal_causal_loss, total_loss
from .model import ResponseEmbeddingModel, STCResponseModel
from .queue import EmbeddingQueue
from .transfer import predictive_transfer_report, transfer_report


class Pool(typing.TypedDict, total=False):
    embeddings: torch.Tensor
    features: torch.Tensor
    responses: torch.Tensor
    masks: torch.Tensor
    city_keys: list[str]
    quality_grades: list[str]
    predictions: dict[str, torch.Tensor]


def _collate_fn(batch, ds: RepresentationDataset, use_images: bool = False):
    return collate_samples(
        batch,
        load_images_fn=ds._get_images if use_images else None,
        max_images_per_grid=ds.max_images_per_grid,
        use_images=use_images,
    )


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
    loss_type: str = "multimodal_causal",
    loss_anchor_weight: float = 0.85,
    loss_align_weight: float = 0.15,
    loss_huber_delta: float = 1.0,
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
    all_response_weights: list[torch.Tensor] = []
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
            response_weight = batch.get("response_weight")
            if response_weight is not None:
                response_weight = response_weight.to(device)
            treatment_orders = batch["treatment_order"].to(device)
            images = batch.get("images")
            image_mask = batch.get("image_mask")
            conditioning_tokens = batch.get("conditioning_tokens")
            station_tokens = batch.get("station_tokens")
            satellite_vectors = batch.get("satellite_vectors")
            satellite_mask = batch.get("satellite_mask")
            if images is not None:
                images = images.to(device)
            if image_mask is not None:
                image_mask = image_mask.to(device)
            if conditioning_tokens is not None:
                conditioning_tokens = conditioning_tokens.to(device)
            if satellite_vectors is not None:
                satellite_vectors = satellite_vectors.to(device)
            if satellite_mask is not None:
                satellite_mask = satellite_mask.to(device)

            if getattr(model, "is_stc_model", False):
                device_batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                embedding, predictions, _ = model.forward_batch(device_batch)
            else:
                embedding, predictions = model(
                    features,
                    images,
                    image_mask,
                    conditioning_tokens=conditioning_tokens,
                    station_tokens=station_tokens,
                    satellite_vectors=satellite_vectors,
                    satellite_mask=satellite_mask,
                )
            temperature_t = model.temperature() if learnable_temperature else temperature
            queue_state = queue.labeled_state() if queue is not None else None
            if loss_type == "multimodal_causal":
                losses = multimodal_causal_loss(
                    embeddings=embedding,
                    predictions=predictions,
                    responses=responses,
                    response_mask=response_mask,
                    response_weight=response_weight,
                    response_se=response_se if se_shrinkage else None,
                    anchor_weight=loss_anchor_weight,
                    align_weight=loss_align_weight,
                    temperature=temperature_t,
                    delta=loss_huber_delta,
                    log_var_rep=model.log_var_rep if model.uncertainty_weighted else None,
                    log_var_pred=model.log_var_pred if model.uncertainty_weighted else None,
                    queue_embeddings=queue_state["embeddings"] if queue_state else None,
                    queue_responses=queue_state["responses"] if queue_state else None,
                    queue_response_mask=queue_state["response_mask"] if queue_state else None,
                    queue_response_se=queue_state["response_se"] if queue_state else None,
                    anchor_ids=treatment_orders,
                    queue_ids=queue_state["ids"] if queue_state else None,
                    queue_response_weight=(
                        queue_state["response_weight"] if queue_state else None
                    ),
                )
                loss = losses["total"]
                losses["representation"] = losses["align_loss"]
                losses["prediction"] = losses["anchor_loss"]
            else:
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
                    response_weight=response_weight,
                    queue_response_weight=(
                        queue_state.get("response_weight") if queue_state else None
                    ),
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
                        response_weight=response_weight,
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
                if response_weight is not None:
                    all_response_weights.append(response_weight.cpu())
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
        val_weight = torch.cat(all_response_weights, dim=0) if all_response_weights else None
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
        if loss_type == "multimodal_causal":
            full_losses = multimodal_causal_loss(
                embeddings=val_emb,
                predictions=val_predictions,
                responses=val_resp,
                response_mask=val_mask,
                response_weight=val_weight,
                response_se=val_se if se_shrinkage else None,
                anchor_weight=loss_anchor_weight,
                align_weight=loss_align_weight,
                temperature=full_temperature,
                delta=loss_huber_delta,
            )
            result.update(
                {
                    "total": float(full_losses["total"].item()),
                    "representation": float(full_losses["align_loss"].item()),
                    "prediction": float(full_losses["anchor_loss"].item()),
                }
            )
        else:
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
                response_weight=val_weight,
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


def _collect_pool(
    model: ResponseEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
) -> Pool | None:
    """Run the model in eval mode and return full-pool tensors (CPU)."""
    model.eval()
    embeddings: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    responses: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    city_keys: list[str] = []
    quality_grades: list[str] = []
    predictions: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in loader:
            feat = batch["features"].to(device)
            resp = batch["responses"].to(device)
            mask = batch["response_mask"].to(device)
            images = batch.get("images")
            image_mask = batch.get("image_mask")
            conditioning_tokens = batch.get("conditioning_tokens")
            station_tokens = batch.get("station_tokens")
            satellite_vectors = batch.get("satellite_vectors")
            satellite_mask = batch.get("satellite_mask")
            if images is not None:
                images = images.to(device)
            if image_mask is not None:
                image_mask = image_mask.to(device)
            if conditioning_tokens is not None:
                conditioning_tokens = conditioning_tokens.to(device)
            if satellite_vectors is not None:
                satellite_vectors = satellite_vectors.to(device)
            if satellite_mask is not None:
                satellite_mask = satellite_mask.to(device)
            if getattr(model, "is_stc_model", False):
                device_batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                embedding, preds, _ = model.forward_batch(device_batch)
                raw_features = torch.cat(
                    [
                        batch["state_tabular"],
                        batch["context"],
                        batch["intervention"],
                        batch["alphaearth"],
                    ],
                    dim=1,
                )
            else:
                raw_features = feat
                embedding, preds = model(
                    feat,
                    images,
                    image_mask,
                    conditioning_tokens=conditioning_tokens,
                    station_tokens=station_tokens,
                    satellite_vectors=satellite_vectors,
                    satellite_mask=satellite_mask,
                )
            embeddings.append(embedding.cpu())
            features.append(raw_features.cpu())
            responses.append(resp.cpu())
            masks.append(mask.cpu())
            city_keys.extend(str(value) for value in batch.get("city_keys", []))
            quality_grades.extend(str(value) for value in batch.get("quality_grades", []))
            if isinstance(preds, dict):
                for k, v in preds.items():
                    predictions.setdefault(k, []).append(v.cpu())
    if not embeddings:
        return None
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "features": torch.cat(features, dim=0),
        "responses": torch.cat(responses, dim=0),
        "masks": torch.cat(masks, dim=0),
        "city_keys": city_keys,
        "quality_grades": quality_grades,
        "predictions": {k: torch.cat(v, dim=0) for k, v in predictions.items()} if predictions else {},
    }


def _visualize_embeddings(
    embeddings: torch.Tensor,
    city_keys: list[str],
    quality_grades: list[str] | None,
    out_path: Path,
) -> dict[str, object]:
    """PCA-2D scatter of a pool coloured by city (best-effort).

    Uses an SVD projection so the coordinate export needs no extra dependency;
    the PNG plot additionally needs matplotlib and is skipped when it is
    missing.
    """
    if embeddings.shape[0] < 3:
        return {"note": "pool smaller than 3 units", "path": None}
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    coords = (centered @ vt[:2].T).numpy()
    points = [
        {"x": float(x), "y": float(y), "city": city}
        for x, y, city in zip(coords[:, 0], coords[:, 1], city_keys, strict=False)
    ]
    if quality_grades:
        for point, grade in zip(points, quality_grades, strict=False):
            point["quality_grade"] = grade
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"points": points, "plot_written": False, "note": "matplotlib_not_installed"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cities = sorted(set(city_keys))
    palette = [plt.get_cmap("tab20")(index) for index in range(20)]
    figure, axis = plt.subplots(figsize=(7, 6))
    for index, city in enumerate(cities):
        keep = [i for i, value in enumerate(city_keys) if value == city]
        axis.scatter(
            coords[keep, 0],
            coords[keep, 1],
            s=24,
            label=city,
            color=palette[index % len(palette)],
            alpha=0.8,
        )
    axis.set_title("PCA-2D projection of learned embeddings (by city)")
    axis.legend(fontsize=7, loc="best", frameon=False)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return {"points": points, "plot_written": True, "path": out_path.as_posix()}


def build_evaluation_report(
    model: ResponseEmbeddingModel,
    loaders: dict[str, DataLoader],
    device: torch.device,
    k: int = 5,
    n_perm: int = 100,
    n_boot: int = 200,
    ridge: float = 1.0,
    seed: int = 42,
    max_perm_units: int = 512,
    probe_min_obs: int = 16,
    run_baselines: bool = True,
    run_transfer: bool = True,
    transfer_shot_sizes: tuple[int, ...] = (4, 8, 16, 32),
    baseline_ae_epochs: int = 5,
    transfer_n_seeds: int = 3,
    visualize: bool = False,
    visualization_dir: Path | None = None,
) -> dict[str, object]:
    """Full statistical evaluation of the best checkpoint on val/test pools.

    Writes retrieval metrics (overall + per family), bootstrap CIs, a
    permutation p-value against shuffled response labels, the raw-feature
    retrieval baseline, and a linear-probe transfer comparison (representation
    vs raw features) fitted on the train pool.

    With ``run_baselines`` the report also contains chance/appearance-only
    baselines (random projection, PCA, frozen DINOv2) on the held-out pool; with
    ``run_transfer`` it contains per-city retrieval and few-shot probe curves
    for every held-out split.
    """
    pools = {name: _collect_pool(model, loader, device) for name, loader in loaders.items()}
    train_pool = pools.get("train")
    response_offsets = None
    test_loader = loaders.get("test")
    if test_loader is not None and hasattr(test_loader.dataset, "response_offsets"):
        response_offsets = test_loader.dataset.response_offsets()
    report: dict[str, object] = {
        "config": {
            "k": k,
            "n_perm": n_perm,
            "n_boot": n_boot,
            "ridge": ridge,
            "seed": seed,
            "max_perm_units": max_perm_units,
            "probe_min_obs": probe_min_obs,
            "run_baselines": run_baselines,
            "run_transfer": run_transfer,
            "transfer_shot_sizes": list(transfer_shot_sizes),
            "transfer_n_seeds": transfer_n_seeds,
        },
    }
    for name, pool in pools.items():
        if name == "train":
            continue
        entry: dict[str, object] = {"n_units": 0}
        if pool is not None and pool["embeddings"].shape[0] >= 2:
            emb_cos = cosine_similarity(pool["embeddings"])
            sim_resp, valid_pairs = response_similarity_with_validity(
                pool["responses"],
                pool["masks"],
                response_offsets=response_offsets,
            )
            entry["n_units"] = int(pool["embeddings"].shape[0])
            entry["retrieval"] = retrieval_metrics(
                pool["embeddings"],
                pool["responses"],
                pool["masks"],
                k=k,
                response_offsets=response_offsets,
            )
            entry["bootstrap_ci"] = bootstrap_ci(
                emb_cos,
                sim_resp,
                k=k,
                n_boot=n_boot,
                seed=seed,
                valid_pairs=valid_pairs,
            )
            entry["permutation"] = permutation_test(
                emb_cos,
                pool["responses"],
                pool["masks"],
                k=k,
                n_perm=n_perm,
                seed=seed,
                max_units=max_perm_units,
                response_offsets=response_offsets,
            )
            entry["raw_feature_baseline"] = retrieval_metrics(
                pool["features"],
                pool["responses"],
                pool["masks"],
                k=k,
                response_offsets=response_offsets,
            )
            if pool.get("predictions"):
                entry["trajectory_prediction"] = trajectory_prediction_metrics(
                    predictions=pool["predictions"],
                    responses=pool["responses"],
                    response_mask=pool["masks"],
                    response_offsets=response_offsets,
                )
                entry["semantic_classification"] = planning_semantic_classification_metrics(
                    predictions=pool["predictions"],
                    responses=pool["responses"],
                    response_mask=pool["masks"],
                    response_offsets=response_offsets,
                )
            if train_pool is not None and train_pool["embeddings"].shape[0] >= 2:
                entry["probe"] = {
                    "embeddings": probe_rmse(
                        train_pool["embeddings"],
                        train_pool["responses"],
                        train_pool["masks"],
                        pool["embeddings"],
                        pool["responses"],
                        pool["masks"],
                        ridge=ridge,
                        min_obs=probe_min_obs,
                        response_offsets=response_offsets,
                    ),
                    "raw_features": probe_rmse(
                        train_pool["features"],
                        train_pool["responses"],
                        train_pool["masks"],
                        pool["features"],
                        pool["responses"],
                        pool["masks"],
                        ridge=ridge,
                        min_obs=probe_min_obs,
                        response_offsets=response_offsets,
                    ),
                }
            if run_transfer:
                entry["transfer"] = transfer_report(
                    pool["embeddings"],
                    pool["features"],
                    pool["responses"],
                    pool["masks"],
                    pool["city_keys"],
                    k=k,
                    shot_sizes=transfer_shot_sizes,
                    ridge=ridge,
                    probe_min_obs=probe_min_obs,
                    n_seeds=transfer_n_seeds,
                    response_offsets=response_offsets,
                )
            if run_transfer and train_pool is not None and train_pool["embeddings"].shape[0] >= 2:
                entry["predictive_transfer"] = predictive_transfer_report(
                    pool["embeddings"],
                    pool["features"],
                    train_pool["embeddings"],
                    train_pool["features"],
                    train_pool["responses"],
                    train_pool["masks"],
                    pool["responses"],
                    pool["masks"],
                    ridge=ridge,
                    min_obs=probe_min_obs,
                    response_offsets=response_offsets,
                )
            if visualize:
                plot_dir = visualization_dir or Path.cwd() / "outputs" / "representation"
                entry["embedding_pca"] = _visualize_embeddings(
                    pool["embeddings"],
                    pool["city_keys"],
                    pool.get("quality_grades"),
                    plot_dir / f"{name}_embedding_pca.png",
                )
        else:
            entry["note"] = "pool empty or smaller than 2 units"
        report[name] = entry
    if run_baselines:
        test_pool = pools.get("test")
        train_pool_for_baselines = pools.get("train")
        baselines: dict[str, object] = {}
        if test_pool is not None and test_pool["embeddings"].shape[0] >= 2:
            baselines["test"] = compute_baselines(
                test_pool["features"],
                test_pool["responses"],
                test_pool["masks"],
                image_batches=loaders.get("test"),
                device=device,
                k=k,
                n_perm=n_perm,
                n_boot=n_boot,
                seed=seed,
                fit_features=(
                    train_pool_for_baselines["features"]
                    if train_pool_for_baselines is not None
                    and train_pool_for_baselines["features"].shape[0] >= 2
                    else None
                ),
                response_offsets=response_offsets,
            )
            if (
                train_pool_for_baselines is not None
                and train_pool_for_baselines["features"].shape[0] >= 8
            ):
                baseline_test = baselines["test"]
                assert isinstance(baseline_test, dict)
                baseline_test["appearance_autoencoder"] = appearance_autoencoder_baseline(
                    train_pool_for_baselines["features"],
                    test_pool["features"],
                    test_pool["responses"],
                    test_pool["masks"],
                    epochs=baseline_ae_epochs,
                    hidden_dims=getattr(model, "hidden_dims", (128,)),
                    embedding_dim=getattr(model, "embedding_dim", getattr(model, "z_dim", 128)),
                    device=device,
                    k=k,
                    n_perm=n_perm,
                    n_boot=n_boot,
                    seed=seed,
                    response_offsets=response_offsets,
                )
        else:
            baselines["test"] = {"note": "pool empty or smaller than 2 units"}
        report["baselines"] = baselines
    return report


def train_representation(
    model_inputs_dir: Path,
    output_dir: Path,
    embedding_dim: int = 128,
    hidden_dims: tuple[int, ...] = (256, 128),
    dropout: float = 0.1,
    temperature: float = 0.07,
    rep_alpha: float = 0.5,
    pred_weight: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 64,
    epochs: int = 100,
    use_images: bool = False,
    max_images_per_grid: int = 4,
    image_pooling: str = "max",
    use_satellite: bool = False,
    satellite_embeddings_path: Path | None = None,
    use_cross_view_synergy: bool = False,
    synergy_heads: int = 4,
    conditioning: str | None = None,
    station_attributes_path: Path | None = None,
    se_shrinkage: bool = True,
    queue_size: int = 0,
    learnable_temperature: bool = False,
    uncertainty_weighted: bool = False,
    device: str | None = None,
    seed: int = 42,
    eval_k: int = 5,
    eval_n_perm: int = 100,
    eval_n_boot: int = 200,
    probe_ridge: float = 1.0,
    probe_min_obs: int = 16,
    run_baselines: bool = True,
    run_transfer: bool = True,
    transfer_shot_sizes: tuple[int, ...] = (4, 8, 16, 32),
    baseline_ae_epochs: int = 5,
    transfer_n_seeds: int = 3,
    visualize: bool = False,
    loss_type: str = "multimodal_causal",
    loss_anchor_weight: float = 0.85,
    loss_align_weight: float = 0.15,
    loss_huber_delta: float = 1.0,
    allow_satellite_proxy: bool = False,
    require_multimodal: bool = False,
    require_satellite: bool = False,
    require_streetview: bool = False,
    verify_bundle: bool = False,
    streetview_index_path: Path | str | None = None,
    satellite_root: Path | str | None = None,
    training_mode: str | None = None,
    max_grad_norm: float | None = 1.0,
    gradient_diagnostics_interval: int = 1,
    prediction_inputs: str = "full",
) -> Path:
    joint_options = locals().copy()
    from urban_intervention.representation.training.joint import (
        train_joint_representation,
        uses_joint_bundle,
    )

    if training_mode is not None or uses_joint_bundle(model_inputs_dir):
        return train_joint_representation(**joint_options)
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for pairwise representation learning")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= rep_alpha <= 1.0:
        raise ValueError("rep_alpha must be between 0 and 1")
    if not 0.0 <= pred_weight <= 1.0:
        raise ValueError("pred_weight must be between 0 and 1")
    if queue_size < 0:
        raise ValueError("queue_size must be non-negative")
    if loss_type not in {"multimodal_causal", "contrastive_legacy"}:
        raise ValueError(
            f"Unknown loss_type: {loss_type!r}, expected 'multimodal_causal' or 'contrastive_legacy'"
        )
    if loss_anchor_weight < 0:
        raise ValueError("loss_anchor_weight must be non-negative")
    if loss_align_weight < 0:
        raise ValueError("loss_align_weight must be non-negative")
    if loss_huber_delta <= 0:
        raise ValueError("loss_huber_delta must be positive")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)
    actual_use_images = bool(use_images or require_streetview or require_multimodal)
    actual_use_satellite = bool(use_satellite or require_satellite or require_multimodal)

    train_ds = RepresentationDataset(
        model_inputs_dir,
        split="train",
        max_images_per_grid=max_images_per_grid,
        load_images=actual_use_images,
        only_training_mask=True,
        station_attributes_path=station_attributes_path,
        satellite_embeddings_path=satellite_embeddings_path,
        allow_satellite_proxy=allow_satellite_proxy,
        require_multimodal=require_multimodal,
        require_satellite=require_satellite,
        require_streetview=require_streetview,
        verify_bundle=verify_bundle,
        streetview_index_path=streetview_index_path,
        satellite_root=satellite_root,
    )
    val_ds = RepresentationDataset(
        model_inputs_dir,
        split="validation",
        max_images_per_grid=max_images_per_grid,
        load_images=actual_use_images,
        only_training_mask=True,
        station_attributes_path=station_attributes_path,
        satellite_embeddings_path=satellite_embeddings_path,
        allow_satellite_proxy=allow_satellite_proxy,
        require_multimodal=require_multimodal,
        require_satellite=require_satellite,
        require_streetview=require_streetview,
        verify_bundle=verify_bundle,
        streetview_index_path=streetview_index_path,
        satellite_root=satellite_root,
    )
    test_ds = RepresentationDataset(
        model_inputs_dir,
        split="test",
        max_images_per_grid=max_images_per_grid,
        load_images=actual_use_images,
        only_training_mask=True,
        station_attributes_path=station_attributes_path,
        satellite_embeddings_path=satellite_embeddings_path,
        allow_satellite_proxy=allow_satellite_proxy,
        require_multimodal=require_multimodal,
        require_satellite=require_satellite,
        require_streetview=require_streetview,
        verify_bundle=verify_bundle,
        streetview_index_path=streetview_index_path,
        satellite_root=satellite_root,
    )
    split_sizes = {"train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)}
    too_small = {name: size for name, size in split_sizes.items() if size < 2}
    if too_small:
        details = ", ".join(f"{name}={size}" for name, size in too_small.items())
        raise ValueError(
            "Every training/evaluation split needs at least 2 eligible units; " + details
        )
    if getattr(train_ds, "is_stc", False):
        legacy_only = []
        if tuple(hidden_dims) != (256, 128):
            legacy_only.append("hidden_dims")
        if image_pooling != "max":
            legacy_only.append("image_pooling")
        if use_cross_view_synergy:
            legacy_only.append("use_cross_view_synergy/synergy_heads")
        if station_attributes_path is not None:
            legacy_only.append("station_attributes_path")
        if legacy_only:
            warnings.warn(
                "STC architecture ignores legacy-only options: "
                + ", ".join(legacy_only),
                stacklevel=2,
            )
        if loss_type == "multimodal_causal" and (
            rep_alpha != 0.5 or pred_weight != 0.5
        ):
            raise ValueError(
                "rep_alpha/pred_weight only apply to contrastive_legacy. "
                "For multimodal_causal use loss_anchor_weight/loss_align_weight."
            )
        if conditioning is not None:
            raise ValueError(
                "conditioning is a legacy ResponseEmbeddingModel option; "
                "STC always consumes the intervention branch T."
            )

    if conditioning in {"station", "opening_year_station"}:
        if station_attributes_path is None:
            raise ValueError(
                "station conditioning requires --station-attributes / station_attributes_path"
            )
        if train_ds.station_token_count == 0:
            raise ValueError(
                "station conditioning is enabled but no station attribute tokens were "
                "loaded; check the station-attributes parquet coverage"
            )
        if not (
            train_ds.station_attributes_binding
            == val_ds.station_attributes_binding
            == test_ds.station_attributes_binding
        ):
            raise ValueError("Station attributes changed while loading the training splits")

    station_attributes_binding = (
        train_ds.station_attributes_binding
        if conditioning in {"station", "opening_year_station"}
        else None
    )

    from typing import Any

    train_loader: DataLoader[Any] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: _collate_fn(b, train_ds, actual_use_images),
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate_fn(b, val_ds, actual_use_images),
    )
    test_loader: DataLoader[Any] = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate_fn(b, test_ds, actual_use_images),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = {
        "schema": "urban_intervention_training_config_v1",
        "model_inputs_dir": str(Path(model_inputs_dir).resolve()),
        "dataset_id": train_ds.manifest.get("dataset_id", ""),
        "strict_production": bool(train_ds.manifest.get("strict_production", False)),
        "station_attributes_binding": station_attributes_binding,
        "streetview_index_path": str(streetview_index_path) if streetview_index_path is not None else None,
        "satellite_root": str(satellite_root) if satellite_root is not None else None,
        "architecture": {
            "feature_dim": train_ds.feature_dim(),
            "hidden_dims": list(hidden_dims),
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "use_image_encoder": actual_use_images,
            "max_images_per_grid": max_images_per_grid,
            "image_pooling": image_pooling,
            "use_satellite_encoder": actual_use_satellite,
            "use_cross_view_synergy": use_cross_view_synergy,
            "synergy_heads": synergy_heads,
            "conditioning": conditioning,
        },
        "optimization": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "temperature": temperature,
            "rep_alpha": rep_alpha,
            "pred_weight": pred_weight,
            "scheduler": "cosine_annealing",
        },
        "loss_type": loss_type,
        "loss_anchor_weight": loss_anchor_weight,
        "loss_align_weight": loss_align_weight,
        "loss_huber_delta": loss_huber_delta,
        "allow_satellite_proxy": allow_satellite_proxy,
        "loss_config": {
            "loss_type": loss_type,
            "anchor_weight": loss_anchor_weight,
            "align_weight": loss_align_weight,
            "huber_delta": loss_huber_delta,
            "allow_satellite_proxy": allow_satellite_proxy,
        },
        "algorithm": {
            "se_shrinkage": se_shrinkage,
            "queue_size": queue_size,
            "learnable_temperature": learnable_temperature,
            "uncertainty_weighted": uncertainty_weighted,
            "conditioning": conditioning,
            "conditioning_schema": "v2",
        },
        "data_splits": {
            "train_units": len(train_ds),
            "validation_units": len(val_ds),
            "test_units": len(test_ds),
            "response_dim": train_ds.response_dim(),
        },
        "evaluation": {
            "k": eval_k,
            "n_perm": eval_n_perm,
            "n_boot": eval_n_boot,
            "probe_ridge": probe_ridge,
            "probe_min_obs": probe_min_obs,
            "baseline_autoencoder_epochs": baseline_ae_epochs,
        },
        "seed": seed,
        "device": str(dev),
        "created_utc": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    if train_ds.is_stc:
        training_config["streetview_input"] = {
            "enabled": actual_use_images,
            "selection": "all_verified_pre_opening_images",
            "temporal_gap_source": "capture_month_to_opening_month",
            "temporal_gap_unit": "calendar_months",
            "temporal_gap_compression": "d/(d+12)",
            "aggregation": "learned_time_conditioned_attention",
            "output": "one_fixed_size_streetview_vector_per_event",
        }
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if train_ds.is_stc:
        single_family = train_ds.families[0] if train_ds.families and len(train_ds.families) == 1 else None
        model = STCResponseModel(
            state_tabular_dim=len(train_ds.state_tabular_columns),
            intervention_dim=len(train_ds.intervention_columns),
            context_tabular_dim=len(train_ds.context_tabular_columns),
            alphaearth_dim=64,
            z_dim=embedding_dim,
            t_dim=32,
            b_dim=64,
            predictor_hidden_dim=128,
            outcome_dim=train_ds.response_dim(),
            outcome_family=single_family,
            use_streetview=actual_use_images,
            use_satellite=actual_use_satellite,
            streetview_time_aware=actual_use_images,
            dropout=dropout,
            learnable_temperature=learnable_temperature,
            uncertainty_weighted=uncertainty_weighted,
        ).to(dev)
    else:
        model = ResponseEmbeddingModel(
            feature_dim=train_ds.feature_dim(),
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_image_encoder=actual_use_images,
            image_pooling=image_pooling,
            use_satellite_encoder=actual_use_satellite,
            use_cross_view_synergy=use_cross_view_synergy,
            synergy_heads=synergy_heads,
            conditioning=conditioning,
            learnable_temperature=learnable_temperature,
            uncertainty_weighted=uncertainty_weighted,
        ).to(dev)

    queue = EmbeddingQueue(embedding_dim, queue_size, dev) if queue_size > 0 else None
    if queue is not None and queue_size < 2 * batch_size:
        warnings.warn(
            f"queue_size ({queue_size}) is smaller than 2*batch_size ({2 * batch_size}); "
            "the response-aware history will contain few cross-batch pairs. "
            "Use a larger queue (e.g. 2048-4096) when memory permits.",
            stacklevel=2,
        )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    checkpoint_path: Path | None = None
    history: list[dict] = []

    for epoch in range(epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            train_ds,
            dev,
            temperature,
            rep_alpha,
            pred_weight,
            actual_use_images,
            is_training=True,
            optimizer=optimizer,
            queue=queue,
            se_shrinkage=se_shrinkage,
            learnable_temperature=learnable_temperature,
            loss_type=loss_type,
            loss_anchor_weight=loss_anchor_weight,
            loss_align_weight=loss_align_weight,
            loss_huber_delta=loss_huber_delta,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            val_ds,
            dev,
            temperature,
            rep_alpha,
            pred_weight,
            actual_use_images,
            is_training=False,
            se_shrinkage=se_shrinkage,
            learnable_temperature=learnable_temperature,
            eval_k=eval_k,
            loss_type=loss_type,
            loss_anchor_weight=loss_anchor_weight,
            loss_align_weight=loss_align_weight,
            loss_huber_delta=loss_huber_delta,
        )

        # Step the schedule only when this epoch actually performed optimizer
        # steps; otherwise PyTorch skips the first LR value and warns that
        # scheduler.step() preceded optimizer.step().
        if train_metrics.get("steps", 0) > 0:
            scheduler.step()

        lr = scheduler.get_last_lr()[0]
        weight_note = ""
        if "rep_weight" in train_metrics:
            weight_note = (
                f" | rep_weight {train_metrics['rep_weight']:.3f}, "
                f"pred_weight {train_metrics['pred_weight']:.3f}"
            )
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss {train_metrics['total']:.4f} "
            f"(rep {train_metrics['representation']:.4f}, "
            f"pred {train_metrics['prediction']:.4f}) | "
            f"val loss {val_metrics['total']:.4f} "
            f"(rep {val_metrics['representation']:.4f}, "
            f"pred {val_metrics['prediction']:.4f}) | "
            f"val nn_corr@{eval_k} {val_metrics.get('mean_nn_corr@k', 0):.4f} | "
            f"lr {lr:.6f}{weight_note}",
            file=sys.stdout,
            flush=True,
        )

        epoch_log = {
            "epoch": epoch,
            "train_total": round(train_metrics["total"], 6),
            "train_rep": round(train_metrics["representation"], 6),
            "train_pred": round(train_metrics["prediction"], 6),
            "val_total": round(val_metrics["total"], 6),
            "val_rep": round(val_metrics["representation"], 6),
            "val_pred": round(val_metrics["prediction"], 6),
            f"val_nn_corr@{eval_k}": val_metrics.get("mean_nn_corr@k", 0),
            "val_baseline_corr": val_metrics.get("baseline_corr", 0),
            "lr": lr,
        }
        if "rep_weight" in train_metrics:
            epoch_log["train_rep_weight"] = round(train_metrics["rep_weight"], 6)
            epoch_log["train_pred_weight"] = round(train_metrics["pred_weight"], 6)
        history.append(epoch_log)

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = output_dir / "best_model.pt"
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["total"],
                "feature_dim": train_ds.feature_dim(),
                "feature_columns": list(train_ds.feature_columns),
                "dataset_id": train_ds.manifest.get("dataset_id", ""),
                "feature_artifact_sha256": train_ds.manifest.get("outputs", {}).get(
                    "state_inputs.parquet" if train_ds.is_stc else "unit_features.parquet", ""
                ),
                "normalization_sha256": hashlib.sha256(
                    json.dumps(
                        train_ds.normalization,
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest(),
                "response_artifact_sha256": train_ds.manifest.get(
                    "response_artifact_sha256", ""
                ),
                "embedding_dim": embedding_dim,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "use_image_encoder": actual_use_images,
                "max_images_per_grid": max_images_per_grid,
                "image_pooling": image_pooling,
                "use_satellite_encoder": actual_use_satellite,
                "use_cross_view_synergy": use_cross_view_synergy,
                "synergy_heads": synergy_heads,
                "conditioning": conditioning,
                "conditioning_schema": "v2",
                "station_attributes_binding": station_attributes_binding,
                "learnable_temperature": learnable_temperature,
                "uncertainty_weighted": uncertainty_weighted,
                "loss_type": loss_type,
                "loss_anchor_weight": loss_anchor_weight,
                "loss_align_weight": loss_align_weight,
                "loss_huber_delta": loss_huber_delta,
                "allow_satellite_proxy": allow_satellite_proxy,
                "loss_config": {
                    "loss_type": loss_type,
                    "anchor_weight": loss_anchor_weight,
                    "align_weight": loss_align_weight,
                    "huber_delta": loss_huber_delta,
                    "allow_satellite_proxy": allow_satellite_proxy,
                },
                "streetview_index_path": str(streetview_index_path) if streetview_index_path is not None else None,
                "satellite_root": str(satellite_root) if satellite_root is not None else None,
            }
            if train_ds.is_stc:
                ckpt_dict["is_stc"] = True
                ckpt_dict["model_type"] = "stc"
                single_fam = train_ds.families[0] if train_ds.families and len(train_ds.families) == 1 else None
                outputs_shas = dict(train_ds.manifest.get("outputs", {}))
                ckpt_dict["bundle_shas"] = outputs_shas
                ckpt_dict["stc_contract"] = {
                    "state_tabular_columns": list(train_ds.state_tabular_columns),
                    "intervention_columns": list(train_ds.intervention_columns),
                    "context_tabular_columns": list(train_ds.context_tabular_columns),
                    "satellite_relative_years": list(train_ds.satellite_relative_years),
                    "families": list(train_ds.families) if train_ds.families else None,
                    "response_offsets": train_ds.response_offsets(),
                    "response_dim": train_ds.response_dim(),
                    "outcome_family": single_fam,
                    "normalization": train_ds.normalization,
                    "bundle_shas": outputs_shas,
                    "streetview_index_path": str(streetview_index_path) if streetview_index_path is not None else None,
                    "streetview_assets_sha256": getattr(train_ds, "streetview_assets_sha256", None),
                    "streetview_time_aware": actual_use_images,
                    "streetview_selection_policy": "all_verified_pre_opening_images",
                    "streetview_time_gap": {
                        "source": "capture_month_to_opening_month",
                        "unit": "calendar_months",
                        "compression": "d/(d+12)",
                        "half_scale_months": 12.0,
                    },
                    "satellite_root": str(satellite_root) if satellite_root is not None else None,
                }
            torch.save(ckpt_dict, checkpoint_path)

    if checkpoint_path is None:
        raise RuntimeError(
            "No valid checkpoint produced — all validation losses may be NaN "
            "or no batches completed. Check dataset size and batch_size/drop_last settings."
        )
    # Evaluate the best-validation model, not the last epoch's weights, so
    # test_metrics.json matches the state saved in best_model.pt.
    best_checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    test_metrics = _run_epoch(
        model,
        test_loader,
        test_ds,
        dev,
        temperature,
        rep_alpha,
        pred_weight,
        actual_use_images,
        is_training=False,
        se_shrinkage=se_shrinkage,
        learnable_temperature=learnable_temperature,
        eval_k=eval_k,
        loss_type=loss_type,
        loss_anchor_weight=loss_anchor_weight,
        loss_align_weight=loss_align_weight,
        loss_huber_delta=loss_huber_delta,
    )
    test_report = {k: round(v, 6) if isinstance(v, float) else v for k, v in test_metrics.items()}
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    evaluation_report = build_evaluation_report(
        model,
        {"train": train_loader, "validation": val_loader, "test": test_loader},
        dev,
        k=eval_k,
        n_perm=eval_n_perm,
        n_boot=eval_n_boot,
        ridge=probe_ridge,
        seed=seed,
        probe_min_obs=probe_min_obs,
        run_baselines=run_baselines,
        run_transfer=run_transfer,
        transfer_shot_sizes=transfer_shot_sizes,
        baseline_ae_epochs=baseline_ae_epochs,
        transfer_n_seeds=transfer_n_seeds,
        visualize=visualize,
        visualization_dir=output_dir,
    )
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(evaluation_report, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    _append_run_record(
        output_dir,
        training_config,
        history,
        test_metrics,
        evaluation_report,
    )
    return checkpoint_path


def _append_run_record(
    output_dir: Path,
    training_config: dict[str, object],
    history: list[dict],
    test_metrics: dict[str, float],
    evaluation_report: dict[str, object],
) -> None:
    """Append one JSONL run record for experiment tracking.

    The record carries a content hash of the training configuration plus the
    headline test metrics and the chance-level baselines, so every run is
    self-describing and comparable without loading the full report.
    """
    # The digest covers only the reproducibility-relevant configuration:
    # created_utc, the absolute model_inputs_dir and the device are run
    # metadata, not hyperparameters, so identical configurations hash equal.
    hash_config = {
        key: value
        for key, value in training_config.items()
        if key not in {"created_utc", "model_inputs_dir", "device", "runtime"}
    }
    config_digest = hashlib.sha256(
        json.dumps(hash_config, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
    best_val = min((entry["val_total"] for entry in history), default=None)
    record: dict[str, object] = {
        "created_utc": training_config["created_utc"],
        "dataset_id": training_config.get("dataset_id", ""),
        "config_sha256": config_digest,
        "config": training_config,
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
    }
    baselines = evaluation_report.get("baselines")
    if isinstance(baselines, dict):
        test_baselines = baselines.get("test")
        if isinstance(test_baselines, dict):
            from .baselines import headline_metric

            record["baseline_nn_corr@k"] = {
                name: headline_metric(entry)
                for name, entry in test_baselines.items()
                if isinstance(entry, dict)
            }
    test_entry = evaluation_report.get("test")
    if isinstance(test_entry, dict):
        retrieval = test_entry.get("retrieval")
        if isinstance(retrieval, dict):
            overall = retrieval.get("overall")
            if isinstance(overall, dict):
                record["test_nn_corr@k"] = overall.get("nn_corr@k")
    runs_path = output_dir / "runs.jsonl"
    with runs_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
