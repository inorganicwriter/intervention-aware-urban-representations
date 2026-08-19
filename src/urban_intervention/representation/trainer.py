"""Training loop for response-aligned urban representation learning."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import typing
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
    probe_rmse,
    response_similarity_with_validity,
    retrieval_metrics,
)
from .loss import total_loss
from .model import ResponseEmbeddingModel
from .queue import EmbeddingQueue
from .transfer import predictive_transfer_report, transfer_report


class Pool(typing.TypedDict):
    embeddings: torch.Tensor
    features: torch.Tensor
    responses: torch.Tensor
    masks: torch.Tensor
    city_keys: list[str]
    quality_grades: list[str]


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
    with torch.no_grad():
        for batch in loader:
            feat = batch["features"].to(device)
            resp = batch["responses"].to(device)
            mask = batch["response_mask"].to(device)
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
            embedding, _ = model(
                feat, images, image_mask,
                conditioning_tokens=conditioning_tokens,
                station_tokens=station_tokens,
            )
            embeddings.append(embedding.cpu())
            features.append(feat.cpu())
            responses.append(resp.cpu())
            masks.append(mask.cpu())
            city_keys.extend(str(value) for value in batch.get("city_keys", []))
            quality_grades.extend(str(value) for value in batch.get("quality_grades", []))
    if not embeddings:
        return None
    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "features": torch.cat(features, dim=0),
        "responses": torch.cat(responses, dim=0),
        "masks": torch.cat(masks, dim=0),
        "city_keys": city_keys,
        "quality_grades": quality_grades,
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
                pool["responses"], pool["masks"]
            )
            entry["n_units"] = int(pool["embeddings"].shape[0])
            entry["retrieval"] = retrieval_metrics(
                pool["embeddings"], pool["responses"], pool["masks"], k=k
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
            )
            entry["raw_feature_baseline"] = retrieval_metrics(
                pool["features"], pool["responses"], pool["masks"], k=k
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
                    device=device,
                    k=k,
                    n_perm=n_perm,
                    n_boot=n_boot,
                    seed=seed,
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
) -> Path:
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)

    train_ds = RepresentationDataset(
        model_inputs_dir,
        split="train",
        max_images_per_grid=max_images_per_grid,
        load_images=use_images,
        only_training_mask=True,
        station_attributes_path=station_attributes_path,
    )
    val_ds = RepresentationDataset(
        model_inputs_dir,
        split="validation",
        max_images_per_grid=max_images_per_grid,
        load_images=use_images,
        only_training_mask=True,
        station_attributes_path=station_attributes_path,
    )
    test_ds = RepresentationDataset(
        model_inputs_dir,
        split="test",
        max_images_per_grid=max_images_per_grid,
        load_images=use_images,
        only_training_mask=True,
        station_attributes_path=station_attributes_path,
    )
    split_sizes = {"train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)}
    too_small = {name: size for name, size in split_sizes.items() if size < 2}
    if too_small:
        details = ", ".join(f"{name}={size}" for name, size in too_small.items())
        raise ValueError(
            "Every training/evaluation split needs at least 2 eligible units; " + details
        )

    from typing import Any

    train_loader: DataLoader[Any] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: _collate_fn(b, train_ds, use_images),
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate_fn(b, val_ds, use_images),
    )
    test_loader: DataLoader[Any] = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate_fn(b, test_ds, use_images),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = {
        "schema": "urban_intervention_training_config_v1",
        "model_inputs_dir": str(Path(model_inputs_dir).resolve()),
        "dataset_id": train_ds.manifest.get("dataset_id", ""),
        "strict_production": bool(train_ds.manifest.get("strict_production", False)),
        "architecture": {
            "feature_dim": train_ds.feature_dim(),
            "hidden_dims": list(hidden_dims),
            "embedding_dim": embedding_dim,
            "dropout": dropout,
            "use_image_encoder": use_images,
            "max_images_per_grid": max_images_per_grid,
            "image_pooling": image_pooling,
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
        "algorithm": {
            "se_shrinkage": se_shrinkage,
            "queue_size": queue_size,
            "learnable_temperature": learnable_temperature,
            "uncertainty_weighted": uncertainty_weighted,
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
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model = ResponseEmbeddingModel(
        feature_dim=train_ds.feature_dim(),
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
        dropout=dropout,
        use_image_encoder=use_images,
        image_pooling=image_pooling,
        conditioning=conditioning,
        learnable_temperature=learnable_temperature,
        uncertainty_weighted=uncertainty_weighted,
    ).to(dev)

    queue = EmbeddingQueue(embedding_dim, queue_size, dev) if queue_size > 0 else None
    if queue is not None and queue_size < 2 * batch_size:
        import warnings

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
            use_images,
            is_training=True,
            optimizer=optimizer,
            queue=queue,
            se_shrinkage=se_shrinkage,
            learnable_temperature=learnable_temperature,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            val_ds,
            dev,
            temperature,
            rep_alpha,
            pred_weight,
            use_images,
            is_training=False,
            se_shrinkage=se_shrinkage,
            learnable_temperature=learnable_temperature,
            eval_k=eval_k,
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
            f"val nn_corr@5 {val_metrics.get('mean_nn_corr@k', 0):.4f} | "
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
            "val_nn_corr@5": val_metrics.get("mean_nn_corr@k", 0),
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["total"],
                    "feature_dim": train_ds.feature_dim(),
                    "feature_columns": list(train_ds.feature_columns),
                    "dataset_id": train_ds.manifest.get("dataset_id", ""),
                    "feature_artifact_sha256": train_ds.manifest.get("outputs", {}).get(
                        "unit_features.parquet", ""
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
                    "use_image_encoder": use_images,
                    "max_images_per_grid": max_images_per_grid,
                    "image_pooling": image_pooling,
                    "conditioning": conditioning,
                    "learnable_temperature": learnable_temperature,
                    "uncertainty_weighted": uncertainty_weighted,
                },
                checkpoint_path,
            )

    if checkpoint_path is None:
        raise RuntimeError(
            "No valid checkpoint produced — all validation losses may be NaN "
            "or no batches completed. Check dataset size and batch_size/drop_last settings."
        )
    # Evaluate the best-validation model, not the last epoch's weights, so
    # test_metrics.json matches the state saved in best_model.pt.
    best_checkpoint = torch.load(
        checkpoint_path, map_location=dev, weights_only=True
    )
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
        use_images,
        is_training=False,
        se_shrinkage=se_shrinkage,
        learnable_temperature=learnable_temperature,
        eval_k=eval_k,
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
