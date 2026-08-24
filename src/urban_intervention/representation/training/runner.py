"""End-to-end modular representation-training runner."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..dataset import RepresentationDataset
from ..model import ResponseEmbeddingModel
from ..queue import EmbeddingQueue
from .batching import _collate_fn
from .epoch import _run_epoch
from .evaluation import build_evaluation_report
from .tracking import _append_run_record


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
