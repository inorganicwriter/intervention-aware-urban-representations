"""Shared implementation of the three controlled STC training experiments."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataset import RepresentationDataset
from ..evaluation import (
    bootstrap_ci,
    cosine_similarity,
    permutation_test,
    probe_rmse,
    response_similarity_with_validity,
    retrieval_metrics,
    trajectory_prediction_metrics,
)
from ..joint_loss import TASK_DIMS, TRAINING_TASKS, joint_causal_loss
from ..model import JointSTCResponseModel
from ..transfer import per_city_retrieval
from .batching import _collate_fn
from .tracking import _append_run_record


def uses_joint_bundle(root: Path) -> bool:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    return manifest.get("schema") == "urban_intervention_stc_dataset" and set(
        manifest.get("families") or []
    ) == set(TASK_DIMS)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _norm(gradients) -> float:
    return math.sqrt(
        sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)
    )


def task_gradient_diagnostics(model, losses) -> dict[str, object]:
    """Observe per-task gradients without changing parameter .grad buffers."""
    params = [p for p in model.state_encoder.parameters() if p.requires_grad]
    gradients = {}
    result = {}
    for family in TASK_DIMS:
        task = losses["tasks"][family]
        gradients[family] = torch.autograd.grad(
            task["diagnostic_loss"], params, allow_unused=True, retain_graph=True
        )
        result[f"state_grad_norm_{family}"] = _norm(gradients[family])
    dot = sum(
        float(a.detach().double().mul(b.detach().double()).sum())
        for a, b in zip(gradients["viirs"], gradients["poi"], strict=True)
        if a is not None and b is not None
    )
    denominator = result["state_grad_norm_viirs"] * result["state_grad_norm_poi"]
    cosine = max(-1.0, min(1.0, dot / denominator)) if denominator > 0 else None
    result.update(
        task_gradient_cosine=cosine,
        task_gradient_conflict=(cosine < 0) if cosine is not None else None,
    )
    return result


def _summarize(losses) -> dict[str, float]:
    result = {key: float(losses[key].detach()) for key in ("total", "anchor", "align")}
    for family, task in losses["tasks"].items():
        for key in (
            "anchor",
            "align",
            "anchor_active",
            "align_active",
            "valid_cells",
            "valid_pairs",
            "valid_anchors",
            "anchor_coefficient",
            "align_coefficient",
        ):
            value = task[key]
            result[f"{family}_{key}"] = (
                float(value.detach()) if torch.is_tensor(value) else float(value)
            )
    result["active_anchor_tasks"] = float(losses["active_anchor_tasks"])
    result["active_align_tasks"] = float(losses["active_align_tasks"])
    return result


def run_joint_epoch(
    model,
    loader,
    device,
    loss_options,
    optimizer=None,
    max_grad_norm=1.0,
    gradient_diagnostics_interval=1,
):
    training = optimizer is not None
    model.train(training)
    totals, diagnostics, parts = [], [], []
    steps = skipped = 0
    offsets = loader.dataset.response_offsets()
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if training:
                valid = (
                    batch["response_mask"]
                    & torch.isfinite(batch["responses"])
                    & torch.isfinite(batch["response_weight"])
                    & (batch["response_weight"] > 0)
                )
                has_supervision = any(
                    valid[:, offsets[f][0] : offsets[f][1]].any().item()
                    for f in TRAINING_TASKS[loss_options["training_mode"]]
                )
                if not has_supervision:
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
            diagnose = (
                training
                and gradient_diagnostics_interval > 0
                and batch_index % gradient_diagnostics_interval == 0
            )
            if diagnose:
                batch["alphaearth"].requires_grad_(True)
                batch["context"].requires_grad_(True)
            z, predictions, fused = model.forward_batch(batch)
            losses = joint_causal_loss(
                z,
                predictions,
                batch["responses"],
                batch["response_mask"],
                offsets,
                response_weight=batch["response_weight"],
                response_se=batch["response_se"] if loss_options["se_shrinkage"] else None,
                **{k: v for k, v in loss_options.items() if k != "se_shrinkage"},
            )
            totals.append(_summarize(losses))
            if training:
                optimizer.zero_grad(set_to_none=True)
                if not losses["update_active"]:
                    skipped += 1
                    continue
                record = {"batch": batch_index}
                if diagnose:
                    record.update(task_gradient_diagnostics(model, losses))
                    input_grads = torch.autograd.grad(
                        losses["total"],
                        [batch["alphaearth"], batch["context"]],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    record["alphaearth_input_grad_norm"] = _norm(input_grads[:1])
                    record["context_tabular_input_grad_norm"] = _norm(input_grads[1:])
                    condition_gradient = torch.autograd.grad(
                        losses["total"], fused, retain_graph=True, allow_unused=True
                    )[0]
                    if condition_gradient is None:
                        condition_gradient = torch.zeros_like(fused)
                    record["alphaearth_condition_grad_norm"] = _norm(
                        [condition_gradient[:, 160:224]]
                    )
                    record["context_tabular_condition_grad_norm"] = _norm(
                        [condition_gradient[:, 224:256]]
                    )
                    background = fused.detach()[:, 160:]
                    record["alphaearth_rms"] = float(background[:, :64].square().mean().sqrt())
                    record["context_tabular_rms"] = float(background[:, 64:].square().mean().sqrt())
                losses["total"].backward()
                threshold = float("inf") if max_grad_norm is None else max_grad_norm
                before = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), threshold, error_if_nonfinite=True
                )
                record["joint_grad_norm_before_clip"] = float(before)
                record["joint_grad_norm_after_clip"] = _norm(p.grad for p in model.parameters())
                record["clip_scale"] = min(1.0, threshold / (float(before) + 1e-6))
                if diagnose:
                    record["context_tabular_parameter_grad_norm"] = _norm(
                        p.grad for p in model.context_encoder.tab_proj.parameters()
                    )
                    record["alphaearth_missing_token_grad_norm"] = _norm(
                        [model.context_encoder.missing_ae_token.grad]
                    )
                diagnostics.append(record)
                optimizer.step()
                steps += 1
            else:
                parts.append(
                    {
                        "embeddings": z.cpu(),
                        "responses": batch["responses"].cpu(),
                        "masks": batch["response_mask"].cpu(),
                        "weights": batch["response_weight"].cpu(),
                        "se": batch["response_se"].cpu(),
                        "features": torch.cat(
                            [
                                batch["state_tabular"],
                                batch["context"],
                                batch["intervention"],
                                batch["alphaearth"],
                            ],
                            -1,
                        ).cpu(),
                        "orders": batch["treatment_order"].cpu(),
                        "city_keys": batch["city_keys"],
                        "predictions": {k: v.cpu() for k, v in predictions.items()},
                    }
                )
    if training:
        metrics = (
            {k: float(np.mean([row[k] for row in totals])) for k in totals[0]} if totals else {}
        )
        metrics.update(steps=steps, skipped_batches=skipped)
        if diagnostics:
            for key in set().union(*(record.keys() for record in diagnostics)):
                values = [r[key] for r in diagnostics if r.get(key) is not None]
                if key != "batch" and values:
                    metrics[key] = float(np.mean(values))
        return metrics, None, diagnostics
    if not parts:
        raise ValueError("Joint evaluation split is empty")
    pool = {
        k: torch.cat([p[k] for p in parts])
        for k in ("embeddings", "responses", "masks", "weights", "se", "features", "orders")
    }
    pool["city_keys"] = [c for part in parts for c in part["city_keys"]]
    pool["predictions"] = {f: torch.cat([p["predictions"][f] for p in parts]) for f in TASK_DIMS}
    losses = joint_causal_loss(
        pool["embeddings"],
        pool["predictions"],
        pool["responses"],
        pool["masks"],
        offsets,
        response_weight=pool["weights"],
        response_se=pool["se"] if loss_options["se_shrinkage"] else None,
        **{k: v for k, v in loss_options.items() if k != "se_shrinkage"},
    )
    metrics = _summarize(losses)
    metrics["supervision_active"] = bool(losses["update_active"])
    return metrics, pool, []


def evaluate_joint_pool(
    pool,
    train_pool,
    offsets,
    *,
    k=5,
    n_perm=100,
    n_boot=200,
    seed=42,
    run_transfer=True,
    mode="joint",
):
    report = {"n_units": len(pool["orders"]), "tasks": {}}
    for family, (start, end, _) in offsets.items():
        y, mask = pool["responses"][:, start:end], pool["masks"][:, start:end]
        mask = mask & torch.isfinite(y) & torch.isfinite(pool["weights"][:, start:end])
        mask = mask & (pool["weights"][:, start:end] > 0)
        layout = {family: (0, end - start, [])}
        pred = pool["predictions"][family]
        target_train = train_pool["responses"][:, start:end]
        train_mask = train_pool["masks"][:, start:end] & torch.isfinite(target_train)
        train_mask = train_mask & (train_pool["weights"][:, start:end] > 0)
        mean = torch.where(train_mask, target_train, 0.0).sum(0) / train_mask.sum(0).clamp_min(1)
        comparable = mask & (train_mask.sum(0) > 0)[None, :]
        prediction = trajectory_prediction_metrics({family: pred}, y, mask, response_offsets=layout)
        mse = (
            float((pred[comparable] - y[comparable]).square().mean()) if comparable.any() else None
        )
        baseline_mse = (
            float((mean.expand_as(y)[comparable] - y[comparable]).square().mean())
            if comparable.any()
            else None
        )
        support = {"orders": pool["orders"].tolist(), "mask": mask.tolist()}
        sim, valid = response_similarity_with_validity(y, mask, response_offsets=layout)
        emb_cos = cosine_similarity(pool["embeddings"])
        entry = {
            "trained": family in TRAINING_TASKS[mode],
            "prediction": prediction.get(family, {"evaluated_cells": 0}),
            "train_mean_baseline": {
                "model_mse": mse,
                "baseline_mse": baseline_mse,
                "improvement": 1 - mse / baseline_mse if baseline_mse and mse is not None else None,
                "comparable_cells": int(comparable.sum()),
            },
            "support_sha256": hashlib.sha256(
                json.dumps(support, sort_keys=True).encode()
            ).hexdigest(),
            "evaluated_cells": int(mask.sum()),
            "test_cities": sorted(set(pool["city_keys"])),
            "retrieval": retrieval_metrics(
                pool["embeddings"], y, mask, k=k, response_offsets=layout
            ),
            "permutation": permutation_test(
                emb_cos, y, mask, k=k, n_perm=n_perm, seed=seed, response_offsets=layout
            ),
            "bootstrap_ci": bootstrap_ci(
                emb_cos, sim, k=k, n_boot=n_boot, seed=seed, valid_pairs=valid
            ),
            "per_city_prediction": {},
        }
        for city in entry["test_cities"]:
            indices = torch.tensor([i for i, c in enumerate(pool["city_keys"]) if c == city])
            metric = trajectory_prediction_metrics(
                {family: pred[indices]}, y[indices], mask[indices], response_offsets=layout
            )
            entry["per_city_prediction"][city] = metric.get(family, {"evaluated_cells": 0})
        if run_transfer:
            entry["per_city_retrieval"] = per_city_retrieval(
                pool["embeddings"], y, mask, pool["city_keys"], k, response_offsets=layout
            )
            entry["probe"] = probe_rmse(
                train_pool["embeddings"],
                target_train,
                train_mask,
                pool["embeddings"],
                y,
                mask,
                min_obs=2,
                response_offsets=layout,
            )
        report["tasks"][family] = entry
    return report


def train_joint_representation(
    model_inputs_dir,
    output_dir,
    training_mode=None,
    embedding_dim=128,
    dropout=0.1,
    learning_rate=1e-3,
    weight_decay=1e-5,
    batch_size=64,
    epochs=100,
    seed=42,
    device=None,
    use_images=False,
    use_satellite=False,
    require_multimodal=False,
    require_satellite=False,
    require_streetview=False,
    max_images_per_grid=4,
    verify_bundle=False,
    streetview_index_path=None,
    satellite_root=None,
    loss_anchor_weight=0.85,
    loss_align_weight=0.15,
    loss_huber_delta=1.0,
    temperature=0.07,
    se_shrinkage=True,
    max_grad_norm=1.0,
    gradient_diagnostics_interval=1,
    prediction_inputs="full",
    eval_k=5,
    eval_n_perm=100,
    eval_n_boot=200,
    run_baselines=True,
    run_transfer=True,
    **legacy_options,
):
    mode = training_mode or "joint"
    if mode not in TRAINING_TASKS:
        raise ValueError(f"Unknown training_mode {mode}")
    if embedding_dim != 128:
        raise ValueError("New shared STC experiments require embedding_dim=128")
    if batch_size < 2 or epochs < 1:
        raise ValueError("batch_size must be at least 2 and epochs must be positive")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max_grad_norm must be positive and finite, or None")
    if gradient_diagnostics_interval < 0:
        raise ValueError("gradient_diagnostics_interval must be non-negative")
    if legacy_options.get("queue_size", 0) or legacy_options.get("uncertainty_weighted", False):
        raise ValueError("Shared STC baseline requires queue_size=0 and fixed task weights")
    if legacy_options.get("loss_type", "multimodal_causal") != "multimodal_causal":
        raise ValueError("Shared STC requires the multimodal_causal objective")
    if legacy_options.get("learnable_temperature", False):
        raise ValueError("Shared STC baseline uses a fixed contrastive temperature")
    if prediction_inputs == "tc_only":
        loss_align_weight = 0.0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root, out = Path(model_inputs_dir), Path(output_dir)
    if (out / "best_model.pt").exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment {out}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    images = bool(use_images or require_streetview or require_multimodal)
    satellite = bool(use_satellite or require_satellite or require_multimodal)
    datasets = {
        s: RepresentationDataset(
            root,
            split=s,
            only_training_mask=True,
            load_images=images,
            load_satellite_images=satellite,
            verify_bundle=True,
            max_images_per_grid=max_images_per_grid,
            require_multimodal=require_multimodal,
            require_satellite=require_satellite,
            require_streetview=require_streetview,
            streetview_index_path=streetview_index_path,
            satellite_root=satellite_root,
        )
        for s in ("train", "validation", "test")
    }
    train_ds = datasets["train"]
    if not train_ds.is_stc or set(train_ds.families or []) != set(TASK_DIMS):
        raise ValueError("All three experiments require the same joint STC VIIRS+POI bundle")
    if any(len(ds) < 2 for ds in datasets.values()):
        raise ValueError("Each city-held-out split needs at least two events")
    city_sets = {s: set(ds.sample_index.city_key) for s, ds in datasets.items()}
    if any(
        city_sets[a] & city_sets[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Joint experiments require disjoint city-level splits")
    keys = ["treatment_order", "outcome_family", "outcome", "event_time"]
    if train_ds.response_targets.duplicated(keys).any():
        raise ValueError("Joint response targets contain duplicate task cells")
    offsets = train_ds.response_offsets()
    model = JointSTCResponseModel(
        len(train_ds.state_tabular_columns),
        len(train_ds.context_tabular_columns),
        len(train_ds.intervention_columns),
        use_streetview=images,
        use_satellite=satellite,
        dropout=dropout,
        prediction_inputs=prediction_inputs,
    ).to(dev)
    loaders = {
        s: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(s == "train"),
            collate_fn=partial(_collate_fn, ds=ds, use_images=images),
        )
        for s, ds in datasets.items()
    }
    # Separate non-shuffled train loader for evaluation; evaluation never advances
    # the training sampler's RNG or changes which events a baseline sees.
    train_eval = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(_collate_fn, ds=train_ds, use_images=images),
    )
    loss_options = dict(
        training_mode=mode,
        anchor_weight=loss_anchor_weight,
        align_weight=loss_align_weight,
        delta=loss_huber_delta,
        temperature=temperature,
        se_shrinkage=se_shrinkage,
    )
    config = dict(
        architecture_version=model.architecture_version,
        model_type="stc_joint",
        training_mode=mode,
        trained_tasks=list(TRAINING_TASKS[mode]),
        dataset_id=train_ds.manifest.get("dataset_id"),
        model_inputs_dir=str(root.resolve()),
        created_utc=datetime.now(UTC).isoformat(),
        seed=seed,
        device=str(dev),
        dimensions={"z": 128, "t": 32, "b": 96, "predictor_input": 256, "viirs": 6, "poi": 12},
        embedding_dim=128,
        dropout=dropout,
        prediction_inputs=prediction_inputs,
        use_images=images,
        use_satellite=satellite,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        gradient_diagnostics_interval=gradient_diagnostics_interval,
        gradient_diagnostic_definition="actual per-task contribution, including independent active-anchor and active-alignment normalization",
        loss=loss_options,
        response_offsets=offsets,
        data_splits={
            s: sorted(ds.sample_index.city_key.unique().tolist()) for s, ds in datasets.items()
        },
        bundle_shas=dict(train_ds.manifest.get("outputs", {})),
        strict_production=bool(train_ds.manifest.get("strict_production", False)),
        eval_k=eval_k,
        eval_n_perm=eval_n_perm,
        eval_n_boot=eval_n_boot,
        run_baselines=run_baselines,
        run_transfer=run_transfer,
    )
    config["data_splits"].update({f"{split}_units": len(ds) for split, ds in datasets.items()})
    config["data_splits"]["response_dim"] = 18
    config["architecture"] = {
        "model_type": "stc_joint",
        "embedding_dim": 128,
        "dimensions": config["dimensions"],
        "hidden_dims": (128,),
        "feature_dim": train_ds.feature_dim(),
        "use_image_encoder": images,
        "use_satellite_encoder": satellite,
        "image_pooling": "time_conditioned_attention",
    }
    config["optimization"] = {
        key: config[key]
        for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "max_grad_norm")
    }
    config["optimization"]["temperature"] = temperature
    config["algorithm"] = {
        "se_shrinkage": se_shrinkage,
        "queue_size": 0,
        "learnable_temperature": False,
        "uncertainty_weighted": False,
        "loss_anchor_weight": loss_anchor_weight,
        "loss_align_weight": loss_align_weight,
    }
    code_root = Path(__file__).resolve().parents[1]
    code_files = {
        str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(code_root.rglob("*.py"))
    }
    config["code_sha256"] = hashlib.sha256(
        json.dumps(code_files, sort_keys=True).encode()
    ).hexdigest()
    config["runtime"] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
    }
    if images:
        from ..encoder import _hub_dir

        backbone_weights = _hub_dir() / "checkpoints/dinov2_vits14_pretrain.pth"
        if backbone_weights.is_file():
            config["backbone_weights_sha256"] = hashlib.sha256(
                backbone_weights.read_bytes()
            ).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "training_config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best, history, ckpt_path = float("inf"), [], out / "best_model.pt"
    for epoch in range(epochs):
        train, _, diagnostics = run_joint_epoch(
            model,
            loaders["train"],
            dev,
            loss_options,
            optimizer,
            max_grad_norm,
            gradient_diagnostics_interval,
        )
        if not train["steps"]:
            raise ValueError("No supervised optimizer updates in training split")
        val, _, _ = run_joint_epoch(model, loaders["validation"], dev, loss_options)
        if not val["supervision_active"]:
            raise ValueError("Validation has no eligible labels for the selected task(s)")
        row = {
            "epoch": epoch,
            "train_total": train["total"],
            "val_total": val["total"],
            "lr": optimizer.param_groups[0]["lr"],
            "train": train,
            "validation": val,
        }
        history.append(row)
        with (out / "gradient_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
            for diagnostic in diagnostics:
                handle.write(json.dumps({"epoch": epoch, **diagnostic}, allow_nan=False) + "\n")
        scheduler.step()
        if val["total"] < best:
            best = val["total"]
            contract = {
                "state_tabular_columns": list(train_ds.state_tabular_columns),
                "intervention_columns": list(train_ds.intervention_columns),
                "context_tabular_columns": list(train_ds.context_tabular_columns),
                "satellite_relative_years": list(train_ds.satellite_relative_years),
                "families": list(train_ds.families),
                "response_offsets": offsets,
                "response_dim": 18,
                "normalization": train_ds.normalization,
                "bundle_shas": config["bundle_shas"],
                "streetview_time_aware": images,
                "streetview_assets_sha256": train_ds.streetview_assets_sha256,
                "architecture_version": model.architecture_version,
                "dimensions": config["dimensions"],
            }
            checkpoint = dict(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                val_loss=best,
                is_stc=True,
                model_type="stc_joint",
                architecture_version=model.architecture_version,
                training_mode=mode,
                prediction_inputs=prediction_inputs,
                feature_dim=train_ds.feature_dim(),
                feature_columns=list(train_ds.feature_columns),
                embedding_dim=128,
                hidden_dims=(128,),
                dropout=dropout,
                dataset_id=config["dataset_id"],
                bundle_shas=config["bundle_shas"],
                use_image_encoder=images,
                use_satellite_encoder=satellite,
                max_images_per_grid=max_images_per_grid,
                stc_contract=contract,
                streetview_index_path=str(streetview_index_path) if streetview_index_path else None,
                satellite_root=str(satellite_root) if satellite_root else None,
                training_config=config,
            )
            torch.save(checkpoint, ckpt_path)
        print(
            f"{mode} epoch {epoch + 1}/{epochs}: train={train['total']:.6f} val={val['total']:.6f}"
        )
    checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test, test_pool, _ = run_joint_epoch(model, loaders["test"], dev, loss_options)
    _, train_pool, _ = run_joint_epoch(model, train_eval, dev, loss_options)
    _, val_pool, _ = run_joint_epoch(model, loaders["validation"], dev, loss_options)
    report = {
        "config": config,
        "validation": evaluate_joint_pool(
            val_pool,
            train_pool,
            offsets,
            k=eval_k,
            n_perm=eval_n_perm,
            n_boot=eval_n_boot,
            seed=seed,
            run_transfer=run_transfer,
            mode=mode,
        ),
        "test": evaluate_joint_pool(
            test_pool,
            train_pool,
            offsets,
            k=eval_k,
            n_perm=eval_n_perm,
            n_boot=eval_n_boot,
            seed=seed,
            run_transfer=run_transfer,
            mode=mode,
        ),
    }
    if run_baselines:
        from ..baselines import run_baselines as compute_baselines

        report["baselines"] = {
            "test": compute_baselines(
                test_pool["features"],
                test_pool["responses"],
                test_pool["masks"],
                fit_features=train_pool["features"],
                response_offsets=offsets,
                k=eval_k,
                n_perm=eval_n_perm,
                n_boot=eval_n_boot,
                seed=seed,
                image_batches=loaders["test"],
                device=dev,
            )
        }
    if legacy_options.get("visualize", False):
        from ..evaluation import latent_manifold_umap

        report["embedding_manifold"] = latent_manifold_umap(
            test_pool["embeddings"].numpy(),
            test_pool["orders"].tolist(),
            test_pool["city_keys"],
            n_neighbors=min(15, max(2, len(test_pool["orders"]) - 1)),
            random_state=seed,
        )
    write_json(out / "training_history.json", history)
    write_json(out / "test_metrics.json", test)
    write_json(out / "evaluation_report.json", report)
    _append_run_record(out, config, history, test, report)
    return ckpt_path
