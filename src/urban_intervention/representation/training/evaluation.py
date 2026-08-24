"""Post-training statistical evaluation orchestration."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..baselines import appearance_autoencoder_baseline
from ..baselines import run_baselines as compute_baselines
from ..evaluation import (
    bootstrap_ci,
    cosine_similarity,
    permutation_test,
    probe_rmse,
    response_similarity_with_validity,
    retrieval_metrics,
)
from ..model import ResponseEmbeddingModel
from ..transfer import predictive_transfer_report, transfer_report
from .pools import _collect_pool
from .visualization import _visualize_embeddings


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
