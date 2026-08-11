from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation.baselines import (
    AppearanceAutoencoder,
    appearance_autoencoder_baseline,
)
from urban_intervention.representation.dataset import RESPONSE_DIM
from urban_intervention.representation.export import export_embeddings
from urban_intervention.representation.trainer import train_representation
from urban_intervention.representation.transfer import _rank_auc, predictive_auc


def _cluster_pool(
    n_per_cluster: int = 8, dims: int = 12, seed: int = 7
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, dims)
    p1 = np.sin(t).astype(np.float32)
    p2 = np.cos(t).astype(np.float32)
    features = torch.zeros(n_per_cluster * 2, dims)
    features[:n_per_cluster] = torch.from_numpy(p1)
    features[n_per_cluster:] = torch.from_numpy(p2)
    features += torch.from_numpy(rng.randn(n_per_cluster * 2, dims).astype(np.float32)) * 0.05
    responses = torch.cat(
        [
            torch.stack([torch.from_numpy(p1) for _ in range(n_per_cluster)]),
            torch.stack([torch.from_numpy(p2) for _ in range(n_per_cluster)]),
        ]
    )
    responses += torch.from_numpy(rng.randn(n_per_cluster * 2, dims).astype(np.float32)) * 0.02
    return features, responses


class TestAppearanceAutoencoder:
    def test_reconstructs_features(self) -> None:
        features, _ = _cluster_pool(n_per_cluster=8)
        model = AppearanceAutoencoder(feature_dim=12, embedding_dim=8)
        embedding, reconstruction = model(features)
        assert embedding.shape == (16, 8)
        assert reconstruction.shape == (16, 12)
        loss = torch.nn.functional.mse_loss(reconstruction, features)
        assert torch.isfinite(loss)
        assert loss.item() < 2.0

    def test_baseline_entry_structure_and_no_response_supervision(self) -> None:
        features, responses = _cluster_pool(n_per_cluster=12)
        masks = torch.ones_like(responses, dtype=torch.bool)
        entry = appearance_autoencoder_baseline(
            features, features, responses, masks, epochs=3, k=4, n_perm=20, n_boot=20
        )
        training = entry["training"]
        assert isinstance(training, dict)
        assert training["response_supervision"] is False
        assert entry["n_units"] == 24
        retrieval = entry["retrieval"]
        assert isinstance(retrieval, dict)
        assert "overall" in retrieval

    def test_captures_appearance_structure(self) -> None:
        features, responses = _cluster_pool(n_per_cluster=12)
        masks = torch.ones_like(responses, dtype=torch.bool)
        entry = appearance_autoencoder_baseline(
            features, features, responses, masks, epochs=20, k=5, n_perm=20, n_boot=20
        )
        retrieval = entry["retrieval"]
        assert isinstance(retrieval, dict)
        overall = retrieval["overall"]
        assert isinstance(overall, dict)
        nn_corr = overall["nn_corr@k"]
        assert isinstance(nn_corr, (int, float))
        assert nn_corr > 0.5

    def test_small_train_pool_returns_note(self) -> None:
        features, responses = _cluster_pool(n_per_cluster=2)
        masks = torch.ones_like(responses, dtype=torch.bool)
        entry = appearance_autoencoder_baseline(features, features, responses, masks, epochs=1)
        assert "note" in entry


class TestPredictiveAuc:
    def test_perfect_linear_signal_reaches_high_auc(self) -> None:
        rng = np.random.RandomState(3)
        n = 60
        x = torch.from_numpy(rng.randn(n, 6).astype(np.float32))
        signal = 2.0 * x[:, :1]
        y = signal.repeat(1, RESPONSE_DIM) + torch.from_numpy(
            rng.randn(n, RESPONSE_DIM).astype(np.float32)
        ) * 0.1
        masks = torch.ones(n, RESPONSE_DIM, dtype=torch.bool)
        result = predictive_auc(x, y, masks, x, y, masks, ridge=1e-6, min_obs=10)
        auc = result["auc_overall"]
        assert isinstance(auc, float)
        assert auc > 0.85

    def test_random_scores_give_chance_auc(self) -> None:
        rng = np.random.RandomState(4)
        n = 60
        x = torch.from_numpy(rng.randn(n, 6).astype(np.float32))
        y = torch.from_numpy(rng.randn(n, RESPONSE_DIM).astype(np.float32))
        masks = torch.ones(n, RESPONSE_DIM, dtype=torch.bool)
        result = predictive_auc(x, y, masks, x, y, masks, ridge=1.0, min_obs=10)
        auc = result["auc_overall"]
        assert isinstance(auc, float)
        assert 0.2 < auc < 0.8

    def test_rank_auc_known_values(self) -> None:
        scores = torch.tensor([0.1, 0.4, 0.9, 0.3, 0.6], dtype=torch.float32)
        labels = torch.tensor([0, 1, 1, 0, 0], dtype=torch.int8)
        auc = _rank_auc(scores, labels)
        assert auc is not None
        assert abs(auc - 5.0 / 6.0) < 1e-9
        assert _rank_auc(scores, torch.ones(5, dtype=torch.int8)) is None

    def test_rank_auc_uses_midranks_for_ties(self) -> None:
        scores = torch.zeros(4)
        labels = torch.tensor([1, 1, 0, 0], dtype=torch.int8)
        assert _rank_auc(scores, labels) == 0.5


class TestEmbeddingExport:
    def test_exports_parquet_with_identity_and_embeddings(self, tmp_path: Path) -> None:
        ds_dir, n_grids = build_synthetic_model_inputs(tmp_path)
        run_dir = tmp_path / "run"
        train_representation(
            model_inputs_dir=ds_dir,
            output_dir=run_dir,
            embedding_dim=16,
            hidden_dims=(32,),
            dropout=0.0,
            batch_size=8,
            epochs=1,
            use_images=False,
            seed=42,
            eval_n_perm=10,
            eval_n_boot=10,
        )
        out_path = tmp_path / "embeddings.parquet"
        export_embeddings(run_dir / "best_model.pt", ds_dir, out_path, batch_size=8)
        frame = pd.read_parquet(out_path)
        assert len(frame) == n_grids
        expected = {
            "treatment_order",
            "city_key",
            "split",
            "quality_grade",
            "final_training_mask",
        }
        assert expected <= set(frame.columns)
        embedding_columns = [c for c in frame.columns if c.startswith("emb_")]
        assert len(embedding_columns) == 16
        norm = np.linalg.norm(frame[embedding_columns].to_numpy(), axis=1)
        assert np.allclose(norm, 1.0, atol=1e-5)
        assert frame["treatment_order"].is_monotonic_increasing

        units_path = ds_dir / "unit_features.parquet"
        units = pd.read_parquet(units_path)
        original = next(column for column in units.columns if column.startswith("z__"))
        units.rename(columns={original: f"{original}_changed"}).to_parquet(
            units_path, index=False
        )
        with pytest.raises(ValueError, match="feature schema"):
            export_embeddings(
                run_dir / "best_model.pt",
                ds_dir,
                tmp_path / "invalid_embeddings.parquet",
                batch_size=8,
            )


class TestFullEvaluationReport:
    def test_report_includes_autoencoder_and_predictive_transfer(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path, n_grids=24, n_cities=8)
        run_dir = tmp_path / "run_full"
        train_representation(
            model_inputs_dir=ds_dir,
            output_dir=run_dir,
            embedding_dim=16,
            hidden_dims=(32,),
            dropout=0.0,
            batch_size=8,
            epochs=1,
            use_images=False,
            seed=42,
            eval_n_perm=10,
            eval_n_boot=10,
        )
        report = json.loads((run_dir / "evaluation_report.json").read_text(encoding="utf-8"))
        test_baselines = report["baselines"]["test"]
        assert isinstance(test_baselines, dict)
        autoencoder = test_baselines["appearance_autoencoder"]
        assert isinstance(autoencoder, dict)
        training = autoencoder["training"]
        assert isinstance(training, dict)
        assert training["response_supervision"] is False
        assert autoencoder["n_units"] >= 2
        test_entry = report["test"]
        predictive = test_entry["predictive_transfer"]
        assert isinstance(predictive, dict)
        assert "embeddings" in predictive and "raw_features" in predictive
        assert "transfer" in test_entry
        assert "few_shot_probe" in test_entry["transfer"]
        runs_rows = (run_dir / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(runs_rows) == 1
        record = json.loads(runs_rows[0])
        assert "test_nn_corr@k" in record
        assert "appearance_autoencoder" in record["baseline_nn_corr@k"]
