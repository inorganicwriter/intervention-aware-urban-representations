from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation.dataset import RESPONSE_DIM
from urban_intervention.representation.export import export_embeddings
from urban_intervention.representation.model import ResponseEmbeddingModel
from urban_intervention.representation.summarize import collect_runs, render_markdown
from urban_intervention.representation.trainer import train_representation
from urban_intervention.representation.transfer import few_shot_probe


class TestModelOptions:
    def test_conditioning_injects_token_embedding(self) -> None:
        model = ResponseEmbeddingModel(
            feature_dim=12, embedding_dim=16, use_image_encoder=False, conditioning="opening_year"
        )
        features = torch.randn(4, 12)
        tokens = torch.tensor([0, 10, 20, 29], dtype=torch.long)
        embedding, _ = model(features, conditioning_tokens=tokens)
        assert embedding.shape == (4, 16)
        assert embedding.norm(p=2, dim=1).sub(1.0).abs().max() < 1e-4
        plain, _ = model(features, conditioning_tokens=tokens)
        other_tokens = torch.tensor([1, 11, 21, 28], dtype=torch.long)
        other, _ = model(features, conditioning_tokens=other_tokens)
        assert not torch.allclose(plain, other)

    def test_conditioning_model_requires_tokens(self) -> None:
        model = ResponseEmbeddingModel(
            feature_dim=12, embedding_dim=16, use_image_encoder=False, conditioning="opening_year"
        )
        with pytest.raises(ValueError):
            model(torch.randn(2, 12))

    def test_meanmax_pooling_forward_without_images(self) -> None:
        model = ResponseEmbeddingModel(
            feature_dim=12, embedding_dim=16, use_image_encoder=False, image_pooling="meanmax"
        )
        embedding, _ = model(torch.randn(3, 12))
        assert embedding.shape == (3, 16)

    def test_all_masked_grid_rows_never_nan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class DummyBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.randn(x.shape[0], 384)

        monkeypatch.setattr(
            "urban_intervention.representation.encoder._ensure_dinov2",
            lambda *args, **kwargs: DummyBackbone(),
        )
        for pooling in ("max", "mean", "meanmax"):
            model = ResponseEmbeddingModel(
                feature_dim=12,
                embedding_dim=16,
                use_image_encoder=True,
                image_pooling=pooling,
            )
            model.eval()
            images = torch.zeros(3, 2, 3, 224, 224)
            image_mask = torch.tensor([[True, True], [False, False], [True, False]])
            embedding, _ = model(torch.randn(3, 12), images, image_mask)
            assert embedding.isfinite().all(), f"pooling={pooling} produced NaN"
            assert embedding.shape == (3, 16)

    def test_invalid_options_rejected(self) -> None:
        with pytest.raises(ValueError):
            ResponseEmbeddingModel(feature_dim=8, image_pooling="attention")
        with pytest.raises(ValueError):
            ResponseEmbeddingModel(feature_dim=8, conditioning="city")

    def test_conditioning_with_images_forward(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class DummyBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.randn(x.shape[0], 384)

        monkeypatch.setattr(
            "urban_intervention.representation.encoder._ensure_dinov2",
            lambda *args, **kwargs: DummyBackbone(),
        )
        model = ResponseEmbeddingModel(
            feature_dim=12,
            embedding_dim=16,
            use_image_encoder=True,
            image_pooling="meanmax",
            conditioning="opening_year",
        )
        images = torch.randn(2, 2, 3, 224, 224)
        image_mask = torch.tensor([[True, False], [True, True]])
        tokens = torch.tensor([5, 15], dtype=torch.long)
        model.eval()
        embedding, _ = model(torch.randn(2, 12), images, image_mask, conditioning_tokens=tokens)
        assert embedding.shape == (2, 16)
        assert embedding.isfinite().all()
        embedding_single, _ = model(
            torch.randn(1, 12), images[:1], image_mask[:1], conditioning_tokens=tokens[:1]
        )
        assert embedding_single.shape == (1, 16)

    def test_frozen_backbone_is_not_serialized_and_stays_in_eval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class DummyBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.new_zeros((x.shape[0], 384))

        monkeypatch.setattr(
            "urban_intervention.representation.encoder._ensure_dinov2",
            lambda *args, **kwargs: DummyBackbone(),
        )
        model = ResponseEmbeddingModel(
            feature_dim=12,
            embedding_dim=16,
            use_image_encoder=True,
        )
        model.train()
        model(
            torch.randn(2, 12),
            torch.randn(2, 1, 3, 224, 224),
            torch.ones(2, 1, dtype=torch.bool),
        )
        assert model.image_encoder is not None
        assert model.image_encoder.backbone.training is False
        state = model.state_dict()
        assert not any(key.startswith("image_encoder._backbone.") for key in state)
        reloaded = ResponseEmbeddingModel(
            feature_dim=12,
            embedding_dim=16,
            use_image_encoder=True,
        )
        reloaded.load_state_dict(state)

    def test_image_checkpoint_exports_with_inferred_architecture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class DummyBackbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.new_zeros((x.shape[0], 384))

        monkeypatch.setattr(
            "urban_intervention.representation.encoder._ensure_dinov2",
            lambda *args, **kwargs: DummyBackbone(),
        )
        ds_dir, n_grids = build_synthetic_model_inputs(tmp_path)
        run_dir = tmp_path / "image_run"
        checkpoint_path = train_representation(
            ds_dir,
            run_dir,
            embedding_dim=8,
            hidden_dims=(16,),
            dropout=0.0,
            batch_size=4,
            epochs=1,
            use_images=True,
            max_images_per_grid=1,
            run_baselines=False,
            run_transfer=False,
            eval_n_perm=2,
            eval_n_boot=2,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["use_image_encoder"] is True
        assert not any(
            key.startswith("image_encoder._backbone.")
            for key in checkpoint["model_state_dict"]
        )
        output_path = tmp_path / "image_embeddings.parquet"
        export_embeddings(checkpoint_path, ds_dir, output_path, batch_size=4)
        assert len(pd.read_parquet(output_path)) == n_grids


class TestFewShotProbeSeeds:
    def test_multi_seed_reports_mean_std(self) -> None:
        rng = np.random.RandomState(9)
        n = 40
        signal = torch.from_numpy(rng.randn(n, 4).astype(np.float32))
        embeddings = torch.cat(
            [signal, torch.from_numpy(rng.randn(n, 4).astype(np.float32) * 0.05)], dim=1
        )
        embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
        responses = torch.cat(
            [signal, torch.from_numpy(rng.randn(n, RESPONSE_DIM - 4).astype(np.float32) * 0.05)],
            dim=1,
        )
        masks = torch.ones(n, RESPONSE_DIM, dtype=torch.bool)
        curve = few_shot_probe(
            embeddings, responses, masks, shot_sizes=(8,), ridge=1e-3, min_obs=2, n_seeds=5
        )
        assert len(curve) == 1
        step = curve[0]
        rmse = step["rmse_overall"]
        std = step["rmse_overall_std"]
        assert isinstance(rmse, float)
        assert isinstance(std, float)
        assert step["seeds"] == 5
        assert std >= 0.0


class TestSummarizeRuns:
    def _make_dataset(self, tmp_path: Path) -> Path:
        return build_synthetic_model_inputs(tmp_path, n_grids=24, n_cities=8)[0]

    def _make_run(self, ds_dir: Path, run_dir: Path) -> Path:
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
        return run_dir

    def test_collect_and_render(self, tmp_path: Path) -> None:
        ds_dir = self._make_dataset(tmp_path)
        first = self._make_run(ds_dir, tmp_path / "main")
        second = self._make_run(ds_dir, tmp_path / "variant")
        rows = collect_runs([first, second])
        assert len(rows) == 2
        assert {row["run"] for row in rows} == {"main", "variant"}
        for row in rows:
            assert "test_nn_corr@k" in row
            assert "baseline_random_projection_nn_corr@k" in row
            assert "baseline_appearance_autoencoder_nn_corr@k" in row
        markdown = render_markdown(rows)
        assert "| run |" in markdown
        assert "| main |" in markdown
        assert "| variant |" in markdown

    def test_csv_roundtrip(self, tmp_path: Path) -> None:
        ds_dir = self._make_dataset(tmp_path)
        run_dir = self._make_run(ds_dir, tmp_path / "r1")
        rows = collect_runs([run_dir])
        out_csv = tmp_path / "summary.csv"
        with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with out_csv.open(encoding="utf-8-sig") as handle:
            frame = pd.read_csv(handle)
        assert len(frame) == 1
        assert "test_nn_corr@k" in frame.columns
        assert frame.loc[0, "run"] == "r1"

    def test_missing_runs_jsonl_reports_skip(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        rows = collect_runs([empty_dir])
        assert rows == []


class TestConditioningEndToEnd:
    def test_train_export_with_conditioning_and_meanmax(self, tmp_path: Path) -> None:
        ds_dir, n_grids = build_synthetic_model_inputs(tmp_path)
        run_dir = tmp_path / "run_cond"
        train_representation(
            model_inputs_dir=ds_dir,
            output_dir=run_dir,
            embedding_dim=16,
            hidden_dims=(32,),
            dropout=0.0,
            batch_size=8,
            epochs=1,
            use_images=False,
            conditioning="opening_year",
            image_pooling="meanmax",
            seed=42,
            eval_n_perm=10,
            eval_n_boot=10,
        )
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["conditioning"] == "opening_year"
        assert checkpoint["image_pooling"] == "meanmax"
        config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
        assert config["architecture"]["conditioning"] == "opening_year"
        assert config["architecture"]["image_pooling"] == "meanmax"
        report = json.loads((run_dir / "evaluation_report.json").read_text(encoding="utf-8"))
        assert "baselines" in report
        out_path = tmp_path / "emb_cond.parquet"
        export_embeddings(run_dir / "best_model.pt", ds_dir, out_path, batch_size=8)
        frame = pd.read_parquet(out_path)
        assert len(frame) == n_grids
        embedding_columns = [c for c in frame.columns if c.startswith("emb_")]
        assert len(embedding_columns) == 16
        assert frame[embedding_columns].to_numpy().dtype == "float64"
