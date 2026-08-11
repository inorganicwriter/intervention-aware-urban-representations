from __future__ import annotations

import json
from pathlib import Path

import pytest

import urban_intervention.representation.ablation_cli as run_ablation
import urban_intervention.representation.model_card_cli as build_model_card
from tests.helpers.representation import build_synthetic_model_inputs


def _trained_run(tmp_path: Path, name: str = "run") -> Path:
    ds_dir, _ = build_synthetic_model_inputs(tmp_path)
    from urban_intervention.representation.trainer import train_representation

    run_dir = tmp_path / name
    train_representation(
        model_inputs_dir=ds_dir,
        output_dir=run_dir,
        embedding_dim=16,
        hidden_dims=(32,),
        dropout=0.0,
        batch_size=8,
        epochs=2,
        use_images=False,
        seed=42,
        eval_n_perm=10,
        eval_n_boot=10,
    )
    return run_dir


class TestModelCard:
    def test_card_json_and_markdown_written(self, tmp_path: Path) -> None:
        run_dir = _trained_run(tmp_path)
        card = build_model_card.build_model_card(run_dir)
        assert set(card) >= {
            "dataset_id",
            "architecture",
            "optimization",
            "data_splits",
            "training_history",
            "test_metrics",
            "evaluation",
            "limitations",
        }
        assert (run_dir / "model_card.json").is_file()
        markdown = (run_dir / "model_card.md").read_text(encoding="utf-8")
        assert "# Model Card" in markdown
        assert "## Validation split" in markdown
        assert "## Limitations" in markdown

    def test_card_reflects_run_artifacts(self, tmp_path: Path) -> None:
        run_dir = _trained_run(tmp_path)
        card = build_model_card.build_model_card(run_dir)
        assert card["training_history"]["epochs_logged"] == 2
        assert card["training_history"]["best_epoch"] is not None
        validation = card["evaluation"]["validation"]
        assert validation["n_units"] >= 2
        assert validation["nn_corr@k"] is not None
        assert validation["permutation_p_value"] is not None
        assert "overall" in {row["family"] for row in validation["families"]}
        assert any("Test split contains only" in note for note in card["limitations"])
        assert any("non-production" in note for note in card["limitations"])

    def test_missing_files_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_model_card.build_model_card(tmp_path)


class TestRunAblation:
    def test_grid_runs_and_summarizes(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        specs = [
            {"name": "baseline", "overrides": {"epochs": 1, "batch_size": 8}},
            {
                "name": "no_prediction",
                "overrides": {"epochs": 1, "batch_size": 8, "pred_weight": 0.0},
            },
        ]
        summary_path = run_ablation.run_ablation(
            ds_dir,
            specs,
            tmp_path / "ablation",
            eval_n_perm=10,
            eval_n_boot=10,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["specs"] == 2
        assert summary["completed"] == 2
        names = {row["name"] for row in summary["rows"]}
        assert names == {"baseline", "no_prediction"}
        row = summary["rows"][0]
        assert row["error"] is None
        assert row["best_val_loss"] is not None
        assert row["validation_nn_corr@k"] is not None
        assert row["validation_permutation_p"] is not None
        for _split in ("validation", "test"):
            assert (tmp_path / "ablation" / "baseline" / "evaluation_report.json").is_file()
        markdown = (tmp_path / "ablation" / "ablation_summary.md").read_text(encoding="utf-8")
        assert "# Ablation Summary" in markdown
        assert "baseline" in markdown
        assert "validation_nn_corr@k" in markdown
        assert "| baseline |" in markdown

    def test_failed_spec_recorded_not_fatal(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        specs = [
            {"name": "broken", "overrides": {"epochs": 0}},
            {"name": "fine", "overrides": {"epochs": 1, "batch_size": 8}},
        ]
        summary_path = run_ablation.run_ablation(
            ds_dir,
            specs,
            tmp_path / "ablation2",
            eval_n_perm=10,
            eval_n_boot=10,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["completed"] == 1
        by_name = {row["name"]: row for row in summary["rows"]}
        assert by_name["broken"]["error"] is not None
        assert by_name["fine"]["error"] is None

    def test_invalid_overrides_rejected(self, tmp_path: Path) -> None:
        ds_dir, _ = build_synthetic_model_inputs(tmp_path)
        with pytest.raises(ValueError):
            run_ablation.run_ablation(
                ds_dir,
                [{"name": "bad", "overrides": "not-a-dict"}],
                tmp_path / "ablation3",
            )
