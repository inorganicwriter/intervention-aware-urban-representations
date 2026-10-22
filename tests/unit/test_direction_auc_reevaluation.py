from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from tests.helpers.representation import build_synthetic_model_inputs
from urban_intervention.representation.dataset import RepresentationDataset
from urban_intervention.representation.model import ResponseEmbeddingModel

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "reevaluate_direction_auc.py"
)
SPEC = importlib.util.spec_from_file_location("auc_reevaluation_test", SCRIPT)
reevaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reevaluation)


def test_reevaluation_preserves_checkpoint_and_original_report(tmp_path: Path) -> None:
    dataset_dir, _ = build_synthetic_model_inputs(tmp_path)
    dataset = RepresentationDataset(dataset_dir)
    model = ResponseEmbeddingModel(
        feature_dim=dataset.feature_dim(),
        hidden_dims=(8,),
        embedding_dim=4,
        use_image_encoder=False,
    )
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": dataset.feature_dim(),
            "feature_columns": dataset.feature_columns,
            "embedding_dim": 4,
            "hidden_dims": (8,),
        },
        checkpoint_path,
    )
    original = tmp_path / "evaluation_report.json"
    original.write_text('{"old": true}\n', encoding="utf-8")
    (tmp_path / "training_config.json").write_text(
        '{"evaluation": {"probe_ridge": 0.01, "probe_min_obs": 2}}', encoding="utf-8"
    )
    before_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    report = reevaluation.reevaluate_direction_auc(checkpoint_path, dataset_dir, tmp_path)
    assert report["checkpoint_sha256"] == before_hash
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == before_hash
    assert original.read_text() == '{"old": true}\n'
    assert report["ridge"] == 0.01
    assert report["min_obs"] == 2
    assert report["retrained"] is False
    # The fixture excludes one unit per city via final_training_mask.
    assert report["eligible_split_units"] == {"train": 4, "validation": 2, "test": 2}
    assert report["test"]["raw_features"]["fitted_cells"] > 0
    assert np.isfinite(report["test"]["raw_features"]["auc_overall"])
    assert report["station_attributes"]["status"] == "not_used"
    saved = json.loads((tmp_path / "direction_auc_report.json").read_text())
    assert saved == report
