from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from urban_intervention.causal.gpu.gsc_reference import (
    CORRECTION_MARKER,
    qualification_gsc_references,
    repair_gsc_reference,
)


def _reference_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    reference = tmp_path / "staging" / "xu_gsc" / "city" / "2015" / "outcome"
    reference.mkdir(parents=True)
    panel = reference / "estimation_panel.parquet"
    labels = reference / "causal_response_labels.parquet"
    pd.DataFrame(
        {
            "gsc_unit_id": [1, 1, 2, 2],
            "time_id": [1, 2, 1, 2],
            "model_value": [1.0, 3.0, 5.0, 7.0],
            "D": [0, 1, 0, 0],
        }
    ).to_parquet(panel, index=False)
    pd.DataFrame(
        {
            "event_time": [-1, 1],
            "causal_response_label": [0.0, 2.0],
            "standard_error": [2.0, 4.0],
            "confidence_lower": [-3.92, -5.84],
            "confidence_upper": [3.92, 9.84],
            "p_value": [1.0, 0.62],
            "uncertainty_source": ["gsynth_parametric_bootstrap"] * 2,
        }
    ).to_parquet(labels, index=False)
    pd.DataFrame(
        {
            "field": ["schema", "production_eligible"],
            "value": ["complete_published_estimators", "TRUE"],
        }
    ).to_csv(reference / "manifest.csv", index=False, encoding="utf-8-sig")

    qualification = tmp_path / "qualification"
    output = qualification / "gsc" / "city" / "2015" / "outcome"
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "estimator": "gsc",
                "panel": str(panel.resolve()),
                "reference_labels": str(labels.resolve()),
            }
        ),
        encoding="utf-8",
    )
    return qualification, panel, labels


def test_repair_gsc_reference_is_audited_and_idempotent(tmp_path: Path) -> None:
    qualification, panel, labels = _reference_fixture(tmp_path)
    references = qualification_gsc_references(qualification)
    assert references == [(panel.resolve(), labels.resolve())]

    dry_run = repair_gsc_reference(panel, labels, apply=False)
    assert dry_run.status == "would_repair"
    assert dry_run.corrected_rows == 2
    original = pd.read_parquet(labels)
    assert original["standard_error"].tolist() == [2.0, 4.0]

    applied = repair_gsc_reference(panel, labels, apply=True)
    assert applied.status == "repaired"
    expected_scale = np.std([1.0, 3.0, 5.0, 7.0], ddof=1)
    assert np.isclose(applied.normalization_scale, expected_scale)
    corrected = pd.read_parquet(labels)
    assert np.allclose(
        corrected["standard_error"], np.array([2.0, 4.0]) / expected_scale
    )
    assert corrected["uncertainty_source"].str.endswith(CORRECTION_MARKER).all()
    expected_lower = (
        corrected["causal_response_label"]
        - 1.959963984540054 * corrected["standard_error"]
    )
    assert np.allclose(corrected["confidence_lower"], expected_lower)

    second = repair_gsc_reference(panel, labels, apply=True)
    assert second.status == "already_corrected"
    assert second.corrected_rows == 0
    unchanged = pd.read_parquet(labels)
    pd.testing.assert_frame_equal(corrected, unchanged)
    manifest = pd.read_csv(labels.with_name("manifest.csv"), dtype="string")
    values = manifest.set_index("field")["value"]
    assert values["gsc_parametric_se_normalization_corrected"] == "TRUE"
    assert values["gsc_parametric_se_normalization_method"] == (
        "divide_once_by_fect_normalization_scale"
    )
    assert values["gsc_reference_labels_sha256_before_correction"] != values[
        "gsc_reference_labels_sha256_after_correction"
    ]
