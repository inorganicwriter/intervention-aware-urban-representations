from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urban_intervention.causal.gpu.contract_builder import build_python_contract
from urban_intervention.causal.gpu.io import (
    cv_contract_artifact_paths,
    load_cv_contract_manifest,
    load_estimation_panel,
    load_gsc_cv_folds,
    load_mc_cv_contract,
)
from urban_intervention.causal.gpu.runtime import RuntimeConfig, TorchRuntime


def _write_panel(directory, estimator: str):
    directory.mkdir()
    unit_column = "gsc_unit_id" if estimator == "gsc" else "mc_unit_id"
    rows = []
    for unit in range(1, 7):
        for time in range(1, 13):
            rows.append(
                {
                    unit_column: unit,
                    "time_id": time,
                    "period": 2000 + time,
                    "model_value": float(
                        2 + 0.1 * time + 0.2 * unit + np.sin(time / 2) * unit
                    ),
                    "D": int(unit == 2 and time >= 9),
                }
            )
    path = directory / "estimation_panel.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_python_gsc_contract_roundtrips_without_r(tmp_path) -> None:
    panel = _write_panel(tmp_path / "gsc", "gsc")
    r_reference = panel.with_name("gsc_cv_folds.parquet")
    r_reference.write_bytes(b"immutable-r-reference")
    manifest = build_python_contract(panel, "gsc")
    loaded = load_estimation_panel(panel, "gsc")
    (fold_path,) = cv_contract_artifact_paths(panel.parent, "gsc")
    folds = load_gsc_cv_folds(fold_path, loaded)
    metadata = load_cv_contract_manifest(panel.parent, "gsc")
    assert manifest.name == "gpu_contract_manifest.csv"
    assert metadata["contract_backend"] == "python_native"
    assert r_reference.read_bytes() == b"immutable-r-reference"
    assert len(folds) == 5
    assert all(fold.scored.any() for fold in folds)
    with pytest.raises(FileExistsError):
        build_python_contract(panel, "gsc")

    changed = pd.read_parquet(panel)
    changed.loc[0, "model_value"] += 1
    changed.to_parquet(panel, index=False)
    with pytest.raises(ValueError, match="panel hash"):
        load_gsc_cv_folds(fold_path, loaded)


def test_python_mc_contract_roundtrips_and_has_zero_lambda(tmp_path) -> None:
    panel = _write_panel(tmp_path / "mc", "mc")
    runtime = TorchRuntime(RuntimeConfig(device="cpu", seed=20260725))
    build_python_contract(panel, "mc", runtime=runtime)
    loaded = load_estimation_panel(panel, "mc")
    folds, lambdas = load_mc_cv_contract(panel.parent, loaded)
    metadata = load_cv_contract_manifest(panel.parent, "mc")
    assert metadata["contract_backend"] == "python_native"
    assert len(folds) == 20
    assert len(lambdas) == 20
    assert lambdas[-1] == 0.0
