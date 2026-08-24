"""Python-native, versioned CV contracts for GSC and matrix completion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from urban_intervention.utils import atomic_write_csv, atomic_write_parquet

from .contracts import GPU_IMPLEMENTATION_VERSION
from .gsc import GSCConfig
from .gsc import make_rolling_cv_folds as make_gsc_folds
from .io import load_estimation_panel
from .linalg import as_panel_tensors
from .matrix_completion import (
    MatrixCompletionConfig,
    make_lambda_grid,
)
from .matrix_completion import (
    make_rolling_cv_folds as make_mc_folds,
)
from .provenance import file_sha256
from .runtime import RuntimeConfig, TorchRuntime

Estimator = Literal["gsc", "mc"]
PYTHON_CONTRACT_SCHEMA = "causal_gpu_input_v2_python_contract"


def _gsc_fold_frame(panel_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    loaded = load_estimation_panel(panel_path, "gsc")
    panel = loaded.panel
    target = panel.single_treated_unit()
    controls = [index for index in range(panel.y.shape[1]) if index != target]
    config = GSCConfig(seed=20260723)
    folds = make_gsc_folds(
        panel.untreated_observed[:, controls],
        folds=config.folds,
        proportion=config.cv_prop,
        min_pre_periods=config.min_pre_periods,
        holdout_periods=config.cv_nobs,
        buffer_periods=config.cv_buffer,
        seed=config.seed,
    )
    rows: list[dict[str, Any]] = []
    control_ids = [loaded.numeric_unit_ids[index] for index in controls]
    for fold_id, fold in enumerate(folds, start=1):
        for time_position, control_position in np.argwhere(fold.removed):
            rows.append(
                {
                    "fold_id": fold_id,
                    "gsc_unit_id": control_ids[int(control_position)],
                    "time_id": loaded.numeric_time_ids[int(time_position)],
                    "scored": bool(fold.scored[time_position, control_position]),
                }
            )
    metadata = {
        "cv_folds": config.folds,
        "cv_nobs": config.cv_nobs,
        "cv_buffer": config.cv_buffer,
        "cv_prop": config.cv_prop,
        "cv_rule": config.cv_rule,
        "cv_seed": config.seed,
        "tol": config.tol,
        "max_iteration": config.max_iter,
    }
    return pd.DataFrame(rows), metadata


def _mc_contract(
    panel_path: Path,
    runtime: TorchRuntime,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    loaded = load_estimation_panel(panel_path, "mc")
    panel = loaded.panel
    config = MatrixCompletionConfig(seed=20260725)
    folds = make_mc_folds(
        panel.untreated_observed,
        np.asarray(panel.treated, dtype=bool),
        folds=config.folds,
        nobs=config.cv_nobs,
        buffer=config.buffer,
        proportion=config.cv_prop,
        min_pre_periods=config.min_pre_periods,
        seed=config.seed,
    )
    available = panel.untreated_observed
    rows: list[dict[str, Any]] = []
    for fold_id, fold in enumerate(folds, start=1):
        removed = available & ~fold.training
        for time_position, unit_position in np.argwhere(removed):
            rows.append(
                {
                    "fold_id": fold_id,
                    "mc_unit_id": loaded.numeric_unit_ids[int(unit_position)],
                    "time_id": loaded.numeric_time_ids[int(time_position)],
                    "scored": bool(fold.score[time_position, unit_position]),
                }
            )
    y, observed, treated = as_panel_tensors(panel, runtime)
    lambdas = make_lambda_grid(y, observed & ~treated, config)
    lambda_frame = pd.DataFrame(
        {"sequence": np.arange(1, len(lambdas) + 1), "lambda": lambdas}
    )
    metadata = {
        "cv_folds": config.folds,
        "cv_nobs": config.cv_nobs,
        "cv_buffer": config.buffer,
        "cv_prop": config.cv_prop,
        "cv_rule": config.cv_rule,
        "cv_seed": config.seed,
        "tol": config.tol,
        "max_iteration": config.max_iter,
    }
    return pd.DataFrame(rows), lambda_frame, metadata


def build_python_contract(
    panel_path: str | Path,
    estimator: Estimator,
    *,
    runtime: TorchRuntime | None = None,
    force: bool = False,
) -> Path:
    """Build one immutable Python-native tuning contract beside a staged panel."""
    panel_path = Path(panel_path).resolve()
    if not panel_path.is_file():
        raise FileNotFoundError(panel_path)
    directory = panel_path.parent
    manifest_path = directory / "gpu_contract_manifest.csv"
    outputs = [manifest_path]
    if estimator not in {"gsc", "mc"}:
        raise ValueError("estimator must be 'gsc' or 'mc'")
    # The native manifest is written last; resolve the explicit native names
    # directly so an existing R reference contract remains untouched.
    native_paths = (
        (directory / "gsc_cv_folds.python.parquet",)
        if estimator == "gsc"
        else (
            directory / "mc_cv_folds.python.parquet",
            directory / "mc_lambda_grid.python.csv",
        )
    )
    outputs.extend(native_paths)
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to replace an existing GPU contract without force=True: "
            + ", ".join(str(path) for path in existing)
        )

    if estimator == "gsc":
        folds, settings = _gsc_fold_frame(panel_path)
        atomic_write_parquet(folds, native_paths[0])
    elif estimator == "mc":
        runtime = runtime or TorchRuntime(RuntimeConfig(device="auto", seed=20260725))
        folds, lambdas, settings = _mc_contract(panel_path, runtime)
        atomic_write_parquet(folds, native_paths[0])
        atomic_write_csv(lambdas, native_paths[1])

    fields = {
        "schema": PYTHON_CONTRACT_SCHEMA,
        "method": f"{estimator.upper()} Python-native GPU contract",
        "run_mode": "python_contract",
        "contract_backend": "python_native",
        "implementation_version": GPU_IMPLEMENTATION_VERSION,
        "rng": "numpy.random.Generator(PCG64)",
        "estimator": estimator,
        "cv_method": "rolling",
        **settings,
        "production_eligible": "FALSE",
        "reference_rng_equivalent": "FALSE",
        "panel_path": str(panel_path),
        "panel_sha256": file_sha256(panel_path),
    }
    atomic_write_csv(
        pd.DataFrame({"field": fields.keys(), "value": fields.values()}),
        manifest_path,
    )
    return manifest_path
