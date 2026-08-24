"""Adapters for estimation panels and reference-label shadow comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import GPU_IMPLEMENTATION_VERSION, PanelData
from .gsc import GSCFold
from .matrix_completion import RollingFold
from .provenance import file_sha256

_CV_CONTRACTS: dict[str, dict[str, str | int | float]] = {
    "gsc": {
        "cv_method": "rolling",
        "cv_folds": 5,
        "cv_nobs": 3,
        "cv_buffer": 1,
        "cv_prop": 0.1,
        "cv_rule": "1se",
        "tol": 1e-5,
        "max_iteration": 5000,
    },
    "mc": {
        "cv_method": "rolling",
        "cv_folds": 20,
        "cv_nobs": 1,
        "cv_buffer": 0,
        "cv_prop": 0.1,
        "cv_rule": "1se",
        "tol": 1e-5,
        "max_iteration": 5000,
    },
}


@dataclass(frozen=True, slots=True)
class LoadedPanel:
    panel: PanelData
    periods: tuple[object, ...]
    numeric_unit_ids: tuple[int, ...]
    numeric_time_ids: tuple[int, ...]


def cv_contract_manifest_path(directory: str | Path) -> Path:
    """Prefer a Python-native contract manifest without overwriting R artifacts."""
    directory = Path(directory)
    native = directory / "gpu_contract_manifest.csv"
    return native if native.is_file() else directory / "manifest.csv"


def cv_contract_artifact_paths(directory: str | Path, estimator: str) -> tuple[Path, ...]:
    """Resolve coexisting R-reference or Python-native contract artifacts."""
    directory = Path(directory)
    native = (directory / "gpu_contract_manifest.csv").is_file()
    suffix = ".python" if native else ""
    if estimator == "gsc":
        return (directory / f"gsc_cv_folds{suffix}.parquet",)
    if estimator == "mc":
        return (
            directory / f"mc_cv_folds{suffix}.parquet",
            directory / f"mc_lambda_grid{suffix}.csv",
        )
    raise ValueError("estimator must be 'gsc' or 'mc'")


def load_cv_contract_manifest(directory: str | Path, estimator: str) -> dict[str, str]:
    """Fail closed when an R CV artifact does not match the frozen GPU contract."""
    if estimator not in _CV_CONTRACTS:
        raise ValueError("estimator must be 'gsc' or 'mc'")
    path = cv_contract_manifest_path(directory)
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if list(frame.columns) != ["field", "value"] or frame["field"].duplicated().any():
        raise ValueError("GPU input manifest must contain unique field/value rows")
    values = dict(zip(frame["field"], frame["value"], strict=True))
    schema = values.get("schema")
    if schema not in {"causal_gpu_input_v1", "causal_gpu_input_v2_python_contract"}:
        raise ValueError("unsupported causal GPU input schema")
    if values.get("production_eligible", "").upper() != "FALSE":
        raise ValueError("GPU input contract must not claim production eligibility")
    if schema == "causal_gpu_input_v1":
        if values.get("run_mode") != "gpu_export":
            raise ValueError("R causal GPU input was not produced in gpu_export mode")
        if values.get("fect_version") != "2.4.5":
            raise ValueError("GPU input contract requires the audited fect version 2.4.5")
        values["contract_backend"] = "r_fect_2.4.5"
    else:
        if values.get("run_mode") != "python_contract":
            raise ValueError("Python causal GPU input was not produced in python_contract mode")
        if values.get("contract_backend") != "python_native":
            raise ValueError("Python causal GPU contract backend must be python_native")
        if values.get("implementation_version") != GPU_IMPLEMENTATION_VERSION:
            raise ValueError("Python causal GPU contract implementation is stale")
        panel_path = Path(values.get("panel_path", "")).resolve()
        expected_panel = Path(directory).resolve() / "estimation_panel.parquet"
        if panel_path != expected_panel or not panel_path.is_file():
            raise ValueError("Python causal GPU contract references another estimation panel")
        if values.get("panel_sha256") != file_sha256(panel_path):
            raise ValueError("Python causal GPU contract panel hash does not match")
    seed = pd.to_numeric(pd.Series([values.get("cv_seed")]), errors="coerce").iloc[0]
    if not np.isfinite(seed):
        raise ValueError("GPU input manifest lacks a finite CV seed")
    for field, expected in _CV_CONTRACTS[estimator].items():
        actual = values.get(field)
        if isinstance(expected, str):
            valid = actual == expected
        else:
            numeric = pd.to_numeric(pd.Series([actual]), errors="coerce").iloc[0]
            valid = bool(np.isfinite(numeric) and np.isclose(numeric, expected))
        if not valid:
            raise ValueError(
                f"{estimator.upper()} GPU contract field {field} must be {expected!r}; "
                f"received {actual!r}"
            )
    return values


def load_estimation_panel(path: str | Path | pd.DataFrame, estimator: str) -> LoadedPanel:
    """Load a formal panel artifact or an in-memory Python-built panel.

    The reference MC implementation replaces an unavailable treated-post
    outcome with the treated unit's pre-period mean before fitting.  Those
    cells are outside every fit mask, so reproducing that placeholder here is
    safe and permits counterfactuals for horizons whose observed outcome is
    unavailable.  The original ``value`` column remains untouched for label
    availability downstream.
    """
    if estimator not in {"gsc", "mc"}:
        raise ValueError("estimator must be 'gsc' or 'mc'")
    frame = path.copy() if isinstance(path, pd.DataFrame) else pd.read_parquet(path)
    unit_column = "gsc_unit_id" if estimator == "gsc" else "mc_unit_id"
    required = {unit_column, "time_id", "D", "model_value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"estimation panel lacks columns: {sorted(missing)}")
    if frame.duplicated(["time_id", unit_column]).any():
        raise ValueError("estimation panel contains duplicate time/unit cells")
    if frame[[unit_column, "time_id", "D"]].isna().any().any():
        raise ValueError("estimation panel identifiers and treatment flags cannot be missing")
    treatment_values = pd.to_numeric(frame["D"], errors="raise")
    if not treatment_values.isin({0, 1}).all():
        raise ValueError("estimation panel treatment flags must be binary")
    frame["D"] = treatment_values.astype(np.int8)
    frame["model_value"] = pd.to_numeric(frame["model_value"], errors="coerce")
    if estimator == "mc" and frame.loc[frame["D"].eq(1), "model_value"].isna().any():
        treated_units = frame.loc[frame["D"].eq(1), unit_column].unique()
        if len(treated_units) != 1:
            raise ValueError("MC panel must contain exactly one treated unit")
        target = frame[unit_column].eq(treated_units[0])
        pre = frame.loc[target & frame["D"].eq(0), "model_value"]
        pre_mean = float(pre.mean())
        if not np.isfinite(pre_mean):
            raise ValueError("MC treated pre-period fill value is unavailable")
        missing_treated = target & frame["D"].eq(1) & frame["model_value"].isna()
        frame.loc[missing_treated, "model_value"] = pre_mean
    for column in (unit_column, "time_id"):
        numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"estimation panel {column} values must be finite integers")
        frame[column] = numeric.astype(np.int64)
    if "period" in frame.columns and frame.groupby("time_id")["period"].nunique().gt(1).any():
        raise ValueError("estimation panel period labels must map one-to-one to time_id")
    times = np.sort(frame["time_id"].unique())
    units = np.sort(frame[unit_column].unique())
    expected = len(times) * len(units)
    if len(frame) != expected:
        raise ValueError(
            f"estimation panel is not rectangular: {len(frame)} rows, expected {expected}"
        )
    values = frame.pivot(index="time_id", columns=unit_column, values="model_value")
    treatment = frame.pivot(index="time_id", columns=unit_column, values="D")
    values = values.reindex(index=times, columns=units)
    treatment = treatment.reindex(index=times, columns=units)
    if treatment.isna().any().any():
        raise ValueError("treatment mask contains missing cells")
    period_lookup = (
        frame[["time_id", "period"]]
        .drop_duplicates()
        .set_index("time_id")["period"]
        .reindex(times)
    if "period" in frame.columns
        else pd.Series(times, index=times)
    )
    if period_lookup.index.duplicated().any() or period_lookup.isna().any():
        raise ValueError("estimation panel period labels must map one-to-one to time_id")
    panel = PanelData(
        y=values.to_numpy(dtype=np.float64),
        observed=np.isfinite(values.to_numpy(dtype=np.float64)),
        treated=treatment.to_numpy(dtype=bool),
        unit_ids=tuple(str(value) for value in units),
        time_ids=tuple(period_lookup.tolist()),
    )
    return LoadedPanel(
        panel=panel,
        periods=tuple(period_lookup.tolist()),
        numeric_unit_ids=tuple(int(value) for value in units),
        numeric_time_ids=tuple(int(value) for value in times),
    )


def load_gsc_cv_folds(path: str | Path, loaded: LoadedPanel) -> list[GSCFold]:
    """Load and validate a rolling-CV mask contract exported by fect/R."""
    metadata = load_cv_contract_manifest(Path(path).parent, "gsc")
    frame = pd.read_parquet(path)
    required = {"fold_id", "gsc_unit_id", "time_id", "scored"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"GSC CV contract lacks columns: {sorted(missing)}")
    if frame.duplicated(["fold_id", "gsc_unit_id", "time_id"]).any():
        raise ValueError("GSC CV contract contains duplicate masked cells")
    if not pd.api.types.is_bool_dtype(frame["scored"]):
        raise ValueError("GSC CV scored flags must be boolean")
    target_index = loaded.panel.single_treated_unit()
    control_ids = [
        unit_id
        for index, unit_id in enumerate(loaded.numeric_unit_ids)
        if index != target_index
    ]
    control_lookup = {unit_id: index for index, unit_id in enumerate(control_ids)}
    time_lookup = {
        int(time_id): index for index, time_id in enumerate(loaded.numeric_time_ids)
    }
    result: list[GSCFold] = []
    fold_ids = sorted(pd.to_numeric(frame["fold_id"], errors="raise").astype(int).unique())
    if fold_ids != list(range(1, len(fold_ids) + 1)):
        raise ValueError("GSC CV fold ids must be consecutive and one-based")
    if len(fold_ids) != int(metadata["cv_folds"]):
        raise ValueError("GSC CV artifact fold count disagrees with its manifest")
    shape = (loaded.panel.y.shape[0], len(control_ids))
    for fold_id in fold_ids:
        removed = np.zeros(shape, dtype=bool)
        scored = np.zeros(shape, dtype=bool)
        rows = frame[frame["fold_id"] == fold_id]
        for row in rows.itertuples(index=False):
            unit_id = int(row.gsc_unit_id)
            time_id = int(row.time_id)
            if unit_id not in control_lookup or time_id not in time_lookup:
                raise ValueError("GSC CV contract references a cell outside the panel")
            coordinate = (time_lookup[time_id], control_lookup[unit_id])
            removed[coordinate] = True
            scored[coordinate] = bool(row.scored)
        result.append(GSCFold(removed=removed, scored=scored))
    return result


def load_mc_cv_contract(
    directory: str | Path,
    loaded: LoadedPanel,
) -> tuple[list[RollingFold], tuple[float, ...]]:
    """Load fect/R rolling folds and the exact matrix-completion lambda grid."""
    directory = Path(directory)
    metadata = load_cv_contract_manifest(directory, "mc")
    folds_path, lambda_path = cv_contract_artifact_paths(directory, "mc")
    folds_frame = pd.read_parquet(folds_path)
    required = {"fold_id", "mc_unit_id", "time_id", "scored"}
    missing = required - set(folds_frame.columns)
    if missing:
        raise ValueError(f"MC CV contract lacks columns: {sorted(missing)}")
    if folds_frame.duplicated(["fold_id", "mc_unit_id", "time_id"]).any():
        raise ValueError("MC CV contract contains duplicate masked cells")
    if not pd.api.types.is_bool_dtype(folds_frame["scored"]):
        raise ValueError("MC CV scored flags must be boolean")
    unit_lookup = {
        int(unit_id): index for index, unit_id in enumerate(loaded.numeric_unit_ids)
    }
    time_lookup = {
        int(time_id): index for index, time_id in enumerate(loaded.numeric_time_ids)
    }
    fold_ids = sorted(
        pd.to_numeric(folds_frame["fold_id"], errors="raise").astype(int).unique()
    )
    if fold_ids != list(range(1, len(fold_ids) + 1)):
        raise ValueError("MC CV fold ids must be consecutive and one-based")
    if len(fold_ids) != int(metadata["cv_folds"]):
        raise ValueError("MC CV artifact fold count disagrees with its manifest")
    shape = loaded.panel.y.shape
    available = loaded.panel.untreated_observed
    folds: list[RollingFold] = []
    for fold_id in fold_ids:
        removed = np.zeros(shape, dtype=bool)
        scored = np.zeros(shape, dtype=bool)
        rows = folds_frame[folds_frame["fold_id"] == fold_id]
        for row in rows.itertuples(index=False):
            unit_id = int(row.mc_unit_id)
            time_id = int(row.time_id)
            if unit_id not in unit_lookup or time_id not in time_lookup:
                raise ValueError("MC CV contract references a cell outside the panel")
            coordinate = (time_lookup[time_id], unit_lookup[unit_id])
            removed[coordinate] = True
            scored[coordinate] = bool(row.scored)
        if np.any(removed & ~available):
            raise ValueError("MC CV contract removes an unavailable panel cell")
        folds.append(RollingFold(training=available & ~removed, score=scored))
    lambda_frame = pd.read_csv(lambda_path, encoding="utf-8-sig")
    if set(lambda_frame.columns) != {"sequence", "lambda"}:
        raise ValueError("MC lambda contract must contain sequence and lambda columns")
    lambda_frame = lambda_frame.sort_values("sequence")
    expected_sequence = np.arange(1, len(lambda_frame) + 1)
    if not np.array_equal(lambda_frame["sequence"].to_numpy(dtype=int), expected_sequence):
        raise ValueError("MC lambda sequence must be consecutive and one-based")
    lambdas = tuple(lambda_frame["lambda"].to_numpy(dtype=np.float64))
    if not lambdas or not np.isfinite(lambdas).all() or min(lambdas) < 0 or any(
        left <= right for left, right in zip(lambdas[:-1], lambdas[1:], strict=True)
    ):
        raise ValueError("MC lambdas must be strictly decreasing and non-negative")
    if len(lambdas) != 20:
        raise ValueError("MC lambda artifact must contain the frozen 20 candidates")
    return folds, lambdas


def compare_counterfactuals(
    periods: tuple[object, ...],
    accelerated: np.ndarray,
    reference_labels: str | Path,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, float | bool | int]]:
    """Align a shadow path to R labels and calculate an explicit parity gate."""
    reference = pd.read_parquet(reference_labels)
    if "period" not in reference or "counterfactual" not in reference:
        raise ValueError("reference labels need period and counterfactual columns")
    shadow = pd.DataFrame({"period_key": [str(value) for value in periods]})
    shadow["gpu_counterfactual"] = np.asarray(accelerated, dtype=np.float64)
    selected = reference.copy()
    selected["period_key"] = selected["period"].astype(str)
    selected = selected[["period_key", "counterfactual"]].rename(
        columns={"counterfactual": "r_counterfactual"}
    )
    comparison = shadow.merge(selected, on="period_key", how="left", validate="one_to_one")
    if comparison["r_counterfactual"].isna().any():
        raise ValueError("reference labels do not cover every panel period")
    difference = comparison["gpu_counterfactual"] - comparison["r_counterfactual"]
    comparison["difference"] = difference
    scale = float(np.max(np.abs(comparison["r_counterfactual"])))
    threshold = absolute_tolerance + relative_tolerance * scale
    metrics: dict[str, float | bool | int] = {
        "periods": len(comparison),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "tolerance": threshold,
        "passed": bool(np.max(np.abs(difference)) <= threshold),
    }
    return comparison, metrics


def compare_inference_paths(
    periods: tuple[object, ...],
    accelerated_effect: np.ndarray,
    accelerated_standard_error: np.ndarray,
    reference_labels: str | Path,
    *,
    relative_rmse_tolerance: float,
    minimum_ci_zero_agreement: float = 0.9,
) -> tuple[pd.DataFrame, dict[str, float | bool | int]]:
    """Compare formal SE paths and 95% zero-coverage decisions to R labels."""
    if relative_rmse_tolerance <= 0 or not 0 < minimum_ci_zero_agreement <= 1:
        raise ValueError("invalid inference parity tolerances")
    reference = pd.read_parquet(reference_labels)
    required = {"period", "causal_response_label", "standard_error"}
    if not required.issubset(reference.columns):
        raise ValueError("reference labels lack effect/standard-error inference columns")
    selected = reference.copy()
    if "event_time" in selected.columns:
        selected = selected.loc[
            pd.to_numeric(selected["event_time"], errors="coerce").gt(0)
        ].copy()
    selected["period_key"] = selected["period"].astype(str)
    selected = selected[
        ["period_key", "causal_response_label", "standard_error"]
    ].rename(
        columns={
            "causal_response_label": "r_effect",
            "standard_error": "r_standard_error",
        }
    )
    shadow = pd.DataFrame(
        {
            "period_key": [str(value) for value in periods],
            "gpu_effect": np.asarray(accelerated_effect, dtype=np.float64),
            "gpu_standard_error": np.asarray(
                accelerated_standard_error, dtype=np.float64
            ),
        }
    )
    comparison = selected.merge(shadow, on="period_key", how="left", validate="one_to_one")
    finite = (
        np.isfinite(comparison["r_effect"])
        & np.isfinite(comparison["r_standard_error"])
        & comparison["r_standard_error"].ge(0)
        & np.isfinite(comparison["gpu_effect"])
        & np.isfinite(comparison["gpu_standard_error"])
        & comparison["gpu_standard_error"].ge(0)
    )
    comparison["inference_comparable"] = finite
    comparable = comparison.loc[finite].copy()
    if len(comparable) < 2:
        metrics: dict[str, float | bool | int] = {
            "periods": len(comparison),
            "comparable_periods": len(comparable),
            "standard_error_relative_rmse": float("nan"),
            "relative_rmse_tolerance": relative_rmse_tolerance,
            "ci_zero_agreement": float("nan"),
            "minimum_ci_zero_agreement": minimum_ci_zero_agreement,
            "passed": False,
        }
        return comparison, metrics
    difference = comparable["gpu_standard_error"] - comparable["r_standard_error"]
    denominator = float(
        np.sqrt(np.mean(np.square(comparable["r_standard_error"])))
    )
    relative_rmse = (
        float(np.sqrt(np.mean(np.square(difference))) / denominator)
        if denominator > np.sqrt(np.finfo(float).eps)
        else float("inf")
    )
    r_covers_zero = (
        comparable["r_effect"].abs() <= 1.959963984540054 * comparable["r_standard_error"]
    )
    gpu_covers_zero = (
        comparable["gpu_effect"].abs()
        <= 1.959963984540054 * comparable["gpu_standard_error"]
    )
    agreement = float((r_covers_zero == gpu_covers_zero).mean())
    comparison.loc[finite, "standard_error_difference"] = difference.to_numpy()
    comparison.loc[finite, "r_ci_covers_zero"] = r_covers_zero.to_numpy()
    comparison.loc[finite, "gpu_ci_covers_zero"] = gpu_covers_zero.to_numpy()
    metrics = {
        "periods": len(comparison),
        "comparable_periods": len(comparable),
        "standard_error_relative_rmse": relative_rmse,
        "relative_rmse_tolerance": relative_rmse_tolerance,
        "ci_zero_agreement": agreement,
        "minimum_ci_zero_agreement": minimum_ci_zero_agreement,
        "passed": bool(
            relative_rmse <= relative_rmse_tolerance
            and agreement >= minimum_ci_zero_agreement
        ),
    }
    return comparison, metrics
