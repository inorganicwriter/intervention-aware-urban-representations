"""Publish per-task causal labels as one versioned training-label artifact."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow

from urban_intervention.causal.gpu.contracts import (
    CONTROL_DESIGN_PROVENANCE,
    CONTROL_DESIGN_SCHEMA,
    CONTROL_DESIGN_VIIRS_CACHE_CONTRACT,
    FORMAL_IMPLEMENTATION_VERSION,
)

OUTCOME_HORIZONS: dict[str, dict[str, tuple[int, ...]]] = {
    "housing": {"housing_log_price": (1, 3, 6, 12, 18, 24)},
    "viirs": {"viirs_avg_asinh": (1, 3, 6, 12, 18, 24)},
    "population": {"population_log": (1, 2, 3)},
    "poi": {
        "poi_count_log": (1, 2, 3),
        "poi_category_entropy": (1, 2, 3),
        "poi_commercial_share": (1, 2, 3),
        "poi_transport_access_log": (1, 2, 3),
    },
}
KEY_COLUMNS = [
    "treatment_order",
    "outcome_family",
    "outcome",
    "event_time",
    "specification_id",
]
TREATMENT_IDENTITY_COLUMNS = [
    "treatment_order",
    "city_key",
    "grid_id",
    "station_event_id",
    "opening_month",
]
TASK_IDENTITY_COLUMNS = [
    "treatment_order",
    "city_key",
    "grid_id",
    "opening_month",
    "outcome_family",
]
SUCCESS_STATUSES = {"matched_labelled", "gsc_labelled", "mc_labelled"}
TERMINAL_STATUSES = SUCCESS_STATUSES | {"skipped"}
LABEL_COLUMNS = [
    "observed",
    "counterfactual",
    "causal_response_label",
    "label_available",
    "transformed_scale",
    "method",
    "control_unit_key",
    "treated_baseline",
    "control_baseline",
    "treated_change",
    "control_change",
    "standard_error",
    "confidence_lower",
    "confidence_upper",
    "p_value",
    "bootstrap_repetitions",
    "valid_inference_repetitions",
    "uncertainty_source",
    "pre_observed_periods",
    "pre_rmspe",
    "pre_mean_effect",
    "pretrend_slope",
    "pretrend_slope_p_value",
    "pretrend_task_flag",
    "mc_lambda",
    "mc_regularized",
    "mc_cv_mspe",
    "selected_factors",
    "minimum_window_n",
    "effective_n_observed",
    "effective_n_counterfactual",
    "window_supported",
    "transaction_count",
    "transaction_count_min",
    "control_transaction_count",
    "transaction_count_threshold",
    "transaction_count_supported",
    "control_transaction_count_supported",
    "price_measure",
    "donor_scope",
    "estimator_backend",
    "implementation_version",
]
CONTROL_COLUMNS = [
    "schema",
    "implementation_version",
    "backend",
    "viirs_cache_contract",
    "status",
    "active_families",
    "selected_method",
    "donor_scope",
    "control_city_key",
    "control_grid_id",
    "control_unit_key",
    "failure_reason",
    "candidate_count",
    "candidate_city_count",
    "training_feature_count",
    "holdout_feature_count",
    "training_distance",
    "holdout_rms_standardized_gap",
    "holdout_max_abs_standardized_gap",
    "training_distance_threshold",
    "holdout_rms_threshold",
    "holdout_max_abs_threshold",
    "control_selection_uses_post_outcome",
]


@dataclass(frozen=True)
class ArtifactInputs:
    treatments: Path
    family_queue: Path
    control_queue: Path
    task_root: Path
    donor_universe: Path | None = None
    target_support: Path | None = None
    treatment_orders: tuple[int, ...] | None = None
    control_task_root: Path | None = None


from urban_intervention.utils import sha256_file  # noqa: E402


def aggregate_file_fingerprint(paths: Iterable[Path], relative_to: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(paths, key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(relative_to).as_posix()
        except ValueError:
            relative = path.resolve().as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
        count += 1
        total_bytes += size
    return {"sha256": digest.hexdigest(), "files": count, "bytes": total_bytes}


def git_state(root: Path) -> dict[str, object]:
    top_result: subprocess.CompletedProcess[str] | None = None
    commit_result: subprocess.CompletedProcess[str] | None = None
    dirty_result: subprocess.CompletedProcess[str] | None = None
    try:
        top_result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "--show-toplevel"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        commit_result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        dirty_result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "status", "--porcelain"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        top_result = commit_result = dirty_result = None
    if (
        top_result is not None
        and commit_result is not None
        and dirty_result is not None
        and top_result.returncode == 0
        and Path(top_result.stdout.strip()).resolve() == root
        and commit_result.returncode == 0
        and dirty_result.returncode == 0
        and commit_result.stdout.strip()
    ):
        return {
            "commit": commit_result.stdout.strip(),
            "dirty": bool(dirty_result.stdout.strip()),
            "source": "git",
        }

    # Production remains reproducible after a source bundle is exported without
    # .git: bind the release to the exact executable source tree instead.
    candidates: list[Path] = []
    for directory in (root / "src", root / "scripts"):
        if directory.exists():
            candidates.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.lower()
                in {
                    ".py",
                    ".r",
                    ".ps1",
                    ".js",
                    ".toml",
                    ".yml",
                    ".yaml",
                }
            )
    for path in (
        root / "pyproject.toml",
        root / "environment.yml",
        root / "requirements.txt",
    ):
        if path.is_file():
            candidates.append(path)
    candidates = sorted(set(candidates), key=lambda path: path.relative_to(root).as_posix())
    if not candidates:
        return {"commit": "unknown", "dirty": None, "source": "unavailable"}
    fingerprint = aggregate_file_fingerprint(candidates, root)
    return {
        "commit": f"tree-sha256:{fingerprint['sha256']}",
        "dirty": False,
        "source": "source_tree_sha256",
        "files": fingerprint["files"],
        "bytes": fingerprint["bytes"],
    }


def require_reproducible_code_state(state: dict[str, object]) -> None:
    version = str(state.get("commit") or "").strip().lower()
    if version in {"", "unknown", "none", "null"}:
        raise ValueError("Strict production release requires a known code version")
    if state.get("source") == "git" and bool(state.get("dirty")):
        raise ValueError("Strict production release refuses a dirty Git worktree")


def _has_value(value: object) -> bool:
    if value is None or value is pd.NA:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() not in {"", "nan", "none", "<na>"}


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _control_record_for_order(
    queue_row: pd.Series, control_task_root: Path | None
) -> pd.Series:
    """Load the durable control record when available, never just its queue summary."""
    order = int(queue_row["treatment_order"])
    if control_task_root is not None:
        path = control_task_root / f"{order:05d}" / "control_record.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Strict production control queue lacks durable record for order {order}: {path}"
            )
        record = pd.read_csv(path)
        if len(record) != 1:
            raise ValueError(f"Control record must contain exactly one row: {path}")
        return record.iloc[0]
    required = {
        "schema",
        "implementation_version",
        "backend",
        "viirs_cache_contract",
        "selected_method",
        "control_selection_uses_post_outcome",
    }
    missing = required - set(queue_row.index)
    if missing:
        raise ValueError(
            "Strict production control validation requires control_task_root or queue provenance "
            f"columns; missing {sorted(missing)}"
        )
    return queue_row


def validate_control_design_provenance(
    control_queue: pd.DataFrame,
    *,
    control_task_root: Path | None = None,
    expected_backend: str | None = None,
) -> list[Path]:
    """Reject stale control records before they can enter a formal release."""
    required_queue = {"treatment_order", "status"}
    missing_queue = required_queue - set(control_queue.columns)
    if missing_queue:
        raise ValueError(f"Control queue lacks columns: {sorted(missing_queue)}")
    if expected_backend is not None and expected_backend not in CONTROL_DESIGN_PROVENANCE:
        raise ValueError(f"Unknown control-design backend: {expected_backend}")

    active = control_queue.loc[
        control_queue["status"].astype(str).isin({"matched", "gsc_pending"})
    ]
    record_paths: list[Path] = []
    for _, queue_row in active.iterrows():
        order = int(queue_row["treatment_order"])
        if control_task_root is not None:
            record_paths.append(
                control_task_root / f"{order:05d}" / "control_record.csv"
            )
        record = _control_record_for_order(queue_row, control_task_root)
        if int(record.get("treatment_order", -1)) != order:
            raise ValueError(f"Control record identity mismatch for order {order}")
        if str(record.get("schema", "")) != CONTROL_DESIGN_SCHEMA:
            raise ValueError(
                f"Control record {order} uses stale control-design schema; "
                f"expected {CONTROL_DESIGN_SCHEMA}"
            )
        if str(record.get("viirs_cache_contract", "")) != CONTROL_DESIGN_VIIRS_CACHE_CONTRACT:
            raise ValueError(
                f"Control record {order} lacks the complete monthly VIIRS cache contract"
            )
        actual_backend = str(record.get("backend", ""))
        actual_version = str(record.get("implementation_version", ""))
        matching_backend = next(
            (
                name
                for name, provenance in CONTROL_DESIGN_PROVENANCE.items()
                if actual_backend == provenance["backend"]
                and actual_version == provenance["implementation_version"]
            ),
            None,
        )
        if matching_backend is None:
            raise ValueError(
                f"Control record {order} has unsupported backend/version "
                f"({actual_backend}, {actual_version})"
            )
        if expected_backend is not None and matching_backend != expected_backend:
            raise ValueError(
                f"Control record {order} backend {matching_backend} disagrees with "
                f"requested {expected_backend}"
            )
        if not _has_value(record.get("control_selection_uses_post_outcome")):
            raise ValueError(f"Control record {order} lacks the post-outcome selection flag")
        if _as_bool(record["control_selection_uses_post_outcome"]):
            raise ValueError(f"Control record {order} uses post-treatment information")
        if str(queue_row.get("status", "")) != str(record.get("status", "")):
            raise ValueError(f"Control queue status disagrees with record for order {order}")
        if str(queue_row["status"]) == "matched":
            expected_method = CONTROL_DESIGN_PROVENANCE[matching_backend]["selected_method"]
            if str(record.get("selected_method", "")) != expected_method:
                raise ValueError(
                    f"Control record {order} uses stale matching method; expected {expected_method}"
                )
    return record_paths


def runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
    }


def expected_label_skeleton(treatments: pd.DataFrame, specification_id: str) -> pd.DataFrame:
    required = {
        "treatment_order",
        "city_key",
        "grid_id",
        "station_event_id",
        "opening_month",
    }
    missing = required - set(treatments.columns)
    if missing:
        raise ValueError(f"Treatment list lacks columns: {sorted(missing)}")
    if treatments["treatment_order"].duplicated().any():
        raise ValueError("Treatment order must be unique")
    blocks: list[pd.DataFrame] = []
    identity = treatments[TREATMENT_IDENTITY_COLUMNS].copy()
    for family, outcomes in OUTCOME_HORIZONS.items():
        for outcome, horizons in outcomes.items():
            block = identity.loc[identity.index.repeat(len(horizons))].reset_index(drop=True)
            block["outcome_family"] = family
            block["outcome"] = outcome
            block["event_time"] = np.tile(np.asarray(horizons, dtype="int64"), len(identity))
            block["specification_id"] = specification_id
            blocks.append(block)
    skeleton = pd.concat(blocks, ignore_index=True)
    if skeleton.duplicated(KEY_COLUMNS).any():
        raise ValueError("Expected Response Artifact key is not unique")
    return skeleton


def _month_key(value: object) -> str:
    try:
        return str(pd.Period(str(value)[:7], freq="M"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid opening month: {value!r}") from exc


def _normalize_task_labels(
    path: Path, expected_identity: dict[str, object], expected_family: str, specification_id: str
) -> pd.DataFrame:
    labels = pd.read_parquet(path)
    required = set(KEY_COLUMNS + TASK_IDENTITY_COLUMNS)
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{path} lacks label identity columns: {sorted(missing)}")
    if labels.empty:
        raise ValueError(f"Successful task contains no label rows: {path}")
    expected_order = int(str(expected_identity["treatment_order"]))
    orders = set(pd.to_numeric(labels["treatment_order"], errors="raise").astype(int))
    if orders != {expected_order}:
        raise ValueError(
            f"{path} treatment_order {sorted(orders)} disagrees with task {expected_order}"
        )
    families = set(labels["outcome_family"].astype(str))
    if families != {expected_family}:
        raise ValueError(
            f"{path} outcome_family {sorted(families)} disagrees with task {expected_family}"
        )
    for column in ("city_key", "grid_id"):
        values = set(labels[column].astype(str))
        expected = str(expected_identity[column])
        if values != {expected}:
            raise ValueError(f"{path} {column} {sorted(values)} disagrees with {expected!r}")
    months = {_month_key(value) for value in labels["opening_month"]}
    expected_month = _month_key(expected_identity["opening_month"])
    if months != {expected_month}:
        raise ValueError(f"{path} opening_month {sorted(months)} disagrees with {expected_month!r}")
    specifications = set(labels["specification_id"].astype(str))
    if specifications != {specification_id}:
        raise ValueError(
            f"{path} specification_id {sorted(specifications)} disagrees with {specification_id!r}"
        )
    allowed = OUTCOME_HORIZONS[expected_family]
    if not set(labels["outcome"].astype(str)).issubset(allowed):
        raise ValueError(f"{path} contains outcomes outside family {expected_family}")
    for outcome, rows in labels.groupby("outcome"):
        horizons = set(pd.to_numeric(rows["event_time"], errors="raise").astype(int))
        if not horizons.issubset(set(allowed[str(outcome)])):
            raise ValueError(f"{path} contains invalid horizons for {outcome}: {sorted(horizons)}")
    if labels.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{path} contains duplicate label keys")
    for column in LABEL_COLUMNS:
        if column not in labels:
            labels[column] = pd.NA
    return labels[KEY_COLUMNS + LABEL_COLUMNS]


def collect_task_products(
    family_queue: pd.DataFrame,
    treatments: pd.DataFrame,
    task_root: Path,
    specification_id: str,
    strict_production: bool = True,
    workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    if family_queue.duplicated(["treatment_order", "outcome_family"]).any():
        raise ValueError("Outcome-family queue key is not unique")
    if treatments["treatment_order"].duplicated().any():
        raise ValueError("Treatment identity lookup is not unique")
    identity_lookup = treatments.set_index("treatment_order")[
        ["city_key", "grid_id", "station_event_id", "opening_month"]
    ].to_dict("index")
    tasks: list[tuple[int, str, Path, Path, dict[str, object], str]] = []
    for row in family_queue.itertuples(index=False):
        directory = task_root / f"{int(row.treatment_order):05d}" / str(row.outcome_family)
        if str(row.status) in SUCCESS_STATUSES:
            order = int(row.treatment_order)
            family = str(row.outcome_family)
            if order not in identity_lookup:
                raise ValueError(f"Task treatment_order is absent from treatment list: {order}")
            label_path = directory / "labels.parquet"
            manifest_path = directory / "manifest.json"
            if not label_path.exists() or not manifest_path.exists():
                raise FileNotFoundError(
                    f"Terminal successful task lacks labels/manifest: {directory}"
                )
            identity = {"treatment_order": order, **identity_lookup[order]}
            tasks.append((order, family, label_path, manifest_path, identity, str(row.status)))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        label_frames = list(
            executor.map(
                lambda item: _normalize_task_labels(item[2], item[4], item[1], specification_id),
                tasks,
            )
        )

    metadata: list[dict[str, object]] = []
    used_paths: list[Path] = []
    for task_index, (order, family, label_path, manifest_path, identity, queue_status) in enumerate(
        tasks
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "causal_response_labels_v1":
            raise ValueError(f"Unknown task manifest schema: {manifest_path}")
        if manifest.get("status") != queue_status:
            raise ValueError(f"Task manifest status disagrees with queue: {manifest_path}")
        if int(manifest.get("treatment_order", -1)) != order:
            raise ValueError(
                f"Task manifest treatment_order disagrees with directory: {manifest_path}"
            )
        if str(manifest.get("outcome_family")) != family:
            raise ValueError(
                f"Task manifest outcome_family disagrees with directory: {manifest_path}"
            )
        for column in ("city_key", "grid_id", "station_event_id"):
            if str(manifest.get(column)) != str(identity[column]):
                raise ValueError(
                    f"Task manifest {column} disagrees with treatment list: {manifest_path}"
                )
        if _month_key(manifest.get("opening_month")) != _month_key(identity["opening_month"]):
            raise ValueError(
                f"Task manifest opening_month disagrees with treatment list: {manifest_path}"
            )
        recorded_hash = manifest.get("labels_sha256")
        if strict_production and not recorded_hash:
            raise ValueError(f"Production task manifest lacks labels_sha256: {manifest_path}")
        if recorded_hash and str(recorded_hash) != sha256_file(label_path):
            raise ValueError(f"Task label hash disagrees with manifest: {label_path}")
        recorded_rows = manifest.get("label_rows")
        if strict_production and recorded_rows is None:
            raise ValueError(f"Production task manifest lacks label_rows: {manifest_path}")
        if recorded_rows is not None and int(recorded_rows) != len(label_frames[task_index]):
            raise ValueError(f"Task label row count disagrees with manifest: {label_path}")
        if strict_production and not bool(manifest.get("production_eligible")):
            raise ValueError(f"Production task is not production eligible: {manifest_path}")
        details = manifest.get("details")
        if strict_production and (
            not isinstance(details, dict) or not str(details.get("run_id") or "")
        ):
            raise ValueError(f"Production task lacks estimator run_id: {manifest_path}")
        task_labels = label_frames[task_index]
        python_task = task_labels.get(
            "estimator_backend", pd.Series(index=task_labels.index, dtype="object")
        ).astype(str).str.contains("python", case=False, na=False).any()
        if strict_production and python_task:
            versions = set(
                task_labels.get(
                    "implementation_version",
                    pd.Series(index=task_labels.index, dtype="object"),
                ).astype(str)
            )
            if versions != {FORMAL_IMPLEMENTATION_VERSION}:
                raise ValueError(
                    f"Production Python task uses a stale implementation: {manifest_path}"
                )
            if (
                not isinstance(details, dict)
                or details.get("formal_qualification_eligible") is not True
                or len(str(details.get("formal_qualification_receipt_sha256", ""))) != 64
            ):
                raise ValueError(
                    f"Production Python task lacks formal qualification proof: {manifest_path}"
                )
        metadata.append(
            {
                "treatment_order": order,
                "outcome_family": family,
                "task_production_eligible": bool(manifest.get("production_eligible", False)),
                "task_schema": manifest.get("schema"),
            }
        )
        used_paths.extend([label_path, manifest_path])
    if label_frames:
        # Drop per-task all-missing columns before concatenation so pandas cannot
        # silently change dtype inference when its deprecated concat behaviour is
        # removed. Required columns are restored immediately afterwards.
        labels = pd.concat(
            [frame.dropna(axis=1, how="all") for frame in label_frames],
            ignore_index=True,
        )
        for column in KEY_COLUMNS + LABEL_COLUMNS:
            if column not in labels:
                labels[column] = pd.NA
        labels = labels[KEY_COLUMNS + LABEL_COLUMNS]
    else:
        labels = pd.DataFrame(columns=KEY_COLUMNS + LABEL_COLUMNS)
    if labels.duplicated(KEY_COLUMNS).any():
        raise ValueError("Labels from different tasks violate the global primary key")
    return labels, pd.DataFrame(metadata), used_paths


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").map(np.isfinite)


def _safe_ratio(numerator: pd.Series | None, denominator: pd.Series | None) -> pd.Series:
    if numerator is None or denominator is None:
        return pd.Series([], dtype=float)
    top = pd.to_numeric(numerator, errors="coerce")
    bottom = pd.to_numeric(denominator, errors="coerce")
    return top / bottom.where(bottom > 0)


def build_response_frame(
    inputs: ArtifactInputs, specification_id: str, strict_production: bool = True, workers: int = 4
) -> tuple[pd.DataFrame, dict[str, object], list[Path]]:
    treatments = pd.read_parquet(inputs.treatments)
    family_queue = pd.read_csv(inputs.family_queue)
    control_queue = pd.read_csv(inputs.control_queue)
    if inputs.treatment_orders is not None:
        selected_orders = {int(order) for order in inputs.treatment_orders}
        if not selected_orders:
            raise ValueError("Sample release requires at least one treatment order")
        treatment_orders = set(treatments["treatment_order"].astype(int))
        missing_orders = selected_orders - treatment_orders
        if missing_orders:
            raise ValueError(
                "Sample treatment orders are absent from the treatment list: "
                f"{sorted(missing_orders)[:10]}"
            )
        treatments = treatments.loc[
            treatments["treatment_order"].astype(int).isin(selected_orders)
        ].copy()
        family_queue = family_queue.loc[
            family_queue["treatment_order"].astype(int).isin(selected_orders)
        ].copy()
        control_queue = control_queue.loc[
            control_queue["treatment_order"].astype(int).isin(selected_orders)
        ].copy()
    expected_families = set(OUTCOME_HORIZONS)
    control_record_paths: list[Path] = []
    if strict_production:
        if len(treatments) != 5_048:
            raise ValueError("Production release requires the immutable 5,048 treatments")
        expected_tasks = len(treatments) * len(expected_families)
        if len(family_queue) != expected_tasks:
            raise ValueError(f"Production family queue must contain {expected_tasks} rows")
        expected_orders = set(treatments["treatment_order"].astype(int))
        if set(family_queue["treatment_order"].astype(int)) != expected_orders:
            raise ValueError("Production family queue treatment orders differ from treatment list")
        families_by_order = family_queue.groupby("treatment_order")["outcome_family"].agg(set)
        if not families_by_order.map(lambda value: value == expected_families).all():
            raise ValueError("Every treatment must contain exactly the four registered families")
        if not set(family_queue["status"]).issubset(TERMINAL_STATUSES):
            raise ValueError("Production release refuses a non-terminal family queue")
        if len(control_queue) != len(treatments):
            raise ValueError("Production control queue must have one row per treatment")
        if control_queue["treatment_order"].duplicated().any():
            raise ValueError("Production control queue treatment order is not unique")
        if set(control_queue["treatment_order"].astype(int)) != expected_orders:
            raise ValueError("Production control queue treatment orders differ from treatment list")
        if not set(control_queue["status"]).issubset({"matched", "gsc_pending"}):
            raise ValueError("Production control queue contains unfinished/error rows")
        if "control_selection_uses_post_outcome" not in control_queue.columns:
            raise ValueError(
                "Production control queue lacks control_selection_uses_post_outcome column"
            )
        if control_queue["control_selection_uses_post_outcome"].fillna(True).map(_as_bool).any():
            raise ValueError("Production release detected post-treatment control-selection leakage")
        control_record_paths = validate_control_design_provenance(
            control_queue,
            control_task_root=inputs.control_task_root,
        )

    skeleton = expected_label_skeleton(treatments, specification_id)
    labels, task_metadata, task_paths = collect_task_products(
        family_queue,
        treatments,
        inputs.task_root,
        specification_id,
        strict_production=strict_production,
        workers=workers,
    )
    artifact = skeleton.merge(labels, on=KEY_COLUMNS, how="left", validate="one_to_one")

    queue_columns = [
        "treatment_order",
        "outcome_family",
        "status",
        "selected_method",
        "failure_reason",
    ]
    queue = family_queue[queue_columns].rename(
        columns={
            "status": "task_status",
            "selected_method": "queue_selected_method",
            "failure_reason": "task_failure_reason",
        }
    )
    artifact = artifact.merge(
        queue,
        on=["treatment_order", "outcome_family"],
        how="left",
        validate="many_to_one",
    )
    if not task_metadata.empty:
        artifact = artifact.merge(
            task_metadata,
            on=["treatment_order", "outcome_family"],
            how="left",
            validate="many_to_one",
        )
    else:
        artifact["task_production_eligible"] = False
        artifact["task_schema"] = pd.NA

    controls = control_queue[
        ["treatment_order"] + [column for column in CONTROL_COLUMNS if column in control_queue]
    ].rename(
        columns={
            "status": "control_design_status",
            "selected_method": "control_selection_method",
            "donor_scope": "control_donor_scope",
            "failure_reason": "control_failure_reason",
        }
    )
    artifact = artifact.merge(controls, on="treatment_order", how="left", validate="many_to_one")

    artifact["label_available"] = (
        artifact["label_available"].astype("boolean").fillna(False).astype(bool)
    )
    available = artifact["label_available"]
    expected_label = pd.to_numeric(artifact["observed"], errors="coerce") - pd.to_numeric(
        artifact["counterfactual"], errors="coerce"
    )
    supplied_label = pd.to_numeric(artifact["causal_response_label"], errors="coerce")
    if not np.allclose(
        supplied_label[available],
        expected_label[available],
        rtol=0,
        atol=1e-10,
        equal_nan=False,
    ):
        raise ValueError("Available label is not observed minus counterfactual")
    artifact["task_production_eligible"] = (
        artifact["task_production_eligible"].astype("boolean").fillna(False).astype(bool)
    )
    matched = artifact["task_status"].eq("matched_labelled")
    gsc = artifact["task_status"].eq("gsc_labelled")
    mc = artifact["task_status"].eq("mc_labelled")
    mc_estimator_proof = (
        (pd.to_numeric(artifact["pre_observed_periods"], errors="coerce") >= 1)
        & (pd.to_numeric(artifact["mc_lambda"], errors="coerce") >= 0)
        & _finite(artifact["mc_cv_mspe"])
        & artifact["method"].astype("string").str.startswith("athey_2021_mc_", na=False)
    )
    invalid_mc = mc & artifact["label_available"] & ~mc_estimator_proof
    if invalid_mc.any():
        raise ValueError(
            "MC labels lack cross-validated estimator proof "
            "(pre periods, non-negative lambda, finite CV MSPE, or method identity)"
        )
    mc_minimal_pre_support = pd.to_numeric(artifact["pre_observed_periods"], errors="coerce") <= 1
    label_scope = artifact.get(
        "donor_scope", pd.Series(index=artifact.index, dtype="object")
    ).astype("string")
    control_scope = artifact.get(
        "control_donor_scope", pd.Series(index=artifact.index, dtype="object")
    ).astype("string")
    method_identity = artifact["method"].astype("string")
    inferred_cross = method_identity.str.contains("all_city|cross_city", na=False)
    inferred_same = method_identity.str.contains("same_city", na=False)
    label_scope = label_scope.fillna(control_scope)
    label_scope = label_scope.mask(inferred_cross, "all_city_standardized")
    label_scope = label_scope.mask(inferred_same, "same_city")
    artifact["donor_scope"] = label_scope
    # Unlabelled skeleton rows legitimately have no donor scope.  They are
    # neither same-city nor cross-city and must not leave nullable booleans in
    # downstream training-mask operations.
    same_city = label_scope.eq("same_city").fillna(False).astype(bool)
    cross_city = label_scope.eq("all_city_standardized").fillna(False).astype(bool)
    # Same-city-first quality ordering: any same-city path ranks above any
    # cross-city path (matched > GSC > MC within each scope).
    artifact["quality_grade"] = np.select(
        [
            matched & same_city,
            gsc & artifact["method"].astype("string").str.contains("same_city", na=False),
            mc
            & mc_minimal_pre_support
            & artifact["method"].astype("string").str.contains("same_city", na=False),
            mc & artifact["method"].astype("string").str.contains("same_city", na=False),
            matched & cross_city,
            gsc & artifact["method"].astype("string").str.contains("all_city", na=False),
            mc
            & mc_minimal_pre_support
            & artifact["method"].astype("string").str.contains("all_city", na=False),
            mc & artifact["method"].astype("string").str.contains("all_city", na=False),
            artifact["task_status"].eq("skipped"),
        ],
        [
            "matched_same_city_pass",
            "gsc_same_city_pass",
            "mc_same_city_minimal_pre_support",
            "mc_same_city_pass",
            "matched_cross_city_pass",
            "gsc_cross_city_pass",
            "mc_cross_city_minimal_pre_support",
            "mc_cross_city_pass",
            "unavailable",
        ],
        default="pending",
    )
    artifact["training_distance_ratio"] = _safe_ratio(
        artifact.get("training_distance"), artifact.get("training_distance_threshold")
    )
    artifact["holdout_rms_ratio"] = _safe_ratio(
        artifact.get("holdout_rms_standardized_gap"), artifact.get("holdout_rms_threshold")
    )
    artifact["holdout_max_ratio"] = _safe_ratio(
        artifact.get("holdout_max_abs_standardized_gap"),
        artifact.get("holdout_max_abs_threshold"),
    )
    artifact["design_pass"] = matched | gsc | (mc & mc_estimator_proof)
    # Same-city-first main-specification marker: 1 for any same-city donor
    # scope, 0 for cross-city.  Cross-city labels are kept and trained on;
    # this column only distinguishes them in every downstream view.
    artifact["main_spec"] = same_city.astype(int)
    artifact["uncertainty_available"] = (
        _finite(artifact["standard_error"])
        & _finite(artifact["confidence_lower"])
        & _finite(artifact["confidence_upper"])
    )
    artifact["production_eligible"] = artifact["task_production_eligible"] & artifact["design_pass"]
    artifact["training_mask"] = (
        artifact["label_available"]
        & artifact["production_eligible"]
        & (~(gsc | mc) | artifact["uncertainty_available"])
    )
    # The first-stage/main analysis excludes cross-city labels by design;
    # retain a separate explicit extension mask so they remain auditable and
    # can be selected deliberately instead of leaking into the default view.
    artifact["main_training_mask"] = artifact["training_mask"] & same_city
    artifact["cross_city_extension_mask"] = artifact["training_mask"] & cross_city
    # The merged task-failure column can be all-NA and therefore inferred as
    # float64 by pandas.  Cast explicitly before assigning string fallback
    # reasons such as ``task_not_terminal`` (pandas 2.x rejects the upcast).
    failure_source = artifact.get(
        "task_failure_reason",
        pd.Series(pd.NA, index=artifact.index, dtype="string"),
    )
    artifact["failure_reason"] = failure_source.astype("string")
    missing_reason = artifact["failure_reason"].isna() & ~artifact["label_available"]
    artifact.loc[missing_reason, "failure_reason"] = np.where(
        artifact.loc[missing_reason, "task_status"].eq("skipped"),
        "task_skipped",
        np.where(
            artifact.loc[missing_reason, "task_status"].isin(SUCCESS_STATUSES),
            "target_period_outcome_or_counterfactual_missing",
            "task_not_terminal",
        ),
    )
    artifact.loc[artifact["label_available"], "failure_reason"] = pd.NA

    diagnostics = {
        "expected_rows": len(skeleton),
        "label_rows_read": len(labels),
        "available_labels": int(artifact["label_available"].sum()),
        "training_labels": int(artifact["training_mask"].sum()),
        "main_training_labels": int(artifact["main_training_mask"].sum()),
        "cross_city_extension_labels": int(
            artifact["cross_city_extension_mask"].sum()
        ),
        "task_status_counts": family_queue["status"].value_counts(dropna=False).to_dict(),
        "quality_grade_counts": artifact["quality_grade"].value_counts(dropna=False).to_dict(),
    }
    return artifact, diagnostics, task_paths + control_record_paths


def publish_response_artifact(
    inputs: ArtifactInputs,
    output_root: Path,
    release_id: str | None = None,
    specification_id: str = "main_a6_r1km",
    strict_production: bool = True,
    workers: int = 4,
    project_root: Path | None = None,
) -> Path:
    root = (project_root or Path.cwd()).resolve()
    state = git_state(root)
    if strict_production:
        require_reproducible_code_state(state)
    release_id = release_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = output_root / release_id
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite release: {destination}")
    staging = output_root / f".{release_id}.tmp-{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        artifact, diagnostics, task_paths = build_response_frame(
            inputs,
            specification_id,
            strict_production=strict_production,
            workers=workers,
        )
        run_id = str(uuid4())
        artifact["release_id"] = release_id
        artifact["run_id"] = run_id
        artifact["data_version"] = release_id
        artifact["code_version"] = state["commit"]
        artifact_path = staging / "response_artifact.parquet"
        artifact.to_parquet(artifact_path, index=False, compression="zstd")
        summary = (
            artifact.groupby(["outcome_family", "quality_grade"], dropna=False)
            .agg(
                rows=("treatment_order", "size"),
                available=("label_available", "sum"),
                training=("training_mask", "sum"),
            )
            .reset_index()
        )
        summary.to_csv(staging / "quality_summary.csv", index=False, encoding="utf-8-sig")

        source_paths = {
            "treatments": inputs.treatments,
            "family_queue": inputs.family_queue,
            "control_queue": inputs.control_queue,
        }
        if inputs.donor_universe is not None:
            source_paths["donor_universe"] = inputs.donor_universe
        if inputs.target_support is not None:
            source_paths["target_support"] = inputs.target_support
        source_hashes = {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in source_paths.items()
        }
        manifest = {
            "schema": "urban_response_artifact_release_v1",
            "release_id": release_id,
            "run_id": run_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "strict_production": strict_production,
            "treatment_scope": {
                "kind": "subset" if inputs.treatment_orders is not None else "full",
                "count": int(len(artifact["treatment_order"].unique())),
                "orders": sorted(int(v) for v in artifact["treatment_order"].unique())
                if inputs.treatment_orders is not None
                else None,
            },
            "specification_id": specification_id,
            "primary_key": KEY_COLUMNS,
            "diagnostics": diagnostics,
            "source_files": source_hashes,
            "task_products": aggregate_file_fingerprint(task_paths, inputs.task_root),
            "code": state,
            "runtime": runtime_versions(),
            "artifact": {
                "path": "response_artifact.parquet",
                "rows": len(artifact),
                "sha256": sha256_file(artifact_path),
            },
            "quality_summary_sha256": sha256_file(staging / "quality_summary.csv"),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        output_root.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    except Exception:
        # Leave no path that could be mistaken for a published release.
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
