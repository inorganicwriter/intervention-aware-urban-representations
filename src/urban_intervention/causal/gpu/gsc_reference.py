"""Repair audited R GSC reference inference after fect normalization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from urban_intervention.utils import (
    atomic_write_csv,
    atomic_write_parquet,
    sha256_file,
)

CORRECTION_MARKER = "_fect_normalization_corrected"
CORRECTION_METHOD = "divide_once_by_fect_normalization_scale"
NORMAL_CRITICAL_VALUE_95 = 1.959963984540054


@dataclass(frozen=True)
class GscReferenceRepair:
    panel: Path
    labels: Path
    manifest: Path
    normalization_scale: float
    corrected_rows: int
    status: str


def fect_normalization_scale(panel: pd.DataFrame) -> float:
    """Return the sample SD used by ``fect(normalize=TRUE)``."""
    if "model_value" not in panel.columns:
        raise ValueError("GSC estimation panel lacks model_value")
    values = pd.to_numeric(panel["model_value"], errors="coerce").to_numpy(float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise ValueError("GSC estimation panel has fewer than two finite model values")
    scale = float(np.std(finite, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("GSC fect normalization scale must be finite and positive")
    return scale


def correct_gsc_reference_labels(
    labels: pd.DataFrame, normalization_scale: float
) -> tuple[pd.DataFrame, int]:
    """Remove one duplicated fect normalization factor from GSC inference."""
    required = {
        "causal_response_label",
        "standard_error",
        "confidence_lower",
        "confidence_upper",
        "p_value",
        "uncertainty_source",
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"GSC reference labels lack columns: {sorted(missing)}")
    if not np.isfinite(normalization_scale) or normalization_scale <= 0:
        raise ValueError("normalization_scale must be finite and positive")

    result = labels.copy()
    sources = result["uncertainty_source"].astype("string")
    marked = sources.str.endswith(CORRECTION_MARKER, na=False)
    if marked.any():
        if not marked.all():
            raise ValueError("GSC reference labels contain a partial normalization repair")
        return result, 0

    effects = pd.to_numeric(result["causal_response_label"], errors="coerce")
    standard_errors = pd.to_numeric(result["standard_error"], errors="coerce")
    corrected_rows = standard_errors.notna() & np.isfinite(standard_errors)
    if not corrected_rows.any():
        raise ValueError("GSC reference labels contain no finite standard errors")
    if standard_errors[corrected_rows].lt(0).any():
        raise ValueError("GSC reference labels contain a negative standard error")

    corrected_se = standard_errors[corrected_rows] / normalization_scale
    corrected_effect = effects[corrected_rows]
    result.loc[corrected_rows, "standard_error"] = corrected_se
    result.loc[corrected_rows, "confidence_lower"] = (
        corrected_effect - NORMAL_CRITICAL_VALUE_95 * corrected_se
    )
    result.loc[corrected_rows, "confidence_upper"] = (
        corrected_effect + NORMAL_CRITICAL_VALUE_95 * corrected_se
    )
    p_values = []
    for effect, standard_error in zip(corrected_effect, corrected_se, strict=True):
        if not np.isfinite(effect):
            p_values.append(np.nan)
        elif standard_error == 0:
            p_values.append(1.0 if effect == 0 else 0.0)
        else:
            p_values.append(math.erfc(abs(float(effect / standard_error)) / math.sqrt(2)))
    result.loc[corrected_rows, "p_value"] = p_values
    result["uncertainty_source"] = sources.fillna("unknown") + CORRECTION_MARKER
    return result, int(corrected_rows.sum())


def _manifest_values(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"R GSC manifest is missing: {path}")
    manifest = pd.read_csv(path, dtype="string")
    if set(manifest.columns) != {"field", "value"}:
        raise ValueError(f"R GSC manifest has unexpected columns: {path}")
    return manifest


def _set_manifest_value(manifest: pd.DataFrame, field: str, value: str) -> None:
    selected = manifest["field"].eq(field)
    if selected.any():
        manifest.loc[selected, "value"] = value
    else:
        manifest.loc[len(manifest)] = {"field": field, "value": value}


def repair_gsc_reference(
    panel_path: str | Path,
    labels_path: str | Path,
    *,
    apply: bool,
) -> GscReferenceRepair:
    """Audit or repair one R reference directory used by qualification."""
    panel = Path(panel_path).resolve()
    labels = Path(labels_path).resolve()
    if panel.name != "estimation_panel.parquet" or labels.name != "causal_response_labels.parquet":
        raise ValueError("qualification reference paths do not name formal GSC artifacts")
    if panel.parent != labels.parent:
        raise ValueError("GSC panel and reference labels must share one output directory")
    manifest_path = labels.with_name("manifest.csv")
    manifest = _manifest_values(manifest_path)
    scale = fect_normalization_scale(pd.read_parquet(panel))
    labels_frame = pd.read_parquet(labels)
    corrected, corrected_rows = correct_gsc_reference_labels(labels_frame, scale)

    already_marked = corrected_rows == 0
    manifest_marked = manifest.loc[
        manifest["field"].eq("gsc_parametric_se_normalization_corrected"), "value"
    ].str.upper().eq("TRUE").any()
    status = "already_corrected" if already_marked and manifest_marked else "would_repair"
    if not apply:
        return GscReferenceRepair(
            panel, labels, manifest_path, scale, corrected_rows, status
        )

    original_hash = sha256_file(labels)
    if not already_marked:
        atomic_write_parquet(corrected, labels)
    corrected_hash = sha256_file(labels)
    _set_manifest_value(manifest, "gsc_parametric_se_normalization_corrected", "TRUE")
    _set_manifest_value(manifest, "gsc_parametric_se_normalization_scale", repr(scale))
    _set_manifest_value(manifest, "gsc_parametric_se_normalization_method", CORRECTION_METHOD)
    _set_manifest_value(
        manifest,
        "gsc_parametric_se_normalization_source",
        "estimation_panel.model_value_sample_sd",
    )
    recorded_fields = set(manifest["field"])
    if "gsc_reference_labels_sha256_before_correction" not in recorded_fields:
        _set_manifest_value(
            manifest, "gsc_reference_labels_sha256_before_correction", original_hash
        )
    if "gsc_reference_labels_sha256_after_correction" not in recorded_fields:
        _set_manifest_value(
            manifest, "gsc_reference_labels_sha256_after_correction", corrected_hash
        )
    atomic_write_csv(manifest, manifest_path)
    status = "already_corrected" if already_marked else "repaired"
    return GscReferenceRepair(
        panel, labels, manifest_path, scale, corrected_rows, status
    )


def qualification_gsc_references(root: str | Path) -> list[tuple[Path, Path]]:
    """Read the exact R GSC references named by qualification manifests."""
    qualification_root = Path(root).resolve()
    gsc_root = qualification_root / "gsc"
    references: set[tuple[Path, Path]] = set()
    for manifest_path in sorted(gsc_root.rglob("manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("estimator") != "gsc":
            continue
        panel = payload.get("panel")
        labels = payload.get("reference_labels")
        if not panel or not labels:
            raise ValueError(f"GSC qualification manifest lacks reference paths: {manifest_path}")
        references.add((Path(panel).resolve(), Path(labels).resolve()))
    if not references:
        raise FileNotFoundError(f"no GSC qualification references found below {gsc_root}")
    return sorted(references, key=lambda item: str(item[1]))
