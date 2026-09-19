"""Audit which MC jackknife convention the installed code follows.

Read-only.  Prints the installed ``jackknife_standard_error`` source and scans
MC outputs that carry jackknife draws (formal task directories and
qualification shadow directories).  For every period it checks whether the
recorded standard error reproduces:

- A: total-refit multiplier, failed refits excluded (fect's R behaviour);
- B: finite-refit multiplier (project convention);
- C: total-refit multiplier with failed refits counted as zero.

When a shadow manifest records its R reference labels, the R standard errors
are matched the same way.  This decides which convention is R-consistent
without re-running qualification.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import urban_intervention.causal.gpu.inference as inf  # noqa: E402

CANDIDATE_NAMES = (
    "A_total_multiplier",
    "B_finite_multiplier",
    "C_total_zero_fill",
)


def candidates(point: float, values: np.ndarray) -> tuple[dict[str, float], int, int]:
    y = np.asarray(values, dtype=float)
    finite = y[np.isfinite(y)]
    total, count = int(y.size), int(finite.size)
    result: dict[str, float] = {}
    if count >= 2:
        pseudo = point * total - finite * (total - 1)
        result["A_total_multiplier"] = float(np.sqrt(np.var(pseudo, ddof=1) / count))
        pseudo = point * count - finite * (count - 1)
        result["B_finite_multiplier"] = float(np.sqrt(np.var(pseudo, ddof=1) / count))
    filled = np.nan_to_num(y, nan=0.0)
    pseudo = point * total - filled * (total - 1)
    result["C_total_zero_fill"] = float(np.sqrt(np.var(pseudo, ddof=1) / total))
    return result, total, count


def matches(recorded: float, value: float) -> bool:
    return bool(
        np.isfinite(recorded)
        and np.isfinite(value)
        and abs(recorded - value) <= 1e-8 + 1e-5 * abs(value)
    )


def print_code() -> None:
    print("=" * 78)
    print("SERVER CODE:", inf.__file__)
    try:
        source = inspect.getsource(inf.jackknife_standard_error)
    except (OSError, TypeError):
        text = Path(inf.__file__).read_text(encoding="utf-8")
        index = text.find("def jackknife_standard_error")
        source = text[index : index + 1600]
    print(source)
    print("flags: total-refit multiplier ->", "local * (total - 1)" in source)
    print("flags: finite-refit multiplier ->", "local * (count - 1)" in source)


def candidate_directories() -> dict[Path, Path]:
    roots = [
        Path(p)
        for p in ("outputs/complete_estimators", "outputs", "data/active/causal")
    ]
    found: dict[Path, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for name in ("inference_draws.parquet", "gpu_inference_draws.parquet"):
            for path in root.rglob(name):
                found[path.parent] = path
    return found


def read_points(directory: Path) -> tuple[pd.DataFrame | None, str, str]:
    labels_path = directory / "labels.parquet"
    gpu_path = directory / "gpu_inference.parquet"
    if labels_path.is_file():
        frame = pd.read_parquet(labels_path)
        methods = frame["method"].astype(str) if "method" in frame.columns else None
        if methods is None or methods.str.contains("athey|mc", case=False, regex=True).any():
            return frame, "causal_response_label", "standard_error"
    if gpu_path.is_file():
        frame = pd.read_parquet(gpu_path)
        effect = "gpu_effect" if "gpu_effect" in frame.columns else "effect"
        error = (
            "gpu_standard_error"
            if "gpu_standard_error" in frame.columns
            else "standard_error"
        )
        return frame, effect, error
    return None, "", ""


def periodic_rows(
    points: pd.DataFrame, draws: pd.DataFrame, effect_col: str, se_col: str
) -> list[tuple[str, int, int, float, list[str], dict[str, float]]]:
    draw_columns = {str(column) for column in draws.columns if column != "replicate"}
    rows: list[tuple[str, int, int, float, list[str], dict[str, float]]] = []
    for _, row in points.iterrows():
        key = str(row["period"])
        if key not in draw_columns:
            continue
        try:
            point = float(row.get(effect_col))
            recorded = float(row.get(se_col))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(point) and np.isfinite(recorded)):
            continue
        values = draws[key].to_numpy(dtype=float)
        if np.isfinite(values).sum() < 2:
            continue
        values_by_candidate, total, count = candidates(point, values)
        hit = [
            name
            for name, value in values_by_candidate.items()
            if matches(recorded, value)
        ]
        rows.append((key, count, total, recorded, hit, values_by_candidate))
    return rows


def r_reference_summary(
    manifest: dict[str, object],
    rows: list[tuple[str, int, int, float, list[str], dict[str, float]]],
) -> None:
    reference_raw = manifest.get("reference_labels")
    if not reference_raw:
        return
    reference_path = Path(str(reference_raw))
    if not reference_path.is_absolute():
        reference_path = Path.cwd() / reference_path
    if not reference_path.is_file():
        return
    reference = pd.read_parquet(reference_path)
    if not {"period", "standard_error"}.issubset(reference.columns):
        return
    reference_map = dict(
        zip(
            reference["period"].astype(str),
            pd.to_numeric(reference["standard_error"], errors="coerce"),
            strict=False,
        )
    )
    agree = {name: 0 for name in CANDIDATE_NAMES}
    compared = 0
    for key, _, _, _, _, values_by_candidate in rows:
        reference_se = reference_map.get(key)
        if reference_se is None or not np.isfinite(float(reference_se)):
            continue
        compared += 1
        for name, value in values_by_candidate.items():
            if matches(float(reference_se), value):
                agree[name] += 1
    print(
        f"  R reference={reference_path} compared_periods={compared} "
        f"R_matches={agree}"
    )


def main() -> int:
    print_code()
    directories = candidate_directories()
    print("=" * 78)
    print("MC outputs with jackknife draws:", len(directories))
    missing_tasks = 0
    for directory, draws_path in sorted(directories.items()):
        manifest_path = directory / "manifest.json"
        manifest: dict[str, object] = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
        estimator = str(manifest.get("estimator", "")).lower()
        if estimator and estimator != "mc" and "/mc/" not in directory.as_posix():
            continue
        points, effect_col, se_col = read_points(directory)
        if points is None or "period" not in points.columns:
            continue
        draws = pd.read_parquet(draws_path)
        rows = periodic_rows(points, draws, effect_col, se_col)
        if not rows:
            continue
        incomplete = [row for row in rows if row[1] < row[2]]
        if incomplete:
            missing_tasks += 1
        print("-" * 78)
        print(directory)
        print(
            f"  estimator={estimator or 'unknown'} periods={len(rows)} "
            f"periods_with_missing_draws={len(incomplete)}"
        )
        print("  recorded-SE verdict sets:", sorted({tuple(row[4]) for row in rows}))
        for row in (incomplete or rows)[:8]:
            key, count, total, recorded, hit, values_by_candidate = row
            print(
                f"    period={key} n_finite={count}/{total} "
                f"se={recorded:.8g} matches={hit}"
            )
            print(
                "      A_total={A_total_multiplier:.8g} "
                "B_finite={B_finite_multiplier:.8g} "
                "C_total_zero={C_total_zero_fill:.8g}".format(**values_by_candidate)
            )
        r_reference_summary(manifest, rows)
    print("=" * 78)
    print("SUMMARY tasks_with_missing_draws:", missing_tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
