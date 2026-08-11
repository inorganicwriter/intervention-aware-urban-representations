"""Robustness-check suite for the causal inference pipeline.

Pre-registered robustness checks (independent of the formal production run):

1. Spatial donor exclusion radius: 1km (main) vs 1.5km / 2km
   (DDR-001 SpatialDonorSpec.sensitivity_exclusion_m).
2. Anticipation window: 6 months (main) vs 0 / 12 months
   (complete_estimator_spec()$timing.sensitivity_anticipation_months).
3. Pre-treatment window length: 36 months (main) vs 24 / 48 months.
4. Covariate set: full vs without the new transit/location variables.
5. Donor pool: same-city vs cross-city (route comparison).

Each check is executed by a dedicated module under scripts/causal_r/ or
scripts/analysis/; this script is the entry point that runs the ones that
can run without the formal production outputs and writes an index report.

Usage:
    python scripts/causal_r/run_robustness_checks.py --all
    python scripts/causal_r/run_robustness_checks.py --spatial-exclusion
    python scripts/causal_r/run_robustness_checks.py --list
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_DIR,
)

ROBUSTNESS_DIR = OUTPUT_DIR / "robustness"


def run_python(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def run_r(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    import os

    env = os.environ.copy()
    r_lib = Path(os.environ.get("MIT_R_LIB", ROOT / ".r-lib"))
    if r_lib.is_dir():
        env["R_LIBS_USER"] = str(r_lib)
    return subprocess.run(
        [os.environ.get("MIT_RSCRIPT", "Rscript"), str(ROOT / script), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def spatial_exclusion() -> dict[str, object]:
    """Run donor-universe audit at 1km, 1.5km, 2km exclusion radii."""
    out = ROBUSTNESS_DIR / "spatial_exclusion"
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    for radius in (1000, 1500, 2000):
        target = out / f"exclusion_{radius}m"
        target.mkdir(parents=True, exist_ok=True)
        result = run_python(
            "src/urban_intervention/causal/spatial_donors.py",
            [
                "--exclusion-radius",
                str(radius),
                "--output-dir",
                str(target),
            ],
        )
        results[str(radius)] = {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1500:],
            "stderr_tail": result.stderr[-500:],
        }
    (out / "index.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def anticipation() -> dict[str, object]:
    """Dry-run label-task sensitivity at anticipation 0 / 6 / 12 months.

    ``--dry-run`` is mandatory here: the label queue marks tasks terminal and
    writes production-eligible manifests; a sensitivity probe must never
    mutate the production queue or emit real task artifacts.
    """
    out = ROBUSTNESS_DIR / "anticipation"
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    for months in (0, 6, 12):
        result = run_python(
            "scripts/causal_r/run_causal_label_queue.py",
            ["--start-order", "1", "--max-tasks", "1", "--anticipation-months", str(months), "--dry-run"],
        )
        results[str(months)] = {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1500:],
            "stderr_tail": result.stderr[-500:],
        }
    (out / "index.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def window_length() -> dict[str, object]:
    """Synthetic estimator-panel smoke at lag 24 / 36 / 48 months."""
    out = ROBUSTNESS_DIR / "window_length"
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    for lag in (24, 36, 48):
        result = run_r(
            "scripts/causal_r/robustness_window_smoke.R",
            [str(lag)],
        )
        results[str(lag)] = {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1500:],
            "stderr_tail": result.stderr[-500:],
        }
    (out / "index.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def covariate_set() -> dict[str, object]:
    """Synthetic matching smoke with full vs reduced covariate sets."""
    out = ROBUSTNESS_DIR / "covariate_set"
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    for variant in ("full", "no_transit"):
        result = run_r(
            "scripts/causal_r/robustness_covariate_smoke.R",
            [variant],
        )
        results[variant] = {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1500:],
            "stderr_tail": result.stderr[-500:],
        }
    (out / "index.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def donor_pool() -> dict[str, object]:
    """Synthetic matching smoke: same-city vs cross-city scope."""
    out = ROBUSTNESS_DIR / "donor_pool"
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    for scope in ("same_city", "all_city_standardized"):
        result = run_r(
            "scripts/causal_r/robustness_donor_scope_smoke.R",
            [scope],
        )
        results[scope] = {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1500:],
            "stderr_tail": result.stderr[-500:],
        }
    (out / "index.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def competing_events() -> dict[str, object]:
    """Exclusion smoke: drop grids with competing events / later openings."""
    out = ROBUSTNESS_DIR / "competing_events"
    out.mkdir(parents=True, exist_ok=True)
    result = run_r(
        "scripts/causal_r/robustness_competing_events.R",
        [],
    )
    results: dict[str, object] = {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1500:],
        "stderr_tail": result.stderr[-500:],
    }
    (out / "index.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


CHECKS = {
    "spatial-exclusion": spatial_exclusion,
    "anticipation": anticipation,
    "window-length": window_length,
    "covariate-set": covariate_set,
    "donor-pool": donor_pool,
    "competing-events": competing_events,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--list", action="store_true", help="List available checks")
    for name in CHECKS:
        parser.add_argument(f"--{name}", action="store_true", help=f"Run {name} check")
    args = parser.parse_args()

    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("Available robustness checks:")
        for name in CHECKS:
            print(f"  --{name}")
        return 0

    selected = [name for name in CHECKS if getattr(args, name.replace("-", "_"))]
    if args.all or not selected:
        selected = list(CHECKS)

    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "checks": {},
    }
    for name in selected:
        print(f"\n=== Running: {name} ===")
        try:
            CHECKS[name]()
            report["checks"][name] = {
                "status": "ok",
                "output": str((ROBUSTNESS_DIR / name).relative_to(ROOT)),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            report["checks"][name] = {"status": "error", "message": str(exc)}
    (ROBUSTNESS_DIR / "index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRobustness report: {ROBUSTNESS_DIR / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
