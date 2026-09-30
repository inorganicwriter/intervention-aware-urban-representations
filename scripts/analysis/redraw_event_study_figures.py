"""Redraw the combined three-method event-study figures from a saved suite.

The suite stores every input needed for the comparison figures, so figures can
be regenerated (for example after a plotting fix) without re-running the
estimators::

    python scripts/analysis/redraw_event_study_figures.py \
        --suite-root outputs/event_study_suite \
        --output-directory outputs/presentations/figures_canonical

Pass ``--write-coverage`` to also refresh the suite's
``event_time_coverage.csv`` (its ``included_in_comparison_figure`` flag then
follows the canonical windows of the redrawn figures instead of the legacy
three-method intersection).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.causal.diagnostic_figures import (  # noqa: E402
    FAMILIES,
    event_study_coverage_table,
    write_pooled_outcome_event_studies,
)
from urban_intervention.causal.event_study import EventStudyResult  # noqa: E402
from urban_intervention.causal.pooled_event_study import (  # noqa: E402
    PooledPathEventStudyResult,
)

SCALAR_DIAGNOSTIC_KEYS = (
    "reference_event_time",
    "grid_clusters",
    "city_clusters",
    "treated_events",
    "nobs",
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _diagnostics(path: Path) -> dict[str, object]:
    frame = _read_csv(path)
    row = frame.iloc[0].to_dict()
    diagnostics: dict[str, object] = {}
    for key, value in row.items():
        if key in SCALAR_DIAGNOSTIC_KEYS:
            diagnostics[key] = int(pd.to_numeric(value, errors="coerce"))
        else:
            diagnostics[key] = value
    return diagnostics


def load_matching(suite_root: Path, family: str) -> EventStudyResult:
    directory = suite_root / "matching_twfe" / family
    return EventStudyResult(
        coefficients=_read_csv(directory / "event_study_coefficients.csv"),
        grid_cluster_pretrend=_read_csv(directory / "parallel_trends_wald.csv"),
        city_cluster_pretrend=_read_csv(
            directory / "parallel_trends_wald_city_cluster.csv"
        ),
        diagnostics=_diagnostics(directory / "diagnostics.csv"),
    )


def load_pooled(suite_root: Path) -> PooledPathEventStudyResult:
    directory = suite_root / "gsc_mc_paths"
    return PooledPathEventStudyResult(
        paths=pd.read_parquet(directory / "effect_paths_with_pretrend.parquet"),
        series=_read_csv(directory / "event_study_series.csv"),
        grid_pretrend=_read_csv(directory / "pretrend_grid_cluster.csv"),
        city_pretrend=_read_csv(directory / "pretrend_city_cluster.csv"),
        diagnostics=_diagnostics(directory / "diagnostics.csv"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--families",
        default=",".join(FAMILIES),
        help="Comma-separated families to redraw.",
    )
    parser.add_argument("--donor-scope", default="same_city")
    parser.add_argument(
        "--write-coverage",
        action="store_true",
        help=(
            "Also rewrite <suite-root>/event_time_coverage.csv so its "
            "included_in_comparison_figure flag follows the canonical windows "
            "of the redrawn figures."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = [value.strip() for value in args.families.split(",") if value.strip()]
    matching_results = {
        family: load_matching(args.suite_root, family) for family in families
    }
    pooled = load_pooled(args.suite_root)
    written = write_pooled_outcome_event_studies(
        matching_results,
        pooled,
        args.output_directory,
        donor_scope=args.donor_scope,
    )
    for path in written:
        print(path)
    if args.write_coverage:
        coverage_path = args.suite_root / "event_time_coverage.csv"
        event_study_coverage_table(
            matching_results, pooled, donor_scope=args.donor_scope
        ).to_csv(coverage_path, index=False, encoding="utf-8-sig")
        print(coverage_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
