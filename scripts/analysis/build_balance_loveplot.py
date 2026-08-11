"""Build a covariate-balance Love plot for the frozen grid-control design.

For every treatment grid whose control design is `matched`, the design task
directory persists a per-feature balance table
(`outputs/control_design/tasks/<order>/feature_balance.parquet`) with the
treated value, the selected control value, the raw gap and the standardized
gap for every active matching feature. This script aggregates those pairs and
compares them with the same-city (or all-city, for cross-city tasks) eligible
donor pool:

- `smd_before`: pooled standardized mean difference between treated grids and
  the donor pool used by the design stage, computed on the same pre-treatment
  lag features (annual stores for POI/population, 12-month blocks for
  monthly VIIRS/housing);
- `smd_after`: pooled standardized mean difference between treated grids and
  their selected matched controls, computed from the persisted pairs.

Neither quantity reads any post-treatment outcome, and both are pure
post-hoc diagnostics over the frozen design; nothing here re-selects
controls. The plot follows the standard Love-plot format: one row per
feature, red = before, blue-green = after, dashed lines at +/- 0.10.

Outputs:
    outputs/figures/balance_loveplot.png
    outputs/figures/balance_smd.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.data.paths import (  # noqa: E402
    CONTROL_DESIGN_QUEUE,
    ELIGIBLE_DONORS,
    FEATURE_STORE_DIR,
    OUTPUT_CONTROL_TASKS_DIR,
    TREATMENT_UNIT_LIST,
)

MONTHLY_FEATURES = {"housing_log_price", "viirs_avg_asinh"}
ANNUAL_FEATURES = {
    "poi_count_log",
    "poi_category_entropy",
    "poi_commercial_share",
    "poi_transport_access_log",
    "population_log",
}
MONTHS_PER_BLOCK = 12
ANTICIPATION_MONTHS = 6
LAG_CALENDAR_OFFSET = ANTICIPATION_MONTHS + 1  # last clean pre month = opening - 7
BALANCE_THRESHOLD = 0.10


def parse_feature_name(feature: str) -> tuple[str, int]:
    variable, lag = feature.rsplit("__lag", 1)
    return variable, int(lag)


def monthly_block_months(opening: pd.Timestamp, lag: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Clean 12-month block for `lag`: lag1 = opening-18..opening-7, etc."""
    end = opening - pd.DateOffset(months=LAG_CALENDAR_OFFSET + MONTHS_PER_BLOCK * (lag - 1))
    start = end - pd.DateOffset(months=MONTHS_PER_BLOCK - 1)
    return start, end


def pooled_smd(treated: np.ndarray, control: np.ndarray) -> float:
    treated = treated[np.isfinite(treated)]
    control = control[np.isfinite(control)]
    if len(treated) < 1 or len(control) < 2:
        return float("nan")
    if treated.std(ddof=1) == 0 and control.std(ddof=1) == 0:
        return float("nan")
    pooled_sd = np.sqrt(
        ((len(treated) - 1) * treated.var(ddof=1) + (len(control) - 1) * control.var(ddof=1))
        / (len(treated) + len(control) - 2)
    )
    if not np.isfinite(pooled_sd) or pooled_sd <= 0:
        return float("nan")
    return float((treated.mean() - control.mean()) / pooled_sd)


def _donor_pool_values(
    store_annual: pd.DataFrame,
    store_monthly: pd.DataFrame,
    donor_grids: pd.DataFrame,
    variable: str,
    lag: int,
    opening_month: str,
) -> pd.Series:
    """Donor-grid pre-treatment values for one feature (windowed, memory-safe)."""
    opening = pd.Timestamp(pd.Period(str(opening_month)[:7], freq="M").to_timestamp())
    if variable in MONTHLY_FEATURES:
        if store_monthly is None or store_monthly.empty or "month" not in store_monthly.columns:
            return pd.Series(dtype=float)
        if variable not in store_monthly.columns:
            return pd.Series(dtype=float)
        start, end = monthly_block_months(opening, lag)
        block = store_monthly.loc[
            store_monthly["month"].between(start, end) & store_monthly[variable].notna()
        ].copy()
        if block.empty:
            return pd.Series(dtype=float)
        minimum = 12 if variable == "viirs_avg_asinh" else 1
        aggregated = block.groupby("grid_id", sort=True)[variable].agg(["count", "mean"])
        aggregated = aggregated.loc[aggregated["count"] >= minimum, "mean"]
        aggregated.index = aggregated.index.astype(str)
        return aggregated.reindex(donor_grids.astype(str)).dropna()
    if variable in ANNUAL_FEATURES:
        if store_annual is None or store_annual.empty or "year" not in store_annual.columns:
            return pd.Series(dtype=float)
        if variable not in store_annual.columns:
            return pd.Series(dtype=float)
        year = int(opening.year - lag)
        block = store_annual.loc[
            store_annual["year"].eq(year) & store_annual[variable].notna()
        ].copy()
        if block.empty:
            return pd.Series(dtype=float)
        values = block.groupby("grid_id", sort=True)[variable].mean()
        values.index = values.index.astype(str)
        return values.reindex(donor_grids.astype(str)).dropna()
    return pd.Series(dtype=float)


def build_balance_report(
    tasks_root: Path,
    store_root: Path,
    donors_path: Path,
    control_queue_path: Path,
    treatments_path: Path,
) -> pd.DataFrame:
    queue = pd.read_csv(control_queue_path)
    matched = queue.loc[queue["status"].eq("matched")].copy()
    if matched.empty:
        raise RuntimeError("No matched control-design rows found in the control queue")
    treatments = pd.read_parquet(treatments_path)
    donors = pd.read_parquet(donors_path, columns=["city_key", "grid_id"])
    donors["city_key"] = donors["city_key"].astype("string")
    donors["grid_id"] = donors["grid_id"].astype("string")

    # Per-task balance rows: (treated/control values come from the persisted
    # design-stage feature_balance.parquet; donor values are windowed per
    # (city, opening month) group to keep memory bounded).
    tasks: list[dict[str, object]] = []
    for _, row in matched.iterrows():
        order = int(row["treatment_order"])
        scope = str(row["donor_scope"])
        task_dir = tasks_root / f"{order:05d}"
        balance_path = task_dir / "feature_balance.parquet"
        if not balance_path.exists():
            continue
        balance = pd.read_parquet(balance_path)
        if balance.empty or "feature" not in balance.columns:
            continue
        opening_month = str(
            treatments.loc[treatments["treatment_order"].eq(order), "opening_month"].iloc[0]
        )
        for feature, group in balance.groupby("feature", sort=True):
            tasks.append(
                {
                    "city_key": str(row["city_key"]),
                    "opening_month": opening_month,
                    "scope": scope,
                    "feature": str(feature),
                    "treated": group["treated_value"].to_numpy(dtype=float),
                    "control": group["control_value"].to_numpy(dtype=float),
                }
            )

    donor_pool_cache: dict[tuple[str, str, str, int], pd.Series] = {}
    city_stores: dict[str, tuple[pd.DataFrame | None, pd.DataFrame | None]] = {}
    by_feature: dict[str, dict[str, list]] = {}
    for task in tasks:
        feature = task["feature"]
        variable, lag = parse_feature_name(feature)
        city = task["city_key"]
        key = (city, task["opening_month"], variable, lag)
        if key not in donor_pool_cache:
            if city not in city_stores:
                annual_path = store_root / f"{city}_annual.parquet"
                monthly_path = store_root / f"{city}_monthly.parquet"
                annual = pd.read_parquet(annual_path) if annual_path.exists() else None
                monthly = pd.read_parquet(monthly_path) if monthly_path.exists() else None
                if monthly is not None and "month" in monthly.columns:
                    monthly["month"] = pd.to_datetime(monthly["month"])
                    # Pre-filter to the union of the city's lag windows.
                    city_openings = [
                        pd.Timestamp(
                            pd.Period(str(t["opening_month"])[:7], freq="M").to_timestamp()
                        )
                        for t in tasks
                        if t["city_key"] == city
                    ]
                    if city_openings:
                        lo = min(
                            o - pd.DateOffset(months=LAG_CALENDAR_OFFSET + MONTHS_PER_BLOCK * 2)
                            for o in city_openings
                        )
                        hi = max(
                            o - pd.DateOffset(months=LAG_CALENDAR_OFFSET - 1) for o in city_openings
                        )
                        monthly = monthly.loc[monthly["month"].between(lo, hi)].copy()
                city_stores[city] = (annual, monthly)
            annual, monthly = city_stores[city]
            # The before-pool is the same-city eligible donor set for every
            # task (including cross-city-scope tasks, whose standardized
            # matching pool is not directly comparable on raw scales).
            donor_grids = donors.loc[donors["city_key"].eq(city), "grid_id"]
            donor_pool_cache[key] = _donor_pool_values(
                annual, monthly, donor_grids, variable, lag, task["opening_month"]
            )
        donor_values = donor_pool_cache[key]
        entry = by_feature.setdefault(
            feature,
            {
                "treated": [],
                "control": [],
                "donor": [],
            },
        )
        entry["treated"].extend(np.asarray(task["treated"], dtype=float).tolist())
        entry["control"].extend(np.asarray(task["control"], dtype=float).tolist())
        entry["donor"].extend(donor_values.astype(float).tolist())

    rows = []
    for feature, entry in sorted(by_feature.items()):
        treated = np.asarray(entry["treated"], dtype=float)
        control = np.asarray(entry["control"], dtype=float)
        donor = np.asarray(entry["donor"], dtype=float)
        smd_before = pooled_smd(treated, donor)
        smd_after = pooled_smd(treated, control)
        rows.append(
            {
                "feature": feature,
                "n_treated": int(len(treated)),
                "n_controls": int(len(control)),
                "n_donors": int(len(donor)),
                "smd_before": round(smd_before, 4) if np.isfinite(smd_before) else np.nan,
                "smd_after": round(smd_after, 4) if np.isfinite(smd_after) else np.nan,
                "pass_0_10_before": bool(
                    np.isfinite(smd_before) and abs(smd_before) <= BALANCE_THRESHOLD
                ),
                "pass_0_10_after": bool(
                    np.isfinite(smd_after) and abs(smd_after) <= BALANCE_THRESHOLD
                ),
            }
        )
    return pd.DataFrame(rows)


def render_loveplot(
    report: pd.DataFrame, out_path: Path, threshold: float = BALANCE_THRESHOLD
) -> None:
    frame = report.dropna(subset=["smd_before", "smd_after"])
    frame = frame.assign(_rank=frame[["smd_before", "smd_after"]].abs().max(axis=1)).sort_values(
        "_rank", ascending=True
    )
    y_positions = np.arange(len(frame))
    fig, axis = plt.subplots(figsize=(9, max(5, 0.38 * len(frame) + 2)))
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.axvline(threshold, color="grey", linestyle="--", linewidth=0.9)
    axis.axvline(-threshold, color="grey", linestyle="--", linewidth=0.9)
    axis.scatter(
        frame["smd_before"],
        y_positions,
        marker="o",
        color="#c0392b",
        label="Before (treated vs donor pool)",
        s=42,
        zorder=3,
    )
    axis.scatter(
        frame["smd_after"],
        y_positions,
        marker="D",
        color="#1f8a70",
        label="After (treated vs matched control)",
        s=42,
        zorder=3,
    )
    axis.axvline(0.1, color="red", linestyle=":", linewidth=1.0, alpha=0.6)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(frame["feature"], fontsize=8)
    axis.set_xlabel("Standardized mean difference (SMD)")
    axis.set_title(
        "Covariate balance, frozen grid-control design "
        f"(n treated = {int(frame['n_treated'].max())})"
    )
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(axis="x", color="grey", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=OUTPUT_CONTROL_TASKS_DIR)
    parser.add_argument("--store-root", type=Path, default=FEATURE_STORE_DIR)
    parser.add_argument("--donors", type=Path, default=ELIGIBLE_DONORS)
    parser.add_argument("--control-queue", type=Path, default=CONTROL_DESIGN_QUEUE)
    parser.add_argument("--treatments", type=Path, default=TREATMENT_UNIT_LIST)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "figures")
    args = parser.parse_args()

    report = build_balance_report(
        args.tasks_root,
        args.store_root,
        args.donors,
        args.control_queue,
        args.treatments,
    )
    if report.empty:
        print("No matched balance records found; nothing to plot.")
        return 1
    csv_path = args.out_dir / "balance_smd.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False, encoding="utf-8-sig")
    render_loveplot(report, args.out_dir / "balance_loveplot.png")
    print(f"Wrote {csv_path} ({len(report)} features) and balance_loveplot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
