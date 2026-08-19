"""Transaction-composition checks around station openings (Lianjia 22 cities).

For every treated grid in the Lianjia 22 cities, this builds event-time
aligned monthly paths of three composition variables from the transaction
layer: transaction count, mean building area, mean building age.  The pooled
paths are plotted and a pre/post jump statistic is reported: a large change
in count or attribute means around the opening would confound price effects
with sample-composition changes.

Outputs under outputs/causal_labels/housing_composition/:
- pooled_paths_{window}.csv / .png for windows around the opening month
- composition_report.json with per-family jump statistics

Usage:
    python scripts/analysis/audit_housing_composition.py [--window 24]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.data.paths import (  # noqa: E402
    STAGING_LIANJIA_TRANSACTIONS_DIR,
    TREATMENT_UNIT_LIST,
    grid_path,
)

OUTPUT_DIR = ROOT / "outputs" / "causal_labels" / "housing_composition"

LIANJIA_CITIES = [
    "beijing", "changzhou", "chongqing", "dongguan", "foshan", "guangzhou",
    "hangzhou", "jinan", "jinhua", "luoyang", "nanjing", "nantong", "ningbo",
    "qingdao", "shaoxing", "shenzhen", "suzhou", "taizhou", "wenzhou", "wuxi",
    "xuzhou", "zhengzhou",
]


def map_grid(frame: pd.DataFrame, city: str) -> pd.DataFrame:
    from pyproj import Transformer

    from urban_intervention.config.project import CITIES

    grids = pd.read_parquet(
        grid_path(city), columns=["grid_id", "row", "col", "centroid_lon", "centroid_lat"]
    )
    transformer = Transformer.from_crs("EPSG:4326", CITIES[city]["projected_crs"], always_xy=True)
    sample = grids.iloc[:: max(1, len(grids) // 2_000)].copy()
    sx, sy = transformer.transform(sample["centroid_lon"], sample["centroid_lat"])
    origin_x = float(np.median(sx - (sample["col"].to_numpy() + 0.5) * 500.0))
    origin_y = float(np.median(sy - (sample["row"].to_numpy() + 0.5) * 500.0))
    lon = pd.to_numeric(frame["lon"], errors="coerce")
    lat = pd.to_numeric(frame["lat"], errors="coerce")
    valid = lon.between(70, 140) & lat.between(10, 60)
    x, y = transformer.transform(lon.loc[valid].to_numpy(), lat.loc[valid].to_numpy())
    cols = np.floor((x - origin_x) / 500.0).astype(np.int64)
    rows = np.floor((y - origin_y) / 500.0).astype(np.int64)
    candidates = pd.Series(
        [f"g{r:05d}x{c:05d}" for r, c in zip(rows, cols, strict=False)], index=lon.index[valid]
    )
    retained = set(grids["grid_id"].astype(str))
    result = pd.Series("", index=frame.index)
    result.loc[valid] = np.where(candidates.isin(retained), candidates, "")
    frame = frame.copy()
    frame["grid_id"] = result
    return frame


def build_city_monthly(city: str) -> pd.DataFrame:
    paths = sorted(STAGING_LIANJIA_TRANSACTIONS_DIR.glob(f"*/{city}.parquet"))
    frames = [pd.read_parquet(p) for p in paths if p.stat().st_size > 10_000]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame["is_valid"].fillna(False)].copy()
    frame = frame.drop_duplicates(subset=["source_record_id"], keep="first")
    frame["_price"] = pd.to_numeric(frame["unit_price_cny_m2"], errors="coerce")
    frame["_area"] = pd.to_numeric(frame["building_area_m2"], errors="coerce")
    built = pd.to_numeric(frame.get("built_year_mid"), errors="coerce")
    frame["_year"] = pd.to_datetime(frame["deal_date"], errors="coerce").dt.year
    frame["_age"] = np.where(
        built.notna() & frame["_year"].notna(),
        np.clip(frame["_year"] - built, 0, 80),
        np.nan,
    )
    frame = frame.loc[
        frame["_price"].gt(0)
        & frame["_area"].gt(0)
        & frame["_year"].notna()
    ].copy()
    frame = map_grid(frame, city)
    frame = frame.loc[frame["grid_id"].ne("")].copy()
    frame["observed_month"] = pd.to_datetime(frame["deal_date"], errors="coerce").dt.to_period(
        "M"
    ).dt.to_timestamp()
    grouped = (
        frame.groupby(["grid_id", "observed_month"], as_index=False)
        .agg(
            n_transactions=("_price", "size"),
            mean_area=("_area", "mean"),
            mean_age=("_age", "mean"),
        )
    )
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    window = args.window

    treatments = pd.read_parquet(TREATMENT_UNIT_LIST)
    treatments["opening_month"] = pd.to_datetime(
        treatments["opening_month"].astype(str) + "-01"
    )
    treated = treatments.loc[
        treatments["city_key"].isin(LIANJIA_CITIES),
        ["treatment_order", "city_key", "grid_id", "opening_month"],
    ].copy()

    parts: list[pd.DataFrame] = []
    for city in LIANJIA_CITIES:
        monthly = build_city_monthly(city)
        if monthly.empty:
            continue
        city_treated = treated.loc[treated["city_key"].eq(city)]
        joined = monthly.merge(
            city_treated[["grid_id", "treatment_order", "opening_month"]],
            on="grid_id",
            how="inner",
        )
        parts.append(joined)
    if not parts:
        raise RuntimeError("No treated grids matched Lianjia transaction panels")
    events = pd.concat(parts, ignore_index=True)
    events["event_time"] = (
        (events["observed_month"] - events["opening_month"]).dt.days / 30.44
    ).round().astype(int)
    events = events.loc[events["event_time"].between(-window, window)].copy()

    pooled = (
        events.groupby("event_time", as_index=False)
        .agg(
            n_grids=("treatment_order", "nunique"),
            mean_n_transactions=("n_transactions", "mean"),
            median_n_transactions=("n_transactions", "median"),
            mean_area=("mean_area", "mean"),
            mean_age=("mean_age", "mean"),
        )
        .sort_values("event_time")
    )

    pre = events.loc[events["event_time"].between(-window, -7)]
    post = events.loc[events["event_time"].between(6, window)]
    jump: dict[str, object] = {}
    for column in ("n_transactions", "mean_area", "mean_age"):
        if pre[column].notna().sum() < 10 or post[column].notna().sum() < 10:
            jump[column] = None
            continue
        pre_mean = float(pre[column].mean())
        post_mean = float(post[column].mean())
        jump[column] = {
            "pre_mean": round(pre_mean, 4),
            "post_mean": round(post_mean, 4),
            "relative_change": round((post_mean - pre_mean) / max(abs(pre_mean), 1e-12), 4),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.output_dir / f"pooled_paths_w{window}.csv", index=False, encoding="utf-8-sig")
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        for axis, column, label in zip(
            axes,
            ("mean_n_transactions", "mean_area", "mean_age"),
            ("transactions per grid-month", "mean building area (m2)", "mean building age (years)"),
            strict=False,
        ):
            axis.plot(pooled["event_time"], pooled[column], marker="o", ms=3, lw=1.2)
            axis.axvline(0, color="grey", ls="--", lw=0.8)
            axis.set_ylabel(label)
            axis.grid(alpha=0.3)
        axes[-1].set_xlabel("event time (months)")
        figure.suptitle(f"Transaction composition around opening (Lianjia 22 cities, W={window})")
        figure.tight_layout()
        figure.savefig(args.output_dir / f"pooled_paths_w{window}.png", dpi=150)
        plt.close(figure)
    except ImportError:
        pass

    report: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "window": window,
        "treated_grids_with_any_transaction": int(events["treatment_order"].nunique()),
        "treated_grids_in_lianjia_cities": int(treated["treatment_order"].nunique()),
        "event_month_rows": int(len(events)),
        "jump_statistics_pre_vs_post": jump,
    }
    (args.output_dir / "composition_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote composition report to {args.output_dir}")
    print(json.dumps(report["jump_statistics_pre_vs_post"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
