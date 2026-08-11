"""Audit overlap between the supplemental Beijing Figshare file and Lianjia.

The Figshare source has month-level dates only, so this report is deliberately
conservative: it reports exact rounded matches and same-location-month support
without deleting or fusing any row.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_HOUSING_ACQUISITION_DIR,
    STAGING_HOUSING_STANDARDIZED_DIR,
    STAGING_LIANJIA_TRANSACTIONS_DIR,
)

FIGSHARE = (
    STAGING_HOUSING_STANDARDIZED_DIR
    / "figshare_14398907_v1_beijing_2020_transactions"
    / "housing_observations.parquet"
)
LIANJIA_ROOT = STAGING_LIANJIA_TRANSACTIONS_DIR
OUTPUT = OUTPUT_HOUSING_ACQUISITION_DIR / "figshare_14398907_beijing_overlap_audit.json"


def match_key(frame: pd.DataFrame, *, date_column: str) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["period"] = pd.to_datetime(frame[date_column], errors="coerce").dt.to_period("M")
    result["lon_5dp"] = pd.to_numeric(frame["lon"], errors="coerce").round(5)
    result["lat_5dp"] = pd.to_numeric(frame["lat"], errors="coerce").round(5)
    result["price_round"] = pd.to_numeric(frame["unit_price_cny_m2"], errors="coerce").round(0)
    return result


def main() -> int:
    supplemental = pd.read_parquet(
        FIGSHARE,
        columns=[
            "source_record_id",
            "deal_date",
            "unit_price_cny_m2",
            "lon",
            "lat",
        ],
    )
    lianjia_frames = []
    for path in sorted(LIANJIA_ROOT.glob("*/beijing.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[
                "source_record_id",
                "deal_date",
                "unit_price_cny_m2",
                "lon",
                "lat",
                "is_valid",
            ],
        )
        lianjia_frames.append(frame[frame["is_valid"]].copy())
    lianjia = pd.concat(lianjia_frames, ignore_index=True)
    lianjia = lianjia.drop_duplicates("source_record_id", keep="first")

    supplemental_keys = match_key(supplemental, date_column="deal_date")
    lianjia_keys = match_key(lianjia, date_column="deal_date")
    exact_columns = ["period", "lon_5dp", "lat_5dp", "price_round"]
    spatial_month_columns = ["period", "lon_5dp", "lat_5dp"]
    exact_reference = pd.MultiIndex.from_frame(
        lianjia_keys[exact_columns].dropna().drop_duplicates()
    )
    spatial_month_reference = pd.MultiIndex.from_frame(
        lianjia_keys[spatial_month_columns].dropna().drop_duplicates()
    )
    exact = pd.MultiIndex.from_frame(supplemental_keys[exact_columns]).isin(exact_reference)
    same_location_month = pd.MultiIndex.from_frame(supplemental_keys[spatial_month_columns]).isin(
        spatial_month_reference
    )
    nearest_distances: list[np.ndarray] = []
    nearest_price_errors: list[np.ndarray] = []
    lon_scale = 111_320 * math.cos(math.radians(40.0))
    lat_scale = 110_540
    for period, supplemental_positions in supplemental_keys.groupby("period").groups.items():
        reference_positions = lianjia_keys.index[lianjia_keys["period"].eq(period)]
        if not len(reference_positions):
            nearest_distances.append(np.full(len(supplemental_positions), np.nan, dtype=float))
            nearest_price_errors.append(np.full(len(supplemental_positions), np.nan, dtype=float))
            continue
        reference_xy = np.column_stack(
            [
                lianjia.loc[reference_positions, "lon"].to_numpy() * lon_scale,
                lianjia.loc[reference_positions, "lat"].to_numpy() * lat_scale,
            ]
        )
        query_xy = np.column_stack(
            [
                supplemental.loc[supplemental_positions, "lon"].to_numpy() * lon_scale,
                supplemental.loc[supplemental_positions, "lat"].to_numpy() * lat_scale,
            ]
        )
        distances, nearest = cKDTree(reference_xy).query(query_xy, k=1)
        reference_prices = lianjia.loc[reference_positions, "unit_price_cny_m2"].to_numpy()[nearest]
        query_prices = supplemental.loc[supplemental_positions, "unit_price_cny_m2"].to_numpy()
        price_error = np.abs(query_prices - reference_prices) / reference_prices
        nearest_distances.append(distances)
        nearest_price_errors.append(price_error)
    distance = np.concatenate(nearest_distances)
    price_error = np.concatenate(nearest_price_errors)

    report = {
        "schema": "figshare_14398907_beijing_overlap_audit_v1",
        "supplemental_rows": int(len(supplemental)),
        "purchased_lianjia_valid_unique_rows": int(len(lianjia)),
        "supplemental_period_first": str(supplemental_keys["period"].min()),
        "supplemental_period_last": str(supplemental_keys["period"].max()),
        "exact_rounded_matches": int(exact.sum()),
        "same_location_month_matches": int(same_location_month.sum()),
        "no_same_location_month_support": int((~same_location_month).sum()),
        "exact_match_share": float(exact.mean()),
        "same_location_month_share": float(same_location_month.mean()),
        "nearest_same_month_distance_m": {
            "median": float(np.nanmedian(distance)),
            "p90": float(np.nanpercentile(distance, 90)),
            "within_50m": int(np.nansum(distance <= 50)),
            "within_100m": int(np.nansum(distance <= 100)),
            "within_250m": int(np.nansum(distance <= 250)),
            "within_500m": int(np.nansum(distance <= 500)),
        },
        "nearest_same_month_joint_support": {
            "within_500m_and_price_within_1pct": int(
                np.nansum((distance <= 500) & (price_error <= 0.01))
            ),
            "within_500m_and_price_within_5pct": int(
                np.nansum((distance <= 500) & (price_error <= 0.05))
            ),
        },
        "key_definition": {
            "exact": "month + longitude/latitude rounded to 5 decimals + unit price rounded to 1 CNY/m2",
            "same_location_month": "month + longitude/latitude rounded to 5 decimals",
        },
        "decision": "retain source separately; deduplicate only during source fusion",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
