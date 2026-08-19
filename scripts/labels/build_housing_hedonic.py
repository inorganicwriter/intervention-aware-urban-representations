"""Build hedonic quality-adjusted housing prices for the Lianjia 22 cities.

Transaction-level, per-city hedonic regression (within-community fixed
effects plus year x quarter dummies):

    log(unit_price_cny_m2) ~ area + area^2 + age + age^2 + age_missing
        + bedrooms + floor-group + orientation + decoration + building-type
        + elevator + community FE + year x quarter FE

The quality-adjusted price of a transaction is exp(log price - X_attrs * b),
i.e. only dwelling attributes are removed; time and community effects stay in
the adjusted level so post-opening price dynamics remain visible.

Output (all under outputs/causal_labels/housing_hedonic/):

- {city}_monthly.parquet: grid_id x observed_month -> median adjusted price,
  n_transactions, n_adjusted (transactions used by the regression)
- diagnostics.json: per-city sample sizes, R2, attribute completeness,
  grid-month transaction-count distribution (feeds the minimum-count rule)

Usage:
    python scripts/labels/build_housing_hedonic.py [--cities all]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.config.project import CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    STAGING_LIANJIA_TRANSACTIONS_DIR,
    grid_path,
)

OUTPUT_DIR = ROOT / "outputs" / "causal_labels" / "housing_hedonic"

LIANJIA_CITIES = [
    "beijing", "changzhou", "chongqing", "dongguan", "foshan", "guangzhou",
    "hangzhou", "jinan", "jinhua", "luoyang", "nanjing", "nantong", "ningbo",
    "qingdao", "shaoxing", "shenzhen", "suzhou", "taizhou", "wenzhou", "wuxi",
    "xuzhou", "zhengzhou",
]

FLOOR_GROUP = re.compile(r"^(低|中|高|顶|底)层")
MIN_TRANSACTIONS_PER_CITY = 500


def load_city_transactions(city: str) -> pd.DataFrame:
    paths = sorted(STAGING_LIANJIA_TRANSACTIONS_DIR.glob(f"*/{city}.parquet"))
    frames = [pd.read_parquet(path) for path in paths if path.stat().st_size > 10_000]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame["is_valid"].fillna(False)].copy()
    frame = frame.drop_duplicates(subset=["source_record_id"], keep="first")
    price = pd.to_numeric(frame.get("unit_price_cny_m2"), errors="coerce")
    area = pd.to_numeric(frame.get("building_area_m2"), errors="coerce")
    usable = (
        price.gt(0)
        & price.lt(1_000_000)
        & area.gt(0)
        & area.lt(2_000)
        & pd.to_numeric(frame.get("lon"), errors="coerce").between(70, 140)
        & pd.to_numeric(frame.get("lat"), errors="coerce").between(10, 60)
    )
    frame = frame.loc[usable].copy()
    frame["_price"] = price.loc[frame.index]
    frame["_area"] = area.loc[frame.index]
    frame["_lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame["_lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    return frame


def map_to_grid(frame: pd.DataFrame, city: str) -> pd.DataFrame:
    grids = pd.read_parquet(
        grid_path(city), columns=["grid_id", "row", "col", "centroid_lon", "centroid_lat"]
    )
    transformer = Transformer.from_crs("EPSG:4326", CITIES[city]["projected_crs"], always_xy=True)
    sample = grids.iloc[:: max(1, len(grids) // 2_000)].copy()
    sample_x, sample_y = transformer.transform(sample["centroid_lon"], sample["centroid_lat"])
    origin_x = float(np.median(sample_x - (sample["col"].to_numpy() + 0.5) * 500.0))
    origin_y = float(np.median(sample_y - (sample["row"].to_numpy() + 0.5) * 500.0))
    residual = max(
        float(np.max(np.abs(sample_x - (origin_x + (sample["col"].to_numpy() + 0.5) * 500.0)))),
        float(np.max(np.abs(sample_y - (origin_y + (sample["row"].to_numpy() + 0.5) * 500.0)))),
    )
    if residual > 1.0:
        raise RuntimeError(f"{city} reference grid is not a regular projected 500 m grid")
    x, y = transformer.transform(frame["_lon"].to_numpy(), frame["_lat"].to_numpy())
    cols = np.floor((x - origin_x) / 500.0).astype(np.int64)
    rows = np.floor((y - origin_y) / 500.0).astype(np.int64)
    candidates = np.array(
        [f"g{row:05d}x{col:05d}" for row, col in zip(rows, cols, strict=False)], dtype=object
    )
    retained = set(grids["grid_id"].astype(str))
    in_retained = pd.Series(candidates).isin(retained).to_numpy()
    frame["grid_id"] = np.where(in_retained, candidates, "")
    return frame


def build_feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (design matrix, attribute column names, log price vector).

    Community fixed effects are absorbed by a within transformation; year x
    quarter dummies stay in the design.  Attribute columns are the leading
    block returned separately so only they are removed from the adjusted
    price.
    """
    area = frame["_area"].to_numpy(dtype=float)
    year = pd.to_datetime(frame["deal_date"], errors="coerce").dt.year.to_numpy(dtype=float)
    built = pd.to_numeric(frame.get("built_year_mid"), errors="coerce").to_numpy(dtype=float)
    age = np.where(np.isfinite(built), np.clip(year - built, 0, 80), 0.0)
    age_missing = (~np.isfinite(built)).astype(float)
    bedrooms = pd.to_numeric(frame.get("bedroom_count"), errors="coerce").fillna(0).to_numpy()
    elevator = (
        frame.get("elevator", pd.Series("", index=frame.index)).astype(str).str.strip()
    )

    attr_blocks: list[np.ndarray] = [
        area,
        area**2,
        age,
        age**2,
        age_missing,
        bedrooms,
        elevator.eq("有").to_numpy(dtype=float),
    ]
    attr_names = ["area", "area2", "age", "age2", "age_missing", "bedrooms", "elevator"]

    floor_raw = frame.get("floor_raw", pd.Series("", index=frame.index)).astype(str)
    floor_group = floor_raw.str.extract(FLOOR_GROUP)[0].fillna("未知")
    for level in ("低", "中", "高", "顶", "底"):
        dummy = floor_group.eq(level).to_numpy(dtype=float)
        attr_blocks.append(dummy)
        attr_names.append(f"floor_{level}")

    for column, name in (
        ("orientation", "orient"),
        ("decoration", "decor"),
        ("building_type", "btype"),
    ):
        values = frame.get(column, pd.Series("", index=frame.index)).astype(str).str.strip()
        for value in sorted(values.dropna().unique()):
            if not value or value in {"", "nan", "未知", "其他"}:
                continue
            attr_blocks.append(values.eq(value).to_numpy(dtype=float))
            attr_names.append(f"{name}_{value}")

    quarter = pd.to_datetime(frame["deal_date"], errors="coerce").dt.to_period("Q")
    time_dummies: list[np.ndarray] = []
    time_names: list[str] = []
    for period in sorted(quarter.dropna().unique()):
        time_dummies.append(quarter.eq(period).to_numpy(dtype=float))
        time_names.append(f"tq_{period}")
    design = np.column_stack(attr_blocks + time_dummies)
    log_price = np.log(frame["_price"].to_numpy(dtype=float))
    return design, attr_names, time_names, log_price


def within_transform(matrix: np.ndarray, keys: np.ndarray) -> np.ndarray:
    keys_flat = keys.astype(str)
    means = pd.DataFrame(matrix).groupby(keys_flat).transform("mean").to_numpy()
    return matrix - means


def fit_city(city: str) -> tuple[pd.DataFrame | None, dict[str, object]]:
    frame = load_city_transactions(city)
    if frame.empty:
        return None, {"city": city, "transactions": 0, "skipped": "no_data"}
    frame = map_to_grid(frame, city)
    frame = frame.loc[frame["grid_id"].ne("")].copy()
    deal_date = pd.to_datetime(frame["deal_date"], errors="coerce")
    frame = frame.loc[deal_date.notna()].copy()
    if len(frame) < MIN_TRANSACTIONS_PER_CITY:
        return None, {"city": city, "transactions": int(len(frame)), "skipped": "too_few"}

    community = frame.get("community_key_exact", pd.Series("", index=frame.index)).astype(str)
    community = community.where(community.ne(""), "unmapped")
    design, attr_names, time_names, log_price = build_feature_matrix(frame)
    if len(attr_names) < 8 or len(time_names) < 4:
        return None, {"city": city, "transactions": int(len(frame)), "skipped": "sparse_design"}

    keys = np.asarray(community)
    # Within transformation removes each community's time-invariant level:
    # singleton communities become all-zero rows and constant-within-community
    # columns become zero vectors.  Drop both before the least-squares fit.
    group_size = pd.Series(keys).groupby(keys).transform("size").to_numpy()
    keep_rows = group_size >= 2
    design_fit = design[keep_rows]
    log_price_fit = log_price[keep_rows]
    keys_fit = keys[keep_rows]
    design_w = within_transform(design_fit, keys_fit)
    y_w = log_price_fit - (
        pd.Series(log_price_fit).groupby(keys_fit).transform("mean").to_numpy()
    )
    col_norm = np.linalg.norm(design_w, axis=0)
    keep_cols = col_norm > 1e-8
    design_w = design_w[:, keep_cols]
    # Pseudo-inverse with an explicit rcond: within-transformed designs can
    # carry a handful of numerically singular directions (e.g. communities
    # whose whole history lies in one quarter); those directions carry no
    # identifiable information and are dropped instead of failing the fit.
    beta = np.linalg.pinv(design_w, rcond=1e-10) @ y_w

    beta_full = np.zeros(design_fit.shape[1])
    beta_full[keep_cols] = beta
    attr_idx = np.arange(len(attr_names))
    attr_effect = design_fit[:, attr_idx] @ beta_full[attr_idx]
    adjusted = np.exp(log_price_fit - attr_effect)
    r2 = 1.0 - float(np.sum((y_w - design_w @ beta) ** 2) / max(np.sum(y_w**2), 1e-12))

    out = frame.loc[keep_rows, ["city_key", "grid_id", "deal_date"]].copy()
    out["adjusted_price"] = adjusted
    out["n_attributes"] = len(attr_names)
    out["observed_month"] = pd.to_datetime(out["deal_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    panel = (
        out.groupby(["city_key", "grid_id", "observed_month"], as_index=False)
        .agg(
            adjusted_price_median=("adjusted_price", "median"),
            n_transactions=("adjusted_price", "size"),
        )
    )
    counts = panel["n_transactions"]
    diagnostics: dict[str, object] = {
        "city": city,
        "transactions": int(len(frame)),
        "grid_months": int(len(panel)),
        "r2": round(r2, 4),
        "rank": int(design_w.shape[1]),
        "n_attributes": len(attr_names),
        "n_time_dummies": len(time_names),
        "attribute_columns": attr_names,
        "attr_coefficients": {name: round(float(b), 4) for name, b in zip(attr_names, beta[: len(attr_names)], strict=False)},
        "counts_ge1": round(float((counts >= 1).mean()), 4),
        "counts_ge2": round(float((counts >= 2).mean()), 4),
        "counts_ge3": round(float((counts >= 3).mean()), 4),
        "counts_ge5": round(float((counts >= 5).mean()), 4),
        "counts_ge10": round(float((counts >= 10).mean()), 4),
        "counts_median": int(counts.median()),
        "monthly_share_of_grid_months": round(
            float(panel["grid_id"].nunique() / max(frame["grid_id"].nunique(), 1)), 4
        ),
    }
    return panel, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cities", default="all", help="Comma-separated city keys or 'all'"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    cities = LIANJIA_CITIES if args.cities == "all" else [c.strip() for c in args.cities.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, object]] = []
    for city in cities:
        panel, diag = fit_city(city)
        if panel is not None:
            panel.to_parquet(
                args.output_dir / f"{city}_monthly.parquet",
                index=False,
                compression="zstd",
            )
            print(f"{city}: {diag['transactions']:,} tx, R2={diag['r2']}, "
                  f"{diag['grid_months']:,} grid-months, counts_ge1={diag['counts_ge1']}")
        else:
            print(f"{city}: skipped ({diag.get('skipped')})")
        diagnostics.append(diag)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "specification": "hedonic_main_v1",
        "cities": diagnostics,
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
