"""Build city × year HPI labels from NBS 70-city monthly HPI data.

The NBS 70-city HPI is a city-level monthly index (上月=100 / 上年同月=100 /
上年同期=100), NOT a grid-level price. We use it as a city-level outcome
variable. It must remain city-level and be joined to grid panels only on
``city_key, year``. For event-study / DiD on metro openings it is useful for:
  - cross-city heterogeneity (how city-level response differs by metro year)
  - sanity-checking grid-level outcomes (e.g. VIIRS) against city aggregates

Output:
  data/active/labels/housing/city_hpi/hpi_city_yearly.parquet
      city_key, year, housing_type, area_class, mom_avg, yoy_avg, ytd_avg,
      hpi_index

Usage:
    python scripts/labels/build_hpi_label.py
    python scripts/labels/build_hpi_label.py --city beijing
    python scripts/labels/build_hpi_label.py --housing new --area total
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import numpy as np
import pandas as pd

from urban_intervention.config.project import ACTIVE_CITIES
from urban_intervention.data.paths import HPI_LABEL_DIR, STAGING_DIR

HPI_CSV = STAGING_DIR / "nbs_hpi" / "monthly.csv"
OUT_DIR = HPI_LABEL_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_hpi() -> pd.DataFrame:
    if not HPI_CSV.exists():
        raise FileNotFoundError(
            f"Missing {HPI_CSV}. Run scripts/collection/nbs_70city_hpi_fetcher.py first."
        )
    df = pd.read_csv(HPI_CSV, encoding="utf-8-sig")
    # Keep only rows mapped to one of our 44 cities
    df = df[df["city_key"].notna()].copy()
    df["city_key"] = df["city_key"].astype(str)
    # Coerce year/month to int (NaN -> dropped) so the ym string is well-formed.
    df = df.dropna(subset=["year", "month"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["ym"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    return df


def monthly_to_yearly(hpi: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly HPI to yearly means per (city, housing, area).

    For yoy/ytd we use the December value (=year-on-year for the year end);
    if December is missing or NaN, fall back to the latest available month
    within that year. mom uses the yearly mean of monthly moms.
    """

    def _dec_or_last(sub_df: pd.DataFrame, col: str) -> float:
        """Prefer December value; fall back to last month's; NaN otherwise."""
        if col not in sub_df.columns or sub_df.empty:
            return np.nan
        dec_rows = sub_df[sub_df["month"] == 12]
        if not dec_rows.empty:
            v = dec_rows[col].iloc[-1]
            if pd.notna(v):
                return float(v)
        non_null = sub_df[col].dropna()
        if not non_null.empty:
            return float(non_null.iloc[-1])
        return np.nan

    hpi_sorted = hpi.sort_values(["city_key", "year", "housing_type", "area_class", "month"])

    rows = []
    keys = ["city_key", "year", "housing_type", "area_class"]
    for (ck, yr, ht, ac), sub in hpi_sorted.groupby(keys):
        rows.append(
            {
                "city_key": ck,
                "year": yr,
                "housing_type": ht,
                "area_class": ac,
                "mom_avg": float(sub["mom"].mean()) if "mom" in sub else np.nan,
                "yoy_december": _dec_or_last(sub, "yoy"),
                "yoy_avg": float(sub["yoy"].mean()) if "yoy" in sub else np.nan,
                "ytd_december": _dec_or_last(sub, "ytd"),
                "n_months": int(sub["mom"].count()) if "mom" in sub else 0,
            }
        )
    yearly = pd.DataFrame(rows)
    return yearly


def compute_chained_index(yearly: pd.DataFrame, base_year: int = 2022) -> pd.DataFrame:
    """Build a synthetic price index per (city, housing, area).

    The index is anchored at 100 in ``base_year``.  For every other year
    with data, a *year-over-year multiplier* is computed and the index is
    chain-multiplied forward (and divided backward).

    Multiplier source (in priority order):
      1. ``yoy_december`` — NBS' "上年同月=100" index for December (or the
         last available month of that year).  This is the canonical YoY
         index and the statistically correct choice for an annual chain.
      2. ``(1 + (mom_avg - 100)/100) ** 12`` — fallback when yoy is missing:
         compound the average monthly mom (上月=100) over 12 months.  This
         is an approximation (assumes constant monthly growth) but is far
         more accurate than the previous arithmetic-mean approach which
         understated annual changes by an order of magnitude.

    Only years with data are chained; gaps remain NaN.
    """

    def _year_multiplier(r: pd.Series) -> float:
        """Return the YoY multiplier for a yearly row, or NaN."""
        yoy = r.get("yoy_december")
        if pd.notna(yoy) and yoy > 0:
            return float(yoy) / 100.0
        mom_avg = r.get("mom_avg")
        if pd.notna(mom_avg) and mom_avg > 0:
            # Compound the (avg) monthly rate over 12 months.
            monthly_factor = 1.0 + (float(mom_avg) - 100.0) / 100.0
            return monthly_factor**12
        return np.nan

    out = []
    for (_ck, _ht, _ac), grp in yearly.groupby(["city_key", "housing_type", "area_class"]):
        grp = grp.sort_values("year").copy()
        idx = pd.Series(index=grp["year"].values, dtype=float)
        if base_year in grp["year"].values:
            idx[base_year] = 100.0

        # Precompute the per-row multiplier once.
        grp["_mult"] = grp.apply(_year_multiplier, axis=1).astype(float)

        # Forward chain: base_year -> ... -> max year
        prev = idx.get(base_year)
        for _, r in grp[grp["year"] > base_year].iterrows():
            if prev is not None and pd.notna(r["_mult"]):
                v = prev * r["_mult"]
                idx[r["year"]] = v
                prev = v
        # Backward chain: base_year -> ... -> min year.
        # idx[Y-1] = idx[Y] / mult[Y] where mult[Y] = price_Y / price_{Y-1};
        # the multiplier belongs to the year we are stepping *from* (prev_year),
        # not the target year we are stepping *to*.
        mult_by_year = dict(zip(grp["year"].values, grp["_mult"].values, strict=False))
        prev = idx.get(base_year)
        prev_year = base_year
        for _, r in grp[grp["year"] < base_year].sort_values("year", ascending=False).iterrows():
            if prev is not None:
                step_mult = mult_by_year.get(prev_year)
                if pd.notna(step_mult) and step_mult > 0:
                    v = prev / step_mult
                    idx[r["year"]] = v
                    prev = v
                    prev_year = int(r["year"])
        grp["hpi_index"] = idx.values
        grp = grp.drop(columns=["_mult"])
        out.append(grp)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="all")
    p.add_argument("--housing", default=None, help="Filter: new | secondhand (default both)")
    p.add_argument(
        "--area", default=None, help="Filter: total | small | medium | large (default all)"
    )
    args = p.parse_args()

    hpi = load_hpi()
    if args.housing:
        hpi = hpi[hpi.housing_type == args.housing]
    if args.area:
        hpi = hpi[hpi.area_class == args.area]
    if hpi.empty:
        print("No HPI rows after filtering. Exiting.")
        return 1
    print(
        f"Loaded HPI: {len(hpi)} rows, {hpi.city_key.nunique()} cities, "
        f"{hpi.ym.min()} ~ {hpi.ym.max()}"
    )

    yearly = monthly_to_yearly(hpi)
    print(f"Yearly aggregation: {len(yearly)} rows, {yearly.year.min()} ~ {yearly.year.max()}")
    yearly = compute_chained_index(yearly, base_year=2022)
    print(f"Chained index added; {yearly.hpi_index.notna().sum()}/{len(yearly)} have index")

    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    yearly = yearly[yearly.city_key.isin(cities)]

    # Save city-level yearly table (one row per city-year-housing-area)
    yearly_path = OUT_DIR / "hpi_city_yearly.parquet"
    yearly.to_parquet(yearly_path, index=False)
    print(f"\nSaved city-level yearly HPI -> {yearly_path}")

    # Quick sanity print
    print("\n=== Sample: Beijing new total, yearly ===")
    s = yearly[
        (yearly.city_key == "beijing")
        & (yearly.housing_type == "new")
        & (yearly.area_class == "total")
    ].sort_values("year")
    if not s.empty:
        print(s[["year", "mom_avg", "yoy_december", "hpi_index"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
