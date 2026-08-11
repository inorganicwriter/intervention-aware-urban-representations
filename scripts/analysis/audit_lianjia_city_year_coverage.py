"""Audit the actual Lianjia transaction coverage by city and calendar year.

The purpose is to distinguish a missing acquisition period from an observed
city-year in which a residential grid has no recorded transaction.  The audit
does not decide the final coverage threshold; it publishes counts and boundary
flags so that the risk-set rule can be frozen before outcome effects are seen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_HOUSING_DID_PREFLIGHT_DIR,
    STAGING_LIANJIA_TRANSACTIONS_DIR,
)

STAGING = STAGING_LIANJIA_TRANSACTIONS_DIR
OUT_DIR = OUTPUT_HOUSING_DID_PREFLIGHT_DIR


def main() -> int:
    paths = sorted(STAGING.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(STAGING)

    frames = []
    file_rows = []
    for path in paths:
        d = pd.read_parquet(
            path,
            columns=[
                "source_record_id",
                "source_file",
                "city_key",
                "year",
                "year_valid",
                "price_valid",
                "community_valid",
                "is_valid",
            ],
        )
        file_rows.append(
            {
                "staging_file": str(path.relative_to(ROOT)),
                "rows": len(d),
                "valid_rows": int(d["is_valid"].fillna(False).sum()),
                "source_files": int(d["source_file"].nunique(dropna=True)),
            }
        )
        frames.append(d)

    raw = pd.concat(frames, ignore_index=True)
    duplicate_ids = int(raw.duplicated("source_record_id").sum())
    raw = raw.drop_duplicates("source_record_id", keep="first")
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    usable = raw[
        raw["is_valid"].fillna(False) & raw["year"].between(2010, 2025) & raw["city_key"].notna()
    ].copy()
    usable["year"] = usable["year"].astype(int)

    coverage = (
        usable.groupby(["city_key", "year"], as_index=False)
        .agg(
            valid_transactions=("source_record_id", "nunique"),
            source_files=("source_file", "nunique"),
        )
        .sort_values(["city_key", "year"])
    )
    city_stats = coverage.groupby("city_key")["valid_transactions"].agg(
        city_year_min="min", city_year_median="median", city_year_max="max"
    )
    coverage = coverage.join(city_stats, on="city_key")
    coverage["share_of_city_year_median"] = (
        coverage["valid_transactions"] / coverage["city_year_median"]
    )
    coverage["at_least_30_transactions"] = coverage["valid_transactions"] >= 30
    coverage["at_least_100_transactions"] = coverage["valid_transactions"] >= 100
    coverage["at_least_20pct_city_median"] = coverage["share_of_city_year_median"] >= 0.20
    coverage["provisional_core_coverage"] = (
        coverage["at_least_100_transactions"] & coverage["at_least_20pct_city_median"]
    )

    years = list(range(2010, 2026))
    city_rows = []
    for city, d in coverage.groupby("city_key", sort=True):
        observed = sorted(map(int, d["year"]))
        core = sorted(map(int, d.loc[d["provisional_core_coverage"], "year"]))
        city_rows.append(
            {
                "city_key": city,
                "first_observed_year": min(observed),
                "last_observed_year": max(observed),
                "observed_year_count": len(observed),
                "observed_years": ";".join(map(str, observed)),
                "internal_missing_years": ";".join(
                    map(
                        str,
                        [
                            y
                            for y in years
                            if min(observed) <= y <= max(observed) and y not in observed
                        ],
                    )
                ),
                "provisional_core_year_count": len(core),
                "provisional_core_years": ";".join(map(str, core)),
                "valid_transactions": int(d["valid_transactions"].sum()),
                "minimum_city_year_transactions": int(d["valid_transactions"].min()),
                "median_city_year_transactions": float(d["valid_transactions"].median()),
                "maximum_city_year_transactions": int(d["valid_transactions"].max()),
            }
        )
    city_summary = pd.DataFrame(city_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(OUT_DIR / "lianjia_city_year_coverage.csv", index=False, encoding="utf-8-sig")
    city_summary.to_csv(
        OUT_DIR / "lianjia_city_coverage_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(file_rows).to_csv(
        OUT_DIR / "lianjia_staging_file_audit.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "staging_files": len(paths),
        "unique_source_records": int(len(raw)),
        "duplicate_source_record_ids_across_staging_files": duplicate_ids,
        "valid_2010_2025_transactions": int(len(usable)),
        "cities": int(usable["city_key"].nunique()),
        "observed_city_years": int(len(coverage)),
        "provisional_core_city_years": int(coverage["provisional_core_coverage"].sum()),
        "provisional_rule": "at least 100 valid transactions and at least 20% of the city's observed-year median",
        "note": "The provisional rule is an audit flag, not yet the frozen analysis-coverage rule.",
    }
    (OUT_DIR / "lianjia_coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
