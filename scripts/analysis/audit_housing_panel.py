"""Audit the canonical housing panel and treated-grid monthly support."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.data.paths import (  # noqa: E402
    CURATED_DIR,
    OUTPUT_HOUSING_DID_DIR,
    OUTPUT_HOUSING_PANEL_DIR,
    PANEL_HOUSING_MONTHLY_DIR,
    PANEL_HOUSING_QUARTERLY_DIR,
    PANEL_HOUSING_YEARLY_DIR,
)

OBS_DIR = CURATED_DIR / "housing" / "housing_observations"
MONTH_DIR = PANEL_HOUSING_MONTHLY_DIR
QUARTER_DIR = PANEL_HOUSING_QUARTERLY_DIR
YEAR_DIR = PANEL_HOUSING_YEARLY_DIR
TREATMENT_PATH = OUTPUT_HOUSING_DID_DIR / "treated_grid_registry.parquet"
REPORT_DIR = OUTPUT_HOUSING_PANEL_DIR


def period_key_duplicates(path: Path, period: str) -> tuple[int, int, int]:
    frame = pd.read_parquet(path)
    duplicate = frame[["city_key", "grid_id", period]].duplicated().sum()
    invalid_price = (
        ~np.isfinite(frame["price_source_balanced_cny_m2"])
        | frame["price_source_balanced_cny_m2"].le(0)
    ).sum()
    return len(frame), int(duplicate), int(invalid_price)


def audit_city(city: str) -> tuple[dict, list[dict], list[dict]]:
    obs_path = OBS_DIR / f"{city}.parquet"
    observations = pd.read_parquet(obs_path)
    duplicate_rows = []
    for (source, duplicate_class), count in (
        observations[observations["duplicate_class"].ne("")]
        .groupby(["source_id", "duplicate_class"])
        .size()
        .items()
    ):
        duplicate_rows.append(
            {
                "city_key": city,
                "source_id": source,
                "duplicate_class": duplicate_class,
                "rows": int(count),
            }
        )
    coordinate_rows = []
    for source, group in observations.groupby("source_id"):
        coordinate_rows.append(
            {
                "city_key": city,
                "source_id": source,
                "rows": int(len(group)),
                "rows_with_coordinates": int(
                    group[["longitude_wgs84", "latitude_wgs84"]].notna().all(axis=1).sum()
                ),
                "rows_with_grid": int(group["grid_id"].ne("").sum()),
                "monthly_eligible_rows": int(group["analysis_eligible_month"].sum()),
                "quarterly_eligible_rows": int(group["analysis_eligible_quarter"].sum()),
                "annual_eligible_rows": int(group["analysis_eligible_year"].sum()),
            }
        )

    month_rows, month_duplicates, month_invalid = period_key_duplicates(
        MONTH_DIR / f"{city}.parquet", "observed_month"
    )
    quarter_rows, quarter_duplicates, quarter_invalid = period_key_duplicates(
        QUARTER_DIR / f"{city}.parquet", "observed_quarter"
    )
    year_rows, year_duplicates, year_invalid = period_key_duplicates(
        YEAR_DIR / f"{city}.parquet", "observed_year"
    )
    year_only_month_leakage = int(
        (observations["time_precision"].eq("year") & observations["observed_month"].notna()).sum()
    )
    quarter_month_leakage = int(
        (
            observations["time_precision"].eq("quarter") & observations["observed_month"].notna()
        ).sum()
    )
    eligible_without_grid = int(
        (observations["analysis_eligible_month"] & observations["grid_id"].eq("")).sum()
    )
    return (
        {
            "city_key": city,
            "observation_rows": int(len(observations)),
            "unique_observation_ids": int(observations["observation_id"].nunique()),
            "duplicate_observation_ids": int(observations["observation_id"].duplicated().sum()),
            "canonical_rows": int(observations["canonical_for_aggregation"].sum()),
            "monthly_eligible_rows": int(observations["analysis_eligible_month"].sum()),
            "year_only_month_leakage": year_only_month_leakage,
            "quarter_month_leakage": quarter_month_leakage,
            "eligible_without_grid": eligible_without_grid,
            "monthly_panel_rows": month_rows,
            "monthly_duplicate_keys": month_duplicates,
            "monthly_invalid_prices": month_invalid,
            "quarterly_panel_rows": quarter_rows,
            "quarterly_duplicate_keys": quarter_duplicates,
            "quarterly_invalid_prices": quarter_invalid,
            "annual_panel_rows": year_rows,
            "annual_duplicate_keys": year_duplicates,
            "annual_invalid_prices": year_invalid,
        },
        coordinate_rows,
        duplicate_rows,
    )


def relative_month(month: pd.Series, opening: pd.Series) -> pd.Series:
    return ((month.dt.year - opening.dt.year) * 12 + month.dt.month - opening.dt.month).astype(
        "Int16"
    )


def treated_grid_support(cities: list[str]) -> tuple[pd.DataFrame, dict]:
    treatment = pd.read_parquet(TREATMENT_PATH)
    treatment = treatment[treatment["analysis_treated_grid"]].copy()
    treatment["opening_month"] = (
        pd.to_datetime(treatment["opening_date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    treatment = treatment[treatment["opening_month"].notna()].copy()
    if treatment[["city_key", "grid_id"]].duplicated().any():
        raise RuntimeError("analysis_treated_grid is not unique by city/grid")
    rows = []
    for city in cities:
        treated = treatment[treatment["city_key"] == city].copy()
        if treated.empty:
            continue
        panel = pd.read_parquet(MONTH_DIR / f"{city}.parquet")
        panel = panel[panel["grid_id"].isin(set(treated["grid_id"]))].copy()
        joined = treated[
            [
                "city_key",
                "grid_id",
                "station_event_id",
                "station_name",
                "opening_date",
                "opening_month",
            ]
        ].merge(panel, on=["city_key", "grid_id"], how="left")
        joined["relative_month"] = relative_month(
            pd.to_datetime(joined["observed_month"], errors="coerce"),
            pd.to_datetime(joined["opening_month"], errors="coerce"),
        )
        for record in treated.itertuples(index=False):
            group = joined[joined["grid_id"] == record.grid_id].copy()
            relative = pd.to_numeric(group["relative_month"], errors="coerce")
            pre12 = group[relative.between(-12, -1)]
            post12 = group[relative.between(1, 12)]
            pre24 = group[relative.between(-24, -1)]
            post24 = group[relative.between(1, 24)]
            pre36 = group[relative.between(-36, -1)]
            post36 = group[relative.between(1, 36)]
            observations_48 = int(
                pd.to_numeric(
                    pd.concat([pre24["n_observations"], post24["n_observations"]]), errors="coerce"
                )
                .fillna(0)
                .sum()
            )
            rows.append(
                {
                    "city_key": city,
                    "grid_id": record.grid_id,
                    "station_event_id": record.station_event_id,
                    "station_name": record.station_name,
                    "opening_date": record.opening_date,
                    "pre12_months": int(pre12["observed_month"].nunique()),
                    "post12_months": int(post12["observed_month"].nunique()),
                    "pre24_months": int(pre24["observed_month"].nunique()),
                    "post24_months": int(post24["observed_month"].nunique()),
                    "pre36_months": int(pre36["observed_month"].nunique()),
                    "post36_months": int(post36["observed_month"].nunique()),
                    "observations_pre_post_24": observations_48,
                    "any_pre_post_24": bool(len(pre24) and len(post24)),
                    "minimal_6m_each_12": bool(
                        pre12["observed_month"].nunique() >= 6
                        and post12["observed_month"].nunique() >= 6
                    ),
                    "baseline_12pre6post": bool(
                        pre24["observed_month"].nunique() >= 12
                        and post12["observed_month"].nunique() >= 6
                        and observations_48 >= 24
                    ),
                    "balanced_12_each_24": bool(
                        pre24["observed_month"].nunique() >= 12
                        and post24["observed_month"].nunique() >= 12
                        and observations_48 >= 36
                    ),
                    "strict_18pre12post": bool(
                        pre36["observed_month"].nunique() >= 18
                        and post36["observed_month"].nunique() >= 12
                        and observations_48 >= 48
                    ),
                }
            )
    result = pd.DataFrame(rows)
    criteria = [
        "any_pre_post_24",
        "minimal_6m_each_12",
        "baseline_12pre6post",
        "balanced_12_each_24",
        "strict_18pre12post",
    ]
    summary = {
        "spatial_treated_grids": int(len(result)),
        "cities": int(result["city_key"].nunique()),
        "criteria_totals": {criterion: int(result[criterion].sum()) for criterion in criteria},
        "criteria_city_counts": {
            criterion: int(result.loc[result[criterion], "city_key"].nunique())
            for criterion in criteria
        },
        "by_city": [
            {
                "city_key": city,
                "spatial_treated_grids": int(len(group)),
                **{criterion: int(group[criterion].sum()) for criterion in criteria},
            }
            for city, group in result.groupby("city_key", sort=True)
        ],
    }
    return result, summary


def main() -> int:
    cities = sorted(path.stem for path in OBS_DIR.glob("*.parquet"))
    city_audits = []
    coordinate_rows = []
    duplicate_rows = []
    for city in cities:
        audit, coordinate, duplicate = audit_city(city)
        city_audits.append(audit)
        coordinate_rows.extend(coordinate)
        duplicate_rows.extend(duplicate)
        print(
            f"{city}: obs={audit['observation_rows']:,}, month={audit['monthly_panel_rows']:,}, "
            f"duplicate_keys={audit['monthly_duplicate_keys']}",
            flush=True,
        )
    treated, treated_summary = treated_grid_support(cities)
    validation_fields = [
        "duplicate_observation_ids",
        "year_only_month_leakage",
        "quarter_month_leakage",
        "eligible_without_grid",
        "monthly_duplicate_keys",
        "monthly_invalid_prices",
        "quarterly_duplicate_keys",
        "quarterly_invalid_prices",
        "annual_duplicate_keys",
        "annual_invalid_prices",
    ]
    failures = {field: int(sum(row[field] for row in city_audits)) for field in validation_fields}
    summary = {
        "schema": "housing_panel_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "cities": len(cities),
        "observation_rows": int(sum(row["observation_rows"] for row in city_audits)),
        "monthly_panel_rows": int(sum(row["monthly_panel_rows"] for row in city_audits)),
        "quarterly_panel_rows": int(sum(row["quarterly_panel_rows"] for row in city_audits)),
        "annual_panel_rows": int(sum(row["annual_panel_rows"] for row in city_audits)),
        "validation_failures": failures,
        "validation_passed": all(value == 0 for value in failures.values()),
        "treated_grid_support": treated_summary,
        "city_audits": city_audits,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "row_closure.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(coordinate_rows).to_csv(
        REPORT_DIR / "coordinate_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(duplicate_rows).to_csv(
        REPORT_DIR / "duplicate_audit.csv", index=False, encoding="utf-8-sig"
    )
    treated.to_csv(REPORT_DIR / "treated_grid_window_audit.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "treated_grid_window_summary.json").write_text(
        json.dumps(treated_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "validation_passed": summary["validation_passed"],
                "validation_failures": failures,
                "treated_grid_support": treated_summary["criteria_totals"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
