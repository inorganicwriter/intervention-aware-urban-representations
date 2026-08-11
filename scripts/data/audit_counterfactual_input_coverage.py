"""Audit four frozen pre-treatment input families without selecting controls."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urban_intervention.data.paths import (
    COUNTERFACTUAL_COVERAGE,
    TREATMENT_UNIT_LIST,
    housing_monthly_panel_path,
    poi_annual_path,
    population_data_path,
    viirs_annual_path,
)


def coverage(path: Path, grid_col: str, time_col: str) -> dict[str, object]:
    if not path.exists():
        return {"rows": 0, "grids": 0, "time_min": pd.NA, "time_max": pd.NA}
    frame = pd.read_parquet(path, columns=[grid_col, time_col])
    return {
        "rows": len(frame),
        "grids": frame[grid_col].nunique(),
        "time_min": frame[time_col].min(),
        "time_max": frame[time_col].max(),
    }


def main() -> None:
    treatments = pd.read_parquet(TREATMENT_UNIT_LIST, columns=["city_key", "grid_id"])
    rows: list[dict[str, object]] = []
    for city in sorted(treatments["city_key"].unique()):
        city_treated = set(treatments.loc[treatments["city_key"].eq(city), "grid_id"])
        inputs = {
            "housing": (
                housing_monthly_panel_path(city),
                "grid_id",
                "observed_month",
            ),
            "poi": (
                poi_annual_path(city),
                "grid_id",
                "year",
            ),
            "viirs": (
                viirs_annual_path(city),
                "grid_id",
                "year",
            ),
            "pop": (
                population_data_path(city),
                "grid_id",
                "year",
            ),
        }
        row: dict[str, object] = {
            "city_key": city,
            "treatment_grids": len(city_treated),
        }
        for label, (path, grid_col, time_col) in inputs.items():
            stats = coverage(path, grid_col, time_col)
            row.update({f"{label}_{key}": value for key, value in stats.items()})
            if path.exists():
                observed_grids = set(pd.read_parquet(path, columns=[grid_col])[grid_col])
                row[f"{label}_treated_grids_observed"] = len(city_treated & observed_grids)
            else:
                row[f"{label}_treated_grids_observed"] = 0
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(COUNTERFACTUAL_COVERAGE, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
