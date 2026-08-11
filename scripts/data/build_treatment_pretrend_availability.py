"""Count unique pre-treatment time points in each matching family."""

from __future__ import annotations

import pandas as pd

from urban_intervention.data.paths import (
    TREATMENT_PREREND_AVAILABILITY,
    TREATMENT_UNIT_LIST,
    housing_monthly_panel_path,
    poi_annual_path,
    population_data_path,
    viirs_annual_path,
)


def main() -> None:
    treatments = pd.read_parquet(TREATMENT_UNIT_LIST)
    products: list[pd.DataFrame] = []
    for city, group in treatments.groupby("city_key"):
        base = group[["treatment_order", "grid_id", "opening_date"]].copy()
        base["opening_year"] = base["opening_date"].dt.year
        result = base.set_index("treatment_order")
        housing_path = housing_monthly_panel_path(city)
        if housing_path.exists():
            housing = pd.read_parquet(housing_path, columns=["grid_id", "observed_month"])
            merged = base.merge(housing, on="grid_id", how="left")
            offset = (
                (merged["observed_month"].dt.year - merged["opening_date"].dt.year) * 12
                + merged["observed_month"].dt.month
                - merged["opening_date"].dt.month
            )
            counts = (
                merged.loc[offset.le(-13)]
                .groupby("treatment_order")["observed_month"]
                .nunique()
                .rename("housing_pre_periods")
            )
            result = result.join(counts)
        annual_specs = {
            "poi": poi_annual_path(city),
            "viirs": viirs_annual_path(city),
            "population": population_data_path(city),
        }
        for family, path in annual_specs.items():
            if not path.exists():
                continue
            frame = pd.read_parquet(path, columns=["grid_id", "year"])
            merged = base.merge(frame, on="grid_id", how="left")
            counts = (
                merged.loc[merged["year"].lt(merged["opening_year"])]
                .groupby("treatment_order")["year"]
                .nunique()
                .rename(f"{family}_pre_periods")
            )
            result = result.join(counts)
        products.append(result.reset_index())
    availability = pd.concat(products, ignore_index=True).fillna(0)
    period_columns = [column for column in availability if column.endswith("_pre_periods")]
    availability[period_columns] = availability[period_columns].astype(int)
    availability["families_with_pretrend"] = availability[period_columns].ge(2).sum(axis=1)
    availability["families_with_holdout_support"] = availability[period_columns].ge(3).sum(axis=1)
    availability = availability.sort_values("treatment_order")
    availability.to_parquet(TREATMENT_PREREND_AVAILABILITY, index=False, compression="zstd")
    print(
        availability[availability["families_with_pretrend"].gt(0)].head(10).to_string(index=False)
    )
    print(availability["families_with_pretrend"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
