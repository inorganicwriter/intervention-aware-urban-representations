"""Freeze the reviewed station-containing grids into an immutable work queue.

This is data preparation only. It does not select controls or estimate effects.
"""

from __future__ import annotations

import json

import pandas as pd

from urban_intervention.data.paths import (
    CAUSAL_DIR,
    COUNTERFACTUAL_QUEUE,
    OUTCOME_FAMILY_QUEUE,
    OUTPUT_HOUSING_DID_DIR,
    PROJECT_ROOT,
    TREATMENT_UNIT_LIST,
)

SOURCE = OUTPUT_HOUSING_DID_DIR / "treated_grid_registry.parquet"


def main() -> None:
    source = pd.read_parquet(SOURCE)
    treated = source.loc[source["analysis_treated_grid"]].copy()
    if treated[["city_key", "grid_id"]].duplicated().any():
        raise RuntimeError("reviewed treatment grids are not unique")
    treated["opening_date"] = pd.to_datetime(treated["opening_date"])
    treated["opening_month"] = treated["opening_date"].dt.to_period("M").astype(str)
    treated = treated.sort_values(
        ["opening_date", "city_key", "grid_id", "station_event_id"],
        kind="stable",
    ).reset_index(drop=True)
    treated.insert(0, "treatment_order", range(1, len(treated) + 1))
    columns = [
        "treatment_order",
        "city_key",
        "grid_id",
        "station_event_id",
        "station_name",
        "wgs84_lon",
        "wgs84_lat",
        "opening_date",
        "opening_month",
        "competing_event_ids",
        "post_treatment_censor_year",
    ]
    treatment_list = treated[columns].copy()
    queue = treatment_list[
        ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]
    ].copy()
    queue["status"] = "pending"
    queue["selected_method"] = pd.NA
    queue["selected_control_grid_id"] = pd.NA
    queue["failure_reason"] = pd.NA
    family_queue = treatment_list[
        ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]
    ].merge(
        pd.DataFrame({"outcome_family": ["housing", "poi", "viirs", "population"]}),
        how="cross",
    )
    family_queue["status"] = "pending"
    family_queue["selected_method"] = pd.NA
    family_queue["failure_reason"] = pd.NA

    CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
    treatment_list.to_parquet(TREATMENT_UNIT_LIST, index=False)
    treatment_list.to_csv(CAUSAL_DIR / "treatment_unit_list.csv", index=False, encoding="utf-8-sig")
    queue.to_csv(COUNTERFACTUAL_QUEUE, index=False, encoding="utf-8-sig")
    family_queue.to_csv(OUTCOME_FAMILY_QUEUE, index=False, encoding="utf-8-sig")
    metadata = {
        "schema": "treatment_unit_list_v1",
        "source": str(SOURCE.relative_to(PROJECT_ROOT)),
        "ordering": ["opening_date", "city_key", "grid_id", "station_event_id"],
        "treatment_units": len(treatment_list),
        "unique_city_grid": not treatment_list[["city_key", "grid_id"]].duplicated().any(),
        "initial_queue_status": "pending",
        "outcome_family_queue_rows": len(family_queue),
    }
    (CAUSAL_DIR / "treatment_unit_list_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
