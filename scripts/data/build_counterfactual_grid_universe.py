"""Build the all-grid universe for unit-level matching and GSC.

No grid is selected as a control here. The output preserves every reference
grid and exposes treatment/contamination flags for later, event-specific audit.
"""

from __future__ import annotations

import json

import pandas as pd

from urban_intervention.data.paths import (
    GRID_UNIVERSE_BY_CITY,
    GRID_UNIVERSE_DIR,
    GRID_UNIVERSE_METADATA,
    OUTPUT_HOUSING_DID_DIR,
    TREATMENT_UNIT_LIST,
)

EXPOSURE_DIR = OUTPUT_HOUSING_DID_DIR / "grid_spatial_exposure"


def main() -> None:
    treatments = pd.read_parquet(TREATMENT_UNIT_LIST, columns=["city_key", "grid_id"])
    treated_keys = set(map(tuple, treatments.itertuples(index=False, name=None)))
    GRID_UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    city_rows: list[dict[str, object]] = []
    found_treatments = 0
    for path in sorted(EXPOSURE_DIR.glob("*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[
                "city_key",
                "grid_id",
                "nearest_station_polygon_distance_m",
                "nearest_station_event_id",
                "nearest_station_opening_year",
                "station_containing_grid",
                "primary_spatial_exclusion_reason",
            ],
        )
        if frame[["city_key", "grid_id"]].duplicated().any():
            raise RuntimeError(f"duplicate city-grid keys in {path}")
        keys = list(map(tuple, frame[["city_key", "grid_id"]].itertuples(index=False, name=None)))
        frame["is_experimental_grid"] = [key in treated_keys for key in keys]
        frame["is_nonexperimental_grid"] = ~frame["is_experimental_grid"]
        frame["known_station_contamination"] = frame["station_containing_grid"] | frame[
            "nearest_station_polygon_distance_m"
        ].lt(1_000)
        found_treatments += int(frame["is_experimental_grid"].sum())
        city = str(frame["city_key"].iloc[0])
        frame.to_parquet(GRID_UNIVERSE_DIR / f"{city}.parquet", index=False, compression="zstd")
        city_rows.append(
            {
                "city_key": city,
                "all_grids": len(frame),
                "experimental_grids": int(frame["is_experimental_grid"].sum()),
                "nonexperimental_grids": int(frame["is_nonexperimental_grid"].sum()),
                "known_station_contaminated_nonexperimental_grids": int(
                    (frame["is_nonexperimental_grid"] & frame["known_station_contamination"]).sum()
                ),
            }
        )
    if found_treatments != len(treatments):
        raise RuntimeError(
            f"only {found_treatments} of {len(treatments)} treatment grids found in universe"
        )
    summary = pd.DataFrame(city_rows).sort_values("city_key")
    summary.to_csv(GRID_UNIVERSE_BY_CITY, index=False, encoding="utf-8-sig")
    metadata = {
        "schema": "counterfactual_grid_universe",
        "cities": len(summary),
        "all_grids": int(summary["all_grids"].sum()),
        "experimental_grids": int(summary["experimental_grids"].sum()),
        "nonexperimental_grids": int(summary["nonexperimental_grids"].sum()),
        "selection_performed": False,
        "note": "known_station_contamination is an audit flag, not a silent prefilter",
    }
    GRID_UNIVERSE_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
