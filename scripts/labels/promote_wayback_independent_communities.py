"""Promote unresolved Wayback names to stable independent communities.

The user-approved rule is source preserving: an unmatched historical name is
not discarded and is not forced onto an existing community.  If no reliable
coordinate exists, the community remains explicitly unlocated and therefore
cannot receive a grid weight until a later geolocation pass.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from urban_intervention.data.paths import (
    COMMUNITY_REGISTRY,
    COMMUNITY_SOURCE_CROSSWALK,
    OUTPUT_HOUSING_FUSION_DIR,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = COMMUNITY_REGISTRY
CROSSWALK_PATH = COMMUNITY_SOURCE_CROSSWALK
REPORT_PATH = OUTPUT_HOUSING_FUSION_DIR / "wayback_independent_communities.json"


def main() -> int:
    registry = pd.read_parquet(REGISTRY_PATH)
    crosswalk = pd.read_parquet(CROSSWALK_PATH)
    mask = (
        crosswalk["source"].fillna("").str.startswith("wayback_") & crosswalk["community_id"].isna()
    )
    unresolved = crosswalk[mask].copy()
    unresolved = unresolved[unresolved["proposed_community_id"].fillna("").ne("")]
    if unresolved.empty:
        print("No unresolved Wayback names to promote")
        return 0

    grouped = (
        unresolved.sort_values(["city_key", "source", "source_name"])
        .groupby("proposed_community_id", as_index=False)
        .agg(
            city_key=("city_key", "first"),
            canonical_name=("source_name", "first"),
            normalized_name=("normalized_name", "first"),
            aliases=("source_name", lambda values: ";".join(sorted(set(map(str, values))))),
            source_count=("source", "nunique"),
        )
        .rename(columns={"proposed_community_id": "community_id"})
    )
    grouped = grouped[~grouped["community_id"].isin(registry["community_id"])].copy()

    defaults = {
        "district": "",
        "centroid_lon": np.nan,
        "centroid_lat": np.nan,
        "has_anjuke_boundary": False,
        "anjuke_source_id": "",
        "transaction_count": 0,
        "first_year": np.nan,
        "last_year": np.nan,
        "aoi_source": "unlocated_wayback_independent",
        "aoi_quality": "E",
        "match_status": "unlocated_wayback_independent",
        "beijing_aoi_id": None,
        "aoi_id": None,
        "boundary_area_m2": np.nan,
        "aoi_bridge_admitted": False,
        "aoi_grid_coverage_share": 0.0,
    }
    for column, value in defaults.items():
        grouped[column] = value
    grouped = grouped.drop(columns=["source_count"])
    for column in registry.columns:
        if column not in grouped.columns:
            grouped[column] = np.nan
    grouped = grouped[registry.columns]
    updated_registry = pd.concat([registry, grouped], ignore_index=True)
    if (
        updated_registry["community_id"].isna().any()
        or updated_registry["community_id"].duplicated().any()
    ):
        raise RuntimeError("Promoted community registry violates primary-key uniqueness")

    crosswalk.loc[mask, "community_id"] = crosswalk.loc[mask, "proposed_community_id"]
    crosswalk.loc[mask, "match_method"] = "independent_source_community"
    crosswalk.loc[mask, "match_score"] = np.nan
    crosswalk.loc[mask, "match_distance_m"] = np.nan
    crosswalk.loc[mask, "proposed_community_id"] = ""
    broken = set(crosswalk["community_id"].dropna()) - set(updated_registry["community_id"])
    if broken:
        raise RuntimeError(f"Crosswalk has {len(broken)} broken community references")

    registry_temp = REGISTRY_PATH.with_suffix(".parquet.tmp")
    crosswalk_temp = CROSSWALK_PATH.with_suffix(".parquet.tmp")
    updated_registry.to_parquet(registry_temp, index=False)
    crosswalk.to_parquet(crosswalk_temp, index=False)
    registry_temp.replace(REGISTRY_PATH)
    crosswalk_temp.replace(CROSSWALK_PATH)

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "unresolved_crosswalk_rows_promoted": int(mask.sum()),
        "independent_communities_added": int(len(grouped)),
        "registry_rows_before": int(len(registry)),
        "registry_rows_after": int(len(updated_registry)),
        "located_primary_aoi_communities": int(updated_registry["aoi_bridge_admitted"].sum()),
        "promotion_rule": "stable independent ID; no forced spatial match",
        "raw_files_modified": False,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
