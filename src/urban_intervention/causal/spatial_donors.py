"""DDR-001 spatial treatment and donor-feasibility audit.

This module deliberately does not read housing outcomes. It maps canonical
station events into the fixed 500 m grid, computes station-to-grid-polygon
distances in each city's metric CRS, and publishes spatial donor eligibility.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import STRtree, from_wkt, points

from urban_intervention.config.project import ACTIVE_CITIES, get_city_config
from urban_intervention.data.paths import (
    OUTPUT_DIR,
    RESOLVED_STATION_EVENTS,
    grid_path,
)
from urban_intervention.utils import require_columns

GRID_COLUMNS = ("grid_id", "geometry_wkt")
STATION_COLUMNS = (
    "city_key",
    "station_event_id",
    "canonical_station_name",
    "wgs84_lon",
    "wgs84_lat",
    "opening_year",
)
OPTIONAL_STATION_DEFAULTS = {
    "opening_date": pd.NA,
    "date_precision": "unknown",
    "resolution_status": "unreviewed_source_event",
    "resolution_grid_id": "",
    "primary_design_excluded": False,
    "primary_design_exclusion_reason": "",
    "competing_event_ids": "",
    "post_treatment_censor_year": pd.NA,
}


@dataclass(frozen=True)
class SpatialDonorSpec:
    """Frozen DDR-001 spatial audit configuration."""

    primary_exclusion_m: int = 1_000
    sensitivity_exclusion_m: tuple[int, ...] = (1_500, 2_000)
    analysis_start_year: int = 2010
    analysis_end_year: int = 2025
    distance_metric: str = "point_to_polygon_minimum"

    @property
    def radii_m(self) -> tuple[int, ...]:
        # Sensitivity radii are "additional" radii beyond the primary; the
        # primary is always included, so drop any sensitivity duplicate and
        # return radii in increasing order regardless of input order.
        extra = tuple(r for r in self.sensitivity_exclusion_m if r != self.primary_exclusion_m)
        return tuple(sorted((self.primary_exclusion_m, *extra)))

    def validate(self) -> None:
        if self.primary_exclusion_m <= 0:
            raise ValueError("primary_exclusion_m must be positive")
        if len(set(self.radii_m)) != len(self.radii_m):
            raise ValueError("spatial exclusion radii must be unique")
        if any(r <= 0 for r in self.radii_m):
            raise ValueError("spatial exclusion radii must be positive")
        if self.analysis_start_year > self.analysis_end_year:
            raise ValueError("analysis_start_year must not exceed analysis_end_year")
        if self.distance_metric != "point_to_polygon_minimum":
            raise ValueError("DDR-001 requires point_to_polygon_minimum distance")


def station_value(station: object, name: str, default: object) -> object:
    value = getattr(station, name, default)
    return default if pd.isna(value) else value


def project_city_geometries(
    grids: pd.DataFrame,
    stations: pd.DataFrame,
    projected_crs: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grid polygons and station points in a city-specific metric CRS."""

    require_columns(grids, GRID_COLUMNS, "grid frame")
    require_columns(stations, STATION_COLUMNS, "station frame")
    if grids.empty:
        raise ValueError("grid frame is empty")
    if stations.empty:
        raise ValueError("station frame is empty")
    if grids["grid_id"].astype(str).duplicated().any():
        raise ValueError("grid frame contains duplicate grid_id values")
    if stations["station_event_id"].astype(str).duplicated().any():
        raise ValueError("station frame contains duplicate station_event_id values")

    polygons_wgs84 = from_wkt(grids["geometry_wkt"].astype(str).to_numpy())
    if bool(np.any(shapely.is_missing(polygons_wgs84))):
        raise ValueError("grid frame contains missing geometry")
    station_points_wgs84 = points(
        pd.to_numeric(stations["wgs84_lon"], errors="raise").to_numpy(float),
        pd.to_numeric(stations["wgs84_lat"], errors="raise").to_numpy(float),
    )
    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    polygons_metric = shapely.transform(polygons_wgs84, transformer.transform, interleaved=False)
    station_points_metric = shapely.transform(
        station_points_wgs84, transformer.transform, interleaved=False
    )
    return polygons_metric, station_points_metric


def compute_spatial_exposure(
    city_key: str,
    grids: pd.DataFrame,
    stations: pd.DataFrame,
    polygons_metric: np.ndarray,
    station_points_metric: np.ndarray,
    spec: SpatialDonorSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute grid exposure and station-to-grid mapping audit for one city."""

    spec.validate()
    if len(grids) != len(polygons_metric):
        raise ValueError("grid geometry count does not match grid frame")
    if len(stations) != len(station_points_metric):
        raise ValueError("station geometry count does not match station frame")
    opening_year = pd.to_numeric(stations["opening_year"], errors="coerce")
    if opening_year.isna().any():
        raise ValueError(
            f"station frame contains {int(opening_year.isna().sum())} missing opening_year values"
        )

    nearest_indices, nearest_distance = STRtree(station_points_metric).query_nearest(
        polygons_metric,
        all_matches=False,
        return_distance=True,
    )
    grid_idx = nearest_indices[0]
    station_idx = nearest_indices[1]
    if len(grid_idx) != len(grids) or not np.array_equal(np.sort(grid_idx), np.arange(len(grids))):
        raise ValueError("nearest-station query did not return exactly one row per grid")

    order = np.argsort(grid_idx)
    station_idx = station_idx[order]
    nearest_distance = nearest_distance[order]
    nearest = stations.iloc[station_idx].reset_index(drop=True)

    station_input_idx, mapped_grid_idx = STRtree(polygons_metric).query(
        station_points_metric,
        predicate="covered_by",
    )
    station_to_grids: dict[int, list[int]] = {i: [] for i in range(len(stations))}
    grid_to_stations: dict[int, list[int]] = {i: [] for i in range(len(grids))}
    for si, gi in zip(station_input_idx.tolist(), mapped_grid_idx.tolist(), strict=False):
        station_to_grids[int(si)].append(int(gi))
        grid_to_stations[int(gi)].append(int(si))

    grid_ids = grids["grid_id"].astype(str).to_numpy()
    mapping_rows: list[dict] = []
    for si, station in enumerate(stations.itertuples(index=False)):
        candidates = sorted(set(station_to_grids[si]))
        if not candidates:
            status = "outside_reference_grid"
        elif len(candidates) == 1:
            status = "unique"
        else:
            status = "boundary_multiple"
        mapping_rows.append(
            {
                "city_key": city_key,
                "station_event_id": str(station.station_event_id),
                "station_name": str(station.canonical_station_name),
                "wgs84_lon": float(station.wgs84_lon),
                "wgs84_lat": float(station.wgs84_lat),
                "opening_year": int(station.opening_year),
                "opening_date": str(station.opening_date),
                "resolution_status": str(
                    station_value(station, "resolution_status", "unreviewed_source_event")
                ),
                "resolution_grid_id": str(station_value(station, "resolution_grid_id", "")),
                "primary_design_excluded": bool(
                    station_value(station, "primary_design_excluded", False)
                ),
                "primary_design_exclusion_reason": str(
                    station_value(station, "primary_design_exclusion_reason", "")
                ),
                "competing_event_ids": str(station_value(station, "competing_event_ids", "")),
                "post_treatment_censor_year": station_value(
                    station, "post_treatment_censor_year", pd.NA
                ),
                "candidate_grid_count": len(candidates),
                "candidate_grid_ids": ";".join(grid_ids[candidates]) if candidates else "",
                "mapping_status": status,
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    mapping["post_treatment_censor_year"] = pd.to_numeric(
        mapping["post_treatment_censor_year"], errors="coerce"
    ).astype("Int64")

    station_counts: list[int] = []
    station_ids: list[str] = []
    station_names: list[str] = []
    station_years: list[str] = []
    for gi in range(len(grids)):
        inside = sorted(set(grid_to_stations[gi]))
        station_counts.append(len(inside))
        station_ids.append(
            ";".join(stations.iloc[inside]["station_event_id"].astype(str)) if inside else ""
        )
        station_names.append(
            ";".join(stations.iloc[inside]["canonical_station_name"].astype(str)) if inside else ""
        )
        station_years.append(
            ";".join(stations.iloc[inside]["opening_year"].astype(int).astype(str))
            if inside
            else ""
        )

    exposure = pd.DataFrame(
        {
            "city_key": city_key,
            "grid_id": grid_ids,
            "nearest_station_polygon_distance_m": nearest_distance.astype(float),
            "nearest_station_event_id": nearest["station_event_id"].astype(str).to_numpy(),
            "nearest_station_name": nearest["canonical_station_name"].astype(str).to_numpy(),
            "nearest_station_opening_year": nearest["opening_year"].astype(int).to_numpy(),
            "station_event_count_in_grid": station_counts,
            "station_event_ids_in_grid": station_ids,
            "station_event_names_in_grid": station_names,
            "station_event_years_in_grid": station_years,
        }
    )
    exposure["station_containing_grid"] = exposure["station_event_count_in_grid"] > 0
    exposure["unique_station_event_in_grid"] = exposure["station_event_count_in_grid"] == 1
    for radius in spec.radii_m:
        exposure[f"spatial_donor_eligible_{radius}m"] = (
            exposure["nearest_station_polygon_distance_m"] >= radius
        )
    exposure["primary_spatial_exclusion_reason"] = np.where(
        exposure["station_containing_grid"],
        "contains_station",
        np.where(
            exposure[f"spatial_donor_eligible_{spec.primary_exclusion_m}m"],
            "eligible_spatial_donor",
            f"spatially_contaminated_{spec.primary_exclusion_m}m",
        ),
    )
    return exposure, mapping


def build_treated_grid_registry(
    exposure: pd.DataFrame,
    station_mapping: pd.DataFrame,
    spec: SpatialDonorSpec,
) -> pd.DataFrame:
    """Return one row per station-containing grid with mapping audit flags."""

    mapped = station_mapping[station_mapping["mapping_status"] == "unique"].copy()
    mapped = mapped.rename(columns={"candidate_grid_ids": "grid_id"})
    columns = [
        "city_key",
        "grid_id",
        "station_event_id",
        "station_name",
        "wgs84_lon",
        "wgs84_lat",
        "opening_year",
        "opening_date",
        "mapping_status",
        "resolution_status",
        "resolution_grid_id",
        "primary_design_excluded",
        "primary_design_exclusion_reason",
        "competing_event_ids",
        "post_treatment_censor_year",
    ]
    mapped = mapped[columns]
    counts = mapped.groupby(["city_key", "grid_id"])["station_event_id"].transform("size")
    mapped["station_events_in_grid"] = counts.astype(int)
    mapped["unique_treatment_event"] = mapped["station_events_in_grid"] == 1
    mapped["within_analysis_window"] = mapped["opening_year"].between(
        spec.analysis_start_year, spec.analysis_end_year
    )
    mapped["analysis_treated_grid"] = (
        mapped["unique_treatment_event"]
        & mapped["within_analysis_window"]
        & ~mapped["primary_design_excluded"]
    )
    mapped["treatment_exclusion_reason"] = "eligible_treated_grid"
    outside_window = ~mapped["within_analysis_window"]
    mapped.loc[outside_window, "treatment_exclusion_reason"] = "outside_analysis_window"
    multiple = ~mapped["unique_treatment_event"]
    mapped.loc[multiple, "treatment_exclusion_reason"] = "multiple_station_events_in_grid"
    design_excluded = mapped["primary_design_excluded"]
    mapped.loc[design_excluded, "treatment_exclusion_reason"] = mapped.loc[
        design_excluded, "primary_design_exclusion_reason"
    ].replace("", "resolved_primary_design_exclusion")

    # Assert that station-containing exposure rows agree with the mapping table.
    expected = set(exposure.loc[exposure["station_containing_grid"], "grid_id"].astype(str))
    observed = set(mapped["grid_id"].astype(str))
    unresolved: set[str] = set()
    unresolved_values = station_mapping.loc[
        station_mapping["mapping_status"] != "unique", "candidate_grid_ids"
    ].astype(str)
    for value in unresolved_values:
        unresolved.update(grid_id for grid_id in value.split(";") if grid_id)
    if expected != observed | unresolved:
        raise ValueError("station-containing grids do not agree with station mapping")
    return mapped.sort_values(
        ["city_key", "opening_year", "grid_id", "station_event_id"]
    ).reset_index(drop=True)


def summarize_city(
    city_key: str,
    exposure: pd.DataFrame,
    mapping: pd.DataFrame,
    treated: pd.DataFrame,
    projected_crs: str,
    spec: SpatialDonorSpec,
) -> dict:
    row: dict[str, object] = {
        "city_key": city_key,
        "projected_crs": projected_crs,
        "distance_metric": spec.distance_metric,
        "primary_exclusion_m": spec.primary_exclusion_m,
        "reference_grids": int(len(exposure)),
        "canonical_station_events": int(len(mapping)),
        "unique_station_grid_mappings": int((mapping["mapping_status"] == "unique").sum()),
        "boundary_multiple_station_mappings": int(
            (mapping["mapping_status"] == "boundary_multiple").sum()
        ),
        "stations_outside_reference_grid": int(
            (mapping["mapping_status"] == "outside_reference_grid").sum()
        ),
        "station_containing_grids": int(exposure["station_containing_grid"].sum()),
        "grids_with_multiple_station_events": int(
            (exposure["station_event_count_in_grid"] > 1).sum()
        ),
        "resolved_design_exclusion_grids": int(
            treated.loc[treated["primary_design_excluded"], "grid_id"].nunique()
        ),
        "analysis_treated_grids": int(treated["analysis_treated_grid"].sum()),
    }
    for radius in spec.radii_m:
        eligible = exposure[f"spatial_donor_eligible_{radius}m"]
        row[f"spatial_donors_{radius}m"] = int(eligible.sum())
        row[f"spatial_donor_share_{radius}m"] = float(eligible.mean())
    return row


def summarize_cohorts(
    treated: pd.DataFrame,
    city_summary: dict,
    spec: SpatialDonorSpec,
) -> pd.DataFrame:
    eligible = treated[treated["analysis_treated_grid"]].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["city_key", "cohort_year", "treated_grids"])
    summary = (
        eligible.groupby(["city_key", "opening_year"], as_index=False)
        .agg(treated_grids=("grid_id", "nunique"))
        .rename(columns={"opening_year": "cohort_year"})
    )
    # The never-treated spatial donor set is time-invariant: the same city
    # total applies to every cohort row.  The column name says "total" so it
    # cannot be mistaken for cohort-specific availability (summing across
    # cohort rows would double-count).
    for radius in spec.radii_m:
        summary[f"total_same_city_spatial_donors_{radius}m"] = int(
            city_summary[f"spatial_donors_{radius}m"]
        )
    return summary


def build_data_quality_issues(
    station_mapping: pd.DataFrame,
    treated: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact review queue for unresolved treatment geography."""

    issue_columns = [
        "issue_type",
        "city_key",
        "grid_id",
        "station_event_ids",
        "station_names",
        "station_wgs84_lons",
        "station_wgs84_lats",
        "opening_years",
        "recommended_action",
    ]
    issues: list[pd.DataFrame] = []

    unresolved = station_mapping[station_mapping["mapping_status"] != "unique"].copy()
    if not unresolved.empty:
        unresolved_issue = pd.DataFrame(
            {
                "issue_type": unresolved["mapping_status"],
                "city_key": unresolved["city_key"],
                "grid_id": unresolved["candidate_grid_ids"],
                "station_event_ids": unresolved["station_event_id"],
                "station_names": unresolved["station_name"],
                "station_wgs84_lons": unresolved["wgs84_lon"].map(
                    lambda value: f"{float(value):.7f}"
                ),
                "station_wgs84_lats": unresolved["wgs84_lat"].map(
                    lambda value: f"{float(value):.7f}"
                ),
                "opening_years": unresolved["opening_year"].astype(str),
                "recommended_action": "repair_station_or_grid_reference_before_admission",
            }
        )
        issues.append(unresolved_issue)

    resolved_group = treated.groupby(["city_key", "grid_id"])["primary_design_excluded"].transform(
        "all"
    )
    multiple = treated[~treated["unique_treatment_event"] & ~resolved_group].copy()
    if not multiple.empty:
        multiple_issue = multiple.groupby(["city_key", "grid_id"], as_index=False).agg(
            station_event_ids=("station_event_id", lambda values: ";".join(values.astype(str))),
            station_names=("station_name", lambda values: ";".join(values.astype(str))),
            station_wgs84_lons=(
                "wgs84_lon",
                lambda values: ";".join(f"{float(value):.7f}" for value in values),
            ),
            station_wgs84_lats=(
                "wgs84_lat",
                lambda values: ";".join(f"{float(value):.7f}" for value in values),
            ),
            opening_years=("opening_year", lambda values: ";".join(values.astype(str))),
        )
        multiple_issue.insert(0, "issue_type", "multiple_station_events_in_grid")
        multiple_issue["recommended_action"] = (
            "resolve_alias_or_true_multi_station_exposure_before_admission"
        )
        issues.append(multiple_issue)

    if not issues:
        return pd.DataFrame(columns=issue_columns)
    return (
        pd.concat(issues, ignore_index=True)[issue_columns]
        .sort_values(["issue_type", "city_key", "grid_id"])
        .reset_index(drop=True)
    )


def audit_city(
    city_key: str,
    grids: pd.DataFrame,
    stations: pd.DataFrame,
    projected_crs: str,
    spec: SpatialDonorSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Run the complete outcome-free spatial audit for one city."""

    polygons, station_points = project_city_geometries(grids, stations, projected_crs)
    exposure, mapping = compute_spatial_exposure(
        city_key, grids, stations, polygons, station_points, spec
    )
    treated = build_treated_grid_registry(exposure, mapping, spec)
    summary = summarize_city(city_key, exposure, mapping, treated, projected_crs, spec)
    cohorts = summarize_cohorts(treated, summary, spec)
    return exposure, mapping, treated, summary, cohorts


def run_audit(
    cities: Iterable[str],
    station_path: Path,
    output_dir: Path,
    spec: SpatialDonorSpec,
) -> dict:
    """Run and write the outcome-free DDR-001 audit."""

    spec.validate()
    stations_all = pd.read_parquet(station_path)
    require_columns(stations_all, STATION_COLUMNS, "canonical station events")
    for column, default in OPTIONAL_STATION_DEFAULTS.items():
        if column not in stations_all.columns:
            stations_all[column] = default
    if stations_all["station_event_id"].astype(str).duplicated().any():
        raise ValueError("canonical station events contain duplicate station_event_id values")

    output_dir.mkdir(parents=True, exist_ok=True)
    exposure_dir = output_dir / "grid_spatial_exposure"
    exposure_dir.mkdir(parents=True, exist_ok=True)
    mappings: list[pd.DataFrame] = []
    treated_rows: list[pd.DataFrame] = []
    city_rows: list[dict] = []
    cohort_rows: list[pd.DataFrame] = []

    for city_key in cities:
        city_stations = stations_all[stations_all["city_key"] == city_key].copy()
        if city_stations.empty:
            raise ValueError(f"{city_key}: no canonical station events")
        grids = pd.read_parquet(grid_path(city_key), columns=list(GRID_COLUMNS))
        config = get_city_config(city_key)
        projected_crs = str(config["projected_crs"])
        exposure, mapping, treated, summary, cohorts = audit_city(
            city_key, grids, city_stations, projected_crs, spec
        )
        exposure.to_parquet(exposure_dir / f"{city_key}.parquet", index=False)
        mappings.append(mapping)
        treated_rows.append(treated)
        city_rows.append(summary)
        if not cohorts.empty:
            cohort_rows.append(cohorts)
        print(
            f"{city_key}: grids={len(exposure):,}, stations={len(mapping):,}, "
            f"treated={summary['analysis_treated_grids']:,}, "
            f"donors@{spec.primary_exclusion_m}m={summary[f'spatial_donors_{spec.primary_exclusion_m}m']:,}",
            flush=True,
        )

    if not mappings:
        raise ValueError("No cities produced spatial donor mappings — check city list")
    mapping_all = pd.concat(mappings, ignore_index=True)
    treated_all = pd.concat(treated_rows, ignore_index=True)
    city_summary = pd.DataFrame(city_rows).sort_values("city_key")
    cohort_summary = pd.concat(cohort_rows, ignore_index=True) if cohort_rows else pd.DataFrame()
    mapping_all.to_csv(output_dir / "station_grid_mapping_audit.csv", index=False)
    treated_all.to_parquet(output_dir / "treated_grid_registry.parquet", index=False)
    city_summary.to_csv(output_dir / "city_spatial_donor_summary.csv", index=False)
    cohort_summary.to_csv(output_dir / "cohort_spatial_donor_summary.csv", index=False)
    data_quality_issues = build_data_quality_issues(mapping_all, treated_all)
    data_quality_issues.to_csv(output_dir / "spatial_data_quality_issues.csv", index=False)

    summary = {
        "design": "DDR-001",
        "reads_housing_outcomes": False,
        "station_path": str(station_path),
        "output_dir": str(output_dir),
        "cities": int(len(city_summary)),
        "spec": asdict(spec),
        "reference_grids": int(city_summary["reference_grids"].sum()),
        "canonical_station_events": int(city_summary["canonical_station_events"].sum()),
        "analysis_treated_grids": int(city_summary["analysis_treated_grids"].sum()),
        "ambiguous_station_mappings": int(
            city_summary["boundary_multiple_station_mappings"].sum()
            + city_summary["stations_outside_reference_grid"].sum()
        ),
        "grids_with_multiple_station_events": int(
            city_summary["grids_with_multiple_station_events"].sum()
        ),
        "resolved_design_exclusion_grids": int(
            city_summary["resolved_design_exclusion_grids"].sum()
        ),
        "station_mapping_status": {
            str(key): int(value)
            for key, value in mapping_all["mapping_status"].value_counts().items()
        },
        "treatment_exclusion_reason": {
            str(key): int(value)
            for key, value in treated_all["treatment_exclusion_reason"].value_counts().items()
        },
        "spatial_data_quality_issues": int(len(data_quality_issues)),
    }
    for radius in spec.radii_m:
        summary[f"spatial_donors_{radius}m"] = int(city_summary[f"spatial_donors_{radius}m"].sum())
    (output_dir / "spatial_feasibility_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DDR-001 station-grid mapping and spatial donor feasibility"
    )
    parser.add_argument("--city", choices=["all", *ACTIVE_CITIES], default="all")
    parser.add_argument("--station-path", type=Path, default=RESOLVED_STATION_EVENTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "housing_did_spatial_feasibility",
    )
    parser.add_argument("--analysis-start-year", type=int, default=2010)
    parser.add_argument("--analysis-end-year", type=int, default=2025)
    parser.add_argument(
        "--exclusion-radius",
        type=int,
        default=1000,
        help="Primary spatial donor exclusion radius in metres "
        "(DDR-001 primary=1000; sensitivity 1500/2000)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    spec = SpatialDonorSpec(
        analysis_start_year=args.analysis_start_year,
        analysis_end_year=args.analysis_end_year,
        primary_exclusion_m=args.exclusion_radius,
    )
    summary = run_audit(cities, args.station_path, args.output_dir, spec)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
