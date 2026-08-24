"""Compile reviewed station-identity decisions into auditable data products.

The compiler never overwrites the source canonical station table. Reviewed
duplicates are collapsed for the primary metro intervention universe, true
multi-station grids remain in the exposure universe but are marked ineligible
for the primary design, and non-metro interchange events are published as a
separate competing-intervention product.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from urban_intervention.config.project import ACTIVE_CITIES
from urban_intervention.data.paths import (
    CANONICAL_STATION_EVENTS,
    COMPETING_TRANSIT_EVENTS,
    EXCLUDED_STATION_EVENTS,
    RESOLVED_STATION_EVENTS,
    STATION_ISSUE_RESOLUTION,
    STATION_RESOLUTION_MANIFEST,
)
from urban_intervention.text import normalize_station_name
from urban_intervention.utils import require_columns, sha256_file

RESOLUTION_VERSION = "station_issue_resolution_v1"
EVENT_REQUIRED_COLUMNS = (
    "city_key",
    "station_event_id",
    "canonical_station_name",
    "normalized_name",
    "lines",
    "wgs84_lon",
    "wgs84_lat",
    "opening_year",
    "opening_date",
)
RESOLUTION_REQUIRED_COLUMNS = (
    "issue_type",
    "original_city_key",
    "grid_id",
    "station_names",
    "review_decision",
    "canonical_station_name",
    "primary_station_name",
    "verified_city_key",
    "study_disposition",
    "review_basis",
)


def _split(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _ordered_union(values: Iterable[object], separators: str = r"[;+]") -> str:
    result: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        for part in re.split(separators, str(value or "")):
            part = part.strip()
            if part and part not in result:
                result.append(part)
    return ";".join(result)


def _add_resolution_columns(events: pd.DataFrame) -> pd.DataFrame:
    resolved = events.copy()
    resolved["resolution_version"] = RESOLUTION_VERSION
    resolved["resolution_status"] = "unchanged"
    resolved["resolution_grid_id"] = ""
    resolved["original_city_key"] = resolved["city_key"].astype(str)
    resolved["original_station_event_ids"] = resolved["station_event_id"].astype(str)
    resolved["primary_design_excluded"] = False
    resolved["primary_design_exclusion_reason"] = ""
    resolved["competing_event_ids"] = ""
    resolved["post_treatment_censor_year"] = pd.Series(pd.NA, index=resolved.index, dtype="Int64")
    return resolved


def _select_issue_events(
    events: pd.DataFrame,
    resolution_row: Any,
) -> pd.DataFrame:
    names = _split(resolution_row.station_names)
    selected_parts: list[pd.DataFrame] = []
    for name in names:
        matches = events[
            (events["city_key"].astype(str) == str(resolution_row.original_city_key))
            & (events["canonical_station_name"].astype(str) == name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{resolution_row.original_city_key}/{name}: expected one source "
                f"event, found {len(matches)}"
            )
        selected_parts.append(matches)
    return pd.concat(selected_parts, ignore_index=True)


def _decorate_disposition(
    events: pd.DataFrame,
    row: Any,
    disposition: str,
    linked_primary_event_id: str = "",
) -> pd.DataFrame:
    result = events.copy()
    result["resolution_version"] = RESOLUTION_VERSION
    result["resolution_grid_id"] = str(row.grid_id or "")
    result["review_decision"] = str(row.review_decision)
    result["study_disposition"] = str(row.study_disposition)
    result["event_disposition"] = disposition
    result["verified_city_key"] = str(row.verified_city_key or "")
    result["linked_primary_station_event_id"] = linked_primary_event_id
    return result


def compile_station_resolution(
    source_events: pd.DataFrame,
    decisions: pd.DataFrame,
    active_cities: Iterable[str] = ACTIVE_CITIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Apply every reviewed decision exactly once without mutating source data."""

    require_columns(source_events, EVENT_REQUIRED_COLUMNS, "source station events")
    require_columns(decisions, RESOLUTION_REQUIRED_COLUMNS, "station decisions")
    if source_events["station_event_id"].astype(str).duplicated().any():
        raise ValueError("source station events contain duplicate station_event_id")
    if decisions.empty:
        resolved = _add_resolution_columns(source_events.copy())
        competing = pd.DataFrame(columns=[*source_events.columns, "event_disposition"])
        excluded = pd.DataFrame(columns=[*source_events.columns, "event_disposition"])
        manifest = {
            "resolution_version": RESOLUTION_VERSION,
            "source_station_events": int(len(source_events)),
            "resolution_rows": 0,
            "source_events_touched": 0,
            "resolved_station_events": int(len(resolved)),
            "primary_design_excluded_events": 0,
            "competing_transit_events": 0,
            "excluded_or_superseded_event_rows": 0,
            "action_counts": {},
            "study_disposition_counts": {},
        }
        return resolved, competing, excluded, manifest

    active_city_set = set(active_cities)
    original = source_events.copy()
    resolved = _add_resolution_columns(original)
    competing_parts: list[pd.DataFrame] = []
    excluded_parts: list[pd.DataFrame] = []
    touched_ids: set[str] = set()
    action_counts: dict[str, int] = {}

    for row in decisions.itertuples(index=False):
        selected = _select_issue_events(original, row)
        selected_ids = selected["station_event_id"].astype(str).tolist()
        overlap = touched_ids.intersection(selected_ids)
        if overlap:
            raise ValueError(f"station events reviewed more than once: {sorted(overlap)}")
        touched_ids.update(selected_ids)
        decision = str(row.review_decision)
        action_counts[decision] = action_counts.get(decision, 0) + 1

        if decision == "same_physical_station":
            primary_matches = selected[
                selected["canonical_station_name"].astype(str) == str(row.primary_station_name)
            ]
            if len(primary_matches) != 1:
                raise ValueError(
                    f"{row.original_city_key}/{row.grid_id}: primary station "
                    f"{row.primary_station_name!r} did not resolve uniquely"
                )
            primary_id = str(primary_matches.iloc[0]["station_event_id"])
            primary = resolved[resolved["station_event_id"].astype(str) == primary_id].copy()
            primary["canonical_station_name"] = str(row.canonical_station_name)
            primary["normalized_name"] = normalize_station_name(row.canonical_station_name)
            primary["lines"] = _ordered_union(selected["lines"])
            if "station_sources" in primary.columns:
                primary["station_sources"] = _ordered_union(selected["station_sources"])
            if "raw_record_count" in primary.columns:
                primary["raw_record_count"] = int(
                    pd.to_numeric(selected["raw_record_count"], errors="coerce").fillna(0).sum()
                )
            primary["resolution_status"] = "merged_same_physical_station"
            primary["resolution_grid_id"] = str(row.grid_id)
            primary["original_station_event_ids"] = ";".join(selected_ids)
            resolved = resolved[~resolved["station_event_id"].astype(str).isin(selected_ids)]
            resolved = pd.concat([resolved, primary], ignore_index=True)
            duplicates = selected[selected["station_event_id"].astype(str) != primary_id]
            if not duplicates.empty:
                excluded_parts.append(
                    _decorate_disposition(duplicates, row, "merged_into_primary_event", primary_id)
                )

        elif decision == "distinct_physical_stations":
            mask = resolved["station_event_id"].astype(str).isin(selected_ids)
            resolved.loc[mask, "resolution_status"] = "resolved_distinct_station_grid_exclusion"
            resolved.loc[mask, "resolution_grid_id"] = str(row.grid_id)
            resolved.loc[mask, "primary_design_excluded"] = True
            resolved.loc[mask, "primary_design_exclusion_reason"] = (
                "multiple_distinct_stations_in_grid"
            )
            excluded_parts.append(
                _decorate_disposition(
                    selected,
                    row,
                    "primary_design_excluded_retained_for_spatial_exposure",
                )
            )

        elif decision == "distinct_modes_same_interchange":
            primary_matches = selected[
                selected["canonical_station_name"].astype(str) == str(row.primary_station_name)
            ]
            if len(primary_matches) != 1:
                raise ValueError(
                    f"{row.original_city_key}/{row.grid_id}: primary mode event "
                    f"{row.primary_station_name!r} did not resolve uniquely"
                )
            primary_id = str(primary_matches.iloc[0]["station_event_id"])
            secondary = selected[selected["station_event_id"].astype(str) != primary_id].copy()
            secondary_ids = secondary["station_event_id"].astype(str).tolist()
            if not secondary_ids:
                raise ValueError(f"{row.original_city_key}/{row.grid_id}: no secondary event")
            primary_mask = resolved["station_event_id"].astype(str) == primary_id
            resolved.loc[primary_mask, "canonical_station_name"] = str(row.canonical_station_name)
            resolved.loc[primary_mask, "normalized_name"] = normalize_station_name(
                row.canonical_station_name
            )
            resolved.loc[primary_mask, "resolution_status"] = (
                "primary_mode_with_competing_intervention"
            )
            resolved.loc[primary_mask, "resolution_grid_id"] = str(row.grid_id)
            resolved.loc[primary_mask, "competing_event_ids"] = ";".join(secondary_ids)
            secondary_years = pd.to_numeric(secondary["opening_year"], errors="coerce").dropna()
            if secondary_years.empty:
                resolved.loc[primary_mask, "post_treatment_censor_year"] = pd.NA
                resolved.loc[primary_mask, "resolution_status"] = (
                    "primary_mode_with_competing_intervention_date_unknown"
                )
            else:
                resolved.loc[primary_mask, "post_treatment_censor_year"] = int(
                    secondary_years.min()
                )
            resolved = resolved[~resolved["station_event_id"].astype(str).isin(secondary_ids)]
            competing_parts.append(
                _decorate_disposition(
                    secondary,
                    row,
                    "competing_intervention",
                    primary_id,
                )
            )
            excluded_parts.append(
                _decorate_disposition(
                    secondary,
                    row,
                    "moved_to_competing_events",
                    primary_id,
                )
            )

        elif decision == "wrong_city_assignment":
            if len(selected) != 1:
                raise ValueError(
                    f"{row.original_city_key}/{row.station_names}: city correction "
                    "must select exactly one event"
                )
            event_id = selected_ids[0]
            mask = resolved["station_event_id"].astype(str) == event_id
            disposition = str(row.study_disposition)
            target_city = str(row.verified_city_key)
            if disposition == "reassign_to_active_city":
                if target_city not in active_city_set:
                    raise ValueError(f"reassignment target is not active: {target_city}")
                resolved.loc[mask, "city_key"] = target_city
                resolved.loc[mask, "resolution_status"] = "reassigned_to_active_city"
            elif disposition == "exclude_outside_study_universe":
                resolved = resolved[~mask]
                excluded_parts.append(
                    _decorate_disposition(selected, row, "outside_active_city_universe")
                )
            else:
                raise ValueError(f"unsupported city disposition: {disposition}")
        else:
            raise ValueError(f"unsupported review_decision: {decision}")

    resolved = resolved.sort_values(["city_key", "opening_year", "station_event_id"]).reset_index(
        drop=True
    )
    if resolved["station_event_id"].astype(str).duplicated().any():
        raise ValueError("resolved station events contain duplicate station_event_id")
    invalid_cities = sorted(set(resolved["city_key"].astype(str)) - active_city_set)
    if invalid_cities:
        raise ValueError(f"resolved events contain non-active cities: {invalid_cities}")

    competing = (
        pd.concat(competing_parts, ignore_index=True)
        if competing_parts
        else pd.DataFrame(columns=[*source_events.columns, "event_disposition"])
    )
    excluded = (
        pd.concat(excluded_parts, ignore_index=True)
        if excluded_parts
        else pd.DataFrame(columns=[*source_events.columns, "event_disposition"])
    )
    manifest = {
        "resolution_version": RESOLUTION_VERSION,
        "source_station_events": int(len(source_events)),
        "resolution_rows": int(len(decisions)),
        "source_events_touched": int(len(touched_ids)),
        "resolved_station_events": int(len(resolved)),
        "primary_design_excluded_events": int(
            resolved["primary_design_excluded"].astype(bool).sum()
        ),
        "competing_transit_events": int(len(competing)),
        "excluded_or_superseded_event_rows": int(len(excluded)),
        "action_counts": action_counts,
        "study_disposition_counts": {
            str(key): int(value)
            for key, value in decisions["study_disposition"].value_counts().items()
        },
    }
    return resolved, competing, excluded, manifest


def write_resolution_products(
    source_path: Path,
    resolution_path: Path,
    resolved_path: Path,
    competing_path: Path,
    excluded_path: Path,
    manifest_path: Path,
) -> dict:
    source = pd.read_parquet(source_path)
    decisions = pd.read_csv(resolution_path, keep_default_na=False)
    resolved, competing, excluded, manifest = compile_station_resolution(source, decisions)
    for path in (resolved_path, competing_path, excluded_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_parquet(resolved_path, index=False)
    competing.to_parquet(competing_path, index=False)
    excluded.to_csv(excluded_path, index=False)
    manifest.update(
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "resolution_path": str(resolution_path),
            "resolution_sha256": sha256_file(resolution_path),
            "resolved_path": str(resolved_path),
            "competing_path": str(competing_path),
            "excluded_path": str(excluded_path),
            "reads_housing_outcomes": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply reviewed station identity and city-assignment decisions"
    )
    parser.add_argument("--source-path", type=Path, default=CANONICAL_STATION_EVENTS)
    parser.add_argument("--resolution-path", type=Path, default=STATION_ISSUE_RESOLUTION)
    parser.add_argument("--resolved-path", type=Path, default=RESOLVED_STATION_EVENTS)
    parser.add_argument("--competing-path", type=Path, default=COMPETING_TRANSIT_EVENTS)
    parser.add_argument("--excluded-path", type=Path, default=EXCLUDED_STATION_EVENTS)
    parser.add_argument("--manifest-path", type=Path, default=STATION_RESOLUTION_MANIFEST)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = write_resolution_products(
        args.source_path,
        args.resolution_path,
        args.resolved_path,
        args.competing_path,
        args.excluded_path,
        args.manifest_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
