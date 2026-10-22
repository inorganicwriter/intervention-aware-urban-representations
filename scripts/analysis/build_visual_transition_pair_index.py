#!/usr/bin/env python3
"""Build a training-only visual state-transition pair index.

The index is separate from causal response labels and from predictor inputs.
Each row identifies one treatment event, frozen grid, target event-time and
visual modality. Post-opening assets are exposed only as transition targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


_PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

from urban_intervention.representation.time_encoding import (  # noqa: E402
    STREETVIEW_AGE_HALF_SCALE_MONTHS,
    compress_time_gap_months,
)


IDENTITY = (
    "treatment_order",
    "city_key",
    "grid_id",
    "station_event_id",
    "opening_month",
    "grid_geometry_sha256",
)
HORIZONS = (0, 1, 2, 3)
SATELLITE_PRE_YEARS = (-6, -5, -4, -3, -2, -1)


def _as_bool(value: Any) -> bool:
    return False if pd.isna(value) else bool(value)


def _as_text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _as_month(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    if len(text) >= 7 and text[4:5] == "-":
        month = text[:7]
        try:
            year, number = (int(part) for part in month.split("-"))
        except ValueError:
            return None
        return month if year >= 1 and 1 <= number <= 12 else None
    return None


def _month_ordinal(value: str) -> int:
    year, month = (int(part) for part in value[:7].split("-"))
    return year * 12 + month - 1


def _month_delta(start_month: str, end_month: str) -> int:
    """Calendar-month difference; source dates remain month precision."""
    return _month_ordinal(end_month) - _month_ordinal(start_month)


def _date_month(value: Any) -> str | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return pd.Timestamp(value).strftime("%Y-%m")
    except (TypeError, ValueError):
        return None


def _json_records(records: list[dict[str, Any]]) -> str:
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_index(path: Path, required: set[str], name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{name} index not found: {path}")
    frame = pd.read_parquet(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} index lacks required columns: {sorted(missing)}")
    return frame


def _validate_identity(
    frame: pd.DataFrame,
    roster: pd.DataFrame,
    name: str,
) -> None:
    """Reject assets that do not bind to the frozen event and grid identity."""
    if frame.empty:
        return
    known = set(roster.treatment_order.astype(int))
    orders = set(pd.to_numeric(frame.treatment_order, errors="coerce").dropna().astype(int))
    unknown = sorted(orders - known)
    if unknown:
        raise ValueError(f"{name} index contains unknown treatment_order values: {unknown[:10]}")

    expected = roster.set_index("treatment_order")
    for field in IDENTITY[1:]:
        left = frame[field].astype("string").fillna("")
        right = frame.treatment_order.map(expected[field]).astype("string").fillna("")
        mismatch = left.ne(right)
        if mismatch.any():
            sample = frame.loc[mismatch, ["treatment_order", field]].head(5).to_dict("records")
            raise ValueError(f"{name} {field} does not match the frozen event roster: {sample}")


def _streetview_refs(frame: pd.DataFrame, opening_month: str, role: str) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        capture_month = _as_month(getattr(row, "capture_month", None))
        asset_path = _as_text(getattr(row, "asset_path", ""))
        valid = _as_bool(getattr(row, "image_valid", False)) and bool(asset_path)
        if capture_month is None or not valid:
            continue
        if role == "pre":
            if capture_month >= opening_month:
                continue
            age = _month_delta(capture_month, opening_month)
            refs.append({
                "asset_path": asset_path,
                "asset_sha256": _as_text(getattr(row, "asset_sha256", "")),
                "panoid": _as_text(getattr(row, "panoid", "")),
                "capture_month": capture_month,
                "age_before_opening_months": age,
                "age_before_opening_compressed": compress_time_gap_months(age),
            })
        else:
            if capture_month <= opening_month:
                continue
            if _as_bool(getattr(row, "state_allowed", False)):
                raise ValueError("Post-opening streetview archive unexpectedly allows predictor-state use")
            lag = _month_delta(opening_month, capture_month)
            refs.append({
                "asset_path": asset_path,
                "asset_sha256": _as_text(getattr(row, "asset_sha256", "")),
                "panoid": _as_text(getattr(row, "panoid", "")),
                "capture_month": capture_month,
                "months_after_opening": lag,
            })
    return sorted(refs, key=lambda item: (item["capture_month"], item["panoid"]))


def _satellite_ref(row: Any, valid: bool) -> dict[str, Any]:
    target_year = int(row.target_year)
    image_path = _as_text(row.image_path)
    return {
        "event_time": int(row.event_time),
        "target_year": target_year,
        "available_mask": bool(valid),
        "image_path": image_path if valid else "",
        "image_sha256": _as_text(row.image_sha256) if valid else "",
        "sensor": _as_text(row.sensor) if valid else "",
        "source_date_min": _as_text(row.source_date_min),
        "source_date_max": _as_text(row.source_date_max),
        "scene_count": int(row.scene_count) if pd.notna(row.scene_count) else None,
        "valid_pixel_fraction": float(row.valid_pixel_fraction)
        if pd.notna(row.valid_pixel_fraction) else None,
    }


def build_pair_index(
    satellite_path: Path,
    streetview_pre_path: Path,
    streetview_post_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    satellite = _read_index(
        satellite_path,
        set(IDENTITY) | {
            "event_time", "target_year", "state_temporal_allowed",
            "state_image_available", "satellite_image_available",
            "calendar_year_complete", "image_path", "image_sha256", "sensor",
            "source_date_min", "source_date_max", "scene_count", "valid_pixel_fraction",
        },
        "satellite sequence",
    )
    street_pre = _read_index(
        streetview_pre_path,
        set(IDENTITY) | {"capture_month", "image_valid", "asset_path", "asset_sha256", "panoid"},
        "pre-opening streetview",
    )
    street_post = _read_index(
        streetview_post_path,
        set(IDENTITY) | {
            "event_time", "target_year", "capture_month", "image_valid",
            "state_allowed", "asset_path", "asset_sha256", "panoid",
        },
        "post-opening streetview",
    )

    satellite["treatment_order"] = pd.to_numeric(satellite.treatment_order, errors="raise").astype(int)
    street_pre["treatment_order"] = pd.to_numeric(street_pre.treatment_order, errors="raise").astype(int)
    street_post["treatment_order"] = pd.to_numeric(street_post.treatment_order, errors="raise").astype(int)
    satellite["event_time"] = pd.to_numeric(satellite.event_time, errors="raise").astype(int)
    street_post["event_time"] = pd.to_numeric(street_post.event_time, errors="raise").astype(int)
    satellite["target_year"] = pd.to_numeric(satellite.target_year, errors="raise").astype(int)
    street_post["target_year"] = pd.to_numeric(street_post.target_year, errors="raise").astype(int)

    roster = satellite.loc[satellite.event_time.eq(-1), list(IDENTITY)].drop_duplicates()
    if roster.treatment_order.duplicated().any():
        raise ValueError("Satellite -1 rows do not define a unique frozen event roster")
    roster = roster.sort_values("treatment_order").reset_index(drop=True)
    if roster.empty:
        raise ValueError("No treatment events found in the satellite sequence index")
    roster["opening_month"] = roster.opening_month.map(_as_month)
    if roster.opening_month.isna().any():
        raise ValueError("Frozen event roster contains invalid opening_month values")

    _validate_identity(satellite, roster, "satellite sequence")
    _validate_identity(street_pre, roster, "pre-opening streetview")
    _validate_identity(street_post, roster, "post-opening streetview")

    if satellite.duplicated(["treatment_order", "event_time"]).any():
        raise ValueError("Satellite sequence has duplicate event-time rows")
    if street_pre.duplicated(["treatment_order", "panoid"]).any():
        raise ValueError("Pre-opening streetview has duplicate event-panoid rows")
    if street_post.duplicated(["treatment_order", "event_time", "panoid"]).any():
        raise ValueError("Post-opening streetview has duplicate event-year-panoid rows")
    if not street_post.event_time.between(0, 3).all():
        raise ValueError("Post-opening streetview includes event_time outside 0..3")

    sv_pre_by_event = {int(k): v for k, v in street_pre.groupby("treatment_order", sort=False)}
    sv_post_by_event_horizon = {
        (int(k[0]), int(k[1])): v
        for k, v in street_post.groupby(["treatment_order", "event_time"], sort=False)
    }
    sat_by_event_time = {
        (int(row.treatment_order), int(row.event_time)): row
        for row in satellite.itertuples(index=False)
    }

    records: list[dict[str, Any]] = []
    for event in roster.itertuples(index=False):
        order = int(event.treatment_order)
        opening_month = str(event.opening_month)
        opening_year = int(opening_month[:4])

        sv_pre_frame = sv_pre_by_event.get(order, street_pre.iloc[0:0])
        sv_pre_refs = _streetview_refs(sv_pre_frame, opening_month, "pre")
        if len(sv_pre_refs) != len(sv_pre_frame.loc[sv_pre_frame.image_valid.fillna(False).astype(bool)]):
            raise ValueError(f"Pre-opening streetview has a temporal or path violation for treatment_order={order}")

        sat_pre_refs: list[dict[str, Any]] = []
        for rel_year in SATELLITE_PRE_YEARS:
            row = sat_by_event_time.get((order, rel_year))
            if row is None:
                raise ValueError(f"Missing satellite pre-year slot {rel_year} for treatment_order={order}")
            if int(row.target_year) != opening_year + rel_year:
                raise ValueError(f"Satellite pre-year target mismatch for treatment_order={order}, event_time={rel_year}")
            valid = (
                _as_bool(row.state_temporal_allowed)
                and _as_bool(row.state_image_available)
                and _as_bool(row.satellite_image_available)
                and _as_bool(row.calendar_year_complete)
                and bool(str(row.image_path or ""))
            )
            if _as_bool(row.state_temporal_allowed) != (rel_year < 0):
                raise ValueError(f"Satellite pre-year temporal mask mismatch at treatment_order={order}")
            sat_pre_refs.append(_satellite_ref(row, valid))

        sat_pre_valid = [ref for ref in sat_pre_refs if ref["available_mask"]]
        sat_ref = next(ref for ref in sat_pre_refs if ref["event_time"] == -1)
        ref_min_month = _date_month(sat_ref["source_date_min"])
        ref_max_month = _date_month(sat_ref["source_date_max"])
        sat_pre_age_min = _month_delta(ref_max_month, opening_month) if ref_max_month else None
        sat_pre_age_max = _month_delta(ref_min_month, opening_month) if ref_min_month else None

        for horizon in HORIZONS:
            target_year = opening_year + horizon
            sv_post_frame = sv_post_by_event_horizon.get(
                (order, horizon), street_post.iloc[0:0]
            )
            if not sv_post_frame.empty and not sv_post_frame.target_year.eq(target_year).all():
                raise ValueError(f"Streetview target year mismatch for treatment_order={order}, event_time={horizon}")
            sv_post_refs = _streetview_refs(sv_post_frame, opening_month, "post")
            if len(sv_post_refs) != len(sv_post_frame.loc[sv_post_frame.image_valid.fillna(False).astype(bool)]):
                raise ValueError(f"Post-opening streetview has a temporal or path violation for treatment_order={order}, event_time={horizon}")
            sv_pre_mask = bool(sv_pre_refs)
            sv_target_mask = bool(sv_post_refs)
            sv_lags = [int(ref["months_after_opening"]) for ref in sv_post_refs]
            sv_ages = [int(ref["age_before_opening_months"]) for ref in sv_pre_refs]
            records.append({
                **{field: getattr(event, field) for field in IDENTITY},
                "pair_id": f"{order}:streetview:{horizon}",
                "modality": "streetview",
                "target_event_time": horizon,
                "target_year": target_year,
                "pre_state_assets_json": _json_records(sv_pre_refs),
                "post_target_assets_json": _json_records(sv_post_refs),
                "pre_state_asset_count": len(sv_pre_refs),
                "post_target_asset_count": len(sv_post_refs),
                "pre_state_mask": sv_pre_mask,
                "post_target_candidate_available": sv_target_mask,
                "post_target_mask": sv_target_mask,
                "identity_mask": True,
                "pre_temporal_mask": all(age > 0 for age in sv_ages),
                "post_temporal_mask": all(lag > 0 for lag in sv_lags),
                "transition_supervision_mask": bool(sv_pre_mask and sv_target_mask),
                "pre_age_min_months": min(sv_ages) if sv_ages else None,
                "pre_age_max_months": max(sv_ages) if sv_ages else None,
                "post_horizon_min_months": min(sv_lags) if sv_lags else None,
                "post_horizon_max_months": max(sv_lags) if sv_lags else None,
                "pre_reference_event_time": None,
                "pre_reference_source_date_min": None,
                "pre_reference_source_date_max": None,
                "post_source_date_min": None,
                "post_source_date_max": None,
                "target_exclusion_reason": None if sv_target_mask else "no_verified_post_streetview_in_target_year",
                "post_target_predictor_use_allowed": False,
                "target_use_allowed": True,
            })

            sat_target_row = sat_by_event_time.get((order, horizon))
            if sat_target_row is None:
                raise ValueError(f"Missing satellite target-year slot {horizon} for treatment_order={order}")
            if int(sat_target_row.target_year) != target_year:
                raise ValueError(f"Satellite target year mismatch for treatment_order={order}, event_time={horizon}")
            raw_sat_candidate = (
                _as_bool(sat_target_row.satellite_image_available)
                and bool(_as_text(sat_target_row.image_path))
            )
            sat_target_temporal_valid = (
                horizon > 0
                and str(sat_target_row.temporal_phase) == "post"
                and _as_bool(sat_target_row.calendar_year_complete)
                and bool(_date_month(sat_target_row.source_date_min))
                and bool(_date_month(sat_target_row.source_date_max))
            )
            sat_target_mask = bool(raw_sat_candidate and sat_target_temporal_valid)
            sat_post_refs = [_satellite_ref(sat_target_row, sat_target_mask)] if sat_target_mask else []
            post_min_month = _date_month(sat_target_row.source_date_min)
            post_max_month = _date_month(sat_target_row.source_date_max)
            sat_post_lags = [
                _month_delta(opening_month, month)
                for month in (post_min_month, post_max_month)
                if month is not None
            ] if sat_target_mask else []
            sat_pre_mask = bool(sat_pre_valid)
            records.append({
                **{field: getattr(event, field) for field in IDENTITY},
                "pair_id": f"{order}:satellite:{horizon}",
                "modality": "satellite",
                "target_event_time": horizon,
                "target_year": target_year,
                "pre_state_assets_json": _json_records(sat_pre_refs),
                "post_target_assets_json": _json_records(sat_post_refs),
                "pre_state_asset_count": len(sat_pre_valid),
                "post_target_asset_count": len(sat_post_refs),
                "pre_state_mask": sat_pre_mask,
                "post_target_candidate_available": raw_sat_candidate,
                "post_target_mask": sat_target_mask,
                "identity_mask": True,
                "pre_temporal_mask": len(sat_pre_refs) == len(SATELLITE_PRE_YEARS)
                and all(ref["event_time"] < 0 for ref in sat_pre_refs),
                "post_temporal_mask": bool(sat_target_temporal_valid),
                "transition_supervision_mask": bool(sat_pre_mask and sat_target_mask),
                "pre_age_min_months": sat_pre_age_min,
                "pre_age_max_months": sat_pre_age_max,
                "post_horizon_min_months": min(sat_post_lags) if sat_post_lags else None,
                "post_horizon_max_months": max(sat_post_lags) if sat_post_lags else None,
                "pre_reference_event_time": -1,
                "pre_reference_source_date_min": sat_ref["source_date_min"],
                "pre_reference_source_date_max": sat_ref["source_date_max"],
                "post_source_date_min": _as_text(sat_target_row.source_date_min) or None,
                "post_source_date_max": _as_text(sat_target_row.source_date_max) or None,
                "target_exclusion_reason": (
                    "opening_year_mixes_pre_and_post_observations" if horizon == 0 else
                    "incomplete_or_unavailable_annual_composite" if not sat_target_mask else None
                ),
                "post_target_predictor_use_allowed": False,
                "target_use_allowed": horizon > 0,
            })

    pairs = pd.DataFrame.from_records(records).sort_values(
        ["treatment_order", "target_event_time", "modality"]
    ).reset_index(drop=True)
    if len(pairs) != len(roster) * len(HORIZONS) * 2:
        raise ValueError("Pair index does not preserve all event × horizon × modality slots")
    if pairs.pair_id.duplicated().any():
        raise ValueError("Pair index contains duplicate pair_id values")
    invalid_supervision = pairs.transition_supervision_mask & ~(
        pairs.pre_state_mask & pairs.post_target_mask & pairs.identity_mask
        & pairs.pre_temporal_mask & pairs.post_temporal_mask
    )
    if invalid_supervision.any():
        raise ValueError("Transition supervision mask includes an invalid source or target")
    sat_open_year_targets = pairs.loc[
        pairs.modality.eq("satellite") & pairs.target_event_time.eq(0), "post_target_mask"
    ]
    if sat_open_year_targets.any():
        raise ValueError("Opening-year satellite imagery cannot be used as a post-state target")

    summary = {
        "schema": "visual_transition_pair_index",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "events": int(len(roster)),
        "target_event_times": list(HORIZONS),
        "modalities": ["streetview", "satellite"],
        "pair_index_rows": int(len(pairs)),
        "source_files": {
            str(satellite_path): _sha256(satellite_path),
            str(streetview_pre_path): _sha256(streetview_pre_path),
            str(streetview_post_path): _sha256(streetview_post_path),
        },
        "mask_semantics": {
            "pre_state_mask": "at least one legal, indexed pre-opening source asset is available",
            "post_target_mask": "at least one legal, indexed post-state target is available at this horizon",
            "identity_mask": "event, city, fixed grid, opening month and geometry hash match frozen satellite roster",
            "transition_supervision_mask": "pre_state_mask AND post_target_mask AND identity_mask AND both temporal masks",
            "post_target_predictor_use_allowed": "false for all post-opening target assets",
        },
        "streetview_pre_age_encoding": {
            "field": "age_before_opening_compressed",
            "formula": "d / (d + tau)",
            "d": "positive calendar-month difference from capture month to opening month",
            "tau_months": STREETVIEW_AGE_HALF_SCALE_MONTHS,
            "usage": "temporal conditioning feature, not sample or loss weight",
        },
        "by_modality_horizon": {},
    }
    grouped = pairs.groupby(["modality", "target_event_time"], sort=True)
    for (modality, horizon), frame in grouped:
        summary["by_modality_horizon"][f"{modality}:{int(horizon)}"] = {
            "rows": int(len(frame)),
            "pre_state_available": int(frame.pre_state_mask.sum()),
            "post_target_available": int(frame.post_target_mask.sum()),
            "supervised_pairs": int(frame.transition_supervision_mask.sum()),
        }
    return pairs, summary


def main() -> int:
    project_root = Path(os.environ.get("MIT_PROJECT_ROOT") or Path.cwd()).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--satellite-index", type=Path,
        default=None,
        help="Satellite sequence index; if omitted, resolve its unique active copy by filename.",
    )
    parser.add_argument(
        "--streetview-pre", type=Path,
        default=None,
        help="Pre-opening streetview index; if omitted, resolve its unique active copy by filename.",
    )
    parser.add_argument(
        "--streetview-post", type=Path,
        default=None,
        help="Post-opening streetview index; if omitted, resolve its unique active copy by filename.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=None,
        help="New output directory outside data/active; defaults to a timestamped outputs directory.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    active_root = root / "data" / "active"

    def resolve_active_index(explicit: Path | None, filename: str, label: str) -> Path:
        if explicit is not None:
            return resolve(explicit)
        candidates = sorted(path.resolve() for path in active_root.rglob(filename) if path.is_file())
        if not candidates:
            raise FileNotFoundError(
                f"No active {label} index named {filename}; pass its path explicitly"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Found multiple active {label} indexes; pass --{label} explicitly: "
                f"{[str(path) for path in candidates]}"
            )
        return candidates[0]

    satellite_path = resolve_active_index(
        args.satellite_index, "satellite_sequence_index.parquet", "satellite-index"
    )
    streetview_pre_path = resolve_active_index(
        args.streetview_pre, "streetview_state_assets.parquet", "streetview-pre"
    )
    streetview_post_path = resolve_active_index(
        args.streetview_post, "streetview_post_assets.parquet", "streetview-post"
    )
    output_dir = (
        resolve(args.output_dir)
        if args.output_dir is not None
        else root / "outputs" / "visual_transition_pairing" /
        f"pair_index_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    active_root = active_root.resolve()
    if output_dir == active_root or active_root in output_dir.parents:
        raise ValueError("Pair index output must be outside data/active")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    pairs, summary = build_pair_index(satellite_path, streetview_pre_path, streetview_post_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_tmp = output_dir / ".visual_transition_pairs.parquet.tmp"
    summary_tmp = output_dir / ".visual_transition_pair_audit.json.tmp"
    pairs.to_parquet(parquet_tmp, index=False, compression="zstd")
    summary["outputs"] = {
        "visual_transition_pairs.parquet": _sha256(parquet_tmp),
    }
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(parquet_tmp, output_dir / "visual_transition_pairs.parquet")
    os.replace(summary_tmp, output_dir / "visual_transition_pair_audit.json")
    print(json.dumps({
        "output_dir": str(output_dir),
        "pair_index_rows": summary["pair_index_rows"],
        "by_modality_horizon": summary["by_modality_horizon"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
