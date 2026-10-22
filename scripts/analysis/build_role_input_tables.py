"""Publish raw S/T/C input tables without fitting preprocessing or a model.

Annual lag 1/2/3 features come only from exact years before opening. City
backgrounds summarize all available reference-grid records in the source city,
not the treated sample. They are coverage-qualified proxies, not yearbook data.
Street view is deferred. Image references are attached after full acquisition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.data.paths import (  # noqa: E402
    ACTIVE_DIR,
    REFERENCE_GRID_DIR,
    TREATMENT_UNIT_LIST,
    poi_annual_path,
    population_data_path,
    viirs_annual_path,
)

OUTPUT = ACTIVE_DIR / "multimodal_inputs_v1"
AE_PATH = ACTIVE_DIR / "alphaearth_grid_v2/alphaearth_treated_embeddings.parquet"
IMAGE_ROOT = ACTIVE_DIR / "satellite_grid_images_v1"
SEQUENCE_ROOT = ACTIVE_DIR / "satellite_event_sequences_v2"
SEQUENCE_AUDIT = ROOT / "outputs/satellite_event_sequences_audit/2026-10-13/audit.json"
T_ROOT = ACTIVE_DIR / "intervention_grid_v1"
KEYS = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"]
LAGS = [1, 2, 3]
SPECS = {
    "poi": {"path": poi_annual_path, "city": "city", "role": "state",
            "columns": {"poi_count": "poi_count_log", "poi_category_entropy": "poi_category_entropy",
                        "poi_commercial_share": "poi_commercial_share",
                        "poi_transport_access_count": "poi_transport_access_log"},
            "logs": ["poi_count", "poi_transport_access_count"], "summary": "poi_count", "summary_log": True},
    "viirs": {"path": viirs_annual_path, "city": "city_key", "role": "state",
              "columns": {"avg_rad": "viirs_avg_asinh"}, "logs": [],
              "summary": "avg_rad", "summary_log": False},
    "population": {"path": population_data_path, "city": "city", "role": "context",
                   "columns": {"pop_count": "population_log"}, "logs": ["pop_count"],
                   "summary": "pop_count", "summary_log": True},
}
SUMMARY_NAMES = {"poi": "city_poi_log_total", "population": "city_population_log_total",
                 "viirs": "city_viirs_mean_asinh"}


def source_city_features(city: str, modality: str, treatment_grids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    spec = SPECS[modality]
    path = spec["path"](city)
    columns = [spec["city"], "grid_id", "year", *spec["columns"]]
    grids_path = REFERENCE_GRID_DIR / city / f"{city}_grids.parquet"
    grid_index = pd.Index(pd.read_parquet(grids_path, columns=["grid_id"]).grid_id)
    if not grid_index.is_unique:
        raise ValueError("Duplicate reference grid identity")
    if not path.exists():
        return pd.DataFrame(columns=["grid_id", "year", *spec["columns"].values()]), pd.DataFrame(), {"path": str(path), "missing": True}
    parts, totals, counts, seen = [], {}, {}, {}
    rows = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=columns):
        frame = batch.to_pandas()
        if not frame[spec["city"]].eq(city).all():
            raise ValueError(f"Source city mismatch: {path}")
        positions = grid_index.get_indexer(frame.grid_id)
        if (positions < 0).any():
            raise ValueError(f"Source contains grid outside reference registry: {path}")
        if frame.year.isna().any() or not frame.year.between(2000, 2030).all() or not frame.year.mod(1).eq(0).all():
            raise ValueError("Invalid annual source year")
        values = pd.to_numeric(frame[spec["summary"]], errors="raise").to_numpy(dtype=float)
        for year, indices in frame.groupby("year").indices.items():
            year = int(year)
            grid_positions = positions[indices]
            prior = seen.setdefault(year, np.zeros(len(grid_index), dtype=bool))
            if len(np.unique(grid_positions)) != len(grid_positions) or prior[grid_positions].any():
                raise ValueError(f"Duplicate city/grid/year in {path}")
            prior[grid_positions] = True
            array = values[indices]
            valid = np.isfinite(array)
            if spec["summary_log"]:
                valid &= array >= 0
            totals[year] = totals.get(year, 0) + float(array[valid].sum())
            counts[year] = counts.get(year, 0) + int(valid.sum())
        local = frame.loc[frame.grid_id.isin(treatment_grids)].copy()
        for raw in spec["logs"]:
            value = pd.to_numeric(local[raw], errors="raise")
            local[raw] = np.log1p(value.where(value >= 0))
        if modality == "viirs":
            local["avg_rad"] = np.arcsinh(local.avg_rad)
        parts.append(local[["grid_id", "year", *spec["columns"]]].rename(columns=spec["columns"]))
        rows += len(frame)
    summaries = []
    for year in sorted(totals):
        value = np.nan
        if counts[year]:
            value = np.log1p(totals[year]) if spec["summary_log"] else totals[year] / counts[year]
            if modality == "viirs":
                value = np.arcsinh(value)
        summaries.append({"year": year, SUMMARY_NAMES[modality]: value,
                          f"city_{modality}_observed_grids": counts[year],
                          f"city_{modality}_grid_coverage": counts[year] / len(grid_index)})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(summaries), {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"), "rows": rows,
        "reference_grids": len(grid_index), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def attach_annual_lags(identity: pd.DataFrame, annual: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    if annual.duplicated(["grid_id", "year"]).any():
        raise ValueError("Duplicate annual grid/year identity")
    output = identity.copy()
    opening_year = pd.to_datetime(identity.opening_month).dt.year
    for lag in LAGS:
        query = identity[["grid_id"]].copy()
        query["year"] = opening_year - lag
        merged = query.merge(annual, on=["grid_id", "year"], how="left", validate="many_to_one", sort=False)
        output[f"lag{lag}_year"] = query.year.to_numpy()
        for field in fields:
            name = f"{field}_lag{lag}"
            value = pd.to_numeric(merged[field], errors="raise").to_numpy(dtype=float, copy=True)
            value[~np.isfinite(value)] = np.nan
            output[name] = value
            output[f"{name}_available"] = np.isfinite(value)
    return output


def attach_image_references(output: Path, *, sequence_refresh: bool = False) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("satellite_state_sequence_table") and not sequence_refresh:
        # A legacy baseline refresh must not point a six-year sequence at the one-year cache.
        attach_sequence_references(output)
        return
    audit = json.loads((IMAGE_ROOT / "satellite_image_audit.json").read_text(encoding="utf-8"))
    images = pd.read_parquet(IMAGE_ROOT / "satellite_image_index.parquet")
    state = pd.read_parquet(output / "state_inputs.parquet")
    context = pd.read_parquet(output / "context_inputs.parquet")
    if not audit["run_finished"] or len(images) != len(state) or images.treatment_order.duplicated().any():
        raise ValueError("Only a finished complete image index can be published")
    bound = state[KEYS].merge(images, on="treatment_order", how="left", validate="one_to_one", suffixes=("", "_image"))
    for key in ["city_key", "grid_id", "opening_month"]:
        if not bound[key].eq(bound[f"{key}_image"]).all():
            raise ValueError(f"Image identity mismatch: {key}")
    hashes = context.set_index("treatment_order").alphaearth_grid_geometry_sha256
    if not bound.grid_geometry_sha256.eq(bound.treatment_order.map(hashes)).all():
        raise ValueError("Image and context polygon hashes disagree")
    if not bound.target_year.eq(pd.to_datetime(bound.opening_month).dt.year - 1).all():
        raise ValueError("Image target year violates pretreatment rule")
    for field in ["image_path", "mask_path", "source_manifest_path", "satellite_image_available",
                  "sensor", "valid_pixel_fraction", "source_date_min", "source_date_max"]:
        state[f"satellite_{field}" if not field.startswith("satellite_") else field] = bound[field].to_numpy()
    state.to_parquet(output / "state_inputs.parquet", index=False, compression="zstd")
    manifest.update(bundle_status="tabular_and_satellite_assets_published",
                    satellite_image_root=IMAGE_ROOT.relative_to(ROOT).as_posix(),
                    satellite_paths_relative_to="satellite_image_root",
                    satellite_band_order=["red", "green", "blue", "nir", "swir1", "swir2"],
                    satellite_valid_mask_band_1based=7, satellite_observation_count_band_1based=8,
                    satellite_available=int(images.satellite_image_available.sum()),
                    image_index_sha256=hashlib.sha256((IMAGE_ROOT / "satellite_image_index.parquet").read_bytes()).hexdigest())
    for role in ["state", "context"]:
        features = [f"{field}_lag{lag}" for field in manifest[f"{role}_numeric_fields"] for lag in LAGS]
        if role == "context":
            features += [f"alphaearth_A{i:02d}" for i in range(64)] + list(SUMMARY_NAMES.values())
        manifest[f"{role}_feature_columns"] = features
        manifest[f"{role}_feature_masks"] = {field: "alphaearth_available" if field.startswith("alphaearth_A")
                                              else f"{field}_available" for field in features}
    intervention_manifest = json.loads((T_ROOT / "manifest.json").read_text(encoding="utf-8"))
    manifest["intervention_feature_columns"] = intervention_manifest["fields"]
    manifest["intervention_mask_columns"] = intervention_manifest["mask_columns"]
    manifest["availability_counts"]["state"] = {name: int(state[name].sum()) for name in state if name.endswith("_available")}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_sequence_binding(state: pd.DataFrame, context: pd.DataFrame,
                              images: pd.DataFrame, latest_complete_year: int) -> pd.DataFrame:
    """Bind all ten years, returning only strictly pre-opening state frames."""
    if images.duplicated(["treatment_order", "target_year"]).any() or len(images) != 10 * len(state):
        raise ValueError("Each frozen event must have exactly ten distinct annual rows")
    if set(images.treatment_order) != set(state.treatment_order) or not images.event_time.between(-6, 3).all():
        raise ValueError("Satellite sequence event coverage/window mismatch")
    for event_time in range(-6, 4):
        part = images.loc[images.event_time.eq(event_time)]
        if len(part) != len(state) or set(part.treatment_order) != set(state.treatment_order):
            raise ValueError("Satellite sequence omits an event/year")
    bound = images.merge(state[KEYS], on="treatment_order", validate="many_to_one", suffixes=("", "_state"))
    for key in KEYS[1:]:
        if not bound[key].eq(bound[f"{key}_state"]).all():
            raise ValueError(f"Satellite sequence identity mismatch: {key}")
    hashes = context.set_index("treatment_order").alphaearth_grid_geometry_sha256
    if not bound.grid_geometry_sha256.eq(bound.treatment_order.map(hashes)).all():
        raise ValueError("Satellite sequence and context polygon hashes disagree")
    if not bound.target_year.eq(pd.to_datetime(bound.opening_month).dt.year + bound.event_time).all():
        raise ValueError("Satellite sequence calendar year mismatch")
    pre = bound.event_time.lt(0)
    complete = bound.target_year.le(latest_complete_year)
    phase = np.where(pre, "pre", np.where(bound.event_time.eq(0), "opening_year", "post"))
    if (not bound.state_temporal_allowed.eq(pre).all() or not bound.temporal_phase.eq(phase).all()
            or not bound.calendar_year_complete.eq(complete).all()
            or not bound.state_image_available.eq(pre & bound.satellite_image_available).all()
            or bound.is_temporal_proxy.any() or bound.is_dry_run.any()):
        raise ValueError("Satellite sequence predictor/time masks violate the input contract")
    if bound.loc[~complete, "satellite_image_available"].any():
        raise ValueError("Incomplete years cannot be annual observations")
    return bound.loc[pre, images.columns].sort_values(["treatment_order", "event_time"]).reset_index(drop=True)


def attach_sequence_references(output: Path) -> None:
    acquisition = json.loads((SEQUENCE_ROOT / "satellite_sequence_audit.json").read_text(encoding="utf-8"))
    verified = json.loads(SEQUENCE_AUDIT.read_text(encoding="utf-8"))
    index_path = SEQUENCE_ROOT / "satellite_sequence_index.parquet"
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    if (not acquisition["run_finished"] or verified["error_count"] or not verified["full_coverage_verified"]
            or verified["image_index_sha256"] != index_sha
            or verified["treatment_list_sha256"] != hashlib.sha256(TREATMENT_UNIT_LIST.read_bytes()).hexdigest()):
        raise ValueError("Only the complete, independently audited sequence index can be published")
    images = pd.read_parquet(index_path)
    state = pd.read_parquet(output / "state_inputs.parquet")
    context = pd.read_parquet(output / "context_inputs.parquet")
    pre = validate_sequence_binding(state, context, images, acquisition["contract"]["latest_complete_year"])
    # Preserve the convenient baseline alias while publishing the full six-year state sequence.
    attach_image_references(output, sequence_refresh=True)
    state = pd.read_parquet(output / "state_inputs.parquet")
    counts = pre.groupby("treatment_order").state_image_available.sum()
    state["satellite_sequence_available_years"] = state.treatment_order.map(counts).astype(int)
    state["satellite_sequence_any_available"] = state.satellite_sequence_available_years.gt(0)
    state["satellite_sequence_all_six_available"] = state.satellite_sequence_available_years.eq(6)
    state.to_parquet(output / "state_inputs.parquet", index=False, compression="zstd")
    sequence_path = output / "satellite_state_sequences.parquet"
    pre.to_parquet(sequence_path, index=False, compression="zstd")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(bundle_status="tabular_and_annual_satellite_sequences_published",
                    satellite_image_root=SEQUENCE_ROOT.relative_to(ROOT).as_posix(),
                    satellite_full_sequence_index=index_path.relative_to(ROOT).as_posix(),
                    satellite_full_sequence_index_sha256=index_sha,
                    satellite_full_event_times=list(range(-6, 4)),
                    satellite_state_sequence_table=sequence_path.name,
                    satellite_state_sequence_sha256=hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
                    satellite_state_event_times=list(range(-6, 0)),
                    satellite_state_frame_rows=len(pre),
                    satellite_state_available_frames=int(pre.state_image_available.sum()),
                    satellite_state_events_with_all_six=int(state.satellite_sequence_all_six_available.sum()),
                    satellite_incomplete_year_frames=int((~images.calendar_year_complete).sum()),
                    satellite_opening_and_post_frames_predictor_allowed=False,
                    satellite_baseline_alias_event_time=-1,
                    satellite_sequence_audit=SEQUENCE_AUDIT.relative_to(ROOT).as_posix())
    manifest["updated_at_utc"] = datetime.now(UTC).isoformat()
    manifest["alphaearth_source"] = {"path": AE_PATH.relative_to(ROOT).as_posix(),
                                     "sha256": hashlib.sha256(AE_PATH.read_bytes()).hexdigest()}
    manifest["intervention_source_manifest"] = {"path": (T_ROOT / "manifest.json").relative_to(ROOT).as_posix(),
                                                "sha256": hashlib.sha256((T_ROOT / "manifest.json").read_bytes()).hexdigest()}
    manifest["availability_counts"]["state"] = {name: int(state[name].sum()) for name in state if name.endswith("_available")}
    manifest["input_table_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                       for name in ["state_inputs.parquet", "context_inputs.parquet",
                                                    "intervention_inputs.parquet", "city_background_proxies.parquet",
                                                    "satellite_state_sequences.parquet"]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    media = parser.add_mutually_exclusive_group()
    media.add_argument("--refresh-media", action="store_true")
    media.add_argument("--refresh-sequences", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.refresh_media:
        attach_image_references(OUTPUT)
        return 0
    if args.refresh_sequences:
        attach_sequence_references(OUTPUT)
        return 0
    treatments = pd.read_parquet(TREATMENT_UNIT_LIST).sort_values("treatment_order")
    ae = pd.read_parquet(AE_PATH)
    states, contexts, city_records, sources = [], [], [], {}
    for city, group in treatments.groupby("city_key"):
        identity = group[KEYS].reset_index(drop=True)
        state, context = identity.copy(), identity.copy()
        for modality, spec in SPECS.items():
            annual, summary, source = source_city_features(city, modality, set(group.grid_id))
            sources[f"{city}/{modality}"] = source
            lagged = attach_annual_lags(identity, annual, list(spec["columns"].values()))
            target = state if spec["role"] == "state" else context
            for field in lagged.columns.difference(KEYS):
                target[field] = lagged[field]
            if not summary.empty:
                summary["city_key"] = city
                city_records.append(summary)
                query = pd.DataFrame({"year": pd.to_datetime(identity.opening_month).dt.year - 1})
                joined = query.merge(summary, on="year", how="left", validate="many_to_one", sort=False)
                for field in summary.columns.difference(["year", "city_key"]):
                    context[field] = joined[field].to_numpy()
                    context[f"{field}_available"] = pd.notna(joined[field]).to_numpy()
        states.append(state)
        contexts.append(context)
        logging.info("Published tabular inputs for %s (%d treatments)", city, len(group))
    state = pd.concat(states, ignore_index=True).sort_values("treatment_order")
    context = pd.concat(contexts, ignore_index=True).sort_values("treatment_order")
    ae_join = context[KEYS].merge(ae, on="treatment_order", validate="one_to_one", how="left", suffixes=("", "_ae"))
    for key in ["city_key", "grid_id", "opening_month"]:
        if not ae_join[key].eq(ae_join[f"{key}_ae"]).all():
            raise ValueError("AlphaEarth identity mismatch")
    if ae_join.is_temporal_proxy.any() or ae_join.is_dry_run.any():
        raise ValueError("Proxy/dry-run AlphaEarth cannot enter the main context table")
    if not ae_join.target_year.eq(pd.to_datetime(ae_join.opening_month).dt.year - 1).all():
        raise ValueError("AlphaEarth target year violates pretreatment rule")
    alphaearth_fields = {f"alphaearth_{band}": ae_join[band].to_numpy() for band in [f"A{i:02d}" for i in range(64)]}
    alphaearth_fields.update(alphaearth_available=ae_join.alphaearth_available.to_numpy(),
                            alphaearth_grid_geometry_sha256=ae_join.grid_geometry_sha256.to_numpy())
    context = pd.concat([context.reset_index(drop=True), pd.DataFrame(alphaearth_fields)], axis=1)
    for frame in [state, context]:
        for field in frame.columns:
            if field.endswith("_available"):
                frame[field] = frame[field].fillna(False).astype(bool)
    state["streetview_available"] = False
    state["streetview_status"] = "deferred_by_user_not_verified_locally"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state.to_parquet(OUTPUT / "state_inputs.parquet", index=False, compression="zstd")
    context.to_parquet(OUTPUT / "context_inputs.parquet", index=False, compression="zstd")
    intervention = pd.read_parquet(T_ROOT / "intervention_inputs.parquet")
    if not intervention[KEYS].reset_index(drop=True).equals(treatments[KEYS].reset_index(drop=True)):
        raise ValueError("Intervention input identity disagrees with frozen treatments")
    intervention.to_parquet(OUTPUT / "intervention_inputs.parquet", index=False, compression="zstd")
    city_summary = pd.concat(city_records, ignore_index=True).groupby(["city_key", "year"], as_index=False).first()
    city_summary.to_parquet(OUTPUT / "city_background_proxies.parquet", index=False, compression="zstd")
    manifest = {
        "built_at_utc": datetime.now(UTC).isoformat(), "rows": len(treatments), "cities": int(treatments.city_key.nunique()),
        "bundle_status": "tabular_published_media_pending", "lags": LAGS,
        "identity_columns": KEYS, "treatment_list_sha256": hashlib.sha256(TREATMENT_UNIT_LIST.read_bytes()).hexdigest(),
        "state_numeric_fields": [name for spec in SPECS.values() if spec["role"] == "state" for name in spec["columns"].values()],
        "context_numeric_fields": [name for spec in SPECS.values() if spec["role"] == "context" for name in spec["columns"].values()],
        "alphaearth_role": "context_only", "alphaearth_available": int(context.alphaearth_available.sum()),
        "city_background_fields": list(SUMMARY_NAMES.values()),
        "city_background_provenance": "all_observed_reference_grids_in_city_source_not_treatment_sample_or_official_yearbook",
        "city_coverage_denominator": "all_frozen_city_reference_grids",
        "fit_preprocessing": False, "response_supervision_included": False, "model_migration_completed": False,
        "streetview_status": "deferred_by_user", "sources": sources,
        "availability_counts": {"state": {name: int(state[name].sum()) for name in state if name.endswith("_available")},
                                "context": {name: int(context[name].sum()) for name in context if name.endswith("_available")}},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
