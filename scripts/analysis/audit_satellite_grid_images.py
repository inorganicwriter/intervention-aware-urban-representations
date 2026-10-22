"""Verify every downloaded image against frozen geometry, source dates and masks."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from shapely import wkt
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.config.project import CITIES  # noqa: E402
from urban_intervention.data.paths import ACTIVE_DIR, TREATMENT_UNIT_LIST  # noqa: E402

IMAGE_ROOT = ACTIVE_DIR / "satellite_grid_images_v1"
BANDS = ("red", "green", "blue", "nir", "swir1", "swir2", "valid_mask", "observation_count")


@lru_cache(maxsize=64)
def city_transformer(city: str):
    return Transformer.from_crs("EPSG:4326", CITIES[city]["projected_crs"], always_xy=True)


def verify_frame(task):
    row_values, ref_values, image_root, sequence, min_valid_fraction, contract_sha, latest_complete_year = task
    row = SimpleNamespace(**row_values)
    try:
        if ref_values is None:
            raise ValueError("unknown_treatment_order")
        ref = SimpleNamespace(**ref_values)
        for key in ["city_key", "grid_id", "opening_month", "grid_geometry_sha256", "geometry_wkt"]:
            if getattr(row, key) != getattr(ref, key):
                raise ValueError(f"identity_or_geometry_mismatch:{key}")
        event_time = int(row.event_time) if sequence else -1
        target_year = pd.Timestamp(ref.opening_month).year + event_time
        if row.target_year != target_year or row.is_temporal_proxy or row.is_dry_run:
            raise ValueError("invalid_temporal_contract")
        record_stem = f"{row.treatment_order:06d}_{target_year}" if sequence else f"{row.treatment_order:06d}"
        record_path = image_root / "records" / row.city_key / f"{record_stem}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for key in ["treatment_order", "city_key", "grid_id", "opening_month", "target_year", "geometry_wkt"]:
            if record[key] != getattr(row, key):
                raise ValueError(f"record_index_mismatch:{key}")
        if record["contract_sha256"] != contract_sha or record["grid_geometry_sha256"] != row.grid_geometry_sha256:
            raise ValueError("record_contract_or_geometry_mismatch")
        if sequence:
            complete = target_year <= latest_complete_year
            phase = "pre" if event_time < 0 else "opening_year" if event_time == 0 else "post"
            if (bool(row.calendar_year_complete) != complete or row.temporal_phase != phase
                    or bool(row.state_temporal_allowed) != (event_time < 0)
                    or bool(row.state_image_available) != (row.satellite_image_available and event_time < 0)):
                raise ValueError("invalid_predictor_or_calendar_mask")
            if not complete and (row.satellite_image_available or row.availability_reason != "calendar_year_not_complete"
                                 or isinstance(getattr(row, "image_path", None), str)):
                raise ValueError("incomplete_year_claimed_as_annual_image")
        if not isinstance(getattr(row, "image_path", None), str):
            if row.satellite_image_available:
                raise ValueError("available_image_missing_path")
            return
        path = image_root / row.image_path
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.image_sha256:
            raise ValueError("image_checksum_mismatch")
        projected = transform(city_transformer(row.city_key).transform, wkt.loads(ref.geometry_wkt))
        with rasterio.open(path) as raster:
            pixels = raster.read()
            if raster.descriptions != BANDS or raster.count != 8:
                raise ValueError("wrong_bands")
            if str(raster.crs) != CITIES[row.city_key]["projected_crs"]:
                raise ValueError("wrong_crs")
            if not np.allclose(raster.bounds, projected.bounds, atol=0.001, rtol=0):
                raise ValueError("image_extent_does_not_match_frozen_grid")
            size = 50 if row.sensor == "sentinel2" else 17
            if (raster.width, raster.height) != (size, size):
                raise ValueError("wrong_image_size")
            if not np.isin(pixels[6], [0, 1]).all():
                raise ValueError("invalid_mask_values")
            valid = pixels[6] == 1
            if not np.isfinite(pixels[:6, valid]).all() or not np.isnan(pixels[:6, ~valid]).all():
                raise ValueError("reflectance_mask_mismatch")
            minimum = 0.0 if row.sensor == "sentinel2" else -0.2
            maximum = 6.5535 if row.sensor == "sentinel2" else 65535 * 0.0000275 - 0.2
            if valid.any() and ((pixels[:6, valid] < minimum - 1e-6).any()
                                or (pixels[:6, valid] > maximum + 1e-6).any()):
                raise ValueError("reflectance_outside_scaled_uint16_range_or_unmasked_fill_value")
            if not (pixels[7, valid] > 0).all() or not (pixels[7, ~valid] == 0).all():
                raise ValueError("observation_count_mask_mismatch")
            fraction = float(valid.mean())
            if not np.isclose(fraction, row.valid_pixel_fraction):
                raise ValueError("valid_fraction_mismatch")
            if bool(row.satellite_image_available) != (fraction >= min_valid_fraction):
                raise ValueError("availability_threshold_mismatch")
            with Image.open(image_root / row.mask_path) as mask:
                if not np.array_equal(np.array(mask), valid.astype(np.uint8) * 255):
                    raise ValueError("mask_preview_mismatch")
            with Image.open(image_root / row.preview_path) as preview:
                if preview.size != (size, size):
                    raise ValueError("rgb_preview_size_mismatch")
        provenance = json.loads((image_root / row.source_manifest_path).read_text(encoding="utf-8"))
        if provenance["source_product"] != row.source_product:
            raise ValueError("source_product_mismatch")
        times = pd.to_datetime(provenance["scene_times_ms"], unit="ms", utc=True)
        if len(times) != row.scene_count or len(provenance["scene_ids"]) != row.scene_count:
            raise ValueError("source_scene_count_mismatch")
        opening_cutoff = pd.Timestamp(ref.opening_date, tz="Asia/Shanghai").tz_convert("UTC")
        if not (times.year == target_year).all():
            raise ValueError("source_date_outside_calendar_year")
        if event_time < 0 and not (times < opening_cutoff).all():
            raise ValueError("state_source_date_not_pretreatment")
        if pixels[7].max() > row.scene_count or record["grid_geometry_sha256"] != row.grid_geometry_sha256:
            raise ValueError("record_or_scene_count_inconsistent")
        return ("summary", {"treatment_order": row.treatment_order, "city_key": row.city_key,
                          "target_year": target_year, "event_time": event_time, "sensor": row.sensor,
                          "available": bool(row.satellite_image_available),
                          "valid_pixel_fraction": fraction,
                          "reflectance_min": float(np.nanmin(pixels[:6])) if valid.any() else None,
                          "reflectance_max": float(np.nanmax(pixels[:6])) if valid.any() else None})
    except Exception as exc:
        return ("error", {"treatment_order": int(row.treatment_order), "target_year": int(row.target_year), "error": str(exc)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="store_true", help="Audit each event's -6..+3 annual frames")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-incomplete", action="store_true", help="Audit an in-progress snapshot; cannot qualify publication")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    image_root = args.image_root or (ACTIVE_DIR / "satellite_event_sequences_v2" if args.sequence else IMAGE_ROOT)
    index_name = "satellite_sequence" if args.sequence else "satellite_image"
    acquisition_audit = json.loads((image_root / f"{index_name}_audit.json").read_text(encoding="utf-8"))
    if not acquisition_audit["run_finished"] and not args.allow_incomplete:
        raise RuntimeError("Full image acquisition has not finished")
    min_valid_fraction = acquisition_audit["contract"]["min_valid_fraction"]
    index_bytes = (image_root / f"{index_name}_index.parquet").read_bytes()
    index = pd.read_parquet(io.BytesIO(index_bytes))
    treatments = pd.read_parquet(TREATMENT_UNIT_LIST)
    ae = pd.read_parquet(ACTIVE_DIR / "alphaearth_grid_v2/alphaearth_treated_embeddings.parquet")
    errors, summaries = [], []
    keys = ["treatment_order", "target_year"] if args.sequence else ["treatment_order"]
    if index.duplicated(keys).any() or set(index.treatment_order) != set(treatments.treatment_order):
        errors.append({"error": "index_does_not_cover_frozen_treatments_exactly"})
    if args.sequence:
        for event_time in range(-6, 4):
            part = index.loc[index.event_time.eq(event_time)]
            if set(part.treatment_order) != set(treatments.treatment_order) or len(part) != len(treatments):
                errors.append({"error": "incomplete_event_time_coverage", "event_time": event_time})
        if len(index) != 10 * len(treatments) or not index.event_time.between(-6, 3).all():
            errors.append({"error": "incorrect_sequence_window"})
    coverage_errors = errors.copy()
    if args.allow_incomplete:
        errors.clear()
    reference = treatments.merge(ae[["treatment_order", "grid_geometry_sha256", "geometry_wkt"]],
                                 on="treatment_order", validate="one_to_one").set_index("treatment_order")
    contract_sha = hashlib.sha256(json.dumps(acquisition_audit["contract"], sort_keys=True).encode()).hexdigest()
    reference_rows = reference.to_dict(orient="index")
    tasks = ((row, reference_rows.get(row["treatment_order"]), image_root, args.sequence,
              min_valid_fraction, contract_sha, acquisition_audit["contract"].get("latest_complete_year"))
             for row in index.to_dict(orient="records"))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for position, result in enumerate(pool.map(verify_frame, tasks, chunksize=32), 1):
            if result is not None:
                kind, value = result
                (summaries if kind == "summary" else errors).append(value)
            if position % 5000 == 0:
                print(f"Audited {position}/{len(index)} indexed frames; errors={len(errors)}", flush=True)
    summary = pd.DataFrame(summaries)
    output = args.output_dir or ROOT / ("outputs/satellite_event_sequences_audit/2026-10-13" if args.sequence
                                       else "outputs/satellite_grid_images_audit/2026-10-13")
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "image_quality_by_frame.csv", index=False, encoding="utf-8")
    report = {
        "audited_at_utc": datetime.now(UTC).isoformat(), "expected_treatments": len(treatments),
        "indexed_frames": len(index), "verified_images": len(summary),
        "available_images": int(index.satellite_image_available.sum()), "error_count": len(errors), "errors": errors,
        "sensor_counts": index.sensor.value_counts().to_dict(),
        "availability_reasons": index.availability_reason.value_counts().to_dict(),
        "valid_fraction_min": float(summary.valid_pixel_fraction.min()) if len(summary) else None,
        "all_image_geometries_match_frozen_grid": not errors,
        "all_state_source_dates_strictly_pretreatment": not errors,
        "full_event_window": list(range(-6, 4)) if args.sequence else [-1],
        "image_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "treatment_list_sha256": hashlib.sha256(TREATMENT_UNIT_LIST.read_bytes()).hexdigest(),
        "full_coverage_verified": acquisition_audit["run_finished"] and not coverage_errors and not errors,
        "coverage_errors": coverage_errors,
    }
    if args.sequence:
        report.update(expected_frames=10 * len(treatments),
                      incomplete_calendar_year_frames=int((~index.calendar_year_complete).sum()),
                      state_available_frames=int(index.state_image_available.sum()),
                      available_by_event_time=index.groupby("event_time").satellite_image_available.sum().to_dict(),
                      available_by_year=index.groupby("target_year").satellite_image_available.sum().to_dict(),
                      contract=acquisition_audit["contract"])
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, ensure_ascii=False, indent=2))
    if errors:
        print(json.dumps(errors[:10], ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
