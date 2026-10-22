"""Collect annual satellite frames for every frozen event from -6 through +3.

Packed strips are temporary network transport. Each frame is restored to its
own original georeference before saving. Post-opening frames cannot be state
predictors; incomplete calendar years remain explicit unavailable rows.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from shapely import wkt
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.collection import download_satellite_grid_images as base  # noqa: E402
from urban_intervention.config.project import CITIES  # noqa: E402
from urban_intervention.data.paths import ACTIVE_DIR, TREATMENT_UNIT_LIST  # noqa: E402

LOG = logging.getLogger("satellite_sequences")
OUTPUT = ACTIVE_DIR / "satellite_event_sequences_v2"
WINDOW = list(range(-6, 4))


def build_queue(treatments: pd.DataFrame, latest_complete_year: int) -> pd.DataFrame:
    if treatments.treatment_order.duplicated().any() or treatments.duplicated(["city_key", "grid_id"]).any():
        raise ValueError("Frozen event/grid identities must be unique")
    years = pd.to_datetime(treatments.opening_month).dt.year
    parts = []
    for event_time in WINDOW:
        part = treatments.copy()
        part["target_year"] = years + event_time
        part["event_time"] = event_time
        part["temporal_phase"] = "pre" if event_time < 0 else "opening_year" if event_time == 0 else "post"
        part["calendar_year_complete"] = part.target_year.le(latest_complete_year)
        part["state_temporal_allowed"] = event_time < 0
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(["treatment_order", "target_year"]).reset_index(drop=True)


def metadata(row: dict, contract_sha: str) -> dict:
    fields = ["treatment_order", "city_key", "grid_id", "station_event_id", "opening_month",
              "target_year", "event_time", "temporal_phase", "calendar_year_complete",
              "state_temporal_allowed", "geometry_wkt", "grid_geometry_sha256"]
    return {**{field: row[field] for field in fields}, "contract_sha256": contract_sha,
            "spatial_support": "fixed_grid_polygon_v1", "is_temporal_proxy": False,
            "is_dry_run": False, "acquired_at_utc": datetime.now(UTC).isoformat()}


def record_path(output: Path, row: dict) -> Path:
    return output / "records" / row["city_key"] / f"{row['treatment_order']:06d}_{row['target_year']}.json"


def compatible_previous_contract(output: Path, contract: dict) -> str | None:
    path = output / "satellite_sequence_audit.json"
    if not path.exists():
        return None
    previous = json.loads(path.read_text(encoding="utf-8"))["contract"]
    def recipe(value: dict) -> dict:
        return {key: item for key, item in value.items() if key != "latest_complete_year"}
    if recipe(previous) != recipe(contract):
        return None
    return hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()


def resume_cached_frame(path: Path, row: dict, contract_sha: str, output: Path,
                        previous_contract_sha: str | None) -> dict | None:
    rec = base.resume_record(path, row, contract_sha, output)
    if rec is None and previous_contract_sha and previous_contract_sha != contract_sha:
        rec = base.resume_record(path, row, previous_contract_sha, output)
        if rec is not None:
            acquired_at = rec["acquired_at_utc"]
            rec.update(metadata(row, contract_sha))
            rec["acquired_at_utc"] = acquired_at
            rec["contract_updated_at_utc"] = datetime.now(UTC).isoformat()
            rec["state_image_available"] = bool(rec["satellite_image_available"] and row["state_temporal_allowed"])
            base._write_json(path, rec)
    return rec


def reuse_baseline(row: dict, output: Path, contract_sha: str) -> dict | None:
    if row["event_time"] != -1:
        return None
    source_record = base.DEFAULT_OUTPUT / "records" / row["city_key"] / f"{row['treatment_order']:06d}.json"
    if not source_record.exists():
        return None
    source = json.loads(source_record.read_text(encoding="utf-8"))
    rec = base.resume_record(source_record, row, source["contract_sha256"], base.DEFAULT_OUTPUT)
    if rec is None:
        return None
    for field in ["image_path", "preview_path", "mask_path", "source_manifest_path"]:
        original, target = base.DEFAULT_OUTPUT / rec[field], output / rec[field]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(original, target)
            except OSError:
                shutil.copy2(original, target)
    rec.update(metadata(row, contract_sha), reused_baseline_v1=True)
    rec["state_image_available"] = bool(rec["satellite_image_available"] and row["state_temporal_allowed"])
    base._write_json(record_path(output, row), rec)
    return rec


def acquire_packed(rows: list[dict], sensor: str, raw, masked, output: Path) -> list[dict]:
    import ee

    geometries = [ee.Geometry.Polygon(base.grid_polygon_coordinates(row["geometry_wkt"]),
                                     proj="EPSG:4326", geodesic=False) for row in rows]
    provenance = ee.List([ee.Dictionary({
        "scene_ids": raw.filterBounds(geometry).aggregate_array("system:index"),
        "scene_times_ms": raw.filterBounds(geometry).aggregate_array("system:time_start"),
    }) for geometry in geometries]).getInfo()
    specs = [base.raster_spec(row["geometry_wkt"], row["projected_crs"], sensor) for row in rows]
    size = specs[0]["width"]
    anchor = specs[0]["crs_transform"]
    moved, positions = [], []
    for position, (row, geometry, source, spec) in enumerate(zip(rows, geometries, provenance, specs, strict=True)):
        if not source["scene_ids"]:
            continue
        start = pd.Timestamp(f"{row['target_year']}-01-01", tz="UTC").timestamp() * 1000
        end = pd.Timestamp(f"{row['target_year'] + 1}-01-01", tz="UTC").timestamp() * 1000
        if not all(start <= timestamp < end for timestamp in source["scene_times_ms"]):
            raise ValueError("Source scene outside requested calendar year")
        local = masked.filterBounds(geometry)
        median = local.median().select(base.BANDS)
        count = local.select("red").count().rename("observation_count").unmask(0)
        valid = median.mask().reduce(ee.Reducer.min()).unmask(0).rename("valid_mask")
        patch = median.unmask(-9999).addBands([valid, count]).clip(geometry).toFloat()
        src_projection = ee.Projection(spec["crs"], spec["crs_transform"])
        destination = list(anchor)
        destination[5] += position * size * anchor[4]
        dst_projection = ee.Projection(spec["crs"], destination)
        moved.append(patch.reproject(src_projection).changeProj(src_projection, dst_projection))
        positions.append(position)
    results = [{"satellite_image_available": False, "availability_reason": "no_source_scenes"} for _ in rows]
    if not moved:
        return results
    strip = ee.ImageCollection.fromImages(moved).mosaic().unmask(-9999)
    url = strip.getDownloadURL({"crs": specs[0]["crs"], "crs_transform": anchor,
                                "dimensions": [size, size * len(rows)], "format": "NPY"})
    response = requests.get(url, timeout=(20, 180))
    if not response.ok:
        try:
            reason = response.json().get("error", {}).get("message", "")
        except ValueError:
            reason = ""
        raise RuntimeError(f"Packed pixels HTTP {response.status_code}: {reason.split('https://')[0][:220]}")
    response.raise_for_status()
    array = np.load(io.BytesIO(response.content), allow_pickle=False)
    if array.shape != (size * len(rows), size):
        raise ValueError("Packed download shape mismatch")
    for position in positions:
        payload = io.BytesIO()
        np.save(payload, array[position * size:(position + 1) * size])
        pixels, valid, observations = base.decode_pixels(payload.getvalue(), size)
        results[position] = base.write_patch_assets(rows[position], sensor, output, pixels, valid,
                                                   observations, provenance[position], specs[position])
    return results


def acquire_resilient(rows: list[dict], sensor: str, raw, masked, output: Path) -> list[dict]:
    """Split memory-limited requests; retry transient failures without losing frames."""
    for attempt in range(3):
        try:
            return acquire_packed(rows, sensor, raw, masked, output)
        except Exception as exc:
            message = str(exc).split("https://")[0][:220]
            if "memory limit" in str(exc).lower() and len(rows) > 1:
                midpoint = len(rows) // 2
                LOG.info("%s/%d %s splitting memory-limited batch %d", rows[0]["city_key"],
                         rows[0]["target_year"], sensor, len(rows))
                return (acquire_resilient(rows[:midpoint], sensor, raw, masked, output)
                        + acquire_resilient(rows[midpoint:], sensor, raw, masked, output))
            LOG.warning("%s/%d %s batch=%d attempt=%d %s", rows[0]["city_key"],
                        rows[0]["target_year"], sensor, len(rows), attempt + 1, message)
            time.sleep(attempt + 1)
    results = []
    for row in rows:
        try:
            results.append(base.acquire_patch(row, sensor, raw, masked, output))
        except Exception as exc:
            results.append({"satellite_image_available": False, "availability_reason": "download_failed",
                            "error_type": type(exc).__name__})
    return results


def collect_chunk(group: pd.DataFrame, output: Path, contract_sha: str, reuse: bool,
                  previous_contract_sha: str | None = None) -> list[dict]:
    import ee

    results, pending = [], []
    for row in group.to_dict(orient="records"):
        path = record_path(output, row)
        if not row["calendar_year_complete"]:
            rec = {**metadata(row, contract_sha), "satellite_image_available": False,
                   "state_image_available": False, "availability_reason": "calendar_year_not_complete"}
            base._write_json(path, rec)
            results.append(rec)
            continue
        rec = resume_cached_frame(path, row, contract_sha, output, previous_contract_sha)
        if rec is None and reuse:
            rec = reuse_baseline(row, output, contract_sha)
        if rec is not None:
            results.append(rec)
        else:
            pending.append(row)
    if not pending:
        return results
    year = pending[0]["target_year"]
    envelope = unary_union([wkt.loads(row["geometry_wkt"]) for row in pending]).bounds
    bounds = ee.Geometry.Rectangle(list(envelope), proj="EPSG:4326", geodesic=False)
    best, attempts = {}, {row["treatment_order"]: [] for row in pending}
    for sensor in base.sensor_candidates(year):
        remaining = [row for row in pending if not best.get(row["treatment_order"], {}).get("satellite_image_available", False)]
        if not remaining:
            break
        raw, masked = base.masked_collection(sensor, year, bounds)
        acquired = acquire_resilient(remaining, sensor, raw, masked, output)
        for row, candidate in zip(remaining, acquired, strict=True):
            key = row["treatment_order"]
            attempts[key].append({"sensor": sensor, "reason": candidate["availability_reason"]})
            if key not in best or candidate.get("valid_pixel_fraction", -1) > best[key].get("valid_pixel_fraction", -1):
                best[key] = candidate
    for row in pending:
        candidate = best[row["treatment_order"]]
        rec = {**metadata(row, contract_sha), **candidate, "sensor_attempts": attempts[row["treatment_order"]],
               "reused_baseline_v1": False,
               "state_image_available": bool(candidate["satellite_image_available"] and row["state_temporal_allowed"])}
        base._write_json(record_path(output, row), rec)
        results.append(rec)
    return results


def write_index(records: list[dict], output: Path, contract: dict, expected: int, finished: bool) -> None:
    frame = pd.DataFrame(records).sort_values(["treatment_order", "target_year"])
    scalar = frame.drop(columns=["sensor_attempts", "native_band_resolution_m", "crs_transform", "bounds"], errors="ignore")
    temporary = output / "satellite_sequence_index.parquet.tmp"
    scalar.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output / "satellite_sequence_index.parquet")
    audit = {"updated_at_utc": datetime.now(UTC).isoformat(), "run_finished": finished,
             "expected_frames": expected, "indexed_frames": len(frame),
             "available_frames": int(frame.satellite_image_available.sum()),
             "state_available_frames": int(frame.state_image_available.sum()),
             "reused_baseline_frames": int(frame.get("reused_baseline_v1", pd.Series(dtype=bool)).fillna(False).sum()),
             "availability_reasons": frame.availability_reason.value_counts().to_dict(),
             "contract": contract}
    base._write_json(output / "satellite_sequence_audit.json", audit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--orders", type=int, nargs="+")
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--no-reuse-baseline", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128 or args.workers < 1:
        parser.error("batch size must be 1..128; workers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    treatments = pd.read_parquet(TREATMENT_UNIT_LIST)
    if args.orders:
        treatments = treatments.loc[treatments.treatment_order.isin(args.orders)]
    if treatments.empty:
        parser.error("No selected treatment events")
    bound = pd.concat([base.bind_grid_polygons(city, group) for city, group in treatments.groupby("city_key")], ignore_index=True)
    bound["projected_crs"] = bound.city_key.map(lambda city: CITIES[city]["projected_crs"])
    bound["min_valid_fraction"] = 0.8
    latest_complete_year = datetime.now(UTC).year - 1
    queue = build_queue(bound, latest_complete_year)
    if args.years:
        queue = queue.loc[queue.target_year.isin(args.years)]
    if queue.empty:
        parser.error("No selected grid/year frames")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    queue.drop(columns=["projected_crs", "min_valid_fraction"]).to_parquet(output / "satellite_sequence_queue.parquet", index=False)
    contract = {**base.CONTRACT, "version": 2, "time_window": "annual_event_time_minus6_through_plus3",
                "event_times": WINDOW, "latest_complete_year": latest_complete_year,
                "min_valid_fraction": 0.8, "predictor_event_times": list(range(-6, 0))}
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    previous_contract_sha = compatible_previous_contract(output, contract)
    LOG.info("Queue: %d frames; %d incomplete calendar years", len(queue), int((~queue.calendar_year_complete).sum()))
    if args.plan_only:
        base._write_json(output / "collection_plan.json", {"expected_frames": len(queue),
                         "downloadable_year_frames": int(queue.calendar_year_complete.sum()), "contract": contract})
        return 0
    import ee
    ee.Initialize(project="macro-city-engine")
    ee.data.setDeadline(60000)
    chunks = []
    for (_, year), group in queue.groupby(["city_key", "target_year"]):
        size = min(args.batch_size, 8) if year >= 2018 else args.batch_size
        chunks.extend(group.iloc[start:start + size] for start in range(0, len(group), size))
    records, last_index = [], 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect_chunk, chunk, output, contract_sha, not args.no_reuse_baseline,
                               previous_contract_sha) for chunk in chunks]
        for future in as_completed(futures):
            records.extend(future.result())
            if time.monotonic() - last_index > 20:
                write_index(records, output, contract, len(queue), False)
                LOG.info("Indexed %d/%d frames", len(records), len(queue))
                last_index = time.monotonic()
    write_index(records, output, contract, len(queue), True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
