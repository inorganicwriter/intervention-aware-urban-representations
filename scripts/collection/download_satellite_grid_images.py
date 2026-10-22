"""Download independent, strictly pretreatment multispectral grid images from GEE.

Each retained treatment uses its frozen 500 m polygon and the calendar year
before opening. Output is six surface-reflectance bands, a validity mask and
clear-observation count in a georeferenced TIFF; RGB PNGs are previews only.
Resume validates the extraction contract, geometry and TIFF checksum.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine
from shapely import wkt
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.collection.extract_alphaearth_embeddings import (  # noqa: E402
    bind_grid_polygons,
    grid_polygon_coordinates,
)
from urban_intervention.config.project import CITIES  # noqa: E402
from urban_intervention.data.paths import ACTIVE_DIR, TREATMENT_UNIT_LIST  # noqa: E402

LOG = logging.getLogger("satellite_images")
DEFAULT_OUTPUT = ACTIVE_DIR / "satellite_grid_images_v1"
BANDS = ["red", "green", "blue", "nir", "swir1", "swir2"]
PRODUCTS = {
    "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "landsat8": "LANDSAT/LC08/C02/T1_L2",
    "landsat7": "LANDSAT/LE07/C02/T1_L2",
    "landsat5": "LANDSAT/LT05/C02/T1_L2",
}
CONTRACT = {
    "version": 1,
    "spatial_support": "fixed_grid_polygon_v1",
    "time_window": "full_calendar_year_before_opening",
    "bands": BANDS,
    "composite": "per_band_clear_pixel_median",
    "s2_scl_excluded": [0, 1, 3, 8, 9, 10, 11],
    "s2_scene_cloud_percentage_max": 80,
    "landsat_qa_pixel_excluded_bits": [0, 1, 2, 3, 4, 5],
    "landsat_qa_radsat": 0,
    "resampling": "nearest",
    "invalid_reflectance": "NaN",
    "preview_rgb_range": [0, 0.3],
}


def sensor_candidates(year: int) -> list[str]:
    if not 1984 <= year < datetime.now(UTC).year:
        raise ValueError(f"Unsupported or incomplete calendar year: {year}")
    if year >= 2018:
        return ["sentinel2", "landsat8"]
    if year >= 2013:
        return ["landsat8", "landsat7"]
    if year == 2012:
        return ["landsat7"]
    return ["landsat5", "landsat7"]


def raster_spec(geometry_wkt: str, crs: str, sensor: str) -> dict:
    polygon = wkt.loads(geometry_wkt)
    projected = transform_geometry(
        Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform, polygon
    )
    xmin, ymin, xmax, ymax = projected.bounds
    if not np.allclose([xmax - xmin, ymax - ymin], [500, 500], atol=0.05):
        raise ValueError("Frozen polygon is not a 500 m projected cell")
    width = 50 if sensor == "sentinel2" else 17
    affine = Affine((xmax - xmin) / width, 0, xmin, 0, -(ymax - ymin) / width, ymax)
    return {
        "crs": crs, "width": width, "height": width,
        "crs_transform": list(affine)[:6],
        "bounds": [xmin, ymin, xmax, ymax],
        "output_pixel_size_m": float(affine.a),
        "native_band_resolution_m": [10, 10, 10, 10, 20, 20]
        if sensor == "sentinel2" else [30] * 6,
    }


def decode_pixels(payload: bytes, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    expected = BANDS + ["valid_mask", "observation_count"]
    if array.shape != (size, size) or not set(expected).issubset(array.dtype.names or ()):
        raise ValueError("Downloaded array shape or bands do not match extraction contract")
    pixels = np.stack([array[band].astype(np.float32) for band in BANDS])
    count = array["observation_count"].astype(np.float32)
    valid = (array["valid_mask"] > 0) & (count > 0) & np.isfinite(pixels).all(axis=0)
    pixels[:, ~valid] = np.nan
    count[~valid] = 0
    return pixels, valid, count


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resume_record(path: Path, row: dict, contract_sha: str, output: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        for field in ["city_key", "grid_id", "treatment_order", "target_year", "grid_geometry_sha256"]:
            if rec[field] != row[field]:
                return None
        if rec["contract_sha256"] != contract_sha:
            return None
        if not rec.get("satellite_image_available", False) and any(
            attempt["reason"] == "download_failed" for attempt in rec.get("sensor_attempts", [])
        ):
            return None
        # Failed acquisitions are retried; real low-coverage images are retained.
        if not rec.get("image_path"):
            return None
        image_path = output / rec["image_path"]
        if not image_path.exists() or _checksum(image_path) != rec["image_sha256"]:
            return None
        if any(not (output / rec[field]).exists() for field in ["preview_path", "mask_path", "source_manifest_path"]):
            return None
        return rec
    except (KeyError, OSError, ValueError):
        return None


def masked_collection(sensor: str, year: int, bounds):
    import ee

    collection = ee.ImageCollection(PRODUCTS[sensor]).filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    ).filterBounds(bounds)
    if sensor == "sentinel2":
        collection = collection.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))

    def mask_image(image):
        if sensor == "sentinel2":
            scl = image.select("SCL")
            clear = scl.neq(0)
            for value in CONTRACT["s2_scl_excluded"][1:]:
                clear = clear.And(scl.neq(value))
            selected = image.select(["B4", "B3", "B2", "B8", "B11", "B12"], BANDS)
            reflectance = selected.multiply(0.0001)
        else:
            clear = image.select("QA_PIXEL").bitwiseAnd(63).eq(0).And(
                image.select("QA_RADSAT").eq(0)
            )
            names = ["SR_B4", "SR_B3", "SR_B2", "SR_B5", "SR_B6", "SR_B7"]
            if sensor != "landsat8":
                names = ["SR_B3", "SR_B2", "SR_B1", "SR_B4", "SR_B5", "SR_B7"]
            selected = image.select(names, BANDS)
            reflectance = selected.multiply(0.0000275).add(-0.2)
        # All six bands must be supported by a clear observation.
        joint_mask = selected.mask().reduce(ee.Reducer.min()).And(clear)
        return reflectance.updateMask(joint_mask).copyProperties(image, ["system:time_start"])

    return collection, collection.map(mask_image)


def acquire_patch(row: dict, sensor: str, raw, masked, output: Path) -> dict:
    import ee

    geometry = ee.Geometry.Polygon(grid_polygon_coordinates(row["geometry_wkt"]),
                                   proj="EPSG:4326", geodesic=False)
    local_raw = raw.filterBounds(geometry)
    provenance = ee.Dictionary({
        "scene_ids": local_raw.aggregate_array("system:index"),
        "scene_times_ms": local_raw.aggregate_array("system:time_start"),
    }).getInfo()
    if not provenance["scene_ids"]:
        return {"satellite_image_available": False, "availability_reason": "no_source_scenes"}
    # Validate actual acquisition dates as well as the requested filter.
    start_ms = int(pd.Timestamp(f"{row['target_year']}-01-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(f"{row['target_year'] + 1}-01-01", tz="UTC").timestamp() * 1000)
    if not all(start_ms <= value < end_ms for value in provenance["scene_times_ms"]):
        raise ValueError("Source scene outside requested calendar year")
    local = masked.filterBounds(geometry)
    median = local.median().select(BANDS)
    count = local.select("red").count().rename("observation_count").unmask(0)
    valid = median.mask().reduce(ee.Reducer.min()).unmask(0).rename("valid_mask")
    image = median.unmask(-9999).addBands([valid, count]).clip(geometry).toFloat()
    spec = raster_spec(row["geometry_wkt"], row["projected_crs"], sensor)
    url = image.getDownloadURL({
        "crs": spec["crs"], "crs_transform": spec["crs_transform"],
        "dimensions": [spec["width"], spec["height"]], "format": "NPY",
    })
    response = requests.get(url, timeout=(20, 100))
    response.raise_for_status()
    pixels, valid_pixels, observations = decode_pixels(response.content, spec["width"])
    return write_patch_assets(row, sensor, output, pixels, valid_pixels, observations, provenance, spec)


def write_patch_assets(row: dict, sensor: str, output: Path, pixels: np.ndarray,
                       valid_pixels: np.ndarray, observations: np.ndarray,
                       provenance: dict, spec: dict) -> dict:
    """Persist independently georeferenced frames; packed transport is never saved as geography."""
    fraction = float(valid_pixels.mean())
    stem = f"{int(row['treatment_order']):06d}_{row['target_year']}_{sensor}"
    folder = output / "images" / row["city_key"]
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / f"{stem}.tif"
    temporary = image_path.with_suffix(".tif.tmp")
    stack = np.concatenate([pixels, valid_pixels[None].astype(np.float32), observations[None]])
    with rasterio.open(temporary, "w", driver="GTiff", width=spec["width"], height=spec["height"],
                       count=8, dtype="float32", crs=spec["crs"],
                       transform=Affine(*spec["crs_transform"]), nodata=np.nan,
                       compress="deflate") as raster:
        raster.write(stack)
        for i, name in enumerate(BANDS + ["valid_mask", "observation_count"], 1):
            raster.set_band_description(i, name)
        raster.update_tags(city_key=row["city_key"], grid_id=row["grid_id"],
                           target_year=row["target_year"], source_product=PRODUCTS[sensor],
                           grid_geometry_sha256=row["grid_geometry_sha256"])
    temporary.replace(image_path)
    rgb = np.moveaxis(np.nan_to_num(np.clip(pixels[:3] / 0.3, 0, 1), nan=0), 0, -1)
    preview_path = folder / f"{stem}_rgb.png"
    mask_path = folder / f"{stem}_mask.png"
    Image.fromarray((rgb * 255).round().astype(np.uint8)).save(preview_path)
    Image.fromarray(valid_pixels.astype(np.uint8) * 255).save(mask_path)
    source_path = output / "sources" / row["city_key"] / f"{stem}.json"
    _write_json(source_path, {"source_product": PRODUCTS[sensor], **provenance})
    available = fraction >= row["min_valid_fraction"]
    return {
        "satellite_image_available": available,
        "availability_reason": "available" if available else "insufficient_valid_pixels",
        "sensor": sensor, "source_product": PRODUCTS[sensor],
        "image_path": image_path.relative_to(output).as_posix(),
        "image_sha256": _checksum(image_path),
        "preview_path": preview_path.relative_to(output).as_posix(),
        "mask_path": mask_path.relative_to(output).as_posix(),
        "source_manifest_path": source_path.relative_to(output).as_posix(),
        "scene_count": len(provenance["scene_ids"]),
        "source_date_min": pd.to_datetime(min(provenance["scene_times_ms"]), unit="ms", utc=True).isoformat(),
        "source_date_max": pd.to_datetime(max(provenance["scene_times_ms"]), unit="ms", utc=True).isoformat(),
        "valid_pixel_fraction": fraction,
        "clear_observation_count_median": float(np.median(observations[valid_pixels]))
        if valid_pixels.any() else 0.0,
        **spec,
    }


def collect_group(group: pd.DataFrame, output: Path, contract_sha: str, dry_run: bool) -> list[dict]:
    import ee

    city, year = str(group.city_key.iloc[0]), int(group.target_year.iloc[0])
    pending, results = [], []
    for row in group.to_dict(orient="records"):
        record_path = output / "records" / city / f"{row['treatment_order']:06d}.json"
        previous = resume_record(record_path, row, contract_sha, output)
        if previous is not None and not dry_run:
            results.append(previous)
        else:
            pending.append((row, record_path))
    if not pending:
        return results
    collections = {}
    if not dry_run:
        envelope = unary_union([wkt.loads(row["geometry_wkt"]) for row, _ in pending]).bounds
        bounds = ee.Geometry.Rectangle(list(envelope), proj="EPSG:4326", geodesic=False)
        for sensor in sensor_candidates(year):
            collections[sensor] = masked_collection(sensor, year, bounds)
    for row, record_path in pending:
        meta = {key: row[key] for key in ["treatment_order", "city_key", "grid_id", "opening_month",
                                         "target_year", "geometry_wkt", "grid_geometry_sha256"]}
        meta.update(contract_sha256=contract_sha, spatial_support="fixed_grid_polygon_v1",
                    is_temporal_proxy=False, is_dry_run=dry_run,
                    acquired_at_utc=datetime.now(UTC).isoformat())
        if dry_run:
            results.append({**meta, "satellite_image_available": False, "availability_reason": "dry_run"})
            continue
        attempts, best = [], None
        for sensor in sensor_candidates(year):
            for attempt in range(3):
                try:
                    candidate = acquire_patch(row, sensor, *collections[sensor], output)
                    attempts.append({"sensor": sensor, "reason": candidate["availability_reason"]})
                    if best is None or candidate.get("valid_pixel_fraction", -1) > best.get("valid_pixel_fraction", -1):
                        best = candidate
                    break
                except Exception as exc:
                    # Log type and bounded error; never log download URLs or credentials.
                    LOG.warning("%s/%s %s attempt %d: %s", city, row["treatment_order"],
                                sensor, attempt + 1, str(exc).split("https://")[0][:250])
                    if attempt == 2:
                        attempts.append({"sensor": sensor, "reason": "download_failed", "error_type": type(exc).__name__})
                    else:
                        time.sleep(attempt + 1)
            if best is not None and best["satellite_image_available"]:
                break
        if best is None:
            best = {"satellite_image_available": False, "availability_reason": "download_failed"}
        rec = {**meta, **best, "sensor_attempts": attempts}
        _write_json(record_path, rec)
        results.append(rec)
        LOG.info("%s/%s %d %s valid=%.3f", city, row["treatment_order"], year,
                 rec["availability_reason"], rec.get("valid_pixel_fraction", 0))
    return results


def write_index(records: list[dict], output: Path, contract: dict, selected_count: int, final: bool) -> None:
    frame = pd.DataFrame(records).sort_values("treatment_order")
    # Complex metadata lives in per-treatment JSON, while the index stays scalar.
    scalar = frame.drop(columns=["sensor_attempts", "native_band_resolution_m", "crs_transform", "bounds"], errors="ignore")
    temporary = output / "satellite_image_index.parquet.tmp"
    scalar.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output / "satellite_image_index.parquet")
    audit = {
        "updated_at_utc": datetime.now(UTC).isoformat(), "run_finished": final,
        "selected_treatments": selected_count, "indexed_treatments": len(frame),
        "available": int(frame.satellite_image_available.sum()),
        "retained_images": int(frame.get("image_path", pd.Series(dtype=str)).notna().sum()),
        "availability_reasons": frame.availability_reason.value_counts().to_dict(),
        "sensors": frame.get("sensor", pd.Series(dtype=str)).value_counts().to_dict(),
        "contract": contract,
    }
    _write_json(output / "satellite_image_audit.json", audit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--project", default="macro-city-engine")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--city", default="all")
    parser.add_argument("--orders", type=int, nargs="+")
    parser.add_argument("--pilot", action="store_true", help="One example for each sensor era")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    args = parser.parse_args()
    if args.workers < 1 or not 0 < args.min_valid_fraction <= 1:
        parser.error("workers must be positive; valid fraction must lie in (0, 1]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    treatments = pd.read_parquet(TREATMENT_UNIT_LIST)
    if args.city != "all":
        treatments = treatments.loc[treatments.city_key.eq(args.city)]
    treatments = treatments.copy()
    treatments["target_year"] = pd.to_datetime(treatments.opening_month).dt.year - 1
    if args.orders:
        treatments = treatments.loc[treatments.treatment_order.isin(args.orders)]
    if args.pilot:
        treatments = pd.concat([treatments.loc[treatments.target_year.eq(year)].head(1)
                                for year in [2010, 2012, 2015, 2023]], ignore_index=True)
    if treatments.empty:
        parser.error("No matching treatment units")
    bound = pd.concat([bind_grid_polygons(city, group) for city, group in treatments.groupby("city_key")], ignore_index=True)
    bound["projected_crs"] = bound.city_key.map(lambda city: CITIES[city]["projected_crs"])
    bound["min_valid_fraction"] = args.min_valid_fraction
    # Validate all raster extents before contacting the service.
    for row in bound.to_dict(orient="records"):
        raster_spec(row["geometry_wkt"], row["projected_crs"], sensor_candidates(row["target_year"])[0])
    contract = {**CONTRACT, "min_valid_fraction": args.min_valid_fraction}
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        import ee
        ee.Initialize(project=args.project)
        ee.data.setDeadline(60000)
    groups = list(bound.groupby(["city_key", "target_year"]))
    LOG.info("Collecting %d treatments in %d city/year groups", len(bound), len(groups))
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect_group, group, args.output_dir, contract_sha, args.dry_run): key
                   for key, group in groups}
        for future in as_completed(futures):
            records.extend(future.result())
            write_index(records, args.output_dir, contract, len(bound), False)
            LOG.info("Indexed %d/%d treatments", len(records), len(bound))
    write_index(records, args.output_dir, contract, len(bound), True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
