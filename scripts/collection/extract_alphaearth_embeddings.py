"""Extract Google AlphaEarth (64-d geospatial embeddings) for treated grids via GEE.

Dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
GEE Project: macro-city-engine
Temporal coverage: 2017 - 2025.

Vectorized extraction:
- Uses ee.ImageCollection.reduceRegions over batched FeatureCollections.
- Reduces network calls by grouping grids by (city_key, target_year).
- For each treated grid:
  - Pre-treatment year target: T_0 = opening_year - 1.
- Uses the frozen reference grid polygon, joined by city_key/grid_id.
- Events without a pre-treatment product year remain unavailable by default.
- Optional retrospective proxies require --allow-temporal-proxy.
  - Output: 64 continuous embedding bands (A00 - A63).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("alphaearth_extractor")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.data.paths import (  # noqa: E402
    ACTIVE_DIR,
    REFERENCE_GRID_DIR,
    TREATMENT_UNIT_LIST,
)

ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
DEFAULT_OUTPUT_DIR = ACTIVE_DIR / "alphaearth_grid_v2"
BANDS = [f"A{i:02d}" for i in range(64)]
SPATIAL_SUPPORT = "fixed_grid_polygon_v1"


def bind_grid_polygons(
    city_key: str,
    treatments: pd.DataFrame,
    grid_dir: Path = REFERENCE_GRID_DIR,
) -> pd.DataFrame:
    """Bind extraction to the same persisted polygons as the causal design."""
    if not treatments["city_key"].eq(city_key).all():
        raise ValueError("Treatment city does not match requested grid city")
    if treatments["treatment_order"].duplicated().any():
        raise ValueError("Duplicate treatment_order")
    path = grid_dir / city_key / f"{city_key}_grids.parquet"
    grids = pd.read_parquet(path, columns=["grid_id", "geometry_wkt"])
    if grids["grid_id"].duplicated().any():
        raise ValueError("Reference grid has duplicate grid_id")
    bound = treatments.merge(grids, on="grid_id", how="left", validate="many_to_one")
    if bound["geometry_wkt"].isna().any():
        raise ValueError("Treatment grid is missing from the frozen reference grid")
    for row in bound.itertuples(index=False):
        polygon = wkt.loads(row.geometry_wkt)
        if polygon.geom_type != "Polygon" or polygon.is_empty or not polygon.is_valid:
            raise ValueError(f"Invalid grid polygon: {city_key}/{row.grid_id}")
        if not polygon.covers(Point(float(row.wgs84_lon), float(row.wgs84_lat))):
            raise ValueError(f"Station is outside its bound grid: {city_key}/{row.grid_id}")
    bound["spatial_support"] = SPATIAL_SUPPORT
    bound["grid_geometry_sha256"] = bound["geometry_wkt"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    return bound


def grid_polygon_coordinates(geometry_wkt: str) -> list[list[list[float]]]:
    polygon = wkt.loads(geometry_wkt)
    return [list(map(list, ring.coords)) for ring in [polygon.exterior, *polygon.interiors]]


def init_ee(project: str = "macro-city-engine") -> None:
    import ee

    try:
        ee.Initialize(project=project)
        logger.info("Initialized Earth Engine with project '%s'", project)
    except Exception as exc:
        logger.error("Failed to initialize Earth Engine: %s", exc)
        raise RuntimeError(f"Earth Engine initialization failed: {exc}") from exc


def extract_city_alphaearth(
    city_key: str,
    city_treatments: pd.DataFrame,
    dry_run: bool = False,
    *,
    grid_dir: Path = REFERENCE_GRID_DIR,
    allow_temporal_proxy: bool = False,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Extract 64-d AlphaEarth embeddings for all treated units in a city using reduceRegions."""
    if city_treatments.empty:
        return pd.DataFrame()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    logger.info("Processing city '%s' with %d treated grids...", city_key, len(city_treatments))
    city_treatments = bind_grid_polygons(city_key, city_treatments, grid_dir)
    city_treatments["opening_year"] = pd.to_datetime(
        city_treatments["opening_month"]
    ).dt.year
    city_treatments["target_year"] = city_treatments["opening_year"] - 1
    city_treatments["is_temporal_proxy"] = False
    if allow_temporal_proxy:
        city_treatments["is_temporal_proxy"] = city_treatments["target_year"] < 2017
        city_treatments["target_year"] = city_treatments["target_year"].clip(lower=2017)

    def metadata(row: pd.Series) -> dict:
        return {
            "treatment_order": int(row["treatment_order"]),
            "city_key": city_key,
            "grid_id": str(row["grid_id"]),
            "opening_month": str(row["opening_month"]),
            "target_year": int(row["target_year"]),
            "is_temporal_proxy": bool(row["is_temporal_proxy"]),
            "spatial_support": SPATIAL_SUPPORT,
            "grid_geometry_sha256": row["grid_geometry_sha256"],
            "geometry_wkt": row["geometry_wkt"],
            "is_dry_run": bool(dry_run),
        }

    records = []
    unavailable = city_treatments["target_year"].lt(2017)
    for _, row in city_treatments.loc[unavailable | dry_run].iterrows():
        rec = metadata(row)
        rec["alphaearth_available"] = False
        rec["availability_reason"] = (
            "no_pretreatment_product_year" if row["target_year"] < 2017 else "dry_run"
        )
        rec.update({band: np.nan for band in BANDS})
        records.append(rec)
    if dry_run:
        return pd.DataFrame(records)

    by_year = city_treatments.loc[~unavailable].groupby("target_year")
    for year, group in by_year:
        for start in range(0, len(group), batch_size):
            batch = group.iloc[start : start + batch_size]
            records.extend(_extract_batch(city_key, int(year), batch, metadata))
    return pd.DataFrame(records)


def _extract_batch(city_key: str, year_int: int, group: pd.DataFrame, metadata) -> list[dict]:
    import ee

    records = []
    if not group.empty:
        logger.info("  Extracting year %d for %d units in %s...", year_int, len(group), city_key)

        features = []
        meta_lookup = {}
        for _, row in group.iterrows():
            t_order = int(row["treatment_order"])
            geom = ee.Geometry.Polygon(
                grid_polygon_coordinates(row["geometry_wkt"]),
                proj="EPSG:4326",
                geodesic=False,
            )
            features.append(ee.Feature(geom, {"treatment_order": t_order}))
            meta_lookup[t_order] = metadata(row)

        fc = ee.FeatureCollection(features)
        col = ee.ImageCollection(ALPHAEARTH_COLLECTION).filterDate(
            f"{year_int}-01-01", f"{year_int + 1}-01-01"
        )
        img = col.mosaic()

        extracted_features = []
        for attempt in range(3):
            try:
                reduced_fc = img.select(BANDS).reduceRegions(
                    collection=fc, reducer=ee.Reducer.mean(), scale=10, tileScale=2
                )
                extracted_features = reduced_fc.getInfo().get("features", [])
                break
            except Exception as exc:
                logger.warning(
                    "Batch failed for %s (%d), attempt %d: %s",
                    city_key, year_int, attempt + 1, exc,
                )
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))

        seen_orders = set()
        for f in extracted_features:
            props = f.get("properties", {})
            t_order = props.get("treatment_order")
            if t_order in meta_lookup:
                seen_orders.add(t_order)
                rec = dict(meta_lookup[t_order])
                is_valid = all(
                    props.get(band) is not None and np.isfinite(props[band]) for band in BANDS
                )
                rec["alphaearth_available"] = bool(is_valid)
                rec["availability_reason"] = (
                    "available" if is_valid else "missing_or_nonfinite_bands"
                )
                for b in BANDS:
                    v = props.get(b)
                    rec[b] = float(v) if v is not None and np.isfinite(v) else np.nan
                records.append(rec)

        # Handle any missing orders that failed reduceRegions
        for t_order, meta in meta_lookup.items():
            if t_order not in seen_orders:
                rec = dict(meta)
                rec["alphaearth_available"] = False
                rec["availability_reason"] = "earthengine_batch_failed_or_missing_feature"
                for b in BANDS:
                    rec[b] = np.nan
                records.append(rec)

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="macro-city-engine", help="GEE project ID")
    parser.add_argument("--city", default="all", help="City key or 'all'")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate geometry without calling GEE")
    parser.add_argument("--limit-per-city", type=int, default=None, help="Optional limit per city")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-temporal-proxy", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch-size must be positive")
    if args.output_dir is None:
        args.output_dir = (
            ROOT / "outputs" / "alphaearth_grid_preview" if args.dry_run else DEFAULT_OUTPUT_DIR
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        init_ee(args.project)
        import ee

        ee.data.setDeadline(60000)

    treatments = pd.read_parquet(TREATMENT_UNIT_LIST)
    logger.info("Loaded %d treatments across %d cities", len(treatments), treatments["city_key"].nunique())

    if args.city != "all":
        treatments = treatments[treatments["city_key"] == args.city]
        if treatments.empty:
            logger.error("No treatments found for city '%s'", args.city)
            return 1

    cities = sorted(treatments["city_key"].unique())
    all_city_results = []

    def collect_city(city_key: str) -> pd.DataFrame:
        city_df = treatments[treatments["city_key"] == city_key]
        if args.limit_per_city:
            city_df = city_df.head(args.limit_per_city)

        return extract_city_alphaearth(
            city_key, city_df, dry_run=args.dry_run,
            allow_temporal_proxy=args.allow_temporal_proxy, batch_size=args.batch_size,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(collect_city, city): city for city in cities}
        for future in as_completed(futures):
            city_key = futures[future]
            city_out = future.result()
            if not city_out.empty:
                out_file = args.output_dir / f"{city_key}_alphaearth.parquet"
                city_out.to_parquet(out_file, index=False)
                logger.info("Saved %s (%d rows)", out_file, len(city_out))
                all_city_results.append(city_out)

    if all_city_results:
        unified_df = pd.concat(all_city_results, ignore_index=True).sort_values("treatment_order")
        unified_path = args.output_dir / "alphaearth_treated_embeddings.parquet"
        unified_df.to_parquet(unified_path, index=False)
        logger.info("Saved unified embeddings to %s (%d rows)", unified_path, len(unified_df))

        audit = {
            "total_extracted": len(unified_df),
            "cities_count": int(unified_df["city_key"].nunique()),
            "valid_embeddings_count": int(unified_df["alphaearth_available"].sum()),
            "temporal_proxies_count": int(unified_df["is_temporal_proxy"].sum()),
            "bands_count": len(BANDS),
            "spatial_support": SPATIAL_SUPPORT,
            "grid_source": str(REFERENCE_GRID_DIR),
            "treatment_list_sha256": hashlib.sha256(TREATMENT_UNIT_LIST.read_bytes()).hexdigest(),
            "geometry_binding_sha256": hashlib.sha256(
                unified_df[["treatment_order", "city_key", "grid_id", "grid_geometry_sha256"]]
                .to_json(orient="records").encode("utf-8")
            ).hexdigest(),
            "dry_run": bool(args.dry_run),
            "allow_temporal_proxy": bool(args.allow_temporal_proxy),
            "availability_reasons": unified_df["availability_reason"].value_counts().to_dict(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        audit_path = args.output_dir / "alphaearth_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        logger.info("Audit report written to %s", audit_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
