"""Verify AlphaEarth against the frozen causal grids and quantify old-window mismatch."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.config.project import CITIES  # noqa: E402
from urban_intervention.data.paths import REFERENCE_GRID_DIR, TREATMENT_UNIT_LIST  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "data/active/alphaearth_grid_v2/alphaearth_treated_embeddings.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/spatial_support_audit/2026-10-13"
    )
    args = parser.parse_args()
    treatments = pd.read_parquet(TREATMENT_UNIT_LIST)
    satellite = pd.read_parquet(args.source)
    if satellite["treatment_order"].duplicated().any():
        raise ValueError("Duplicate source events")
    if set(satellite.treatment_order) != set(treatments.treatment_order):
        raise ValueError("Satellite source does not preserve the frozen event support")
    bound = treatments.merge(
        satellite, on=["treatment_order", "city_key", "grid_id"],
        how="left", validate="one_to_one", suffixes=("", "_sat"),
    )
    if not bound.spatial_support.eq("fixed_grid_polygon_v1").all():
        raise ValueError("Satellite source lacks the canonical spatial support")
    parts = []
    for city, events in bound.groupby("city_key", sort=True):
        grids = pd.read_parquet(
            REFERENCE_GRID_DIR / city / f"{city}_grids.parquet",
            columns=["grid_id", "geometry_wkt"],
        )
        events = events.merge(grids, on="grid_id", suffixes=("_sat", ""), validate="many_to_one")
        hashes = events.geometry_wkt.map(
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
        if not hashes.eq(events.grid_geometry_sha256).all():
            raise ValueError(f"Reference geometry binding mismatch: {city}")
        if not events.geometry_wkt.eq(events.geometry_wkt_sat).all():
            raise ValueError(f"Extracted polygon differs from reference: {city}")
        forward = Transformer.from_crs("EPSG:4326", CITIES[city]["projected_crs"], always_xy=True)
        official = shapely.transform(
            shapely.from_wkt(events.geometry_wkt.to_numpy()), forward.transform, interleaved=False
        )
        legacy = np.array([
            box(lon - .0025, lat - .0025, lon + .0025, lat + .0025)
            for lon, lat in zip(events.wgs84_lon, events.wgs84_lat, strict=True)
        ], dtype=object)
        legacy = shapely.transform(legacy, forward.transform, interleaved=False)
        intersection = shapely.area(shapely.intersection(official, legacy))
        parts.append(pd.DataFrame({
            "treatment_order": events.treatment_order.to_numpy(),
            "city_key": city, "grid_id": events.grid_id.to_numpy(),
            "old_window_iou": intersection / shapely.area(shapely.union(official, legacy)),
            "old_window_grid_coverage": intersection / shapely.area(official),
            "official_area_m2": shapely.area(official),
            "old_window_area_m2": shapely.area(legacy),
        }))
    overlap = pd.concat(parts, ignore_index=True)
    bands = [f"A{i:02d}" for i in range(64)]
    finite = np.isfinite(bound[bands].to_numpy(dtype=float)).all(axis=1)
    available = bound.alphaearth_available.fillna(False).to_numpy(dtype=bool)
    opening_year = pd.to_datetime(bound.opening_month).dt.year.to_numpy()
    target_year = bound.target_year.to_numpy(dtype=float)
    if not np.array_equal(finite, available):
        raise ValueError("Band completeness disagrees with availability")
    if np.any(available & ~(target_year < opening_year)):
        raise ValueError("Available embeddings violate the pre-treatment boundary")
    if bound.is_dry_run.any() or bound.is_temporal_proxy.any():
        raise ValueError("Main spatial release contains dry-run rows or temporal proxies")
    audit = {
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "treatment_list_sha256": hashlib.sha256(TREATMENT_UNIT_LIST.read_bytes()).hexdigest(),
        "spatial_support": "fixed_grid_polygon_v1",
        "total_events": len(bound), "cities": int(bound.city_key.nunique()),
        "geometry_binding_valid": len(bound), "available_pre_treatment": int(available.sum()),
        "unavailable_events": int((~available).sum()),
        "availability_reasons": bound.availability_reason.value_counts().to_dict(),
        "old_window_identical_to_grid": int(overlap.old_window_iou.gt(.99999).sum()),
        "old_window_iou_median": float(overlap.old_window_iou.median()),
        "old_window_iou_mean": float(overlap.old_window_iou.mean()),
        "old_window_iou_below_half": int(overlap.old_window_iou.lt(.5).sum()),
        "old_window_grid_coverage_mean": float(overlap.old_window_grid_coverage.mean()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(args.output_dir / "alphaearth_grid_overlap.csv", index=False)
    (args.output_dir / "alphaearth_spatial_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
