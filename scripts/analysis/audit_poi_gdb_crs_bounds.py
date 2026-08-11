"""Audit CRS and coordinate bounds for extracted POI FileGDB sources."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "src"))

from collection.poi.gdb import discover_extracted_gdb_sources

from urban_intervention.data.paths import CATALOG_DIR, OUTPUT_POI_QUALITY_DIR


def main() -> int:
    import geopandas as gpd

    inventory_path = CATALOG_DIR / "inventories" / "poi_gdb_extracted_inventory.csv"
    missing_keys = None
    if inventory_path.exists():
        inv = pd.read_csv(inventory_path)
        missing = inv[inv["crs"].fillna("").eq("")]
        missing_keys = {
            (int(row.year), str(row.path), "" if pd.isna(row.layer) else str(row.layer))
            for row in missing.itertuples(index=False)
        }

    rows = []
    for source in discover_extracted_gdb_sources():
        if missing_keys is not None:
            key = (int(source.year), str(source.path), source.layer or "")
            if key not in missing_keys:
                continue
        read_kwargs = {}
        if source.layer is not None:
            read_kwargs["layer"] = source.layer
        gdf = gpd.read_file(source.path, **read_kwargs)
        bounds = gdf.total_bounds if not gdf.empty else [None, None, None, None]
        rows.append(
            {
                "year": source.year,
                "category": source.category,
                "layer": source.layer or "",
                "path": str(source.path),
                "crs": gdf.crs.to_string() if gdf.crs is not None else "",
                "rows": len(gdf),
                "minx": bounds[0],
                "miny": bounds[1],
                "maxx": bounds[2],
                "maxy": bounds[3],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        print(
            "No GDB sources to audit; verify discover_extracted_gdb_sources() or missing_keys filter"
        )
        return 0
    out["looks_lonlat"] = (
        out["minx"].between(-180, 180)
        & out["maxx"].between(-180, 180)
        & out["miny"].between(-90, 90)
        & out["maxy"].between(-90, 90)
    )
    out_path = OUTPUT_POI_QUALITY_DIR / "poi_gdb_crs_bounds.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {out_path} ({len(out)} rows)")
    print(out.groupby(["year", "crs"], dropna=False).size().to_string())
    bad = out[~out["looks_lonlat"]]
    print(f"non_lonlat_bounds={len(bad)}")
    if not bad.empty:
        print(
            bad[
                ["year", "category", "layer", "crs", "rows", "minx", "miny", "maxx", "maxy"]
            ].to_string(index=False)
        )
    missing = out[out["crs"].eq("")]
    print(f"missing_crs={len(missing)}")
    if not missing.empty:
        print(
            missing[
                [
                    "year",
                    "category",
                    "layer",
                    "rows",
                    "minx",
                    "miny",
                    "maxx",
                    "maxy",
                    "looks_lonlat",
                ]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
