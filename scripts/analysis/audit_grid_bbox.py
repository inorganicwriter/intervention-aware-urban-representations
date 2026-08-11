"""Compare configured city bounding boxes with generated grid extents."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import shapely

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "src"))

from urban_intervention.config.project import ACTIVE_CITIES, CITIES, GRID_DIR
from urban_intervention.data.paths import OUTPUT_POI_QUALITY_DIR


def grid_bounds(city_key: str) -> tuple[float, float, float, float]:
    path = GRID_DIR / city_key / f"{city_key}_grids.parquet"
    df = pd.read_parquet(path, columns=["geometry_wkt"])
    bounds = shapely.bounds(shapely.from_wkt(df["geometry_wkt"].to_numpy()))
    return (
        float(bounds[:, 0].min()),
        float(bounds[:, 1].min()),
        float(bounds[:, 2].max()),
        float(bounds[:, 3].max()),
    )


def build_audit() -> pd.DataFrame:
    rows = []
    for city_key in ACTIVE_CITIES:
        cfg = CITIES[city_key]["bbox"]
        grid = grid_bounds(city_key)
        rows.append(
            {
                "city": city_key,
                "cfg_minx": cfg[0],
                "cfg_miny": cfg[1],
                "cfg_maxx": cfg[2],
                "cfg_maxy": cfg[3],
                "grid_minx": grid[0],
                "grid_miny": grid[1],
                "grid_maxx": grid[2],
                "grid_maxy": grid[3],
                "d_minx": grid[0] - cfg[0],
                "d_miny": grid[1] - cfg[1],
                "d_maxx": grid[2] - cfg[2],
                "d_maxy": grid[3] - cfg[3],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    out = build_audit()
    out_path = OUTPUT_POI_QUALITY_DIR / "grid_bbox_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {out_path} ({len(out)} rows)")
    print(out[["d_minx", "d_miny", "d_maxx", "d_maxy"]].abs().max().to_string())
    diff = out[(out[["d_minx", "d_miny", "d_maxx", "d_maxy"]].abs() > 0.001).any(axis=1)]
    print(f"diff_cities={len(diff)}")
    if not diff.empty:
        print(diff.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
