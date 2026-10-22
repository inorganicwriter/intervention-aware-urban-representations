"""Build the street-view index that attaches Baidu imagery to 500m grids.

Pipeline position::

    outputs/streetview/<city>/{metadata,images}      (package_streetview.py)
      -> this script -> data/active/streetview/streetview_index.parquet
      -> build_pretraining_dataset.py --streetview-index ...
      -> urban-train-representation --use-images

Metadata columns (extra columns are ignored)::

    province, city, county, wgs_lon, wgs_lat, panoid, capture_date, status

Only successful captures with finite coordinates are kept. Grid assignment
uses the frozen reference polygons. Ambiguous shared-boundary captures are
left unassigned, matching the causal design's unique-grid requirement.

Usage::

    python scripts/collection/build_streetview_index.py --city all
    python scripts/collection/build_streetview_index.py --city beijing --limit 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from urban_intervention.config.project import ACTIVE_CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_STREETVIEW_DIR,
    STREETVIEW_INDEX,
    grid_path,
)

REQUIRED_METADATA_COLUMNS = {"wgs_lon", "wgs_lat", "capture_date", "panoid", "status"}
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")


def read_metadata(paths: list[Path], status_value: str) -> pd.DataFrame:
    """Read and concatenate metadata CSVs, keeping successful finite captures."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame: pd.DataFrame | None = None
        for encoding in ("utf-8-sig", "gbk", "utf-8"):
            try:
                frame = pd.read_csv(path, encoding=encoding, dtype=str)
                break
            except UnicodeDecodeError:
                continue
        if frame is None:
            raise ValueError(f"cannot decode metadata file: {path}")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=sorted(REQUIRED_METADATA_COLUMNS))
    data = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_METADATA_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"metadata lacks columns: {sorted(missing)}")
    data = data.loc[
        data["status"].astype("string").str.strip().eq(status_value)
    ].copy()
    for column in ("wgs_lon", "wgs_lat"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["capture_date"] = pd.to_datetime(data["capture_date"], errors="coerce")
    data = data.dropna(subset=["wgs_lon", "wgs_lat", "capture_date", "panoid"])
    data = data.drop_duplicates("panoid")
    return data.reset_index(drop=True)


def assign_grids_bbox(
    longitudes: np.ndarray, latitudes: np.ndarray, grids: pd.DataFrame
) -> pd.Series:
    """Legacy axis-aligned lookup; not valid for the projected production grids."""
    required = {"grid_id", "lon_min", "lon_max", "lat_min", "lat_max"}
    missing = required - set(grids.columns)
    if missing:
        raise ValueError(f"grids lack columns: {sorted(missing)}")
    grouped = grids.groupby(["lat_min", "lat_max"], sort=True)
    row_keys = list(grouped.groups.keys())
    row_lat_min = np.array([key[0] for key in row_keys], dtype=np.float64)
    row_lat_max = np.array([key[1] for key in row_keys], dtype=np.float64)
    row_lons: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for key in row_keys:
        part = grouped.get_group(key).sort_values("lon_min")
        row_lons.append(
            (
                part["lon_min"].to_numpy(dtype=np.float64),
                part["lon_max"].to_numpy(dtype=np.float64),
                part["grid_id"].to_numpy(dtype=object),
            )
        )
    result = np.full(longitudes.shape[0], None, dtype=object)
    rows = np.searchsorted(row_lat_min, latitudes, side="right") - 1
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    for row_index in range(len(row_keys)):
        start = int(np.searchsorted(sorted_rows, row_index, side="left"))
        end = int(np.searchsorted(sorted_rows, row_index, side="right"))
        if start == end:
            continue
        selection = order[start:end]
        within_lat = latitudes[selection] <= row_lat_max[row_index]
        lon_mins, lon_maxs, grid_ids = row_lons[row_index]
        columns = np.searchsorted(lon_mins, longitudes[selection], side="right") - 1
        valid = within_lat & (columns >= 0)
        if valid.any():
            clamped = np.clip(columns, 0, None)
            valid &= longitudes[selection] <= lon_maxs[clamped]
        if valid.any():
            chosen = selection[valid]
            result[chosen] = grid_ids[columns[valid]]
    return pd.Series(result)


def assign_grids_polygon(
    longitudes: np.ndarray, latitudes: np.ndarray, grids: pd.DataFrame
) -> pd.Series:
    """Assign captures by actual polygon containment, never bounding boxes."""
    if "geometry_wkt" not in grids or "grid_id" not in grids:
        raise ValueError("Reference grids require grid_id and geometry_wkt")
    if grids["grid_id"].duplicated().any():
        raise ValueError("Reference grids have duplicate grid_id")
    longitudes = np.asarray(longitudes, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)
    if longitudes.ndim != 1 or latitudes.shape != longitudes.shape:
        raise ValueError("Coordinate arrays must be matching one-dimensional arrays")
    result = np.full(len(longitudes), None, dtype=object)
    if grids.empty or not len(result):
        return pd.Series(result, dtype=object)
    polygons = shapely.from_wkt(grids["geometry_wkt"].to_numpy())
    if not np.all(shapely.is_valid(polygons) & ~shapely.is_empty(polygons)):
        raise ValueError("Reference grids contain invalid polygons")
    valid = np.isfinite(longitudes) & np.isfinite(latitudes)
    source_indices = np.flatnonzero(valid)
    points = shapely.points(longitudes[valid], latitudes[valid])
    point_indices, grid_indices = STRtree(polygons).query(points, predicate="covered_by")
    match_counts = np.bincount(point_indices, minlength=len(points))
    unique = match_counts[point_indices] == 1
    result[source_indices[point_indices[unique]]] = grids["grid_id"].to_numpy()[
        grid_indices[unique]
    ]
    return pd.Series(result, dtype=object)


def resolve_image_path(
    images_dir: Path, panoid: str, extensions: tuple[str, ...] = IMAGE_EXTENSIONS
) -> Path | None:
    """Resolve one panoid to an existing image file, or None."""
    for extension in extensions:
        candidate = images_dir / f"{panoid}.{extension}"
        if candidate.is_file():
            return candidate
    for match in sorted(images_dir.glob(f"{panoid}*")):
        if match.is_file() and match.suffix.lower().lstrip(".") in extensions:
            return match
    return None


def build_city_index(
    city: str,
    metadata_files: list[Path],
    images_dir: Path,
    grids: pd.DataFrame,
    *,
    status_value: str = "成功",
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach one city's captures to grids and resolve image paths."""
    data = read_metadata(metadata_files, status_value)
    audit: dict[str, object] = {
        "metadata_rows": int(len(data)),
        "matched_grid_rows": 0,
        "rows_with_image": 0,
        "missing_image_rows": 0,
    }
    if data.empty:
        return pd.DataFrame(
            columns=["city_key", "grid_id", "capture_date", "asset_path", "panoid"]
        ), audit
    if limit is not None and len(data) > limit:
        data = data.sample(n=limit, random_state=0).reset_index(drop=True)
    data["grid_id"] = assign_grids_polygon(
        data["wgs_lon"].to_numpy(dtype=np.float64),
        data["wgs_lat"].to_numpy(dtype=np.float64),
        grids,
    )
    matched = data.dropna(subset=["grid_id"])
    audit["matched_grid_rows"] = int(len(matched))
    rows: list[dict[str, object]] = []
    missing = 0
    for record in matched.itertuples(index=False):
        image_path = resolve_image_path(images_dir, str(record.panoid), extensions)
        if image_path is None:
            missing += 1
            continue
        rows.append(
            {
                "city_key": city,
                "grid_id": str(record.grid_id),
                "capture_date": record.capture_date,
                "asset_path": str(image_path.resolve()),
                "panoid": str(record.panoid),
            }
        )
    audit["missing_image_rows"] = missing
    audit["rows_with_image"] = len(rows)
    frame = pd.DataFrame(
        rows, columns=["city_key", "grid_id", "capture_date", "asset_path", "panoid"]
    )
    return frame, audit


def _metadata_files(city_root: Path) -> list[Path]:
    return sorted((city_root / "metadata").glob("*.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streetview-root", type=Path, default=OUTPUT_STREETVIEW_DIR)
    parser.add_argument("--output", type=Path, default=STREETVIEW_INDEX)
    parser.add_argument("--city", default="all", help="City key, comma-separated, or 'all'.")
    parser.add_argument("--status-value", default="成功")
    parser.add_argument("--limit", type=int, help="Sample at most N captures per city.")
    parser.add_argument(
        "--allow-unknown-cities",
        action="store_true",
        help="Include directories outside the active city registry.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.streetview_root
    if not root.is_dir():
        raise SystemExit(f"streetview root not found: {root}")
    if args.city == "all":
        cities = sorted(path.name for path in root.iterdir() if path.is_dir())
    else:
        cities = [value.strip() for value in args.city.split(",") if value.strip()]
    known = set(ACTIVE_CITIES)
    unknown = sorted(set(cities) - known)
    if unknown and not args.allow_unknown_cities:
        print(f"skipping non-registry cities: {unknown}", file=sys.stderr)
        cities = [city for city in cities if city in known]
    frames: list[pd.DataFrame] = []
    audits: dict[str, dict[str, object]] = {}
    for city in cities:
        city_root = root / city
        metadata = _metadata_files(city_root)
        grids_file = grid_path(city)
        if not metadata or not (city_root / "images").is_dir():
            audits[city] = {"skipped": "metadata or images directory missing"}
            continue
        if not grids_file.is_file():
            audits[city] = {"skipped": f"grids not found: {grids_file}"}
            continue
        grids = pd.read_parquet(
            grids_file, columns=["grid_id", "geometry_wkt"]
        )
        frame, audit = build_city_index(
            city,
            metadata,
            city_root / "images",
            grids,
            status_value=args.status_value,
            limit=args.limit,
        )
        frames.append(frame)
        audits[city] = audit
        print(
            f"{city}: captures={audit['metadata_rows']} "
            f"grid={audit['matched_grid_rows']} images={audit['rows_with_image']} "
            f"missing={audit['missing_image_rows']}"
        )
    if not frames:
        raise SystemExit("no city index could be built")
    index = pd.concat(frames, ignore_index=True)
    index = index.drop_duplicates(["city_key", "grid_id", "capture_date", "panoid"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(args.output, index=False, compression="zstd")
    summary = {
        "spatial_support": "fixed_grid_polygon_v1",
        "grid_assignment": "unique_point_in_polygon",
        "output": str(args.output),
        "rows": int(len(index)),
        "cities": int(index["city_key"].nunique()) if not index.empty else 0,
        "per_city": audits,
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nindex: {args.output} ({len(index)} rows)")
    print(f"audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
