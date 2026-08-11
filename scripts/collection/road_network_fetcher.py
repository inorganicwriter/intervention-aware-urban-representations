"""Fetch OSM road network and aggregate to grid-level metrics for 44 cities.

For each city:
  1. Download the driveable road network from OSM via osmnx
  2. Project to the city's UTM zone
  3. Load existing 500m grids
  4. For each grid cell, compute:
     - total_road_length_m
     - intersection_count
     - node_count
     - average_circuity
     - road_density_km_per_km2
     - intersection_density_per_km2
  5. Save to data/active/curated/road_network/{city}_road_network.parquet

Usage:
    python scripts/collection/road_network_fetcher.py --city beijing
    python scripts/collection/road_network_fetcher.py --city all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from urban_intervention.config.project import ACTIVE_CITIES, CITIES, GRID_DIR, get_proxy
from urban_intervention.data.paths import ROAD_NETWORK_DIR

OUT_DIR = ROAD_NETWORK_DIR

HIGHWAY_TYPES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
]


def fetch_city_road_network(city_key: str, timeout: int = 300) -> pd.DataFrame | None:
    """Download OSM road network and aggregate to grid-level metrics.

    Returns a DataFrame with columns:
        grid_id, total_road_length_m, intersection_count, node_count,
        average_circuity, road_density_km_per_km2, intersection_density_per_km2
    """
    import geopandas as gpd
    import osmnx as ox
    from shapely import wkt

    city_cfg = CITIES[city_key]
    print(f"\n{'=' * 60}", flush=True)
    print(f"City: {city_key} ({city_cfg['name']})", flush=True)

    # ── Load grids ──────────────────────────────────────────
    grid_path = GRID_DIR / city_key / f"{city_key}_grids.parquet"
    if not grid_path.exists():
        print(f"  ERROR: grid file not found: {grid_path}", flush=True)
        return None
    grids = pd.read_parquet(grid_path, columns=["grid_id", "geometry_wkt"])
    grids["geometry"] = grids["geometry_wkt"].apply(wkt.loads)
    grids = grids.drop(columns=["geometry_wkt"])
    grids = gpd.GeoDataFrame(grids, geometry="geometry", crs="EPSG:4326")
    print(f"  Grids: {len(grids):,} cells", flush=True)

    proxy = get_proxy()
    if proxy:
        print(f"  Using proxy: {proxy}", flush=True)
        ox.settings.http_proxy = proxy

    # Increase Overpass timeout for large cities
    ox.settings.overpass_rate_limit = True
    ox.settings.max_query_area_size = 25_000_000_000  # allow large areas
    ox.settings.timeout = 600
    ox.settings.memory = 1073741824  # 1GB

    # Use Overpass Turbo endpoint (more reliable for China)
    ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"
    print(f"  Overpass endpoint: {ox.settings.overpass_endpoint}", flush=True)

    cfg_bbox = city_cfg["bbox"]
    buf = 0.05  # ~5km buffer
    query_bbox = (cfg_bbox[0] - buf, cfg_bbox[1] - buf, cfg_bbox[2] + buf, cfg_bbox[3] + buf)
    print(f"  Downloading OSM road network (bbox={query_bbox})...", flush=True)

    t0 = time.time()
    # osmnx 2.x: bbox = (left, bottom, right, top) = (west, south, east, north)
    osm_bbox = (query_bbox[0], query_bbox[1], query_bbox[2], query_bbox[3])
    G = None
    for attempt in range(3):
        try:
            G = ox.graph_from_bbox(
                osm_bbox,
                network_type="drive",
                custom_filter=f'["highway"~"{"|".join(HIGHWAY_TYPES)}"]',
            )
            break
        except Exception as exc:
            print(f"  Attempt {attempt + 1}/3 failed: {exc}", flush=True)
            if attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"  Waiting {wait}s before retry...", flush=True)
                time.sleep(wait)

    if G is None:
        print("  ERROR: all 3 download attempts failed", flush=True)
        return None
    t1 = time.time()
    print(
        f"  Downloaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges ({t1 - t0:.1f}s)",
        flush=True,
    )

    # ── Convert to projected GeoDataFrame ────────────────────
    projected_crs = city_cfg.get("projected_crs", "EPSG:32650")
    print(f"  Projecting to {projected_crs}...", flush=True)

    # Nodes
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    nodes_gdf = nodes_gdf.set_crs("EPSG:4326")
    nodes_proj = nodes_gdf.to_crs(projected_crs)

    # Edges
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    edges_gdf = edges_gdf.set_crs("EPSG:4326")
    edges_proj = edges_gdf.to_crs(projected_crs)

    # ── Project grids to same CRS ────────────────────────────
    grids_proj = grids.to_crs(projected_crs)

    # ── Compute edge metrics per grid (vectorized) ──────────
    print("  Computing road metrics per grid (vectorized)...", flush=True)

    # Clip all edges to grid boundaries at once via sjoin + intersection
    edges_simple = edges_proj[["geometry"]].copy()
    edges_simple["edge_id"] = range(len(edges_simple))

    # Spatial join: each edge gets assigned to overlapping grids
    edges_joined = gpd.sjoin(
        edges_simple, grids_proj[["grid_id", "geometry"]], how="inner", predicate="intersects"
    )

    # Clip edges to grid polygon
    grid_polys = grids_proj.set_index("grid_id")["geometry"]
    edges_joined["clipped"] = edges_joined.apply(
        lambda row: row.geometry.intersection(grid_polys[row["grid_id"]]), axis=1
    )
    edges_joined = edges_joined.set_geometry("clipped")

    # Sum road length per grid
    edges_joined["length_m"] = edges_joined.geometry.length
    road_length = edges_joined.groupby("grid_id")["length_m"].sum().reset_index()
    road_length.columns = ["grid_id", "total_road_length_m"]

    # Count nodes within each grid
    if "street_count" in nodes_proj.columns:
        nodes_proj["has_street_count"] = nodes_proj["street_count"].fillna(0).astype(float)
    else:
        nodes_proj["has_street_count"] = 0

    nodes_joined = gpd.sjoin(
        nodes_proj[["geometry", "has_street_count"]],
        grids_proj[["grid_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    # Nodes per grid
    node_count = nodes_joined.groupby("grid_id").size().reset_index(name="node_count")

    # Intersections per grid (nodes with street_count >= 3)
    if "street_count" in nodes_proj.columns:
        intersections = nodes_proj[nodes_proj["street_count"] >= 3]
        int_joined = gpd.sjoin(
            intersections[["geometry"]],
            grids_proj[["grid_id", "geometry"]],
            how="inner",
            predicate="within",
        )
        int_count = int_joined.groupby("grid_id").size().reset_index(name="intersection_count")
    else:
        int_count = node_count.copy()
        int_count.columns = ["grid_id", "intersection_count"]

    # Merge all metrics
    df = pd.DataFrame({"grid_id": grids_proj["grid_id"]})
    df = df.merge(road_length, on="grid_id", how="left")
    df = df.merge(node_count, on="grid_id", how="left")
    df = df.merge(int_count, on="grid_id", how="left")

    df["total_road_length_m"] = df["total_road_length_m"].fillna(0.0)
    df["node_count"] = df["node_count"].fillna(0).astype(int)
    df["intersection_count"] = df["intersection_count"].fillna(0).astype(int)

    # Derived metrics
    grid_area_km2 = (
        grids_proj.geometry.unary_union.area / 1e6 / len(grids_proj)
    )  # average, approximate
    df["road_density_km_per_km2"] = df["total_road_length_m"] / 1000 / grid_area_km2
    df["intersection_density_per_km2"] = df["intersection_count"] / grid_area_km2
    df["average_circuity"] = 0.0  # placeholder — not meaningful without per-edge routing

    total_road_km = df["total_road_length_m"].sum() / 1000
    total_int = int(df["intersection_count"].sum())
    print(
        f"  Results: {len(df):,} grids, road={total_road_km:.1f} km, intersections={total_int:,}",
        flush=True,
    )
    return df


def save_city_road_network(city_key: str, df: pd.DataFrame) -> Path:
    """Save road network metrics to parquet."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{city_key}_road_network.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df):,} rows)", flush=True)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch OSM road network and aggregate to grid metrics."
    )
    parser.add_argument("--city", default="all", help="City key, comma-separated, or 'all'.")
    parser.add_argument("--timeout", type=int, default=300, help="OSM query timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Download but don't save.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.city == "all":
        cities = list(ACTIVE_CITIES)
    else:
        cities = [c.strip() for c in args.city.split(",") if c.strip()]
        unknown = [c for c in cities if c not in CITIES]
        if unknown:
            raise KeyError(f"Unknown city key(s): {unknown}")

    print(f"Processing {len(cities)} city/cities: {', '.join(cities)}", flush=True)

    for city_key in cities:
        df = fetch_city_road_network(city_key, timeout=args.timeout)
        if df is not None and not args.dry_run:
            save_city_road_network(city_key, df)
        elif df is not None and args.dry_run:
            print(f"  [dry-run] {city_key}: {len(df):,} rows (not saved)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
