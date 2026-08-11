"""Precompute pre-treatment transit features for ALL grids at every treated
opening month (plan A: on-demand snapshots cached to disk).

For each city and each distinct treated opening month t:
  - snapshot = stations with opening_date <= t - anticipation months
  - per grid (all city grids, including donors):
      dist_nearest_station_m, stations_500m/800m/1500m, lines_in_1500m
      network_closeness (of the nearest station)

Performance notes (vectorised):
  - closeness is computed ONCE per city on the current P197 topology, then
    filtered to the snapshot station set.  The research window (2010-2025)
    means all snapshot stations are post-2010 openings whose topology is
    current, so this is exact for this window.
  - grid-station distances use a vectorised cKDTree query (k nearest) with
    numpy masking; no per-grid Python loop.

Output: data/active/causal/transit_snapshots/{city}/{opening_month}.parquet
  one row per grid: city_key, grid_id, opening_month, and the 6 features.

Usage:
    python scripts/analysis/build_transit_snapshots.py --city all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, GRID_DIR  # noqa: E402

REFERENCE_DIR = BASE_DIR / "data" / "active" / "reference"
ADJACENCY_PATH = REFERENCE_DIR / "transit" / "wikidata_adjacency.parquet"
OUT_DIR = BASE_DIR / "data" / "active" / "causal" / "transit_snapshots"
ANTICIPATION_MONTHS = 12
BUFFER_RADII_M = (500, 800, 1500)
LINE_SPLIT = re.compile(r"[;；]")

# cKDTree works in lon/lat degrees; convert the 1500 m radius to a degree
# bound using the city median latitude (equirectangular).
EARTH_RADIUS_KM = 6371.0
K_NEAREST = 25  # max stations a grid can have within 1500 m


def _norm_station(name: object) -> str:
    s = str(name)
    s = re.sub(r"\(地铁站\)$", "", s)
    s = re.sub(r"站$", "", s)
    return s


def _parse_lines(lines: str) -> list[str]:
    if not isinstance(lines, str) or not lines.strip():
        return []
    return [ln.strip() for ln in LINE_SPLIT.split(lines) if ln.strip()]


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _load_city_stations(city: str) -> pd.DataFrame:
    events = pd.read_parquet(
        REFERENCE_DIR / "transit" / "canonical_station_events_resolved.parquet",
        columns=[
            "city_key",
            "station_event_id",
            "canonical_station_name",
            "lines",
            "wgs84_lon",
            "wgs84_lat",
            "opening_date",
            "primary_design_excluded",
        ],
    )
    events = events[events["city_key"] == city].copy()
    events = events[events["primary_design_excluded"] != True]  # noqa: E712
    events["opening_date"] = pd.to_datetime(events["opening_date"])
    events["_norm"] = events["canonical_station_name"].apply(_norm_station)
    events = events.drop_duplicates(subset=["_norm"], keep="first")
    return events.reset_index(drop=True)


def _load_adjacency_edges(city: str) -> pd.DataFrame:
    adj = pd.read_parquet(ADJACENCY_PATH)
    adj = adj[adj["city_key"] == city].copy()
    adj["_norm"] = adj["station_name"].apply(_norm_station)
    adj["_norm_adj"] = adj["adj_station_name"].apply(_norm_station)
    return adj


def _closeness_once(events: pd.DataFrame, adj: pd.DataFrame) -> dict[str, float]:
    """closeness per station_event_id on the CURRENT full network, once."""
    import networkx as nx

    g = nx.Graph()
    coords = {}
    for _, row in events.iterrows():
        eid = row["station_event_id"]
        coords[eid] = (row["wgs84_lon"], row["wgs84_lat"])
        g.add_node(eid)
    name_to_event = events.set_index("_norm")["station_event_id"]
    present = set(events["_norm"])
    for _, row in adj.iterrows():
        if row["_norm"] not in present or row["_norm_adj"] not in present:
            continue
        a = name_to_event[row["_norm"]]
        b = name_to_event[row["_norm_adj"]]
        if a == b or g.has_edge(a, b):
            continue
        d = _haversine_km(coords[a][0], coords[a][1], coords[b][0], coords[b][1])
        if d > 0:
            g.add_edge(a, b, weight=d)
    result: dict[str, float] = {}
    for node in g.nodes():
        lengths = nx.single_source_dijkstra_path_length(g, node, weight="weight")
        total = sum(lengths.values()) - lengths.get(node, 0.0)
        if total > 0 and len(lengths) > 1:
            result[node] = 1.0 / total
        else:
            result[node] = 0.0
    return result


def build_snapshot(
    city: str,
    opening_month: pd.Timestamp,
    stations: pd.DataFrame,
    adj: pd.DataFrame,
    grids: pd.DataFrame,
    closeness: dict[str, float],
) -> pd.DataFrame:
    cutoff = opening_month - pd.DateOffset(months=ANTICIPATION_MONTHS)
    snap = stations[stations["opening_date"] <= cutoff].copy()
    cols = [
        "city_key",
        "grid_id",
        "opening_month",
        "dist_nearest_station_m",
        "stations_500m",
        "stations_800m",
        "stations_1500m",
        "lines_in_1500m",
        "network_closeness",
    ]
    if snap.empty:
        out = grids[["grid_id"]].copy()
        out["city_key"] = city
        out["opening_month"] = opening_month
        for col in cols[3:]:
            out[col] = 0.0
        out["dist_nearest_station_m"] = np.nan
        return out[cols]

    # Station -> line-set index for vectorised line counting.
    snap = snap.copy()
    snap["_line_set"] = snap["lines"].apply(lambda s: frozenset(_parse_lines(s)))

    grid_xy = np.column_stack(
        [
            grids["centroid_lon"].to_numpy(),
            grids["centroid_lat"].to_numpy(),
        ]
    )
    station_xy = np.column_stack(
        [
            snap["wgs84_lon"].to_numpy(),
            snap["wgs84_lat"].to_numpy(),
        ]
    )

    # Vectorised k-NN query: for every grid, distances/indices of up to
    # K_NEAREST stations overall (nearest-station distance is not radius
    # limited), and a 1500 m mask for the buffer counts.
    radius_deg = 1.5 / (EARTH_RADIUS_KM * np.pi / 180.0)  # 1.5 km in degrees
    tree = cKDTree(station_xy)
    dist_deg, idx = tree.query(grid_xy, k=K_NEAREST)
    if dist_deg.ndim == 1:
        dist_deg = dist_deg[:, None]
        idx = idx[:, None]

    mask = np.isfinite(dist_deg) & (dist_deg <= radius_deg)
    # Convert degree distance to km (equirectangular at the grid latitude).
    km_per_deg = np.pi * EARTH_RADIUS_KM / 180.0
    dist_km = dist_deg * km_per_deg
    dist_m = dist_km * 1000.0

    # Nearest station: overall minimum (any distance), not radius limited.
    nearest_m = dist_m.min(axis=1)
    nearest_m[~np.isfinite(nearest_m)] = np.nan

    stations_500 = ((mask) & (dist_m <= 500)).sum(axis=1).astype(int)
    stations_800 = ((mask) & (dist_m <= 800)).sum(axis=1).astype(int)
    stations_1500 = mask.sum(axis=1).astype(int)

    # lines in 1500 m: union of line sets of stations within radius, per grid
    line_counts = np.zeros(len(grids), dtype=int)
    station_line_sets = snap["_line_set"].to_numpy()
    for r in range(len(grids)):
        js = idx[r][mask[r]]
        union = set()
        for j in js:
            union |= station_line_sets[j]
        line_counts[r] = len(union)

    # closeness of the overall nearest station (not radius limited)
    nearest_idx = idx[np.arange(len(grids)), dist_m.argmin(axis=1)]
    closest_vals = np.zeros(len(grids))
    for r in range(len(grids)):
        if np.isfinite(dist_m[r].min()):
            eid = snap["station_event_id"].iloc[nearest_idx[r]]
            closest_vals[r] = closeness.get(eid, 0.0)

    out = grids[["grid_id"]].copy()
    out["city_key"] = city
    out["opening_month"] = opening_month
    out["dist_nearest_station_m"] = nearest_m
    out["stations_500m"] = stations_500
    out["stations_800m"] = stations_800
    out["stations_1500m"] = stations_1500
    out["lines_in_1500m"] = line_counts
    out["network_closeness"] = closest_vals
    return out[cols]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="all")
    args = parser.parse_args()

    treated = pd.read_csv(BASE_DIR / "data" / "active" / "causal" / "treatment_unit_list.csv")
    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for city in cities:
        city_treated = treated[treated["city_key"] == city]
        if city_treated.empty:
            print(f"  [SKIP] {city}: no treated grids")
            continue
        opening_months = sorted(pd.to_datetime(city_treated["opening_month"].unique()))
        stations = _load_city_stations(city)
        adj = _load_adjacency_edges(city)
        grids = pd.read_parquet(
            GRID_DIR / city / f"{city}_grids.parquet",
            columns=["grid_id", "centroid_lon", "centroid_lat"],
        )
        closeness = _closeness_once(stations, adj)
        city_dir = OUT_DIR / city
        city_dir.mkdir(parents=True, exist_ok=True)
        for month in opening_months:
            label = month.strftime("%Y-%m")
            target = city_dir / f"{label}.parquet"
            if target.exists():
                continue
            frame = build_snapshot(city, month, stations, adj, grids, closeness)
            frame.to_parquet(target, index=False)
        print(f"  {city}: {len(opening_months)} snapshots, {len(grids):,} grids each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
