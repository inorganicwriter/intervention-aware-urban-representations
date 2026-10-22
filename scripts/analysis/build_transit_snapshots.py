"""Precompute pre-treatment transit features for ALL grids at every treated
opening month (plan A: on-demand snapshots cached to disk).

For each city and each distinct treated opening month t:
  - snapshot = stations with opening_date <= t - anticipation months
  - per grid (all city grids, including donors):
      dist_nearest_station_m, stations_500m/800m/1500m, lines_in_1500m
      network_closeness (of the nearest station)

Distances use a spherical 3D KDTree, including every station in each buffer.
Closeness is computed on cutoff-eligible nodes and edges, cached by graph.
Undated P197 adjacency remains a topology proxy, not a verified historical
edge inventory. Optional valid_from / valid_to columns constrain dated edges.

Output: outputs/transit_snapshots_v2/{city}/{opening_month}.parquet
  one row per grid: city_key, grid_id, opening_month, and the 6 features.
Each file has a versioned metadata sidecar. Frozen production inputs are
preserved; promoting corrected snapshots is a separate data-release step.

Usage:
    python scripts/analysis/build_transit_snapshots.py --city all
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
OUT_DIR = BASE_DIR / "outputs" / "transit_snapshots_v2"
ANTICIPATION_MONTHS = 12
BUFFER_RADII_M = (500, 800, 1500)
LINE_SPLIT = re.compile(r"[;；]")

EARTH_RADIUS_KM = 6371.0
SNAPSHOT_SCHEMA = "urban_intervention_transit_snapshot_v2"


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
    """Closeness on the supplied eligible graph, preserving the old definition."""
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


def _unit_sphere(lon_lat: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(lon_lat).T
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def _chord_to_m(chord: np.ndarray) -> np.ndarray:
    return 2 * EARTH_RADIUS_KM * 1000 * np.arcsin(np.clip(chord / 2, 0, 1))


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_json(orient="split", date_format="iso").encode("utf-8")
    ).hexdigest()


def _edges_at_cutoff(adj: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    eligible = adj.copy()
    for column, is_start in (("valid_from", True), ("valid_to", False)):
        if column not in eligible:
            continue
        dates = pd.to_datetime(eligible[column], errors="coerce")
        if (eligible[column].notna() & dates.isna()).any():
            raise ValueError(f"Invalid edge dates in {column}")
        keep = dates.isna() | (dates <= cutoff if is_start else dates > cutoff)
        eligible = eligible.loc[keep].copy()
    return eligible


def _coordinate_mask(lon_lat: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(lon_lat).all(axis=1)
        & (np.abs(lon_lat[:, 0]) <= 180)
        & (np.abs(lon_lat[:, 1]) <= 90)
    )


def build_snapshot(
    city: str,
    opening_month: pd.Timestamp,
    stations: pd.DataFrame,
    adj: pd.DataFrame,
    grids: pd.DataFrame,
    closeness: dict[str, float] | None = None,
    *,
    closeness_cache: dict[tuple[tuple[str, ...], str], dict[str, float]] | None = None,
) -> pd.DataFrame:
    # Historical callers supplied full-network scores. Never trust those
    # scores: rebuild from the eligible graph even when this argument is set.
    del closeness
    cutoff = opening_month - pd.DateOffset(months=ANTICIPATION_MONTHS)
    snap = stations[stations["opening_date"] <= cutoff].copy()
    station_xy = snap[["wgs84_lon", "wgs84_lat"]].to_numpy(dtype=float)
    snap = snap.loc[_coordinate_mask(station_xy)].reset_index(drop=True)
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

    edges = _edges_at_cutoff(adj, cutoff)
    cache_key = (
        tuple(sorted(str(value) for value in snap["station_event_id"])),
        _frame_sha256(snap) + _frame_sha256(edges),
    )
    if closeness_cache is not None and cache_key in closeness_cache:
        scores = closeness_cache[cache_key]
    else:
        scores = _closeness_once(snap, edges)
        if closeness_cache is not None:
            closeness_cache[cache_key] = scores
    snap["_line_set"] = snap["lines"].apply(lambda s: frozenset(_parse_lines(s)))
    grid_xy = grids[["centroid_lon", "centroid_lat"]].to_numpy(dtype=float)
    valid_rows = np.flatnonzero(_coordinate_mask(grid_xy))
    station_xyz = _unit_sphere(snap[["wgs84_lon", "wgs84_lat"]].to_numpy(dtype=float))
    grid_xyz = _unit_sphere(grid_xy[valid_rows])
    tree = cKDTree(station_xyz)
    nearest_chord, nearest_idx = tree.query(grid_xyz, k=1)
    nearest_m = np.full(len(grids), np.nan)
    nearest_m[valid_rows] = _chord_to_m(nearest_chord)
    radius_chord = 2 * np.sin(max(BUFFER_RADII_M) / (2 * EARTH_RADIUS_KM * 1000))
    candidates = tree.query_ball_point(grid_xyz, r=radius_chord + 1e-14)
    station_counts = {radius: np.zeros(len(grids), dtype=int) for radius in BUFFER_RADII_M}
    line_counts = np.zeros(len(grids), dtype=int)
    station_line_sets = snap["_line_set"].to_numpy()
    for position, row in enumerate(valid_rows):
        js = np.asarray(candidates[position], dtype=int)
        dist_m = _chord_to_m(np.linalg.norm(station_xyz[js] - grid_xyz[position], axis=1))
        for radius in BUFFER_RADII_M:
            station_counts[radius][row] = int((dist_m <= radius + 1e-7).sum())
        union = set()
        for j in js[dist_m <= max(BUFFER_RADII_M) + 1e-7]:
            union |= station_line_sets[j]
        line_counts[row] = len(union)
    closest_vals = np.zeros(len(grids))
    for position, row in enumerate(valid_rows):
        eid = snap["station_event_id"].iloc[nearest_idx[position]]
        closest_vals[row] = scores.get(eid, 0.0)

    out = grids[["grid_id"]].copy()
    out["city_key"] = city
    out["opening_month"] = opening_month
    out["dist_nearest_station_m"] = nearest_m
    for radius in BUFFER_RADII_M:
        out[f"stations_{radius}m"] = station_counts[radius]
    out["lines_in_1500m"] = line_counts
    out["network_closeness"] = closest_vals
    return out[cols]


def _cache_matches(target: Path, identity: dict[str, object]) -> bool:
    metadata = target.with_suffix(".metadata.json")
    if not target.is_file() or not metadata.is_file():
        return False
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        record.get("identity") == identity
        and record.get("sha256") == hashlib.sha256(target.read_bytes()).hexdigest()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="all")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild incompatible cached snapshots"
    )
    args = parser.parse_args()

    treated = pd.read_csv(BASE_DIR / "data" / "active" / "causal" / "treatment_unit_list.csv")
    cities = ACTIVE_CITIES if args.city == "all" else [args.city]
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        closeness_cache: dict[tuple[tuple[str, ...], str], dict[str, float]] = {}
        source_hashes = {
            "stations": _frame_sha256(stations),
            "adjacency": _frame_sha256(adj),
            "grids": _frame_sha256(grids),
        }
        city_dir = args.output_dir / city
        city_dir.mkdir(parents=True, exist_ok=True)
        for month in opening_months:
            label = month.strftime("%Y-%m")
            target = city_dir / f"{label}.parquet"
            identity = {
                "schema": SNAPSHOT_SCHEMA,
                "city": city,
                "opening_month": label,
                "anticipation_months": ANTICIPATION_MONTHS,
                "buffer_radii_m": list(BUFFER_RADII_M),
                "distance": "great_circle_sphere_6371km",
                "closeness": "inverse_sum_reachable_distance_km_on_cutoff_graph",
                "source_hashes": source_hashes,
                "edge_dates_available": "valid_from" in adj.columns,
                "topology_scope": "P197 adjacency restricted to cutoff-eligible stations and dated edges",
            }
            if target.exists():
                if _cache_matches(target, identity):
                    continue
                if not args.overwrite:
                    raise ValueError(
                        f"Incompatible or unversioned snapshot: {target}; "
                        "use a new --output-dir or --overwrite"
                    )
            frame = build_snapshot(
                city, month, stations, adj, grids, closeness_cache=closeness_cache
            )
            frame.to_parquet(target, index=False)
            target.with_suffix(".metadata.json").write_text(
                json.dumps(
                    {
                        "identity": identity,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(f"  {city}: {len(opening_months)} snapshots, {len(grids):,} grids each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
