"""Build pre-treatment transit accessibility and station-attribute features.

Network topology comes from Wikidata P197 adjacent-station relations
(``data/active/reference/transit/wikidata_adjacency.parquet``), which are the
authoritative station-level graph (To 2015-style network analysis), replacing
any coordinate-sorted approximation.

Measures (each mapped to literature in docs/research/transit_accessibility_method.md):

1. Pre-treatment network snapshot per treated grid (Gao & Wang 2026): only
   stations opened before the grid's treatment month minus the 12-month
   anticipation window.
2. Network construction (To 2015): stations as nodes; edges from Wikidata
   P197 (adjacent station), with straight-line km weights between the two
   stations (To 2015 uses station-pair distances as edge weights).
3. Closeness centrality per station (To 2015; Gao & Wang 2026), Dijkstra via
   networkx; isolated stations get 0.
4. Grid assignment (Wu et al. 2022): grid takes the closeness of its nearest
   pre-treatment station.
5. Buffer counts (Debrezion et al. 2007): stations_500m/800m/1500m and
   lines_in_1500m from the same pre-treatment snapshot.
6. Station attributes (from the treated grid's station event):
   - is_transfer_station: station belongs to >= 2 distinct lines
   - is_terminal_station: station has degree 1 on every line it belongs to
     within the pre-treatment network (Wikidata P197 gives the true line
     topology)
   - line_opening_year: earliest opening year among stations of each line;
     is_new_line = treated station's line had no earlier station; is_extension
     = the line already had stations opened earlier
   - stations_opened_same_month: number of stations in the same city opened
     in the treated station's opening month

All features use pre-treatment information only (DDR-004 constraint).

Output: data/active/causal/accessibility_features/{city}_accessibility.parquet
one row per treated grid (treatment_order key):
  treatment_order, city_key, grid_id,
  dist_nearest_station_m, stations_500m, stations_800m, stations_1500m,
  lines_in_1500m, network_closeness,
  is_transfer_station, is_terminal_station, is_new_line, is_extension,
  stations_opened_same_month

Usage:
    python scripts/analysis/build_accessibility_features.py --city all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from urban_intervention.config.project import ACTIVE_CITIES, GRID_DIR  # noqa: E402

REFERENCE_DIR = BASE_DIR / "data" / "active" / "reference"
ADJACENCY_PATH = REFERENCE_DIR / "transit" / "wikidata_adjacency.parquet"
OUT_DIR = BASE_DIR / "data" / "active" / "causal" / "accessibility_features"

ANTICIPATION_MONTHS = 12
BUFFER_RADII_M = (500, 800, 1500)
LINE_SPLIT = re.compile(r"[;；]")


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


def _load_city_topology(city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load project stations (mapped to event ids) and Wikidata P197 edges."""
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

    adj = pd.read_parquet(ADJACENCY_PATH)
    adj = adj[adj["city_key"] == city].copy()
    adj["_norm"] = adj["station_name"].apply(_norm_station)
    adj["_norm_adj"] = adj["adj_station_name"].apply(_norm_station)

    name_to_event = events.set_index("_norm")["station_event_id"]
    adj = adj[
        adj["_norm"].isin(name_to_event.index) & adj["_norm_adj"].isin(name_to_event.index)
    ].copy()
    adj["station_event_id"] = adj["_norm"].map(name_to_event)
    adj["adj_event_id"] = adj["_norm_adj"].map(name_to_event)
    return events, adj


def _line_opening_map(events: pd.DataFrame) -> dict[str, int]:
    """line -> earliest opening year among its stations."""
    mapping: dict[str, list[int]] = {}
    for _, row in events.iterrows():
        year = row["opening_date"].year
        for line in _parse_lines(row["lines"]):
            mapping.setdefault(line, []).append(year)
    return {line: min(years) for line, years in mapping.items()}


def _build_graph(snapshot_events: pd.DataFrame, adj: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    coords: dict[str, tuple[float, float]] = {}
    for _, row in snapshot_events.iterrows():
        eid = row["station_event_id"]
        coords[eid] = (row["wgs84_lon"], row["wgs84_lat"])
        g.add_node(eid)
    snap_ids = set(snapshot_events["station_event_id"])
    for _, row in adj.iterrows():
        a, b = row["station_event_id"], row["adj_event_id"]
        if a not in snap_ids or b not in snap_ids or a == b:
            continue
        if g.has_edge(a, b):
            continue
        d = _haversine_km(coords[a][0], coords[a][1], coords[b][0], coords[b][1])
        if d <= 0:
            continue
        g.add_edge(a, b, weight=d)
    return g


def _closeness(g: nx.Graph) -> dict[str, float]:
    result: dict[str, float] = {}
    nodes = list(g.nodes())
    for node in nodes:
        lengths = nx.single_source_dijkstra_path_length(g, node, weight="weight")
        total = sum(lengths.values()) - lengths.get(node, 0.0)
        if total > 0 and len(lengths) > 1:
            result[node] = 1.0 / total
        else:
            result[node] = 0.0
    return result


def build_city(city: str, treated: pd.DataFrame) -> pd.DataFrame:
    events, adj = _load_city_topology(city)
    line_first_year = _line_opening_map(events)
    grids = pd.read_parquet(
        GRID_DIR / city / f"{city}_grids.parquet",
        columns=["grid_id", "centroid_lon", "centroid_lat"],
    )
    grid_lookup = grids.set_index("grid_id")
    events_indexed = events.set_index("station_event_id")

    rows = []
    for _, trow in treated.iterrows():
        grid_id = trow["grid_id"]
        opening = pd.to_datetime(trow["opening_month"])
        cutoff = opening - pd.DateOffset(months=ANTICIPATION_MONTHS)
        snap = events[events["opening_date"] <= cutoff].copy()
        if snap.empty:
            rows.append(_empty_row(trow, grid_id))
            continue

        g = _build_graph(snap, adj)
        closeness = _closeness(g)
        coords = {
            eid: (r["wgs84_lon"], r["wgs84_lat"])
            for eid, r in snap.set_index("station_event_id").iterrows()
        }

        # Network as of the treated station's own opening month (includes the
        # treated station and any line opened simultaneously).  The month end
        # boundary ensures stations with day-level dates inside the same month
        # are included.  Used for the terminal-station judgement: a terminal
        # is degree <= 1 on this graph.
        month_end = opening + pd.offsets.MonthEnd(0)
        network_at_opening = events[events["opening_date"] <= month_end].copy()
        g_opening = _build_graph(network_at_opening, adj)

        glon, glat = (
            grid_lookup.loc[grid_id, "centroid_lon"],
            grid_lookup.loc[grid_id, "centroid_lat"],
        )
        dists = {eid: _haversine_km(glon, glat, lon, lat) for eid, (lon, lat) in coords.items()}
        if not dists:
            rows.append(_empty_row(trow, grid_id))
            continue
        nearest = min(dists, key=dists.get)
        dist_nearest_m = dists[nearest] * 1000.0

        n500 = int(sum(1 for d in dists.values() if d * 1000 <= 500))
        n800 = int(sum(1 for d in dists.values() if d * 1000 <= 800))
        n1500 = int(sum(1 for d in dists.values() if d * 1000 <= 1500))
        lines_in = set()
        for eid, d in dists.items():
            if d * 1000 <= 1500:
                lines_in.update(_parse_lines(events_indexed.loc[eid, "lines"]))
        row = {
            "treatment_order": trow["treatment_order"],
            "city_key": city,
            "grid_id": grid_id,
            "dist_nearest_station_m": dist_nearest_m,
            "stations_500m": n500,
            "stations_800m": n800,
            "stations_1500m": n1500,
            "lines_in_1500m": len(lines_in),
            "network_closeness": closeness.get(nearest, 0.0),
        }

        # ── Station attributes of the treated station ──────────────
        if trow["station_event_id"] in events_indexed.index:
            treated_event = events_indexed.loc[trow["station_event_id"]]
        else:
            treated_event = None
        if treated_event is None:
            row.update(
                {
                    "is_transfer_station": 0,
                    "is_terminal_station": 0,
                    "is_new_line": 0,
                    "is_extension": 0,
                    "stations_opened_same_month": 0,
                }
            )
        else:
            treated_lines = _parse_lines(treated_event["lines"])
            row["is_transfer_station"] = int(len(treated_lines) >= 2)
            row["is_terminal_station"] = int(
                g_opening.degree(trow["station_event_id"]) <= 1
                if g_opening.has_node(trow["station_event_id"])
                else 0
            )
            row["stations_opened_same_month"] = int(
                (events["opening_date"].dt.to_period("M") == opening.to_period("M")).sum()
            )
            new_line = 1
            extension = 0
            for line in treated_lines:
                first = line_first_year.get(line)
                if first is not None and first < treated_event["opening_date"].year:
                    extension = 1
                    new_line = 0
            row["is_new_line"] = new_line
            row["is_extension"] = extension
        rows.append(row)
    return pd.DataFrame(rows)


def _empty_row(trow: pd.Series, grid_id: str) -> dict:
    return {
        "treatment_order": trow["treatment_order"],
        "city_key": trow["city_key"],
        "grid_id": grid_id,
        "dist_nearest_station_m": np.nan,
        "stations_500m": 0,
        "stations_800m": 0,
        "stations_1500m": 0,
        "lines_in_1500m": 0,
        "network_closeness": 0.0,
        "is_transfer_station": 0,
        "is_terminal_station": 0,
        "is_new_line": 0,
        "is_extension": 0,
        "stations_opened_same_month": 0,
    }


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
        frame = build_city(city, city_treated)
        out = OUT_DIR / f"{city}_accessibility.parquet"
        frame.to_parquet(out, index=False)
        n_valid = int(frame["dist_nearest_station_m"].notna().sum())
        print(
            f"  {city}: {len(frame)} treated grids, {n_valid} with station, "
            f"transfer {(frame['is_transfer_station'] == 1).sum()}, "
            f"terminal {(frame['is_terminal_station'] == 1).sum()}, "
            f"new_line {(frame['is_new_line'] == 1).sum()}, "
            f"ext {(frame['is_extension'] == 1).sum()}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
