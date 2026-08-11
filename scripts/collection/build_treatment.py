"""Build grid × year treatment matrix from a chosen station source.

Multi-city (44). Source selection via --source:
  auto      : pick best source per city from comparison report, else merged union
  merged    : use {city}_metro_stations_merged.csv (union of all sources)
  osm|amap|wikidata|wiki : use that single source CSV
  legacy    : old OSM+Amap+Wiki merge behavior (preserved for reproducibility)

Drops under-construction / planned stations so they can't pollute nearest-
station or treatment fields.

Usage:
    python build_treatment.py --city all --source auto
    python build_treatment.py --city beijing --source osm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "src"))
from urban_intervention.config.project import ACTIVE_CITIES, CITIES
from urban_intervention.config.project import norm_station_name as _norm
from urban_intervention.data.paths import (
    OUTPUT_TRANSIT_COMPARISON_DIR,
    RAW_TRANSIT_DIR,
)

# ── Grid treatment computation ──────────────────────────────────


def _vec_haversine_m(grid_lons, grid_lats, sta_lons, sta_lats):
    """Vectorized Haversine: (G,) x (S,) → (G, S) distance matrix in meters."""
    R = 6371000
    glon_r = np.radians(grid_lons)
    glat_r = np.radians(grid_lats)
    slon_r = np.radians(sta_lons)
    slat_r = np.radians(sta_lats)
    dlat = glat_r[:, None] - slat_r[None, :]
    dlon = glon_r[:, None] - slon_r[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(glat_r[:, None]) * np.cos(slat_r[None, :]) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_grid_treatment(
    grids: pd.DataFrame,
    stations: pd.DataFrame,
    buffers_m: tuple = (200, 500, 800, 1500),
    year_range: range = range(2010, 2026),
) -> pd.DataFrame:
    """For each (grid, year), count stations within each buffer that opened <= that year.
    Uses vectorized Haversine — O(G×S) in C, much faster than Python loops."""
    grid_coords = grids[["grid_id", "centroid_lon", "centroid_lat"]].values
    G = len(grid_coords)

    # ── Empty-stations guard ───────────────────────────────────────
    # If stations is empty (e.g. all were dropped by _drop_under_construction),
    # emit an all-zero treatment table so downstream code does not crash on
    # `masked.min(axis=1)` with a zero-size axis.
    if stations.empty:
        print(f"  Computing treatment for {G} grids x 0 stations x {len(year_range)} years...")
        print("  [WARN] no stations; emitting all-zero treatment table")
        rows = []
        for gi in range(G):
            gid = grid_coords[gi, 0]
            for yr in year_range:
                row = {"grid_id": gid, "year": yr}
                for buf in buffers_m:
                    row[f"stations_{buf}m"] = 0
                    row[f"has_metro_{buf}m"] = 0
                    row[f"first_treatment_year_{buf}m"] = None
                row["nearest_station_m"] = np.nan
                row["nearest_station_name"] = ""
                row["nearest_station_year"] = None
                rows.append(row)
        return pd.DataFrame(rows)

    station_coords = stations[["wgs84_lon", "wgs84_lat"]].values
    station_years = stations["opening_year"].fillna(9999).values

    S = len(station_coords)
    print(f"  Computing treatment for {G} grids x {S} stations x {len(year_range)} years...")

    glons = grid_coords[:, 1].astype(float)
    glats = grid_coords[:, 2].astype(float)
    slons = station_coords[:, 0].astype(float)
    slats = station_coords[:, 1].astype(float)

    # Vectorized: all pairwise Haversine distances at once
    print("    Computing distance matrix (vectorized)...")
    dists = _vec_haversine_m(glons, glats, slons, slats)  # (G, S)

    print("    Building year-by-year treatment table...")
    # Precompute per-grid, per-buffer first treatment year (does not depend
    # on yr). Done once outside the year loop instead of len(year_range) times.
    # Replace invalid (year==9999) with +inf so np.min skips them.
    sy_for_min = np.where(station_years <= 9998, station_years, np.iinfo(np.int64).max)
    sy_broadcast = sy_for_min[None, :]  # (1, S)
    # For each buffer: where dist <= buf, take min year over stations; else None.
    first_year_per_buf: dict[int, np.ndarray] = {}  # buf -> (G,) array
    for _bi, buf in enumerate(buffers_m):
        within = dists <= buf  # (G, S)
        # Use a masked reduction: set non-within to +inf, then min over axis=1.
        masked = np.where(within, sy_broadcast, np.iinfo(np.int64).max)
        min_yr = masked.min(axis=1)  # (G,)
        first_year_per_buf[buf] = min_yr

    rows = []
    for gi in range(G):
        if gi % 500 == 0:
            print(f"    Grid {gi}/{G}...")
        gid = grid_coords[gi, 0]
        gdists = dists[gi]  # (S,)

        # Nearest station (independent of year)
        min_idx = int(np.argmin(gdists))
        nearest_station_m = round(float(gdists[min_idx]), 1)
        nearest_station_name = stations.iloc[min_idx].get("station_name", "")
        nearest_station_year = (
            int(station_years[min_idx]) if station_years[min_idx] < 9999 else None
        )

        # Precompute first-treatment-year per buffer for this grid
        first_year_bufs = {
            buf: int(first_year_per_buf[buf][gi])
            if first_year_per_buf[buf][gi] != np.iinfo(np.int64).max
            else None
            for buf in buffers_m
        }

        for yr in year_range:
            row = {"grid_id": gid, "year": yr}
            yr_mask = station_years <= yr
            for buf in buffers_m:
                mask = (gdists <= buf) & yr_mask
                count = int(mask.sum())
                row[f"stations_{buf}m"] = count
                row[f"has_metro_{buf}m"] = int(count > 0)
                row[f"first_treatment_year_{buf}m"] = first_year_bufs[buf]
            # Nearest station
            row["nearest_station_m"] = nearest_station_m
            row["nearest_station_name"] = nearest_station_name
            row["nearest_station_year"] = nearest_station_year
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Treatment table: {len(df)} rows, {len(df.columns)} columns")
    return df


# ── Merge OSM + Amap (coordinate-level merge) ──────────────────


def merge_osm_amap(osm_df: pd.DataFrame, amap_df: pd.DataFrame) -> pd.DataFrame:
    """Merge OSM and Amap station data. Overlaps resolved by coordinate proximity.
    Returns merged DataFrame with station_name, wgs84_lon, wgs84_lat, source.

    Does NOT mutate the caller's DataFrames — both inputs are copied first.
    """
    if osm_df.empty:
        return amap_df.assign(source="amap").copy()
    if amap_df.empty:
        return osm_df.assign(source="osm").copy()

    # Copy to avoid mutating caller's frames when we add _n / source columns.
    osm_df = osm_df.copy()
    amap_df = amap_df.copy()

    # Normalize names for overlap detection
    osm_df["_n"] = osm_df["station_name"].apply(_norm)
    amap_df["_n"] = amap_df["station_name"].apply(_norm)

    # Find overlaps by normalized name
    osm_names = set(osm_df["_n"])
    amap_names = set(amap_df["_n"])
    overlap = osm_names & amap_names

    # Build merged: start with OSM, add non-overlapping Amap
    osm_df["source"] = "osm"
    amap_only = amap_df[~amap_df["_n"].isin(overlap)].copy()
    amap_only["source"] = "amap"

    merged = pd.concat([osm_df, amap_only], ignore_index=True)
    merged = merged.drop(columns=["_n"], errors="ignore")
    print(
        f"  OSM+Amap merge: OSM={len(osm_df)}, Amap={len(amap_df)}, overlap={len(overlap)}, merged={len(merged)}"
    )
    return merged


# ── Main per city ───────────────────────────────────────────────

from urban_intervention.config.project import GRID_DIR, TREATMENT_DIR, city_dir, get_city_config

# Future opening years beyond this are treated as "not yet operating" and the
# station is dropped so it can't pollute nearest-station / treatment fields.
MAX_OPERATING_YEAR = 2024
BEST_REPORT = OUTPUT_TRANSIT_COMPARISON_DIR / "best_source_per_city.csv"


def _drop_under_construction(df: pd.DataFrame) -> pd.DataFrame:
    """Remove stations that are not yet operating.

    Two filters are applied:
      1. Name/line keyword filter: station_name or line contains 在建 / 规划 / (拟).
      2. Year filter: opening_year > MAX_OPERATING_YEAR (future openings).
    """
    if df.empty:
        return df

    def _is_uc(s):
        return isinstance(s, str) and ("在建" in s or "规划" in s or "(拟)" in s)

    # Build boolean Series on df's own index so | alignment is correct even
    # when df has a non-default index (e.g. after a filter without reset_index).
    bad_name = df["station_name"].apply(_is_uc)
    if "line" in df.columns:
        bad_line = df["line"].fillna("").apply(_is_uc)
    else:
        bad_line = pd.Series([False] * len(df), index=df.index)
    # Year filter: drop stations with opening_year beyond MAX_OPERATING_YEAR
    if "opening_year" in df.columns:

        def _is_future(y):
            if not pd.notna(y) or y is None:
                return False
            try:
                return float(y) > MAX_OPERATING_YEAR
            except (ValueError, TypeError):
                # Non-numeric year (e.g. "待定") — can't determine, keep it
                return False

        bad_year = df["opening_year"].apply(_is_future)
    else:
        bad_year = pd.Series([False] * len(df), index=df.index)
    mask = ~(bad_name | bad_line | bad_year)
    dropped = int((~mask).sum())
    if dropped:
        print(f"  Dropped {dropped} under-construction / planned / future-opening stations")
    return df[mask].reset_index(drop=True)


def _load_source_csv(ck: str, tag: str) -> pd.DataFrame | None:
    """Load {city}_metro_stations_{tag}.csv, falling back to legacy filenames."""
    transit_dir = RAW_TRANSIT_DIR / ck
    cands = [transit_dir / f"{ck}_metro_stations_{tag}.csv"]
    if tag == "wiki":
        cands = [transit_dir / f"{ck}_stations_wiki.csv"]
    for p in cands:
        if p.exists():
            return pd.read_csv(p, encoding="utf-8-sig")
    return None


def _resolve_source(ck: str, requested: str) -> tuple[str, pd.DataFrame]:
    """Return (tag, df). Resolves 'auto' via best-source report, then merged,
    then any available source. Raises if nothing available."""
    if requested not in ("auto", "legacy"):
        df = _load_source_csv(ck, requested)
        if df is None:
            raise FileNotFoundError(f"source '{requested}' not found for {ck}")
        return requested, df

    if requested == "auto":
        # 1. best-source report
        if BEST_REPORT.exists():
            rep = pd.read_csv(BEST_REPORT, encoding="utf-8-sig")
            row = rep[rep["city"] == ck]
            if not row.empty:
                best = str(row.iloc[0]["best_source"])
                # wiki source has no coordinates — skip as standalone and
                # fall through to merged / other sources below.
                if best != "wiki":
                    df = _load_source_csv(ck, best)
                    if df is not None:
                        return best, df
        # 2. merged union
        df = _load_source_csv(ck, "merged")
        if df is not None:
            return "merged", df
        # 3. first available of preferred order
        for tag in ["osm", "wikidata", "amap", "wiki"]:
            df = _load_source_csv(ck, tag)
            if df is not None:
                return tag, df

    raise FileNotFoundError(f"no station source available for {ck} (requested='{requested}')")


def _fill_year_fallback(df: pd.DataFrame, ck: str) -> pd.DataFrame:
    """If opening_year still missing for some rows, fill with city first-line year."""
    if "opening_year" not in df.columns:
        df["opening_year"] = None
    n_missing = int(df["opening_year"].isna().sum())
    if n_missing == 0:
        return df
    from urban_intervention.config.project import METRO_REFERENCE

    first_yr = METRO_REFERENCE.get(ck, {}).get("first_line_opened")
    if first_yr:
        df["opening_year"] = df["opening_year"].fillna(int(first_yr))
        print(f"  Fallback year={first_yr} for {n_missing} stations missing year")
    return df


def _compute_and_save(
    grids: pd.DataFrame, stations: pd.DataFrame, ck: str, suffix: str = ""
) -> None:
    """Compute treatment for one grid type and save the two parquet outputs."""
    treatment = compute_grid_treatment(grids, stations)
    treatment_dir = city_dir(ck, TREATMENT_DIR)
    treatment_dir.mkdir(parents=True, exist_ok=True)
    treatment.to_parquet(treatment_dir / f"{ck}_grid_treatment{suffix}.parquet")
    print(f"  Saved: {treatment_dir / f'{ck}_grid_treatment{suffix}.parquet'}")

    merge_cols = ["grid_id", "nearest_station_m", "nearest_station_name", "nearest_station_year"]
    merge_cols += ["has_metro_200m", "has_metro_500m", "has_metro_800m", "has_metro_1500m"]
    merge_cols += ["stations_200m", "stations_500m", "stations_800m", "stations_1500m"]
    merge_cols += [
        "first_treatment_year_200m",
        "first_treatment_year_500m",
        "first_treatment_year_800m",
        "first_treatment_year_1500m",
    ]
    latest = treatment.sort_values("year").drop_duplicates("grid_id", keep="last")
    available = [c for c in merge_cols if c in latest.columns and c != "grid_id"]
    grids_out = grids.merge(latest[["grid_id"] + available], on="grid_id", how="left")
    grids_out.to_parquet(treatment_dir / f"{ck}_grids_with_treatment{suffix}.parquet")
    print(f"  Saved: {treatment_dir / f'{ck}_grids_with_treatment{suffix}.parquet'}")


def run_city(ck: str, source: str = "auto") -> None:
    cfg = get_city_config(ck)
    name = cfg["name"]
    print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")

    grid_p = city_dir(ck, GRID_DIR) / f"{ck}_grids.parquet"
    if not grid_p.exists():
        print(f"  [!] Missing grid data: {grid_p}")
        return

    # ── Legacy path: keep original OSM+Amap+Wiki merge behavior ──
    if source == "legacy":
        return _run_city_legacy(ck, grid_p)

    # ── New path: pick one source (or merged union) ──
    tag, stations = _resolve_source(ck, source)
    print(f"  Source: {tag} ({len(stations)} rows)")
    stations = _drop_under_construction(stations)

    if "line" not in stations.columns:
        stations["line"] = ""
    stations = _fill_year_fallback(stations, ck)
    # Drop stations still missing coords (some sources may lack them)
    if "wgs84_lon" not in stations.columns or "wgs84_lat" not in stations.columns:
        print(f"  [!] source '{tag}' has no wgs84_lon/lat columns")
        return
    before = len(stations)
    stations = stations.dropna(subset=["wgs84_lon", "wgs84_lat"]).reset_index(drop=True)
    if len(stations) < before:
        print(f"  Dropped {before - len(stations)} rows missing coords")

    # ── Admin-boundary grid (primary) ───────────────────────────
    grids = pd.read_parquet(grid_p)
    print(f"  [Admin grid] {len(grids)} cells, {len(stations)} stations")
    _compute_and_save(grids, stations, ck, suffix="")

    # ── Station-centred grid (secondary) ────────────────────────
    station_grid_p = city_dir(ck, GRID_DIR) / f"{ck}_grids_station.parquet"
    if station_grid_p.exists():
        grids_st = pd.read_parquet(station_grid_p)
        print(f"  [Station grid] {len(grids_st)} cells, {len(stations)} stations")
        _compute_and_save(grids_st, stations, ck, suffix="_station")
    else:
        print("  [Station grid] not found — run grid_builder.py first")


def _run_city_legacy(ck: str, grid_p: Path) -> None:
    """Original OSM+Amap+Wiki merge behavior (preserved for reproducibility)."""
    transit_dir = RAW_TRANSIT_DIR / ck
    osm_p = transit_dir / f"{ck}_metro_stations_osm.csv"
    amap_p = transit_dir / f"{ck}_metro_stations_amap.csv"
    wiki_p = transit_dir / f"{ck}_stations_wiki.csv"

    stations_list = []
    if osm_p.exists():
        osm_df = pd.read_csv(osm_p, encoding="utf-8-sig")
        osm_df["_src"] = "osm"
        stations_list.append(osm_df)
        print(f"  OSM: {len(osm_df)} stations")
    if amap_p.exists():
        amap_df = pd.read_csv(amap_p, encoding="utf-8-sig")
        amap_df["_src"] = "amap"
        stations_list.append(amap_df)
        print(f"  Amap: {len(amap_df)} stations")

    if not stations_list:
        print("  [!] No station data (need osm or amap CSV)")
        return

    if len(stations_list) == 2:
        stations = merge_osm_amap(stations_list[0], stations_list[1])
    else:
        stations = stations_list[0]
        stations["source"] = stations["_src"]
    stations = stations.drop(columns=["_src"], errors="ignore")

    if wiki_p.exists():
        wiki = pd.read_csv(wiki_p, encoding="utf-8-sig")
        wiki["_n"] = wiki["station_name"].apply(_norm)
        wiki_map = {}
        for _, row in wiki.iterrows():
            key = row["_n"]
            line = str(row.get("line", "")).strip()
            yr = row.get("opening_year")
            yr = int(yr) if pd.notna(yr) and yr is not None else None
            if key not in wiki_map:
                wiki_map[key] = (line, yr)
            else:
                old_line, old_yr = wiki_map[key]
                new_line = old_line + ";" + line if old_line != line else old_line
                new_yr = yr if yr and (old_yr is None or yr < old_yr) else old_yr
                wiki_map[key] = (new_line, new_yr)

        # ── OSM-first merge ────────────────────────────────────────
        # Previous code unconditionally overwrote `line` and `opening_year`
        # with wiki values, even when wiki had no entry (which wiped OSM's
        # data to "" / None).  Now we only fill in fields that are missing
        # in the OSM/Amap merged frame.
        stations["_n"] = stations["station_name"].apply(_norm)

        # Build wiki-aligned series, then use combine_first / where to fill
        # only the missing values.
        wiki_lookup = stations["_n"].map(lambda n: wiki_map.get(n, ("", None)))
        wiki_line_series = wiki_lookup.apply(lambda x: x[0] if x else "")
        wiki_year_series = wiki_lookup.apply(lambda x: x[1] if x else None)

        # Ensure columns exist
        if "line" not in stations.columns:
            stations["line"] = ""
        if "opening_year" not in stations.columns:
            stations["opening_year"] = None

        # Replace empty/NaN line values with wiki's value (if any).
        empty_line_mask = stations["line"].isna() | (stations["line"].astype(str) == "")
        stations.loc[empty_line_mask, "line"] = wiki_line_series[empty_line_mask]

        # Replace missing opening_year with wiki's value (if any).
        missing_year_mask = stations["opening_year"].isna()
        stations.loc[missing_year_mask, "opening_year"] = wiki_year_series[missing_year_mask]

        stations = stations.drop(columns=["_n"], errors="ignore")
        n_wiki = stations["opening_year"].notna().sum()
        print(
            f"  Wiki: {n_wiki}/{len(stations)} with year (OSM-first merge — wiki fills gaps only)"
        )
    else:
        if "opening_year" not in stations.columns:
            stations["opening_year"] = None
        from urban_intervention.config.project import METRO_REFERENCE

        first_yr = METRO_REFERENCE.get(ck, {}).get("first_line_opened")
        if first_yr:
            stations["opening_year"] = stations["opening_year"].fillna(int(first_yr))

    if "line" not in stations.columns:
        stations["line"] = ""

    # Drop under-construction / planned / future-opening stations so
    # they don't pollute the treatment matrix (matches new-path behavior).
    stations = _drop_under_construction(stations)

    grids = pd.read_parquet(grid_p)
    print(f"  Grids: {len(grids)}, Stations: {len(stations)}")

    _compute_and_save(grids, stations, ck, suffix="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="all")
    parser.add_argument(
        "--source",
        default="auto",
        help="Station source: auto|legacy|merged|osm|amap|wikidata|wiki "
        "(auto = best from comparison report; legacy = old OSM+Amap+Wiki)",
    )
    args = parser.parse_args()
    cities = []
    for c in args.city.split(","):
        c = c.strip()
        if c == "all":
            cities = list(ACTIVE_CITIES)
            break
        if c in CITIES:
            cities.append(c)
        else:
            print(f"[WARN] unknown city key '{c}' — skipping")
    for ck in cities:
        try:
            run_city(ck, source=args.source)
        except FileNotFoundError as e:
            print(f"\n[{ck}] {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
