"""Grid aggregation and panel persistence for normalized POI rows."""

import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import geopandas as gpd  # noqa: F401  (used in string annotations and lazily)
import pandas as pd

from urban_intervention.config.project import GRID_DIR

from .config import ANALYSIS_CATEGORIES, OUT_DIR
from .taxonomy import shannon_entropy


@contextmanager
def _exclusive_file_lock(path: Path):
    """Hold an OS-level exclusive lock for one city-panel destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if hasattr(msvcrt, "lockf"):
                msvcrt.lockf(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl  # type: ignore[attr-defined]  # noqa: F401

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # type: ignore[attr-defined]  # noqa: F401

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        finally:
            handle.close()


def load_city_grid(city_key: str):
    import geopandas as gpd
    from shapely import wkt

    path = GRID_DIR / city_key / f"{city_key}_grids.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    grids = pd.read_parquet(path, columns=["grid_id", "geometry_wkt"])
    grids["geometry"] = grids["geometry_wkt"].apply(wkt.loads)
    grids = grids.drop(columns=["geometry_wkt"])
    return gpd.GeoDataFrame(grids, geometry="geometry", crs="EPSG:4326")


def load_city_grids(city_keys: list[str]):
    import geopandas as gpd

    parts = []
    for city_key in city_keys:
        grids = load_city_grid(city_key)
        grids.insert(0, "city", city_key)
        parts.append(grids)
    if not parts:
        return gpd.GeoDataFrame(
            columns=["city", "grid_id", "geometry"], geometry="geometry", crs="EPSG:4326"
        )
    return gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )


def aggregate_chunk(chunk: "pd.DataFrame | gpd.GeoDataFrame", grids) -> pd.DataFrame:
    import geopandas as gpd

    needed = {"name", "category", "is_commercial", "is_chain", "is_community_commerce"}
    missing = needed - set(chunk.columns)
    if missing:
        raise ValueError(f"Normalized POI chunk is missing columns: {sorted(missing)}")

    if chunk.empty:
        return pd.DataFrame()

    _has_geom = hasattr(chunk, "geometry") and chunk.geometry is not None

    if _has_geom:
        points = chunk.dropna(subset=["geometry"])
        if points.empty:
            return pd.DataFrame()
        points = points.drop(columns=["city"], errors="ignore")
        bounds = grids.total_bounds
        points = points.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]]
        if points.empty:
            return pd.DataFrame()
    else:
        chunk = chunk.dropna(subset=["lon", "lat"]).copy()
        if chunk.empty:
            return pd.DataFrame()
        bounds = grids.total_bounds
        chunk = chunk[
            (chunk["lon"] >= bounds[0])
            & (chunk["lon"] <= bounds[2])
            & (chunk["lat"] >= bounds[1])
            & (chunk["lat"] <= bounds[3])
        ].copy()
        if chunk.empty:
            return pd.DataFrame()
        points = gpd.GeoDataFrame(
            chunk,
            geometry=gpd.points_from_xy(chunk["lon"], chunk["lat"]),
            crs="EPSG:4326",
        )

    joined = gpd.sjoin(points, grids[["grid_id", "geometry"]], how="inner", predicate="within")
    if joined.empty:
        return pd.DataFrame()

    agg = joined.groupby("grid_id").agg(
        poi_count=("name", "size"),
        poi_commercial_count=("is_commercial", "sum"),
        poi_chain_count=("is_chain", "sum"),
        poi_community_commerce_count=("is_community_commerce", "sum"),
    )
    cat = pd.crosstab(joined["grid_id"], joined["category"])
    for category in ANALYSIS_CATEGORIES:
        if category not in cat.columns:
            cat[category] = 0
    cat = cat[ANALYSIS_CATEGORIES].rename(columns=lambda c: f"poi_{c}_count")
    return agg.join(cat, how="left").reset_index()


def aggregate_chunk_multi_city(chunk: "pd.DataFrame | gpd.GeoDataFrame", grids) -> pd.DataFrame:
    import geopandas as gpd

    needed = {"name", "category", "is_commercial", "is_chain", "is_community_commerce"}
    missing = needed - set(chunk.columns)
    if missing:
        raise ValueError(f"Normalized POI chunk is missing columns: {sorted(missing)}")

    if chunk.empty:
        return pd.DataFrame()

    _has_geom = hasattr(chunk, "geometry") and chunk.geometry is not None

    if _has_geom:
        points = chunk.dropna(subset=["geometry"])
        if points.empty:
            return pd.DataFrame()
        points = points.drop(columns=["city"], errors="ignore")
        bounds = grids.total_bounds
        points = points.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]]
        if points.empty:
            return pd.DataFrame()
    else:
        chunk = chunk.dropna(subset=["lon", "lat"]).copy()
        if chunk.empty:
            return pd.DataFrame()
        bounds = grids.total_bounds
        chunk = chunk[
            (chunk["lon"] >= bounds[0])
            & (chunk["lon"] <= bounds[2])
            & (chunk["lat"] >= bounds[1])
            & (chunk["lat"] <= bounds[3])
        ].copy()
        if chunk.empty:
            return pd.DataFrame()
        point_attrs = chunk.drop(columns=["city"], errors="ignore")
        points = gpd.GeoDataFrame(
            point_attrs,
            geometry=gpd.points_from_xy(point_attrs["lon"], point_attrs["lat"]),
            crs="EPSG:4326",
        )

    joined = gpd.sjoin(
        points, grids[["city", "grid_id", "geometry"]], how="inner", predicate="within"
    )
    if joined.empty:
        return pd.DataFrame()

    group_keys = ["city", "grid_id"]
    agg = joined.groupby(group_keys).agg(
        poi_count=("name", "size"),
        poi_commercial_count=("is_commercial", "sum"),
        poi_chain_count=("is_chain", "sum"),
        poi_community_commerce_count=("is_community_commerce", "sum"),
    )
    cat = pd.crosstab([joined["city"], joined["grid_id"]], joined["category"])
    for category in ANALYSIS_CATEGORIES:
        if category not in cat.columns:
            cat[category] = 0
    cat = cat[ANALYSIS_CATEGORIES].rename(columns=lambda c: f"poi_{c}_count")
    return agg.join(cat, how="left").reset_index()


def finalize_year(city_key: str, year: int, parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["city", "grid_id", "year", "poi_count"])
    df = pd.concat(parts, ignore_index=True)
    numeric_cols = [c for c in df.columns if c != "grid_id"]
    out = df.groupby("grid_id", as_index=False)[numeric_cols].sum()
    cat_cols = [f"poi_{c}_count" for c in ANALYSIS_CATEGORIES]
    out["poi_category_entropy"] = out[cat_cols].apply(shannon_entropy, axis=1)
    out["poi_commercial_share"] = out["poi_commercial_count"] / out["poi_count"].where(
        out["poi_count"] > 0
    )
    out["poi_chain_share"] = out["poi_chain_count"] / out["poi_count"].where(out["poi_count"] > 0)
    out["poi_community_commerce_share"] = out["poi_community_commerce_count"] / out[
        "poi_count"
    ].where(out["poi_count"] > 0)
    out.insert(0, "city", city_key)
    out.insert(2, "year", year)
    return out


def finalize_multi_city_year(year: int, parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["city", "grid_id", "year", "poi_count"])
    df = pd.concat(parts, ignore_index=True)
    numeric_cols = [c for c in df.columns if c not in {"city", "grid_id"}]
    out = df.groupby(["city", "grid_id"], as_index=False)[numeric_cols].sum()
    cat_cols = [f"poi_{c}_count" for c in ANALYSIS_CATEGORIES]
    out["poi_category_entropy"] = out[cat_cols].apply(shannon_entropy, axis=1)
    out["poi_commercial_share"] = out["poi_commercial_count"] / out["poi_count"].where(
        out["poi_count"] > 0
    )
    out["poi_chain_share"] = out["poi_chain_count"] / out["poi_count"].where(out["poi_count"] > 0)
    out["poi_community_commerce_share"] = out["poi_community_commerce_count"] / out[
        "poi_count"
    ].where(out["poi_count"] > 0)
    out.insert(2, "year", year)
    return out


def add_change_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    panel = panel.sort_values(["city", "grid_id", "year"]).copy()
    panel["poi_count_lag"] = panel.groupby(["city", "grid_id"])["poi_count"].shift(1)
    diff = panel["poi_count"] - panel["poi_count_lag"]
    panel["poi_net_new_count_proxy"] = diff.clip(lower=0).fillna(0)
    panel["poi_net_exit_count_proxy"] = (-diff).clip(lower=0).fillna(0)
    panel["poi_growth_rate"] = diff / panel["poi_count_lag"].where(panel["poi_count_lag"] > 0)
    return panel


def save_city_panel(city_key: str, frames: list[pd.DataFrame]) -> Path | None:
    frames = [f for f in frames if not f.empty]
    if not frames:
        print(f"{city_key}: no POI rows to save")
        return None
    out_path = OUT_DIR / f"{city_key}_poi_grid_yearly.parquet"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    with _exclusive_file_lock(lock_path):
        panel = pd.concat(frames, ignore_index=True)
        if out_path.exists():
            existing_years = set(pd.read_parquet(out_path, columns=["year"])["year"].unique())
            new_years = set(panel["year"].unique())
            years_to_keep = existing_years - new_years
            if years_to_keep:
                old = pd.read_parquet(out_path)
                old = old[old["year"].isin(years_to_keep)]
                panel = pd.concat([old, panel], ignore_index=True)

        panel = add_change_metrics(panel)
        tmp_path = out_path.with_suffix(out_path.suffix + f".{uuid4().hex}.tmp")
        try:
            panel.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, out_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    print(f"Saved {out_path} ({len(panel):,} rows)")
    return out_path
