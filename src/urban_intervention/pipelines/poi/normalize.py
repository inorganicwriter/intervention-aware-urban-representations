"""Schema and CRS normalization for POI sources."""

from pathlib import Path

import geopandas as gpd  # noqa: F401
import pandas as pd

from .config import NORMALIZED_COLUMNS
from .gdb import category_from_2020_fields
from .sources import category_from_gdb_path
from .taxonomy import add_feature_flags

_GDB_KEEP_COLS = {"name", "typecode", "typename", "tag", "type", "大类", "中类", "小类", "geometry"}
# 2020 nationwide GDBs vary in which classification field they carry
# ("typename" / "type" / "tag"); read whichever exist so
# category_from_2020_fields can use all of them.
_GDB_KEEP_COLS_2020 = {
    "name",
    "typecode",
    "typename",
    "type",
    "tag",
    "大类",
    "中类",
    "小类",
    "geometry",
    "wgs_x",
    "wgs_y",
}
_GDB_READ_COLS = [
    "name",
    "typecode",
    "typename",
    "tag",
    "type",
    "大类",
    "中类",
    "小类",
    "wgs_x",
    "wgs_y",
]
_GDB_READ_COLS_2020 = [
    "name",
    "typecode",
    "typename",
    "type",
    "tag",
    "大类",
    "中类",
    "小类",
    "wgs_x",
    "wgs_y",
]


def _list_gdb_fields(path: Path, layer: str | None = None) -> set[str]:
    """List field names in a GDB without reading data."""
    try:
        import pyogrio

        info = pyogrio.read_info(str(path), layer=layer) if layer else pyogrio.read_info(str(path))
        return set(info.get("fields", []))
    except Exception:
        return set()


def _columns_to_read(path: Path, year: int, layer: str | None = None) -> list[str] | None:
    """Return the list of columns to request from the GDB, or None to read all."""
    candidates = _GDB_READ_COLS_2020 if year == 2020 else _GDB_READ_COLS
    available = _list_gdb_fields(path, layer=layer)
    if not available:
        return None
    cols = [c for c in candidates if c in available]
    return cols if cols else None


def normalize_csv_poi_chunk(chunk: pd.DataFrame, year: int, city_key: str) -> pd.DataFrame:
    needed = {"name", "lon", "lat", "typecode", "cate_A", "cate_B", "cate_C"}
    missing = needed - set(chunk.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    out = chunk[list(needed)].copy()
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out = out.dropna(subset=["lon", "lat"])
    out.insert(0, "year", year)
    out.insert(0, "city", city_key)
    out = add_feature_flags(out)
    return out[NORMALIZED_COLUMNS]


def normalize_crs_to_wgs84(gdf):
    """Return (GeoDataFrame in EPSG:4326, conversion_method)."""
    if gdf.crs is None:
        if gdf.empty:
            return gdf.set_crs("EPSG:4326"), "assume_wgs84:empty_missing_crs"
        bounds = gdf.total_bounds
        looks_lonlat = (
            -180 <= bounds[0] <= 180
            and -180 <= bounds[2] <= 180
            and -90 <= bounds[1] <= 90
            and -90 <= bounds[3] <= 90
        )
        if looks_lonlat:
            return gdf.set_crs("EPSG:4326"), "assume_wgs84:missing_crs"
        raise ValueError(f"Missing CRS and coordinates do not look like lon/lat: bounds={bounds}")
    epsg = gdf.crs.to_epsg()
    if epsg == 4326:
        return gdf, "already_wgs84"
    before = gdf.crs.to_string()
    if len(gdf) == 1:
        # pyproj 3.7 treats one-element NumPy coordinate arrays as scalars and
        # emits a NumPy deprecation warning. Transforming the lone geometry
        # through scalar coordinates avoids that upstream edge case without
        # slowing normal FileGDB batches.
        from pyproj import Transformer
        from shapely.ops import transform

        transformer = Transformer.from_crs(gdf.crs, "EPSG:4326", always_xy=True)
        out = gdf.copy()
        out.geometry = out.geometry.map(
            lambda geometry: (
                transform(transformer.transform, geometry) if geometry is not None else None
            )
        )
        out = out.set_crs("EPSG:4326", allow_override=True)
    else:
        out = gdf.to_crs("EPSG:4326")
    return out, f"to_crs:{before}->EPSG:4326"


def category_series_from_gdb(
    df: pd.DataFrame,
    fallback_path: str | Path,
    category_override: str | None = None,
    infer_from_fields: bool = False,
) -> pd.Series:
    if category_override:
        return pd.Series(category_override, index=df.index)
    if infer_from_fields:
        return df.apply(category_from_2020_fields, axis=1)
    return pd.Series(category_from_gdb_path(str(fallback_path)), index=df.index)


def _trim_gdf_columns(gdf, year: int):
    keep = _GDB_KEEP_COLS_2020 if year == 2020 else _GDB_KEEP_COLS
    cols = [c for c in keep if c in gdf.columns]
    if "geometry" not in cols:
        cols.append("geometry")
    return gdf[cols]


def read_filegdb(
    path: Path,
    year: int,
    city_key: str,
    max_rows: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    category_override: str | None = None,
    infer_category_from_fields: bool = False,
    layer: str | None = None,
) -> tuple["gpd.GeoDataFrame", str]:
    import geopandas as gpd

    read_kwargs: dict[str, object] = {}
    if max_rows is not None:
        read_kwargs["rows"] = max_rows
    if bbox is not None:
        read_kwargs["bbox"] = bbox
    if layer is not None:
        read_kwargs["layer"] = layer
    read_cols = _columns_to_read(path, year, layer=layer)
    if read_cols:
        read_kwargs["columns"] = read_cols
    gdf = gpd.read_file(path, **read_kwargs)
    gdf, crs_method = normalize_crs_to_wgs84(gdf)

    gdf = _trim_gdf_columns(gdf, year)

    cate_a = category_series_from_gdb(
        gdf,
        fallback_path=path,
        category_override=category_override,
        infer_from_fields=infer_category_from_fields,
    )

    gdf["city"] = city_key
    gdf["year"] = year
    gdf["cate_A"] = cate_a
    gdf["cate_B"] = gdf["typename"] if "typename" in gdf.columns else ""
    gdf["cate_C"] = gdf["tag"] if "tag" in gdf.columns else ""
    gdf["typecode"] = gdf["typecode"] if "typecode" in gdf.columns else ""

    gdf = add_feature_flags(gdf)

    keep = [c for c in NORMALIZED_COLUMNS if c in gdf.columns]
    if "geometry" not in keep:
        keep.append("geometry")
    return gdf[keep], crs_method
