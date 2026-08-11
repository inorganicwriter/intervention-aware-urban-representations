"""Discovery and metadata helpers for extracted Amap FileGDB assets."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import INTERIM_DIR
from .sources import category_from_gdb_path
from .taxonomy import map_poi_category

GDB_ARCHIVE_DIR = INTERIM_DIR / "gdb_archives"
_GDB_SRC_CACHE: dict[str, list[ExtractedGdbSource]] = {}

TYPECODE_MAJOR_PREFIX = {
    "01": "汽车服务",
    "02": "汽车销售",
    "03": "汽车维修",
    "04": "摩托车服务",
    "05": "餐饮服务",
    "06": "购物服务",
    "07": "生活服务",
    "08": "体育休闲服务",
    "09": "医疗保健服务",
    "10": "住宿服务",
    "11": "风景名胜",
    "12": "商务住宅",
    "13": "政府机构及社会团体",
    "14": "科教文化服务",
    "15": "交通设施服务",
    "16": "金融保险服务",
    "17": "公司企业",
    "18": "道路附属设施",
    "19": "地名地址信息",
    "20": "公共设施",
    "97": "室内设施",
    "99": "通行设施",
}


@dataclass(frozen=True)
class ExtractedGdbSource:
    path: Path
    year: int
    category: str | None
    is_nested: bool
    layer: str | None = None


def is_nested_gdb(path: Path) -> bool:
    return any(parent.suffix == ".gdb" for parent in path.parents)


def is_valid_filegdb(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.gdbtable")) and any(path.glob("*.gdbtablx"))


def infer_year_from_path(path: Path) -> int | None:
    for part in path.parts:
        if re.fullmatch(r"20(?:1[8-9]|2[0-4])", part):
            return int(part)
    years = re.findall(r"20(?:1[8-9]|2[0-4])", path.name)
    return int(years[0]) if years else None


def normalize_gdb_category(value: str | None) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"20(?:1[8-9]|2[0-4])", "", value)
    cleaned = cleaned.replace("全国", "").strip()
    aliases = {
        "体育休闲": "体育休闲服务",
        "科教文化": "科教文化服务",
        "政府机构与社会团体": "政府机构及社会团体",
    }
    return aliases.get(cleaned, cleaned)


def sources_from_2020_layers(path: Path, layers: list[str]) -> list[ExtractedGdbSource]:
    sources = []
    for layer in layers:
        sources.append(
            ExtractedGdbSource(
                path=path,
                year=2020,
                category=normalize_gdb_category(layer),
                is_nested=is_nested_gdb(path),
                layer=layer,
            )
        )
    return sources


def list_filegdb_layers(path: Path) -> list[str]:
    import geopandas as gpd

    layers = gpd.list_layers(path)
    if "name" not in layers.columns:
        return []
    return layers["name"].astype(str).tolist()


def discover_extracted_gdb_sources(
    base_dir: Path = GDB_ARCHIVE_DIR,
    year: int | None = None,
    categories: set[str] | None = None,
) -> list[ExtractedGdbSource]:
    cache_key = f"{base_dir}|{year}|{sorted(categories) if categories else 'all'}"
    if cache_key in _GDB_SRC_CACHE:
        return _GDB_SRC_CACHE[cache_key]

    if not base_dir.exists():
        _GDB_SRC_CACHE[cache_key] = []
        return []

    valid_gdbs = [path for path in base_dir.rglob("*.gdb") if is_valid_filegdb(path)]
    nested_valid = {path for path in valid_gdbs if is_nested_gdb(path)}
    container_gdbs = {nested.parent for nested in nested_valid}

    sources: list[ExtractedGdbSource] = []
    for path in valid_gdbs:
        if path in container_gdbs:
            continue
        source_year = infer_year_from_path(path)
        if year is not None and source_year != year:
            continue
        if source_year is None:
            warnings.warn(
                f"Skipping FileGDB whose year cannot be inferred: {path}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if source_year == 2020:
            layer_sources = sources_from_2020_layers(path, list_filegdb_layers(path))
            if categories:
                layer_sources = [
                    source
                    for source in layer_sources
                    if source.category in categories
                    or map_poi_category(source.category) in categories
                ]
            sources.extend(layer_sources)
            continue
        category = category_from_gdb_path(str(path))
        category = normalize_gdb_category(category) if category else category
        if (
            categories
            and category is not None
            and (category not in categories and map_poi_category(category) not in categories)
        ):
            continue
        sources.append(
            ExtractedGdbSource(
                path=path,
                year=source_year,
                category=category,
                is_nested=is_nested_gdb(path),
            )
        )
    result = sorted(sources, key=lambda source: str(source.path))
    _GDB_SRC_CACHE[cache_key] = result
    return result


def inspect_filegdb(path: Path, layer: str | None = None) -> dict:
    import geopandas as gpd

    try:
        layers = gpd.list_layers(path)
        layer_names = (
            ",".join(layers["name"].astype(str).tolist()) if "name" in layers.columns else ""
        )
        read_kwargs: dict[str, object] = {"rows": 1}
        if layer is not None:
            read_kwargs["layer"] = layer
        sample = gpd.read_file(path, **read_kwargs)
        crs = sample.crs.to_string() if sample.crs is not None else ""
        columns = ",".join(str(col) for col in sample.columns)
        row_readable = True
        error = ""
    except Exception as exc:
        layer_names = ""
        crs = ""
        columns = ""
        row_readable = False
        error = str(exc)
    return {
        "layers": layer_names,
        "crs": crs,
        "columns": columns,
        "row_readable": row_readable,
        "error": error,
    }


def build_extracted_gdb_inventory(
    base_dir: Path = GDB_ARCHIVE_DIR, inspect: bool = False
) -> pd.DataFrame:
    rows = []
    for source in discover_extracted_gdb_sources(base_dir=base_dir):
        row = {
            "year": source.year,
            "category": source.category or "",
            "layer": source.layer or "",
            "path": str(source.path),
            "relative_path": str(source.path.relative_to(base_dir)),
            "is_nested": source.is_nested,
            "is_valid_filegdb": is_valid_filegdb(source.path),
            "size_bytes": sum(
                file.stat().st_size for file in source.path.rglob("*") if file.is_file()
            ),
        }
        if inspect:
            row.update(inspect_filegdb(source.path, layer=source.layer))
        rows.append(row)
    return pd.DataFrame(rows)


def category_from_2020_fields(row: pd.Series) -> str:
    for col in ["typename", "type", "tag"]:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            first = re.split(r"[，,;；/|>]", value.strip(), maxsplit=1)[0]
            if first:
                return first
    typecode = str(row.get("typecode", "") or "").strip()
    return TYPECODE_MAJOR_PREFIX.get(typecode[:2], "其他")
