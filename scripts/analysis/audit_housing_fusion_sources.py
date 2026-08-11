"""Audit raw housing sources and AOI assets before canonical fusion.

This command is read-only with respect to source data.  It writes a compact
machine-readable report and a human-readable CSV inventory used by the housing
fusion contract.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.config.project import ACTIVE_CITIES, CITIES  # noqa: E402
from urban_intervention.data.paths import (  # noqa: E402
    OUTPUT_HOUSING_FUSION_DIR,
    RAW_HOUSING_DIR,
)

RAW = RAW_HOUSING_DIR
OUT = OUTPUT_HOUSING_FUSION_DIR


def _file_row_count(path: Path, encoding: str = "utf-8") -> int:
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _field_map(columns: list[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in columns:
        name = str(column)
        if "小区" in name and "边界" not in name:
            result.setdefault("community", name)
        if "城市" in name:
            result.setdefault("city", name)
        if "经" in name:
            result.setdefault("lon", name)
        if "纬" in name:
            result.setdefault("lat", name)
        if "成交时间" in name or "成交日期" in name:
            result.setdefault("deal_date", name)
        if "成交年份" in name:
            result.setdefault("year", name)
        if "单价" in name:
            result.setdefault("unit_price", name)
        elif "成交价" in name:
            result.setdefault("price", name)
    return result


def audit_lianjia() -> list[dict]:
    rows: list[dict] = []
    # Several provincial workbooks are stored one directory below the root.
    # Resolve duplicate root/nested copies by file name and size so the audit
    # describes distinct source workbooks rather than archive packaging.
    candidates = sorted(
        (RAW / "platform_exports" / "lianjia" / "purchased_transactions").rglob("*.xlsx")
    )
    distinct: dict[tuple[str, int], Path] = {}
    for candidate in candidates:
        distinct.setdefault((candidate.name, candidate.stat().st_size), candidate)
    for path in sorted(distinct.values()):
        item = {
            "dataset": "lianjia_purchased",
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
        }
        try:
            sample = pd.read_excel(path, nrows=100)
            fields = _field_map(list(sample.columns))
            item.update(
                {
                    "status": "readable",
                    "columns": json.dumps(
                        [str(value) for value in sample.columns], ensure_ascii=False
                    ),
                    "field_map": json.dumps(fields, ensure_ascii=False),
                    "has_required_fields": (
                        all(key in fields for key in ("community", "lon", "lat"))
                        and ("unit_price" in fields or "price" in fields)
                        and ("year" in fields or "deal_date" in fields)
                    ),
                }
            )
            year_column = fields.get("year") or fields.get("deal_date")
            if year_column:
                values = sample[year_column].dropna().astype(str).head(20).tolist()
                item["sample_time_values"] = json.dumps(values, ensure_ascii=False)
        except Exception as exc:  # audit must retain failures
            item.update({"status": "unreadable", "error": repr(exc), "has_required_fields": False})
        rows.append(item)
    return rows


def audit_anjuke() -> list[dict]:
    base = RAW / "platform_exports" / "anjuke" / "cross_section"
    rows: list[dict] = []
    for city_key in ACTIVE_CITIES:
        city_name = CITIES[city_key]["name"]
        boundary_candidates = sorted(base.glob(f"{city_name}*_community_ext.csv"))
        house_candidates = sorted(base.glob(f"{city_name}*_house.csv"))
        item = {
            "dataset": "anjuke_cross_section",
            "city_key": city_key,
            "boundary_path": str(boundary_candidates[0].relative_to(ROOT))
            if boundary_candidates
            else "",
            "house_path": str(house_candidates[0].relative_to(ROOT)) if house_candidates else "",
            "boundary_file_present": bool(boundary_candidates),
            "house_file_present": bool(house_candidates),
        }
        if boundary_candidates:
            path = boundary_candidates[0]
            try:
                frame = pd.read_csv(path)
                columns = list(frame.columns)
                boundary_columns = [col for col in columns if "边界" in str(col)]
                name_columns = [col for col in columns if "名称" in str(col)]
                item.update(
                    {
                        "boundary_rows": len(frame),
                        "boundary_columns": json.dumps(
                            [str(value) for value in columns], ensure_ascii=False
                        ),
                        "boundary_nonnull": int(frame[boundary_columns[0]].notna().sum())
                        if boundary_columns
                        else 0,
                        "boundary_schema_ok": bool(boundary_columns and name_columns),
                    }
                )
            except Exception as exc:
                item.update({"boundary_error": repr(exc), "boundary_schema_ok": False})
        if house_candidates:
            path = house_candidates[0]
            try:
                frame = pd.read_csv(path)
                item.update(
                    {
                        "house_rows": len(frame),
                        "house_columns": json.dumps(
                            [str(value) for value in frame.columns], ensure_ascii=False
                        ),
                    }
                )
            except Exception as exc:
                item["house_error"] = repr(exc)
        rows.append(item)
    return rows


def audit_wayback() -> list[dict]:
    rows: list[dict] = []
    base = RAW / "web_archives" / "wayback" / "parsed_pages"
    for path in sorted(base.glob("*_wayback_*.csv")):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
            year_columns = [
                value for value in ("deal_year", "snapshot_year") if value in frame.columns
            ]
            years: set[int] = set()
            for column in year_columns:
                years.update(
                    pd.to_numeric(frame[column], errors="coerce").dropna().astype(int).tolist()
                )
            rows.append(
                {
                    "dataset": "wayback",
                    "path": str(path.relative_to(ROOT)),
                    "rows": len(frame),
                    "columns": json.dumps(list(frame.columns), ensure_ascii=False),
                    "year_columns": ";".join(year_columns),
                    "min_year": min(years) if years else None,
                    "max_year": max(years) if years else None,
                    "has_community": "community" in frame.columns,
                    "has_unit_price": "unit_price" in frame.columns,
                }
            )
        except Exception as exc:
            rows.append(
                {"dataset": "wayback", "path": str(path.relative_to(ROOT)), "error": repr(exc)}
            )
    return rows


def audit_grid2023() -> list[dict]:
    rows: list[dict] = []
    for path in sorted((RAW / "spatial_support" / "grid_price_2023_05").glob("*/表格/*.csv")):
        try:
            frame = pd.read_csv(path)
            rows.append(
                {
                    "dataset": "grid_2023",
                    "path": str(path.relative_to(ROOT)),
                    "rows": len(frame),
                    "columns": json.dumps(
                        [str(value) for value in frame.columns], ensure_ascii=False
                    ),
                    "year": 2023,
                }
            )
        except Exception as exc:
            rows.append(
                {"dataset": "grid_2023", "path": str(path.relative_to(ROOT)), "error": repr(exc)}
            )
    return rows


def audit_beijing_aoi() -> dict:
    path = RAW / "spatial_support" / "community_aoi" / "baidu_beijing" / "房地产.shp"
    result = {
        "dataset": "beijing_independent_aoi",
        "path": str(path.relative_to(ROOT)),
        "present": path.exists(),
    }
    if not path.exists():
        return result
    try:
        import geopandas as gpd

        frame = gpd.read_file(path)
        result.update(
            {
                "status": "readable",
                "rows": len(frame),
                "crs": str(frame.crs),
                "columns": list(frame.columns),
                "geometry_types": frame.geometry.geom_type.value_counts().to_dict(),
                "valid_geometries": int(frame.geometry.is_valid.sum()),
                "empty_geometries": int(frame.geometry.is_empty.sum()),
            }
        )
    except Exception as exc:
        result.update({"status": "unreadable", "error": repr(exc)})
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = audit_lianjia() + audit_anjuke() + audit_wayback() + audit_grid2023()
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(OUT / "source_inventory.csv", index=False, encoding="utf-8-sig")

    beijing = audit_beijing_aoi()
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "raw_sources_modified": False,
        "lianjia_files": int((inventory_frame["dataset"] == "lianjia_purchased").sum()),
        "anjuke_research_cities": int((inventory_frame["dataset"] == "anjuke_cross_section").sum()),
        "wayback_files": int((inventory_frame["dataset"] == "wayback").sum()),
        "grid2023_files": int((inventory_frame["dataset"] == "grid_2023").sum()),
        "beijing_independent_aoi": beijing,
    }
    (OUT / "source_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved {OUT / 'source_inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
