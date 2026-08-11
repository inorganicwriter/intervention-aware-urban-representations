"""Create a compact, reproducible inventory and distribution audit for project data.

The audit deliberately uses Parquet metadata and existing quality audit outputs where
possible, so it is safe to run on the full project without loading multi-GB geometry
or raster-derived tables into memory.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from urban_intervention.data.paths import (  # noqa: E402
    DATA_ROOT,
    HPI_LABEL_DIR,
    OUTPUT_DATA_QUALITY_DIR,
    OUTPUT_GEE_QUALITY_DIR,
    OUTPUT_POI_QUALITY_DIR,
    OUTPUT_TRANSIT_COMPARISON_DIR,
    PANEL_DIR,
    POI_DIR,
    POPULATION_DIR,
    RAW_LIANJIA_DIR,
    RAW_WAYBACK_PARSED_DIR,
    REFERENCE_GRID_DIR,
    SENTINEL2_DIR,
    TREATMENT_DIR,
    VIIRS_ANNUAL_DIR,
)

DATA = DATA_ROOT
OUT = OUTPUT_DATA_QUALITY_DIR


def parquet_meta(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    return {
        "file": str(path.relative_to(ROOT)),
        "rows": pf.metadata.num_rows,
        "columns": pf.schema_arrow.names,
        "bytes": path.stat().st_size,
    }


def parse_xlsx(path: Path) -> dict:
    """Read the lightweight Excel worksheet dimension, falling back cleanly."""
    item = {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
    try:
        with zipfile.ZipFile(path) as zf:
            sheets = [n for n in zf.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)]
            dimensions = []
            for sheet in sheets:
                head = zf.read(sheet)[:2048].decode("utf-8", "ignore")
                match = re.search(r'<dimension ref="(?:[A-Z]+\d+:)?[A-Z]+(\d+)"', head)
                dimensions.append(int(match.group(1)) if match else None)
            item.update({"sheets": len(sheets), "sheet_row_dimensions": dimensions})
    except (OSError, zipfile.BadZipFile) as exc:
        item["error"] = str(exc)
    return item


def housing_file_kind(path: Path) -> tuple[str, str]:
    name = path.name
    if name.endswith("_wayback_anjuke.csv"):
        return "anjuke", "community"
    if "_wayback_beike_chengjiao.csv" in name:
        return "beike", "chengjiao"
    if "_wayback_beike_xiaoqu.csv" in name:
        return "beike", "xiaoqu"
    if name.endswith("_wayback_chengjiao.csv"):
        return "lianjia", "chengjiao"
    if name.endswith("_wayback_xiaoqu.csv"):
        return "lianjia", "xiaoqu"
    return "other", "other"


def housing_audit() -> tuple[list[dict], list[dict]]:
    rows = []
    by_city = defaultdict(lambda: Counter())
    for path in sorted(RAW_WAYBACK_PARSED_DIR.glob("*.csv")):
        platform, page_type = housing_file_kind(path)
        n = 0
        years = Counter()
        priced = 0
        deal_dated = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                n += 1
                year = record.get("snapshot_year", "")
                if year:
                    years[year] += 1
                if record.get("unit_price", "") not in ("", None):
                    priced += 1
                if record.get("deal_date", "") not in ("", None):
                    deal_dated += 1
                city = record.get("city_key") or path.name.split("_wayback_")[0]
                by_city[city]["rows"] += 1
                by_city[city][f"{platform}_rows"] += 1
        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "city": path.name.split("_wayback_")[0],
                "platform": platform,
                "page_type": page_type,
                "rows": n,
                "priced_rows": priced,
                "deal_dated_rows": deal_dated,
                "min_snapshot_year": min(years, default=""),
                "max_snapshot_year": max(years, default=""),
                "snapshot_year_counts": dict(sorted(years.items())),
            }
        )
    city_rows = [dict(city=city, **counts) for city, counts in sorted(by_city.items())]
    return rows, city_rows


def summarize_file_group(paths: list[Path]) -> dict:
    metas = [parquet_meta(path) for path in paths]
    return {
        "files": len(metas),
        "rows": sum(x["rows"] for x in metas),
        "bytes": sum(x["bytes"] for x in metas),
        "common_columns": sorted(set.intersection(*(set(x["columns"]) for x in metas)))
        if metas
        else [],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grids = sorted(REFERENCE_GRID_DIR.glob("*/*_grids.parquet"))
    treatment = sorted(TREATMENT_DIR.glob("*/*_grid_treatment.parquet"))
    viirs = sorted(VIIRS_ANNUAL_DIR.glob("*_viirs_annual.parquet"))
    sentinel = sorted(SENTINEL2_DIR.glob("*_s2.parquet"))
    population = sorted(POPULATION_DIR.glob("*_pop.parquet"))
    poi = sorted(POI_DIR.glob("*_poi_grid_yearly.parquet"))
    panels = sorted(PANEL_DIR.glob("*.parquet"))

    gee_path = OUTPUT_GEE_QUALITY_DIR / "gee_processed_audit.csv"
    poi_path = OUTPUT_POI_QUALITY_DIR / "poi_coverage_by_city_year.csv"
    transit_path = OUTPUT_TRANSIT_COMPARISON_DIR / "per_city_summary.csv"
    gee = pd.read_csv(gee_path) if gee_path.exists() else pd.DataFrame()
    poi_cov = pd.read_csv(poi_path) if poi_path.exists() else pd.DataFrame()
    transit = pd.read_csv(transit_path) if transit_path.exists() else pd.DataFrame()

    hpi_path = HPI_LABEL_DIR / "hpi_city_yearly.parquet"
    hpi = (
        pd.read_parquet(hpi_path, columns=["city_key", "year", "housing_type", "area_class"])
        if hpi_path.exists()
        else pd.DataFrame()
    )
    housing_files, housing_cities = housing_audit()
    xlsx = [parse_xlsx(p) for p in sorted(RAW_LIANJIA_DIR.rglob("*.xlsx"))]

    folder_sizes = {}
    for child in sorted(DATA.iterdir()):
        if child.is_dir():
            folder_sizes[child.name] = sum(
                p.stat().st_size for p in child.rglob("*") if p.is_file()
            )

    report = {
        "project_data_bytes": sum(folder_sizes.values()),
        "folder_bytes": folder_sizes,
        "parquet_groups": {
            "grids": summarize_file_group(grids),
            "treatment": summarize_file_group(treatment),
            "viirs": summarize_file_group(viirs),
            "sentinel2": summarize_file_group(sentinel),
            "population": summarize_file_group(population),
            "poi_grid_year": summarize_file_group(poi),
            "analysis_panels": summarize_file_group(panels),
        },
        "gee_quality": {
            "rows": int(len(gee)),
            "sources": gee.groupby("source")
            .agg(
                cities=("city", "nunique"),
                rows=("rows", "sum"),
                duplicate_grid_year_rows=("duplicate_grid_year_rows", "sum"),
                min_year=("min_year", "min"),
                max_year=("max_year", "max"),
            )
            .reset_index()
            .to_dict("records")
            if not gee.empty
            else [],
        },
        "poi_quality": {
            "city_year_rows": int(len(poi_cov)),
            "cities": int(poi_cov["city"].nunique()) if not poi_cov.empty else 0,
            "years": sorted(map(int, poi_cov["year"].unique())) if not poi_cov.empty else [],
            "total_pois": int(poi_cov["poi_count_total"].sum()) if not poi_cov.empty else 0,
            "year_totals": poi_cov.groupby("year")
            .agg(
                cities=("city", "nunique"),
                poi_total=("poi_count_total", "sum"),
                median_grid_poi=("median_grid_poi_count", "median"),
            )
            .reset_index()
            .to_dict("records")
            if not poi_cov.empty
            else [],
        },
        "hpi": {
            "rows": int(len(hpi)),
            "cities": int(hpi["city_key"].nunique()) if not hpi.empty else 0,
            "years": sorted(map(int, hpi["year"].unique())) if not hpi.empty else [],
            "strata": hpi[["housing_type", "area_class"]]
            .drop_duplicates()
            .sort_values(["housing_type", "area_class"])
            .to_dict("records")
            if not hpi.empty
            else [],
        },
        "transit": {
            "rows": int(len(transit)),
            "cities": int(transit["city"].nunique()) if not transit.empty else 0,
            "sources": transit.groupby("source")
            .agg(
                city_rows=("city", "nunique"),
                stations=("count", "sum"),
                stations_with_year=("n_year", "sum"),
                stations_with_line=("n_line", "sum"),
            )
            .reset_index()
            .to_dict("records")
            if not transit.empty
            else [],
        },
        "wayback_housing_files": housing_files,
        "wayback_housing_by_city": housing_cities,
        "xlsx_housing_archives": xlsx,
    }
    (OUT / "data_distribution_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pd.DataFrame(housing_files).drop(columns=["snapshot_year_counts"]).to_csv(
        OUT / "wayback_housing_file_distribution.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(housing_cities).fillna(0).to_csv(
        OUT / "wayback_housing_city_distribution.csv", index=False, encoding="utf-8-sig"
    )
    lines = [
        "# 项目数据分布审计",
        "",
        f"- 数据目录总大小：{report['project_data_bytes'] / 1024**3:.2f} GiB",
        f"- 网格：{report['parquet_groups']['grids']['files']} 城市文件，{report['parquet_groups']['grids']['rows']:,} 行",
        f"- 地铁 treatment：{report['parquet_groups']['treatment']['files']} 文件，{report['parquet_groups']['treatment']['rows']:,} 行",
        f"- POI：{report['poi_quality']['cities']} 城 × {len(report['poi_quality']['years'])} 年 = {report['poi_quality']['city_year_rows']} 城市年，POI 总量 {report['poi_quality']['total_pois']:,}",
        f"- HPI：{report['hpi']['cities']} 城，年份 {report['hpi']['years']}",
        f"- Wayback 住房：{sum(x['rows'] for x in housing_files):,} 网页行，{len(housing_cities)} 城市有有效解析行",
        "",
        "## 使用注意",
        "",
        "- Wayback 行是快照页面中解析出的列表行，跨快照会重复；建模前须按业务主键和日期去重，不能直接作为独立成交样本。",
        "- GEE 质量表中 `duplicate_grid_year_rows` 非零，合并网格年面板前须先按 `grid_id, year` 聚合或保留唯一观测。",
        "- `wayback_housing_*distribution.csv` 为文件级和城市级覆盖明细；JSON 保留完整机器可读统计。",
    ]
    (OUT / "data_distribution_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
