"""
Pipeline Configuration
Intervention-aware Urban Representation Learning

Multi-city support: 44 Chinese metro cities.
"""

import json
import math
import os
import socket
import warnings
from pathlib import Path
from typing import TypedDict

from urban_intervention.data.paths import (
    BOUNDARY_DIR,
    DATA_ROOT,
    HPI_LABEL_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    REFERENCE_GRID_DIR,
    STAGING_DIR,
    TREATMENT_DIR,
)
from urban_intervention.interventions.transit.station_names import (
    normalize_station_name as _normalize_station_name,
)

BASE_DIR = PROJECT_ROOT
DATA_DIR = DATA_ROOT
GRID_DIR = REFERENCE_GRID_DIR


# ── City registry (44 metro cities, auto-generated) ──


class CityConfig(TypedDict, total=False):
    name: str
    country: str
    city_id: str
    bbox: list[float]
    crs: str
    projected_crs: str
    center_lon: float
    center_lat: float


CITIES: dict[str, CityConfig] = {
    "beijing": {
        "name": "北京",
        "country": "China",
        "city_id": "北京_cn",
        "bbox": [115.8, 39.3, 117.0, 40.5],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 116.4,
        "center_lat": 39.9,
    },
    "changchun": {
        "name": "长春",
        "country": "China",
        "city_id": "长春_cn",
        "bbox": [124.72, 43.22, 125.92, 44.42],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 125.32,
        "center_lat": 43.82,
    },
    "changsha": {
        "name": "长沙",
        "country": "China",
        "city_id": "长沙_cn",
        "bbox": [112.37, 27.63, 113.57, 28.83],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32649",
        "center_lon": 112.97,
        "center_lat": 28.23,
    },
    "changzhou": {
        "name": "常州",
        "country": "China",
        "city_id": "常州_cn",
        "bbox": [119.37, 31.17, 120.57, 32.37],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 119.97,
        "center_lat": 31.77,
    },
    "chengdu": {
        "name": "成都",
        "country": "China",
        "city_id": "成都_cn",
        "bbox": [103.47, 29.97, 104.67, 31.17],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32648",
        "center_lon": 104.07,
        "center_lat": 30.57,
    },
    "chongqing": {
        "name": "重庆",
        "country": "China",
        "city_id": "重庆_cn",
        "bbox": [105.95, 28.96, 107.15, 30.16],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32648",
        "center_lon": 106.55,
        "center_lat": 29.56,
    },
    "dalian": {
        "name": "大连",
        "country": "China",
        "city_id": "大连_cn",
        "bbox": [121.02, 38.31, 122.22, 39.51],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 121.62,
        "center_lat": 38.91,
    },
    "dongguan": {
        "name": "东莞",
        "country": "China",
        "city_id": "东莞_cn",
        "bbox": [113.15, 22.42, 114.35, 23.62],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32649",
        "center_lon": 113.75,
        "center_lat": 23.02,
    },
    "foshan": {
        "name": "佛山",
        "country": "China",
        "city_id": "佛山_cn",
        "bbox": [112.52, 22.42, 113.72, 23.62],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32649",
        "center_lon": 113.12,
        "center_lat": 23.02,
    },
    "fuzhou": {
        "name": "福州",
        "country": "China",
        "city_id": "福州_cn",
        "bbox": [118.7, 25.47, 119.9, 26.67],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 119.3,
        "center_lat": 26.07,
    },
    "guangzhou": {
        "name": "广州",
        "country": "China",
        "city_id": "广州_cn",
        "bbox": [112.66, 22.53, 113.86, 23.73],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32649",
        "center_lon": 113.26,
        "center_lat": 23.13,
    },
    "guiyang": {
        "name": "贵阳",
        "country": "China",
        "city_id": "贵阳_cn",
        "bbox": [106.11, 26.05, 107.31, 27.25],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32648",
        "center_lon": 106.71,
        "center_lat": 26.65,
    },
    "hangzhou": {
        "name": "杭州",
        "country": "China",
        "city_id": "杭州_cn",
        "bbox": [119.61, 29.67, 120.81, 30.87],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 120.21,
        "center_lat": 30.27,
    },
    "harbin": {
        "name": "哈尔滨",
        "country": "China",
        "city_id": "哈尔滨_cn",
        "bbox": [126.03, 45.15, 127.23, 46.35],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32652",
        "center_lon": 126.63,
        "center_lat": 45.75,
    },
    "hefei": {
        "name": "合肥",
        "country": "China",
        "city_id": "合肥_cn",
        "bbox": [116.63, 31.22, 117.83, 32.42],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 117.23,
        "center_lat": 31.82,
    },
    "hohhot": {
        "name": "呼和浩特",
        "country": "China",
        "city_id": "呼和浩特_cn",
        "bbox": [111.07, 40.22, 112.27, 41.42],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32649",
        "center_lon": 111.67,
        "center_lat": 40.82,
    },
    "jinan": {
        "name": "济南",
        "country": "China",
        "city_id": "济南_cn",
        "bbox": [116.4, 36.05, 117.6, 37.25],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 117.0,
        "center_lat": 36.65,
    },
    "jinhua": {
        "name": "金华",
        "country": "China",
        "city_id": "金华_cn",
        "bbox": [119.05, 28.48, 120.25, 29.68],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 119.65,
        "center_lat": 29.08,
    },
    "kunming": {
        "name": "昆明",
        "country": "China",
        "city_id": "昆明_cn",
        "bbox": [102.13, 24.44, 103.33, 25.64],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32648",
        "center_lon": 102.73,
        "center_lat": 25.04,
    },
    "lanzhou": {
        "name": "兰州",
        "country": "China",
        "city_id": "兰州_cn",
        "bbox": [103.13, 35.43, 104.33, 36.63],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32648",
        "center_lon": 103.73,
        "center_lat": 36.03,
    },
    "luoyang": {
        "name": "洛阳",
        "country": "China",
        "city_id": "洛阳_cn",
        "bbox": [111.85, 34.02, 113.05, 35.22],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32649",
        "center_lon": 112.45,
        "center_lat": 34.62,
    },
    "nanchang": {
        "name": "南昌",
        "country": "China",
        "city_id": "南昌_cn",
        "bbox": [115.26, 28.08, 116.46, 29.28],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 115.86,
        "center_lat": 28.68,
    },
    "nanjing": {
        "name": "南京",
        "country": "China",
        "city_id": "南京_cn",
        "bbox": [118.2, 31.46, 119.4, 32.66],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 118.8,
        "center_lat": 32.06,
    },
    "nanning": {
        "name": "南宁",
        "country": "China",
        "city_id": "南宁_cn",
        "bbox": [107.72, 22.22, 108.92, 23.42],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32649",
        "center_lon": 108.32,
        "center_lat": 22.82,
    },
    "nantong": {
        "name": "南通",
        "country": "China",
        "city_id": "南通_cn",
        "bbox": [120.3, 31.38, 121.5, 32.58],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 120.9,
        "center_lat": 31.98,
    },
    "ningbo": {
        "name": "宁波",
        "country": "China",
        "city_id": "宁波_cn",
        "bbox": [120.94, 29.27, 122.14, 30.47],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 121.54,
        "center_lat": 29.87,
    },
    "qingdao": {
        "name": "青岛",
        "country": "China",
        "city_id": "青岛_cn",
        "bbox": [119.78, 35.47, 120.98, 36.67],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 120.38,
        "center_lat": 36.07,
    },
    "shanghai": {
        "name": "上海",
        "country": "China",
        "city_id": "上海_cn",
        "bbox": [120.87, 30.63, 122.07, 31.83],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 121.47,
        "center_lat": 31.23,
    },
    "shaoxing": {
        "name": "绍兴",
        "country": "China",
        "city_id": "绍兴_cn",
        "bbox": [119.98, 29.43, 121.18, 30.63],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 120.58,
        "center_lat": 30.03,
    },
    "shenyang": {
        "name": "沈阳",
        "country": "China",
        "city_id": "沈阳_cn",
        "bbox": [122.83, 41.2, 124.03, 42.4],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 123.43,
        "center_lat": 41.8,
    },
    "shenzhen": {
        "name": "深圳",
        "country": "China",
        "city_id": "深圳_cn",
        "bbox": [113.46, 21.94, 114.66, 23.14],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 114.06,
        "center_lat": 22.54,
    },
    "shijiazhuang": {
        "name": "石家庄",
        "country": "China",
        "city_id": "石家庄_cn",
        "bbox": [113.91, 37.44, 115.11, 38.64],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 114.51,
        "center_lat": 38.04,
    },
    "suzhou": {
        "name": "苏州",
        "country": "China",
        "city_id": "苏州_cn",
        "bbox": [119.99, 30.7, 121.19, 31.9],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 120.59,
        "center_lat": 31.3,
    },
    "taiyuan": {
        "name": "太原",
        "country": "China",
        "city_id": "太原_cn",
        "bbox": [111.95, 37.27, 113.15, 38.47],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32649",
        "center_lon": 112.55,
        "center_lat": 37.87,
    },
    "taizhou": {
        "name": "台州",
        "country": "China",
        "city_id": "台州_cn",
        "bbox": [120.82, 28.06, 122.02, 29.26],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 121.42,
        "center_lat": 28.66,
    },
    "tianjin": {
        "name": "天津",
        "country": "China",
        "city_id": "天津_cn",
        "bbox": [116.6, 38.53, 117.8, 39.73],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 117.2,
        "center_lat": 39.13,
    },
    "urumqi": {
        "name": "乌鲁木齐",
        "country": "China",
        "city_id": "乌鲁木齐_cn",
        "bbox": [87.02, 43.19, 88.22, 44.39],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32645",
        "center_lon": 87.62,
        "center_lat": 43.79,
    },
    "wenzhou": {
        "name": "温州",
        "country": "China",
        "city_id": "温州_cn",
        "bbox": [120.1, 27.4, 121.3, 28.6],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 120.7,
        "center_lat": 28.0,
    },
    "wuhan": {
        "name": "武汉",
        "country": "China",
        "city_id": "武汉_cn",
        "bbox": [113.7, 29.99, 114.9, 31.19],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 114.3,
        "center_lat": 30.59,
    },
    "wuxi": {
        "name": "无锡",
        "country": "China",
        "city_id": "无锡_cn",
        "bbox": [119.7, 30.97, 120.9, 32.17],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32651",
        "center_lon": 120.3,
        "center_lat": 31.57,
    },
    "xiamen": {
        "name": "厦门",
        "country": "China",
        "city_id": "厦门_cn",
        "bbox": [117.49, 23.88, 118.69, 25.08],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 118.09,
        "center_lat": 24.48,
    },
    "xian": {
        "name": "西安",
        "country": "China",
        "city_id": "西安_cn",
        "bbox": [108.34, 33.66, 109.54, 34.86],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32649",
        "center_lon": 108.94,
        "center_lat": 34.26,
    },
    "xuzhou": {
        "name": "徐州",
        "country": "China",
        "city_id": "徐州_cn",
        "bbox": [116.58, 33.66, 117.78, 34.86],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32650",
        "center_lon": 117.18,
        "center_lat": 34.26,
    },
    "zhengzhou": {
        "name": "郑州",
        "country": "China",
        "city_id": "郑州_cn",
        "bbox": [113.05, 34.16, 114.25, 35.36],
        "crs": "EPSG:4326",
        "projected_crs": "EPSG:32649",
        "center_lon": 113.65,
        "center_lat": 34.76,
    },
}

ACTIVE_CITIES = [
    "beijing",
    "changchun",
    "changsha",
    "changzhou",
    "chengdu",
    "chongqing",
    "dalian",
    "dongguan",
    "foshan",
    "fuzhou",
    "guangzhou",
    "guiyang",
    "hangzhou",
    "harbin",
    "hefei",
    "hohhot",
    "jinan",
    "jinhua",
    "kunming",
    "lanzhou",
    "luoyang",
    "nanchang",
    "nanjing",
    "nanning",
    "nantong",
    "ningbo",
    "qingdao",
    "shanghai",
    "shaoxing",
    "shenyang",
    "shenzhen",
    "shijiazhuang",
    "suzhou",
    "taiyuan",
    "taizhou",
    "tianjin",
    "urumqi",
    "wenzhou",
    "wuhan",
    "wuxi",
    "xiamen",
    "xian",
    "xuzhou",
    "zhengzhou",
]

PIPELINE_CONFIG = {
    "cities": CITIES,
    "active_cities": ACTIVE_CITIES,
    "grid": {
        "cell_size_m": 500,
        "min_poi_count": 5,  # minimum POIs for a grid to be valid
        "min_road_length_m": 100,  # minimum road length for valid grid
    },
    "transit": {
        "buffer_radius_m": [200, 500, 800, 1500],  # distance bands from metro station
        "min_opening_year": 2000,
        "max_opening_year": 2024,
    },
    "poi": {
        "categories": {
            "food": ["restaurant", "cafe", "fast_food", "bar", "pub", "food_court"],
            "retail": [
                "supermarket",
                "convenience",
                "mall",
                "department_store",
                "clothes",
                "electronics",
                "bakery",
                "butcher",
                "greengrocer",
            ],
            "service": [
                "bank",
                "atm",
                "hairdresser",
                "beauty",
                "laundry",
                "post_office",
                "pharmacy",
                "hospital",
                "clinic",
                "dentist",
                "veterinary",
            ],
            "education": ["school", "university", "college", "kindergarten", "library", "training"],
            "leisure": [
                "park",
                "gym",
                "sports_centre",
                "theatre",
                "cinema",
                "museum",
                "art_gallery",
            ],
            "office": ["office", "coworking", "government", "embassy"],
            "lodging": ["hotel", "hostel", "guest_house", "motel"],
            "transport": ["bus_station", "subway_entrance", "taxi", "parking", "bicycle_rental"],
        },
    },
    "road_network": {
        "highway_types": [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "residential",
            "unclassified",
        ],
        "metrics": ["total_length", "intersection_density", "node_density", "average_circuity"],
    },
    "viirs": {
        "product": "VIIRS/BlackMarble/VNP46A2",  # monthly, 500m
        "start_year": 2012,
        "end_year": 2024,
        "metrics": ["mean_radiance", "median_radiance", "pct_above_threshold"],
    },
    "sentinel": {
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "start_year": 2018,
        "end_year": 2024,
        "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
        "cloud_filter": 20,  # max cloud cover percentage
        "indices": ["ndvi", "ndbi", "ndwi"],
    },
    "streetview": {
        "provider": "baidu",
        "heading_angles": [0, 90, 180, 270],
        "pitch": 0,
        "fov": 90,
        "image_size": "600x400",
        "sampling_strategy": "grid_center_one_point",
    },
    "hpi": {
        "source": "NBS 70-city HPI (国家统计局 70 个大中城市商品住宅销售价格指数)",
        "raw_csv": str(STAGING_DIR / "nbs_hpi" / "monthly.csv"),
        "article_index": str(STAGING_DIR / "nbs_hpi" / "article_index.csv"),
        "html_cache": str(RAW_DIR / "nbs_hpi" / "html_cache"),
        "city_yearly": str(HPI_LABEL_DIR / "hpi_city_yearly.parquet"),
        "grid_yearly": str(HPI_LABEL_DIR / "all_cities_hpi_yearly.parquet"),
        "coverage_months": "2021-12 ~ 2026-05 (53 months); 2010-2021 needs separate historical fetch",
        "city_coverage": "37 of our 44 cities (missing: changzhou, dongguan, foshan, nantong, shaoxing, suzhou, taizhou)",
        "housing_types": ["new", "secondhand"],
        "area_classes": ["total", "small (≤90m²)", "medium (90-144m²)", "large (>144m²)"],
        "metrics": [
            "mom (上月=100)",
            "yoy (上年同月=100)",
            "ytd (上年同期=100, until 2023-12)",
            "hpi_index (chained from base_year=2022)",
        ],
        "note": "HPI is city-level. Grid-year rows replicate the city value across all grids — "
        "use for cross-city heterogeneity & sanity check, not as grid-level price.",
        "fetch_command": "python scripts/collection/nbs_70city_discover.py && "
        "python scripts/collection/nbs_70city_hpi_fetcher.py && "
        "python scripts/labels/build_hpi_label.py",
    },
}

# ── Metro quick-reference (first-line opening years, verified 2025-06) ──
# Sources: official municipal / operator announcements; cross-checked with
# Wikipedia "List of metro systems" and the MoT 2025-05 statistics.
# Used as the missing-year fallback in build_treatment._fill_year_fallback.

METRO_REFERENCE = {
    "beijing": {"first_line_opened": 1969, "ref_lines": {}},  # 1号线 1969-10-01 trial
    "changchun": {"first_line_opened": 2002, "ref_lines": {}},  # 轻轨3号线 2002-10 trial
    "changsha": {"first_line_opened": 2014, "ref_lines": {}},  # 2号线 2014-04
    "changzhou": {"first_line_opened": 2020, "ref_lines": {}},  # 1号线 2020-09
    "chengdu": {"first_line_opened": 2011, "ref_lines": {}},  # 1号线 2010-09 trial, 2011 formal
    "chongqing": {"first_line_opened": 2005, "ref_lines": {}},  # 2号线 2005-06
    "dalian": {"first_line_opened": 2003, "ref_lines": {}},  # 快轨3号线 2003-05
    "dongguan": {"first_line_opened": 2012, "ref_lines": {}},  # R2线 2012-12 (now 2号线)
    "foshan": {"first_line_opened": 2010, "ref_lines": {}},  # 广佛线 2010-11
    "fuzhou": {"first_line_opened": 2016, "ref_lines": {}},  # 1号线 2016-05
    "guangzhou": {"first_line_opened": 1997, "ref_lines": {}},  # 1号线 1997-06 trial
    "guiyang": {"first_line_opened": 2018, "ref_lines": {}},  # 1号线 2018-12
    "hangzhou": {"first_line_opened": 2015, "ref_lines": {}},  # 1号线 2015-11
    "harbin": {"first_line_opened": 2014, "ref_lines": {}},  # 1号线 2014-09
    "hefei": {"first_line_opened": 2016, "ref_lines": {}},  # 1号线 2016-12
    "hohhot": {"first_line_opened": 2020, "ref_lines": {}},  # 1号线 2020-12
    "jinan": {"first_line_opened": 2019, "ref_lines": {}},  # 1号线 2019-12
    "jinhua": {"first_line_opened": 2022, "ref_lines": {}},  # 义乌-金华 2022-08
    "kunming": {"first_line_opened": 2012, "ref_lines": {}},  # 1号线 2012-06
    "lanzhou": {"first_line_opened": 2017, "ref_lines": {}},  # 1号线 2017-06
    "luoyang": {"first_line_opened": 2021, "ref_lines": {}},  # 1号线 2021-03
    "nanchang": {"first_line_opened": 2015, "ref_lines": {}},  # 1号线 2015-12
    "nanjing": {"first_line_opened": 2005, "ref_lines": {}},  # 1号线 2005-09
    "nanning": {"first_line_opened": 2016, "ref_lines": {}},  # 1号线 2016-06
    "nantong": {"first_line_opened": 2022, "ref_lines": {}},  # 1号线 2022-11
    "ningbo": {"first_line_opened": 2014, "ref_lines": {}},  # 1号线 2014-05
    "qingdao": {"first_line_opened": 2015, "ref_lines": {}},  # 3号线 2015-12
    "shanghai": {"first_line_opened": 1995, "ref_lines": {}},  # 1号线 1995-04
    "shaoxing": {"first_line_opened": 2021, "ref_lines": {}},  # 1号线 2021-06 (杭州-S1)
    "shenyang": {"first_line_opened": 2010, "ref_lines": {}},  # 1号线 2010-09
    "shenzhen": {"first_line_opened": 2004, "ref_lines": {}},  # 1号线 2004-12
    "shijiazhuang": {"first_line_opened": 2017, "ref_lines": {}},  # 1号线 2017-06
    "suzhou": {"first_line_opened": 2012, "ref_lines": {}},  # 1号线 2012-04
    "taiyuan": {"first_line_opened": 2020, "ref_lines": {}},  # 2号线 2020-12
    "taizhou": {"first_line_opened": 2022, "ref_lines": {}},  # S1线 2022-12
    "tianjin": {"first_line_opened": 1984, "ref_lines": {}},  # 1号线(老线) 1984-12
    "urumqi": {"first_line_opened": 2020, "ref_lines": {}},  # 1号线 2020-10 (full)
    "wenzhou": {"first_line_opened": 2019, "ref_lines": {}},  # S1线 2019-01
    "wuhan": {"first_line_opened": 2004, "ref_lines": {}},  # 1号线 2004-07
    "wuxi": {"first_line_opened": 2014, "ref_lines": {}},  # 1号线 2014-07
    "xiamen": {"first_line_opened": 2019, "ref_lines": {}},  # 1号线 2017-12 trial, 2019 full
    "xian": {"first_line_opened": 2013, "ref_lines": {}},  # 2号线 2013-09
    "xuzhou": {"first_line_opened": 2019, "ref_lines": {}},  # 1号线 2019-09
    "zhengzhou": {"first_line_opened": 2014, "ref_lines": {}},  # 1号线 2013-12 trial, 2014 full
}

# ── Admin boundary cache ─────────────────────────────────────────

try:
    from shapely import affinity as _shapely_affinity
    from shapely.geometry import MultiPolygon, Point, Polygon, shape  # noqa: F401
    from shapely.ops import unary_union as _shapely_unary_union

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


def shapely_affine_scale(geom, xfact: float, yfact: float):
    """Wrapper around shapely.affinity.scale that degrades gracefully when
    shapely is unavailable (returns the original geom).

    Uses ``origin=(0, 0)`` so the scaling is a pure coordinate
    transformation (no translation) — required for the latitude-corrected
    buffer in :func:`clip_grids_to_boundary`.
    """
    if not HAS_SHAPELY:
        return geom
    return _shapely_affinity.scale(geom, xfact=xfact, yfact=yfact, origin=(0, 0))


def _load_boundary_geojson(city_key: str) -> dict | None:
    """Load cached admin boundary GeoJSON, or None if unavailable."""
    p = BOUNDARY_DIR / f"{city_key}_boundary.geojson"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get_admin_boundary(city_key: str):
    """Return shapely Polygon/MultiPolygon for a city, or None."""
    if not HAS_SHAPELY:
        return None
    geojson = _load_boundary_geojson(city_key)
    if geojson is None:
        return None
    try:
        geojson_type = geojson.get("type")
        if geojson_type == "FeatureCollection":
            geometries = [
                shape(feature["geometry"])
                for feature in geojson.get("features", [])
                if feature.get("geometry") is not None
            ]
            if not geometries:
                raise ValueError("FeatureCollection contains no geometries")
            return _shapely_unary_union(geometries)
        if geojson_type == "Feature":
            return shape(geojson["geometry"])
        return shape(geojson)
    except Exception as exc:
        warnings.warn(
            f"Invalid boundary GeoJSON for {city_key}: {exc}; using bbox fallback",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def get_effective_bbox(city_key: str, buffer_km: float = 10.0) -> list[float]:
    """Return bbox from cached admin boundary + buffer, or hardcoded fallback.

    Hardcoded bbox is center ±0.6°; admin-derived bbox is boundary.extent
    + buffer in all directions.  This is the single source of truth for
    grid generation, GEE extraction, and transit queries.

    Buffer conversion accounts for latitude: longitude degrees are scaled
    by ``cos(lat_c)`` so the buffer is approximately square in meters.
    """
    cfg = get_city_config(city_key)
    boundary = get_admin_boundary(city_key)
    if boundary is None:
        return list(cfg["bbox"])

    minx, miny, maxx, maxy = boundary.bounds
    lat_c = (miny + maxy) / 2.0
    buf_lat_deg = buffer_km / 111.0
    buf_lon_deg = buffer_km / (111.0 * max(0.1, math.cos(math.radians(lat_c))))
    return [
        round(minx - buf_lon_deg, 4),
        round(miny - buf_lat_deg, 4),
        round(maxx + buf_lon_deg, 4),
        round(maxy + buf_lat_deg, 4),
    ]


def clip_grids_to_boundary(grids: list[dict], city_key: str, buffer_km: float = 10.0) -> list[dict]:
    """Filter grid cells to those intersecting the (buffered) admin boundary.

    When no boundary is cached, returns grids unchanged.  When the boundary
    is available, keeps only cells whose centroid lies within the buffered
    polygon.

    Buffer conversion accounts for latitude via ``cos(lat_c)``.
    """
    if not HAS_SHAPELY:
        return grids

    boundary = get_admin_boundary(city_key)
    if boundary is None:
        return grids

    # Buffer the boundary outward by buffer_km with latitude-corrected degrees
    minx, miny, maxx, maxy = boundary.bounds
    lat_c = (miny + maxy) / 2.0
    buf_lat_deg = buffer_km / 111.0
    cos_lat = max(0.1, math.cos(math.radians(lat_c)))
    # shapely.buffer is isotropic in the units of the geometry (degrees here),
    # so we approximate a latitude-corrected buffer by transforming to a
    # pseudo-isotropic space, buffering, then transforming back:
    #   1. COMPRESS x by cos(lat)  — now 1 deg-x ≈ 1 deg-y in meters
    #   2. Buffer isotropically by buf_lat_deg
    #   3. STRETCH x by 1/cos(lat)  — back to original coordinate system
    # The x-direction buffer in original space = buf_lat_deg / cos(lat),
    # which equals the desired buf_lon_deg.  (The previous implementation
    # had the scale factors reversed, making the x-buffer too small by
    # a factor of cos²(lat).)
    scaled = shapely_affine_scale(boundary, xfact=cos_lat, yfact=1.0)
    buffered_scaled = scaled.buffer(buf_lat_deg)
    buffered = shapely_affine_scale(buffered_scaled, xfact=1.0 / cos_lat, yfact=1.0)

    kept = []
    for cell in grids:
        pt = Point(cell["centroid_lon"], cell["centroid_lat"])
        if buffered.contains(pt) or buffered.touches(pt):
            kept.append(cell)
    return kept


# ── Helper functions ──────────────────────────────────────────────


def get_city_config(city_key: str) -> CityConfig:
    """Return city config dict for a given city key (e.g. 'beijing')."""
    if city_key not in CITIES:
        raise KeyError(f"Unknown city key '{city_key}'. Available: {list(CITIES.keys())}")
    return CITIES[city_key]


def city_dir(city_key: str, base: Path = GRID_DIR) -> Path:
    """Return a city sub-directory below a canonical dataset root."""
    return base / city_key


def ensure_dirs():
    for d in [DATA_DIR, GRID_DIR, TREATMENT_DIR, RAW_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for ck in ACTIVE_CITIES:
        if ck not in CITIES:
            continue
        for base in [GRID_DIR, RAW_DIR, TREATMENT_DIR]:
            city_dir(ck, base).mkdir(parents=True, exist_ok=True)


# ── Station name normalization (shared across sources) ──────────


def norm_station_name(name) -> str:
    """Normalize a metro station name for cross-source matching.

    Strips parentheticals, common suffixes (站 / trailing 路 / · / -), and
    whitespace, then lowercases.  Identical station names from different
    sources (Amap / OSM / Wikidata / Wikipedia) collapse to the same key
    so that overlap detection and dedup produce consistent results.

    Note on ``路``: only the *trailing* ``路`` is stripped (e.g. "建国路"
    → "建国") so that stations with ``路`` in the middle of the name
    (e.g. "五路居", "十路口") are not corrupted.

    Examples:
        "西二旗站"         -> "西二旗"
        "西二旗(地铁)"     -> "西二旗"
        "西二旗（地铁）"   -> "西二旗"
        "建国路"           -> "建国"
        "五路居"           -> "五路居"   (preserved — 路 is not trailing)
        "海淀黄庄·换乘"    -> "海淀黄庄换乘"
        "Xierqi Station"  -> "xierqistation"
    """
    return _normalize_station_name(name)


# ── Proxy auto-detection (shared across all fetchers) ───────────
# On networks where Overpass/Wikipedia/Wikidata return 406 or timeout
# to direct Python requests, routing through a local Clash proxy at
# 127.0.0.1:7890 fixes the issue.  We auto-detect the proxy by probing
# the port; callers can override with --proxy or the HTTPS_PROXY env var.

_PROXY_CACHE: str | None = None
_PROXY_DETECTED = False


def detect_proxy() -> str | None:
    """Return proxy URL if explicitly configured, else None.

    Checks the HTTPS_PROXY/HTTP_PROXY env var first, then an optional
    MIT_AUTO_PROXY_PORT variable for environments that run a local proxy.
    Automatic TCP probing of arbitrary ports is forbidden per
    code_standards.md §1.
    """
    env_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if env_proxy:
        return env_proxy
    auto_port = os.environ.get("MIT_AUTO_PROXY_PORT")
    if not auto_port:
        return None
    try:
        with socket.create_connection(("127.0.0.1", int(auto_port)), timeout=1):
            return f"http://127.0.0.1:{auto_port}"
    except OSError:
        return None


def get_proxy() -> str | None:
    """Lazily detect and cache the proxy URL (singleton)."""
    global _PROXY_CACHE, _PROXY_DETECTED
    if not _PROXY_DETECTED:
        _PROXY_CACHE = detect_proxy()
        _PROXY_DETECTED = True
    return _PROXY_CACHE


def get_proxies() -> dict:
    """Return ``{"http": url, "https": url}`` for requests, or empty dict.

    Usage:
        import requests
        from urban_intervention.config.project import get_proxies
        resp = requests.get(url, proxies=get_proxies(), ...)
    """
    p = get_proxy()
    return {"http": p, "https": p} if p else {}


def set_proxy(proxy_url: str | None) -> None:
    """Override the detected proxy (used by --proxy CLI argument)."""
    global _PROXY_CACHE, _PROXY_DETECTED
    _PROXY_CACHE = proxy_url
    _PROXY_DETECTED = True
