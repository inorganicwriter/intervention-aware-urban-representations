"""Pipeline parameter configuration."""

from urban_intervention.data.paths import HPI_LABEL_DIR, RAW_DIR, STAGING_DIR

from .registry import ACTIVE_CITIES, CITIES

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
