"""City registry and metro reference configuration."""

from typing import TypedDict


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


def get_city_config(city_key: str) -> CityConfig:
    """Return city config dict for a given city key (e.g. 'beijing')."""
    if city_key not in CITIES:
        raise KeyError(f"Unknown city key '{city_key}'. Available: {list(CITIES.keys())}")
    return CITIES[city_key]
