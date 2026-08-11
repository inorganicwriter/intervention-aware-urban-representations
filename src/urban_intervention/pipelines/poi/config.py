"""Shared constants for Amap POI processing."""

from urban_intervention.data.paths import (
    CATALOG_DIR,
    CURATED_DIR,
    PROJECT_ROOT,
    STAGING_DIR,
)

POI_ZIP = PROJECT_ROOT / "data" / "高德poi数据.zip"
POI_DIR = PROJECT_ROOT / "data" / "高德poi数据"
INTERIM_DIR = STAGING_DIR / "poi"
INVENTORY_PATH = CATALOG_DIR / "inventories" / "poi_asset_inventory.csv"
OUT_DIR = CURATED_DIR / "poi"

CSV_YEARS = set(range(2012, 2018))

CATEGORY_MAP = {
    "餐饮服务": "food",
    "购物服务": "retail",
    "生活服务": "life_service",
    "体育休闲服务": "leisure",
    "商务住宅": "business_residential",
    "公司企业": "office_enterprise",
    "住宿服务": "lodging",
    "科教文化服务": "education_culture",
    "医疗保健服务": "healthcare",
    "交通设施服务": "transport",
    "通行设施": "transport_access",
    "金融保险服务": "finance",
    "风景名胜": "scenic",
    "政府机构及社会团体": "government",
    "公共设施": "public_facility",
    "汽车服务": "auto_service",
    "汽车销售": "auto_sales",
    "汽车维修": "auto_repair",
    "摩托车服务": "motorcycle",
    "道路附属设施": "road_facility",
    "地名地址信息": "address",
    "室内设施": "indoor",
    "事件活动": "event",
}

ANALYSIS_CATEGORIES = [
    "food",
    "retail",
    "life_service",
    "leisure",
    "business_residential",
    "office_enterprise",
    "lodging",
    "education_culture",
    "healthcare",
    "transport",
    "transport_access",
    "finance",
    "scenic",
    "government",
    "public_facility",
    "auto_service",
    "auto_sales",
    "auto_repair",
    "motorcycle",
    "road_facility",
    "address",
    "indoor",
    "event",
    "other",
]

COMMERCIAL_CATEGORIES = {"food", "retail", "life_service", "leisure", "lodging", "finance"}
COMMUNITY_KEYWORDS = (
    "便利店",
    "超市",
    "菜市场",
    "综合市场",
    "洗衣",
    "美容",
    "美发",
    "维修",
    "药房",
    "药店",
    "餐馆",
    "快餐",
    "咖啡",
    "茶饮",
    "水果",
    "生鲜",
)
CHAIN_BRANDS = (
    "肯德基",
    "麦当劳",
    "星巴克",
    "瑞幸",
    "必胜客",
    "海底捞",
    "喜茶",
    "奈雪",
    "蜜雪冰城",
    "霸王茶姬",
    "库迪",
    "全家",
    "罗森",
    "711",
    "7-11",
    "屈臣氏",
    "永辉",
    "盒马",
    "华润万家",
    "沃尔玛",
    "家乐福",
    "麦德龙",
    "苏宁",
    "国美",
    "优衣库",
    "宜家",
    "迪卡侬",
)

NORMALIZED_COLUMNS = [
    "city",
    "year",
    "name",
    "lon",
    "lat",
    "typecode",
    "cate_A",
    "cate_B",
    "cate_C",
    "category",
    "is_commercial",
    "is_chain",
    "is_community_commerce",
]
