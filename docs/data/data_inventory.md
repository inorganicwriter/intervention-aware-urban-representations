# 数据资产总览（Data Inventory）

更新：2026-08-27

本文档汇总当前项目的数据资产。与 `data/active/catalog/datasets.yaml` 互补：
本文档说明"有什么、在哪、干什么用"，datasets.yaml 记录每条目的 schema/质量/状态。

## 目录布局

```
data/
├── active/       冻结研究资产与队列基线（只读；服务器使用独立工作副本）
│   ├── curated/  标准协变量与结果（VIIRS/S2/人口/POI/房价/路网/区位特征）
│   ├── reference/ 冻结资产（网格/边界/站点/中心/adjacency）
│   ├── causal/   处理清单/队列/feature_store/轨道特征/快照
│   ├── panels/   房价网格面板（月/季/年）
│   ├── labels/   房价标签
│   └── catalog/  datasets.yaml 注册表
└── archive/      原始与中间数据（存档，不传服务器）
    ├── raw/      不可变原始（房价/transit/R2024B tif）
    └── staging/  可重建中间（GEE CSV、R2024B 聚合）
```

上传服务器时传输 `data/active/` 的冻结快照（archive 排除），见
`docs/operations/server_deployment.md`。队列运行产生的写入落在服务器独立
working copy，本地冻结目录保持只读。

## 数据集一览（样本量 + 来源）

实测样本量（2026-08-09）与数据来源。n = 空间单元数，m = 时间单元数。

| 数据集 | 样本量（n × m） | 来源 | 来源网站 |
|---|---|---|---|
| 500m 网格 | n = 3,839,581（44 城，每城 17,289-424,995） | 项目自建（GADM 边界 + UTM 500m） | https://gadm.org |
| 处理网格 | n = 5,048（44 城，2010-2025 开通） | 站点决议产物（wikidata/amap/osm/wikipedia） | https://www.wikidata.org |
| eligible donor | n = 3,771,800 | 空间 donor 审计（1km 排除） | 无 |
| 站点事件 | n = 5,615（44 城） | 四源交叉决议 | https://www.wikidata.org |
| VIIRS 月度 | n = 3.84M × m = 156（2012-01 至 2024-12）→ 6,864 分区 | NASA VNP46A2（GEE） | https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP46A2 |
| VIIRS 年度 | n = 3.84M × m = 13（2012-2024） | 月度聚合（项目内） | 无 |
| Sentinel-2 | n = 3.84M × m = 11（2014-2024） | Landsat 8 / Sentinel-2（GEE） | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED |
| 人口 | n = 3.84M × m = 15（2010-2024） | WorldPop（GEE 2010-14 + R2024B 2015-24） | https://data.worldpop.org |
| POI | n = 3.84M × m = 13（2012-2024） | 高德 POI 历史归档 | https://lbs.amap.com |
| 房价月度 | n ≈ 3.84M × m = 156（每城 236,278 网格-月） | 链家/安居客/wayback 等 4 源 | 见 `docs/data/housing_data_catalog.md` |
| 房价季度/年度 | 每城 99,779 / 35,843 | 同上聚合 | 同上 |
| 路网 | n = 3,839,581（2026 截面） | OSM | https://www.openstreetmap.org |
| 区位特征 | n = 3,839,581（静态） | 中心识别（McMillen 2001，composite） | 无 |
| 轨道可达性 | n = 5,048（处理网格） | Wikidata P197 + 站点事件 | https://www.wikidata.org |
| 轨道快照 | 482 快照 × 3.84M 网格 | 同上（方案 A 按开通月） | 同上 |
| Wikidata adjacency | 19,602 边（44 城） | Wikidata SPARQL（P197） | https://query.wikidata.org |
| 中心注册表 | 392 中心（44 城 main + 348 subcenter） | McMillen (2001) LWR | 无 |

## 1. 核心协变量与结果数据（`data/active/curated/`）

| 数据 | 路径 | 覆盖 | 用途 | 状态 |
|---|---|---|---|---|
| VIIRS 月度 | `curated/viirs/monthly/` | 44 城 × 156 月（6,864 分区） | 月度结果族（VNP46A2） | ✅ 0 重复 |
| VIIRS 年度 | `curated/viirs_annual_aggregated/` | 44 城 2012-2024 | 年度特征/标签 | ✅ 0 重复 |
| Sentinel-2 | `curated/sentinel2/` | 44 城 2014-2024 | NDVI/NDBI 特征 | ✅ 0 重复 |
| 人口 | `curated/population/` | 44 城 2010-2024 | 结果族/协变量（GEE 2010-14 + R2024B 2015-24） | ✅ 0 重复 |
| POI | `curated/poi/` | 44 城 2012-2024 | 结果族/协变量 | ✅ 0 重复（2018/19 量级断档已文档化） |
| 房价观测 | `curated/housing/housing_observations/` | 44 城 | 标准化房价观测（多源，经 importer 纳入检查） | ✅ |
| 房价面板 | `panels/housing_grid_{month,quarter,year}/` | 44 城 88×3 文件 | 房价结果族 | ✅ |
| 路网 | `curated/road_network/` | 44 城 2026 截面 | 静态协变量（未启用） | ✅ |
| 处理矩阵 | `curated/treatment/` | 44 城 176 文件 | 多半径站点覆盖 | ✅ |

## 2. 新变量（本轮构建）

| 数据 | 路径 | 覆盖 | 用途 |
|---|---|---|---|
| 区位特征 | `curated/location_features/` | 全 383.9 万网格 | dist_main / dist_subcentre / dist_centre |
| 轨道可达性（处理网格） | `causal/accessibility_features/` | 5,048 处理网格 | 最近站距离/站数/线路数/closeness + 站点属性 |
| 轨道快照（含 donor） | `causal/transit_snapshots/` | 482 快照 × 全网格 | 按处理时点对齐的 donor 轨道特征（方案 A） |
| Wikidata 相邻关系 | `reference/transit/wikidata_adjacency.parquet` | 44 城 19,602 边 | P197 拓扑（网络构建/终点站） |

方法依据：`docs/research/transit_accessibility_method.md`、`docs/research/related_work_literature.md`。

## 3. 冻结研究资产（`data/active/reference/`、`data/active/causal/`）

| 资产 | 路径 | 内容 |
|---|---|---|
| 网格 | `reference/grids/{city}/` | 500m 网格（含 GeoJSON，GEE 资产源） |
| 站点事件（决议后） | `reference/transit/canonical_station_events_resolved.parquet` | 5,615 事件 |
| 中心注册表 | `reference/city_centers.csv` | 44 城 main + 348 subcenter（composite 源） |
| 小区资产 | `reference/housing/` | 166,079 小区注册表 + AOI + 网格桥接 + 来源交叉索引（96.5% 已桥接） |
| 处理清单 | `causal/treatment_unit_list.csv` | 5,048 网格（冻结） |
| donor universe | `causal/grid_universe/` | 377 万 donor |
| eligible donors | `causal/formal_matching_inputs/` | 3,771,800 候选 + housing_annual 输入 |
| feature_store | `causal/feature_store/` | 88 文件；**无消费者**（R 匹配直接读 panels/VIIRS 分区），仅历史预计算输出，可跳过 |
| 队列基线 | `causal/*_queue.csv` | 控制、结果族和反事实队列；本地快照只读，服务器运行副本的实时状态见运行状态文档 |

## 3a. 房价标签（`data/active/labels/housing/`）

| 标签 | 路径 | 说明 |
|---|---|---|
| 安居客挂牌（截面） | `labels/housing/listing_price/anjuke/` | 2023-05 网格价格快照 |
| 安居客历史（采集管线） | `labels/housing/listing_price/anjuke_history/` | 小区历史月度（方案 2 产物，待采集） |
| 链家挂牌（2023 网格） | `labels/housing/listing_price/grid_2023/` | 链家 AOI 快照 |
| Wayback 历史快照 | `labels/housing/historical_snapshot/wayback/` | 5 源（lianjia/anjuke/beike 成交与小区） |
| 链家成交 | `labels/housing/transaction_price/lianjia/` | 购买导出 |
| NBS HPI | `labels/housing/city_hpi/` | 70 城月度指数 |

## 3b. 注册表（`data/active/catalog/`）

`datasets.yaml`（32 数据集注册表）、`sources.yaml`、`housing_acquisition_sources.yaml`、
schemas/（3）、mappings/（4）、inventories/（2）、quality/（2，人工决议）、
migrations/（2）、snapshots/（11，布局迁移前快照）。

## 4. 原始数据（`data/archive/raw/`、`data/archive/staging/`）

| 数据 | 路径 | 说明 |
|---|---|---|
| GEE S2 导出 | `staging/gee/s2/` | 484 CSV（reduceRegions，原始保留） |
| GEE POP 导出 | `staging/gee/pop/` | 484 CSV |
| R2024B tif | `raw/worldpop_r2024b/` | 10 年 GeoTIFF（27.1 GB） |
| R2024B 网格聚合 | `staging/worldpop_r2024b/` | 10 年 parquet |
| 房价原始 | `raw/housing/` | 1,710 文件（sha256 审计） |
| 交通原始 | `raw/transit/` | amap/osm/wikidata/wikipedia |

## 5. 输出（`outputs/`，可重建不入库）

| 目录 | 内容 |
|---|---|
| `figures/centers/` | 44 城中心识别审计图 |
| `figures/balance_loveplot.*` | 匹配 SMD 诊断 |
| `viirs_monthly/` | 6,864 月度分区审计/缓存（保留，不参与本次清理） |
| `data_quality/`、`gee_quality/` | 审计报告 |

## 6. 数据流（当前）

```
raw/staging → curated → 新变量 → feature_store / transit_snapshots
                                    → 6 轮匹配路由 → 标签 → Response Artifact
                                                          → 表示学习（独立管线）
```

## 已知且已接受的数据特性（非缺陷）

- POP 2014→2015 产品跳变（GEE vs R2024B 口径）
- S2 2018+ 缺失 ~16% 网格（数据源固有）
- POI 2018/2019 量级断档
- Wikidata P197 为当前拓扑（研究窗口 2010-2025 内与当年一致）
