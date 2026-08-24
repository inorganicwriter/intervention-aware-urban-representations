# 轨道网络可达性：方法-文献对照

本文档定义开通前轨道网络可达性、站点属性和线路属性协变量，并列出变量定义、
时间边界、计算步骤与文献依据。

## 数据源

| 数据 | 内容 | 获取方式 |
|------|------|----------|
| `canonical_station_events_resolved` | 项目站点事件（坐标、线路、开通日期） | 既有（wikidata/amap/osm/wikipedia 决议） |
| `wikidata_adjacency.parquet` | 19,602 条站点相邻关系（P197），44 城 | `wikidata_transit_fetch.py` SPARQL 分页抓取 |
| `transit_snapshots/{city}/{month}.parquet` | 每城每处理开通月的全网格处理前轨道特征（含 donor），482 个快照 | `build_transit_snapshots.py`（方案 A：按需快照缓存） |

### 快照结构（方案 A）

对每城每个去重后的处理开通月 t：
- 快照 = opening_date <= (t − 12 个月) 的已开通站点
- 对**该城全部网格**（含 donor）计算：最近站距离、500/800/1500m 站数、1500m 线路数、最近站 closeness
- closeness 在每城当前 P197 拓扑上计算一次（研究窗口 2010-2025 内拓扑与当年一致），快照只取已开通站的值
- 匹配时（R 侧 round 1/4）按 target 的处理前时点读取对应快照文件，donor 与 target 使用同一时点

这解决了"donor 网格轨道特征缺失"问题：donor 特征按 target 处理前时点动态对齐，不预存全量（5,048 时点 × 377 万网格的爆炸存储被 482 个快照文件替代）。

## 变量与文献对照

| # | 变量 | 定义 | 文献依据 | 实施细节 |
|---|------|------|----------|----------|
| 1 | `dist_nearest_station_m` | 网格质心到最近已开通站的直线距离 | Smersh & Smith (2000, JHE)；Yang et al. (2021, Springer) | 处理前快照：只考虑 opening_date <= 处理前时点（开通月 − 12 个月）的站 |
| 2 | `stations_500m` / `stations_800m` / `stations_1500m` | 处理前各半径内已开通站点数 | Debrezion et al. (2007, JRERF 元分析) | 与既有 treatment 文件一致的三档半径 |
| 3 | `lines_in_1500m` | 处理前 1.5km 缓冲区内经过的线路条数 | Debrezion et al. (2007)：线路密度维度 | 缓冲区内站点归属线路的并集数 |
| 4 | `network_closeness` | 处理前网络快照的站点 closeness centrality | To (2015, Urban Rail Transit)；Gao & Wang (2026, CBM 动态快照)；Wu et al. (2022, ICRT) | 站点节点 + **Wikidata P197 真实相邻边**，closeness=1/Σ最短路径距离（Dijkstra）；网格取最近站值 |
| 5 | `is_transfer_station` | 处理站是否换乘站（>=2 条线路） | 城市轨道运营惯例（Wikidata lines 字段） | 解析 `lines` 字段（`;` 分隔） |
| 6 | `is_terminal_station` | 处理站在开通月网络中是否终点站 | Wikidata P197 拓扑：度 ≤ 1 | 开通月网络（含同月开通站）中该站度 ≤ 1 |
| 7 | `is_new_line` / `is_extension` | 处理站所在线路是否新线（无更早开通站）或延长线（已有更早站） | 基于 opening_date 的运营史推断 | 线路首开年 = 该线路最早站开通年 |
| 8 | `stations_opened_same_month` | 处理站开通同月同城开通站数（同期开通规模） | 同期效应（anticipation 文献惯例） | opening_month 聚合计数 |

## 计算步骤

### 步骤 1：处理前网络快照（Gao & Wang 2026 动态网络）

对每个处理网格 i（开通于月 t_i）：
- 从 `canonical_station_events_resolved` 取 opening_date <= (t_i − 12 个月) 的站点
- 这些站点构成"处理前网络"（只含处理前信息，DDR-004 约束）

### 步骤 2：网络构建（To 2015，真实拓扑版）

- 节点 = 处理前已开通站点（station_event_id）
- 边 = **Wikidata P197 相邻站关系**（`wikidata_adjacency.parquet`，经归一化站名映射到项目 event id），边权 = 两站 haversine 直线距离 km
- 这是真实线路拓扑：换乘站在多条线路上各有相邻站，度 ≥ 2；终点站度 = 1
- 无 P197 边的站保持孤点（closeness = 0）

### 步骤 3：中心性计算（To 2015；Gao & Wang 2026）

- 每站 closeness = 1 / Σ_所有其他站 (最短路径距离 km)
- 用 Dijkstra（networkx），边权为站间直线距离
- 孤岛站（无任何连边）：closeness = 0

### 步骤 4：网格赋值（Wu et al. 2022）

- 每个网格取其最近已开通站（处理前快照内）的 closeness 值
- 快照内无站的网格：特征缺失（NaN/0 按设计文档处理）

### 步骤 5：线路/站点数（Debrezion et al. 2007）

- 对处理前快照的站点，按网格缓冲区分档计数
- 线路数 = 缓冲区内站点所属不同线路的并集数

### 步骤 6：站点属性（处理站自身）

- 换乘站：`lines` 字段线路数 >= 2
- 终点站：开通月网络（含同月开通站）中该站度 <= 1
- 新线/延长线：线路首开年（=该线路最早站开通年）vs 处理站开通年
- 同期开通：同城同月开通站数

## 一致性约束

- 所有变量只用处理前信息（与 DDR-004 匹配约束一致）
- 不读取任何处理后结果
- 结果写入 `data/active/causal/accessibility_features/` 供匹配使用

## 匹配接入（两阶段精炼）

静态协变量已接入控制设计匹配（`grid_control_design_lib.R`，2026-08-10）：

- 区位特征（`location_features/{city}_location.parquet`）与开通前轨道特征
  （`transit_snapshots/{city}/{opening_month}.parquet`，donor 与处理网格按同一
  处理前时点对齐）合并进匹配 frame，列名加 `loc_` / `transit_` 前缀。
- **两阶段控制选择**：阶段 1 按处理前结果滞后（lag2/lag3）Mahalanobis 匹配
  M=5 个候选；阶段 2 在候选中精炼静态协变量平衡最优者（`static_balance_refine()`）。
- 共同支持门禁与 holdout/placebo 门禁仍只作用于结果历史特征。处理网格按构造
  比 1km 排除后的 donor 更接近轨道网络，静态特征进入支持门禁会使匹配路径
  系统性失败（详见 `docs/research/matching_and_gsc_methodology.md` §5.4.1）。
- SMD 诊断（`feature_balance.parquet`）同时报告结果滞后与静态协变量。

## 文献列表

1. Smersh, G.T. & Smith, M.T. (2000). Accessibility changes and urban house price appreciation: A constrained optimization approach to determining distance effects. *Journal of Housing Economics* 9(3):187-205.
2. Debrezion, G., Pels, E. & Rietveld, P. (2007). The impact of railway stations on residential and commercial property value: A meta-analysis. *Journal of Real Estate Finance and Economics* 35(2):161-180.
3. Yang, L. et al. (2021). Place-varying impact of metro accessibility on property prices. In: *Property Price Impacts of Environment-Friendly Transport Accessibility in Chinese Cities*. Springer.
4. To, W.M. (2015). Centrality of an urban rail system. *Urban Rail Transit* 1(3):155-162.
5. Gao, Z. & Wang, Y. (2026). Subway network accessibility and carbon balance: Evidence from China's dynamic centrality analysis. *Carbon Balance and Management* 21:5.
6. Wu, Q. et al. (2022). Regional impact of urban rail transit network accessibility on residential property price. *ICRT 2021*, ASCE.
7. Zhang, X. et al. (2018). Urban rail transit network vulnerability measurement based on complex network theory: A case study of Chongqing rail transit. *DEStech Transactions on Computer Science and Engineering*.
