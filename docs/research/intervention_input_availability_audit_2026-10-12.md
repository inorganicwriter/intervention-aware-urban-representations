# 干预输入资产与空间单元核对

日期：2026-10-12

后续更新：2026-10-13 已按正式网格完成 AlphaEarth 真实重提取，旧站点中心窗口的说明仅描述发现问题时的历史实现。详见 [空间口径统一与重提取记录](spatial_support_alignment_2026-10-13.md)。

补齐更新：2026-10-13 已实际生成 `outputs/causal_labels/station_attributes/station_attributes.parquet`（5,615 站点、5,048 处理事件），并发布 `data/active/intervention_grid_v1/intervention_inputs.parquet` 与 manifest。正式输入与冻结事件的 city、event ID、开通日期全部一致，保留全部事件。7 个字段及 masks：年份、同月开通数各 5,048 个可用；线路数、换乘、新线、延伸各 5,047 个；终点 4,264 个。原始 783 个终点未知及线路数为 0 的异常事件 `3537` 的相关类型字段均保留缺失。结构可用不等于历史归属已独立核验，manifest 保留 `requires_source_review`。以下“本地不存在/只在内存重算”描述 2026-10-12 审计时点。

范围：本地工作区内拟议干预 `T_i` 的数据可用性、字段构造与网格口径。读取真实数据，在内存中执行现有站点属性构造函数；没有重写正式站点属性、处理清单、网格或响应标签。服务器资产未核验。

结论：现有原始数据基本足够构建第一版干预输入，但本地尚无统一的、按开通时口径核验完成的干预属性发布表。正式分析网格是城市固定 500m × 500m 网格；AlphaEarth 的站点中心提取窗口是另一种空间支持域。站点到格心距离从核心干预输入调整为可选空间特征。

## 资产和覆盖

| 资产或字段 | 本地核对结果 | 第一版用途 |
|---|---|---|
| `data/active/causal/treatment_unit_list.parquet` | 5,048 行、44 城；5,048 个唯一站点事件和唯一 city/grid 组合。开通日期、月份、站点坐标、网格和事件 ID 均无缺失 | 已有开通时间和绑定键 |
| `data/active/reference/transit/canonical_station_events_resolved.parquet` | 5,615 个站点事件；覆盖全部 5,048 个处理事件；`date_precision` 全为 `day` | 属性构造原料；日期精度标记不等同于独立历史来源验收 |
| `data/active/reference/transit/wikidata_adjacency.parquet` | 19,602 条邻接记录 | 属性构造原料 |
| `data/active/reference/grids/{city}/{city}_grids.parquet` | 44 城文件均存在；全部 5,048 个处理事件可匹配格心坐标 | 正式固定网格空间单元 |
| `data/active/causal/accessibility_features/` | 44 个文件、5,048 行、5,048 个唯一 treatment_order，覆盖全部处理事件。旧站点类型和同月开通字段无空值 | 旧口径资产，不能直接重命名为新的开通时属性；最近既有站距离有 669 个空值，且不是新站到格心距离 |
| `outputs/causal_labels/station_attributes/station_attributes.parquet` | 本地不存在，其他本地目录也未检索到同名 Parquet | 需生成或取得服务器已生成产物，并按已有审阅流程核对 |

旧可达性脚本 `build_accessibility_features.py` 的换乘属性取当前线路归属，且终点、新线等定义与新属性脚本不同。旧表“无空值”不能证明开通时口径齐全。当前模型接口支持新属性编码，不等于本地已存在对应输入文件。

## 现有脚本可重算的开通时属性

本次通过 `scripts/analysis/build_station_attributes.py` 的 `load_frames()` 与 `compute_station_attributes()` 在内存中重算，未调用写正式文件的 `main()`。

| 建议字段 | 非空覆盖 / 5,048 | 核对结果 |
|---|---|---|
| `is_transfer_at_opening` | 5,048 | 582 个事件标为换乘 |
| `is_new_line_opening` | 5,048 | 2,883 个事件标为新线开通 |
| `is_extension_opening` | 5,048 | 2,469 个事件标为延伸，与新线属性可同时成立 |
| `is_terminal_at_opening` | 4,265 | 783 个未知：553 个换乘邻接度歧义，230 个缺少邻接证据；保留缺失 mask |
| `n_lines_at_opening` | 5,048 | 1 个事件线路数为 0，需核对或屏蔽该字段，不能据非空率认定完全正确 |
| `same_month_openings` | 5,048 | 均值 19.61；是同城同月开通规模，包含本站，并非网格附近新增站数 |

上述属性依赖当前线路归属及邻接记录，用开通日期重建历史状态。结构覆盖率与历史事实准确性分别核验，沿用 `docs/operations/station_attribute_audit_runbook.md` 的历史来源审阅，不要求重新采集整套站点资料。未知终点不需要导致事件被删除。

## 网格究竟是否以站点为中心

正式空间单元生成与绑定链路：

1. `scripts/collection/grid_builder.py::generate_grids_proj()` 从城市范围生成米制投影下的规则 500m 格网，`build_grids_for_city()` 按城市边界裁剪并保存。
2. `src/urban_intervention/causal/spatial_donors.py::compute_spatial_exposure()` 用 `covered_by` 将站点点位绑定到固定网格多边形；`build_treated_grid_registry()` 筛选合法处理网格。
3. `scripts/data/build_treatment_unit_queue.py` 冻结处理网格，清单元数据记录源为 `outputs/housing_did_spatial_feasibility/treated_grid_registry.parquet`。

冻结规则 `docs/research/decisions/DDR-001_spatial_treatment_and_donor_exclusion.md` 同样规定：分析单元为固定 500m × 500m 网格，实验组为包含正式站点的网格。`generate_station_grids()` 虽有站点中心相关命名，但函数生成的是站点整体范围加缓冲的统一格网，也不保证每个站点位于格心。

对全部真实处理事件，连接正式网格格心并计算球面直线距离（Haversine，地球半径 6,371km）：

| 统计量 | 站点到格心距离 |
|---|---|
| 有效匹配 | 5,048 / 5,048 |
| 最小值 | 3.13m |
| 中位数 | 198.11m |
| 均值 | 191.03m |
| 第 95 百分位 | 298.45m |
| 最大值 | 352.92m |
| 距离不超过 1m | 0 个事件 |

因此正式处理网格并不逐站居中。不过，这个距离主要刻画站点在格子中的位置；全体事件都已是“站点所在格”，不能仅因距离有差异就认为它是可解释的干预剂量。第一版先用开通时间、站点类型和规模，距离保留为可选空间特征或消融项。

另一个确实采用站点中心的流程是 `scripts/collection/extract_alphaearth_embeddings.py`：从处理清单读取站点 `wgs84_lon/lat`，构造各方向 ±0.0025° 的经纬度窗口后提取嵌入。此窗口并非正式网格多边形，也非各城市均精确 500m 的米制正方形。需要在模态元数据中区分正式分析单元与提取窗口；不能据此认定所有 POI、VIIRS、街景或未来独立影像都采用站点中心空间支持域。

## 第一版落实

核心干预字段为开通年份、四个站点类型属性、两个规模字段，共七个基础字段及可用性 mask。现有原料能够支持这一版，不新增街道路网采集任务。还需发布或取得新属性表、处理未知或异常值、核对历史口径，并接入干预编码器与检查点字段契约。此核对没有声明新输入已经完成训练验收。
