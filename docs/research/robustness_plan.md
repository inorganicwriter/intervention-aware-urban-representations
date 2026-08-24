# Robustness Checks（稳健性检验）

预注册的稳健性检验。入口：`scripts/causal_r/run_robustness_checks.py`，
输出：`outputs/robustness/`。全部方法均有文献依据，不新造方法。

## 0. 推断有效性与聚类（Abadie et al. 2023）

地铁开通是**城市级政策**：同一城市网格共享城市冲击（土地政策、经济增长、
规划预期）。按网格聚类可能低估标准误，因此事件研究同时报告：

- **网格聚类**（主）：`cluster ~ grid_id`
- **城市聚类**（稳健）：`cluster ~ city_key`（Abadie, Athey, Imbens &
  Wooldridge 2023, QJE 138(1)，DOI: 10.1093/qje/qjac038）

GSC/MC 聚合侧对应增加**城市级均值检验**
（`joint_pretrend_tests_city`），与网格级检验并列报告。

## 1. 空间排除半径敏感性（已实现并可运行）

主规格 donor 排除 1km（DDR-001）。敏感性：1.5km / 2km（Yu et al. 2013 的
空间溢出检验框架）。

```
python scripts/causal_r/run_robustness_checks.py --spatial-exclusion
```

- 实现：`spatial_donors.py --exclusion-radius`（SpatialDonorSpec 支持）
- 产出：`outputs/robustness/spatial_exclusion/exclusion_{radius}m/`
- 实测（2026-08-09）：donor 数随半径变化（1km 最严）
- 判定：正式匹配后对比三种半径下的 ATT 稳定性

## 2. Anticipation 窗口敏感性（已实现，待正式产物）

主规格 6 个月。敏感性：0 / 12 个月（`complete_estimator_spec()$timing`）。

```
python scripts/causal_python/run_causal_label_queue.py \
  --start-order 1 --max-tasks 1 --anticipation-months 0
python scripts/causal_python/run_causal_label_queue.py \
  --start-order 1 --max-tasks 1 --anticipation-months 12
```

- 实现：`run_causal_label_queue.py --anticipation-months`
<!-- 旧路径 scripts/causal_r/run_causal_label_queue.py 由兼容包装器保留。 -->
- 判定：正式标签后对比 0/6/12 个月窗口的估计

## 3. 预处理窗口长度敏感性（已实现并可运行）

主规格 36 个月。敏感性：24 / 48 个月。

```
python scripts/causal_r/run_robustness_checks.py --window-length
```

- 实现：`robustness_window_smoke.R`（验证 monthly_event_calendar 在
  24/36/48 个月下正确构建）

## 4. 协变量集敏感性（已实现并可运行）

主规格全变量（含区位/轨道新变量）。敏感性：剔除轨道/区位变量。

```
python scripts/causal_r/run_robustness_checks.py --covariate-set
```

- 实现：`robustness_covariate_smoke.R`
- 实测（2026-08-09）：full=8 特征 / no_transit=6 特征，匹配均成功

## 5. Donor 池敏感性（已实现并可运行）

主规格同城优先（6 轮路由）。敏感性：跨城 standardized 对比。

```
python scripts/causal_r/run_robustness_checks.py --donor-pool
```

- 实现：`robustness_donor_scope_smoke.R`
- 实测（2026-08-09）：same=40 donors / cross=80 donors，均成功

## 6. 竞争事件排除（已实现并可运行）

de Chaisemartin & D'Haultfœuille (2020, AER) 指出 TWFE 在"处理单位被再次
处理"时存在偏误。本项目处理定义为网格**首站**开通，已核实同网格无第二站
（0 例多站网格）；竞争事件标记仅 2 例。检验逻辑：

```
python scripts/causal_r/run_robustness_checks.py --competing-events
```

- 实现：`robustness_competing_events.R`（剔除竞争事件网格后重估）
- 判定：全样本 vs 剔除子样本估计一致 → 多重处理不构成威胁

## 7. Spillover / 网络效应异质性（已实现）

地铁网络效应可能使邻近网格受益（Yu et al. 2013, JTG：
交通基础设施的空间溢出）。检验方式：

- 事件研究按**同期开通站数**（`stations_opened_same_month`）中位数分层
  （small vs large），对比两层的 Sun-Abraham 事件研究
- 输出：`outputs/event_study/matching/{family}/spillover_{stratum}_*.csv`

## 运行状态

| 检查 | 代码 | 可运行（无正式产物） | 正式运行 |
|---|---|---|---|
| 聚类（网格 vs 城市） | ✅ | ✅（smoke） | 匹配后 |
| 空间排除 | ✅ | ✅（真实数据已跑） | 匹配后对比 |
| Anticipation | ✅ | ✅（smoke） | 标签后对比 |
| 窗口长度 | ✅ | ✅（smoke 24/36/48） | 需接入匹配 lag 参数 |
| 协变量集 | ✅ | ✅（smoke full/no_transit） | 匹配后对比 |
| Donor 池 | ✅ | ✅（smoke same/cross） | 匹配后对比 |
| 竞争事件 | ✅ | ✅（smoke） | 匹配后对比 |
| Spillover 分层 | ✅ | ✅（逻辑内置） | 匹配后 |

## 文献

1. Abadie, A., Athey, S., Imbens, G.W. & Wooldridge, J.M. (2023). When should you adjust standard errors for clustering? *QJE* 138(1):1-35.
2. de Chaisemartin, C. & D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *AER* 110(9):2964-2996.
3. Yu, N., de Jong, M., Storm, S. & Mi, J. (2013). Spatial spillover effects of transport infrastructure: evidence from Chinese regions. *Journal of Transport Geography* 29:56-66.
