# 因果识别与平行趋势诊断

状态：当前识别说明

本文说明项目如何执行 DID 和事件研究的平行趋势诊断，并说明输出文件和结果标记。计量对象、识别假设、估计量、推断方法和稳健性框架见 [`econometric_methods.md`](econometric_methods.md)。

## 1. 诊断分组

事件研究诊断按处理网格、结果变量、结果族、估计方法和 donor scope 分组：

```text
L[i, k, h] = Y_observed[i, k, h] - Y_0_counterfactual[i, k, h]
```

匹配路径使用冻结的物理控制网格，GSC 和 MC 针对具体结果路径拟合反事实。housing、VIIRS、POI 和 population 分别保留诊断结果。

## 2. 事件时间和可用的处理前时期

| 频率 | 规格 | 估计前期 | 正式冲击期 | 处理后期 | 参考期 |
|---|---:|---|---|---|---:|
| 月度 | 主规格 | `-36:-1` | `0` 开通月 | `1:24` | `-1` |
| 月度 | anticipation 稳健性 | `-42:-1`（其中 `-6:-1` 为提前反应系数） | `0` 开通月 | `1:24` | `-7` |
| 年度 | 主规格 | 所有可用开通前年份 | `0` 开通年 | `1:3` | `-1` |

主规格把开通前一期作为标准事件研究参考期，便于四类 outcome 和三种方法直接比较。`t=0` 是正式冲击期，必须由真实开通月/年数据估计、保存和绘制，不能删除、插值或用 `t=1` 代替。月度稳健性规格允许开通前六个月已经受预期影响，因此省略 `t=-7`，并估计 `t=-6,...,-1,0,...,24`；它必须作为独立 Phase 2 拟合和独立产物报告，不能与主规格混合。

## 3. 匹配路径：事件研究 TWFE

在冻结的处理与控制配对上估计：

```text
Y_it = α_i + γ_t + Σ_{k∈K, k≠k0} β_k [treated_i × 1(event_time=k)] + ε_it
```

- `α_i` 是配对中每个网格的固定效应；`γ_t` 是日历期固定效应；
- 控制网格始终作为未处理单元，事件期虚拟变量只作用于 treated 单元；
- 主规格统一令 `k0=-1`；月度 anticipation 稳健性规格令 `k0=-7`；
- 主检验使用参考期之前最近 5 个清洁处理前 lead（`k0=-1` 时为 `-6..-2`），联合检验包含这 5 个系数。完整清洁处理前窗口保留在图形和系数表中。该低维检验避免约束数超过城市聚类自由度。联合原假设为：

```text
H0: β_k = 0  for every clean pre-treatment k < 0, k ≠ k0
```

结果判读同时查看：每个 `β_k` 的点估计与置信区间、最近 5 个 clean periods 的联合 Wald/F 检验、城市聚类和网格聚类两套结果，以及处理前支持数量。完整 clean lead 路径用于检查更早期的漂移，不在聚类自由度不足时强行构造高维联合 p 值。

### 3.1 聚类协方差与有限样本修正

设残差化后的事件期设计矩阵为 `X`，回归残差为 `u`，聚类集合为 `g=1,...,G`。
每个聚类的得分向量为：

```text
S_g = Σ_{i in g} X_i u_i
```

单向聚类协方差为：

```text
V = [G/(G-1)] [(n-1)/(n-K)] (X'X)^(-1) [Σ_g S_g S_g'] (X'X)^(-1)
```

`K` 按 fixest 默认的 nonnested 规则计算。事件期系数计入 `K`。嵌套于当前聚类键的
固定效应不重复计数。未嵌套的固定效应按其水平数计入 `K`。网格聚类和城市聚类分别
计算自己的 `G` 与 `K`。系数检验使用自由度 `G-1` 的 t 分布。clean leads 的联合检验为：

```text
F = (β_pre' V_pre^(-1) β_pre) / q
```

其中 `q` 是参与联合检验的 clean lead 数，参考分布为 `F(q, G-1)`。聚类数少于 2、
无可识别 clean lead 或协方差子矩阵不可识别时，检验结果写为缺失并记录支持不足。

### 当前实现和输出

Python 主事件研究入口是 `scripts/causal_python/run_all_method_event_study.py`，匹配部分调用 `urban_intervention.causal.event_study`：

- `event_study_coefficients.csv`：各事件期系数、网格/城市聚类标准误和置信区间；
- `parallel_trends_wald.csv`：按处理网格聚类的联合检验；
- `parallel_trends_wald_city_cluster.csv`：按城市聚类的联合检验；
- `pretrend_metadata.csv`：诊断标记，`admission_policy=diagnostic_only_not_an_automatic_label_gate`；
- `diagnostics.csv`：样本数、聚类数、参考期和方差估计类型。

R `scripts/causal_r/run_event_study_matching.R` 提供带 Sun and Abraham 敏感性路径的参考和审计实现。Python GPU 标签队列在生产运行中独立执行。

## 4. GSC/MC 路径：反事实 gap 的处理前诊断

GSC/MC 保留每个任务的处理前反事实 gap：

```text
gap[i, k, t] = Y_observed[i, k, t] - Y_hat_0[i, k, t]
```

项目先在每个 treated grid 内计算相对于所声明参考期的差分
`gap(h)-gap(k0)`，再按 `frequency × outcome_family × outcome × method × donor_scope × observation_window × reference_event_time` 分开聚合。不同参考期或 observation window 的产物不会混在同一条曲线中：

1. 每个事件期先报告网格间均值、标准差、样本量和 95% 区间；
2. GSC/MC 的差分标准误由同一任务的联合 bootstrap/jackknife draws 计算，不能用两个边际标准误拼接；聚合方差再包含 within-task 和网格间成分；
3. 处理前联合检验先在每个网格内对选定 clean pre-period 求均值，再在网格层面做 one-sample t 检验；
4. 由于同一城市的网格共享城市冲击，再把处理前 gap 先聚合到城市均值，在城市之间做同样的 one-sample t 检验；
5. 主规格使用参考期之前最近 5 个时期（即最多 `-6:-2`）做处理前联合诊断；稳健性规格只使用 `t<-7`，不把 `-6:-1` 的潜在提前反应误当作 clean pre-trend。

三种方法的跨方法图按 outcome 动态选择最长共同事件时间支持。图形保留所有可用
事件期，然后分别取 Matching、GSC、MC 中实际可用方法的最早/最晚区间交集。
主规格中 `h=-1` 是省略参考期、`h=0` 是正式冲击期，clean-pre 检验只使用
`h<-1`；月度稳健性中 `h=-7` 是参考期，`h=-6:-1` 是提前反应系数，clean-pre
检验只使用 `h<-7`。任何规格的 `h=0` 都不进入平行趋势检验。
因此早期 history 不再被固定的 3 年展示窗口人为删除，但某一种方法独有的额外时期也
不会冒充跨方法比较。主规格下月度 housing 与 VIIRS 的 Matching、GSC/MC 统一使用
`observation_window=1`，不做 post moving-window 聚合；housing 的
`observation_window=3` 仅作为独立敏感性规格；年度窗口固定为 1。读取器按结果族校验窗口，对缺失、非法或
不符合结果族口径的 estimator manifest 采用 fail-closed。

“pooled”表示把任务清单内所有被各方法接纳的 treated grids，在**同一结果族、同一
方法、同一 donor scope 和同一窗口**内聚合。housing、VIIRS、POI、population 的
量纲和频率不同，不把四类 outcome 强行混成一个系数。入口同时写出
`sample_aggregation_coverage.csv`、`event_time_support.csv` 和
`event_time_coverage.csv`，用于核对每个结果族、方法和事件期实际进入聚合的网格数。

输出目录由运行参数决定，通常包含：

- `event_study_series.csv`；
- `pretrend_grid_cluster.csv`；
- `pretrend_city_cluster.csv`；
- `effect_paths_with_pretrend.parquet`；
- `diagnostics.csv` 和事件研究图。

这些检验回答的是“估计出的处理前 gap 是否系统偏离 0”，不是“真实潜在结果必然满足平行趋势”。GSC 的额外识别依据是处理前因子结构、载荷稳定、donor 未受处理以及足够的处理前期；MC 的额外依据是低秩结构、交叉验证和处理前重构能力。

## 5. 设计阶段的配套证据

事件研究提供聚合诊断。正式结果还需在同一规格和同一来源记录下汇总以下证据：

| 证据 | 检查什么 | 作用 |
|---|---|---|
| 空间风险集和 1km 排除 | donor 是否可能被站点污染 | donor 纳入 |
| 共同支持 | 处理网格是否落在 donor 训练特征范围内 | 匹配路由 |
| 训练距离 | 处理与控制的 pre-only 状态差异 | 预设 q95 门槛 |
| lag1 holdout | 未用于选控制的最近处理前块能否复现 | 匹配质量 |
| donor-donor placebo | 在未处理 donor 中伪造处理时的经验误差分布 | 匹配质量 |
| GSC 跨城 masked placebo | 目标的遮蔽预处理预测误差是否超过 donor q95 | 跨城 GSC 路由 |
| 事件研究 lead/Wald | 聚合后的处理前 gap 或相对差异是否偏离 0 | 诊断记录 |
| anticipation/半径敏感性 | 预期效应和空间溢出是否改变结论 | 稳健性报告 |

holdout/placebo 属于控制设计阶段的 pre-only 选择门禁，事件研究属于结果生成后的聚合诊断。全部选择规则、q95 阈值和事件时间窗口在读取处理后效应前冻结。

## 6. 如何判读结果

- **未拒绝 H0**：当前样本、窗口、聚类方式和统计功效下未观察到明显的处理前偏离。该结果支持当前设定下的趋势一致性，解释范围限于本次样本和规格。
- **拒绝 H0**：标记为红旗，并检查处理前图形、各期支持数、城市聚类结果、pre-RMSPE、holdout/placebo、站点预期效应和竞争事件。标签保留依据支持范围和失败规则共同确定。
- **城市聚类优先用于主解释**：城市内网格共享政策、宏观和规划冲击，网格聚类结果作为补充结果展示。
- **支持不足单独报告**：p 值为 NA、聚类数不足、缺少 clean pre-period 或标准误无法识别时，记录 `insufficient_clusters` 或缺失状态。
- **各结果族分别解释**：housing、VIIRS、POI、population 的可用期、变换尺度和反事实模型分别记录，报告中分别展示对应的平行趋势结果。

事件研究诊断使用固定显著性阈值只是报告标记，不改变 `label_available`。正式 release 的可用性仍由 Response Artifact 的任务完整性、质量门禁、输入哈希和方法溯源决定。

## 7. 最低报告表

正式报告至少保存以下字段：

```text
frequency, outcome_family, outcome, method, donor_scope,
specification_fingerprint, pre_window, anticipation_window,
n_grids, n_cities, n_pre_observations,
pre_mean, pre_sd, grid_cluster_p_value, city_cluster_p_value,
pretrend_flag, pre_rmspe_summary, holdout_summary,
placebo_threshold, spatial_exclusion_km, run_id, input_hashes
```

图形显示所有有支持的事件期、95% 区间、参考期、`event_time=0` 正式冲击边界和样本量。稳健性规格的 anticipation 系数保留在图中，但从 clean-pre 联合检验中排除。

## 8. 复核入口

- Python TWFE：`src/urban_intervention/causal/event_study.py`；
- Python GSC/MC 聚合：`src/urban_intervention/causal/pooled_event_study.py`；
- R 事件研究聚合：`scripts/causal_r/event_study_lib.R`；
- 主设计：[`counterfactual_response_label_design.md`](counterfactual_response_label_design.md)；
- 稳健性规格：[`robustness_plan.md`](robustness_plan.md)；
- 单元测试：`tests/unit/test_python_event_study.py`、`tests/unit/test_pooled_event_study.py`、`tests/causal_r/test_event_study_aggregation.R`。
