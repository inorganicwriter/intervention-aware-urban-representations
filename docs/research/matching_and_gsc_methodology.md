# 对照组匹配、广义合成控制与矩阵补全：方法说明

状态：active  
日期：2026-07-24  
适用范围：5,048 个地铁站点处理网格的控制组选择、匹配标签生成、GSC 回退和 MC 第三层回退

## 1. 文档定位

本文件完整描述对照组选择的算法逻辑、文献出处、项目设计与原论文的异同。  
权威冻结决策见 DDR-003 和 DDR-004；本文件是方法层面的可读说明。

---

## 2. 整体路由

```text
对每个处理网格 i：
  1. 提取处理前多变量特征（房价/VIIRS/POI/人口，按可用性筛选活跃变量族）
  2. 活跃变量族 < 1 → gsc_pending
  3. 同城 Abadie-Imbens 最近邻匹配（lag2/lag3 训练，lag1 holdout）
  4. 共同支持检查 → 不通过则 gsc_pending
  5. donor-donor placebo q95 门禁 → 不通过则 gsc_pending
  6. 同城通过 → matched（冻结控制身份）
  7. 同城未通过 → 跨城标准化匹配（相同流程）
  8. 跨城通过 → matched
  9. 跨城未通过 → gsc_pending
  10. gsc_pending 的网格按结果变量运行 Xu GSC
  11. GSC 失败 → 运行 MC（矩阵补全，Athey et al. 2021）
  12. MC 失败 → skipped（不降级补标签）
```

---

## 3. 文献出处与官方软件

### 3.1 Abadie-Imbens 最近邻匹配

| 项目 | 内容 |
|------|------|
| 论文 | Abadie, A. & Imbens, G. W. (2006). *Large Sample Properties of Matching Estimators for Average Treatment Effects.* Econometrica 74(1):235-267. |
| 偏差修正 | Abadie, A. & Imbens, G. W. (2011). *Bias-Corrected Matching Estimators for Average Treatment Effects.* Econometrica 79(1):33-65. |
| R 包 | `Matching` 4.10-15，`Match()` 函数 |
| 代码位置 | `scripts/causal_r/grid_control_design_lib.R`（控制选择），`scripts/causal_r/run_complete_abadie_imbens.R`（完整 ATT） |

### 3.2 Imai-Kim-Wang PanelMatch

| 项目 | 内容 |
|------|------|
| 论文 | Imai, K., Kim, I. S. & Wang, E. H. (2023). *Matching Methods for Causal Inference with Time-Series Cross-Sectional Data.* AJPS 67(3):587-605. |
| R 包 | `PanelMatch` 3.1.3 |
| 代码位置 | `scripts/causal_r/run_complete_panelmatch.R` |
| 角色 | 独立稳健性估计器，不参与逐网格标签生产 |

### 3.3 Xu 广义合成控制

| 项目 | 内容 |
|------|------|
| 论文 | Xu, Y. (2017). *Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models.* Political Analysis 25(1):57-76. |
| R 包 | `gsynth` 1.4.0 |
| 代码位置 | `scripts/causal_r/run_complete_xu_gsc.R` |
| 角色 | 匹配失败后的回退估计器 |

### 3.4 辅助文献

| 文献 | 用途 |
|------|------|
| Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), *Matrix Completion Methods for Causal Panel Data Models*, JASA 116:1716-1730 | MC 第三层回退的方法依据 |
| Ratledge et al. (2022), *Using Machine Learning to Assess the Livelihood Impact of Electricity Access*, Nature 611:491-495 | MC 在数据稀疏环境下的因果推断可靠性验证 |
| Callaway & Sant'Anna (2021), *DiD with Multiple Time Periods*, JoE | 交错采纳设计的时间合同参考 |
| Sun & Abraham (2021), *Event Study with Heterogeneous Treatment Effects*, JoE | 事件时间定义参考 |
| Roth (2022), *Pretest with Caution* | 平行趋势预检的警示，指导 placebo 设计 |
| Rambachan & Roth (2023), *A More Credible Approach to Parallel Trends* | anticipation window 的设计依据 |

---

## 4. 项目设计与原论文的异同

### 4.1 Abadie-Imbens 匹配

#### 与论文一致的部分

- `estimand = "ATT"`，`M = 1`（一对一最近邻）
- `replace = TRUE`（允许放回），`Weight = 2`（Mahalanobis 距离）
- `BiasAdjust = TRUE`，`Var.calc = 1`（完整 cohort ATT 估计时使用异方差稳健解析方差）
- 共同支持（common support）是 Abadie-Imbens 框架的标准要求

#### 项目自行设计的部分（不属于论文算法）

| 设计 | 论文是否涉及 | 项目理由 |
|------|------------|---------|
| `Y = NULL` 两阶段分离 | 否 | 防止处理后结果泄漏到控制身份选择；DDR-004 第 3 节规定 |
| lag1 holdout + donor-donor placebo q95 门禁 | 否 | 项目预注册的质量规则；用 200 个确定性 donor 伪处理匹配校准阈值 |
| 同城优先 → 跨城标准化 fallback 路由 | 否 | 城市内经济环境更相似；跨城需处理前城市内标准化（median/MAD） |
| 三块 12 月特征（lag1/2/3） | 否 | 月度数据稀疏（房价非每月交易），分块求均值避免要求特定月份有观测 |
| 最少 1 个完整变量族（项目保守选择） | 否 | 放宽以增加标签覆盖；单变量标签标记较低质量等级 |
| 控制身份冻结后不可因结果改变 | 否 | 防止基于处理后结果重新选择对照 |

#### 关键区别

论文定义的是**群体 ATT 估计器**：给定处理组和 donor pool，计算平均处理效应及其标准误。  
项目使用的是**个体控制身份选择**：为每个处理网格找到一个对照网格，冻结其身份后用于生成逐网格响应标签。完整 ATT 估计器独立运行（`run_complete_abadie_imbens.R`），但不替代逐网格标签。

具体差异：
- 控制选择阶段：`BiasAdjust = FALSE`, `Var.calc = 0`（不识别偏差修正和方差，只选控制身份）
- 完整 ATT 阶段：`BiasAdjust = TRUE`, `Var.calc = 1`（独立报告 cohort 效应和标准误）
- 匹配标签不伪造个体标准误；其质量依据是共同支持、训练距离、holdout 误差和 placebo 分布

### 4.2 Xu GSC

#### 与论文一致的部分

- `estimator = "gsynth"`（交互固定效应模型）
- `force = "two-way"`（个体效应 + 时间效应）
- `CV = TRUE`，`criterion = "mspe"`（处理前交叉验证选择因子数）
- `r = 0:5`（因子候选 0-5）
- `min.T0 = 5`（最少 5 个预处理期）
- `se = TRUE`，`inference = "parametric"`，`nboots = 200`（参数 bootstrap 标准误）
- `normalize = TRUE`（Xu 2017 推荐的标准化选项）
- donor admission 只依据处理前完整路径

#### 项目自行设计的部分

| 设计 | 论文是否涉及 | 项目理由 |
|------|------------|---------|
| anticipation window（月度 6 个月，年度 0/1 年） | 否 | TOD 文献（Cao & Porter-Nelson 2016）表明开通前已有预期效应；月度路径排除前 6 个月，年度路径用 `annual_anticipation_years` 参数近似 |
| 跨城标准化（城市内 median/MAD 标准化后跨城 donor） | 否 | 不同城市经济规模差异大，直接混用会导致大城市 donor 主导因子结构 |
| 跨城 masked-placebo 门禁 | 否 | 掩盖目标最后 12 个月，用 20 个 donor 伪处理校准 q95 RMSPE 门槛 |
| VIIRS 结构性边界检查 | 否 | VIIRS 数据始于 2012-01，早于此的站点不满足 gsynth `min.T0=5` |
| 两步 gsynth（先 CV 选 r，再固定 r 跑 bootstrap） | 否 | 控制内存：CV 可并行 8 核，bootstrap 顺序执行避免复制多 GB 面板 |
| 跨城标准化标签的反标准化 | 否 | `Y.ct` 在标准化空间估计，用目标城市的 `(pre_center, pre_scale)` 乘回原始尺度 |

#### 关键区别

论文的 GSC 是**群体-level 因果推断方法**：给定处理组和 donor pool，估计 ATT、反事实路径和不确定性。  
项目将 GSC 作为**逐网格标签生成器**：每次为一个处理网格及其 donor pool 拟合一个 gsynth 模型，提取该网格的反事实路径作为标签。

具体差异：
- 论文中处理组可以有多个单元；项目中 GSC 的处理组通常只有一个目标网格（`treatment_order` 指定）
- 论文不讨论 anticipation；项目在数据准备阶段排除 anticipation 期，不修改 `gsynth()` 调用本身
- 论文不约束 donor 数量下限；项目在跨城 placebo 路径需要至少 20 个 donor（技术需求：抽样 20 个做伪处理校准）

### 4.3 PanelMatch

#### 与论文一致的部分

- `refinement.method = "mahalanobis"`，`size.match = 1`
- `matching = TRUE`，`match.missing = FALSE`，`listwise.delete = TRUE`
- `forbid.treatment.reversal = TRUE`
- `placebo.test = TRUE`，`number.iterations = 1000`，`se.method = "bootstrap"`

#### 项目角色

PanelMatch 作为**独立稳健性估计器**运行，不参与逐网格标签生产。其结果用于校验匹配和 GSC 标签的合理性，不作为训练信号。DDR-003 明确规定三种估计器不得混合。

---

## 5. 匹配算法详细逻辑

### 5.1 特征构造

#### 月度变量（housing, viirs）

36 个干净预处理月分为三个不重叠的 12 月块：

```text
time_id 1-12  → lag3（最早 12 月，训练）
time_id 13-24 → lag2（中间 12 月，训练）
time_id 25-36 → lag1（最近 12 月，holdout）
```

每块内对有限月值求均值。housing 允许月份稀疏（`minimum_observations = 1`），VIIRS 要求更完整支持（`minimum_observations = 12`）。

特征命名：`{variable}__lag{1|2|3}`

#### 年度变量（poi, population）

```text
opening_year - 3 → lag3（训练）
opening_year - 2 → lag2（训练）
opening_year - 1 → lag1（holdout）
```

### 5.2 训练/holdout 分割

```r
training <- features[grepl("__lag[23]$", features)]  # lag2 + lag3
holdout  <- features[grepl("__lag1$", features)]     # lag1
```

控制身份只用 `training` 特征选择；`holdout` 特征在匹配完成后用于独立验证。

### 5.3 共同支持

对每个活跃训练特征，要求处理单元的值落在 donor 的闭区间 `[min, max]` 内：

```r
lower <- apply(X[donor_rows, ], 2, min)
upper <- apply(X[donor_rows, ], 2, max)
inside <- sweep(X, 2, lower, ">=") & sweep(X, 2, upper, "<=")
```

不在共同支持内的处理单元转入 GSC。

### 5.4 Mahalanobis 距离匹配

```r
Matching::Match(
  Y = NULL,           # 不读结果
  Tr = frame$Tr,      # 处理标识
  X = matching_matrix, # 训练特征
  estimand = "ATT",
  M = 5,               # 候选数（两阶段精炼用）
  replace = TRUE,      # 允许放回
  Weight = 2,          # Mahalanobis
  BiasAdjust = FALSE,  # 控制选择阶段不识别偏差修正
  CommonSupport = FALSE, # 已显式执行
  ties = FALSE,
  Var.calc = 0
)
```

零方差特征在 `active_matching_matrix` 中被移除（`feature_sd > sqrt(.Machine$double.eps)`），避免协方差矩阵不可逆。

### 5.4.1 静态协变量（区位/轨道）两阶段精炼

处理前时不变的区位与轨道协变量不进入共同支持门禁，也不直接进入单阶段距离，而是采用两阶段控制选择：

1. **阶段 1（结果历史）**：按处理前结果滞后特征（lag2/lag3）用 Mahalanobis 距离匹配 M=5 个候选控制；
2. **阶段 2（静态平衡精炼）**：在 5 个候选中，选择与处理单元在静态协变量上的 donor 协方差 Mahalanobis 距离最小的候选作为最终控制。

静态协变量清单（见 `docs/research/transit_accessibility_method.md`）：

- 区位：`loc_dist_main_km`、`loc_dist_nearest_subcentre_km`、`loc_dist_nearest_centre_km`
- 开通前轨道（按处理时点快照，donor 与处理同快照对齐）：`transit_dist_nearest_station_m`、`transit_stations_500m/800m/1500m`、`transit_lines_in_1500m`、`transit_network_closeness`

**设计理由**：处理网格按构造比 1km 排除后的 donor 更接近轨道网络（`dist_nearest_station_m` 等静态特征上处理与 donor 分布几乎不相交），若把这些特征放进共同支持门禁或单阶段距离，匹配路径会系统性失败（实测 16 网格采样接受率从 9/14 降至 1/14）。两阶段设计在保持结果历史匹配质量（接受率恢复至 9/16）的同时，让新协变量实际影响控制选择（静态 SMD 显著改善）。

零方差静态特征在精炼前被移除；`static_balance_refine()` 在全部静态特征零方差时退化为取第一个候选（等效 M=1）。

### 5.5 Placebo 校准

从 donor 中确定性抽样 200 个作为伪处理，对每个伪处理执行**与目标完全相同的两阶段匹配流程**（M=5 结果历史候选 + 静态精炼，排除自身），计算：

1. 训练 Mahalanobis 距离
2. holdout RMS 标准化差异
3. holdout 最大绝对标准化差异

三个指标各自的第 95 百分位作为门槛。目标匹配对必须**同时**通过三项门槛。

协方差逆矩阵用特征值分解构造（Moore-Penrose 伪逆），处理奇异协方差。

### 5.6 跨城标准化

同城匹配失败时，启用跨城 donor pool。所有特征按城市内 donor 的 median 和 MAD 标准化：

```r
center = median(donor_values)  # by city
scale  = mad(donor_values)      # by city, fallback to sd, fallback to 1
```

标准化后用相同的 Match + placebo 流程。处理单元的标准化参数来自其所在城市的 donor（不含处理单元自身）。

### 5.7 标签生成

匹配通过后，冻结控制身份 `j`，按结果变量读取处理后结果：

```text
月度: L[i,k,h] = (Y[i,k,h] - mean(Y[i,k,baseline_12m])) - (Y[j,k,h] - mean(Y[j,k,baseline_12m]))
年度: L[i,k,h] = (Y[i,k,h] - Y[i,k,opening_year-1]) - (Y[j,k,h] - Y[j,k,opening_year-1])
```

---

## 6. GSC 算法详细逻辑

### 6.1 进入条件

以下任一情况进入 `gsc_pending`：

- 活跃变量族 < 1
- 共同支持失败
- 最近邻不存在
- placebo/holdout 门禁失败（同城和跨城均失败）

### 6.2 donor pool

- 同城 GSC：处理网格所在城市的全部合格 never-treated donor
- 跨城 GSC：全部 44 城的合格 never-treated donor

donor admission 只要求处理前路径完整（`pre_finite = TRUE` 且 `pre_periods == length(pre)`），不读取处理后结果。

### 6.3 时间窗口

#### 月度 GSC

```text
opening_month          → 排除（部分暴露月）
opening_month - 1 ~ -6 → 排除（anticipation window）
opening_month - 7 ~ -42 → 干净预处理期（36 个月）
opening_month + 1 ~ +24 → 处理后期
```

VIIRS 受数据起点限制：`pre_months >= 2012-01-01`。

#### 年度 GSC

```text
opening_year           → 排除（过渡年）
opening_year - 1       → annual_anticipation_years=0 时保留；=1 时排除
opening_year - 2 ~ -3  → 预处理期
opening_year + 1 ~ +3  → 处理后期
```

`annual_anticipation_years` 是项目对 DDR-004 anticipation 合同在年粒度上的近似实现，不修改 gsynth 算法本身。

### 6.4 两步估计

```r
# 步骤 1：交叉验证选因子数
selection_fit <- gsynth(Y ~ D, CV = TRUE, r = 0:5, se = FALSE, ...)

# 步骤 2：固定因子数跑 bootstrap
fit <- gsynth(Y ~ D, CV = FALSE, r = selected_r, se = TRUE, nboots = 200, ...)
```

### 6.5 跨城标准化

跨城 GSC 先对每个城市的 donor 处理期计算 `mean` 和 `sd`，标准化后输入 gsynth。估计完成后，目标网格的反事实路径用其所在城市的 `(pre_center, pre_scale)` 反标准化回原始尺度。

### 6.6 跨城 masked-placebo 门禁

掩盖目标网格最后 12 个月（月度）或 1 年（年度）的预处理数据，用 gsynth 拟合剩余预处理期。从 donor 中抽样 20 个做相同的伪处理。目标网格的 masked RMSPE 不得超过 donor 伪处理的第 95 百分位。

### 6.7 标签生成

```text
L[i,k,h] = Y_observed[i,k,h] - Y_counterfactual[i,k,h]
```

其中 `Y_counterfactual` 是 gsynth 的 `fit$Y.ct` 列。不确定性（SE、CI、p 值）来自 `fit$est.att` 的参数 bootstrap，按事件期写入标签路径。

---

## 7. 信息边界（防泄漏设计）

| 阶段 | 允许读取 | 禁止读取 |
|------|---------|---------|
| 控制身份选择 | 处理前特征（lag2/lag3） | 处理后结果、处理后缺失状态、lag1 holdout |
| placebo 校准 | donor 的 lag2/lag3 和 lag1 | 处理后结果、目标网格的 lag1 |
| 匹配标签生成 | 冻结的控制身份 + 处理前基期 + 处理后结果 | 不允许因结果改变控制身份 |
| GSC donor admission | 处理前完整路径 | 处理后结果、处理后缺失 |
| GSC 标签生成 | 处理前路径 + 处理后结果 | 不允许因结果重选 donor |
| 跨城标准化参数 | donor 的处理前值 | 处理单元的处理前值（不参与标准化参数计算） |

---

## 7. Matrix Completion 回退

### 7.1 进入条件

匹配和 GSC 均失败后自动触发。

### 7.2 文献出处

| 项目 | 内容 |
|------|------|
| 论文 | Athey, Bayati, Doudchenko, Imbens & Khosravi (2021). *Matrix Completion Methods for Causal Panel Data Models.* JASA 116:1716-1730 |
| 应用验证 | Ratledge et al. (2022), Nature — 在乌格兰电力接入评估中验证 MC 在数据稀疏环境下的因果推断可靠性 |
| R 包 | `gsynth` 1.4.0, estimator=`"mc"` |
| 代码位置 | `scripts/causal_r/run_complete_mc.R` |

### 7.3 算法原理

矩阵补全将面板 N×T 处理为矩阵，处理单元的处理后值视为缺失。通过核范数
（nuclear norm）正则化，用剩余单元的观测值补全缺失的反事实路径：

```text
minimize ||Y_obs - L||_F^2 + λ||L||_*
```

其中 `||L||_*` 为矩阵的核范数（奇异值之和），等价于假设矩阵低秩（少数因子
驱动大部分变异）。lambda 通过交叉验证选择。

### 7.4 与 GSC 的关键区别

| 特性 | GSC | MC |
|------|-----|----|
| 模型 | `Y_it = α_i + ξ_t + λ_i'f_t + ε_it` | 核范数正则化矩阵补全 |
| 最少预处理期 | 5 | 1 |
| 共同支持 | 不要求 | 不要求 |
| 完整预处理路径 | 要求（missing 即 exclude） | 不要求（内部补全缺失值） |
| 平行趋势 | 通过交互固定效应近似 | 不要求 |
| 交叉验证 | 因子数 r ∈ {0,...,5} | 正则化参数 λ |
| donor 数量 | 无上限（项目实践不加 cap） | 有 soft cap（受内存限制，项目按预处理完整度选 top 2000） |
| bootstrap | 参数 bootstrap 200 次 | 非参数 bootstrap 200 次 |

### 7.5 项目设计与原文的关系

MC 方法来自 Athey et al. (2021)，项目不做算法修改。`min.T0=1` 和
`max_donors=2000` 是项目层面的应用参数，不属于论文算法内部步骤。Ratledge
et al. (2022) 验证了 MC 在数据稀疏环境下的可靠性——他们面对的是**结果变量
完全缺失**（需要 CNN 从卫星影像预测），我们面对的是**预处理期不足**（数据
有观测但不够 GSC 的 min.T0=5），因此不需要 CNN 预测步骤，直接使用 MC
补全反事实路径。

### 7.6 标签生成

```text
L[i,k,h] = Y_observed[i,k,h] - Y_counterfactual[i,k,h]
```

其中 `Y_counterfactual` 为 MC 模型的 `fit$Y.ct` 列。不确定性（SE、CI、p 值）
来自 200 次非参数 bootstrap。MC 必须以 `CV=TRUE` 对 20 个候选正则化强度进行
MSPE 交叉验证；`lambda.cv`、CV MSPE、处理前观测期数和处理前 RMSPE 均写入标签与
manifest。处理单元的处理后结果在 CV 和拟合前统一替换为仅由处理前数据计算的均值，
避免响应泄漏。`min.T0=1` 的标签标记为
`mc_*_minimal_pre_support`，但仍保留为可用响应标签。

## 8. 已知局限

### 8.1 SUTVA 假设

PanelMatch、Abadie-Imbens 和 Xu GSC 均假设 SUTVA（稳定单元处理值假设）。跨城 donor 所在城市可能已有地铁开通，1km 空间排除确保 donor 网格附近无站点，但 2km 外的网络溢出无法完全消除。DDR-001 预留了 1.5km/2km 敏感性阈值。

### 8.2 年度 anticipation 近似

月度路径的 6 个月 anticipation 无法在年粒度面板中精确表达。`annual_anticipation_years = 0`（默认）不排除开通前一年，对应月度 0 个月敏感性规格；`= 1` 排除开通前一年，对应 12 个月敏感性。Xu (2017) 不讨论 anticipation。

### 8.3 GSC donor 数量

Xu (2017) 和 `gsynth` 包均不约束 donor 数量下限。项目在跨城 placebo 路径需要至少 20 个 donor（抽样需求），同城路径无额外下限。`min.T0 = 5` 已间接保证最低数据质量。

### 8.4 MC 内存限制

Athey et al. (2021) 不设 donor 数量上限。项目实践中，`gsynth` 的 MC 模式在
大 donor pool（>~5000）下会触发内存溢出。当前按预处理期完整度降序选取前
2,000 个 donor。论文中应明确记录此限制及其对估计结果的潜在影响。

---

## 9. 代码文件索引

| 文件 | 职责 |
|------|------|
| `scripts/causal_r/grid_control_design_lib.R` | 匹配特征构造、共同支持、Match 调用、placebo 校准、跨城标准化 |
| `scripts/causal_r/fixed_control_label_lib.R` | 冻结控制身份后的标签生成 |
| `scripts/causal_r/run_grid_control_design_queue.py` | 控制设计事务队列 |
| `scripts/causal_r/run_causal_label_queue.py` | 标签生产事务队列和 GSC 路由 |
| `scripts/causal_r/complete_estimators_lib.R` | 共享的数据读取、面板构造和估计器 spec |
| `scripts/causal_r/run_complete_abadie_imbens.R` | 完整 Abadie-Imbens cohort ATT（独立估计器） |
| `scripts/causal_r/run_complete_panelmatch.R` | 完整 PanelMatch（独立估计器） |
| `scripts/causal_r/run_complete_xu_gsc.R` | Xu GSC 估计和标签生成 |
| `scripts/causal_r/run_complete_mc.R` | Matrix Completion 估计和标签生成（第三层回退） |
| `scripts/causal_r/formal_matching_lib.R` | 匹配 spec、Mahalanobis 距离、Moore-Penrose 逆 |
| `src/urban_intervention/causal/spatial_donors.py` | 空间 donor universe 和 1km 排除 |
| `src/urban_intervention/causal/response_artifact.py` | Response Artifact 严格发布器 |
| `src/urban_intervention/causal/pretraining_dataset.py` | 训练前数据集发布器 |


## 6. 事件研究（平行趋势验证）

三路径采用不同但互补的事件研究设计，统一输出到
\outputs/event_study/\。

### 6.1 匹配路径（TWFE 回归，新增）

对 matched pairs（处理网格 + 冻结控制网格）估计标准双向固定效应
事件研究回归（fixest）：

    Y_it = sum_k beta_k * D_it^k + alpha_i + gamma_t + eps_it

- Y_it：结果水平值（房价 log price / VIIRS asinh 辐亮度）
- D_it^k：事件时间虚拟变量，基期 k = -1 省略
- alpha_i：网格固定效应；gamma_t：日历月固定效应
- 标准误按网格聚类
- 平行趋势检验：H0: beta_k = 0 for all k < 0 的联合 Wald 检验

脚本：un_event_study_matching.R\。

### 6.2 GSC / MC 路径（counterfactual gap 聚合）

GSC（gsynth）与 MC（fect）估计器本身输出每期的处理效应序列
（est.att），pre-period 的 observed - counterfactual gap 即平行趋势
证据。聚合方式（\event_study_lib.R\ 的
\ggregate_event_study_series\）：

- 每 event_time 的均值 = 跨网格 ATT 均值
- 聚合标准误 = sqrt(within_var + between_var / n)
  - within_var = mean(SE_i^2) / n（bootstrap SE 聚合）
  - between_var = var(grid_mean)（网格间方差）
- 联合零 pre-trend 检验：网格级均值 one-sample t 检验

### 6.3 与文献的一致性

- Roth (2022) 警示：pre-trend 检验不能证明平行趋势，只作支持证据
- Sun & Abraham (2021)：匹配路径的 TWFE 作为基准，异质性稳健版本
  作为敏感性（待实现）
- 匹配路径的选择阶段 holdout/placebo q95 门禁独立于事件研究，
  两者互为补充

## 7. 公式汇总

全部计量公式（响应标签 / 匹配 / GSC / MC / 事件研究 / Sun-Abraham / SMD）
集中见 [estimator_formulas.md](estimator_formulas.md)，作为论文 Methods 底稿。
