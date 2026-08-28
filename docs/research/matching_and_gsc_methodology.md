# 对照组匹配、广义合成控制与矩阵补全：方法说明

状态：active  
日期：2026-08-27
适用范围：5,048 个地铁站点处理网格的控制组选择、匹配标签生成、GSC 回退和 MC 第三层回退

## 1. 文档定位

本文件完整描述对照组选择的算法逻辑、文献出处、项目设计与原论文的异同。  
权威冻结决策见 DDR-003 和 DDR-004；本文件是方法层面的可读说明。Python/GPU 是当前正式
生产后端，R 包流程保留为参考、资格和敏感性审计。

---

## 2. 整体路由

```text
对每个处理网格 i：
  1. 提取处理前多变量特征（房价/VIIRS/POI/人口，按可用性筛选活跃变量族）
  2. 没有完整变量族 → 该结果族跳过，记录结构化原因
  3. 同城 Abadie-Imbens 最近邻匹配（lag2/lag3 训练，lag1 holdout）
  4. 匹配通过 → matched（冻结控制身份）
  5. 匹配失败 → 同城 Xu GSC
  6. 同城 GSC 失败 → 同城 MC（矩阵补全，Athey et al. 2021）
  7. 同城 MC 失败 → 跨城标准化匹配
  8. 跨城匹配通过 → matched
  9. 跨城匹配失败 → 跨城 Xu GSC
  10. 跨城 GSC 失败 → 跨城 MC
  11. 跨城 MC 失败 → skipped，记录结构化原因
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

#### 匹配与论文一致的部分

- `estimand = "ATT"`，`M = 1`（一对一最近邻）
- `replace = TRUE`（允许放回），`Weight = 2`（Mahalanobis 距离）
- `BiasAdjust = TRUE`，`Var.calc = 1`（完整 cohort ATT 估计时使用异方差稳健解析方差）
- 共同支持（common support）是 Abadie-Imbens 框架的标准要求

#### 项目新增规则

| 设计 | 项目关系 | 本项目说明 |
|------|------------|---------|
| `Y = NULL` 两阶段分离 | 项目新增 | 控制身份选择只读取处理前数据，依据 DDR-004 第 3 节执行 |
| lag1 holdout + donor-donor placebo q95 门禁 | 项目新增 | 使用 200 个确定性 donor 伪处理匹配校准项目质量阈值 |
| 同城优先 → 跨城标准化 fallback 路由 | 项目新增 | 同城优先保留相近经济环境，跨城路径使用处理前城市内标准化（median/MAD） |
| 三块 12 月特征（lag1/2/3） | 项目新增 | 月度房价交易较稀疏，分块求均值以覆盖实际观测月份 |
| 最少 1 个完整变量族（覆盖优先边界） | 项目新增 | 保留可用标签覆盖，单变量标签记录较低质量等级 |
| 控制身份冻结后保持不变 | 项目新增 | 控制身份在读取结果前完成冻结，后续结果只用于生成标签 |

#### 匹配路径的关键区别

论文定义的是**群体 ATT 估计器**：给定处理组和 donor pool，计算平均处理效应及其标准误。  
项目使用**个体控制身份选择**：为每个处理网格找到一个对照网格，冻结其身份后生成逐网格响应标签。完整 ATT 估计器独立运行（`run_complete_abadie_imbens.R`），逐网格标签保留独立的响应路径记录。

具体差异：
- 控制选择阶段：`BiasAdjust = FALSE`, `Var.calc = 0`（不识别偏差修正和方差，只选控制身份）
- 完整 ATT 阶段：`BiasAdjust = TRUE`, `Var.calc = 1`（独立报告 cohort 效应和标准误）
- 匹配标签不伪造个体标准误；其质量依据是共同支持、训练距离、holdout 误差和 placebo 分布

### 4.2 Xu GSC

#### GSC 与论文一致的部分

- `estimator = "gsynth"`（交互固定效应模型）
- `force = "two-way"`（个体效应 + 时间效应）
- `CV = TRUE`，`criterion = "mspe"`（处理前交叉验证选择因子数）
- `r = 0:5`（因子候选 0-5）
- `min.T0 = 5`（最少 5 个预处理期）
- `se = TRUE`，`inference = "parametric"`，`nboots = 200`（参数 bootstrap 标准误）
- `normalize = TRUE`（Xu 2017 推荐的标准化选项）
- donor 纳入只依据处理前完整路径

#### 项目自行设计的部分

| 设计 | 项目关系 | 本项目说明 |
|------|------------|---------|
| anticipation window（月度 6 个月，年度 0/1 年） | 项目新增 | 月度路径排除前 6 个月，年度路径使用 `annual_anticipation_years` 参数近似预期效应 |
| 跨城标准化（城市内 median/MAD 标准化后跨城 donor） | 项目新增 | 先消除城市经济规模差异，再使用跨城 donor 拟合因子结构 |
| 跨城 masked-placebo 门禁 | 项目新增 | 掩盖目标最后 12 个月，用 20 个 donor 伪处理校准 q95 RMSPE 阈值 |
| VIIRS 结构性边界检查 | 项目新增 | VIIRS 数据始于 2012-01，早期站点按 `min.T0=5` 检查处理前支持 |
| 两步 gsynth（先 CV 选 r，再固定 r 跑 bootstrap） | 项目新增 | CV 使用并行计算，bootstrap 顺序执行，控制多 GB 面板的内存占用 |
| 跨城标准化标签的反标准化 | 项目新增 | `Y.ct` 在标准化空间估计，使用目标城市的 `(pre_center, pre_scale)` 恢复原始尺度 |

#### GSC 路径的关键区别

论文的 GSC 是**群体-level 因果推断方法**：给定处理组和 donor pool，估计 ATT、反事实路径和不确定性。  
项目将 GSC 作为**逐网格标签生成器**：每次为一个处理网格及其 donor pool 拟合一个 gsynth 模型，提取该网格的反事实路径作为标签。

具体差异：
- 论文中处理组可以有多个单元；项目中 GSC 的处理组通常只有一个目标网格（`treatment_order` 指定）
- 论文不讨论 anticipation；项目在数据准备阶段排除 anticipation 期，不修改 `gsynth()` 调用本身
- 论文不约束 donor 数量下限；项目在跨城 placebo 路径需要至少 20 个 donor（技术需求：抽样 20 个做伪处理校准）

### 4.3 PanelMatch

#### PanelMatch 的一致部分

- `refinement.method = "mahalanobis"`，`size.match = 1`
- `matching = TRUE`，`match.missing = FALSE`，`listwise.delete = TRUE`
- `forbid.treatment.reversal = TRUE`
- `placebo.test = TRUE`，`number.iterations = 1000`，`se.method = "bootstrap"`

#### 项目角色

PanelMatch 作为**独立稳健性估计器**运行，不参与逐网格标签生产。其结果用于校验匹配和 GSC 标签的合理性，不作为训练信号；三种估计器分别报告。

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

零方差特征在 `active_matching_matrix` 中被移除（`feature_sd > sqrt(.Machine$double.eps)`），保证协方差矩阵可逆。

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

对至少有一个完整变量族的网格，以下匹配失败情况进入 `gsc_pending`：

- 共同支持失败
- 最近邻不存在
- placebo/holdout 门禁失败

没有完整变量族的结果族直接跳过，并记录结构化失败原因。

### 6.2 donor pool

- 同城 GSC：处理网格所在城市的全部合格 never-treated donor
- 跨城 GSC：全部 44 城的合格 never-treated donor

donor 纳入要求处理前路径完整（`pre_finite = TRUE` 且 `pre_periods == length(pre)`），处理后结果在此阶段保持隔离。

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

掩盖目标网格最后 12 个月（月度）或 1 年（年度）的预处理数据，用 gsynth 拟合剩余预处理期。从 donor 中抽样 20 个做相同的伪处理。目标网格的 masked RMSPE 需要处于 donor 伪处理第 95 百分位以内。

### 6.7 标签生成

```text
L[i,k,h] = Y_observed[i,k,h] - Y_counterfactual[i,k,h]
```

其中 `Y_counterfactual` 是 gsynth 的 `fit$Y.ct` 列。不确定性（SE、CI、p 值）来自 `fit$est.att` 的参数 bootstrap，按事件期写入标签路径。

---

## 7. 信息使用边界

| 阶段 | 使用的信息 | 隔离的信息 |
|------|---------|---------|
| 控制身份选择 | 处理前特征（lag2/lag3） | 处理后结果、处理后缺失状态、lag1 holdout |
| placebo 校准 | donor 的 lag2/lag3 和 lag1 | 处理后结果、目标网格的 lag1 |
| 匹配标签生成 | 冻结的控制身份 + 处理前基期 + 处理后结果 | 控制身份保持冻结 |
| GSC donor 纳入 | 处理前完整路径 | 处理后结果、处理后缺失 |
| GSC 标签生成 | 处理前路径 + 处理后结果 | donor 保持冻结 |
| 跨城标准化参数 | donor 的处理前值 | 处理单元的处理前值（不参与标准化参数计算） |

---

## 8. Matrix Completion 回退

### 8.1 进入条件

匹配和 GSC 均失败后自动触发。

### 8.2 文献出处

| 项目 | 内容 |
|------|------|
| 论文 | Athey, Bayati, Doudchenko, Imbens & Khosravi (2021). *Matrix Completion Methods for Causal Panel Data Models.* JASA 116:1716-1730 |
| 应用验证 | Ratledge et al. (2022), Nature，在乌格兰电力接入评估中验证 MC 在数据稀疏环境下的因果推断可靠性 |
| R 包 | `gsynth` 1.4.0, estimator=`"mc"` |
| 代码位置 | `scripts/causal_r/run_complete_mc.R` |

### 8.3 算法原理

矩阵补全将面板 N×T 处理为矩阵，处理单元的处理后值视为缺失。通过核范数
（nuclear norm）正则化，用剩余单元的观测值补全缺失的反事实路径：

```text
minimize ||Y_obs - L||_F^2 + λ||L||_*
```

其中 `||L||_*` 为矩阵的核范数（奇异值之和），等价于假设矩阵低秩（少数因子
驱动大部分变异）。lambda 通过交叉验证选择。

### 8.4 与 GSC 的关键区别

| 特性 | GSC | MC |
|------|-----|----|
| 模型 | `Y_it = α_i + ξ_t + λ_i'f_t + ε_it` | 核范数正则化矩阵补全 |
| 最少预处理期 | 5 | 1 |
| 共同支持 | 不要求 | 不要求 |
| 完整预处理路径 | 要求（missing 即 exclude） | 不要求（内部补全缺失值） |
| 平行趋势 | 通过交互固定效应近似 | 不要求 |
| 交叉验证 | 因子数 r ∈ {0,...,5} | 正则化参数 λ |
| donor 数量 | 无上限（项目实践不加 cap） | 有 soft cap（受内存限制，项目按预处理完整度选 top 2000） |
| inference | 参数 bootstrap 200 次 | 固定 lambda 后 jackknife；`nboots=200` 仅保留为兼容性元数据 |

### 8.5 项目设计与原文的关系

MC 方法来自 Athey et al. (2021)，项目不做算法修改。`min.T0=1` 和
`max_donors=2000` 是项目层面的应用参数，不属于论文算法内部步骤。Ratledge
et al. (2022) 验证了 MC 在数据稀疏环境下的可靠性。他们面对的是**结果变量
完全缺失**（需要 CNN 从卫星影像预测），我们面对的是**预处理期不足**（数据
有观测但不够 GSC 的 min.T0=5），因此不需要 CNN 预测步骤，直接使用 MC
补全反事实路径。

### 8.6 标签生成

```text
L[i,k,h] = Y_observed[i,k,h] - Y_counterfactual[i,k,h]
```

其中 `Y_counterfactual` 为 MC 模型的 `fit$Y.ct` 列。不确定性（SE、CI、p 值）
来自固定 lambda 后的 jackknife。MC 必须以 `CV=TRUE` 对 20 个候选正则化强度进行
MSPE 交叉验证；`lambda.cv`、CV MSPE、处理前观测期数和处理前 RMSPE 均写入标签与
manifest。处理单元的处理后结果在 CV 和拟合前统一替换为仅由处理前数据计算的均值，
保持响应信息隔离。`min.T0=1` 的标签标记为
`mc_*_minimal_pre_support`，但仍保留为可用响应标签。

## 9. 已知局限

### 9.1 SUTVA 假设

PanelMatch、Abadie-Imbens 和 Xu GSC 均假设 SUTVA（稳定单元处理值假设）。跨城 donor 所在城市可能已有地铁开通，1km 空间排除确保 donor 网格附近无站点，但 2km 外的网络溢出无法完全消除。DDR-001 预留了 1.5km/2km 敏感性阈值。

### 9.2 年度 anticipation 近似

月度路径的 6 个月 anticipation 无法在年粒度面板中精确表达。`annual_anticipation_years = 0`（默认）不排除开通前一年，对应月度 0 个月敏感性规格；`= 1` 排除开通前一年，对应 12 个月敏感性。Xu (2017) 不讨论 anticipation。

### 9.3 GSC donor 数量

Xu (2017) 和 `gsynth` 包均不约束 donor 数量下限。项目在跨城 placebo 路径需要至少 20
个 donor（抽样需求），同城路径无额外下限。Python/GPU 跨城 GSC 默认在读取结果前按固定
seed 做稳定哈希抽样，最多保留 50,000 个 donor；该上限和 seed 写入 specification
fingerprint。`min.T0 = 5` 已间接保证最低数据质量。

### 9.4 MC 内存限制

Athey et al. (2021) 不设 donor 数量上限。项目实践中，Python/GPU 和 R 参考 MC 都按
预处理期完整度降序最多保留 2,000 个 donor，以控制核范数补全的内存；该限制及其对估计
结果的影响必须在正式报告中明确记录。

---

## 10. 代码文件索引

| 文件 | 职责 |
|------|------|
| `scripts/causal_r/grid_control_design_lib.R` | 匹配特征构造、共同支持、Match 调用、placebo 校准、跨城标准化 |
| `scripts/causal_r/fixed_control_label_lib.R` | 冻结控制身份后的标签生成 |
| `scripts/causal_r/run_grid_control_design_queue.py` | 控制设计事务队列 |
| `scripts/causal_python/run_causal_label_queue.py` | 默认 Python/GPU 标签生产事务队列和 GSC/MC 路由 |
| `scripts/causal_r/complete_estimators_lib.R` | 共享的数据读取、面板构造和估计器 spec |
| `scripts/causal_r/run_complete_abadie_imbens.R` | 完整 Abadie-Imbens cohort ATT（独立估计器） |
| `scripts/causal_r/run_complete_panelmatch.R` | 完整 PanelMatch（独立估计器） |
| `scripts/causal_r/run_complete_xu_gsc.R` | Xu GSC 估计和标签生成 |
| `scripts/causal_r/run_complete_mc.R` | Matrix Completion 估计和标签生成（第三层回退） |
| `scripts/causal_r/formal_matching_lib.R` | 匹配 spec、Mahalanobis 距离、Moore-Penrose 逆 |
| `src/urban_intervention/causal/spatial_donors.py` | 空间 donor universe 和 1km 排除 |
| `src/urban_intervention/causal/response_artifact.py` | Response Artifact 严格发布器 |
| `src/urban_intervention/causal/pretraining_dataset.py` | 训练前数据集发布器 |


事件研究和 DID 平行趋势的实现与解释已集中到
[`identification_and_diagnostics.md`](identification_and_diagnostics.md)。公式附录见
[`estimator_formulas.md`](estimator_formulas.md)。
