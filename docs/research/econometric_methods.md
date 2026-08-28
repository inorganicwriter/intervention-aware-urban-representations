# 计量经济学方法与因果识别手册

更新：2026-08-28
状态：当前计量方法主文档
适用范围：地铁站开通事件、500m × 500m 网格、housing、VIIRS、POI 和 population 结果族

本手册说明项目如何定义因果效应、构造未处理反事实、选择估计量、检验识别假设、计算不确定性和解释结果。它是计量理论与统计推断的集中入口。

项目参数、队列状态和文件合同分别见：

- [`counterfactual_response_label_design.md`](counterfactual_response_label_design.md)：处理清单、数据窗口、控制设计和标签路由；
- [`identification_and_diagnostics.md`](identification_and_diagnostics.md)：可执行的事件研究、平行趋势诊断和输出文件；
- [`matching_and_gsc_methodology.md`](matching_and_gsc_methodology.md)：匹配、GSC 和 MC 的算法映射；
- [`estimator_formulas.md`](estimator_formulas.md)：公式索引；
- [`decisions/`](decisions/)：冻结研究决策。

## 1. 研究问题和计量对象

### 1.1 研究单位

研究单位是城市内固定的 500m × 500m 网格。每个网格有以下标识：

| 符号 | 含义 |
|---|---|
| `i` | 网格 |
| `c(i)` | 网格所属城市 |
| `t` | 月度或年度日历期 |
| `k` | 结果变量，例如 housing_log_price、viirs_avg_asinh、poi_count_log、population_log |
| `f` | 结果族，例如 housing、VIIRS、POI、population |
| `G_i` | 网格内首个正式地铁站的开通期 |
| `h=t-G_i` | 相对事件期 |
| `D_it` | 处理状态 |

处理清单固定包含 5,048 个处理网格。donor 来自非处理网格，并受到处理站点周边空间排除规则约束。主规格把站点点位到 donor 网格多边形的最短投影距离设为至少 1km。

### 1.2 潜在结果

对结果变量 `k`，定义：

```text
Y_ikt(0) = 网格 i 在 t 期没有地铁开通影响时的潜在结果
Y_ikt(1) = 网格 i 在 t 期接受本次地铁开通影响时的潜在结果
```

观察到的结果满足一致性关系：

```text
Y_ikt = D_it * Y_ikt(1) + (1 - D_it) * Y_ikt(0)
```

这里的 `D_it` 表示与当前处理事件相对应的暴露状态。开通当月和开通当年属于部分暴露边界，主事件研究将其排除。下一个完整月或完整年定义为 `h=1`。

### 1.3 动态处理效应

网格 `i`、结果变量 `k` 和相对事件期 `h` 的局部动态效应为：

```text
τ_ikh = Y_ik,G_i+h(1) - Y_ik,G_i+h(0)
```

项目实际生成的标签是：

```text
L_ikh = Y_observed,ik,G_i+h - Y_counterfactual,ik,G_i+h
```

当反事实模型满足识别条件时，标签对应于局部动态处理效应的估计值。标签的数值尺度由结果变量决定：

| 结果族 | 标签尺度 | 直接解释 |
|---|---|---|
| housing | 对数价格差 | 近似百分比变化 |
| VIIRS | `asinh` 辐亮度差 | 变换后夜间灯光变化 |
| POI | `log1p` 计数差、比例差或熵差 | 对应变量尺度上的变化 |
| population | `log1p` 人口差 | 变换后人口变化 |

### 1.4 标签和 ATT 的区别

项目同时保存两类计量对象：

1. 逐网格动态标签：为表示学习提供监督信号；
2. 队列层面的 ATT 或动态 ATT：用于计量报告和方法比较。

逐网格标签和全体处理网格平均后的 ATT 属于两个不同的计量对象。一个任务对应一个处理网格、一个结果族、一个结果变量、一个 donor scope 和一个估计方法。队列汇总时，对同一方法、同一 donor scope、同一结果变量和同一事件期的可用任务做平均：

```text
μ_kh = (1 / N_kh) * Σ_i L_ikh
```

Matching、GSC 和 MC 使用不同的反事实结构，汇总时保持方法分组。跨方法结果保持分组，不构造跨方法均值。

## 2. 处理时间和样本窗口

### 2.1 月度事件时间

设站点在月份 `G_i` 内任意日期开通。月度事件时间为：

```text
h = calendar_month - opening_month
```

主规格使用：

| 区间 | 相对事件期 | 计量用途 |
|---|---:|---|
| 清洁处理前 | -42 到 -7 | 平行趋势、控制特征和预处理拟合 |
| anticipation | -6 到 -1 | 预期效应敏感性，不进入主清洁趋势 |
| 部分暴露边界 | 0 | 排除 |
| 处理后 | 1 到 24 | 动态响应 |

例如开通月份为 2019-12：

```text
2016-06 到 2019-05  -> 清洁处理前
2019-06 到 2019-11  -> anticipation
2019-12             -> 部分暴露月，排除
2020-01             -> h = 1
```

月度 TWFE 事件研究把 `h=-7` 作为参考期。该参考期是清洁处理前窗口的最后一个月，因此每个系数表示相对于 `h=-7` 的结果差异。

### 2.2 年度事件时间

年度结果使用开通年份 `G_i^year`：

```text
h = calendar_year - opening_year
```

主规格使用：

| 区间 | 相对事件期 | 计量用途 |
|---|---:|---|
| 清洁处理前 | -4 到 -1 | 年度趋势和年度控制特征 |
| 部分暴露边界 | 0 | 排除 |
| 处理后 | 1 到 3 | 年度动态响应 |

年度 TWFE 事件研究把 `h=-1` 作为参考期。年度 anticipation 使用 0 年和 1 年两种预先规定规格，结果中必须记录实际使用的参数。

### 2.3 处理前信息集

对处理网格 `i`，控制设计只能使用：

```text
X_i = {处理前结果历史、处理前数据可用性、处理前区位和轨道特征}
```

控制身份确定后，流程才读取处理后结果。处理后结果、处理后缺失状态、处理后显著性和最终标签不进入 donor 选择。

## 3. 识别假设

不同估计量需要不同的识别条件。平行趋势检验是其中一类诊断证据，不能替代完整的设计假设。

### 3.1 一致性和处理定义

一致性要求处理状态和结果变量具有明确含义：

1. `treatment_order` 对应唯一处理网格和唯一首个正式开通事件；
2. `opening_month` 保留原始日期映射到的月份；
3. housing、VIIRS、POI、population 的变换规则在估计前固定；
4. 同一网格和同一时期的重复观测按照数据合同处理；
5. 原始结果中的缺失值保持为缺失，不用面向因果结果的插值创造观察值。

一致性使 `Y_ikt(1)` 和 `Y_ikt(0)` 对应于项目实际定义的处理和结果。

### 3.2 清洁处理前的无提前反应

对清洁处理前窗口，项目采用：

```text
Y_ik,t(1) = Y_ik,t(0)  for t in clean pre-treatment window
```

开通前 6 个月允许存在预期反应，因此从主月度平行趋势窗口中移除。若 0、6、12 个月 anticipation 规格给出不同结果，应把差异作为处理时点和预期反应的敏感性证据。

### 3.3 匹配路径的条件平行趋势

令 `X_i` 为只由处理前信息构造的特征，`T_i=1` 表示处理网格，`C_i=1` 表示 donor。匹配路径需要在相似的 `X_i` 条件下，处理网格与 donor 的未处理结果变化具有相同条件均值：

```text
E[Y_i,t(0) - Y_i,t-1(0) | T_i = 1, X_i = x]
  = E[Y_j,t(0) - Y_j,t-1(0) | C_j = 1, X_j = x]
```

实际项目中使用有限维的处理前结果块、区位特征和轨道特征近似 `X_i`。共同支持保证处理网格的特征位置落在 donor 的可比较范围内。

该条件给出匹配差分的识别基础：

```text
E[(Y_i,t - Y_i,base) - (Y_j,t - Y_j,base) | matched pair]
```

在清洁处理前检验中，匹配对的相对变化应围绕 0 波动。

### 3.4 donor 未受处理和空间干扰

donor 在其用于构造反事实的时期内需要保持未处理。地铁网络会通过邻近站点、换乘、规划预期和土地市场影响周边网格，因此 donor 风险集使用空间排除规则。

主规格：

```text
站点点位到 donor 网格多边形的最短投影距离 >= 1km
```

1.5km 和 2km 用于空间敏感性。0.5km 作为探索性规格时，结果必须单独标注。后续站点、竞争交通和污染事件进入处理后观察期后，需要记录删失或污染状态。

### 3.5 共同支持和正值概率

匹配路径需要 donor 在每个活跃训练特征上覆盖处理网格：

```text
min_j X_jm <= X_im <= max_j X_jm  for every active feature m
```

共同支持是逐特征闭区间检查。高维特征空间的支持范围还需要用 Mahalanobis 训练距离和 holdout 误差描述。

### 3.6 GSC 的交互固定效应结构

GSC 允许未处理结果由单位效应、时间效应和低维共同因子共同决定：

```text
Y_it(0) = α_i + δ_t + λ_i' F_t + ε_it
```

其中：

| 符号 | 含义 |
|---|---|
| \(\alpha_i\) | 网格固定效应 |
| \(\delta_t\) | 日历期固定效应 |
| \(\lambda_i\) | 网格因子载荷 |
| \(F_t\) | 公共时间因子 |
| \(\epsilon_{it}\) | 剩余扰动 |

GSC 的目标反事实使用 donor 面板估计 \(\widehat F_t\)，再用目标网格的处理前结果估计其载荷或因子组合。主要条件包括：

1. donor 面板能表示目标网格的主要未处理变化；
2. 因子载荷在处理前后保持稳定；
3. donor 在整个估计窗口内没有受到当前事件污染；
4. 目标网格拥有足够的清洁处理前观测；
5. 处理后缺失不会改变 donor 选择和模型结构。

### 3.7 MC 的低秩结构

MC 将未处理结果矩阵写为：

```text
Y(0) = L + E
```

其中 `L` 是低秩信号矩阵，`E` 是残差。当前实现把两向固定效应作为初始结构，并用核范数惩罚控制低秩复杂度：

```text
min_L  (1 / |Ω|) * Σ_(i,t)∈Ω (Y_it - L_it)^2 + λ ||L||_*
```

Ω 是未处理观察单元集合，\(\|L\|_*\) 是奇异值之和。MC 的主要条件包括：

1. 未处理结果具有可由低秩结构表示的共同变化；
2. 缺失模式保留足够的处理前和 donor 支持；
3. 交叉验证选择的 \(\lambda\) 能控制拟合和复杂度；
4. 处理后结果不进入模型训练；
5. 目标处理前期数满足当前规格的最低要求。

MC 允许 `min.T0=1` 的覆盖优先边界。此时估计主要依赖跨单元结构，结果应报告实际处理前支持期数。

### 3.8 结果观测和测量稳定性

因果估计还需要结果观测过程在处理前后具有可比较性。项目按结果族分别检查：

- housing 的交易数量和有效网格月份；
- VIIRS 的有效采样点数、有效天数和时间覆盖；
- POI 的年度更新和类别覆盖；
- population 的网格年份支持和重复记录聚合。

结果观测概率发生变化时，估计值可能同时反映真实结果变化和观测机制变化。报告需要把 `n_observed`、有效月份或有效年份、交易数量阈值和结果变量尺度一起保存。

## 4. 控制设计和匹配估计量

### 4.1 控制设计的两阶段结构

控制设计和结果估计分为两个阶段：

```text
处理前信息 -> 风险集 -> 特征 -> 共同支持 -> 候选 donor -> 静态精炼 -> 质量门禁 -> 冻结控制身份
                                                                   |
                                                                   v
                                                            读取结果并生成标签
```

月度结果族的处理前特征分为三个 12 月块：

| 特征块 | 内容 | 用途 |
|---|---|---|
| `lag3` | 最早 12 个清洁处理前月 | 训练 |
| `lag2` | 中间 12 个月 | 训练 |
| `lag1` | 最近 12 个清洁处理前月 | holdout |

年度结果族使用开通年前第 3 年、第 2 年和第 1 年，前两年参与训练，前一年作为 holdout。

### 4.2 处理前特征向量

对结果变量或结果族 `m`，月度块均值为：

```text
X_im,b = mean(Y_im,t : t belongs to block b and Y_im,t is finite)
```

房价允许月份稀疏，但每个要求的块需要至少一个有效观测。VIIRS 需要更完整的月度支持。零方差特征从实际距离矩阵中移除，移除信息写入来源记录。

静态特征包括区位和处理前轨道可达性，例如：

```text
loc_dist_main_km
loc_dist_nearest_subcentre_km
loc_dist_nearest_centre_km
transit_dist_nearest_station_m
transit_stations_500m
transit_stations_800m
transit_stations_1500m
transit_lines_in_1500m
transit_network_closeness
```

### 4.3 Mahalanobis 距离

对处理网格 `i` 和 donor `j`，令 `X_i` 和 `X_j` 为训练特征，\(\widehat\Sigma\) 为 donor 训练特征协方差矩阵，则：

```text
d_ij = sqrt((X_i - X_j)' Σ_hat^(-1) (X_i - X_j))
```

当前 GPU 实现先对特征做样本标准化，再按特征值阈值构造对称伪逆。距离计算使用双精度和分块 GPU 矩阵运算，保留与原 donor 顺序一致的平局处理。

### 4.4 M=5 候选和静态精炼

主控制设计的候选数为 5：

1. 用处理前结果历史的 Mahalanobis 距离选择 5 个候选；
2. 在 5 个候选中，用区位和轨道静态特征的 Mahalanobis 距离选择最终 donor；
3. 静态特征全部无有效方差时，保留第一候选；
4. donor 允许被多个处理网格重复使用。

这套设计让结果历史决定总体趋势相似性，让静态特征影响最终控制身份。共同支持主要作用于活跃训练特征，静态特征用于候选精炼和平衡报告。

### 4.5 训练距离、holdout 和 placebo 门禁

对最终 donor `j(i)`，计算：

```text
training_distance_i = d_i,j(i)
```

holdout 特征按 donor 样本标准差标准化：

```text
gap_im = (X_holdout,im - X_holdout,j(i)m) / sd_donor,holdout,m
```

汇总为：

```text
RMS_i = sqrt(mean(gap_im^2))
MAX_i = max_m |gap_im|
```

在 donor 中按确定性位置抽取最多 200 个伪处理单元，对每个伪处理单元执行同样的 M=5 候选和静态精炼。训练距离、`RMS_i` 和 `MAX_i` 的第 95 百分位分别形成门槛：

```text
training_distance_i <= q95(training_distance_placebo)
RMS_i <= q95(RMS_placebo)
MAX_i <= q95(MAX_placebo)
```

三个条件同时通过时，记录 `quality_passed = TRUE`。q95 使用当前实现的 median-unbiased quantile type 8。该门槛属于项目质量规则，和 Abadie and Imbens 的 ATT 方差公式分开解释。

### 4.6 匹配标签

匹配控制身份冻结后，月度主标签为：

```text
L_ikh = (Y_i,k,G_i+h - B_i,k) - (Y_j(i),k,G_i+h - B_j,k)
```

月度 (B) 使用最近的清洁处理前块统计量。年度主标签使用开通前一年作为 baseline：

```text
L_ikh = (Y_i,k,G_i+h - Y_i,k,G_i-1) - (Y_j(i),k,G_i+h - Y_j(i),G_i-1)
```

缺失目标期和缺失 donor 期会使该期 `label_available = FALSE`。可用标签保留 0 值和接近 0 的值。

### 4.7 完整 Abadie and Imbens ATT

逐网格控制设计与完整 cohort ATT 使用不同计算对象。完整 ATT 使用处理组和 donor 组的结果值：

```text
ATT = (1 / N_T) * Σ_i [ΔY_i - Σ_j w_ij ΔY_j]
```

主匹配设置为：

```text
estimand = ATT
M = 1
replace = TRUE
Weight = 2
BiasAdjust = TRUE
Var.calc = 1
```

偏差修正把处理前协变量差异与控制组结果变化之间的线性关系纳入：

```text
ATT_adj = (1 / N_T) * Σ_i [ΔY_i - ΔY_j(i) + (X_i - X_j(i))' β_hat]
```

\(\widehat\beta\) 来自控制组结果变化对偏差修正协变量的加权回归。Abadie and Imbens 解析方差使用处理组和控制组的局部结果方差，以及控制 donor 被重复使用的权重。

该 ATT 的识别对象是给定 donor 风险集的处理组平均效应。逐网格标签承担响应路径记录功能，Abadie and Imbens 解析方差用于完整 cohort ATT。

## 5. GSC 估计量

### 5.1 donor 面板

每个 GSC 任务包含一个处理目标和一组 never-treated donor。估计面板按结果变量单独构造：

```text
Y = [目标网格, donor_1, ..., donor_N]
```

模型训练只使用 `observed & ~treated` 单元。目标网格处理后的观测不参与 donor 因子结构估计；目标的因子组合由其处理前观测拟合。

同城 GSC 使用目标城市的合格 donor。跨城 GSC 先按处理前城市内的 median 和 MAD 进行尺度标准化，结果在目标尺度上反标准化，并单独记录 donor scope。

### 5.2 交互固定效应模型

控制面板的拟合目标为：

```text
Y_it = α_i + δ_t + λ_i'F_t + ε_it
```

当前 GPU 实现使用 `force = two-way`，即单位固定效应和时间固定效应。因子数 `r` 的候选集合为 0 到 5。`r=0` 表示只使用两向固定效应。

对目标网格，先由 donor 面板估计公共时间结构，再用目标清洁处理前观测拟合目标的截距和因子载荷：

```text
Y_i,t = a_i + F_t' b_i + ε_i,t,  t in clean pre-treatment
Y_hat_i,t(0) = a_hat_i + F_hat_t' b_hat_i
```

GSC 标签为：

```text
L_i,t = Y_i,t - Y_hat_i,t(0)
```

### 5.3 因子数交叉验证

GSC 使用 forward rolling CV：

| 参数 | 当前默认值 |
|---|---:|
| 候选因子数 | `0, 1, 2, 3, 4, 5` |
| folds | 5 |
| 抽取 donor 比例 | 0.1 |
| 每次评分期数 | 3 |
| 训练缓冲期 | 1 |
| 最低处理前期数 | 5 |
| 选择规则 | 1SE |

每个 fold 从 donor 处理前路径抽取部分单位，移除评分期以及其后的训练区域，模型只在更早的观测上拟合。评分指标为被移除观测上的 pooled mean squared prediction error：

```text
MSPE(r) = Σ_(i,t)∈S [Y_it - Y_hat_it(r)]^2 / |S|
```

1SE 规则先找到 MSPE 最小的因子数 `r_min`，再选择 MSPE 位于 `MSPE(r_min) + SE(r_min)` 内的最小模型复杂度。结果记录每个候选 `r` 的 MSPE、标准误、选定因子数和 CV 来源。

### 5.4 GSC 不确定性

正式 GSC 使用 200 次参数 bootstrap。当前 `auto` 规则为：

| donor 面板 | bootstrap 来源 |
|---|---|
| 完整面板 | empirical reference |
| 存在缺失 | reference AR |

每次抽样重新生成目标和 donor 路径，重新拟合固定的因子数，并保存每个时期的 effect draw。时期 `t` 的标准误为有效 draw 的样本标准差：

```text
SE_t = sd({τ_t^(b) : b = 1,...,B and draw is finite})
```

置信区间和双侧 p 值使用正态近似：

```text
CI_95,t = τ_hat_t ± 1.959964 * SE_t
p_t = 2 * Φ(-|τ_hat_t / SE_t|)
```

结果必须同时记录请求次数和实际收敛次数。任何处理后时期有效 draw 数低于最低比例时，任务进入失败处理。

### 5.5 GSC 诊断

每个 GSC 任务至少保存：

```text
selected_rank
cv_mean_mspe
cv_se_mspe
pre_rmspe
post_rmspe
converged
iterations
n_pre_observed
n_post_observed
bootstrap_repetitions
valid_repetitions
donor_scope
```

处理前 RMSPE 定义为：

```text
pre_RMSPE_i = sqrt(mean((Y_i,t - Y_hat_i,t(0))^2 : t in clean pre-treatment))
```

GSC 结果还需要结合 cross-city masked placebo。该 placebo 把 untreated 目标的最近处理前一段遮蔽，用 donor 预测被遮蔽区域，比较目标误差与伪处理 donor 误差分布。

## 6. MC 估计量

### 6.1 矩阵结构

MC 在每个结果变量上单独构造时间 × 网格矩阵。观察集合为：

```text
Ω = {(t,i): outcome is finite and unit i is untreated at t}
```

两向固定效应提供初始拟合。核范数惩罚后的更新使用奇异值软阈值：

```text
s_l^shrunk = max(s_l - λ * T * N, 0)
```

当前实现将收敛的最终拟合矩阵作为未处理潜在结果的估计，目标网格的反事实路径为：

```text
Y_hat_i,t(0) = L_hat_ti
```

### 6.2 Lambda 交叉验证

MC 使用 rolling CV 选择 \(\lambda\)：

| 参数 | 当前默认值 |
|---|---:|
| lambda 候选数 | 20 |
| lambda 最小比例 | \(10^{-3}\) |
| folds | 20 |
| 抽取单位比例 | 0.1 |
| 每次评分期数 | 1 |
| 缓冲期 | 0 |
| 最低处理前期数 | 1 |
| 选择规则 | 1SE |

正值候选从两向固定效应残差的最大奇异值构造，并追加 \(\lambda=0\) 的无惩罚候选。每个 fold 把评分单元格从训练集合移除：

```text
MSPE(λ) = Σ_(i,t)∈S [Y_it - Y_hat_it(λ)]^2 / |S|
```

1SE 规则选择满足误差范围的最大 \(\lambda\)，从而优先选择更强正则化和更低复杂度的模型。输出保存完整 lambda 网格、每个候选的 MSPE 和选定值。

### 6.3 MC 标签和 jackknife

MC 标签为：

```text
L_i,t = Y_i,t - Y_hat_i,t(0)
```

正式 MC 使用单位 leave-one-out jackknife。对每个 donor 单位 `j` 删除该单位，固定已经选择的 \(\lambda\)，重新拟合 MC 并得到目标效应路径 \(\widehat\tau_t^{(-j)}\)。令总单位数为 `N`，伪值为：

```text
p_t^(-j) = N * τ_hat_t - (N - 1) * τ_hat_t^(-j)
```

jackknife 标准误为有效伪值的样本方差除以有效数量：

```text
SE_t^JK = sqrt(var({p_t^(-j)}) / n_valid,t)
```

当前实现使用 (N-1) 个 donor 删除重拟合，处理目标本身保留在每次拟合中。有效重拟合数量和失败原因写入推断记录。

### 6.4 MC 诊断

每个 MC 任务至少保存：

```text
selected_lambda
effective_rank
cv_mean_mspe
cv_se_mspe
pre_rmspe
post_rmspe
converged
iterations
n_pre_observed
n_post_observed
jackknife_repetitions
valid_repetitions
```

MC 的处理前拟合质量直接影响目标处理后的反事实。`min.T0=1` 的任务应单独报告，因为单个处理前期无法形成传统的多期趋势诊断。

## 7. DID 与事件研究

### 7.1 匹配样本上的 TWFE

在控制身份已经冻结的处理网格和 donor 网格上，项目估计：

```text
Y_it = α_i + γ_t + Σ_(h∈H, h≠h0) β_h [Treated_i * 1(event_time=h)] + ε_it
```

其中：

| 部分 | 含义 |
|---|---|
| \(Y_{it}\) | 匹配样本结果变量 |
| \(\alpha_i\) | 网格固定效应 |
| \(\gamma_t\) | 日历期固定效应 |
| \(\beta_h\) | 相对于参考期 \(h_0\) 的动态系数 |
| \(H\) | 清洁处理前和处理后事件期集合 |
| \(h_0\) | 月度 -7，年度 -1 |

控制网格的事件期虚拟变量全部编码为参考期，事件虚拟变量只作用于 treated 网格。Python 实现先吸收单位和时期固定效应，再用最小二乘估计事件系数。R 实现用 `fixest::feols` 做参考和审计。

### 7.2 系数解释

\(\widehat\beta_h\) 表示：在固定网格效应和共同日历期冲击之后，处理网格在相对事件期 h 相对于参考期 h_0 的平均结果差异。

处理前系数表达相对于参考期的相对变化，因此平行趋势检验应检查处理前系数是否共同接近 0。

处理后系数描述动态路径。它们同时受处理效应、剩余趋势、竞争事件和结果观测支持影响，报告中需要和样本支持、处理后缺失、空间规格一起解释。

### 7.3 平行趋势联合 Wald 检验

令 \(\widehat\beta_{pre}\) 为所有清洁处理前 lead 的系数向量，\(\widehat V\) 为相应聚类协方差矩阵。检验：

```text
H0: β_h = 0 for every h < 0 and h != h0
H1: at least one clean pre coefficient differs from zero
```

联合统计量为：

```text
F = (β_hat_pre' V_hat_pre^(-1) β_hat_pre) / q
```

其中 `q` 是清洁处理前系数数量。Python 实现使用 `F(q, G-1)` 参考分布，`G` 是聚类数量；R `fixest::wald` 输出独立的参考结果。

单期系数和联合检验回答不同问题：

- 单期检验识别哪个事件期出现偏离；
- 联合检验识别处理前 lead 是否整体偏离 0；
- 图形显示系数大小、置信区间和支持数量。

### 7.4 聚类协方差

Python TWFE 使用 CRV1 聚类协方差：

```text
V_hat = correction * (X'X)^(-1) [Σ_g X_g' e_g e_g' X_g] (X'X)^(-1)
```

有限样本修正为：

```text
G / (G - 1) * (n - 1) / (n - p)
```

其中 `G` 是聚类数量，`n` 是样本量，`p` 是事件系数数量。每个系数使用聚类数减 1 的 t 分布计算 p 值和 95% 置信区间。

项目同时输出两种聚类：

| 聚类层级 | 解释 |
|---|---|
| `grid_cluster` | 处理网格或网格配对层面的相关误差 |
| `city_cluster` | 同一城市共同面对规划、土地、宏观和网络冲击 |

城市层级更接近地铁政策的实施层级，主解释优先报告城市聚类结果。网格聚类结果用于展示更细空间层面的统计精度。城市聚类数过少或协方差不可识别时，p 值和置信区间写为 NA，并在报告中保留支持信息。

### 7.5 交错处理和 Sun and Abraham 敏感性

5,048 个站点具有不同开通时间。TWFE 事件研究在处理效应随 cohort 或事件期变化时，可能把不同 cohort 的处理前后观测互相作为比较基准。

R 参考入口 `scripts/causal_r/run_event_study_matching.R` 使用 `fixest::sunab` 估计 cohort 与事件时间的交互加权效应：

```text
Y_it = α_i + γ_t + Σ_(g,h) β_g,h^IW * 1(G_i=g, event_time=h) + ε_it
```

该结果用于交错处理下的异质性敏感性分析。报告需要同时保存：

```text
TWFE coefficients
Sun-Abraham coefficients
clean pre lead tests
reference period
cohort support
cluster definition
```

Matching 主路径的逐网格标签不使用 TWFE 系数作为标签。TWFE 和 Sun-Abraham 结果属于队列层诊断和动态效应汇总。

### 7.6 GSC 和 MC 的事件路径诊断

GSC 和 MC 直接生成目标网格的 gap 路径：

```text
gap_i,k,h = Y_i,k,G_i+h - Y_hat_i,k,G_i+h(0)
```

项目把 gap 按以下键分开汇总：

```text
frequency × outcome_family × outcome × method × donor_scope × event_time
```

平均路径为：

```text
mean_gap_h = (1 / N_h) * Σ_i gap_i,h
```

对于任务带有方法层标准误的情形，汇总方差为：

```text
within_var_h = mean(SE_i,h^2) / N_h
between_var_h = var(gap_i,h)
SE_h = sqrt(within_var_h + between_var_h / N_h)
```

没有任务层标准误时，使用 `sd(gap_i,h) / sqrt(N_h)`。处理前联合检验先在每个网格或城市内部对选定 clean pre 期取均值，再在聚类层级上做 one-sample t 检验。

主 Python 聚合默认使用每个任务最近 5 个清洁处理前事件期。月度主窗口下这通常对应 `-11:-7`，不包含 `-6:-1` anticipation。

## 8. 平行趋势如何检验

### 8.1 Matching 的检验步骤

1. 固定处理清单和 donor 风险集；
2. 删除开通边界和 anticipation 期；
3. 在 matched treated-control panel 上估计 TWFE；
4. 记录每个 clean pre lead 的 \(\widehat\beta_h\)、标准误、p 值和 95% 区间；
5. 对全部 clean pre lead 做联合 Wald/F 检验；
6. 分别使用 grid cluster 和 city cluster；
7. 检查每期 treated、control 和 city 支持数量；
8. 用 Sun-Abraham 作为交错处理敏感性；
9. 将结果与 holdout、placebo、空间半径和 anticipation 敏感性联合归档。

### 8.2 GSC 和 MC 的检验步骤

1. 保存每个任务的处理前 observed、counterfactual 和 gap；
2. 按事件时间对齐 gap；
3. 选择最近 5 个 clean pre 期，或按指定 `min_pre_event_time` 使用固定窗口；
4. 对每个网格取 clean pre gap 均值；
5. 在网格层面检验均值是否为 0；
6. 先把网格 gap 聚合到城市，再在城市层面检验均值是否为 0；
7. 记录 pre-RMSPE、有效任务数、有效城市数和方法层标准误；
8. 将 GSC 的因子数和 MC 的 lambda 与处理前预测误差一起报告。

### 8.3 检验统计量

对网格级或城市级聚类 g=1,...,G，令 \(\bar L_g^{pre}\) 是该聚类在 clean pre 期的平均 gap：

```text
mean_pre = (1 / G) * Σ_g Lbar_g^pre
sd_pre = sd({Lbar_g^pre})
t = mean_pre / (sd_pre / sqrt(G))
```

自由度为 `G-1`，双侧 p 值为：

```text
p = 2 * P(T_(G-1) >= |t|)
```

当 `G<2`、`sd_pre` 不可用或 `sd_pre=0` 时，统计量写为 NA。诊断表保留 `n_pre_observations`，用于区分时间观测数量和独立聚类数量。

### 8.4 结果判读

| 结果 | 计量含义 | 报告方式 |
|---|---|---|
| clean pre lead 联合检验未拒绝 0 | 当前样本和规格中没有发现明显处理前偏离 | 报告统计量、p 值、自由度和支持数量 |
| clean pre lead 联合检验拒绝 0 | 处理前路径存在系统偏离 | 标记趋势诊断红旗，检查窗口、空间污染、cohort 和竞争事件 |
| city cluster 拒绝，grid cluster 未拒绝 | 城市共同冲击可能驱动网格层结果 | 主解释采用城市聚类，并报告两套结果 |
| p 值为 NA | 聚类或方差支持不足 | 保留任务，不生成可用的统计推断字段 |
| pre-RMSPE 较大 | 反事实路径在处理前拟合较差 | 降低结果可信度，检查 donor 和方法参数 |

未拒绝 0 表示检验没有发现明显偏离。该结果的解释范围由样本、窗口、聚类数、统计功效和反事实模型共同决定。

## 9. 统计推断和多重检验

### 9.1 单期和联合推断

事件研究通常包含多个 lead 和 lag。单期 5% 显著性会产生多重检验问题。项目报告使用以下层次：

1. 动态系数图和 95% 置信区间；
2. 所有 clean pre lead 的联合检验；
3. 关键处理后期的单期结果；
4. Sun-Abraham、空间半径和 anticipation 敏感性；
5. 方法和结果族分开汇总。

结果表需要标注探索性系数和预先规定的主要事件期，例如月度 `h=1,3,6,12,18,24`。完整逐月路径的单期显著性与联合检验分别报告。

### 9.2 置信区间的计算

TWFE 使用聚类协方差和 t 分布临界值。GSC bootstrap 和 MC jackknife 使用方法层标准误，再使用正态近似生成区间。每个结果记录：

```text
estimate
standard_error
confidence_lower
confidence_upper
p_value
inference_method
requested_repetitions
valid_repetitions
```

结果中 `requested_repetitions` 不能替代 `valid_repetitions`。收敛不足时期的推断字段为 NA。

### 9.3 城市聚类数和自由度

城市聚类反映处理实施层级，有效城市数决定有限样本自由度。城市数较少时，报告应优先展示置信区间、系数大小和支持数量，并把 p 值作为辅助信息。网格聚类数量很大时，独立政策数量仍由城市和处理事件决定。

## 10. 稳健性和安慰剂设计

### 10.1 时间窗口

预先规定：

```text
anticipation_months = 0, 6, 12
monthly_pre_window = 24, 36, 48 months
annual_anticipation_years = 0, 1
```

每个规格重新生成事件日历、控制特征和标签。窗口改变后，控制身份、支持数量和方法路由可能发生变化，最终回归窗口也要同步更新。

### 10.2 空间排除

主规格为 1km，敏感性为 1.5km 和 2km。比较内容包括：

```text
n_treated
n_donor
same_city donor support
cross_city fallback rate
pre-fit quality
event-study path
city-cluster inference
```

### 10.3 donor 范围

同城 donor 保留城市内经济和规划环境。跨城 donor 先做城市内标准化，再单独估计。两个 donor scope 的结果不能合并成一个无条件平均值。

### 10.4 结果和变量集

按结果族和变换尺度分别进行：

- housing 的 median 和 hedonic 价格测度；
- VIIRS 的 `asinh` 结果和有效观察支持；
- POI 的计数、比例和类别熵；
- population 的 `log1p` 结果；
- 移除区位特征或轨道特征后的控制设计；
- 不同交易数量阈值下的 housing 结果。

### 10.5 处理属性异质性

R 事件研究入口按处理属性分层：

```text
transfer / non_transfer
new_line / existing_line
terminal / non_terminal
small_batch / large_batch
```

分层模型保留同一控制网格规则和事件时间定义。每层至少需要 3 个 treated events 才输出联合趋势统计量。小样本层只展示路径和支持数量。

### 10.6 Placebo

项目使用四类 placebo：

| placebo | 构造 | 检查内容 |
|---|---|---|
| donor-donor matching placebo | donor 中伪造 treated | 训练距离和 holdout 误差门槛 |
| in-time placebo | 把处理前期指定为伪开通期 | 时间趋势和预期效应 |
| masked untreated prediction | 遮蔽未处理目标的一段处理前结果 | GSC 跨城预测能力 |
| competing-event placebo | 对存在竞争交通事件的网格单独评估 | 污染和替代解释 |

Placebo 的结果用于标定设计质量和反事实预测误差。它们不创造正式处理效应。

## 11. 结果表和图形规范

### 11.1 动态效应表

每个 `frequency × outcome_family × outcome × method × donor_scope` 组合至少报告：

```text
event_time
estimate
standard_error
confidence_lower
confidence_upper
p_value
n_grids
n_cities
n_pre_observations
inference_method
specification_fingerprint
```

TWFE 另保存 `grid_clusters`、`city_clusters`、`reference_event_time`、`variance` 和 `absorption_iterations`。

GSC 另保存 `selected_rank`、CV MSPE、pre-RMSPE、bootstrap 有效次数。MC 另保存 `selected_lambda`、effective rank、CV MSPE、pre-RMSPE 和 jackknife 有效次数。

### 11.2 平行趋势表

最小字段为：

```text
frequency
outcome_family
outcome
method
donor_scope
pre_window
anticipation_window
n_grids
n_cities
n_pre_observations
pre_mean
pre_sd
grid_cluster_statistic
grid_cluster_p_value
city_cluster_statistic
city_cluster_p_value
pretrend_flag
```

TWFE 表把 `statistic` 解释为 Wald/F 统计量。GSC 和 MC 表把 `statistic` 解释为聚类均值的一样本 t 统计量。字段定义不能混用。

### 11.3 图形

事件研究图应包含：

1. 横轴为相对事件期；
2. 纵轴标注结果尺度；
3. 点估计和 95% 置信区间；
4. `h=0` 的处理边界；
5. 参考期或处理前窗口说明；
6. 每期有效网格数或城市数；
7. anticipation 期和缺失期的可见标记。

Matching 的 TWFE 图显示相对于参考期的系数。GSC 和 MC 图显示直接 gap 均值，两者的纵轴含义需要在图例中区分。

## 12. 计量结论的写法

推荐按以下顺序写结果：

1. 先说明估计对象、结果尺度和样本；
2. 再说明 donor scope、方法和处理前窗口；
3. 报告处理前支持和城市聚类检验；
4. 报告处理后动态系数、置信区间和有效样本；
5. 对照空间、anticipation、donor 范围和方法敏感性；
6. 说明结论适用的结果族和事件期。

示例句式：

```text
在 housing 月度对数价格、同城 donor、清洁处理前 -42 到 -7 月和城市聚类规格下，h=1 的估计效应为 β_hat，95% CI 为 [L,U]。clean pre lead 的联合 Wald 检验为 F(q,G-1)=...，p=...。样本包含 ... 个 treated grids 和 ... 个城市。该结果表示相对于 h=-7 的动态价格差异。
```

对于 GSC 和 MC，应把“事件研究系数”改写为“反事实 gap 均值”，并说明因子数、lambda、处理前 RMSPE 和有效推断次数。

## 13. 代码和文档对应关系

| 计量内容 | 当前代码入口 | 主要测试或输出 |
|---|---|---|
| TWFE 事件研究 | [`src/urban_intervention/causal/event_study.py`](../../src/urban_intervention/causal/event_study.py) | `tests/unit/test_python_event_study.py` |
| GSC/MC 路径汇总 | [`src/urban_intervention/causal/pooled_event_study.py`](../../src/urban_intervention/causal/pooled_event_study.py) | `tests/unit/test_pooled_event_study.py` |
| R 事件研究 | [`scripts/causal_r/run_event_study_matching.R`](../../scripts/causal_r/run_event_study_matching.R) | `parallel_trends_wald*.csv` |
| R 路径汇总 | [`scripts/causal_r/event_study_lib.R`](../../scripts/causal_r/event_study_lib.R) | `pretrend_grid_cluster.csv`、`pretrend_city_cluster.csv` |
| GPU Matching | [`src/urban_intervention/causal/gpu/matching.py`](../../src/urban_intervention/causal/gpu/matching.py) | `tests/unit/test_gpu_matching.py` |
| 完整 ATT Matching | [`src/urban_intervention/causal/gpu/abadie_imbens.py`](../../src/urban_intervention/causal/gpu/abadie_imbens.py) | `tests/unit/test_gpu_abadie_imbens.py` |
| 控制设计 | [`src/urban_intervention/causal/gpu/control_design.py`](../../src/urban_intervention/causal/gpu/control_design.py) | 控制队列和来源记录 |
| GPU GSC | [`src/urban_intervention/causal/gpu/gsc.py`](../../src/urban_intervention/causal/gpu/gsc.py) | `tests/unit/test_gpu_gsc.py` |
| GPU MC | [`src/urban_intervention/causal/gpu/matrix_completion.py`](../../src/urban_intervention/causal/gpu/matrix_completion.py) | `tests/unit/test_gpu_matrix_completion.py` |
| 方法推断 | [`src/urban_intervention/causal/gpu/inference.py`](../../src/urban_intervention/causal/gpu/inference.py) | `tests/unit/test_gpu_inference.py` |
| 任务标签入口 | [`scripts/causal_python/run_causal_label_queue.py`](../../scripts/causal_python/run_causal_label_queue.py) | Response Artifact 校验 |

## 14. 当前实施边界

### 14.1 已实现的计量组件

- Python TWFE 事件研究包含单位和时期固定效应、grid/city 两种聚类、单期 t 检验和联合 Wald/F 检验；
- R 事件研究包含 `fixest::feols` 和 Sun-Abraham 敏感性路径；
- Python GSC 包含 rolling CV、因子数选择、目标处理前载荷拟合和 bootstrap 路径；
- Python MC 包含 rolling CV、lambda 选择、核范数补全和单位 jackknife；
- Python Matching 包含共同支持、M=5 候选、静态精炼和 donor-donor q95 门禁；
- 完整 Abadie and Imbens ATT 包含偏差修正和解析方差；
- GSC/MC 路径聚合包含网格级和城市级处理前一样本 t 检验。

### 14.2 发布前仍需完成的验证

Python 单元测试和静态检查已经覆盖当前纯 Python 实现。R 运行时、四张 RTX 4090 的服务器资格运行、R/Python 数值一致性和真实生产队列仍需要在服务器副本中完成。

当前 active 冻结规格仍需要通过启动校验。若 active 文件中的 `minimum_complete_families` 与代码的覆盖优先规则不一致，生产入口会停止，需先更新冻结规格或恢复代码参数，再运行正式队列。active 数据和 `outputs/viirs_monthly/` 缓存保持只读。

## 15. 文献依据

1. Abadie, A. and Imbens, G. W. (2006). Large Sample Properties of Matching Estimators for Average Treatment Effects. *Econometrica*, 74(1), 235-267.
2. Abadie, A. and Imbens, G. W. (2011). Bias-Corrected Matching Estimators for Average Treatment Effects. *Journal of Business and Economic Statistics*, 29(1), 1-11.
3. Xu, Y. (2017). Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models. *Political Analysis*, 25(1), 57-76.
4. Athey, S., Bayati, M., Doudchenko, N., Imbens, G. and Khosravi, K. (2021). Matrix Completion Methods for Causal Panel Data Models. *Journal of the American Statistical Association*, 116(536), 1716-1730.
5. Sun, L. and Abraham, S. (2021). Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects. *Journal of Econometrics*, 225(2), 175-199.
6. Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing. *Econometrica*, 89(5), 2387-2421.
7. Roth, J. (2022). Pretest with Caution: Event-Study Estimates After Testing for Parallel Trends. *AEA Papers and Proceedings*, 112, 433-439.
8. Rambachan, A. and Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*, 90(5), 2555-2591.
9. Abadie, A., Athey, S., Imbens, G. W. and Wooldridge, J. M. (2023). When Should You Adjust Standard Errors for Clustering? *Quarterly Journal of Economics*, 138(1), 1-35.
10. de Chaisemartin, C. and D'Haultfœuille, X. (2020). Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects. *American Economic Review*, 110(9), 2964-2996.

## 16. 复核清单

每次正式结果发布前，按以下顺序复核：

1. 处理清单、开通期和空间风险集的哈希是否与冻结规格一致；
2. 结果变换、时间窗口和 anticipation 是否写入 manifest；
3. 控制设计是否只读取处理前信息；
4. 匹配是否通过共同支持、训练距离、holdout 和 placebo 门禁；
5. GSC 的因子数和 MC 的 lambda 是否来自当前面板的 rolling CV；
6. pre-RMSPE、有效处理前观测和 donor 支持是否达到规格要求；
7. TWFE 是否保存两种聚类结果和联合 lead 检验；
8. GSC/MC 是否保存网格级与城市级处理前检验；
9. 每个置信区间是否有对应的推断方法和有效重复次数；
10. Matching、GSC、MC 是否分开汇总；
11. Response Artifact 是否包含输入哈希、规格指纹、方法、donor scope 和失败原因；
12. 服务器四张 RTX 4090 的资格记录和生产运行日志是否完整。
