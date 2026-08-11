# 计量模型公式（Estimator Formulas）

本文档集中项目全部计量模型的公式与文献依据，作为论文 Methods 的底稿。
每节给出公式、参数定义、实现位置。三路径（匹配 / GSC / MC）的估计量独立，
互不混合（DDR-003/004）。

---

## 1. 核心定义：响应标签

每个处理网格 i 在结果族 f、事件时间 h 的因果响应标签：

```
causal_response_label(i, f, h) = observed(i, f, h) − counterfactual(i, f, h)
```

- observed：处理网格实际观测值（水平或差分，见下）
- counterfactual：由三路径之一构造的反事实

实现：`scripts/causal_r/complete_estimators_lib.R`（pair_change_labels /
normalize_gsc_labels）、`run_complete_mc.R`。

---

## 2. 匹配路径（Abadie–Imbens 最近邻 + DiD 差分）

### 2.1 匹配距离（Mahalanobis）

对处理前协变量向量 X（匹配特征，见 DDR-004）：

```
d_M(x_t, x_c) = (x_t − x_c)' Σ⁻¹ (x_t − x_c)
```

- Σ：donor 协变量协方差矩阵（`active_matching_matrix` 稳定化逆）
- M = 1 最近邻，replace = TRUE，ties = FALSE

### 2.2 共同支持

处理网格必须落在 donor 特征凸包内（逐协变量 min/max 检查）：
```
x_t ∈ [min_donor(X_j), max_donor(X_j)]  ∀ j
```

### 2.3 响应标签（配对差分）

```
ΔY_it = Y_i(post_time) − Y_i(baseline_time)      # attach_prepost_outcome
label_i = ΔY_treated − ΔY_control                # pair_change_labels
```

### 2.4 聚合 ATT（Abadie–Imbens 偏差校正）

```
ATT = (1/N_t) Σ_i [ ΔY_treated,i − ΔY_control,i ]
```
偏差校正项（BiasAdjust）：
```
ATT_adj = ATT + (1/N_t) Σ_i (X_t,i − X_c,i)' β̂
```
其中 β̂ 为控制组 ΔY 对 X 的 OLS 系数。

实现：`run_complete_abadie_imbens.R`（`Matching::Match`）。

---

## 3. Xu (2017) 广义合成控制（GSC）

交互固定效应模型：

```
Y_it = D_it τ + X_it' β + λ_i' F_t + ε_it
```

- λ_i：单位因子载荷；F_t：公共因子（r 维，CV 选择 r ∈ 0..5）
- 反事实：处理单元 i 在 t > T0 的 Ŷ_ct = X_it' β̂ + λ̂_i' F̂_t
- 响应标签：label_it = Y_it − Ŷ_ct
- 推断：200 次参数 bootstrap（单位重抽样），得到每期 SE/CI

实现：`run_complete_xu_gsc.R`（`gsynth`）、`normalize_gsc_labels`。

---

## 4. Athey et al. (2021) 矩阵补全（MC）

潜在因子面板（矩阵补全）模型：

```
Y_it = L_it + ε_it,   L_it 低秩
```
用 `fect(method="mc")` 估计，λ 由 MSPE 交叉验证选择；反事实为
补全矩阵在 (i, t > T0) 的元素；推断为 200 次 bootstrap。

实现：`run_complete_mc.R`。

---

## 5. 事件研究（平行趋势验证）

### 5.1 匹配路径：TWFE 回归

```
Y_it = Σ_{k∈K} β_k · D_it^k + α_i + γ_t + ε_it
```
- D_it^k：事件时间 k 的虚拟变量（k = 事件月 − 开通月），基期 k = −1 省略
- α_i：网格固定效应；γ_t：日历月固定效应
- 标准误按网格聚类
- 平行趋势：H₀: β_k = 0 ∀ k < 0（联合 Wald 检验）

### 5.2 匹配路径：Sun–Abraham (2021) 异质性稳健

```
Y_it = Σ_k β_k^IW · 1(k, cohort) 加权交互估计（fixest::sunab）
```
对交错处理下的异质性偏误稳健（Goodman-Bacon 2021）。

实现：`run_event_study_matching.R`。

### 5.3 GSC/MC 路径：counterfactual gap 聚合

对 GSC/MC 的 est.att 序列（模型自身的事件研究系数）聚合：
```
mean_k = (1/N) Σ_i label_i,k
SE_k = √( within_var_k + between_var_k / N )
within_var_k  = mean(SE_i,k²)/N   （bootstrap SE 聚合）
between_var_k = var(grid_mean_k)   （网格间方差）
```
联合零 pre-trend：网格级均值 one-sample t 检验。

实现：`event_study_lib.R`。

---

## 6. 匹配质量诊断：SMD

标准化均值差（协变量 j）：

```
SMD_j = (x̄_treated,j − x̄_control,j) / SD_donor,j
```
实现：`pair_preonly_diagnostics`、`build_balance_loveplot.py`。

---

## 7. 六轮路由

```
同城匹配 → 同城 GSC → 同城 MC → 跨城匹配 → 跨城 GSC → 跨城 MC → skip
```
实现：`grid_control_design_lib.R`、`run_causal_label_queue.py`。

---

## 8. 聚类与推断有效性

地铁开通是城市级政策，同一城市网格共享城市冲击。事件研究同时报告：

- 网格聚类（主）：SE 按 grid_id 聚类
- 城市聚类（稳健）：SE 按 city_key 聚类（Abadie et al. 2023）

GSC/MC 聚合的联合 pre-trend 检验提供网格级与城市级两个版本。

## 9. Spillover 异质性

按同期开通站数（`stations_opened_same_month`）中位数分两层
（small / large），分别估计 Sun-Abraham 事件研究：

```
Y_it = Σ_k β_k^IW(stratum) · 1(event_time=k) + α_i + γ_t + ε_it
```
两层的 β_k 差异提示网络效应/空间溢出（Yu et al. 2013）。

---

## 文献

1. Abadie, A. & Imbens, G.W. (2006). Large sample properties of matching estimators. *Econometrica* 74(1):235-267.
2. Abadie, A. & Imbens, G.W. (2011). Bias-corrected matching estimators. *Journal of Business & Economic Statistics* 29(1):1-11.
3. Xu, Y. (2017). Generalized synthetic control method. *Political Analysis* 25(1):57-76.
4. Athey, S., Bayati, M., Doudchenko, N., Imbens, G. & Khosravi, K. (2021). Matrix completion methods for causal panel data models. *JASA* 116(536):1716-1730.
5. Sun, L. & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics* 225(2):175-199.
6. Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Econometrica* 89(5):2387-2421.
7. Roth, J. (2022). Pretest with caution: Event-study estimates after testing for parallel trends. *AER: Insights* 4(3):305-322.
8. Abadie, A., Athey, S., Imbens, G.W. & Wooldridge, J.M. (2023). When should you adjust standard errors for clustering? *QJE* 138(1):1-35.
9. de Chaisemartin, C. & D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *AER* 110(9):2964-2996.
10. Yu, N., de Jong, M., Storm, S. & Mi, J. (2013). Spatial spillover effects of transport infrastructure: evidence from Chinese regions. *Journal of Transport Geography* 29:56-66.
