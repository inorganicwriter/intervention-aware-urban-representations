# 计量公式索引

更新：2026-08-28
状态：公式导航页

完整的估计对象、识别假设、参数含义、推断规则和结果解释见
[`econometric_methods.md`](econometric_methods.md)。本页只列出项目使用的核心公式和代码入口，便于从公式定位到方法章节。

## 1. 因果对象和响应标签

| 对象 | 公式 | 详细章节 |
|---|---|---|
| 潜在结果 | \(Y_{ikt}=D_{it}Y_{ikt}(1)+(1-D_{it})Y_{ikt}(0)\) | [1.2](econometric_methods.md#12-潜在结果) |
| 动态效应 | \(τ_{ikh}=Y_{ik,G_i+h}(1)-Y_{ik,G_i+h}(0)\) | [1.3](econometric_methods.md#13-动态处理效应) |
| 响应标签 | \(L_{ikh}=Y_{ik,G_i+h}-\widehat Y_{ik,G_i+h}(0)\) | [1.3](econometric_methods.md#13-动态处理效应) |
| 队列均值 | \(μ_{kh}=N_{kh}^{-1}\sum_i L_{ikh}\) | [1.4](econometric_methods.md#14-标签和-att-的区别) |

代码入口：[`src/urban_intervention/causal/response_artifact.py`](../../src/urban_intervention/causal/response_artifact.py)。

## 2. Matching

| 对象 | 公式 | 详细章节 |
|---|---|---|
| Mahalanobis 距离 | \(d_{ij}=\sqrt{(X_i-X_j)'\widehat\Sigma^{-1}(X_i-X_j)}\) | [4.3](econometric_methods.md#43-mahalanobis-距离) |
| 条件支持 | \(\min_jX_{jm}\le X_{im}\le\max_jX_{jm}\) | [3.5](econometric_methods.md#35-共同支持和正值概率) |
| 匹配标签 | \(L_{ikh}=(Y_{i,h}-B_i)-(Y_{j(i),h}-B_{j(i)})\) | [4.6](econometric_methods.md#46-匹配标签) |
| ATT | \(ATT=N_T^{-1}\sum_i[\Delta Y_i-\sum_jw_{ij}\Delta Y_j]\) | [4.7](econometric_methods.md#47-完整-abadie-and-imbens-att) |
| 偏差修正 ATT | \(ATT_{adj}=N_T^{-1}\sum_i[\Delta Y_i-\Delta Y_{j(i)}+(X_i-X_{j(i)})'\widehat\beta]\) | [4.7](econometric_methods.md#47-完整-abadie-and-imbens-att) |

代码入口：[`src/urban_intervention/causal/gpu/matching.py`](../../src/urban_intervention/causal/gpu/matching.py)、[`src/urban_intervention/causal/gpu/abadie_imbens.py`](../../src/urban_intervention/causal/gpu/abadie_imbens.py)。

## 3. GSC

| 对象 | 公式 | 详细章节 |
|---|---|---|
| 未处理因子结构 | \(Y_{it}(0)=\alpha_i+\delta_t+\lambda_i'F_t+\epsilon_{it}\) | [3.6](econometric_methods.md#36-gsc-的交互固定效应结构) |
| 目标拟合 | \(Y_{i,t}=a_i+F_t'b_i+\epsilon_{i,t}\) | [5.2](econometric_methods.md#52-交互固定效应模型) |
| GSC 标签 | \(L_{i,t}=Y_{i,t}-\widehat Y_{i,t}(0)\) | [5.2](econometric_methods.md#52-交互固定效应模型) |
| CV 误差 | \(MSPE(r)=|S|^{-1}\sum_{(i,t)\in S}[Y_{it}-\widehat Y_{it}(r)]^2\) | [5.3](econometric_methods.md#53-因子数交叉验证) |
| 处理前 RMSPE | \(RMSPE_i=\sqrt{mean[(Y_{i,t}-\widehat Y_{i,t}(0))^2]}\) | [5.5](econometric_methods.md#55-gsc-诊断) |
| bootstrap 标准误 | \(SE_t=sd(\{\widehat τ_t^{(b)}\})\) | [5.4](econometric_methods.md#54-gsc-不确定性) |

代码入口：[`src/urban_intervention/causal/gpu/gsc.py`](../../src/urban_intervention/causal/gpu/gsc.py)。

## 4. MC

| 对象 | 公式 | 详细章节 |
|---|---|---|
| 低秩分解 | \(Y(0)=L+E\) | [3.7](econometric_methods.md#37-mc-的低秩结构) |
| 核范数目标 | \(\min_L |\Omega|^{-1}\sum_{(i,t)\in\Omega}(Y_{it}-L_{it})^2+\lambda\|L\|_*\) | [3.7](econometric_methods.md#37-mc-的低秩结构) |
| MC 标签 | \(L_{i,t}=Y_{i,t}-\widehat L_{i,t}\) | [6.1](econometric_methods.md#61-矩阵结构) |
| CV 误差 | \(MSPE(\lambda)=|S|^{-1}\sum_{(i,t)\in S}[Y_{it}-\widehat Y_{it}(\lambda)]^2\) | [6.2](econometric_methods.md#62-lambda-交叉验证) |
| Jackknife 伪值 | \(p_t^{(-j)}=N\widehat τ_t-(N-1)\widehat τ_t^{(-j)}\) | [6.3](econometric_methods.md#63-mc-标签和-jackknife) |
| Jackknife 标准误 | \(SE_t^{JK}=\sqrt{var(p_t^{(-j)})/n_{valid,t}}\) | [6.3](econometric_methods.md#63-mc-标签和-jackknife) |

代码入口：[`src/urban_intervention/causal/gpu/matrix_completion.py`](../../src/urban_intervention/causal/gpu/matrix_completion.py)。

## 5. DID 和事件研究

| 对象 | 公式 | 详细章节 |
|---|---|---|
| TWFE 事件研究 | \(Y_{it}=\alpha_i+\gamma_t+\sum_{h\ne h_0}\beta_h[Treated_i\times1(event\_time=h)]+\epsilon_{it}\) | [7.1](econometric_methods.md#71-匹配样本上的-twfe) |
| 清洁 pre 原假设 | \(H_0:\beta_h=0\) for every clean \(h<0,h\ne h_0\) | [7.3](econometric_methods.md#73-平行趋势联合-wald-检验) |
| Wald/F 统计量 | \(F=q^{-1}\widehat\beta_{pre}'\widehat V_{pre}^{-1}\widehat\beta_{pre}\) | [7.3](econometric_methods.md#73-平行趋势联合-wald-检验) |
| 聚类协方差 | \(\widehat V=correction(X'X)^{-1}[\sum_gX_g'e_ge_g'X_g](X'X)^{-1}\) | [7.4](econometric_methods.md#74-聚类协方差) |
| GSC/MC pooled gap | \(\widehat μ_h=N_h^{-1}\sum_i gap_{i,h}\) | [7.6](econometric_methods.md#76-gsc-和-mc-的事件路径诊断) |
| pooled 标准误 | \(SE_h=\sqrt{mean(SE_{i,h}^2)/N_h+var(gap_{i,h})/N_h}\) | [7.6](econometric_methods.md#76-gsc-和-mc-的事件路径诊断) |

代码入口：[`src/urban_intervention/causal/event_study.py`](../../src/urban_intervention/causal/event_study.py)、[`src/urban_intervention/causal/pooled_event_study.py`](../../src/urban_intervention/causal/pooled_event_study.py)。

## 6. 统计推断

| 推断类型 | 标准误来源 | 区间和 p 值 |
|---|---|---|
| TWFE grid/city cluster | CRV1，聚类数减 1 的 t 分布 | 系数 ± t 临界值 × SE |
| GSC | 200 次参数 bootstrap 的有效路径 | 正态近似 |
| MC | donor 单位 jackknife 伪值 | 正态近似 |
| pooled GSC/MC | 任务层 within 方差和网格间 between 方差 | 95% 正态区间 |

完整的有效重复次数、聚类层级和多重检验要求见
[`econometric_methods.md`](econometric_methods.md#9-统计推断和多重检验)。

## 7. 实现索引

| 内容 | 文件 |
|---|---|
| 事件研究窗口和参考期 | [`src/urban_intervention/causal/event_study.py`](../../src/urban_intervention/causal/event_study.py) |
| pooled pretrend t 检验 | [`src/urban_intervention/causal/pooled_event_study.py`](../../src/urban_intervention/causal/pooled_event_study.py) |
| R Sun-Abraham 参考 | [`scripts/causal_r/run_event_study_matching.R`](../../scripts/causal_r/run_event_study_matching.R) |
| Matching 设计门禁 | [`src/urban_intervention/causal/gpu/control_design.py`](../../src/urban_intervention/causal/gpu/control_design.py) |
| 推断对象封装 | [`src/urban_intervention/causal/gpu/inference.py`](../../src/urban_intervention/causal/gpu/inference.py) |
