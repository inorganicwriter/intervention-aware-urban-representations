# DDR-004：因果响应标签生产规范

状态：frozen

日期：2026-07-23

实现修订：2026-09-04；2026-09-19（队列状态与质量排序澄清）

## 1. 目标

估计对象是处理网格 `i`、结果族 `k`、事件期 `h` 的局部因果响应标签：

`L[i,k,h] = Y_observed[i,k,h] - Y_counterfactual[i,k,h]`。

处理单元固定为 5,048 个含正式地铁站点的 500m × 500m 网格。房价、VIIRS、POI、人口
是不同结果族，允许拥有不同的可信控制组和不同的缺失掩码。

## 2. 时间合同

- 精确开通日保留为来源信息，但月度分析以日历月为单位。
- 开通当月是正式冲击期 `event_time = 0`，必须读取真实当月观测、估计、保存并绘制；
  不得删除、插值或用 `event_time = 1` 代替。
- 四类结果的主事件研究统一省略 `event_time = -1`；月度主规格因此使用
  `anticipation_months = 0`。
- housing/VIIRS 另做独立月度稳健性：省略 `event_time = -7`，把 `-6:-1`
  显示为可能的提前反应，clean-pre 检验只使用 `event_time < -7`。
- 月度主处理前窗口为参考期及其之前的 36 个模型期（主规格 `-36:-1`）；正式路径
  包含 `t=0` 与 1-24 月，正向标签重点汇总 1、3、6、12、18、24 月。主规格使用原始
  逐月 `observation_window = 1`。
- 年度路径包含开通年 `event_time = 0` 与 1-3 年，并统一省略 `event_time = -1`；
  `annual_anticipation_years = 0`。
- anticipation 或 observation window 改变时必须重新拟合并写入独立 queue、task root、
  estimator root 和 manifest，不能把不同参考期或窗口的结果混合。

## 3. 控制组与信息边界

控制组选择只读取处理发生前可观测的信息。处理后结果值、处理后结果是否缺失、处理后显著性
和最终标签均保持在 donor 纳入、距离、共同支持和阈值校准之外。

设计和估计必须分成两个持久化阶段：

1. `pre-only design`：冻结风险集、共同支持、控制身份或 GSC donor 身份；
2. `label estimation`：在冻结设计后读取处理后结果并计算标签。

自动测试必须证明：任意修改处理后数值或缺失状态，均不会改变第一阶段选出的控制身份。

主匹配质量门禁把最近一个干净处理前观测块留作 holdout；控制身份只用更早的处理前轨迹
确定。对同一风险集中的 donor 进行确定性的 donor-donor 伪处理匹配，以其训练距离、holdout
RMS 标准化差异和最大绝对标准化差异的第 95 百分位作为门槛。目标对三项均处于门槛以内时
进入 matched-label 路径，其他目标路由到 GSC。门槛校准只使用处理前信息。

## 4. 三条标签路径

### 4.1 单一控制组（matched change-in-changes）

使用正式最近邻匹配设计选出控制网格 `j` 后：

`L[i,k,h] = (Y[i,k,h] - B[i,k]) - (Y[j,k,h] - B[j,k])`，

其中 `B` 是预先冻结的处理前基期或处理前均值。匹配设计遵循 Abadie and Imbens 最近邻匹配
的共同支持、允许放回和 Mahalanobis 距离规范。完整 cohort ATT 的偏差修正和解析方差由
R `Matching::Match` 单独报告；测试版距离函数的结果单独标记。

### 4.2 Generalized synthetic control

若没有达到预先冻结质量标准的单一控制组，则使用 Xu (2017) 的 interactive fixed-effects
generalized synthetic control：

`L[i,k,h] = Y[i,k,h] - Y_hat_0[i,k,h]`。

正式 Python/GPU 实现与 R `gsynth` 参考合同对齐，使用 two-way effects、处理前交叉验证
选择因子数、`r=0:5`、`min.T0=5` 和 200 次 parametric bootstrap。R `gsynth` 仅用于
资格/参考比较。donor 身份只由处理前支持决定；处理后缺失可使某一期标签缺失或使模型
失败，donor 身份保持冻结。

## 5. 输出主键与字段

主键：`treatment_order × outcome_family × outcome × event_time × specification_id`。

必需字段包括：处理网格、开通月、观测值、反事实值、标签值、原始/变换尺度、方法、
控制网格或 donor 权重/集合、匹配距离、共同支持、处理前 RMSPE、placebo/held-out 诊断、
标准误或区间（若方法可识别）、质量等级、`label_available`、失败原因和代码/数据版本。

接近零或统计不显著的标签仍然有效，显著性只用于报告。输入支持不足、共同支持失败、
预拟合不合格、估计器失败或目标期缺失时设置缺失掩码。

## 6. 路由和队列安全

合法状态转换为：

`pending → matching_running → matched_labelled`；或  
`pending → matching_running → gsc_pending → gsc_running → gsc_labelled`；或  
`pending → matching_running → gsc_pending → gsc_running → (GSC fail) → mc_pending → mc_running → mc_labelled`；或  
`… → skipped`（匹配、GSC 与 MC 均失败并记录结构化原因）。

同城三轮回退失败后，跨城扩展按 `cross_matching_running → cross_gsc_running →
cross_mc_running` 继续，跨城 matched 标签记为 `matched_labelled`。

MC 回退仅在 GSC 失败后触发，不参与匹配失败后的直接路由；结构性支持失败（如处理后
支持不足或结果面板缺失）直接记为 `skipped`，不进入 MC。

质量等级采用同城优先排序：在同一 donor scope 内 MC 低于 matched 和 GSC；但任一
same-city 路径（含 MC）高于任一 cross-city 路径，且 `mc_*_pass` 高于
`mc_*_minimal_pre_support`。质量等级只用于报告和样本索引分层，训练掩码由
`label_available`、生产合格性和不确定性支持共同决定。

队列和结果必须通过临时文件原子替换；每一任务目录必须包含 manifest。重新运行已完成任务
时复用现有结果，保持记录一致。生产队列只有在合成真值、处理后泄漏、事件时间、真实数据、失败
路由和中断恢复六类门禁全部通过后才能解除冻结。

## 7. 方法依据

- Imai, Kim, and Wang (2023), *Matching Methods for Causal Inference with Time-Series Cross-Sectional Data*.
- Abadie and Imbens (2006), *Large Sample Properties of Matching Estimators for Average Treatment Effects*.
- Abadie and Imbens (2011), *Bias-Corrected Matching Estimators for Average Treatment Effects*.
- Xu (2017), *Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models*.
- Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), *Matrix Completion Methods for Causal Panel Data Models*, JASA.
- Ratledge et al. (2022), *Using Machine Learning to Assess the Livelihood Impact of Electricity Access*, Nature.
