# 城市轨道交通干预的反事实与响应标签完整设计

状态：项目级主设计（authoritative design）  
日期：2026-07-23  
适用范围：5,048 个地铁站点处理网格的控制组寻找、反事实构造和响应标签生成

## 1. 文档目的与解释优先级

本文件统一规定当前项目的研究对象、经济学识别逻辑、匹配方法、Generalized Synthetic
Control（GSC）回退路径、标签定义、质量门禁、失败规则、软件输出和正式执行顺序。

本文件是当前唯一的主动因果设计。对应 DDR 用于冻结关键决定；测试说明只解释实现，历史输出
只能作为诊断，不能反向决定本设计。

本设计的最终目的不是报告一个总体 DID 系数，而是为表示学习构造带有质量和不确定性信息的
局部干预响应标签。

## 2. 项目研究问题

项目题目为：

> Learning How Cities Respond: Intervention-Conditioned Urban Representations

核心问题是：给定一个地点在干预前的多模态城市状态，能否学习一种表示，使得在相同地铁
干预下响应相似的地点在表示空间中接近，而不只是让视觉或功能相似的地点接近。

因此需要先为已有地铁开通事件构造尽可能可信的局部反事实：如果该地点没有开通地铁，其
房价、夜间活动、POI 和人口会如何变化。观测路径与反事实路径之差构成响应标签。

## 3. 因果对象与标签对象

### 3.1 处理单元

- 空间单元为固定的 500m × 500m 城市网格。
- 站点坐标落入网格多边形时，该网格成为候选处理单元。
- 当前冻结的处理单元清单共有 5,048 个唯一网格。
- 站点别名、同物理站点记录、跨市归属和竞争交通事件已经过决议表处理。
- 一个网格只能在冻结清单中出现一次；站点记录数不等于处理网格数。

### 3.2 干预

主干预为网格内正式运营地铁站点的开通。云巴、有轨电车等不是主地铁处理，但作为竞争
干预记录，并可用于删失或敏感性分析。

### 3.3 最终标签

对处理网格 `i`、结果变量 `k` 和事件期 `h`：

```text
L[i,k,h] = Y_observed[i,k,h] - Y_counterfactual[i,k,h]
```

标签是特定设计下的因果响应证据，不是真实个体处理效应的无误差观测。每条标签必须同时
保存方法、处理前拟合质量、不确定性、支持范围和失败/缺失掩码。

## 4. 空间风险集

### 4.1 处理网格

主分析仅使用冻结的 5,048 个站点所在网格。将周围 8 个相邻网格扩展为处理区域属于后续
空间扩展，不进入第一轮标签生产。

### 4.2 非实验 donor universe

候选 donor 必须同时满足：

1. 不属于 5,048 个实验网格；
2. 无已知站点事件污染；
3. 满足空间排除规则；
4. 在目标事件的处理前窗口中保持未处理；
5. 不使用任何处理后结果决定是否进入 donor pool。

当前空间审计得到约 3,771,800 个合格非实验网格。

### 4.3 空间排除半径

- 主规格：站点点位到 donor 网格多边形的最短投影距离至少 1km；
- 稳健性规格：1.5km 和 2km；
- 不使用网格中心距离替代点到多边形最短距离；
- 后续站点或竞争交通进入排除范围时，对相应后处理期进行删失或污染标记。

### 4.4 城市范围

控制搜索采用分层顺序：

1. 首先搜索同城合格 donor；
2. 同城不存在可信控制时，可搜索所有研究城市的非实验网格；
3. 跨城匹配必须使用处理前的城市内标准化特征，并记录来源城市；
4. 同城和跨城结果分别报告，不得静默混合；
5. GSC 可使用跨城 donor，但必须进行尺度统一和跨城 placebo 验证。

当前 R 正式估计器实现的是同城主规格；全城市 donor 扩展是批量运行前需要补齐并单独验证的
实现项。

## 5. 事件时间

### 5.1 月度结果

- 原始站点日期精确到日并保留。
- 开通当月是部分暴露月，主分析中完全排除。
- 开通后的下一个完整日历月定义为 `event_time = 1`。
- 主 anticipation window 为开通前 6 个完整月。
- anticipation 0 个月和 12 个月为预先规定的敏感性规格。
- 主处理前窗口为 anticipation 之前的 36 个干净月份。
- 月度重点事件期为 1、3、6、12、18、24 月，同时保存完整逐月路径。

例如站点在 2019-12 任意一天开通：

- 2019-06 至 2019-11：主 anticipation window；
- 2019-12：部分暴露月，排除；
- 2020-01：`event_time = 1`；
- 主干净处理前窗口结束于 2019-05。

### 5.2 年度结果

- 开通年视为过渡年并排除；
- 下一完整年份定义为 `event_time = 1`；
- 年度主处理前窗口为开通年前 3 年；
- 年度标签期为开通后第 1、2、3 个完整年份。

## 6. 数据和变量

### 6.1 房价

- 主时间单位：月；年度聚合只用于年度协变量和年度稳健性模型。
- 主价格字段为网格月价格的对数变换。
- 挂牌价和成交价均代表该地点在对应时间的价格观测，但保留来源语义。
- 不要求住房品质调整变量；当前任务关注价格、时间和坐标。
- 同一经济记录的重复副本必须先去重。
- 缺失价格保持缺失，不进行面向因果结果的插值。

### 6.2 VIIRS

- 主结果使用月度 VIIRS。
- 原始输入位于 `MIT_VIIRS_RAW` 指定的外部目录，覆盖 44 城、2012-01 至 2024-12。
- 原始点通过投影坐标下的半开 500m 网格规则精确归属，不使用最近质心。
- 同一坐标的重复记录先折叠；冲突值直接报错。
- 同一网格内不同 VIIRS 采样点取等权均值，并保留采样点数和有效天数。
- 有限负辐亮度是合法观测，不截断为 0。
- 模型变量使用 `asinh(avg_rad)`，不用 `log1p(max(avg_rad,0))`。
- 月度分区按需处理并缓存，不能为每个目标反复读取原始 CSV。

### 6.3 POI

年度特征至少包括：

- `log1p(POI count)`；
- POI 类别熵；
- 商业 POI 占比；
- `log1p(transport-access POI count)`。

POI 既可作为处理前状态变量，也可作为结果族。作为结果时不得使用处理后 POI 选择 donor。

### 6.4 人口

- 年度人口使用 `log1p(population count)`；
- 同一网格—年份重复记录先聚合；
- 人口既可作为处理前状态变量，也可作为年度结果。

## 7. 处理前特征构造

### 7.1 原则

- 只读取目标事件前的可观测信息；
- 处理后值、处理后缺失状态、处理后显著性和最终标签均禁止进入控制搜索；
- 房价、VIIRS、POI、人口应联合使用，而不是只依赖房价；
- 每个目标只使用其实际具有合法处理前支持的变量族；
- 变量族和特征列必须在匹配前冻结并写入 manifest。

### 7.2 月度变量的三个处理前年块

36 个干净处理前月划为三个不重叠的 12 个月块：

- `lag1`：最近的 12 个干净月，作为 holdout；
- `lag2`：中间 12 个月，用于控制搜索；
- `lag3`：最早 12 个月，用于控制搜索。

每个变量在每个块内对有限月值求均值。房价允许月份稀疏，但每个要求的块至少需要一个合法
观测；VIIRS 应具有更完整的月度支持。三个块共同表示处理前水平和走势，避免要求某个指定
月份恰好发生房价交易。

### 7.3 年度变量

- 开通年前第 3、2 年用于控制搜索；
- 开通年前第 1 年作为 holdout；
- 三年原始值共同用于完整平衡报告。

### 7.4 最低变量支持

单一物理控制匹配的主规格要求目标至少有 2 个完整处理前变量族。只有 0 或 1 个变量族时，
不得强制产生最近邻，直接进入 GSC 可行性检查。

当前支持审计结果：

| 完整变量族数 | 处理网格数 |
|---:|---:|
| 0 | 501 |
| 1 | 420 |
| 2 | 537 |
| 3 | 3,179 |
| 4 | 411 |

因此最多 4,127 个网格可进入主多变量单一控制匹配；这不是最终成功匹配数。

## 8. 单一控制网格匹配

### 8.1 目标

对每一个处理网格寻找一个在干预前多变量状态和走势上可信的非实验网格。控制可以被多个
处理网格重复使用，因为禁止放回可能迫使算法选择较差反事实。

### 8.2 正式算法

主匹配使用 R `Matching` 包的 `Match()`：

```text
estimand = "ATT"
M = 1
replace = TRUE
Weight = 2                 # Mahalanobis
Y = NULL                   # 控制身份选择阶段不读取结果
BiasAdjust = FALSE         # 只用于冻结控制身份
Var.calc = 0               # 只用于冻结控制身份
```

这一步对应 Abadie–Imbens 最近邻匹配设计。完整 cohort ATT 作为独立统计估计时，才使用：

```text
BiasAdjust = TRUE
Var.calc = 1
```

若处理样本数或 donor 数不足以识别偏差修正或解析方差，完整 ATT 必须标记为不可识别；不得
接受软件自动降级。个体控制身份和响应标签不应冒充完整 cohort ATT。

### 8.3 共同支持

在调用 `Match()` 前，显式要求处理单元每个活跃训练特征均处于 donor 的闭区间范围内。
不在共同支持内的目标不得匹配，转入 GSC。

### 8.4 处理后泄漏防护

匹配分为两个持久化阶段：

1. `pre-only design`：冻结风险集、特征、共同支持和控制身份；
2. `outcome/label stage`：冻结后才允许读取处理后结果。

自动测试必须证明：修改任意处理后数值或处理后缺失状态，控制身份保持不变。

### 8.5 Holdout 与 donor-donor placebo 门禁

最近处理前年块/年份不参与控制身份选择，而用于独立验证。对同一风险集中的 donor 执行
确定性的 donor-donor 伪处理匹配，形成以下三个经验分布：

1. 训练 Mahalanobis 距离；
2. holdout RMS 标准化差异；
3. holdout 最大绝对标准化差异。

当前主门槛使用 200 个确定性分布位置的 placebo donor，并取三个指标各自的第 95 百分位。
目标匹配对必须同时通过三项门槛。该 95% 规则是项目预注册的质量规则，不是
Abadie–Imbens 定理的一部分。

### 8.6 PanelMatch 的角色

R `PanelMatch` 的完整 Imai–Kim–Wang 流程作为时间序列截面匹配和动态 ATT 的独立稳健性
估计器运行，包括 matched sets、Mahalanobis refinement、placebo、逐集合效应和 1,000 次
bootstrap。它不与 Abadie–Imbens 或 GSC 拼接成一个虚构的“论文算法”。

## 9. Generalized Synthetic Control 回退

### 9.1 何时进入 GSC

以下任一情况进入 `gsc_pending`：

- 少于 2 个完整处理前变量族；
- donor 共同支持失败；
- 最近邻不存在；
- holdout/placebo 质量门禁失败；
- 单一控制在不同预定规格中不稳定。

### 9.2 为什么 GSC 按结果变量运行

单一控制网格可以作为处理网格级设计被多个结果共享；GSC 的交互固定效应模型则直接拟合
具体结果路径，因此权重和反事实原则上是结果变量特定的。不得假定房价 GSC 权重自动适用于
VIIRS、POI 或人口。

### 9.3 正式 Xu-GSC 设置

使用 R `gsynth`：

```text
estimator = "gsynth"
force = "two-way"
CV = TRUE
criterion = "mspe"
r = 0:5
min.T0 = 5
se = TRUE
inference = "parametric"
nboots = 200
normalize = TRUE
```

- 因子数由处理前交叉验证选择；
- 主规格不对同城合法 donor 任意设置 2,000 个上限；
- donor admission 只依据处理前路径；
- 处理后缺失不能触发 donor 重选；
- 月度 GSC 最多使用 36 个干净处理前月，但在数据起点限制下至少需要 5 期；
- VIIRS 2012-01 前不存在，因此早期事件必须进行结构性支持判断。

### 9.4 跨城 GSC

跨城 GSC 是允许的扩展，但必须先完成：

- 城市内处理前尺度标准化；
- 城市/时间系统差异处理；
- 跨城 donor 组成报告；
- 同城与跨城 masked-placebo 预测比较。

在这些门禁完成前，同城 GSC 是当前正式实现。

## 10. 响应标签计算

### 10.1 匹配标签

单一控制 `j` 通过门禁后：

```text
L[i,k,h]
  = (Y[i,k,h] - B[i,k])
  - (Y[j,k,h] - B[j,k])
```

- 月度 `B` 使用最近干净处理前年块的预先冻结统计量；
- 年度 `B` 使用开通年前第 1 年；
- 不显著或接近 0 的标签仍然是合法标签；
- 只有设计失败或目标期缺失才设置 `label_available = FALSE`。

### 10.2 GSC 标签

```text
L[i,k,h] = Y[i,k,h] - Y_hat_0[i,k,h]
```

其中 `Y_hat_0` 为 Xu-GSC 预测的未处理反事实。

### 10.3 变换尺度

每条标签必须声明尺度：

- 房价：对数价格差；
- VIIRS：`asinh` 辐亮度差；
- POI：对应原始比例、熵或 `log1p` 计数差；
- 人口：`log1p` 人口差。

需要原始尺度解释时另行反变换，不在标签表中静默混用尺度。

## 11. 平行趋势与可信度

处理前检验不能证明平行趋势。可信度来自一组预先规定的证据：

- 共同支持；
- 多变量训练距离；
- 未参与匹配的 holdout 趋势；
- donor-donor placebo；
- in-time placebo；
- masked untreated-outcome prediction；
- anticipation 0/6/12 月敏感性；
- 1/1.5/2km 空间排除敏感性；
- 同城/跨城 donor 敏感性；
- 后续站点和竞争交通删失；
- 房价观察概率与交易数量变化。

不得因为处理后效应“不显著”而重新匹配，也不得反复调门槛直到得到理想方向。

## 12. 标签质量和失败规则

### 12.1 标签质量字段

至少保存：

- `method`；
- `control_grid_id` 或 GSC donor/权重；
- `training_distance`；
- `holdout_rms`；
- `holdout_max_abs_gap`；
- `common_support`；
- `pre_rmspe`；
- `placebo_percentile`；
- `standard_error` / 区间（方法可识别时）；
- `quality_grade`；
- `label_available`；
- `failure_reason`。

### 12.2 允许的失败原因

- `insufficient_pre_treatment_families`；
- `insufficient_pre_treatment_periods`；
- `outside_common_support`；
- `no_credible_single_control`；
- `holdout_quality_failed`；
- `gsc_cross_validation_failed`；
- `gsc_counterfactual_missing`；
- `post_outcome_missing`；
- `competing_intervention_contamination`；
- `estimator_runtime_error`（只用于真正程序错误，不能代替经济学原因）。

匹配和 GSC 均失败后进入 MC；MC 仍失败时才舍弃该网格—结果—事件期，不生成 0 标签。

## 13. 输出数据结构

### 13.1 控制设计表

主键：`treatment_order × design_id`。

字段包括处理网格、开通时间、活跃变量族、donor universe、控制网格、训练特征、距离、共同
支持、holdout、placebo、控制来源城市、空间排除规格和设计状态。

### 13.2 GSC 设计表

主键：`treatment_order × outcome × design_id`。

字段包括 donor 集合、因子数、处理前期数、pre-MSPE、CV 结果、模型版本和失败原因。

### 13.3 Response Artifact

主键：

```text
treatment_order × outcome_family × outcome × event_time × specification_id
```

至少包含身份、事件时间、观测值、反事实值、标签、尺度、方法、不确定性、质量、掩码、数据
版本、代码版本和运行 ID。

当前发布器构造完整期望骨架，保留 skipped/缺失结果；将 GSC `est.att` 的标准误、置信区间
和 p 值写入单网格路径；匹配标签使用设计距离和 holdout/placebo 指标，不伪造个体 ATT
标准误。正式产物写入不可覆盖的 `data/active/causal/releases/<release_id>/`，并保存输入、任务、
代码和运行时哈希。

## 14. 队列与可恢复执行

### 14.1 单一控制搜索队列

每个处理网格只出现一次，共 5,048 行。合法状态：

```text
pending
→ matching_running
→ matched_control
或 gsc_required
或 skipped
```

### 14.2 结果族/GSC 队列

仅 `gsc_required` 的处理网格按结果族展开。合法状态：

```text
gsc_pending → gsc_running → gsc_labelled 或 skipped
```

### 14.3 事务规则

- 每个任务独立目录和 manifest；
- 输出先写临时文件，再在同目录原子替换；
- 队列每完成一个任务更新一次；
- 已完成任务重跑必须幂等；
- 中断后根据 manifest 恢复，不得重复追加；
- 不允许长批次只在最后一次性落盘。

当前代码已经使用 5,048 行 `control_design_queue.csv` 先冻结每个处理网格的物理控制身份；
只有未通过单一控制质量门禁的网格才标记为 `gsc_pending`，再由结果族队列运行 GSC。匹配
控制不再随结果变量重复搜索。

## 15. 正式执行顺序

1. 校验 5,048 个处理网格和 donor universe 哈希；
2. 校验站点时间、竞争事件和 1km 空间污染；
3. 构造每个处理网格的处理前变量签名；
4. 运行全部单一控制匹配，不读取处理后结果；
5. 冻结通过质量门禁的控制设计；
6. 将失败目标标记为 `gsc_required`；
7. 按结果变量运行正式 Xu-GSC；
8. 冻结 matched/GSC 反事实；
9. 读取处理后结果并生成 Response Artifact；
10. 运行 placebo、anticipation、空间半径和跨城稳健性；
11. 生成标签质量分层与训练掩码；
12. 才允许表示学习模块读取版本化标签。

## 16. 软件实现与文献边界

正式估计器：

- Imai–Kim–Wang：R `PanelMatch` 3.1.3；
- Abadie–Imbens 最近邻与完整 ATT：R `Matching` 4.10-15；
- Xu-GSC：R `gsynth` 1.4.0。

项目风险集、空间排除、变量签名、holdout、placebo 百分位、失败路由和队列事务属于项目设计
层，不得宣称为上述论文算法内部步骤。三种估计器是相互独立的完整实现，不拼接成不存在的
“混合论文算法”。

## 17. 必须通过的代码门禁

- 处理后值变化不改变匹配控制身份；
- 处理后缺失变化不改变匹配控制身份；
- 开通当月排除、次月为事件期 1；
- 年度开通年排除、下一年为事件期 1；
- 月度三块特征对稀疏房价有效；
- VIIRS 负值通过 `asinh` 保留；
- VIIRS 同坐标重复和同网格多采样点聚合符合合同；
- horizon 1/2/3 与实际年度一一对应；
- GSC 列正确映射到 `treatment_order`；
- 已知合成真值可恢复；
- 非法软件降级被拒绝；
- 队列中断可以恢复；
- GSC bootstrap 不确定性与路径逐行对齐；
- Response Artifact 完整骨架、主键、标签公式和哈希通过验证；
- 训练特征年份严格早于开通年；
- 城市不跨 train/validation/test，归一化只拟合训练城市；
- 失败原因是结构化经济学/数据原因，而不是泛化 runtime error。

## 18. 当前状态与生产边界

已完成：

- 5,048 个处理网格冻结；
- 约 377 万非实验 donor 空间审计；
- 房价面板重建；
- VIIRS 原始月度数据下载和按需缓存代码；
- PanelMatch、Matching 和 gsynth 正式包调用；
- 处理后泄漏、事件时间、GSC 映射和队列恢复测试；
- 5,048 行处理网格级控制设计队列及事务恢复；
- 同城优先、全城市处理前标准化的 Matching fallback；
- 月度房价和 VIIRS 三个 12 月块，以及最近干净 12 月标签基期；
- 10 个真实成都网格审计：9 个匹配成功，其中 1 个使用跨城控制，1 个进入 GSC；
- 一个真实 81,285 donor 的成都 Xu-GSC smoke test，因子交叉验证、20 次参数 bootstrap、
  反事实路径和标签均成功落盘；该结果明确标记为 `production_eligible=FALSE`。
- GSC 不确定性规范化、Response Artifact 严格发布器、质量/训练掩码、输入/代码/run 哈希；
- 处理前多模态特征、街景时间过滤、城市级防泄漏切分和 train-only 归一化发布器；
- 上述发布模块的微型合成合同测试。

正式全量运行前仅需：

1. 在目标服务器记录 R、包版本、CPU、内存和输入文件哈希；
2. 清空 10 行本机审计状态并重新生成 5,048 行全 pending 队列；
3. 先运行一批服务器 canary，复核耗时、峰值内存和输出 manifest；
4. 获得明确的全量启动确认后再运行 5,048 个控制设计任务；
5. 对 `gsc_pending` 按结果变量运行生产模式的 200 次参数 bootstrap；
6. 不把 smoke/pilot 状态或旧住房 DID 结果当成最终标签。

## 19. 主要参考文献

- Abadie, A. and Imbens, G. W. (2006). Large Sample Properties of Matching Estimators for Average Treatment Effects.
- Abadie, A. and Imbens, G. W. (2011). Bias-Corrected Matching Estimators for Average Treatment Effects.
- Imai, K., Kim, I. S., and Wang, E. H. (2023). Matching Methods for Causal Inference with Time-Series Cross-Sectional Data.
- Xu, Y. (2017). Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models.
- Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-Differences with Multiple Time Periods.
- Sun, L. and Abraham, S. (2021). Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects.
- Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends.
- Rambachan, A. and Roth, J. (2023). A More Credible Approach to Parallel Trends.
