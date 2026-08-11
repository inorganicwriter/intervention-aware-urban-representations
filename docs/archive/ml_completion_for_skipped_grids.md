# ML 补全方法在缺失对照网格中的应用分析

状态：研究提案（proposed）  
日期：2026-07-24  
参考文献：Ratledge et al. (2022), *Using machine learning to assess the livelihood impact of electricity access*, Nature 611:491-495

## 1. 问题背景

当前项目中有部分处理网格在匹配和 GSC 两条路径中均失败：

| 完整变量族数 | 处理网格数 | 匹配路径 | GSC 路径 |
|---:|---:|---|---|
| 0 | 501 | 无法进入（没有完整变量族） | GSC 需 ≥ 5 个预处理期，多数失败 |
| 1 | 420 | 可进入匹配（放宽后的最低边界） | 匹配失败后可进入 GSC/MC，必须记录单变量支持 |
| 2+ | 4,127 | 可进入匹配 | 匹配失败后可进入 GSC/MC |

主要失败原因是早期开通站点（2010-2014）的预处理数据不足：
- VIIRS 始于 2012-01，2012 年前开通的站点无 VIIRS 预处理期
- 房价数据按城市差异较大，部分城市 2015 年后才有覆盖
- POI 和人口年度数据从 2010-2012 年开始

只有 0 个完整变量族的网格在匹配入口直接失败。420 个仅有 1 个完整变量族的网格允许进入匹配；匹配和 GSC 均失败后，MC 使用 `min.T0=1` 的覆盖优先边界继续尝试生成响应标签。

## 2. Ratledge et al. (2022) 的方法

### 2.1 论文核心思路

Ratledge et al. 在乌格兰电力接入的因果评估中面临类似问题：缺乏详细的微观经济结果数据。他们的解决方案分为两步：

**第一步：ML 预测结果变量**

- 用卫星影像 + CNN（ResNet 架构）预测社区级资产财富指数
- 训练数据来自 DHS（Demographic and Health Surveys）的 640,000+ 户家庭数据，覆盖 25 个撒哈拉以南非洲国家的 27,174 个社区
- 关键创新：在 CNN 损失函数中加入分位数偏差惩罚项（quintile-specific bias penalty），确保预测值在整个财富分布上无系统偏差，而不只是最小化 MSE
- 输出：2005-2017 年逐年的社区级财富预测面板

**第二步：ML 因果推断**

在 ML 预测的完整面板上，使用两种方法估计因果效应：

1. **Matrix Completion (MC)**（Athey, Bayati, Doudchenko, Imbens & Khosravi, 2021, JASA）
   - 将面板视为一个 N×T 矩阵，处理单元的 post-treatment 值视为缺失
   - 用核范数正则化（nuclear norm regularization）补全缺失值
   - 补全值即为反事实估计
   - 优势：不要求共同支持，不要求平行趋势，可处理不平衡面板

2. **Synthetic Control with Elastic Net (SC-EN)**
   - 用 Elastic Net 回归在 donor pool 中估计合成控制权重
   - 比 Abadie et al. (2010) 的传统 SC 更适合大 donor pool
   - 放松了非负权重和权重和为 1 的约束

### 2.2 论文的关键发现

- MC 和 SC-EN 在交叉验证中能准确预测 held-out 控制单元的值（平均偏差 0.017 和 0.018）
- 在模拟数据中，MC 在存在处理相关的时间趋势时比 DiD 偏差更小
- 两种 ML 方法比传统 DiD 给出更可靠的因果估计
- CNN 预测中的分位数偏差修正对因果估计有显著影响——偏差越大，因果估计被低估越多

### 2.3 与传统方法的区别

| 特征 | 传统匹配/GSC | Ratledge ML 方法 |
|------|-------------|-----------------|
| 结果变量来源 | 直接观测（房价、VIIRS 等） | ML 预测（卫星影像→财富/经济指标） |
| 预处理期要求 | ≥ 5 期（GSC）或 ≥ 2 年（匹配） | MC 可处理更短/不平衡面板 |
| 共同支持 | 匹配必需 | MC 不要求 |
| Donor 权重 | 非负、和为 1（SC）或最近邻 | SC-EN 放松约束 |
| 偏差控制 | 依赖平行趋势假设 | CNN 分位数偏差惩罚 + MC 核范数正则 |
| 适用场景 | 数据充足时 | 数据稀疏或缺失时 |

## 3. 在本项目中的应用可能性

### 3.1 可行的应用路径

**路径 A：ML 预测补全缺失结果变量**

对于预处理数据不足的网格，用卫星影像（Sentinel-2 NDVI/NDBI、VIIRS 夜间灯光）训练一个回归模型，预测缺失的房价或 POI 值，构造完整面板后再跑 GSC 或 MC。

- 优势：利用已有的 Sentinel-2 和 VIIRS 数据
- 挑战：需要训练标签（有房价/POI 数据的网格作为训练集）
- 风险：ML 预测值的测量误差会传递到因果估计中（Berkson 型衰减）

**路径 B：直接用 Matrix Completion 补全反事实**

不预测缺失的结果变量，而是将现有面板（含缺失值）直接输入 MC 算法，补全处理单元的处理后反事实路径。

- 优势：不需要额外的 ML 预测步骤
- 挑战：MC 对缺失模式有要求（随机缺失比结构性缺失表现更好）
- 风险：早期站点的预处理期太少，MC 可能无法识别个体效应

**路径 C：CNN 预测 + MC 推断（完整复刻 Ratledge）**

用卫星影像训练 CNN 预测网格级经济指标，然后在整个预测面板上用 MC 估计因果效应。

- 优势：与 Ratledge 方法完全一致，有 Nature 发表支撑
- 挑战：需要大量训练数据（DHS 类似的调查数据在中国不可得）；需要训练 CNN
- 风险：训练数据来源和标签质量是关键瓶颈

### 3.2 推荐方案

**短期（当前项目阶段）：路径 B**

在不引入新 ML 预测的前提下，对匹配和 GSC 均失败的网格，尝试 Matrix Completion 方法。具体做法：

1. 对每个失败网格，收集其在所有可用结果族上的观测值（即使不完整）
2. 从同城和跨城 donor pool 中选取有完整预处理路径的 donor
3. 将 treated + donor 的面板输入 MC 算法（R 包 `gsynth` 的 `estimator="mc"` 模式或 Athey et al. 的官方实现）
4. MC 补全的反事实路径即为标签

**中期（论文扩展阶段）：路径 A**

用 Sentinel-2 + VIIRS 训练房价/POI 预测模型，补全早期站点的缺失结果变量，然后重新跑 GSC 或 MC。

### 3.3 与当前框架的兼容性

| 维度 | 当前框架 | MC 补全路径 |
|------|---------|------------|
| 标签定义 | `L = observed - counterfactual` | 不变 |
| 信息边界 | 控制选择不读处理后结果 | MC 用 donor 的处理前路径补全 treated 的处理后反事实，不读 treated 的处理后结果 |
| 质量门禁 | 共同支持 + placebo q95 | MC 无共同支持要求；可用 cross-validation RMSPE 替代 |
| 不确定性 | GSC bootstrap 或匹配诊断 | MC 可用 subsampling 或 bootstrap |
| 路由 | matched → gsc_pending → skipped | 在 skipped 前加 MC 作为第三层回退 |

### 3.4 需要解决的问题

1. **MC 的最小数据要求**：Athey et al. (2021) 没有明确的最小预处理期要求，但实际需要足够的预处理期来识别个体效应。0 个完整变量族的网格可能仍然无法用 MC 补救。

2. **测量误差传播**：如果用 ML 预测的结果变量，预测误差会导致因果估计的衰减（Ratledge 论文 Extended Data Fig. 7 证实了这一点）。需要在标签质量字段中记录 `outcome_source = "ml_predicted"` 或 `"observed"`。

3. **标签质量分级**：MC 补全的标签应标记为比 matched/GSC 更低的质量等级，例如 `mc_fallback`，在训练 mask 中可选排除。

4. **软件实现**：`gsynth` 包支持 `estimator="mc"` 模式（矩阵补全），可直接复用现有代码框架。或者使用 Athey et al. 的官方 R 实现。

## 4. 文献依据

| 文献 | 核心贡献 | 与本项目的关系 |
|------|---------|--------------|
| Ratledge et al. (2022), Nature | ML 预测 + MC/SC-EN 因果推断在数据稀疏环境下的应用 | 直接参考：补全 skipped 网格 |
| Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), JASA | Matrix Completion 方法用于因果面板数据 | MC 方法的理论依据 |
| Doudchenko & Imbens (2016), arXiv | 合成控制、DiD 和平衡方法的综合框架 | SC-EN 方法的理论基础 |
| Abadie, Diamond & Hainmueller (2010), JASA | 传统合成控制法 | 当前 GSC 的基础方法 |
| Yeh et al. (2020), Nat. Commun. | 用卫星影像和深度学习预测非洲经济福祉 | CNN 预测结果变量的方法参考 |
| Jean et al. (2016), Science | 卫星影像 + ML 预测贫困 | 早期开创性工作 |

## 5. 已实现状态与下一步

1. 队列已实现 `gsc_running → mc_pending → mc_running → mc_labelled`；MC 失败则进入带原因的 `skipped`。
2. MC 使用 Athey et al. 方法的官方 `fect(method="mc")` 实现，仅用处理前信息选择正则化参数，并把处理后处理组结果从拟合输入中遮蔽。为保持 `min.T0=1` 的目标准入边界，第一阶段以 `se=FALSE, CV=TRUE, cv.nobs=1, cv.donut=0, cv.buffer=0` 选择 lambda；第二阶段固定 lambda，以 `CV=FALSE, se=TRUE, nboots=200` 估计反事实和不确定性。两阶段设计规避 `fect 2.4.5` bootstrap 包装器未转发低支持 CV 参数的问题，同时不取消交叉验证。lambda 仍必须由至少一个额外 donor 验证观测选择，不能用单个观测伪造 CV。
3. 每个结果族允许部分成功；成功结果必须具有有限的交叉验证 MSPE、正的 `selected_lambda`、反事实和不确定性，才能进入 Response Artifact。
4. 下一步先运行有限的生产参数 canary，核验耗时和产物合同；审核通过后再启动全量队列，并分别报告零族与单族网格的标签成功率。
