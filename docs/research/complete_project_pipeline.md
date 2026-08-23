# Learning How Cities Respond：完整项目与生产管线

状态：因果标签、训练前数据、表示模型与训练器均已实现；正式 5,048 网格生产尚未完成
更新日期：2026-08-19

本文保留稳定的研究与生产合同；当前队列状态和已完成的 canary 记录见
[`docs/operations/current_project_status.md`](../operations/current_project_status.md)，服务器命令见
[`docs/operations/server_deployment.md`](../operations/server_deployment.md)。

## 1. 项目目标

题目：**Learning How Cities Respond: Intervention-Conditioned Urban Representations**。

项目不以一个总体 DID 系数为最终产物，而是为每个发生地铁站开通的城市网格构造未处理
反事实，并形成随事件时间变化的局部响应：

```text
causal_response_label = observed outcome - untreated counterfactual
```

响应标签作为监督信号，使表示空间中的距离表达“在同类干预下是否产生相似响应”，而不只
表达视觉或语义相似。

## 2. 研究对象与时间定义

- 城市：44；
- 空间单元：固定 500m × 500m 网格；
- 处理单元：5,048 个包含正式地铁站点开通事件的唯一网格；
- 初始 donor universe：3,771,800 个非处理网格；
- 主空间污染排除：1km；
- 月度事件期 1：开通后的第一个完整自然月；
- 年度事件期 1：开通后的第一个完整自然年；
- 开通月/年不作为完整处理后时期；
- 主 anticipation window：6 个月，0/12 个月作为敏感性规范。

| 结果族 | 变量 | 事件期 |
|---|---|---|
| housing | `housing_log_price` | 1/3/6/12/18/24 月 |
| viirs | `viirs_avg_asinh` | 1/3/6/12/18/24 月 |
| population | `population_log` | 1/2/3 年 |
| poi | 4 个 POI 构成指标 | 1/2/3 年 |

## 3. 数据层

权威空间身份来自 `city_key × grid_id`。站点、城市归属、别名、多站网格、竞争交通事件及
城市边界问题通过人工决议表和可追踪应用器处理。

- 房价：标准化网格月度/年度面板；
- VIIRS：月度按需缓存及年度多采样点网格聚合；
- POI：2012–2024 网格年度构成；
- 人口：重复地理采样按网格年度聚合；
- Sentinel-2：训练阶段将同一网格年度多采样点聚合为 NDVI/NDBI 均值；
- 街景：通过可选且带 `capture_date` 的资产索引接入，只允许开通前图像；
- 2026 道路网络是处理后静态截面，默认不进入历史处理前特征。

## 4. 反事实路由

### 4.1 单一物理控制设计

控制设计阶段先进行同城控制搜索；搜索阶段使用 `Matching::Match` 的 `Y=NULL`、Mahalanobis
距离、`M=5` 候选和允许替换设置。匹配失败时记录 `gsc_pending`，跨城控制在结果族路由需要时单独生成，
不覆盖同城设计记录。所有控制选择均不读取处理后结果或处理后缺失状态。

月度房价和 VIIRS 使用三个完整 12 月块：lag 2/3 用于训练距离，lag 1 完全留出。年度 POI
和人口使用开通前第 1/2/3 个完整年度。目标至少需要一个完整变量族才能进入物理匹配。

1. 同城市 donor 匹配：显式共同支持、lag1 holdout 和 200 个确定性 donor-placebo q95 门禁；
2. 同城匹配通过后冻结控制身份；
3. 同城匹配失败后依次运行同城 GSC 和同城 MC；
4. 同城两条回退路径均失败后，再运行跨城标准化匹配、跨城 GSC 和跨城 MC；
5. 六条路径均失败时进入 `skipped`，并记录结构化原因。

同城优先、跨城标准化、holdout 和 q95 属于项目设计层，与 Abadie–Imbens 估计器分开报告。
正式研究必须报告 placebo 数量、分位门槛、空间半径和跨城规范敏感性。

### 4.2 匹配标签

月度基期是最近一个干净 12 月块的均值；年度基期是开通前最后完整年度：

```text
label(i,h) = [Y(i,h)-Y(i,baseline)] - [Y(j,h)-Y(j,baseline)]
```

单一匹配标签不伪造 Abadie–Imbens 个体标准误。其质量依据是共同支持、匹配距离、处理前
留出误差和 placebo 分布。完整 Abadie–Imbens ATT 估计器独立保留，用于 cohort/城市层
平均效应校准，而不是替代逐网格标签。

### 4.3 Xu generalized synthetic control

匹配失败后按结果变量运行 Xu（2017）交互固定效应：

```text
Y_it(0) = alpha_i + xi_t + lambda_i' f_t + error_it
```

正式设置：`force="two-way"`、`CV=TRUE`、`criterion="mspe"`、`r=0:5`、
`min.T0=5`、`se=TRUE`、`inference="parametric"`、`nboots=200`。因子交叉验证可并行，
选定因子后的 bootstrap 顺序执行以控制内存。正式任务是一网格一模型；GSC 的 `est.att`
标准误、置信区间和 p 值按路径位置写入标签。跨城标准化模型的效应及不确定性同时乘回目标
城市处理前尺度，避免原尺度标签与标准化尺度标准误混用。

`smoke_test` 只运行 20 次 bootstrap，独立目录且 `production_eligible=FALSE`。

## 5. Response Artifact

全量任务结束后运行 `build_response_artifact.py`。发布器先构造所有期望标签的完整骨架，
因此成功、结果缺失和被跳过任务都有明确记录。主键：

```text
treatment_order × outcome_family × outcome × event_time × specification_id
```

核心字段包括身份、事件时间、观测值、反事实、响应标签、控制设计、GSC 不确定性、质量、
失败原因、训练掩码、release ID、run ID、数据版本和代码版本。

质量类别：

```text
matched_same_city_pass
matched_cross_city_pass
gsc_same_city_pass
gsc_cross_city_pass
unavailable
pending
```

正式发布器拒绝非 5,048 处理清单、非 20,192 结果族任务、非终态队列、重复/未完成控制、
处理后信息泄漏、任务文件缺失、标签公式错误和输出目录覆盖。

```text
data/active/causal/releases/<release_id>/
  response_artifact.parquet
  quality_summary.csv
  manifest.json
```

manifest 保存处理清单、donor、队列、支持表和任务产品哈希，以及 Python/包/代码版本。
训练前数据发布时必须再次读取该 manifest 中的 treatment SHA-256，并逐项核对
`treatment_order` 对应的城市、网格、站点事件和开通月份；不能只按顺序号连接。
任务标签也必须在队列写入和 Response Artifact 汇总两个边界分别验证目录、
manifest、treatment 身份、结果族与事件期。

## 6. 训练前多模态数据集

`build_pretraining_dataset.py` 读取已发布 Response Artifact，并输出：

```text
data/active/model_inputs/<dataset_id>/
  unit_features.parquet
  response_targets.parquet
  sample_index.parquet
  normalization.json
  manifest.json
```

处理前特征默认使用开通年前第 1/2/3 年：房价、POI、VIIRS、人口和 Sentinel-2。所有源
数据先按 `city_key × grid_id × year` 唯一化；代码断言 `feature_year < opening_year`。
街景资产只有 `capture_date < opening_month` 才能进入索引。

城市通过固定 seed 的 SHA256 排序分为 train/validation/test。同一城市绝不跨 split；均值
和标准差只在训练城市拟合，再应用于验证和测试城市。缺失值不填成零，由各模态 availability
mask 表达。

```text
final_training_mask = response.training_mask AND feature_training_mask
```

`feature_training_mask` 由可配置的最少可用模态数控制。模型可以进一步要求图像模态，但不能
放宽 Response Artifact 的因果质量掩码。

## 7. 服务器环境

Python 调度器支持：

```text
MIT_RSCRIPT   Rscript 可执行文件
MIT_R_LIB     项目 R 包库
MIT_VIIRS_RAW VIIRS 原始月度根目录
```

未设置时才回退到本机默认路径。服务器运行必须记录 CPU、内存、R/Python/包版本和输入哈希。

## 8. 正式执行顺序

1. 配置服务器环境并运行测试；
2. 快照输入及哈希；
3. 重置三张队列为全 pending；
4. 运行小规模 control-design canary；
5. 审计资源、共同支持、holdout 和 placebo；
6. 经明确批准后运行 5,048 个控制设计；
7. 为 matched 控制生成结果标签；
8. 为 `gsc_pending` 按结果变量运行正式 200-bootstrap GSC；
9. 确认四个结果族全部终态；
10. 发布严格 Response Artifact；
11. 构造城市隔离的训练前数据集；
12. 冻结 release 后才允许模型训练读取。

`--allow-partial` 仅用于 canary 和开发验证；部分产物带有非生产标记。

## 9. 识别与验证边界

代码可以检验共同支持、处理前拟合、placebo 相对质量、donor 污染、信息泄漏和输出完整性；
现有数据只能支持对可观测识别条件的检查，不能替代未观测混杂和空间干扰的敏感性分析。正式报告保留：

- 0/6/12 月 anticipation；
- 0.5/1/1.5/2km 空间半径；
- 同城-only 与跨城 fallback；
- placebo 200/500/1000 和 q90/q95/q99；
- 匹配与 GSC 方法一致性；
- 处理前拟合与标签幅度关系；
- 城市留出泛化。

## 10. 主要实现

- 网格控制设计：`scripts/causal_r/grid_control_design_lib.R`；
- 固定控制标签：`scripts/causal_r/fixed_control_label_lib.R`；
- PanelMatch：`scripts/causal_r/run_complete_panelmatch.R`；
- Abadie–Imbens：`scripts/causal_r/run_complete_abadie_imbens.R`；
- Xu GSC：`scripts/causal_r/run_complete_xu_gsc.R`；
- 事务队列：`scripts/causal_r/run_*_queue.py`；
- Response Artifact：`src/urban_intervention/causal/response_artifact.py`；
- 训练前数据：`src/urban_intervention/causal/pretraining_dataset.py`。

## 11. 文献边界

- Abadie and Imbens（2006；2011）：最近邻与偏差修正匹配；
- Imai、Kim and Wang（2023）：PanelMatch；
- Xu（2017）：generalized synthetic control；
- Abadie（2021）：synthetic-control feasibility and diagnostics；
- Roth（2022）及 Rambachan and Roth（2023）：平行趋势预检与敏感性。

论文估计器、项目路由、质量门禁和机器学习发布层分别报告，不合并为单一估计器。
