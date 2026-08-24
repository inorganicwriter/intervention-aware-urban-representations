# DDR-003：完整已发表估计器的隔离实现

<!-- GPU 迁移前状态：估计器实现完成；正式批量估计冻结，等待计算路由冻结。 -->
状态：R 参考实现冻结；Python/GPU 生产实现完成，等待 GSC/MC 服务器资格门  
日期：2026-07-22；GPU 迁移增补：2026-08-24  
替代：早期已撤销的 prototype 设计

## 2026-08-24 Python/GPU 迁移增补

本 DDR 下列 A–C 节保留原始 R 官方包工作流，作为论文方法解释、一次性
资格参考和显式 `r_reference` 后端；它们不再是逐任务默认生产后端。当前默认
后端为 `python_gpu`：

- Matching 控制设计与最终固定控制标签由 Python/PyTorch 执行，并同时对 R
  候选集、最终控制、质量门和完整标签路径做 parity；
- Xu GSC 的交互固定效应、rolling CV、反事实路径和参数 bootstrap 由
  Python/PyTorch 执行；
- matrix completion 的 lambda CV、最终拟合和 unit jackknife 由
  Python/PyTorch 执行，`lambda=0` 是合法的未正则化端点；
- production 运行必须提供至少 3 个 Matching、3 个 GSC、3 个 MC 代表任务
  形成的环境绑定资格凭证；preview 不得晋升为正式结果；
- 每个 shard 完整验证凭证和绑定源码一次，子进程只验证该凭证摘要，避免逐任务
  重复哈希资格面板。

截至 2026-08-24，本机 R 4.6.1 与 RTX 4060 已完成 3 个真实 Matching 任务
（orders 507、509、530）的设计和最终标签 parity，三例标签最大绝对误差均为
0。GSC/MC 各 3 个代表任务和完整凭证仍需在 RTX 4090 服务器运行；因此不得把
“实现完成”写成“全量生产已完成”。规范入口为：

- `scripts/causal_python/run_causal_label_queue.py`；
- `scripts/causal_python/run_formal_estimator.py`；
- `scripts/causal_gpu/run_shadow_queue.py`；
- `scripts/causal_gpu/audit_formal_qualification.py`。

## 原则

项目不再把多个论文组件拼接后称为某篇论文的完整算法。三种方法作为相互独立的完整估计器运行，分别保存输入、官方软件对象、诊断、估计量和推断结果。项目路由规则只决定哪个数据子样本交给哪个估计器，不修改估计器内部步骤。

## A. Imai–Kim–Wang PanelMatch

使用 R `PanelMatch` 3.1.3 的官方完整流程：

1. `PanelData()` 构造平衡时间序列截面数据；
2. `PanelMatch()` 构造相同处理历史 matched sets；
3. `refinement.method="mahalanobis"`、`size.match=1`、matching with replacement；
4. `get_covariate_balance()` 生成官方平衡诊断；
5. `get_set_treatment_effects()` 保存每个 matched set 的动态效应；
6. `PanelEstimate()` 使用 1,000 次 bootstrap 产生动态 ATT、置信区间和 placebo 输出。

年度结果按城市、开通 cohort、处理前变量签名和结果族运行。月度房价使用月度处理时间、36个月处理历史、处理前年度变量的第13/25/37月滞后以及1–24月动态效应。

## B. Abadie–Imbens

使用 R `Matching` 4.10-15 的官方 `Match()`：

- `estimand="ATT"`；
- `M=1`；
- `replace=TRUE`；
- `Weight=2`（Mahalanobis）；
- `BiasAdjust=TRUE`；
- `Var.calc=1`（Abadie–Imbens异方差稳健解析方差）；
- 不对最近邻匹配使用普通 bootstrap。

结果变量是预先定义的处理前—处理后变化，因此该估计器作为匹配差分 ATT 独立报告，不冒充 PanelMatch 或 staggered DiD。

## C. Xu generalized synthetic control

使用 R `gsynth` 1.4.0 的官方完整流程：

- `estimator="gsynth"`；
- `force="two-way"`；
- `CV=TRUE`、`criterion="mspe"`；
- `r=0:5`；
- `min.T0=5`；
- `se=TRUE`；
- `inference="parametric"`；
- `nboots=200`；
- 保存完整 `gsynth` 对象、反事实路径、ATT、处理前拟合和不确定性。

主规范使用同城市全部合格never-treated网格，不进行2,000 donor截断。跨城市规范单独运行和报告。

## 项目路由（不属于论文算法）

- 处理网格固定为5,048个站点网格；
- donor必须是空间合格、非实验且无已知站点污染的网格；
- 处理前变量签名只用处理前数据形成；
- 同城市是主规范，跨城市是稳健性规范；
- 数据不足只决定估计器是否可运行，不改变任何算法公式；
- 三种估计器的结果不得混合成一个没有理论定义的标准误或ATT。

## 实现与门禁记录

实现文件：

- `scripts/causal_r/run_complete_panelmatch.R`；
- `scripts/causal_r/run_complete_abadie_imbens.R`；
- `scripts/causal_r/run_complete_xu_gsc.R`；
- 共享的只读数据构造层 `scripts/causal_r/complete_estimators_lib.R`。

2026-07-22 门禁结果：

1. 三套官方流程均通过合成数据测试。PanelMatch 测试覆盖 matched sets、官方平衡诊断、逐集合动态效应、bootstrap 和 placebo；Abadie–Imbens 测试覆盖偏差修正和解析方差；GSC 测试覆盖 CV、反事实路径和参数 bootstrap。
2. 厦门 2019 cohort 的 Abadie–Imbens 真实运行成功：26 个处理网格、2,882 个完整 donor，`BiasAdjust=TRUE` 和 `Var.calc=1` 均未降级。零方差变量不进入不可逆的协方差矩阵，并在 manifest 中逐项记录。
3. 厦门 2019 cohort 的 Xu GSC 真实全 donor 运行成功：16,514 个 donor、9 个处理前年度、`r=0:5` 完整 CV、200 次参数 bootstrap，完整官方对象和反事实路径已经保存。
4. PanelMatch 的 300-donor 真实数据集成门禁通过；该结果明确标记为 `formal_estimate=FALSE`。全 16,514 donor 的官方 PanelMatch 生产基准在当前机器运行 10 分钟仍未完成，且没有产生正式结果。正式批量运行前必须冻结一个具有论文依据、只依赖处理前信息的 coarse risk-set 规则，或配置可承受全 donor 计算的运行资源。不得把测试 fixture 结果当作估计结果。

## 不允许的静默降级

- 当完整处理单元数不多于偏差修正协变量数时，`Matching` 会自动关闭 `BiasAdjust`。运行器必须提前停止并报告“不识别”，不得保存为完整 Abadie–Imbens 结果。
- 当 donor 数不足以支持 `Var.calc=1` 时必须停止，不得接受包自动改为 `Var.calc=0`。
- GSC 的因子候选数必须由足够的处理前期支持；年度 GSC 使用该结果变量全部可用的处理前年份，而不是人为只保留五期后再让包裁剪 `r=0:5`。
- PanelMatch 的真实子集门禁、降低 bootstrap 次数或 donor 截断只能用于测试，不能写入正式队列。
