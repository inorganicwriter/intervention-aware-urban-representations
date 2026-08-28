# DDR-003：估计器与生产后端边界

状态：frozen

日期：2026-07-22

修订：2026-08-27

## 生产后端

默认后端为 `python_gpu`：

- Matching 控制设计与固定控制标签由 Python/PyTorch 执行，并与 R 参考结果
  比较候选集、最终控制、质量门和标签路径；
- Xu GSC 使用交互固定效应、rolling CV、反事实路径和参数 bootstrap；
- matrix completion 使用 lambda CV、最终拟合和 unit jackknife，`lambda=0`
  是合法端点；
- production 运行要求环境绑定资格凭证至少包含 3 个 Matching、3 个 GSC 和
  3 个 MC 代表任务；
- shard 启动时验证完整凭证与源码绑定，子进程验证凭证摘要。

R 官方包流程保留为方法解释、资格参考和显式 `r_reference` 后端，不作为逐任务
默认生产后端。规范入口为：

- `scripts/causal_python/run_causal_label_queue.py`；
- `scripts/causal_python/run_formal_estimator.py`；
- `scripts/causal_gpu/run_shadow_queue.py`；
- `scripts/causal_gpu/audit_formal_qualification.py`。

## 方法边界

Matching、GSC 和 MC 作为独立估计器运行，分别保存输入、参考软件对象、诊断、
估计量和推断结果。项目路由只决定数据子样本与估计器的对应关系，不改变估计器
内部步骤。

## A. Imai, Kim, and Wang PanelMatch

使用 R `PanelMatch` 3.1.3 的官方完整流程：

1. `PanelData()` 构造平衡时间序列截面数据；
2. `PanelMatch()` 构造相同处理历史 matched sets；
3. `refinement.method="mahalanobis"`、`size.match=1`、matching with replacement；
4. `get_covariate_balance()` 生成官方平衡诊断；
5. `get_set_treatment_effects()` 保存每个 matched set 的动态效应；
6. `PanelEstimate()` 使用 1,000 次 bootstrap 产生动态 ATT、置信区间和 placebo 输出。

年度结果按城市、开通 cohort、处理前变量签名和结果族运行。月度房价使用月度处理时间、36个月处理历史、处理前年度变量的第13/25/37月滞后以及1-24月动态效应。

## B. Abadie and Imbens

使用 R `Matching` 4.10-15 的官方 `Match()`：

- `estimand="ATT"`；
- `M=1`；
- `replace=TRUE`；
- `Weight=2`（Mahalanobis）；
- `BiasAdjust=TRUE`；
- `Var.calc=1`（Abadie and Imbens 异方差稳健解析方差）；
- 不对最近邻匹配使用普通 bootstrap。

结果变量是预先定义的处理前到处理后变化，因此该估计器作为匹配差分 ATT 独立报告，
与 PanelMatch 和 staggered DiD 分开解释。

## C. Xu generalized synthetic control

R `gsynth` 1.4.0 是资格/参考实现；正式队列使用与该合同对齐的 Python/PyTorch GPU
实现，并由环境绑定资格凭证确认数值和推断一致性：

- `estimator="gsynth"`；
- `force="two-way"`；
- `CV=TRUE`、`criterion="mspe"`；
- `r=0:5`；
- `min.T0=5`；
- `se=TRUE`；
- `inference="parametric"`；
- `nboots=200`；
- 保存完整 `gsynth` 对象、反事实路径、ATT、处理前拟合和不确定性。

主规范使用同城市全部合格 never-treated 网格，不设置 2,000 donor 截断。跨城市规范单独运行和报告。

## 项目路由（项目新增规则）

- 处理网格固定为5,048个站点网格；
- donor必须是空间合格、非实验且无已知站点污染的网格；
- 处理前变量签名只用处理前数据形成；
- 同城市是主规范，跨城市是稳健性规范；
- 数据不足只决定估计器是否可运行，不改变任何算法公式；
- 三种估计器分别报告标准误和 ATT，项目报告中保留各自的理论定义。

## 降级处理规则

- 当完整处理单元数不多于偏差修正协变量数时，`Matching` 会自动关闭 `BiasAdjust`。运行器提前停止并报告“不识别”，结果标记为非完整 Abadie and Imbens 估计。
- 当 donor 数不足以支持 `Var.calc=1` 时停止运行，保留 `Var.calc=1` 的正式设置。
- GSC 的因子候选数依据处理前期支持确定；年度 GSC 使用该结果变量全部可用的处理前年份，保持 `r=0:5` 的完整候选范围。
- PanelMatch 的真实子集门禁、降低 bootstrap 次数和 donor 截断属于测试配置，正式队列沿用完整配置。
