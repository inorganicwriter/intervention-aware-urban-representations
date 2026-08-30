# 项目总览与阅读入口

更新：2026-08-27
状态：当前项目入口

这份文档说明项目的估计对象、结果生成流程和运行入口。具体参数以冻结决策和因果主设计为准，实时队列状态以状态页为准。

## 1. 项目要解决什么问题

项目题目是 **Learning How Cities Respond: Intervention-Conditioned Urban Representations**。
研究对象是地铁站正式开通后，发生开通事件的 500m × 500m 城市网格相对于其未处理反事实的响应。

对处理网格 `i`、结果变量 `k` 和事件期 `h`，响应标签定义为：

```text
L[i, k, h] = observed outcome[i, k, h] - untreated counterfactual[i, k, h]
```

项目输出带有方法、质量、支持范围和不确定性记录的逐网格响应路径。该响应路径作为表示学习的监督信号。

## 2. 两层管线

```text
冻结空间与事件资产
        ↓
处理前特征、风险集和 donor 审计
        ↓
控制设计（只读处理前信息）
        ↓
匹配 / GSC / MC 六轮路由
        ↓
观测路径 - 未处理反事实路径
        ↓
Response Artifact（质量、掩码、不确定性、溯源）
        ↓
城市隔离的训练前数据集
        ↓
干预条件化城市表示学习
```

反事实标签层和表示学习层分开管理。训练器只读取已经发布的 Response Artifact 与训练前特征，控制选择和因果队列由前一层负责。

## 3. 当前冻结研究范围

| 项目 | 当前口径 |
|---|---|
| 城市 | 44 个中国大陆地铁城市 |
| 空间单元 | 固定 500m × 500m 网格 |
| 处理网格 | 5,048 个唯一站点开通网格 |
| 初始合格 donor | 3,771,800 行空间风险集 |
| 主空间排除 | 处理站点到 donor 网格多边形最短距离至少 1km |
| 结果族 | housing、VIIRS、POI、population |
| 月度事件期 | 开通月排除；下一个完整月为 `event_time=1`；主 anticipation 为 6 个月 |
| 年度事件期 | 开通年排除；下一个完整年为 `event_time=1`；标签期为 1-3 年 |

月度主干净处理前窗口为 `-42:-7`，其中 `-6:-1` 是 anticipation 窗口；月度后窗口为 `1:24`。年度事件研究主窗口为 `-4:-1` 与 `1:3`。0、6、12 个月 anticipation 和 0.5、1、1.5、2km 空间半径属于敏感性规格，运行时需要单独记录。

## 4. 反事实路由

对每个处理网格，控制设计先固定一个 pre-only 风险集，再按以下顺序路由：

```text
同城物理匹配
  → 同城 GSC
  → 同城 MC
  → 跨城标准化匹配
  → 跨城 GSC
  → 跨城 MC
  → skipped
```

- 物理匹配阶段使用处理前结果历史与静态区位/轨道特征；`M=5` 候选再做静态协变量精炼，最终冻结一个控制网格。
- 匹配阶段使用共同支持、较早处理前块训练距离、最近处理前块 holdout 和 200 个 donor-donor placebo 的 q95 门禁。
- GSC 按结果变量拟合交互固定效应；MC 是 GSC 失败后的矩阵补全回退。
- 控制身份或 donor 集合冻结后，流程才读取处理后结果。处理后缺失只记录为结果支持信息，不改变 donor。
- Matching、GSC、MC 是独立估计器；路由是项目层的样本与失败处理规则，不是一个“混合估计器”。

## 5. 数据与写入边界

- `data/active/` 是当前冻结研究资产和队列基线。本地工作区只读使用这些内容。
- `outputs/viirs_monthly/` 及 `outputs/` 中的 VIIRS 缓存视为保留资产，不参与本次清理。
- 正式 GPU 运行在服务器的独立工作副本中执行。`--reset-queues` 只允许用于该副本。
- `outputs/` 中除保留的 VIIRS 缓存外，其余审计、preview、staging 和估计器结果均可重建。每次运行使用独立的运行 ID 和规格指纹。
- `data/active/causal/releases/` 是正式发布目录。完整、终态且来源记录齐全的任务才能进入 release。

## 6. 服务器执行主线（4 × RTX 4090）

服务器上先完成 CUDA、R 参考环境、输入哈希和资格凭证检查，再进行正式运行。四张 4090 是四个独立进程/任务槽，不是拼接成一张 96GB 显卡。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count() == 4"

python scripts/causal_gpu/run_shadow_queue.py \
  --formal-qualification --estimators matching,gsc,mc \
  --max-tasks-per-estimator 3 --gpu-ids 0,1,2,3 --retry \
  --output-root outputs/complete_estimators/gpu_formal_qualification

python scripts/causal_gpu/audit_formal_qualification.py \
  --root outputs/complete_estimators/gpu_formal_qualification \
  --output outputs/causal_gpu/formal_qualification.json

python scripts/causal_r/run_parallel_production.py \
  --run-all --estimator-backend python_gpu \
  --qualification-receipt outputs/causal_gpu/formal_qualification.json \
  --gpu-ids 0,1,2,3 --shard-count 4 --workers 4
```

上面的正式命令不包含 `--reset-queues`：当前 active 输入是冻结的，服务器若需建立全新运行，必须先制作独立的服务器工作副本并记录输入哈希，再只在该副本中重建队列。

若服务器副本保留了旧的控制记录，先完成规格核验，再运行
`--phase 1 --repair-stale-controls --estimator-backend python_gpu`；该选项只重算
来源检查未通过的记录，已经合格的记录会被复用。

## 7. 文档权威层级

遇到冲突时按下列顺序判断：

1. 冻结研究决策：[`research/decisions/`](research/decisions/README.md)；
2. 因果主设计：[`research/counterfactual_response_label_design.md`](research/counterfactual_response_label_design.md)；
3. 计量方法与识别：[`research/econometric_methods.md`](research/econometric_methods.md)；
4. 执行诊断：[`research/identification_and_diagnostics.md`](research/identification_and_diagnostics.md)；
5. 算法说明与公式：[`research/matching_and_gsc_methodology.md`](research/matching_and_gsc_methodology.md)、[`research/estimator_formulas.md`](research/estimator_formulas.md)；
6. 运行状态：[`operations/current_project_status.md`](operations/current_project_status.md)；
7. 服务器操作：[`operations/server_deployment.md`](operations/server_deployment.md)。

## 8. 当前状态

代码检查结果、队列数量和服务器待办事项集中记录在
[`operations/current_project_status.md`](operations/current_project_status.md)。
本页提供阅读入口，状态信息统一在状态页维护。
