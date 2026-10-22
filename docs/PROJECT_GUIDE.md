# 项目总览与阅读入口

更新：2026-10-13  
状态：因果阶段已闭环；新多模态数据准备、服务器入库与绑定完成，下一步为模型接入和小规模训练验证。

这份文档说明项目的估计对象、结果生成流程和运行入口。具体参数以冻结决策和因果主设计为准，实时队列状态以状态页为准。

当前输入约定：状态使用 POI、VIIRS、街景和独立卫星图像；背景使用完整 AlphaEarth64、房价、人口和城市统计代理；干预使用站点与开通属性。本轮不加入路网。最新资产已在 `/home/nas/lsr/MIT_summer` 完成正式发布，37 项绑定与 17 项路径验证通过；历史缺失和未结束年度保留 masks，干预历史属性仍需来源复核。临时传输与重复暂存已清理，正式数据和审计证据保留。版本、回执及清理范围见 [输入同步记录](operations/input_server_sync_2026-10-13.md)，模型迁移按 [输入契约](research/state_context_input_contract_2026-10-12.md) 验收。

## 1. 项目要解决什么问题

项目题目是 **Learning How Cities Respond: Intervention-Conditioned Urban Representations**。
研究对象是地铁站正式开通后，发生开通事件的 500m × 500m 城市网格相对于其未处理反事实的响应。

对处理网格 `i`、结果变量 `k` 和事件期 `h`，响应标签定义为：

```text
L[i, k, h] = observed outcome[i, k, h] - untreated counterfactual[i, k, h]
```

项目通过严格因果推断构造局部因果响应路径作为监督信号，并通过多模态深度学习（街景全景图、Google AlphaEarth 卫星嵌入、时序表格先验）学习能够跨城市外推预测干预响应的城市空间表征。

## 2. 两层解耦系统与当前阶段

```text
[第一层：因果反事实推断层 (已全量闭环)]
冻结空间与事件资产 (44城 / 5,048处理网格 / 377万donors)
        ↓
控制设计（1km 空间溢出排除，只读处理前信息）
        ↓
匹配 / GSC / MC 六轮级联路由 (服务器4090队列已完成)
        ↓
严格发布 Response Artifact (13.6万行生产标签，含SE、D分数、质量掩码)
        ↓
事件研究最终筛选：收敛至 VIIRS与POI的同城Matching与GSC (Housing/Pop与MC排除)
        ↓
[第二层：干预条件化表征学习层 (当前重点推进)]
多模态高维感知输入：Google AlphaEarth 64维天基场 + 地基街景全景图 (DINOv2)
        ↓
VIGOR 式空天地跨视角注意力协同 (Cross-View Cross-Attention)
        ↓
融合 27 维处理前表格先验 (Housing, POI, Pop, VIIRS) 与 5-bit 站点属性
        ↓
以 Soft Causal Anchoring (加权因果响应预测) 为主干的表征学习
        ↓
四级验证体系：连续轨迹预测 → 规划语义分类 → 严格跨城5折迁移 → UMAP流形解释
```

反事实标签层和表示学习层分开管理。因果推断阶段已经闭环发布，训练器只读发布的 Response Artifact，不再回写因果队列。

## 3. 当前冻结研究范围与监督口径

| 项目 | 当前口径 |
|---|---|
| 城市 | 44 个中国大陆地铁城市 |
| 空间单元 | 固定 500m × 500m 网格 |
| 处理网格 | 5,048 个唯一站点开通网格 |
| 初始合格 donor | 3,771,800 行空间风险集（严格排除处理站点 1km 缓冲区） |
| 因果分析族 | housing、VIIRS、POI、population |
| **表征学习因果监督标签** | **仅使用 VIIRS（6维）与 POI（12维）的同城 Matching 与 GSC**（Housing 稀疏度高、Population 信噪比低、MC 缺锚点，均被排除） |
| 月度事件期 | 开通月为正式 `event_time=0`；主规格省略 `t=-1`；VIIRS 重点期为 1, 3, 6, 12, 18, 24 月 |
| 年度事件期 | 开通年为正式 `event_time=0`；省略 `t=-1`；POI 正向标签期为 1-3 年 |
| **表征输入模态** | **AlphaEarth 64维空间嵌入（天基）+ 街景全景图（地基）+ 27维表格先验 + 5-bit 站点属性** |

月度主估计路径使用 `anticipation=0`，保留省略参考期 `t=-1`、真实开通期 `t=0`
和正向 `t=1:24`。housing/VIIRS 的 `anticipation=6`（省略 `t=-7`）是独立稳健性规格，
不能与主结果混合。Phase 1 控制选择仍可使用更保守的六个月 pre-only 设计缓冲；
该缓冲不改变 Phase 2 主事件研究的参考期。年度路径保留 `t=0:3`。
12 个月 anticipation 和 1.5、2km 空间半径属于额外敏感性规格，运行时需单独记录。

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
