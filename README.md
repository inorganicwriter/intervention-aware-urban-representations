# Learning How Cities Respond

**Intervention-Conditioned Urban Representations**：从地铁站开通事件中构造逐网格因果响应标签，并用处理前多模态城市特征学习可跨城市迁移的表示。

项目最终需要学习的不是“空间外观是否相似”，而是两个地方在同类干预下是否产生相似响应。当前仓库首先解决标签问题：为每个处理网格寻找或构造可信的未处理反事实，并计算

```text
response(i, h) = observed_outcome(i, h) - counterfactual_outcome(i, h)
```

## 当前实现状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 站点身份、城市归属与竞争交通事件决议 | 已实现 | 人工决议由可追溯应用器编译，原始站点表不被覆盖 |
| 500m 网格处理清单与 1km 空间 donor 排除 | 已实现 | 固定 5,048 个处理网格；空间 donor universe 已生成 |
| 处理前多变量控制匹配 | 已实现 | **6 轮路由**：同城匹配 → 同城 GSC → 同城 MC → 跨城匹配 → 跨城 GSC → 跨城 MC → 显式跳过；选择阶段不读处理后结果 |
| PanelMatch、Abadie–Imbens、Xu GSC | 已实现 | 三个独立的文献估计器；正式参数、路由和审计见 DDR-003/004 |
| 事件研究聚合与平行趋势验证 | 已实现 | `scripts/causal_r/run_event_study_aggregation.R` 汇总 GSC/MC 的逐期 pre-period 系数（均值/SE/CI + 网格级联合检验 + 事件研究图），只接受生产 manifest；匹配路径的平行趋势证据由选择阶段 holdout/placebo q95 门禁与 PanelMatch placebo 承担 |
| 可恢复生产队列 | 已实现 | 原子更新、断点续跑、失败原因和任务级 provenance；队列已重置待正式运行 |
| Response Artifact | 已实现 | 严格发布、完整标签骨架、质量等级、训练 mask、输入/代码哈希 |
| 训练前多模态数据集 | 已实现 | 只读处理前特征；按城市切分；仅用训练城市拟合标准化参数 |
| 5,048 网格正式全量计算 | **全量待执行** | 队列已重置（5,048 控制 + 20,192 族级全 pending）；两阶段匹配 canary 已验证（orders 1–10 正确路由 GSC；orders 906–915 中 3/10 同城匹配、order 906 全族走完 match→GSC→MC→skip 链路），canary 产物已清理；待服务器正式运行 |
| 多模态表示模型与训练器 | 已实现 | 干预条件化表示模型 + 训练器（`src/urban_intervention/representation/`、`urban-train-representation` 入口），已用 demo 数据集完成端到端验证（30 epochs 收敛、无 NaN；按 `final_training_mask` 过滤训练样本）；每次训练输出 `evaluation_report.json`（全池检索 + 逐族 `nn_corr@k`、bootstrap CI、响应打乱 permutation 检验、原始特征基线、线性探针迁移指标、**机会/外观基线**（随机投影 / PCA / 冻结 DINOv2 / 外观自编码器）、**逐城市检索 + few-shot 探针曲线 + 响应方向预测 AUC**），`runs.jsonl` 实验追踪（config 哈希 + 头部指标），`--seeds` 多 seed 汇总；`urban-build-model-card` 生成模型卡，`urban-run-ablation` 运行消融网格（模板见 `configs/representation/`），`urban-export-embeddings` 导出嵌入 parquet，`urban-summarize-runs` 汇总对比表；可选 `--conditioning opening_year` 显式干预条件化与 `--image-pooling max/mean/meanmax` 图像池化；评估协议见 [docs/research/representation_learning.md](docs/research/representation_learning.md) |
| 匹配协变量（本轮新增） | 已实现并接入匹配 | 区位（到主中心/副中心距离）、开通前轨道条件（最近站距离/站数/线路数/closeness）、站点属性（换乘/终点/新线/延长/同期开通）已收集；**两阶段控制选择**：先按处理前结果滞后匹配 M=5 候选，再在其中精炼区位/轨道协变量平衡最优者；共同支持与 holdout/placebo 门禁仍只作用于结果历史特征；SMD 诊断同时报告结果与静态协变量。见 `docs/research/transit_accessibility_method.md` |

因此，“反事实标签和训练前数据生产代码”已经闭环；表示学习模型、训练、跨城市评估与机会/外观基线的端到端管线已经实现（见[表示学习文档](docs/research/representation_learning.md)），待正式响应标签发布后即可在 5,048 网格全量数据上运行正式训练。

## 冻结研究口径

- 研究范围：44 个中国大陆地铁城市。
- 空间单元：固定 `500m × 500m` 网格，主处理为网格内正式地铁站开通。
- 处理清单：5,048 个经站点决议和唯一性审计确认的处理网格。
- donor：从全部非处理网格开始，主规格排除距相关站点网格多边形不足 1km 的单元。
- 时间：站点日期保留到日；月度估计把开通后的第一个完整自然月定义为事件期 1。
- 结果族：房价、VIIRS、POI、人口。各结果族允许因数据支持不同而使用不同反事实或缺失 mask。
- 控制选择：只能使用处理前房价、VIIRS、POI、人口及其可用性；不得读取处理后结果或处理后缺失状态。
- 覆盖优先边界：至少 1 个完整变量族即可进入物理匹配；GSC 失败后的 MC 允许 `min.T0=1`，并在产物中保留实际处理前支持期数。
- 路由：单一物理控制未通过共同支持、holdout 或 placebo 门禁时进入 Xu GSC；GSC 失败后运行 MC；三种路径均失败才明确跳过。

精确定义以 [因果响应标签设计](docs/research/counterfactual_response_label_design.md) 和 [冻结决策](docs/research/decisions/README.md) 为准。

## 仓库架构

```text
src/urban_intervention/
  causal/                 空间 donor、Response Artifact、训练前数据发布
  representation/         干预条件化表示模型、训练器、统计评估、基线与迁移评测
  interventions/transit/  站点人工决议应用器
  pipelines/housing/      房价导入、来源适配和规范面板
  pipelines/poi/          POI 读取、标准化与聚合
  config/                 城市和项目配置
  data/                   数据路径与注册表

scripts/
  causal_r/               正式计量估计器、控制设计和可恢复队列
  collection/             外部数据采集与来源导入
  data/                   因果输入与固定任务清单构建
  data_management/        数据布局、注册表和快照工具
  labels/                 房价等观测结果构建
  analysis/               可重复的数据质量审计；不包含正式估计器

data/
  catalog/                来源、schema、字段映射和人工质量决议
  reference/              固定边界、网格和已决议站点事件
  raw/                    不可变原始材料
  staging/                可重建的来源级中间数据
  curated/                标准化协变量
  labels/                 规范结果变量
  panels/                 分析就绪的规范面板
  causal/                 固定处理清单、donor universe、输入和生产队列

outputs/                  审计、估计器对象与 staging 结果（可重建，不入库）
tests/                    Python 合同测试与 R 估计器门禁
docs/                     研究设计、数据合同和操作说明
```

脚本职责索引见 [scripts/README.md](scripts/README.md)，数据分层合同见 [docs/architecture/data_layout.md](docs/architecture/data_layout.md)。

## 环境

Python 使用项目专属 `mit` conda 环境；R 正式版本和包锁定位于 `scripts/causal_r/RUNTIME_LOCK.csv`。

```powershell
conda env update -n mit -f environment.yml --prune
conda run -n mit python -m pip install -e ".[dev]"
conda run -n mit python -c "import sys; assert sys.version_info[:2] == (3, 11)"
```

R 可位于系统 `PATH` 中，或通过环境变量配置；无需修改代码中的研究规则：

```powershell
$env:MIT_RSCRIPT = (Get-Command Rscript).Source
$env:MIT_R_LIB = (Resolve-Path '.r-lib').Path
$env:MIT_VIIRS_RAW = '<external-monthly-VIIRS-directory>'
```

密钥只允许通过环境变量或未提交的 `config.yaml` 提供。

`--use-images` 训练路径首次运行时会通过 `torch.hub` 下载 DINOv2
（`facebookresearch/dinov2`），可用 `TORCH_HUB` 环境变量指定缓存目录；
torchvision 必须与已安装的 torch 构建匹配（见 `requirements.txt` 注释）。

正式任务统一使用 Python 3.11 的 `mit` 环境。执行前运行
`conda env update -n mit -f environment.yml --prune`，确保包括 `openpyxl`
在内的运行与测试依赖均已安装；仓库要求 Python 3.11 或更高版本。

## 标准工作流

### 1. 基础验证

```powershell
conda run -n mit python scripts/data_management/validate_registry.py
conda run -n mit python -m pytest -q
& $env:MIT_RSCRIPT tests/causal_r/test_complete_estimators.R
```

`tests/causal_r/verify_complete_implementation.R` 依赖正式估计器产物，应在控制设计与标签队列运行后执行，不在基础验证范围内。

### 2. 重建正式因果输入

以下命令会重置生产队列。仅在确认输入快照和哈希后执行：

```powershell
& $env:MIT_RSCRIPT scripts/causal_r/reset_counterfactual_queues.R
& $env:MIT_RSCRIPT scripts/causal_r/build_formal_matching_inputs.R
& $env:MIT_RSCRIPT scripts/causal_r/audit_formal_target_support.R
```

### 3. 运行控制设计与标签队列

先 dry-run，再运行一个有限 canary；未审核 canary 前不要启动全量 5,048 网格。

```powershell
conda run -n mit python scripts/causal_r/run_grid_control_design_queue.py --start-order 1 --max-units 10 --dry-run
conda run -n mit python scripts/causal_r/run_causal_label_queue.py --start-order 1 --max-tasks 4 --dry-run
```

队列语义、正式参数和服务器运行方法见 [scripts/causal_r/README.md](scripts/causal_r/README.md)。

### 4. 发布标签与训练前数据

只有所有结果族任务进入终态后，才能创建正式 release：

```powershell
conda run -n mit python scripts/causal_r/build_response_artifact.py --release-id production_YYYYMMDD
conda run -n mit python scripts/causal_r/build_pretraining_dataset.py --response-release data/active/causal/releases/production_YYYYMMDD --dataset-id production_YYYYMMDD
```

`--allow-partial` 只允许用于 canary 和测试；其产物带非生产标记，不能进入正式训练。
严格发布会验证 treatment 文件哈希、逐处理单元身份、任务目录/manifest/标签
一致性和代码版本。在没有 `.git` 的源码包中，代码版本使用可执行源码树的
`tree-sha256`，不会接受 `unknown`。

## 文档入口

- [服务器部署与生产运行](docs/operations/server_deployment.md)
- [完整项目与生产管线](docs/research/complete_project_pipeline.md)
- [因果响应标签设计](docs/research/counterfactual_response_label_design.md)
- [干预条件化表示学习](docs/research/representation_learning.md)
- [核心研究架构](docs/architecture/research_architecture.md)
- [当前运行状态](docs/operations/current_project_status.md)
- [数据合同](docs/data/README.md)
- [操作手册](docs/operations/README.md)
- [文献依据](docs/research/related_work_literature.md)
- [计量模型公式](docs/research/estimator_formulas.md)

若历史输出、脚本注释或其他说明与冻结 DDR 冲突，以 DDR 和当前生产代码为准。旧 pooled DID、旧年度面板、prototype matching/GSC、smoke 和 partial release 均不是正式响应标签。
