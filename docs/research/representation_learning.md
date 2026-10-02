# 干预条件化城市表示学习（Representation Learning）

本文档描述表示学习模块的研究设计、评估协议与产物，是
[counterfactual_response_label_design.md](counterfactual_response_label_design.md) 的下游。

## 1. 研究问题

表示学习目标是刻画不同地点对同类干预的响应相似性。
表示学习模块把这一点形式化为：

> 学习一个映射 $f: X \to \mathbb{R}^d$（$X$ = 处理前多模态城市特征），使得
> 嵌入空间中的近邻关系 $\text{sim}_\text{emb}(f(x_i), f(x_j))$ 与响应相似性
> $\text{sim}_\text{resp}(r_i, r_j)$（因果响应标签向量在共同观测单元上的相关）
> 一致，并且该一致性**迁移到训练中未见过的城市**。

标签 $r_i$ 由因果管线（Response Artifact）提供，训练只用处理前特征
（leakage 屏障见 `build_pretreatment_features` 的严格正 lag 约束）。

## 2. 模型与训练

- **输入**：处理前年度特征（房价/VIIRS/POI/人口/Sentinel-2，开年前 1-3 年 lag，z 标准化，
  只由训练城市拟合标准化参数）+ 可选街景图像（DINOv2 冻结 backbone + 投影头）。
- **结构**：tabular MLP 编码器 + 图像编码器 → 融合 → 投影头（归一化嵌入）；
  多任务预测头输出 4 个结果族的响应向量（`ResponseEmbeddingModel`）。
  - **图像池化**（`--image-pooling`）：`max`（默认）/ `mean` / `meanmax`（拼接后投影）。
  - **显式干预条件化**（`--conditioning`，默认关闭）：`opening_year` 分桶嵌入与
    `station`（四个 0/1 开通属性 + 终点点缺失位，32 codes）以 L2 归一化、固定比例 0.1
    混入表格流；比例固定是为了避免随机初始化压缩地点差异。开通年份与站点属性都是
    处理前信息，任何新站点推理时都可得，不破坏跨城市迁移；默认路径的"条件化"由响应
    对齐目标隐式实现。
- **目标**：`total = (1-pred_weight)·L_rep + pred_weight·L_pred`
  - `L_rep`：InfoNCE（正样本权重 = 响应相似度）+ 嵌入距离 MSE（`rep_alpha` 混合）；
    self-logit 与无足够共同观测的 pair 均从目标中排除，跨 outcome family 只对可比较
    family 求平均；
  - `L_pred`：逐 cell 加权 MSE。发布数据集始终携带 `response_weight`，实际训练走设计
    权重分支 `sum(m·w·diff²)/sum(m·w)`；`1/SE²` 精度加权只在缺少权重列的旧数据集上
    生效。
- **权重现状**：设计权重在每个网格的各 horizon 上恒定（VIIRS 全部、POI matched 全部），
  而恒定样本权重在加权 Pearson 中被约去；因此 W-design 主要影响预测损失，
  **不重塑成对相似度**。详见
  [`representation_learning_audit.md`](representation_learning_audit.md)。
- **算法优化选项**（均默认关闭或保持原语义，可单独消融）：
  - `--no-se-shrinkage`（默认开）：对比损失中响应相似度按标签可靠性 `1/(1+SE²)` 收缩，
    高噪声配对降权（measurement-error 衰减）；SE 缺失/恒定时退化为原损失。
  - `--queue-size N`：响应感知的 sample-level FIFO 队列（如 4096），在 batch 之外继续
    区分正样本、负样本、不可比较 pair 与同一 unit 的历史副本。
  - `--learnable-temperature`：InfoNCE 温度改为可学习 logit scale（CLIP 风格，初始化
    1/0.07、上限 100），替代固定温度扫描。
  - `--uncertainty-weighted`：表示/预测两个任务权重改为不确定性加权
    （Kendall et al. 2018，`exp(-log_var)` 权重 + log_var 正则），初始化
    `log_var = log 2` 即默认 0.5/0.5 平衡；开启后 `pred_weight` 被忽略。
- **训练协议**：城市级划分（train/validation/test），只读 `final_training_mask` 网格；
  CosineAnnealing、按验证 loss 保存最佳 checkpoint、`--seeds` 多 seed 汇总。
- **逐族独立口径**：VIiRS 与 POI 分别单独训练模型，监督只使用同城的
  matched 与 GSC 标签（数据上 MC 与跨城样本不足或未通过平行趋势诊断）；
  housing / population 不进入下游监督。共享嵌入的融合模型作为后续对照，
  操作流程见 [`../operations/per_family_training_runbook.md`](../operations/per_family_training_runbook.md)。

## 3. 评估协议（`evaluation_report.json`）

每个训练运行在最佳 checkpoint 上输出：

| 区块 | 内容 | 回答的问题 |
|------|------|-----------|
| `retrieval` | `nn_corr@k`（总体 + 逐结果族）+ 随机近邻基线 | 嵌入近邻是否共享响应 |
| `bootstrap_ci` | 单元重采样 95% CI | 指标稳定性 |
| `permutation` | 响应打乱 100/200 次的 p 值 | 关联是否优于随机 |
| `raw_feature_baseline` | 原始特征直接检索 | 模型是否胜过"直接用特征" |
| `probe` | 含截距（训练均值中心化）的逐 cell ridge → 目标池 RMSE（嵌入 vs 原始特征），附训练均值预测基线；ridge 可按表示独立选择（`select_probe_ridge`） | 迁移能力；只有同时优于均值基线才算特征有信号 |
| `baselines`（test 池） | 随机投影 / PCA（无监督；中心与基在训练池拟合后投影 test 池）/ 冻结 DINOv2 / **外观自编码器**（继承主模型 `hidden_dims/embedding_dim`，无响应监督训练） | 模型是否胜过机会水平与外观-only 基线 |
| `transfer.per_city` | 每个未见城市的 `nn_corr@k` | 迁移是否在各城市一致 |
| `transfer.few_shot_probe` | **逐目标城市**抽取 4/8/16/32 个网格的探针 RMSE 曲线（多 seed 均值 ± std） | 少量标签下的城内适应速度，各目标城市分别计算 |
| `transfer.cross_validated_probe` | 目标池 disjoint-fold 交叉验证探针 RMSE | 有监督适配上限；不在拟合样本上回报 RMSE |
| `predictive_transfer` | 逐 cell 响应方向（正/负）ridge 分类的 rank-AUC（嵌入 vs 原始特征） | 表示能否预测新城市响应的方向 |

**基线的意义**：随机投影 = 纯机会水平；PCA = 外观几何（无响应监督）；
冻结 DINOv2 = 最强冻结外观表示；**外观自编码器** = 与主模型同结构、可训练但
只用特征重建目标的外观表示。模型只有同时显著高于全部基线，才能支撑
"学习到的是响应结构而非外观结构"的核心主张。

## 4. 实验追踪与可复现

- 每次训练追加一行 `runs.jsonl`（`output_dir/`），含 `config_sha256`（训练配置内容哈希）、
  `test_metrics`、`test_nn_corr@k`、`baseline_nn_corr@k`，跨运行可比。
- `urban-summarize-runs RUN_DIR... --output summary`：把多个运行目录的 runs.jsonl 汇总为
  论文 Table 1 雏形（CSV + markdown 对比表，含每个基线的 nn_corr@k）。
- `evaluation_report.json` 记录全部统计细节；`training_config.json` 记录输入数据集与
  超参数；模型卡 `urban-build-model-card` 汇总为论文友好格式。
- 消融/超参网格：`urban-run-ablation` 直接消费 specs JSON；正式模板见
  `configs/representation/ablation_specs.example.json`（消融清单）与
  `sweep_specs.example.json`（超参数扫描）。

## 5. 当前状态与可预测性诊断

- 评估修正（提交 `b945ed6`）：probe 含截距与训练均值基线、`precision="float64"`
  相似度选项、station 条件化尺度与缺失位修复
- 诊断脚本：
  [`../../scripts/analysis/representation_diagnostics.py`](../../scripts/analysis/representation_diagnostics.py)
  （标签支撑、权重恒定性、float32 风险、嵌入几何、逐城市/逐 outcome）
- 最小对照：
  [`../../scripts/analysis/representation_controls.py`](../../scripts/analysis/representation_controls.py)
  （城市级 5 折 CV + 内层选参；raw 检索、含截距 ridge、kNN 对照与训练均值基线）
- 结论边界：截至诊断，多 seed 下没有稳定超过 raw 特征或外观基线的增量；
  在可预测性对照确认前不启动多 seed 正式矩阵。审计与实测记录见
  [`representation_learning_audit.md`](representation_learning_audit.md)。

## 6. 复现命令

逐族独立训练（当前主口径；数据集构建、核验、冒烟与消融矩阵见
[`../operations/per_family_training_runbook.md`](../operations/per_family_training_runbook.md)）：

```powershell
conda run -n mit python scripts/causal_r/build_pretraining_dataset.py `
  --response-release data/active/causal/releases/production `
  --dataset-id viirs-mg-design --families viirs --methods matched,gsc --weight-alpha 0
conda run -n mit python scripts/causal_r/build_pretraining_dataset.py `
  --response-release data/active/causal/releases/production `
  --dataset-id poi-mg-design --families poi --methods matched,gsc --weight-alpha 0
conda run -n mit urban-train-representation data/active/model_inputs/viirs-mg-design `
  --output outputs/representation/viirs-mg-design --epochs 100 --seeds 1 2 3
conda run -n mit urban-train-representation data/active/model_inputs/poi-mg-design `
  --output outputs/representation/poi-mg-design --epochs 100 --seeds 1 2 3
conda run -n mit urban-summarize-runs outputs/representation/viirs-mg-design/seed_* `
  outputs/representation/poi-mg-design/seed_* --output outputs/representation/summary
conda run -n mit urban-build-model-card outputs/representation/viirs-mg-design/seed_1
conda run -n mit urban-export-embeddings outputs/representation/viirs-mg-design/seed_1/best_model.pt `
  data/active/model_inputs/viirs-mg-design --output outputs/representation/viirs_embeddings.parquet
conda run -n mit urban-run-ablation data/active/model_inputs/viirs-mg-design `
  --specs configs/representation/ablation_specs.example.json
```

街景是可选输入（默认关闭）：数据集构建加 `--streetview-index`，训练与导出
加 `--use-images`（与 checkpoint 架构一致），见
[`../operations/streetview_pipeline.md`](../operations/streetview_pipeline.md)。

## 7. 与因果管线的边界

- 表示学习**只读** Response Artifact 发布产物与训练前数据集；不参与队列、不写因果产物。
- 训练掩码由发布者决定（`final_training_mask`），训练器不做推断。
- sample_index 的 `quality_grade` 优先取 Response Artifact 的 `label_grade`（未获得
  标签的行已掩为 `unavailable`），仅在缺少该列时回退到设计等级 `quality_grade`；
  它可作可视化着色，不作为训练监督。
