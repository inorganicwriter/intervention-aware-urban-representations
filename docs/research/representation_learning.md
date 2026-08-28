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
  - **显式干预条件化**（`--conditioning opening_year`，默认关闭）：开通年份分桶嵌入以
    残差方式注入表格流。开通年份是处理前信息，任何新站点推理时都可得，不破坏跨城市
    迁移；默认路径的"条件化"由响应对齐目标隐式实现。
- **目标**：`total = (1-pred_weight)·L_rep + pred_weight·L_pred`
  - `L_rep`：InfoNCE（正样本权重 = 响应相似度）+ 嵌入距离 MSE（`rep_alpha` 混合）；
    self-logit 与无足够共同观测的 pair 均从目标中排除，跨 outcome family 只对可比较
    family 求平均；
  - `L_pred`：逐 cell 预测 MSE，按 `1/SE²` 加权（使用标签的 `standard_error`）。
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

## 3. 评估协议（`evaluation_report.json`）

每个训练运行在最佳 checkpoint 上输出：

| 区块 | 内容 | 回答的问题 |
|------|------|-----------|
| `retrieval` | `nn_corr@k`（总体 + 逐结果族）+ 随机近邻基线 | 嵌入近邻是否共享响应 |
| `bootstrap_ci` | 单元重采样 95% CI | 指标稳定性 |
| `permutation` | 响应打乱 100/200 次的 p 值 | 关联是否优于随机 |
| `raw_feature_baseline` | 原始特征直接检索 | 模型是否胜过"直接用特征" |
| `probe` | 训练池拟合的线性探针 → 目标池 RMSE（嵌入 vs 原始特征） | 迁移能力 |
| `baselines`（test 池） | 随机投影 / PCA（无监督）/ 冻结 DINOv2 / **外观自编码器**（与主模型同结构、无响应监督训练） | 模型是否胜过机会水平与外观-only 基线 |
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

## 5. 复现命令

```powershell
conda run -n mit python scripts/causal_r/build_pretraining_dataset.py `
  --response-release data/active/causal/releases/production_YYYYMMDD `
  --dataset-id production_YYYYMMDD
conda run -n mit urban-train-representation data/model_inputs/production_YYYYMMDD `
  --output outputs/representation/main --epochs 100 --use-images --seeds 1 2 3
conda run -n mit urban-run-ablation data/model_inputs/production_YYYYMMDD --specs specs.json
conda run -n mit urban-build-model-card outputs/representation/main
conda run -n mit urban-export-embeddings outputs/representation/main/best_model.pt `
  data/model_inputs/production_YYYYMMDD --output outputs/representation/embeddings.parquet
conda run -n mit urban-run-ablation data/model_inputs/production_YYYYMMDD `
  --specs configs/representation/ablation_specs.example.json
conda run -n mit urban-summarize-runs outputs/representation/main `
  outputs/representation/seed_1 outputs/representation/seed_2 --output outputs/representation/summary
```

## 6. 与因果管线的边界

- 表示学习**只读** Response Artifact 发布产物与训练前数据集；不参与队列、不写因果产物。
- 训练掩码由发布者决定（`final_training_mask`），训练器不做推断。
- 因果标签的质量等级（`quality_grade`）进入 sample_index，可作可视化着色，不作为训练监督。
