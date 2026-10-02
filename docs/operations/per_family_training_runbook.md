# 逐族表示学习训练流程

当前训练口径：**VIIRS 与 POI 各自独立训练一个模型**，监督只使用同城的
matched 与 GSC 标签（`--scope-view same_city --methods matched,gsc`）；
housing / population 与 MC 标签不进入训练（数据原因：样本不足、缺数据或
未通过平行趋势诊断）。融合模型留待两族结果确认后再评估。

## 监督矩阵（数据决定）

| 变量 | 保留（同城） | 排除（原因） |
|---|---|---|
| VIIRS | matched、GSC | MC（城市簇过少，检验无功效）；跨城（样本不足） |
| POI | matched、GSC | MC（pre-trend 被拒绝）；跨城（样本不足） |

标签按 cell 展开：VIiRS 每格 6 个 horizon，POI 每格 12 个（4 outcome × 3 年）。

## 数据集命名

| 数据集 | 变量 | 方法 | 权重 | 用途 |
|---|---|---|---|---|
| `viirs-mg-design` / `poi-mg-design` | 单族 | matched+GSC | design-only | **主训练** |
| `*-mg-full` | 单族 | matched+GSC | design × precision | W-full 消融 |
| `*-mg-w0` | 单族 | matched+GSC | 单位权重 | W0 基线 |
| `*-3m-design` | 单族 | 三方法 | design-only | MC 影响消融 |
| `viirs-poi-mg-design` | 两族联合 | matched+GSC | design-only | 后续融合对照 |

## 1. 构建数据集

```bash
python scripts/causal_r/build_pretraining_dataset.py \
  --response-release data/active/causal/releases/production \
  --dataset-id viirs-mg-design --families viirs --methods matched,gsc --weight-alpha 0
python scripts/causal_r/build_pretraining_dataset.py \
  --response-release data/active/causal/releases/production \
  --dataset-id poi-mg-design --families poi --methods matched,gsc --weight-alpha 0
# 消融：把 --weight-alpha 换成 0.5（*-mg-full）或加 --weight-gamma 0（*-mg-w0），
#       去掉 --methods（*-3m-design）；融合对照用 --families viirs,poi。
```

## 2. 数据集核验

```bash
python - <<'PY'
import pandas as pd
for name in ("viirs-mg-design", "poi-mg-design"):
    x = pd.read_parquet(f"data/active/model_inputs/{name}/response_targets.parquet",
                        columns=["treatment_order","method","outcome_family","outcome",
                                 "eligible","response_weight","final_training_mask"])
    print(name, "| grids:", x.treatment_order.nunique(),
          "| eligible:", int(x.eligible.sum()),
          "| training cells:", int((x.eligible & x.final_training_mask).sum()))
    print(x.groupby(["method","outcome_family"])["response_weight"]
           .apply(lambda s: s[x.loc[s.index,"eligible"]].mean()).to_string())
    print("eligible zero:", int(x.loc[x.eligible,"response_weight"].eq(0).sum()),
          "| ineligible non-zero:", int(x.loc[~x.eligible,"response_weight"].ne(0).sum()),
          "| outcomes:", x.loc[x.eligible].groupby("outcome_family")["outcome"].nunique().to_dict())
PY
```

发布端已强制：过滤后无 eligible 标签、或 eligible 权重非正，都会直接拒绝出集。
W-full 前另按方法统计 release 的两列覆盖（非有限 SE 现在按中性精度处理，
不再产生零权重）：

```bash
python - <<'PY'
import pandas as pd
x = pd.read_parquet("data/active/causal/releases/production/response_artifact.parquet",
                    columns=["method", "pre_observed_periods", "standard_error"])
print(x.groupby("method")[["pre_observed_periods", "standard_error"]]
       .agg(["count", "isna"]).to_string())
PY
```

## 3. 训练前冒烟（先跑 VIiRS，再跑 POI）

**训练前门禁（先诊断，后训练）**：正式训练前必须先跑诊断与最小对照——

```bash
python scripts/analysis/representation_diagnostics.py \
  --dataset data/active/model_inputs/viirs-mg-design \
  --output-directory outputs/representation_diagnostics/viirs
python scripts/analysis/representation_diagnostics.py \
  --dataset data/active/model_inputs/poi-mg-design \
  --output-directory outputs/representation_diagnostics/poi
python scripts/analysis/representation_controls.py \
  --dataset data/active/model_inputs/viirs-mg-design \
  --output-directory outputs/representation_controls/viirs
python scripts/analysis/representation_controls.py \
  --dataset data/active/model_inputs/poi-mg-design \
  --output-directory outputs/representation_controls/poi
```

判读：对照中 `ridge` / `knn` 的 `improvement_vs_baseline`（正 = 优于训练均值）
与 `retrieval.mean` 决定下一步——若两者均 ≈0 且检索 ≈ raw，说明当前特征没有城市外
可预测信号，停止模型侧调参，转而补充处理前特征或调整论文定位；只有当特征显示可预测
信号、而表示模型仍不超 raw 时，才处理目标函数冲突。审计与实测记录见
[`../research/representation_learning_audit.md`](../research/representation_learning_audit.md)。

站点条件化实验需先构建属性表：
`python scripts/analysis/build_station_attributes.py`
（输出 `outputs/causal_labels/station_attributes/station_attributes.parquet`）。

```bash
conda run -n mit urban-train-representation data/active/model_inputs/viirs-mg-design \
  --output outputs/representation/viirs-mg-smoke \
  --epochs 10 --device cuda --seeds 1 --batch-size 256 --queue-size 4096
```

看 `evaluation_report.json`（按 split 分组）：
- `test.raw_feature_baseline.overall.nn_corr@k`：原始特征是否已含信号；
- `test.retrieval.overall.nn_corr@k` 是否高于该基线并随 epoch 上升；
- `test.permutation.p_value` 是否显著；
- `test_metrics.json` 的汇总字段名是 `mean_nn_corr@k`（不是 `nn_corr@k`）。

## 4. 正式训练

```bash
tmux new -s train
conda run -n mit urban-train-representation data/active/model_inputs/viirs-mg-design \
  --output outputs/representation/viirs-mg-design \
  --epochs 100 --device cuda --seeds 1 2 3 --batch-size 256 --queue-size 4096
conda run -n mit urban-train-representation data/active/model_inputs/poi-mg-design \
  --output outputs/representation/poi-mg-design \
  --epochs 100 --device cuda --seeds 1 2 3 --batch-size 256 --queue-size 4096
```

- 可选街景：加 `--use-images`（需带 `--streetview-index` 重建的数据集，
  见 [`streetview_pipeline.md`](streetview_pipeline.md)）
- 导出嵌入时保持相同开关（`--use-images` 与 checkpoint 一致）

## 5. 产物与成功判据

- 每个 seed：`best_model.pt`、`training_config.json`、`training_history.json`、
  `test_metrics.json`、`evaluation_report.json`、`runs.jsonl`
- 汇总：`urban-summarize-runs outputs/representation/viirs-mg-design/seed_* \
  outputs/representation/poi-mg-design/seed_* --output outputs/representation/summary`
- 成功判据：`nn_corr@k` 显著高于四类基线（随机投影 / PCA / 冻结 DINOv2 /
  外观自编码器）、未见城市结果一致、跨 seed 稳定

## 6. 消融与后续融合

- 消融：W0（`*-mg-w0`）、W-full（`*-mg-full`）、三方法（`*-3m-design`）
- 融合：在两族独立结果确认后，用 `viirs-poi-mg-design` 训练共享嵌入 +
  每族头的联合模型，与两个独立模型对比检索与迁移指标；若无增益则维持独立模型。
