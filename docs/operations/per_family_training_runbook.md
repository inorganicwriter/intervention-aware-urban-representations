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

## 3. 训练前冒烟（先跑 VIiRS，再跑 POI）

```bash
conda run -n mit urban-train-representation data/active/model_inputs/viirs-mg-design \
  --output outputs/representation/viirs-mg-smoke \
  --epochs 10 --device cuda --seeds 1 --batch-size 256 --queue-size 4096
```

看 `evaluation_report.json`：
- `raw_feature_baseline` 的 `nn_corr@k`：原始特征是否已含信号；
- `retrieval.overall.nn_corr@k` 是否高于随机基线并随 epoch 上升；
- `permutation` 的 p 值是否显著。

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
