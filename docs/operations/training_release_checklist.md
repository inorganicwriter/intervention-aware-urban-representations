# 训练 release 本地检查与服务器 checklist

执行顺序（分析闭环优先）：

> 四族 ES 封版 → 选 VIIRS+POI → 提取 labels → 附加 design-only weights
> → 本地验证 → 冻结 training_ready → 服务器训练

服务器本轮只保留本文件第 3 节的 checklist，不推进；先完成第 2 节本地闭环。

## 1. 命名约定（防止误归因）

| 名称 | 含义 |
|---|---|
| Historical | 引入标签权重之前的训练器（prediction head 用 `1/SE²` precision） |
| W0 | 无 design differentiation（`gamma=0, alpha=0`），但遵循新训练框架 |
| W-design | `alpha=0` design-only，主训练版本 |
| W-full | `alpha=0.5`，分析/消融 |

不得把 W0 写成 historical baseline。检索指标 `nn_corr` 保持无权重口径，
作为跨权重定义可比的历史指标；`weighted_nn_corr` 是以后的增量，不是当前
blocker。详见 `docs/research/label_reliability_weights.md`。

## 2. 本地闭环状态

- [x] 逐标签权重训练接入（pairwise `sqrt(w_i w_j)`、逐维 prediction
  权重、queue 携带权重；637 tests 通过）。
- [x] 四族 ES 统一导出器与单元测试
  （`scripts/causal_python/export_event_study_final.py`）。
- [x] 训练显存/耗时本地 smoke（真实 forward + backward + optimizer +
  labeled queue，本机 RTX 4060 Laptop 8 GiB）：

  | batch | queue | steps | peak allocated | peak reserved | 中位 step |
  |---:|---:|---:|---:|---:|---:|
  | 64 | 0 | 10 | 21 MB | 26 MB | 0.046 s |
  | 256 | 4096 | 30 | 295 MB | 938 MB | 0.075 s |
  | 512 | 8192 | 20 | 1111 MB | 3728 MB | 0.159 s |

  结论：计划配置 B=256 / Q=4096 显存与耗时都很宽裕；stress 配置
  （512×8192）也无 OOM、无碎片尖峰。**无需 queue chunking**，权重数学
  定义保持不变。脚本可复跑：
  `python scripts/representation/smoke_weighted_training.py --batch-size 256 --queue-size 4096 --steps 30`。
- [x] 四族 ES 服务器重跑与封版（跨城 `no_data` 属正常；本地封版
  `outputs/event_study_final`，validation 通过；见
  `docs/operations/event_study_final_runbook.md`）
- [x] VIIRS+POI label 提取：`data/active/model_inputs/viirs-poi-design`
  （`--families viirs,poi --weight-alpha 0`）
- [x] 逐标签权重核验（第 3 节清单全部通过；`viirs-poi-full` 消融版一并核验）
- [x] 冻结 `training_ready` 发布（名称按内容，不带版本号）
- [ ] 服务器训练（W-design 主版本；随后 W0 / W-full 消融）

## 3. 服务器 checklist

前置：第 2 节全部完成，且冻结版 zip 的 sha256 已记录。

1. **上传代码**：zip 传到 `/home/lsr/`，解压时排除状态型队列文件——
   服务器生产队列状态不能跟着代码包回滚：

   ```bash
   unzip -o /home/lsr/MIT_summer_code.zip -d /home/nas/lsr/MIT_summer \
     -x 'data/active/causal/*queue*.csv'
   ```

   确认源码与冻结 commit 一致。若队列已被覆盖（症状：`matching_twfe:*`
   全 `no_data`、pooled 正常），按
   [`event_study_final_runbook.md`](event_study_final_runbook.md) 第 6 节恢复，
   不重算任何匹配。
2. **测试**：`python -m pytest tests/unit/test_representation_weighted_response.py
   tests/unit/test_event_study_release.py tests/unit/test_pretraining_dataset.py -q`。
3. **ES 封版**：按 `event_study_final_runbook.md` 重跑四族并导出
   `mit_event_study_final.zip`。
4. **数据集构建**（逐族独立、同城、matched+GSC；完整流程见
   [`per_family_training_runbook.md`](per_family_training_runbook.md)）：

   ```bash
   python scripts/causal_r/build_pretraining_dataset.py \
       --response-release data/active/causal/releases/production \
       --dataset-id viirs-mg-design --families viirs \
       --methods matched,gsc --weight-alpha 0
   python scripts/causal_r/build_pretraining_dataset.py \
       --response-release data/active/causal/releases/production \
       --dataset-id poi-mg-design --families poi \
       --methods matched,gsc --weight-alpha 0
   ```

   W-full 消融把 `--weight-alpha` 换成 `0.5`、dataset-id 用 `*-mg-full`；
   W0 用 `--weight-gamma 0 --weight-alpha 0`、dataset-id 用 `*-mg-w0`；
   三方法对照去掉 `--methods`、dataset-id 用 `*-3m-design`。
   封版路径不带日期：release 固定为 `data/active/causal/releases/production`。

5. **权重分布核验清单**（逐条贴回结果）：
   - `response_weight` 在 `(method, family)` 内 eligible 均值 = 1；
   - `eligible = false` 的行 `response_weight` 全为 0；
   - matched ties（同 outcome 同分位）与 GSC 分布合理，无冻结常量；
   - VIiRS 1 个 outcome；POI 四个 outcome 的权重独立（不因 task-level min 被拉平）；
   - percentile pooling 在 n≥30 分层生效（小 outcome 上溯 family）。
6. **训练**：逐族独立模型（VIiRS、POI 各一），主版本 W-design；
   消融 W0 / W-full / 三方法 / 后续融合对照，实验表使用第 1 节命名。
7. **性能 profiling**：正式大规模训练时再记录吞吐与显存曲线（非当前阻塞）。
