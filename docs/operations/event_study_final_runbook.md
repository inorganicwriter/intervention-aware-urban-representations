# 四族 event study 封版流程（分析闭环优先）

状态：当前最高优先级。旧的 `2026-09-24` 导出（`mit_event_study_results/` 与
`.zip`）视为历史分析，不再修补；最终产物以本流程重新生成的
`mit_event_study_final.zip` 为准。

## 1. 服务器统一重跑（四族 + 同城/跨城 + Match/GSC/MC）

```bash
cd /home/nas/lsr/MIT_summer
python scripts/causal_python/run_all_method_event_study.py \
    --output-directory outputs/event_study_suite \
    --families housing,viirs,poi,population \
    --matching-window-preset canonical
```

- 如沿用上次隔离队列/任务根的运行方式，追加与规范运行一致的
  `--matching-family-queue`、`--matching-control-queue`、
  `--matching-task-root`；不要混用不同隔离目录。
- 窗口为预注册规范：月度（housing/viirs）±24，年度（poi/population）
  `[-6, +3]`，reference `-1`。
- Matching 默认按两个 donor scope 独立分析：`same_city` 写
  `matching_twfe/<family>`，`all_city_standardized` 写
  `matching_twfe_cross/<family>`；family 无跨城 matched 标签时该项记为
  `no_data`，不视为失败（`--matching-donor-scopes` 可显式指定）。
- 完成后检查 `outputs/event_study_suite/suite_status.csv` 全部
  `completed`（`matching_twfe_cross:*` 允许 `no_data`），以及
  `gsc_mc_paths/event_study_series.csv` 同时包含
  `same_city` 与 `all_city_standardized` 行。

## 2. 下载（保持目录结构）

需要下载：

- `matching_twfe/<family>/` 与 `matching_twfe_cross/<family>/` 下全部 csv
  与图（png/svg/pdf/tiff）；
- `gsc_mc_paths/` 下 `event_study_series.csv`、`diagnostics.csv`、
  `pretrend_city_cluster.csv`、`pretrend_grid_cluster.csv`、`figures/`；
- `pooled_outcomes/` 四族合并图；
- `suite_status.csv`、`event_time_support.csv`、`event_time_coverage.csv`、
  `sample_aggregation_coverage.csv`。

## 3. 本地导出封版

```powershell
python scripts/causal_python/export_event_study_final.py `
    --suite-root <下载目录> `
    --output-directory outputs/event_study_final `
    --zip
```

如目录分层（matching 与 pooled 分离），也可显式传
`--matching-root` + `--pooled-root` + `--suite-root`；显式参数优先于
`--suite-root` 的默认推断。

产物（字段语义按阶段拆分，不再出现混用的 `treated_events`）：

| 文件 | 内容 |
|---|---|
| `event_study_panel.csv` | 统一长表：family × method × donor scope × outcome × event time；coefficient、grid/city SE、CI、p、样本量、`event_phase`、推断元数据（`se_type`、`p_value_method`、`n_city_clusters`），已过滤到规范窗口 |
| `availability.csv` | 每格一行（含 `analysis_type`）。Matching：`availability_treated_events`、`event_study_max_n_treated`、`coverage`；Pooled：`max_supported_treated_grids`（不计算 matching 式 coverage）。另有 `available` / `no_data` + reason、source 范围、`insufficient_clusters` 警告 |
| `pretrend_summary.csv` | Matching 联合 Wald（grid/city cluster）与 pooled GSC/MC cluster 均值检验 |
| `family_selection.csv` / `.md` | 选择记录：VIIRS+POI 进入 downstream causal supervision（`selected_family` / `selection_reason`）；matching 保留 `availability_treated_events`、`event_study_max_n_treated`、`n_treated_gap`、`coverage`，pooled 记录 `max_supported_treated_grids`；并记录信号稳定性、前趋势、方法一致性 |
| `manifest.json` | 窗口、方法、输入/输出 sha256、行数、downstream 选择、caveats |
| `validation.json` | 结构与窗口检查；`treated_event_consistency`（仅 matching）违反即 hard fail |
| `figures/`、`source_tables/` | 底层图与审计表原样拷贝 |

读取旧版表时用 `normalize_treated_event_columns()`：matching 行的旧
`treated_events` 映射到 `availability_treated_events`，pooled 行映射到
`max_supported_treated_grids`；新 release 不再写该旧字段。

## 4. 验收门槛（缺一不可）

1. `validation.json` 的 `ok = true`（0 个 blocking error）；
2. `availability.csv` 24 格齐全——没有数据的格子也必须有一行并写明
   `no_data` 原因，不允许静默缺失；
3. `family_selection.csv` 中 `selected_family` 仅 `viirs`、`poi` 为 true，且
   matching 行满足 `coverage ≤ 1`
   （`event_study_max_n_treated ≤ availability_treated_events`；违反时
   `validation.json` 的 `treated_event_consistency` hard fail）；
4. 跨城格子若 `n_cities < 3`，必须带 `insufficient_clusters` 警告；
5. `manifest.json:caveats` 保留（标准化尺度、t0 非干净 lead、零锚 MC
   降级、仅规范窗口、pooled p 为正态近似等）；
6. 每个 `family × method × scope` 组合独立聚合；样本量太少无法估计的组合
   记为 `no_data` 并在该行写明原因（不视为主动舍弃），不影响其他组合。

## 5. 旧结果

旧的 `mit_event_study_results/` 与 `.zip` 已清理；后续引用只使用
`mit_event_study_final.zip`。

## 6. 附：队列状态被代码包覆盖时的恢复

症状：`suite_status.csv` 中 `matching_twfe:*` 全部 `no_data`，而
`gsc_mc_paths` 正常完成（pooled 不读队列，直接读 staging）。

原因：`data/active/causal/*queue*.csv` 是 git 跟踪的冻结基线，`unzip -o`
覆盖了服务器上的终态队列。标签产物与匹配计算不受影响，只恢复账本：

```bash
python scripts/analysis/recover_causal_queues.py \
  --family-queue-backup outputs/causal_labels/qc/outcome_family_work_queue.before_<stamp>.csv \
  --demotions outputs/causal_labels/qc/mc_insufficient_pre_demotions.csv \
  --response-release data/active/causal/releases/production \
  --output-directory outputs/causal_labels/recovered
```

- family 队列 = qc 降级前备份 + 226 条 MC 降级记录；
- control 队列 = release 中 same-city matched 标签的 `control_unit_key`；
- 期望状态计数：skipped 9,162 / matched_labelled 7,553 / gsc_labelled 2,322 /
  mc_labelled 1,155；`matched_pairs_without_matched_label` 应为 0；
  `matched_orders_without_same_city_control` 若大于 0，是只有跨城 matched
  标签的站点（其跨城 component 用 `matching_twfe_cross` 独立分析），属正常；
- 跨城 matched 任务不需要进 same-city 控制队列：匹配分析的跨城 scope 直接读
  manifest 的 `control_unit_key`。

重跑时加：

```
--matching-family-queue outputs/causal_labels/recovered/outcome_family_work_queue.csv \
--matching-control-queue outputs/causal_labels/recovered/control_design_queue.csv
```

生产任务根仍用原来的 `--matching-task-root`（默认
`outputs/causal_labels/tasks`）；不重算任何匹配。
