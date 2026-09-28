# 四族 event study 封版流程（分析闭环优先）

状态：当前最高优先级。旧的 `2026-09-24` 导出（`mit_event_study_results/` 与
`.zip`）视为历史分析，不再修补；最终产物以本流程重新生成的
`mit_event_study_final.zip` 为准。

## 1. 服务器统一重跑（四族 + 同城/跨城 + Match/GSC/MC）

```bash
cd /home/nas/lsr/MIT_summer
python scripts/causal_python/run_all_method_event_study.py \
    --output-directory outputs/event_study_python_final \
    --families housing,viirs,poi,population \
    --matching-window-preset canonical
```

- 如沿用上次隔离队列/任务根的运行方式，追加与规范运行一致的
  `--matching-family-queue`、`--matching-control-queue`、
  `--matching-task-root`；不要混用不同隔离目录。
- 窗口为预注册规范：月度（housing/viirs）±24，年度（poi/population）
  `[-6, +3]`，reference `-1`。
- 完成后检查 `outputs/event_study_python_final/suite_status.csv` 全部
  `completed`，以及 `gsc_mc_paths/event_study_series.csv` 同时包含
  `same_city` 与 `all_city_standardized` 行。

## 2. 下载（保持目录结构）

需要下载：

- `matching_twfe/<family>/` 下全部 csv 与图（png/svg/pdf/tiff）；
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

产物：

| 文件 | 内容 |
|---|---|
| `event_study_panel.csv` | 统一长表：family × method × donor scope × outcome × event time；coefficient、grid/city SE、CI、p、样本量、`event_phase`，已过滤到规范窗口 |
| `availability.csv` | 4 族 × 3 方法 × 2 scope 每格一行：`available` / `no_data` + reason、admitted units、source 范围、`insufficient_clusters` 警告 |
| `pretrend_summary.csv` | Matching 联合 Wald（grid/city cluster）与 pooled GSC/MC cluster 均值检验 |
| `family_selection.csv` / `.md` | 选择记录：VIIRS+POI 进入 downstream causal supervision；Housing/Population 已分析但不进入；记录信号稳定性、前趋势、有效样本、方法一致性 |
| `manifest.json` | 窗口、方法、输入/输出 sha256、行数、downstream 选择、caveats |
| `validation.json` | 结构与窗口检查（error 会阻断干净运行） |
| `figures/`、`source_tables/` | 底层图与审计表原样拷贝 |

## 4. 验收门槛（缺一不可）

1. `validation.json` 的 `ok = true`（0 个 blocking error）；
2. `availability.csv` 24 格齐全——没有数据的格子也必须有一行并写明
   `no_data` 原因，不允许静默缺失；
3. `family_selection.csv` 中 `in_downstream_supervision` 仅
   `viirs`、`poi` 为 true；
4. 跨城格子若 `n_cities < 3`，必须带 `insufficient_clusters` 警告；
5. `manifest.json:caveats` 保留（标准化尺度、t0 非干净 lead、零锚 MC
   降级、仅规范窗口等）。

## 5. 归档

封版完成并人工确认后，将旧的 `mit_event_study_results/` 与
`mit_event_study_results.zip` 移入 `D:\User\Desktop\MIT_summer_archive\`
（不进 git），后续引用只使用 `mit_event_study_final.zip`。
