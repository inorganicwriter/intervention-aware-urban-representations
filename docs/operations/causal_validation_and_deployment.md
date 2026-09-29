# 因果代码验证、服务器覆盖与生产复核命令

更新：2026-09-04
适用环境：本地 Windows 工作区与 Linux 生产服务器 `/home/nas/lsr/MIT_summer`
主规格：Python/GPU；四类 outcome 统一以 `t=-1` 为参考；月度 anticipation=0；`t=0` 为真实开通冲击期；housing 和 VIIRS 均使用逐月 observation window=1；年度 window=1；跨城 donor 上限 5,000。housing/VIIRS 另以 `t=-7`（anticipation=6）做独立稳健性分析。事件研究主窗口预注册为与标签期限对齐的有界窗口：月度族（housing/VIIRS）`[-24, +24]` 个月，年度族（POI/population）`[-6, +3]` 年；窗口敏感性（月度 ±12/±36、年度 `[-6, +10]`）单独报告，主检验为城市聚类联合检验。

本文是因果流程的单一命令清单。命令均写成单行，不使用反斜杠续行。R 只用于冻结参考和 parity；通过资格后，Phase 1–3 的正式生产只依赖 Python/CUDA。

## 1. 本地打包（不包含数据、outputs 和打包文件）

在 PowerShell 中运行：

```powershell
cd D:\User\Desktop\MIT_summer
$archive='D:\User\Desktop\MIT_summer_code.tar.gz'
$sidecar="$archive.sha256"
tar --create --gzip --file $archive --exclude=.git --exclude=data --exclude=outputs --exclude=.pytest_cache --exclude=.ruff_cache --exclude=__pycache__ --exclude='*.pyc' --exclude='*.pyo' --exclude='*.tar.gz' --exclude='*.sha256' --exclude='*.pptx' src scripts tests docs pyproject.toml requirements.txt environment.yml README.md config.yaml.template Dockerfile.R
$hash=(Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower()
"$hash  MIT_summer_code.tar.gz" | Set-Content -LiteralPath $sidecar -Encoding ascii
Get-Item $archive,$sidecar | Select-Object Name,Length,LastWriteTime
Get-Content $sidecar
```

把 `MIT_summer_code.tar.gz` 和 `MIT_summer_code.tar.gz.sha256` 上传到服务器 `/home/lsr/`。不要把压缩包放进项目目录，也不要把它提交到 Git。

## 2. 服务器校验并覆盖代码

先确认没有生产进程仍在写任务文件：

```bash
pgrep -af 'run_parallel_production.py|run_grid_control_design_queue.py|run_control_design_batch.py|run_causal_label_queue.py|run_causal_label_round_queue.py'
```

若该命令没有输出，执行校验和覆盖：

```bash
cd /home/lsr
sha256sum -c MIT_summer_code.tar.gz.sha256
tar -tzf MIT_summer_code.tar.gz | grep -E '(^|/)src/urban_intervention/causal/pooled_event_study.py$|(^|/)scripts/causal_python/run_all_method_event_study.py$|(^|/)docs/operations/causal_validation_and_deployment.md$'
tar --extract --gzip --file /home/lsr/MIT_summer_code.tar.gz --directory /home/nas/lsr/MIT_summer --no-same-owner --no-same-permissions --overwrite
conda activate mit
cd /home/nas/lsr/MIT_summer
grep -n 'restrict_to_common_event_support' src/urban_intervention/causal/pooled_event_study.py
grep -n 'sample_aggregation_coverage.csv' scripts/causal_python/run_all_method_event_study.py
```

`tar` 成功时通常无输出。`grep` 必须返回对应代码行；否则不要启动生产。

## 3. 覆盖后的基础门禁

```bash
export MIT_PROJECT_ROOT=/home/nas/lsr/MIT_summer
python -c "import torch; print(torch.__version__,torch.version.cuda,torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count()==4"
python -m compileall -q src scripts tests
python -m pytest tests/unit -q
python -m ruff check src scripts tests
test -d data/active/causal && echo 'causal inputs present'
test -d outputs/viirs_monthly && echo 'VIIRS cache present'
```

预期：`compileall` 和 Ruff 成功时可无输出；pytest 全部通过；CUDA 返回 4 张卡。

## 4. 形式资格与代码指纹

若 Matching、GSC 或 MC 任一资格绑定源码发生变化，必须重跑对应的 3 个 parity 任务。绑定范围是三个方法各自的 `estimator_source_files`（含公共文件）加上 `gpu/workers.py`；只改该范围之外的代码（队列安全、Response Artifact 发布、事件研究 admission、表示学习、文档）不需要重跑 parity，直接对已有 shadow 根目录重新运行 `audit_formal_qualification.py` 即可。

绑定范围内的改动若经复核确认不涉及估计数值（注释、类型注解、纯错误路径），可以签发 Drift Waiver 复用现有凭证：

```bash
python scripts/causal_gpu/issue_qualification_waiver.py \
  --receipt outputs/causal_gpu/formal_qualification.json \
  --reason "<reviewed diff summary>" --confirm-no-numerical-change
```

生产运行时加 `--qualification-waiver outputs/causal_gpu/qualification_drift_waiver.json`（或设置 `MIT_CAUSAL_QUALIFICATION_WAIVER`）。Waiver 绑定凭证摘要、当前估计器指纹和全部绑定源码哈希；parity、质量、收敛、数值环境和绑定输入任一不合格时，签发脚本会拒绝，不能用 waiver 绕过。任何可能影响估计数值的改动仍必须重跑对应 parity。

当前 Matching/VIIRS 处理前特征与控制设计代码发生变化时，至少重跑 Matching；为避免遗漏，发布前推荐三方法一起跑：

```bash
python scripts/causal_gpu/run_shadow_queue.py --formal-qualification --estimators matching,gsc,mc --max-tasks-per-estimator 3 --gpu-ids 0,1,2,3 --retry --rebuild-python-contracts --output-root outputs/complete_estimators/gpu_formal_qualification
python scripts/causal_gpu/audit_formal_qualification.py --root outputs/complete_estimators/gpu_formal_qualification --output outputs/causal_gpu/formal_qualification.json
python -c 'import json; p="outputs/causal_gpu/formal_qualification.json"; d=json.load(open(p,encoding="utf-8")); print("eligible=",d["eligible"]); print("counts=",d["counts"]); print("reasons=",d["reasons"]); assert d["eligible"] is True and not d["reasons"] and all(d["counts"].get(k,0)>=3 for k in ("matching","gsc","mc"))'
```

若仅 Matching 指纹改变且 GSC/MC manifest 已与当前指纹一致，可把第一条的 `--estimators matching,gsc,mc` 改成 `--estimators matching`；最终 audit 仍必须看到三方法各至少 3 条且 `eligible=true`。

## 5. Phase 1：控制设计

控制设计源码或特征发生变化后，旧控制记录必须按来源检查修复。服务器实测 Phase 1 可使用 24 workers；这一阶段主要是 CPU，不以 GPU 利用率判断是否正常。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 python scripts/causal_r/run_parallel_production.py --phase 1 --estimator-backend python_gpu --qualification-receipt outputs/causal_gpu/formal_qualification.json --gpu-ids 0,1,2,3 --shard-count 4 --workers 24 --repair-stale-controls 2>&1 | tee outputs/phase1.log
```

完成检查：

```bash
python -c "import pandas as pd; q=pd.read_csv('data/active/causal/control_design_queue.csv'); print(q['status'].value_counts(dropna=False).to_string()); print('rows=',len(q),'unique_orders=',q['treatment_order'].nunique()); assert len(q)==5048 and q['treatment_order'].nunique()==5048 and q['status'].isin(['matched','gsc_pending']).all()"
```

`matched + gsc_pending` 必须等于 5,048。warning 需按日志逐条判断；不能只凭进程退出码认定设计正确。

## 6. Phase 2：正式 causal labels

### 6.1 全量

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 python scripts/causal_r/run_parallel_production.py --phase 2 --estimator-backend python_gpu --qualification-receipt outputs/causal_gpu/formal_qualification.json --gpu-ids 0,1,2,3 --shard-count 4 --task-scheduling global_rounds --run-mode production --price-measure main --window 1 --anticipation-months 0 --transaction-count-threshold 1 --max-gsc-cross-city-donors 5000 2>&1 | tee outputs/phase2.log
```

### 6.2 已冻结的汇报样本

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 python scripts/causal_r/run_parallel_production.py --phase 2 --tasks-file outputs/causal_labels/presentation_sample_tasks.csv --estimator-backend python_gpu --qualification-receipt outputs/causal_gpu/formal_qualification.json --gpu-ids 0,1,2,3 --shard-count 4 --task-scheduling global_rounds --run-mode production --price-measure main --window 1 --anticipation-months 0 --transaction-count-threshold 1 --max-gsc-cross-city-donors 5000 2>&1 | tee outputs/phase2_sample.log
```

四张 GPU 最多使用四个 shard。`global_rounds` 的顺序是：全部同城 Matching → 同城 GSC → 同城 MC → 跨城 Matching → 跨城 GSC → 跨城 MC。主规格必须显式写 `--anticipation-months 0 --window 1`。重复同一命令可从持久化 shard 续跑；不要为续跑添加 `--reset-queues`。

可选的 `--task-scheduling round_pool` 使用 SQLite 状态库、严格六轮屏障和动态领任务，
要求 `--phase 2 --queue-phase all`，不能与 `--dry-run` 组合。用 `--round-pool-db`
和 `--round-pool-run-root` 指定状态库与私有队列目录，续跑传 `--round-pool-run-id`，
硬失败任务用 `--requeue-failed` 重新入队；运行状态用
`scripts/causal_python/round_pool_status.py --db ... --run-id ...` 查看。调度模式
不改变估计合同，任务产物与 `global_rounds` 一致。

实时监控：

```bash
pgrep -af 'run_parallel_production.py|run_causal_label_round_queue.py|run_causal_label_queue.py'
nvidia-smi dmon -s pucm -d 2
tail -n 80 outputs/phase2.log
```

### 6.3 月度 `t=-7` 稳健性规格（与主规格完全隔离）

先从目标任务清单中只保留 housing/VIIRS，然后使用独立 queue、task root 和 estimator root。不要对这组命令添加 `--reset-queues`，也不要复用主规格输出目录。

```bash
python -c "import pandas as pd; p='outputs/causal_labels/presentation_sample_tasks.csv'; q='outputs/causal_labels/presentation_sample_monthly_tasks.csv'; x=pd.read_csv(p); x=x[x['outcome_family'].isin(['housing','viirs'])]; x.to_csv(q,index=False); print(x['outcome_family'].value_counts().to_string(), '\nrows=',len(x))"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 python scripts/causal_r/run_parallel_production.py --phase 2 --tasks-file outputs/causal_labels/presentation_sample_monthly_tasks.csv --estimator-backend python_gpu --qualification-receipt outputs/causal_gpu/formal_qualification.json --gpu-ids 0,1,2,3 --shard-count 4 --task-scheduling global_rounds --run-mode production --price-measure main --window 1 --anticipation-months 6 --transaction-count-threshold 1 --max-gsc-cross-city-donors 5000 --queue-namespace robust_m7 --task-output-root outputs/causal_labels/tasks_robust_m7 --estimator-output-root outputs/complete_estimators/python_production_robust_m7 2>&1 | tee outputs/phase2_robust_m7.log
python scripts/causal_r/run_parallel_production.py --phase 3 --tasks-file outputs/causal_labels/presentation_sample_monthly_tasks.csv --shard-count 4 --run-mode production --queue-namespace robust_m7 --task-output-root outputs/causal_labels/tasks_robust_m7 --estimator-output-root outputs/complete_estimators/python_production_robust_m7
```

首次运行会复制已完成的 Phase 1 控制设计，但把独立 family/unit queue 重置为 pending。主规格的三个 master queues、task manifests 和 estimator paths 均不会被修改。

## 7. Phase 3：合并队列

样本任务先核对终态，再合并：

```bash
python scripts/causal_r/run_parallel_production.py --phase 3 --tasks-file outputs/causal_labels/presentation_sample_tasks.csv --shard-count 4 --run-mode production
```

全量运行则移除 `--tasks-file`。Phase 3 必须报告 `Terminal tasks = total tasks`，并清理四个 shard queue 文件。

## 8. 发布与标签汇总

样本允许 partial release；全量正式发布不要加 `--allow-partial`：

```bash
python scripts/causal_r/build_response_artifact.py --release-id presentation_sample_20260902 --tasks-file outputs/causal_labels/presentation_sample_tasks.csv --allow-partial
python scripts/analysis/summarize_causal_labels.py --response-release data/active/causal/releases/presentation_sample_20260902 --tasks-file outputs/causal_labels/presentation_sample_tasks.csv
python -c "import pandas as pd; r='outputs/causal_labels/summary_presentation_sample_20260902'; print(pd.read_csv(r+'/label_generation_by_family.csv').to_string(index=False)); print(pd.read_csv(r+'/failure_reasons_by_family.csv').to_string(index=False)); print(pd.read_csv(r+'/label_distribution_by_family.csv').to_string(index=False))"
```

## 9. 所有样本的 pooled event-study

入口按任务文件纳入所有可接纳样本，但不跨 outcome 量纲混合：每个结果族内分别汇总 Matching、GSC、MC。事件研究诊断不再把图形机械截断为固定 3 年；它先保留每个 outcome 的全部可用 clean history，再按该 outcome 实际存在的方法计算最长共同重叠区间：

- 主规格：四类 outcome 都省略 `t=-1`，`t=0` 是正式冲击期；平行趋势检验仅使用 `t<-1`；
- 月度稳健性：housing/VIIRS 单独省略 `t=-7`，把 `-6:-1` 作为可能的提前反应系数显示，但 clean-pre 检验仅使用 `t<-7`；
- housing/VIIRS 均使用逐月 window=1，不做 post moving-window 聚合；年度 window=1；
- Matching、GSC、MC 的 t=0 都保留；主规格下 housing 和 VIIRS 的正向 horizon 均按逐月 window=1；
- 比较图使用各可用方法区间的交集，不用某一种方法的额外历史冒充跨方法证据；
- `-6:-1` 月度 anticipation 不参加平行趋势检验，但不再从图形删除，因此曲线不会在处理边界前人为中断。重新生成的 GSC/MC 产物必须保存 t=0；旧产物若未保存 t=0，只能标记为旧口径并重跑，不能伪造该点。

```bash
PYTHONUNBUFFERED=1 python scripts/causal_python/run_all_method_event_study.py --staging-root outputs/complete_estimators/python_production --tasks-file outputs/causal_labels/presentation_sample_tasks.csv --output-directory outputs/event_study_presentation_sample --latest-pre-periods 5 --observation-window 1 --reference-event-time -1 2>&1 | tee outputs/event_study_presentation_sample.log
python -c "import pandas as pd; r='outputs/event_study_presentation_sample'; print(pd.read_csv(r+'/suite_status.csv').to_string(index=False)); print(pd.read_csv(r+'/sample_aggregation_coverage.csv').to_string(index=False)); assert pd.read_csv(r+'/suite_status.csv')['status'].eq('completed').all()"
python -c "import pandas as pd; r='outputs/event_study_presentation_sample'; x=pd.read_parquet(r+'/gsc_mc_paths/effect_paths_with_pretrend.parquet'); m=x.loc[x.frequency.eq('monthly')]; assert set(m.loc[m.outcome_family.isin(['housing','viirs']),'observation_window'])=={1}; assert set(x['reference_event_time'])=={-1}; assert x.loc[x.event_time.eq(0),'label_available'].any(); print(pd.read_csv(r+'/event_time_support.csv').to_string(index=False)); print(pd.read_csv(r+'/event_time_coverage.csv').to_string(index=False))"
```

`event_time_support.csv` 给出每种方法自身的最早/最晚时期及该 outcome 的共同区间；`event_time_coverage.csv` 给出每个事件期实际贡献的 treated grids。最长尾部时期若只剩很少网格，只能作为探索性趋势，不能与支持充分的中心时期同等解释。

月度 `t=-7` 稳健性图使用对应的独立 Phase 2 产物：

```bash
PYTHONUNBUFFERED=1 python scripts/causal_python/run_all_method_event_study.py --staging-root outputs/complete_estimators/python_production_robust_m7 --tasks-file outputs/causal_labels/presentation_sample_monthly_tasks.csv --families housing,viirs --matching-family-queue data/active/causal/outcome_family_work_queue_robust_m7.csv --matching-control-queue data/active/causal/control_design_queue_robust_m7.csv --matching-task-root outputs/causal_labels/tasks_robust_m7 --output-directory outputs/event_study_presentation_sample_robust_m7 --latest-pre-periods 5 --observation-window 1 --reference-event-time -7 2>&1 | tee outputs/event_study_presentation_sample_robust_m7.log
python -c "import pandas as pd; r='outputs/event_study_presentation_sample_robust_m7'; x=pd.read_parquet(r+'/gsc_mc_paths/effect_paths_with_pretrend.parquet'); assert set(x['reference_event_time'])=={-7}; assert x.loc[x.event_time.eq(0),'label_available'].any(); print(pd.read_csv(r+'/suite_status.csv').to_string(index=False))"
```

如果正式 estimator paths 被复制到单独目录，应把 `--staging-root` 指向那个目录；不能指向只含汇总 labels、没有 estimator manifest 的 release。

## 10. 因果论证复核

### 10.1 平行趋势

```bash
python -c "from pathlib import Path; import pandas as pd; root=Path('outputs/event_study_presentation_sample'); files=sorted(root.rglob('parallel_trends_wald_city_cluster.csv'))+sorted(root.rglob('pretrend_city_cluster.csv')); [print('\n',p,'\n',pd.read_csv(p).to_string(index=False)) for p in files]"
```

主解释优先城市聚类结果。`p >= 0.05` 只表示在当前样本和功效下未拒绝处理前系数为零，不等于证明平行趋势。还必须同时检查 clean leads 的图形方向、置信区间、城市数、样本量和 pre-RMSPE。t=0 是开通边界，不进入 clean-pre 联合检验。

### 10.2 匹配平衡

```bash
python scripts/analysis/build_balance_loveplot.py --out-dir outputs/figures/causal_balance
python -c "import pandas as pd; p='outputs/figures/causal_balance/balance_smd.csv'; x=pd.read_csv(p); print(x.to_string(index=False)); print('\nabs SMD summary\n',x.filter(regex='smd').abs().describe().to_string())"
```

同时复核 Phase 1 的共同支持、holdout、donor-donor placebo 和空间 1 km 排除。Love plot 只是其中一项，不能替代门禁记录。

### 10.3 标签支持与失败原因

必须分别报告四类 outcome 的 complete/partial/failed、主要失败原因、分布与 Tukey 异常值。housing 因交易稀疏可以明显低于其他结果族；不能把 post 缺失通过重新抽样或改变 donor 来隐藏。

## 11. 何时必须重跑什么

| 修改范围 | 资格 | Phase 1 | Phase 2 | event-study |
|---|---|---|---|---|
| 仅文档、图形样式 | 否 | 否 | 否 | 重新出图 |
| pooled event-study 读取/共同支持 | 否 | 否 | 否 | 必须重跑 |
| Matching 特征、距离或控制门禁 | Matching（发布前推荐全审计） | 必须 | 必须 | 必须 |
| GSC/MC 数值算法或推断 | 对应方法 | 否，除非 donor 设计也变 | 对应任务 | 必须 |
| window、anticipation、price measure、交易阈值 | 对应资格/规格审计 | 视 Matching 规格而定 | 必须 | 必须 |

不要因事件研究图不理想而事后修改窗口、抽样或门禁。任何敏感性规格必须使用独立输出目录，并与主规格并列报告。

（历史记录）本次 `t=0` 修复曾改变资格绑定源码，当时要求重新生成三方法
qualification receipt；该问题最终以 attestation 形式的 qualification drift
waiver 与 control-design drift waiver 解决，生产未重跑 parity。Phase 1 的
实际匹配特征和 6 个月设计缓冲没有改变，`--repair-stale-controls` 复核后
Phase 2 已完成并发布严格 release（见
[`current_project_status.md`](current_project_status.md) 的资格章节）。

## 12. 四族 event-study 最终封版（当前分析阶段）

Phase 1/Phase 2 已在服务器完成，严格 release
`data/active/causal/releases/production` 已发布（含两个 drift
waiver）。四族封版已于 2026-09-29 完成：按规范窗口重跑
`run_all_method_event_study.py`（同城四族 + 跨城 viirs/poi/population，
housing 跨城无 matched 标签记 `no_data`），再由
`export_event_study_final.py` 生成 `mit_event_study_final.zip`
（validation ok）。完整命令、下载清单与验收门槛见
[`event_study_final_runbook.md`](event_study_final_runbook.md)（封版记录见
§7）；服务器训练前置检查与 checklist 见
[`training_release_checklist.md`](training_release_checklist.md)。
