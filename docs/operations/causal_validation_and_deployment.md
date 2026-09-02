# 因果代码验证、服务器覆盖与生产复核命令

更新：2026-09-02
适用环境：本地 Windows 工作区与 Linux 生产服务器 `/home/nas/lsr/MIT_summer`
主规格：Python/GPU，月度 anticipation=6，observation window=3，年度 window=1，跨城 donor 上限 5,000。

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

若 Matching、GSC 或 MC 任一资格绑定源码发生变化，必须重跑对应的 3 个 parity 任务。只改绘图代码或文档不需要重跑 R reference。当前 Matching/VIIRS 处理前特征与控制设计代码发生变化时，至少重跑 Matching；为避免遗漏，发布前推荐三方法一起跑：

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
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 python scripts/causal_r/run_parallel_production.py --phase 2 --estimator-backend python_gpu --qualification-receipt outputs/causal_gpu/formal_qualification.json --gpu-ids 0,1,2,3 --shard-count 4 --task-scheduling global_rounds --run-mode production --price-measure main --window 3 --anticipation-months 6 --transaction-count-threshold 1 --max-gsc-cross-city-donors 5000 2>&1 | tee outputs/phase2.log
```

### 6.2 已冻结的汇报样本

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 python scripts/causal_r/run_parallel_production.py --phase 2 --tasks-file outputs/causal_labels/presentation_sample_tasks.csv --estimator-backend python_gpu --qualification-receipt outputs/causal_gpu/formal_qualification.json --gpu-ids 0,1,2,3 --shard-count 4 --task-scheduling global_rounds --run-mode production --price-measure main --window 3 --anticipation-months 6 --transaction-count-threshold 1 --max-gsc-cross-city-donors 5000 2>&1 | tee outputs/phase2_sample.log
```

四张 GPU 最多使用四个 shard。`global_rounds` 的顺序是：全部同城 Matching → 同城 GSC → 同城 MC → 跨城 Matching → 跨城 GSC → 跨城 MC。重复同一命令可从持久化 shard 续跑；不要为续跑添加 `--reset-queues`。

实时监控：

```bash
pgrep -af 'run_parallel_production.py|run_causal_label_round_queue.py|run_causal_label_queue.py'
nvidia-smi dmon -s pucm -d 2
tail -n 80 outputs/phase2.log
```

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

入口按任务文件纳入所有可接纳样本，但不跨 outcome 量纲混合：每个结果族内分别汇总 Matching、GSC、MC。三种方法只比较共同支持：

- 月度：clean pre `-42:-7`；post `1,3,6,12,18,24`；window=3；t=0 仅 Matching 作为边界诊断正常绘制；
- 年度：clean pre `-4:-1`；post `1,2,3`；window=1；
- GSC/MC 拟合可使用更长年度 pre 历史，但共同图不会显示额外时期；
- `-6:-1` 月度 anticipation 不参加平行趋势检验。

```bash
PYTHONUNBUFFERED=1 python scripts/causal_python/run_all_method_event_study.py --staging-root outputs/causal_labels/presentation_sample_estimator_paths_20260902 --tasks-file outputs/causal_labels/presentation_sample_tasks.csv --output-directory outputs/event_study_presentation_sample --latest-pre-periods 5 --observation-window 3 2>&1 | tee outputs/event_study_presentation_sample.log
python -c "import pandas as pd; r='outputs/event_study_presentation_sample'; print(pd.read_csv(r+'/suite_status.csv').to_string(index=False)); print(pd.read_csv(r+'/sample_aggregation_coverage.csv').to_string(index=False)); assert pd.read_csv(r+'/suite_status.csv')['status'].eq('completed').all()"
python -c "import pandas as pd; p='outputs/event_study_presentation_sample/gsc_mc_paths/effect_paths_with_pretrend.parquet'; x=pd.read_parquet(p); assert set(x.loc[x.frequency.eq('monthly'),'observation_window'])=={3}; assert set(x.loc[x.frequency.eq('annual'),'observation_window'])<= {1}; assert x.loc[x.frequency.eq('monthly'),'event_time'].isin(list(range(-42,-6))+[1,3,6,12,18,24]).all(); assert x.loc[x.frequency.eq('annual'),'event_time'].isin(list(range(-4,0))+[1,2,3]).all(); print(x.groupby(['outcome_family','method','frequency','observation_window'])['treatment_order'].nunique().to_string())"
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
