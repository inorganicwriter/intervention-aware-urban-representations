# 服务器部署与生产运行

本文件说明如何把仓库完整部署到 Linux 服务器，运行 5,048 网格全量因果标签管线与后续表示学习训练。

## 1. 服务器建议规格

| 资源 | 建议 | 说明 |
|---|---|---|
| CPU | ≥ 32 核 | 数据读取、面板构建与 VIIRS 缓存校验 |
| 内存 | ≥ 64 GB | PanelMatch 16k+ donor 基准单任务 >10 分钟 |
| 磁盘 | ≥ 120 GB | `data/` 约 17 GB + VIIRS 原始月度数据 + outputs |
| GPU | 4 × RTX 4090 | 因果估计默认一张卡一个 Python 进程；训练另行调度 |
| OS | Linux (Ubuntu 22.04+) | Python 3.11；R 仅作参考验证 |

## 2. 传输清单

源码包通过 `zip` / `rsync` / `scp` 传输（`.gitignore` 已覆盖可重建产物）。
实测大小（2026-08-10 复核）：`data/` 全量约 92 GB，但其中 ~60 GB 是
可重建的中间/原始物（WorldPop tif、GEE 导出、VIIRS 月度缓存）。精确分层：

```text
必须传输（约 11 GB；= data/active 全部 − feature_store）：
  src/  scripts/  tests/  docs/
  pyproject.toml  requirements.txt  environment.yml
  README.md  config.yaml  config.yaml.template  Dockerfile.R

  data/active/reference/                # 2.9 GB  固定 500m 网格、站点事件、小区
                                  #         AOI/注册表/桥接（不可再生，必须传）
  data/active/curated/                  # 8.2 GB  全部含 viirs/monthly（4.4 GB）：
                                  #         控制设计前置校验要求全部 6,864 分区；
                                  #         服务器无 MIT_VIIRS_RAW 时必须整体上传
  data/active/panels/  data/active/labels/      # 0.3 GB 房价网格面板与标签
  data/active/causal/（排除 feature_store/）
                                  # 0.8 GB  处理清单、队列、donor universe、
                                  #   grid_universe、accessibility_features、
                                  #   transit_snapshots（482 快照）
  data/active/catalog/               # 3 MB   datasets.yaml 注册表与决议

可不传：
  data/active/causal/feature_store/   # 3.0 GB  无消费者：R 匹配直接读原始分区
                                  #         （panels + viirs monthly），该目录
                                  #         仅为孤立预计算脚本输出，跳过
  data/archive/（raw + staging，约 60 GB）   # 生产标签管线不重建 curated/panels/
                                  #         labels 时无需传输；若需重建
                                  #         population 面板，只传
                                  #         staging/worldpop_r2024b/（0.4 GB）
  data/active/model_inputs/          # demo 数据；生产 release 由
                                  #          build_pretraining_dataset.py 生成

可重建，无需传输（本地自行保留）：
  outputs/                      # 审计与队列产物
  .r-lib/                       # 服务器重建 R 包库
  .runtime/  .pytest_cache/  __pycache__/
```

无 `.git` 时，严格发布会用源码树的 `tree-sha256` 作为代码版本（不会接受 `unknown`），
无需初始化 git。

## 3. Python 环境（mit）

```bash
conda env create -f environment.yml -n mit        # 或 conda env update -n mit -f environment.yml --prune
conda activate mit
pip install -e ".[dev,ml]"                          # 权威依赖来源是 pyproject.toml
python -c "import sys; assert sys.version_info[:2] == (3, 11)"
```

### 重要：torch 必须安装 CUDA 版本

`environment.yml` / `pyproject.toml` 的 `torch>=2.0` 可能解析为 CPU wheel。
服务器必须按 PyTorch 官方安装选择器安装与驱动兼容的 CUDA wheel，再做硬门禁：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4"
```

### playwright（仅采集脚本需要）

```bash
playwright install chromium
```

## 4. R 参考环境（可选）

Python/GPU 是输入初始化、控制设计与正式标签队列的默认后端。仅在运行
`--estimator-backend r_reference` 或 R/Python 一致性审计时需要 R。

两种方式：

- **容器**：使用根目录 `Dockerfile.R`（rocker/r-base:4.6.1 + PanelMatch/Matching/gsynth/fect/arrow/fixest 等全部正式包）。
- **裸机**：安装 R 4.6.1，包库指向项目内 `.r-lib`：

```bash
export R_LIBS_USER="$(pwd)/.r-lib"
mkdir -p "$R_LIBS_USER"
Rscript -e 'options(repos=c(CRAN="https://mirrors.pku.edu.cn/CRAN/")); install.packages(c("data.table","dplyr","fixest","lmtest","sandwich","Matching","PanelMatch","gsynth","fect","future","doParallel","foreach"), lib=Sys.getenv("R_LIBS_USER"), Ncpus=1)'
Rscript -e 'options(repos=c(PPM="https://packagemanager.posit.co/cran/latest/bin/linux/noble-x86_64/4.6")); install.packages("arrow", lib=Sys.getenv("R_LIBS_USER"), Ncpus=1)'
```

当前服务器为 Ubuntu 24.04 x86_64。Arrow 使用 Posit Package Manager 的 R 4.6 Linux 二进制包；安装日志必须显示 `* installing *binary* package 'arrow'`。如果回退为 `* installing *source* package 'arrow'`，应停止安装并检查二进制仓库地址，避免再次触发本地 C++ 编译。

验证（门禁测试）：

```bash
export R_LIBS_USER="$(pwd)/.r-lib"
Rscript tests/causal_r/test_complete_estimators.R
```

`scripts/causal_r/RUNTIME_LOCK.csv` 记录当前规范环境的 R 包版本；其中路径字段是机器本地路径，不能直接复制到服务器。当前规范版本以服务器为准，R 4.6.1 使用 `arrow 25.0.0`；本地 Windows 环境已同步到同一 Arrow 版本。服务器实际路径仍通过下面第 5 节环境变量覆盖，无需修改代码中的数据路径。

## 5. 环境变量与配置

```bash
export MIT_PROJECT_ROOT="$(pwd)"            # R 侧路径基准（paths.R）
export MIT_RSCRIPT="$(command -v Rscript)"
export MIT_R_LIB="$(pwd)/.r-lib"
export MIT_VIIRS_RAW="/data/VIIRS/monthly"  # 原始月度 VIIRS 根目录（44 城 2012-01~2024-12）
export MIT_CAUSAL_RUN_ID="$(date +%Y%m%d%H%M%S)"   # 可选：任务级 run-id 溯源
export MIT_CAUSAL_GPU_IDS="0,1,2,3"                 # Phase 1 worker 分卡
```

`config.yaml`：复制 `config.yaml.template` 并填入高德/百度密钥；GEE 使用
`macro-city-engine` 项目时保留 `config.yaml` 现有内容，并在服务器上完成 GEE 认证：

```bash
earthengine authenticate           # 交互式
# 或 service account:
# export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## 6. 基础验证（部署后必跑）

```bash
conda run -n mit python scripts/data_management/validate_registry.py
conda run -n mit python -m pytest -q
# 仅在维护 R 参考后端时：Rscript tests/causal_r/test_complete_estimators.R
```

## 7. 生产运行序列

完全不调用 R 的输入重建入口如下；并行器的 `--reset-queues` 会自动执行同一流程：

```bash
conda run -n mit python scripts/causal_python/prepare_causal_inputs.py --all
```

### 7.1 VIIRS 月度缓存（控制设计前置，先于一切）

控制设计需要完整 44 城 2012-01 至 2024-12 月度 VIIRS 缓存（6,864 个 Parquet+audit 对）。
缺失文件**不得**被当作 VIIRS 不可用而静默丢城：

```bash
conda run -n mit python scripts/causal_r/run_grid_control_design_queue.py --prepare-viirs-cache-only
```

### 7.2 队列 dry-run 与 canary

```bash
# 控制设计（每格一行，5,048 行）
conda run -n mit python scripts/causal_r/run_grid_control_design_queue.py --start-order 1 --max-units 10 --dry-run

# 结果族任务（5,048 × 4 family = 20,192 任务）
conda run -n mit python scripts/causal_python/run_causal_label_queue.py --start-order 1 --max-tasks 4 --dry-run
```

canary 审核通过前不要启动全量。服务器执行从下述 dry-run 开始，并检查路由、
manifest、GPU 使用、峰值显存和任务耗时。

### 7.3 全量并行生产

队列 CSV 是单写入者：**同一队列禁止并行启动多个进程**。使用官方并行编排器：

```bash
conda run -n mit python scripts/causal_r/run_parallel_production.py \
  --run-all --reset-queues --estimator-backend python_gpu \
  --gpu-ids 0,1,2,3 --shard-count 4 --workers 4
```

- `4` 是四卡服务器的安全默认并行度：一张卡对应一个控制设计 worker 或标签分片。
- 启动器会拒绝 `shard-count > GPU 数量`，避免正式 bootstrap/jackknife 在同一卡争抢显存。
- Phase 1 控制设计 → Phase 2 因果标签（分片并行，可断点续跑）→ Phase 3 合并回 master 队列。
- 中途中断后重跑同命令（不带 `--reset-queues` 保留进度）即可续跑。
- GSC 控制面板秩选择和完全相同 MC 面板的 lambda 选择使用内容寻址缓存；配置、实现版本或决定调参的单元格变化会自动失效。

### 7.4 发布标签与训练前数据（全部任务终态后）

```bash
conda run -n mit python scripts/causal_r/build_response_artifact.py --release-id production_$(date +%Y%m%d)
conda run -n mit python scripts/causal_r/build_pretraining_dataset.py \
  --response-release data/active/causal/releases/production_$(date +%Y%m%d) --dataset-id production_$(date +%Y%m%d)
```

`--allow-partial` 仅限 canary/测试，产物带非生产标记，不能进入正式训练。

### 7.5 表示学习训练

```bash
conda run -n mit urban-train-representation \
  data/active/model_inputs/production_$(date +%Y%m%d) \
  --output outputs/representation/production_$(date +%Y%m%d) \
  --epochs 100 --device cuda \
  2>&1 | tee outputs/representation/production_$(date +%Y%m%d).log
```

每轮输出 `Epoch x/100 | train/val loss | nn_corr@5 | lr`；终局产物
`best_model.pt`、`training_history.json`、`test_metrics.json`。
若启用街景多模态（DINOv2 需联网下载 hub 权重，离线时设置 `TORCH_HUB` 缓存目录）。

## 8. 监控与续跑

- 队列状态：`data/active/causal/*_queue.csv`（原子更新，可随时中断）
- 任务产物：`data/active/causal/tasks/<order>/<family>/`（manifest.json 终态判定）
- 断点续跑：重复上一命令即可；`--retry-matching` / `--phase` 可定向重试
- 全量完成后按 README「发布标签与训练前数据」一节执行严格发布校验

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: urban_intervention` | 未 `pip install -e .`，或 PYTHONPATH 未含 `src/` |
| R 参考门禁失败（gsynth/fect 版本） | 仅影响 `r_reference` 审计；用 Dockerfile.R 重建并核对 RUNTIME_LOCK.csv |
| VIIRS 缓存缺失对 | 先跑 `--prepare-viirs-cache-only`；严禁静默跳过 |
| torch.cuda.is_available() = False | 用 `--index-url` 重装 CUDA wheel |
| 所有任务只占用 GPU 0 | 使用并行启动器并传 `--gpu-ids 0,1,2,3 --shard-count 4` |
| 发布会拒绝 `unknown` 代码版本 | 保持源码树完整传输（tree-sha256 校验） |
| 队列卡住 | 检查 `data/active/causal/*_queue.csv` 与任务目录日志；断点续跑即可 |
