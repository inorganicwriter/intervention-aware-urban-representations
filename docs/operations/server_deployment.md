# 服务器部署与生产运行

本文件说明如何把仓库完整部署到 Linux 服务器，运行 5,048 网格全量因果标签管线与后续表示学习训练。

## 1. 服务器建议规格

| 资源 | 建议 | 说明 |
|---|---|---|
| CPU | ≥ 32 核 | Phase 1/2 并行 R 估计器与 VIIRS 缓存校验 |
| 内存 | ≥ 64 GB | PanelMatch 16k+ donor 基准单任务 >10 分钟 |
| 磁盘 | ≥ 120 GB | `data/` 约 17 GB + VIIRS 原始月度数据 + outputs |
| GPU | 可选 | 训练表示模型时使用（`--device cuda`）；无 GPU 自动回退 CPU |
| OS | Linux (Ubuntu 22.04+) | R 4.6.1 与 Python 3.11 |

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

`environment.yml` / `pyproject.toml` 的 `torch>=2.0` 默认解析为 CPU wheel。
服务器有 GPU 时必须在装 torch 后显式替换为 CUDA 版（以 cu121 为例，按服务器 CUDA 驱动选）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # 期望 True
```

### playwright（仅采集脚本需要）

```bash
playwright install chromium
```

## 4. R 环境（4.6.1）

两种方式：

- **容器**：使用根目录 `Dockerfile.R`（rocker/r-base:4.6.1 + PanelMatch/Matching/gsynth/fect/arrow/fixest 等全部正式包）。
- **裸机**：安装 R 4.6.1，包库指向项目内 `.r-lib`：

```bash
export R_LIBS_USER="$(pwd)/.r-lib"          # 首次 mkdir -p .r-lib
R -e 'install.packages(c("data.table","dplyr","arrow","fixest","lmtest","sandwich",
                         "Matching","PanelMatch","gsynth","fect",
                         "future","doParallel","foreach"),
     repos="https://cloud.r-project.org")'
```

验证（门禁测试）：

```bash
export R_LIBS_USER="$(pwd)/.r-lib"
Rscript tests/causal_r/test_complete_estimators.R
```

`scripts/causal_r/RUNTIME_LOCK.csv` 是 Windows 本机的运行锁定记录；服务器上以
实际安装路径为准，通过下面第 5 节环境变量覆盖，无需修改任何代码。

## 5. 环境变量与配置

```bash
export MIT_PROJECT_ROOT="$(pwd)"            # R 侧路径基准（paths.R）
export MIT_RSCRIPT="$(command -v Rscript)"
export MIT_R_LIB="$(pwd)/.r-lib"
export MIT_VIIRS_RAW="/data/VIIRS/monthly"  # 原始月度 VIIRS 根目录（44 城 2012-01~2024-12）
export MIT_CAUSAL_RUN_ID="$(date +%Y%m%d%H%M%S)"   # 可选：任务级 run-id 溯源
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
conda run -n mit python -m pytest -q                  # 本地基线 230 passed
Rscript tests/causal_r/test_complete_estimators.R
```

## 7. 生产运行序列

### 7.1 VIIRS 月度缓存（控制设计前置，先于一切）

控制设计需要完整 44 城 2012-01~2024-12 月度 VIIRS 缓存（6,864 个 Parquet+audit 对）。
缺失文件**不得**被当作 VIIRS 不可用而静默丢城：

```bash
conda run -n mit python scripts/causal_r/run_grid_control_design_queue.py --prepare-viirs-cache-only
```

### 7.2 队列 dry-run 与 canary

```bash
# 控制设计（每格一行，5,048 行）
conda run -n mit python scripts/causal_r/run_grid_control_design_queue.py --start-order 1 --max-units 10 --dry-run

# 结果族任务（5,048 × 4 family = 20,192 任务）
conda run -n mit python scripts/causal_r/run_causal_label_queue.py --start-order 1 --max-tasks 4 --dry-run
```

canary 审核通过前不要启动全量。2026-08-10 两阶段匹配 canary 已验证
（orders 1–10 正确路由 GSC；orders 906–915 中 3/10 同城匹配、order 906
全族走完 match→GSC→MC→skip 链路）；canary 产物已清理，队列已重置全 pending，
服务器上从下述 dry-run 开始。

### 7.3 全量并行生产

队列 CSV 是单写入者：**同一队列禁止并行启动多个进程**。使用官方并行编排器：

```bash
conda run -n mit python scripts/causal_r/run_parallel_production.py \
  --run-all --reset-queues --shard-count 16 --workers 48
```

- Phase 1 控制设计 → Phase 2 因果标签（分片并行，可断点续跑）→ Phase 3 合并回 master 队列。
- 中途中断后重跑同命令（不带 `--reset-queues` 保留进度）即可续跑。

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

- 队列状态：`data/active/causal/queues/*.csv`（原子更新，可随时中断）
- 任务产物：`data/active/causal/tasks/<order>/<family>/`（manifest.json 终态判定）
- 断点续跑：重复上一命令即可；`--retry-matching` / `--phase` 可定向重试
- 全量完成后按 README「发布标签与训练前数据」一节执行严格发布校验

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: urban_intervention` | 未 `pip install -e .`，或 PYTHONPATH 未含 `src/` |
| R 门禁失败（gsynth/fect 版本） | 用 Dockerfile.R 重建包库，或核对 RUNTIME_LOCK.csv 版本 |
| VIIRS 缓存缺失对 | 先跑 `--prepare-viirs-cache-only`；严禁静默跳过 |
| torch.cuda.is_available() = False | 用 `--index-url` 重装 CUDA wheel |
| 发布会拒绝 `unknown` 代码版本 | 保持源码树完整传输（tree-sha256 校验） |
| 队列卡住 | 检查 `data/active/causal/queues/` 状态与任务目录日志；断点续跑即可 |
