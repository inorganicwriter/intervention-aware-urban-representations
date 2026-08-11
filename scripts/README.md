# Script entrypoints

`scripts/` 只保存可重复运行的命令行入口。可复用逻辑应放在 `src/urban_intervention/`；正式计量逻辑集中在 `scripts/causal_r/`。

分层契约：**可复用、可测试逻辑 → `src/urban_intervention/`（pip install -e 安装的包，正式操作暴露为 `urban-*` console scripts）；正式生产队列 → `scripts/causal_r/`；一次性采集/审计 → `scripts/collection|labels|analysis/`。** 目录下的 Python 文件是薄入口（或 `python scripts/...py` 直接运行），不再定义领域逻辑；`scripts/` 内不得 import 同目录兄弟模块，一律走 `src` 包。

| 目录 | 职责 | 是否可写正式结果 |
|---|---|---|
| `causal_r/` | 控制设计、PanelMatch、Abadie–Imbens、Xu GSC、生产队列和标签发布入口 | 是 |
| `collection/` | 外部数据发现、下载、导入和来源特定解析 | 仅写 `raw/` 或 `staging/` |
| `data/` | 固定处理清单、donor universe、处理前支持表 | 是，写 `data/active/causal/` |
| `data_management/` | 数据迁移、快照、注册表与布局验证 | 是，但迁移必须先 dry-run |
| `labels/` | 规范观测结果与房价社区/网格桥接 | 是，写 `data/active/labels/` 或 `data/active/panels/` |
| `analysis/` | 数据质量、覆盖率和重复记录审计 | 否，只写审计输出 |

## 生产入口

- 站点人工决议：`urban-resolve-stations`
- 空间 donor 审计：`urban-spatial-donor-audit --city all`
- 因果队列：`scripts/causal_r/run_grid_control_design_queue.py`、`scripts/causal_r/run_causal_label_queue.py`
- Response Artifact：`scripts/causal_r/build_response_artifact.py`
- 训练前数据集：`scripts/causal_r/build_pretraining_dataset.py`

## 表示学习入口（console scripts，来自 `src/urban_intervention/representation/`）

- 训练：`urban-train-representation DATA_DIR --output OUT`（`--seeds 1 2 3` 多 seed 汇总）
- 模型卡：`urban-build-model-card OUT`（生成 `model_card.json` / `model_card.md`）
- 消融：`urban-run-ablation DATA_DIR --specs specs.json --output OUT`

`scripts/train_representation.py`、`scripts/build_model_card.py`、`scripts/run_ablation.py`
是保持向后兼容的薄包装，等价于对应 console script。

更具体的运行顺序与环境变量见 [`causal_r/README.md`](causal_r/README.md)。

## 维护规则

- 不在仓库根目录放临时脚本。
- 一次性检查应转成测试或可参数化审计；完成后删除，不建立第二个 archive。
- 脚本不得包含用户绝对路径、密钥或未经声明的网络副作用。
- 不允许脚本直接覆盖 `raw/`；人工决议必须保留源表和应用 manifest。
- 测试、smoke 和 partial 结果必须与正式输出隔离并带 `production_eligible=FALSE`。
