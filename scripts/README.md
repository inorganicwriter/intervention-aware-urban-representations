# Script entrypoints

`scripts/` 只保存可重复运行的命令行入口。可复用逻辑放在
`src/urban_intervention/`；默认生产队列和事件研究入口位于
`scripts/causal_python/`，GPU shadow/parity/资格工具位于
`scripts/causal_gpu/`，`scripts/causal_r/` 保留经审计的 R 参考实现和部署兼容入口。

代码职责：

- 可复用逻辑：`src/urban_intervention/`；
- Python/GPU 生产入口：`scripts/causal_python/`；
- 资格与 parity：`scripts/causal_gpu/`；
- R 参考实现：`scripts/causal_r/`；
- 采集和审计：`scripts/collection/`、`scripts/labels/`、`scripts/analysis/`。

入口脚本应保持薄层。兼容包装器可以导入规范入口，但不得复制领域逻辑。

| 目录 | 职责 | 是否可写正式结果 |
|---|---|---|
| `causal_python/` | 默认 `python_gpu` 控制/标签生产、GSC/MC 正式执行和三方法事件研究 | 是；production 需合格 parity receipt |
| `causal_gpu/` | GPU shadow、R/Python parity、Matching 参考导出和资格凭证审计 | 否；只写不可晋升的 shadow/qualification 产物 |
| `causal_r/` | PanelMatch、Abadie–Imbens、`gsynth`/`fect` 参考实现及兼容/部署包装器 | 仅显式 `r_reference` 或参考产物 |
| `collection/` | 外部数据发现、下载、导入和来源特定解析 | 仅写 `raw/` 或 `staging/` |
| `data/` | 固定处理清单、donor universe、处理前支持表 | 是，写 `data/active/causal/` |
| `data_management/` | 数据迁移、快照、注册表与布局验证 | 是，但迁移必须先 dry-run |
| `labels/` | 规范观测结果与房价社区/网格桥接 | 是，写 `data/active/labels/` 或 `data/active/panels/` |
| `analysis/` | 数据质量、覆盖率和重复记录审计 | 否，只写审计输出 |

## 生产入口

- 站点人工决议：`urban-resolve-stations`
- 空间 donor 审计：`urban-spatial-donor-audit --city all`
- 控制队列：`scripts/causal_r/run_grid_control_design_queue.py`（默认内部调用 Python/GPU 控制设计）
- 因果标签队列：`scripts/causal_python/run_causal_label_queue.py`
- 标签队列模块化入口：`scripts/causal_python/run_causal_label_queue_modular.py`（非生产入口）
- 单任务 GSC/MC：`scripts/causal_python/run_formal_estimator.py`
- 三方法事件研究：`scripts/causal_python/run_all_method_event_study.py`
- GPU 资格：`scripts/causal_gpu/run_shadow_queue.py`、`scripts/causal_gpu/audit_formal_qualification.py`
- Response Artifact：`scripts/causal_r/build_response_artifact.py`
- 训练前数据集：`scripts/causal_r/build_pretraining_dataset.py`

## 表示学习入口（console scripts，来自 `src/urban_intervention/representation/`）

- 训练：`urban-train-representation DATA_DIR --output OUT`（`--seeds 1 2 3` 多 seed 汇总）
- 模型卡：`urban-build-model-card OUT`（生成 `model_card.json` / `model_card.md`）
- 消融：`urban-run-ablation DATA_DIR --specs specs.json --output OUT`

`scripts/train_representation.py`、`scripts/build_model_card.py`、`scripts/run_ablation.py`
是保持向后兼容的薄包装，等价于对应 console script。

生产资格和 GPU 命令见
[`../src/urban_intervention/causal/gpu/README.md`](../src/urban_intervention/causal/gpu/README.md)；
R 参考参数与部署环境见 [`causal_r/README.md`](causal_r/README.md)。

## POI 面板入口

- 全国 FileGDB 正式批处理：`scripts/collection/poi_batch_panel_builder.py`，核心为
  `urban_intervention.pipelines.poi.batch`。
- 单城重建、CSV 年份和定向回填：`scripts/collection/poi_panel_builder.py`，核心为
  `urban_intervention.pipelines.poi.pipeline`。

两者共享标准化和聚合模块。全国生产使用 batch 入口；单城入口用于回填和诊断。

## 维护规则

- 不在仓库根目录放临时脚本。
- 一次性检查应转成测试或可参数化审计；完成后删除，不建立第二个 archive。
- 脚本不得包含用户绝对路径、密钥或未经声明的网络副作用。
- 不允许脚本直接覆盖 `raw/`；人工决议必须保留源表和应用 manifest。
- 测试、smoke 和 partial 结果必须与正式输出隔离并带 `production_eligible=FALSE`。
