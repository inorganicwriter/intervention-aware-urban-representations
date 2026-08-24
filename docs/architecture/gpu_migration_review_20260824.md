# GPU migration architecture review

日期：2026-08-24

本文记录 GPU 迁移后架构审阅中 13 项问题的处理结果。它只说明当前代码
边界和验证状态，不把尚未完成的服务器资格运行写成已经完成。

<!-- 原审阅摘要（保留）：
高优先级：文档滞后于 GPU 迁移；Python 生产编排器仍位于 causal_r，且新目录
尚未进入版本控制。中优先级：config/interventions 反向依赖、SHA-256 与列校验
重复、原子写分散、新入口缺少直接测试。低优先级：私有 API 跨模块调用、GPU
门面误导、巨型文件、POI 双入口、文档断链及状态文档过期。
-->

## 逐项处置

| # | 状态 | 处置 |
|---|---|---|
| 1 | 已修复 | 根 `README.md`、`scripts/README.md`、DDR-003 和当前状态页已写明：默认生产后端是受资格凭证约束的 `python_gpu`；R 是独立审计参考。旧表述保留在 HTML 注释中。 |
| 2 | 代码结构已修复 | 规范编排器已迁至 `scripts/causal_python/run_causal_label_queue.py`；旧 `scripts/causal_r/` 路径只保留薄兼容包装器。`causal_python/` 与 `causal_gpu/` 的职责已进入架构文档。按项目要求，本轮不读取或修改 Git 记录，因此 tracked/untracked 和远端提交状态不在本报告结论内。 |
| 3 | 已修复 | 站名规范化下沉至中立模块 `urban_intervention.text`。配置层和领域层均依赖该模块，`config` 不再反向依赖 `interventions`；旧 `station_names` 路径保留兼容导出。 |
| 4 | 已修复 | SHA-256 统一由 `urban_intervention.utils.sha256_file` 实现。`causal.gpu.provenance.file_sha256` 仅为兼容包装器；分块大小不影响摘要语义。 |
| 5 | 已修复 | `require_columns` 统一到 `urban_intervention.utils`，空间 donor 与站点决议模块不再各自维护副本。 |
| 6 | 已修复 | CSV、Parquet、JSON 的原子文件发布统一到 `urban_intervention.utils`，使用同目录唯一临时文件、刷新后替换，并处理 Windows 短时文件占用。目录级发布和内容寻址缓存仍保留各自事务语义，不强行套用文件助手。 |
| 7 | 已修复 | 新增 Python 生产入口直接测试，覆盖 6 个规范入口的 `--help`、旧路径兼容包装器和正式输出事务敏感路径。 |
| 8 | 已修复 | panel reader 已改为公开 API：`read_annual_outcome`、`read_monthly_housing`、`read_monthly_viirs`；跨模块不再导入下划线私有函数。 |
| 9 | 已修复 | `causal.gpu.__init__` 改为无导入、仅说明边界的包入口，不再把 Abadie–Imbens 误示为整个 GPU 生产门面。生产代码继续显式导入具体子模块。 |
| 10 | 有意延期 | 已迁移放错目录的队列入口，但不按行数机械拆分 `trainer.py`、`project.py` 或队列实现。拆分应以稳定的职责边界和回归测试为前提，安排在 GSC/MC 服务器资格验证之后，避免在正确性冻结期引入行为变化。 |
| 11 | 已澄清，无需合并 | `pipelines/poi/batch.py` 是全国 FileGDB 批生产入口；`pipelines/poi/pipeline.py` 是单城市/CSV 年份与定向回填入口。两者不是同一工作流的重复实现；规范用途已写入 `scripts/README.md`。 |
| 12 | 已修复 | 3 个指向可重建但当前不存在 output 的链接改为指向生成脚本，旧 output 路径保留在 HTML 注释；决策索引已加入 DDR-005。 |
| 13 | 已修复 | `current_project_status.md` 更新至 2026-08-24，明确 Python/GPU 默认后端、R 参考边界，以及尚待 4090 完成的 3 个 GSC 与 3 个 MC 资格任务。 |

## 当前边界

- Matching 的 3 个真实参考任务已通过设计、质量和最终标签 parity。
- GSC 与 MC 的 Python/GPU 实现和测试已具备，但正式生产仍须在 RTX 4090
  环境各完成 3 个代表任务并签发环境绑定资格凭证。
- 未取得完整凭证时，生产入口必须 fail closed；不能用 smoke、shadow 或部分
  parity 结果替代正式资格。
- R 不再是默认逐任务生产依赖，但仍是当前学术参考与迁移审计依赖。取得完整
  GSC/MC parity 证据后，日常全量生产可以只运行 Python/GPU；复现参考结果时
  仍需要锁定的 R 环境。

## 验证原则

本轮只做与审阅项直接相关的结构修复，不以“更防御性”为理由扩大业务规则。
工具函数合并必须保持既有输入输出合同；入口迁移必须保留旧路径兼容；任何
估计器公式、筛选阈值、fallback 顺序或正式资格门槛均不得因重构而改变。

## 本轮验证

- Ruff 对所有受影响 Python 模块和入口检查通过。
- Python 完整单元测试：396 passed。
- R 4.6.1（项目 `.r-lib`）参考合同：标签合同、PanelMatch/Abadie–Imbens
  等价性、GSC 标签映射、event-study 聚合、四类正式参考估计器的 5 个测试
  全部通过。
- Markdown 本地链接扫描：0 个缺失目标；HTML 注释均成对闭合。

完整单测还暴露并修复了一个与架构清理同时出现的合同回归：未标注 skeleton
行的 `donor_scope` 可以为空，新版 pandas 会把作用域比较保留为 nullable boolean，
使 `main_training_mask` 的布尔运算失败。现在空 scope 明确映射为
`same_city=False` 且 `cross_city=False`；已标注样本的作用域、质量等级和训练资格
不变，并增加了回归断言。
