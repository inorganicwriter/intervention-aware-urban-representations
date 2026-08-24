# Code and reproducibility standards

## 1. 模块边界

- 可复用 Python 逻辑放在 `src/urban_intervention/`。
<!-- GPU 迁移前：正式计量估计器和 R 数据构造统一放在 scripts/causal_r/。 -->
- 默认 Python/GPU 生产入口放在 `scripts/causal_python/`，资格/parity 工具放在
  `scripts/causal_gpu/`；`scripts/causal_r/` 只保留 R 数据构造、学术参考估计器
  和兼容/部署包装器。
- `scripts/collection/` 只负责采集和来源适配；`scripts/analysis/` 只负责审计。
<!-- 迁移前状态：正式训练代码尚未建立，计划创建独立的
src/urban_intervention/representation/。 -->
- 正式训练代码位于 `src/urban_intervention/representation/`；不得复用旧
  TWFE 或旧年度 DID 占位模块。
- 一次性调试代码不得进入长期主树；稳定检查应转换成测试或参数化审计。

## 2. 数据边界

- `data/archive/raw/` append-only，不覆盖原始材料。
- `data/archive/staging/` 可重建；不得被模型直接读取。
- `data/active/curated/`、`data/active/labels/` 和 `data/active/panels/` 必须声明主键和字段语义。
- `data/active/causal/` 保存冻结处理身份、donor、设计输入和队列；不得根据处理后结果改写处理清单。
- `outputs/` 保存可重建审计和估计器对象，不是数据注册表。

所有路径应通过 `urban_intervention.data.paths`、明确的 CLI 参数或环境变量解析。禁止提交用户绝对路径和密钥。

## 3. 因果信息边界

- 控制选择只能读取处理前变量、处理前缺失状态和冻结空间规则。
- 开通月不作为完整处理后月份；事件期 1 是开通后的第一个完整自然月。
- 匹配、GSC 和 Response Artifact 必须保留 treatment、control/donor、输入版本、代码版本、run ID 和质量诊断。
- Response Artifact 与训练数据发布器必须同时校验 treatment 文件 SHA-256
  以及 `treatment_order → city_key/grid_id/station_event_id/opening_month` 映射。
- 每个任务的目录、队列、manifest 和标签必须具有相同的处理身份、结果族、
  specification 和允许的事件期；只检查主键不重复是不充分的。
- `smoke_test`、`--allow-partial` 和失败后人工改写的结果不得进入正式训练。
- 项目路由规则必须与论文估计器分开表述，不得把二者拼成不存在的“新算法”。

## 4. 表合同

- 空间身份：`city_key, grid_id`。
- 网格年度事实：`city_key, grid_id, year`。
- 网格月度事实：`city_key, grid_id, observed_month`。
- Response Artifact：`treatment_order, outcome_family, outcome, event_time, specification_id`。

写入前必须检查主键唯一性、必需字段、有限数值、时间范围和 CRS。重复行不能通过无条件平均被静默消除；聚合规则必须属于数据合同。

## 5. 输出与写入

- 新输出先写临时文件，校验成功后原子替换目标。
- 不覆盖已存在的正式 release；使用新的 `release_id` 或 `dataset_id`。
- 严格发布不得写入 `code_version=unknown`。无 Git 元数据时使用可执行源码树
  SHA-256；有 Git 元数据时拒绝 dirty worktree。
- 队列更新必须可恢复，错误原因必须结构化记录。
- 大型可重建目录、运行时包库、缓存和密钥不纳入 Git。

## 6. 测试要求

任何生产模块修改至少应运行相关单元测试。因果代码还必须运行对应 R 合同测试；涉及发布器时必须同时验证失败门禁和成功路径。

```powershell
conda run -n mit python -m pytest -q
& $env:MIT_RSCRIPT tests/causal_r/test_complete_estimators.R
& $env:MIT_RSCRIPT tests/causal_r/verify_complete_implementation.R
```

真实数据 smoke 只能证明代码可运行，不能代替全量共同支持、placebo、敏感性和泛化审计。
