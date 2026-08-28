# Code and reproducibility standards

## 1. 模块边界

- 可复用 Python 逻辑放在 `src/urban_intervention/`。
- 默认 Python/GPU 生产入口放在 `scripts/causal_python/`，资格/parity 工具放在
  `scripts/causal_gpu/`；`scripts/causal_r/` 只保留 R 数据构造、学术参考估计器
  和兼容/部署包装器。
- `scripts/collection/` 只负责采集和来源适配；`scripts/analysis/` 只负责审计。
- 正式训练代码位于 `src/urban_intervention/representation/`；旧 TWFE 和旧年度 DID
  占位模块保留在历史目录。
- 一次性调试代码留在临时目录；长期检查转换成测试或参数化审计。

## 2. 数据边界

- `data/archive/raw/` 采用追加写入，原始材料保留原文件。
- `data/archive/staging/` 可重建，模型读取标准化后的正式数据。
- `data/active/curated/`、`data/active/labels/` 和 `data/active/panels/` 必须声明主键和字段语义。
- `data/active/causal/` 保存冻结处理身份、donor、设计输入和队列；处理清单只根据冻结设计更新。
- `outputs/` 保存可重建审计和估计器对象，数据注册表位于 `data/active/catalog/`。

所有路径通过 `urban_intervention.data.paths`、明确的 CLI 参数或环境变量解析。提交内容只包含相对路径和配置占位符，密钥放入环境变量。

## 3. 因果信息边界

- 控制选择读取处理前变量、处理前缺失状态和冻结空间规则。
- 开通月不作为完整处理后月份；事件期 1 是开通后的第一个完整自然月。
- 匹配、GSC 和 Response Artifact 必须保留 treatment、control/donor、输入版本、代码版本、run ID 和质量诊断。
- Response Artifact 与训练数据发布器必须同时校验 treatment 文件 SHA-256
  以及 `treatment_order → city_key/grid_id/station_event_id/opening_month` 映射。
- 每个任务的目录、队列、manifest 和标签必须具有相同的处理身份、结果族、
  specification 和允许的事件期；只检查主键不重复是不充分的。
- `smoke_test`、`--allow-partial` 和人工改写结果归入测试范围；正式训练只读取经过发布检查的结果。
- 项目路由规则与论文估计器分开表述，文档中分别说明数据路径和算法步骤。

## 4. 表合同

- 空间身份：`city_key, grid_id`。
- 网格年度事实：`city_key, grid_id, year`。
- 网格月度事实：`city_key, grid_id, observed_month`。
- Response Artifact：`treatment_order, outcome_family, outcome, event_time, specification_id`。

写入前检查主键唯一性、必需字段、有限数值、时间范围和 CRS。重复行按数据合同处理，
聚合规则和保留规则均写入合同。

## 5. 输出与写入

- 新输出先写临时文件，校验成功后原子替换目标。
- 不覆盖已存在的正式 release；使用新的 `release_id` 或 `dataset_id`。
- 发布校验拒绝 `code_version=unknown`。无 Git 元数据时使用可执行源码树 SHA-256；
  有 Git 元数据时要求工作区没有未提交修改。
- 队列更新必须可恢复，错误原因必须结构化记录。
- 大型可重建目录、运行时包库、缓存和密钥不纳入 Git。

## 6. 测试要求

任何生产模块修改至少应运行相关单元测试。因果代码还必须运行对应 R 合同测试；涉及发布器时必须同时验证失败门禁和成功路径。

```powershell
conda run -n mit python -m pytest -q
& $env:MIT_RSCRIPT tests/causal_r/test_complete_estimators.R
& $env:MIT_RSCRIPT tests/causal_r/verify_complete_implementation.R
```

真实数据 smoke 用于确认代码可运行。正式结果还需完成全量共同支持、placebo、敏感性和泛化审计。
