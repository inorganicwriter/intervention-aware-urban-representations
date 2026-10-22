# S/T/C 代码就绪记录

更新：2026-10-19。

当前正式服务器验收于2026-10-19在指定服务器既有 `mit` 环境完成，运行目录为`/home/nas/lsr/MIT_summer/outputs/stc_server_acceptance/stc_acceptance_20261019T114650Z`。VIIRS、POI分别使用strict-production真实Response Artifact，各64个事件完成开通前多模态读取、两轮短训练、检查点恢复和64事件导出。六个指定回归文件62 passed、0 failed、0 skipped、0 deselected、0 excluded，UMAP真实测试执行且PCA回退由独立测试覆盖。工程闭环不代表正式模型性能已验证。

UMAP补验：真实UMAP单项先独立通过（78.39秒，umap-learn 0.5.11）。运行到45秒的堆栈显示仍在导入UMAP附带加载的TensorFlow模块，本次最终正常完成；这不能追溯证明历史未完成运行的原因。测试现明确断言`method=umap`，缺少可选依赖时显式跳过；PCA回退由独立测试验证。后续相关回归不再默认排除UMAP，验收需同时记录通过、跳过及排除数量。本次临时测试数据和Numba编译缓存随临时目录自动清除。

模型口径以《研究思路.docx》的公式为准：`z_i = f_theta(x_i)`，`tau_hat_i = g_phi(z_i, T_i, C_i)`。当前项目契约将 `x_i` 明确为开通前状态 `S_i`：POI/VIIRS 历史、全部合法开通前街景、独立卫星影像序列及 masks 一起学习一个定长多维状态 embedding `z_i`。干预 `T_i` 与背景 `C_i` 单独编码，作为响应预测条件；不拼入 `z_i`。因果响应标签监督响应预测，并以响应相似性损失塑造 `z_i`。不使用开通后视觉影像作为预测目标。

此前服务器验收记录见下文历史回执。最新运行已重新验证当前 S/T/C 输入路径，包括“全部开通前街景 + 逐图时间间隔 + 时间感知集合编码”。

## 模型输入与监督边界

每个事件的模型输入由三类向量共同构成：

| 分支 | 输入内容 | 编码结果 |
|---|---|---|
| 状态 `S` | 15 个开通前 POI/VIIRS 滞后字段、全部合规开通前街景、开通前卫星影像 `-6..-1` 年序列 | 唯一状态 embedding `z` |
| 干预 `T` | 开通年份、4 个开通时站点类型字段、开通线路数、同月开通数及字段 mask | 响应条件 `t` |
| 背景 `C` | 76 个数值字段：AlphaEarth64 + 12 个表格背景字段（人口滞后与城市 POI/VIIRS/人口统计，不含房价）；另有12个表格字段 mask 和1个 AlphaEarth 可用性 mask | `ContextEncoder` 将 AlphaEarth 投影为32维、表格值与 mask 投影为32维，再融合为64维条件向量 `b` |

唯一的项目状态 embedding 是 `z=f_theta(S)`；街景、卫星、POI/VIIRS等状态模态共同编码成一个定长多维向量。原始 `C` 是76个数值字段加13个缺失/可用性 mask，编码后为64维 `b`；房价字段已从背景中移除。响应预测为 `tau_hat=g_phi(z,T,C)`，实现中预测器读取 `concat(z,t,b)`。响应相似性对齐作用于 `z`，不作用于预测器拼接张量。监督目标是正式 Response Artifact 中有 mask 和可靠性权重的因果响应标签；开通后的街景或卫星图不是预测目标。最新服务器验收已覆盖时间感知街景主链路。

核心入口：

- 训练包发布：`scripts/causal_r/build_pretraining_dataset.py --stc`。
- 数据加载及模型：`src/urban_intervention/representation/`。
- 服务器验收：`scripts/analysis/verify_stc_server_smoke.py`。
- 回归测试：`tests/unit/test_stc_model.py` 及相关既有测试。

清理说明：保留源码、回归测试、正式资产和必要文档；移除临时候选源码副本、审阅复现脚本、测试夹具、测试检查点、Python/pytest/ruff/mypy缓存及LaTeX辅助文件。测试夹具由pytest临时生成，不再依赖审阅输出。清理共移除7,826个文件，按文件逻辑大小约241 MiB。审阅结论、实验指标、正式数据的采集/绑定/同步回执、历史审计记录及研究结果均保留；正式数据不在本次清理范围内。

## 首轮 2026-10-19 指定服务器真实验收（历史记录）

运行目录为`/home/nas/lsr/MIT_summer/outputs/stc_server_acceptance/stc_acceptance_20261019T050410Z`。隔离候选代码包441个文件校验通过；VIIRS六维381个有效监督单元，POI十二维720个有效监督单元。每类均读取31张真实街景和383张真实卫星TIFF，保留合法缺失mask，无读取失败；完成前向/反向、S/T/C分离、mask不变性、两轮短训、检查点恢复和64事件导出。恢复与导出最大数值差为0。六个回归文件60 passed、0 failed、0 skipped、0 deselected；UMAP方法断言通过。验收前后32个正式资产及队列文件的SHA256一致。

历史服务器验收中的损失以开通前S/T/C为输入，并由正式Response Artifact中的因果响应标签监督。同期生成的前后视觉资产配对索引只用于覆盖率与时序审计，不是模型训练目标或输入；详见[历史视觉资产配对审计](../research/visual_state_transition_pair_contract_2026-10-19.md)。

## 街景时间感知输入状态

街景属于 `S` 的子模态。主数据集按事件读取全部通过身份、SHA256 和 `capture_month < opening_month` 校验的图片；对每张图片计算正的日历月间隔 `d`，再用 `d / (d + 12)` 压缩。`StateEncoder` 将图片 embedding 与对应压缩时间编码融合，再通过可学习注意力得到每个事件一个定长街景向量。该向量与 POI/VIIRS 历史及卫星序列共同生成状态 embedding `z`。街景缺失使用 mask；读取失败或来源绑定错误在严格数据中报错，不伪装成合法缺失。

该时间感知街景输入已在本次 S/T/C 主训练入口中完成指定回归与服务器复验。验收实际读取31张街景和383张开通前卫星TIFF；街景间隔为1–30个日历月，按 `d/(d+12)` 压缩后送入可学习时间感知注意力。开通后图片不进入 `z`，正式因果响应标签仍是监督目标。详细结果见最新服务器验收回执。
