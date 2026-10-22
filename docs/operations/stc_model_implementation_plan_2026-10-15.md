# S/T/C 模型接入实施方案

方案日期：2026-10-15。按《研究思路.docx》的目标，开通前多模态城市状态共同编码为一个响应敏感 embedding，因果响应标签用于训练响应预测及表示对齐。2026-10-19最新真实服务器验收已覆盖时间感知街景输入、六个回归文件以及 VIIRS/POI 短训练保存恢复导出闭环。工程验收不代表正式模型性能已经验证。

## 目标与首版范围

在已验收资产上贯通“训练包发布 → 读取真实数据 → S/T/C 独立编码 → 训练 → 保存恢复 → 导出评估”。正式资产和生产因果响应使用服务器 `/home/nas/lsr/MIT_summer`；本地用于开发与合成数据回归，完整街景继续只保存在服务器。

首版保留现有 soft causal anchoring 主目标与 W-design 权重设置，沿用同城 matched/GSC 的 VIIRS、POI 监督范围。先分别训练 VIIRS（6 个预测维度）和 POI（12 个预测维度），不把房价、人口或 MC/跨城估计重新加入正式监督。旧模型与旧检查点作为显式兼容路径保留。

| 分支 | 首版内容 | 输出 |
|---|---|---|
| S | POI/VIIRS 已发布历史字段、全部合法开通前街景、独立卫星前六年影像与 masks | 唯一状态 embedding `z=f_theta(S)` |
| T | 已发布七字段及逐字段 masks | 独立编码为响应条件 `t` |
| C | 82个数值字段：AlphaEarth64、房价/人口历史值及城市统计代理；另有18个表格字段 mask 和1个 AlphaEarth 可用性 mask | 独立编码为64维响应条件 `b` |
| 预测 | `concat(z,t,b)` 作为响应预测器输入 | `tau_hat=g_phi(z,T,C)` 因果响应路径 |
| 响应相似性对齐 | 用因果响应相似度对齐状态 embedding `z_i` 与 `z_j` | 不把 T/C 拼入对齐 embedding |

路网、站名/站点 ID 编码、站点到格心距离和新的轨道增量不加入首版。城市背景继续使用已发布的观测网格统计代理，不新增 GDP 等外部采集。既有数据缺口用 masks 表达，工程接入不等待所有历史字段补齐。

## 1. 发布可直接训练的新角色数据包

扩展现有 `publish_pretraining_dataset()` 和 CLI，增加显式角色资产入口及输入契约版本。使用冻结 `treatment_order/city_key/grid_id/station_event_id/opening_month` 连接已发布 S/T/C、前六年卫星索引和生产响应；禁止按行号或站名隐式拼接。

发布到 `data/model_inputs/<dataset_id>/` 的文件：

- `state_inputs.parquet`、`context_inputs.parquet`、`intervention_inputs.parquet`；
- `satellite_state_sequences.parquet`，仅保留 `event_time=-6..-1`；
- `response_targets.parquet`、`sample_index.parquet`；
- `normalization.json`、`manifest.json`。

影像只写来源引用与 SHA256，不重复复制。保留全部 5,048 个事件记录；训练资格由监督有效性及既定实验范围决定，不要求所有背景或视觉模态均有观测。城市划分沿用既有确定性规则，预处理仅拟合训练城市中的有效观测。验证与测试城市不参与均值、尺度、降维或特征选择拟合。

清单保存字段白名单和顺序、监督维度顺序、来源清单/处理清单 SHA256、时间规则、城市划分、角色 masks、传感器/反射率口径及预处理参数。样本身份、来源路径和原始站点字符串只用于绑定与审计，不通过自动数值列扫描进入模型。

联调可使用现有七字段及结构 masks；正式实验采用完成来源复核的历史属性，或限定到已确认字段。该选择写入数据包和实验配置，未确认值不能被标记为历史已核验。

完成标准：训练包能直接创建新 Dataset；字段身份匹配、城市集合互斥；非法事后输入被拒绝；现有合法历史缺失不导致整批样本报错。

## 2. Dataset 与 collate 输出明确分支

扩展原有 `RepresentationDataset`，按 schema 分派新角色输入与旧发布包。定义共享的 batch 契约供 canonical/modular 训练、导出和评估使用。

| 张量/引用 | 形状或含义 |
|---|---|
| `state_tabular`、`state_field_mask` | `(B, n_S)`；已发布 POI/VIIRS 历史字段 |
| `streetview_features`、`streetview_mask` | `(B, K, D_img)` 与 `(B,K)`；当前精选策略 `K≤2` |
| `satellite_images` | `(B,6,6,H,W)`；六个相对年份、六个反射率波段 |
| `satellite_pixel_mask` | `(B,6,1,H,W)` |
| `satellite_year_mask`、`satellite_event_times` | `(B,6)`；固定顺序 −6 至 −1，缺年不压缩成相邻年份 |
| `context`、`context_field_mask` | `(B,n_C)`；AlphaEarth 另保留整体可用性 mask |
| `intervention`、`intervention_mask` | `(B,7)` |
| `targets`、`target_mask`、`target_weight` | 与所选结果族的监督维度一致 |

卫星读取正式 TIFF 的六个反射率波段，保留已有像素有效掩码；mask 和观测数波段不当作额外反射率。NaN 与无效像素先按 mask 处理，不能作为真实零反射率参与统计。传感器共有波段顺序显式记录。若统一分辨率，重采样同时处理有效掩码，避免把缺测区域插值成有效观测。

街景优先复用冻结视觉骨干，在服务器预计算合法精选图片的特征；缓存绑定图片 SHA256、骨干权重及预处理版本。缓存是可重建派生资产，原始图像和正式选择表继续保留。AlphaEarth 不进入该视觉特征缓存。

完成标准：混合完整、缺年、缺街景、缺背景的 batch 可前向；缺年位置和图像/字段 masks 正确；同一个事件只绑定到其正式网格及合法时间范围。

## 3. 实现小型 S/T/C 模型

```text
POI/VIIRS + field masks ── TabularEncoder ───────┐
streetview features + image masks ── pooling ───┼── StateEncoder ── z
6-band CNN ── masked six-year GRU ──────────────┘

AlphaEarth64 + housing/population/city proxies + masks ── ContextEncoder ── b
seven intervention fields + masks ────────────────────── InterventionEncoder ── t

concat(z, t, b) ── ResponsePredictor ── response path
z_i, z_j ── response-similarity alignment
```

首版采用六通道小 CNN、带相对年份信息的轻量 GRU 时序编码、掩码池化和 MLP 融合。缺失年度跳过 GRU 状态更新，保留真实相对年份；不把中间缺年误当作末尾 padding。整段卫星或街景缺失时使用专门的缺失表示，防止全 mask 的 softmax/池化产生 NaN。

背景分支接收完整 64 维 AlphaEarth，再学习低维投影；不按维度编号拆分语义。干预分支接收七字段和 masks，不沿用仅有旧站点 bit token 的接口作为新分支的替代。

`z` 是由开通前状态 `S` 的多模态特征共同学习的唯一响应表征 embedding；完整状态向量经过训练映射到一个定长多维向量。干预 `T` 和背景 `C` 通过独立编码形成预测条件 `t` 和 `b`，只与 `z` 一起输入响应预测器。因果响应预测损失训练预测器，响应相似性对齐损失作用于 `z`。不得把开通后视觉资产作为训练目标。

完成标准：固定 S，改变有效 T/C 时，状态 embedding `z` 保持不变而响应预测发生变化；响应相似性损失可对 S 编码器回传梯度，预测损失对有效 S/T/C 分支回传梯度；改变已屏蔽像素、年份或字段不改变 `z` 与预测；全模态缺失边界条件仍产生有限结果。

## 4. 贯通训练、检查点、导出和评估

训练、CLI、检查点恢复和导出共同调用一个 batch→model 适配层，避免 canonical/modular 路径各自消费不同字段。原有 soft anchoring/监督 masks 和权重逻辑复用；监督维度从发布清单配置，不继续依赖固定四族 27 维常量。

检查点保存架构/输入契约版本、S/T/C 字段顺序、年度顺序、视觉处理、背景投影、预处理参数、监督维度、数据包 SHA256 和实验配置。旧检查点按旧 schema 恢复，新检查点必须完整恢复同一接口。正式导出先验证数据及代码绑定。

新增验收重点：时间越界拒绝；只拟合训练城市；mask 扰动不影响结果；`z` 只由 S 生成而响应预测确实使用 S/T/C；保存恢复后的 eval 预测与 `z` 在约定数值容差内一致；导出事件不丢失；禁用分支的消融确实禁用对应输入。

完成标准：本地小型合成数据闭环通过，现有旧模型回归继续通过；没有把旧 24 项测试通过当成新链路已经验收。

## 5. 在服务器候选版本进行真实数据闭环

当前服务器源码不得与候选入口混用。先核对并保存服务器版本；将白名单代码放在服务器项目内部的本次运行目录 `$RUN_ROOT/candidate/`。代码根目录与正式资产根目录分开配置：`PYTHONPATH` 指向候选目录的 `src/`，`MIT_PROJECT_ROOT` 始终指向 `/home/nas/lsr/MIT_summer`。

代码部署采用白名单，只包含所需源码、CLI、测试、依赖声明和必要文档；排除正式数据、原始图像、生产队列、既有结果、缓存、临时文件、凭证及 `.git`。真实服务器验收复用现有 `mit` conda 环境，不新建、删除或重建环境，不默认全量升级依赖。发现依赖问题时先定位，再记录并实施必要的最小修复。

先从多个城市选 64–128 个真实事件，保留既定城市划分并覆盖缺失模态案例，分别做 VIIRS 和 POI 的读取、前向、反向、短训练、保存、恢复、导出。输出进入独立 smoke 目录并标记非正式结果，不改写生产输入或队列。记录梯度、有限性、数据读取失败、吞吐与显存；批量与 DataLoader 参数以测量结果确定。

真实 smoke 验收后，冻结训练数据包与候选代码版本，再发布正式模型入口；更新代码 SHA256、部署回执和数据角色清单中的模型迁移状态。模型迁移状态只在真实训练→恢复→导出完成后置为已完成。

完成标准：本地与部署关键文件内容一致；真实资产闭环通过；正式目录和队列 SHA256 不因代码部署而变化；没有未经标记的试验输出进入正式结论。

## 6. 首轮比较实验

每个结果族在相同城市划分、监督支持集、时间截止和权重下比较：

1. `T+C` 背景/干预基线；
2. `z+T` 状态基线；
3. `z+T+C` 完整模型；
4. 完整模型移除 AlphaEarth、移除独立卫星、移除街景的三个消融。

缺失背景不能成为某一模型单独删减测试事件的理由。分别报告可用模态分层及响应 mask 覆盖。先回答各分支的增量作用，再考虑更复杂融合与多任务训练，不预先承诺预测收益。

## 文件修改地图与交付物

| 文件/模块 | 修改内容 |
|---|---|
| `causal/pretraining_dataset.py`、`build_pretraining_dataset.py` | 新角色 schema、来源绑定、逐族监督与训练城市预处理 |
| `representation/dataset.py` | S/T/C 加载、影像时序、字段/像素/年度 masks、共享 collate |
| `representation/encoder.py` | 六波段 CNN、掩码时序编码、背景与干预编码 |
| `representation/model.py` | S 状态 embedding、T/C 响应条件、配置化预测维度、旧检查点兼容 |
| `representation/trainer.py`、`training/*`、`cli.py` | 共享新 batch 接口、损失接入、配置和日志 |
| `representation/export.py`、`evaluation.py`、相关恢复入口 | 同契约恢复、状态/融合表示区分、来源与预测一致性 |
| 测试与运行文档 | 新契约回归、真实 smoke 回执、部署与实验可复现命令 |

交付按顺序为：可读取的逐族训练包、新模型及共享接口、本地回归结果、服务器真实 smoke 回执、冻结代码/数据版本、正式基线与消融结果。前三步先完成可运行的最小实现，再逐项扩展验收；工程阶段不新增数据重采任务。

依据：[输入契约](../research/state_context_input_contract_2026-10-12.md)、[代码就绪记录](stc_code_readiness.md)、[逐族训练清单](training_release_checklist.md)、[正式资产状态](input_server_sync_2026-10-13.md)。
