# 历史视觉资产配对审计（不属于模型训练契约）

日期：2026-10-19（2026-10-19 更新街景时间编码）  
状态：此文档记录此前生成的前后视觉资产配对审计及其历史输出。它不是当前模型的输入/目标契约，也不要求训练视觉状态转移。当前模型以结构化 S/T/C 输入形成多维向量 `[z, t, b]`，并由因果响应标签监督；街景多图聚合属于 S 输入。开通后视觉资产不进入该模型。

## 历史审计范围

历史上曾为检查不同年份视觉资产的覆盖、空间身份和时间窗口，构造过前后配对索引。以下字段、mask、统计量和服务器输出仅用于解释该审计产物，不应接入当前响应模型的训练数据或损失。当前任务没有开通后街景目标，也没有视觉状态转移损失。

## 配对粒度与空间身份

一个索引行对应：

```text
treatment event × frozen grid × target event_time (0..3) × modality
```

冻结身份字段为 `treatment_order`、`city_key`、`grid_id`、`station_event_id`、`opening_month` 和 `grid_geometry_sha256`。所有输入表须逐行与卫星 −1 年事件 roster 完全匹配；不匹配时构建失败，不按行号、站名或最近邻静默拼接。

每个事件保留四个目标跨度和两个模态，因此完整索引含 `N_events × 4 × 2` 行。一个后状态槽位可以绑定同月的一张或多张已核验图片；街景图与卫星 patch 表示同一固定网格的状态，不声称相机视角逐图相同。

## 模态时间规则

### 街景

- 源状态来自正式开通前资产表，合法条件为 `capture_month < opening_month`。
- 不只取最近一张：每个事件的所有合法开通前街景都进入 `pre_state_assets_json`，形成长度可变的影像集合；每张图独立保留月份差和压缩年龄。无街景事件由 `pre_state_mask=false` 表示，批处理时用逐图有效性 mask 屏蔽 padding。
- 每个目标 event time 的后状态来自独立事后档案，合法条件为 `capture_month > opening_month`、年份与 `opening_year + event_time` 一致且 `image_valid=true`。
- 月份按 `YYYY-MM` 保留。每张源图和目标图各自记录月份及距开通月份数；不把标准化为月初的 `capture_date` 当成日精度。
- 每张开通前街景保留原始 `age_before_opening_months`，并增加 `age_before_opening_compressed = d / (d + 12)`，其中 `d` 是距开通的正月份数。该单调、有界变换用于让转移头感知影像新旧程度；它只是时间条件特征，不是样本权重或损失权重，不会因街景较早而删除影像或降低其监督权重。12 个月对应压缩值 0.5，例如 1、12、139 个月分别约为 0.077、0.5、0.921。
- `event_time=0` 的后街景可以作为目标，因为其选中图片严格晚于开通月。档案中的 `state_allowed=false` 表示禁止用作 predictor 输入，不妨碍它在独立的 target-only 索引中作为监督目标。
- 现有事后档案每事件年最多两张、优先较晚拍摄月份；其时间不等于开通后最近一张。索引如实记录已选月份，不重新声称或伪造“最近拍摄日”。

### 卫星

- 源状态保留六个固定槽位 −6..−1，包括缺失槽位及其 mask；只有 `state_temporal_allowed`、`state_image_available`、完整日历年和 `satellite_image_available` 同时成立时，该槽位才有可用资产。
- event time `+1..+3` 的完整年度合成可作为后状态目标，保存 `source_date_min/max`、传感器、场景数、有效像素率和资产校验值。
- event time `0` 的年度合成可能混合开通前后观测，即使档案中有影像，也令 `post_target_mask=false`，并标记排除原因。
- 年度合成源日期范围是场景窗口元数据，不代表每个像素对应某个精确拍摄月或某天。后状态跨度以日期窗口和月份边界表达。

## Mask 契约

所有 mask 只对该行指定的模态和目标跨度有效：

| 字段 | 定义 |
|---|---|
| `pre_state_mask` | 至少存在一个合法、已发布的开通前源资产；卫星源为固定六年槽位，其中至少一个可用 |
| `post_target_candidate_available` | 原始索引报告目标期存在资产候选；允许与 `post_target_mask` 不同，例如开通年卫星有候选但不能监督 |
| `post_target_mask` | 至少一个目标资产通过模态的时间规则、可用性与年份校验 |
| `identity_mask` | 事件、地点、开通月和固定网格几何均通过严格绑定；当前构建器不允许身份失败后继续 |
| `pre_temporal_mask` | 源状态时间槽位符合开通前定义 |
| `post_temporal_mask` | 目标状态符合模态及跨度定义 |
| `transition_supervision_mask` | `pre_state_mask & post_target_mask & identity_mask & pre_temporal_mask & post_temporal_mask` |
| `post_target_predictor_use_allowed` | 恒为 false：后状态资产不能进入 predictor 输入；配对行中的前状态资产仍可作为 predictor 输入 |
| `target_use_allowed` | true 表示该目标角色可用于转移监督；卫星开通年为 false |

街景和卫星分别生成 mask，不要求两种模态同时有图。训练时按模态 mask 计算并归一化各自损失，避免将街景缺失误当作卫星目标也缺失。

## 输出

执行 `scripts/analysis/build_visual_transition_pair_index.py`，默认在 `data/active/` 下按文件名唯一解析以下输入：

- `satellite_sequence_index.parquet`
- `streetview_state_assets.parquet`
- `streetview_post_assets.parquet`

同名输入不唯一时，构建器报错并要求通过命令行显式传入路径，避免按目录名版本号静默选错数据。

默认在新的 `outputs/visual_transition_pairing/pair_index_<UTC时间>/` 目录写入。服务器当前产物位于：

`/home/nas/lsr/MIT_summer/outputs/stc_server_acceptance/stc_acceptance_20261019T091255Z/results/visual_transition_pairing/`

- `visual_transition_pairs.parquet`：配对资产引用、实际月份/窗口、逐模态 mask 和 target-only 角色；街景源引用逐图保存原始与压缩后的开通前年龄。
- `visual_transition_pair_audit.json`：来源文件哈希、按模态与目标跨度汇总的源/目标/有效监督配对数，以及输出哈希。

当前构建器写入不带版本号的 schema `visual_transition_pair_index`。较早生成的历史产物保持原样；当前输出目录包含逐图压缩年龄字段。

构建器不复制或修改正式图片、标签、active 索引或当前训练包；输出目录必须在 `data/active/` 之外，且不会覆盖已有目录。

### 2026-10-19 服务器构建结果

5,048 个事件各保留 4 个目标跨度（0..3）和 2 个模态，共 40,384 行。`transition_supervision_mask` 只在前状态、后目标、严格身份及双方时间 mask 都有效时为 true。以下“目标可用”是索引中通过来源和时间检查的后状态；监督配对数还要求该事件有合法前状态：

| 模态 | event_time | 前状态可用事件 | 后目标可用 | 有效监督配对 |
|---|---:|---:|---:|---:|
| 卫星 | 0 | 5,048 | 0 | 0 |
| 卫星 | +1 | 5,048 | 4,738 | 4,738 |
| 卫星 | +2 | 5,048 | 4,351 | 4,351 |
| 卫星 | +3 | 5,048 | 4,045 | 4,045 |
| 街景 | 0 | 3,778 | 504 | 433 |
| 街景 | +1 | 3,778 | 1,744 | 1,526 |
| 街景 | +2 | 3,778 | 1,688 | 1,391 |
| 街景 | +3 | 3,778 | 1,553 | 1,077 |

开通年卫星即使有候选影像也不作为目标；开通年街景只保留拍摄月份严格晚于开通月的图像。所有开通后资产均标记为 target-only，不能进入生成预测状态的 predictor。构建器、输入表和产物 SHA256 均记录在本次输出目录的 `visual_transition_pair_audit.json` 中；上表配对数量与当前重建输出一致。

本轮索引构建于 `2026-10-19T10:10:07Z` 完成。配对 Parquet SHA256 为 `a15a9b73802f627981d0110a53ca3dd37b405c2da4464ad51443b9bb2ad6dc65`。

另用正式配对索引和真实图像完成了街景输入探针：6 个有效配对、6 个事件，共实际加载 12 张开通前图和 12 张 target-only 开通后图。开通前各图的压缩时间差为 0.333–0.600；集合编码输出形状为 6×64，数值有限。预测编码器只接收开通前图、逐图时间差和 mask；开通后图没有传入预测编码器。机器记录位于同一运行目录的 `reports/streetview_transition_input_probe.json`。

## 当前模型的正确输入路径

当前街景输入位于 `src/urban_intervention/representation/dataset.py`、`encoder.py` 和 `model.py`：全部合法开通前图片及逐图开通月间隔进入状态分支，聚合成一个定长街景向量；该向量与其余状态输入一起编码为 `z`。`[z, t, b]` 经响应预测器拟合正式因果响应标签。上文列出的开通后配对数据仅为历史资产审计输出，不由训练入口读取。
