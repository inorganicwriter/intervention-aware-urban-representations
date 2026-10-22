# 表示学习与轨道快照修复：重算范围

日期：2026-10-07。

## 本次修复

- `transfer.predictive_auc` 按每个 response cell 的观测训练行拟合等价截距；测试只使用训练侧均值。
- Dataset 对实际解析的站点属性文件字节计算 SHA-256，同时记录 token schema、覆盖数量和 treatment_order/token 映射指纹。正式 trainer 与模块化 runner 将绑定信息写入配置和 checkpoint，并拒绝三个 split 加载期间的输入变更。
- 导出核对站点绑定。旧 station-conditioned checkpoint 默认拒绝未绑定导出；显式 `--allow-unbound-station-attributes` 才能生成标记为 `unverified_legacy` 的历史导出。已绑定 checkpoint 的不匹配不能用该参数绕过。
- 轨道距离改为半径 6371 km 的球面距离，使用单位球面三维 KDTree；最近站和半径候选使用同一几何定义，半径查询无 25 站上限。
- closeness 在 cutoff 允许的节点和边上重新计算，按图缓存；保持既有 `1 / sum(reachable distance in km)` 定义。外部传入的旧全网分数不会再被采用。
- 快照默认写到 `outputs/transit_snapshots_v2/`，配版本、源数据指纹和输出哈希 sidecar。旧版或变更后的缓存不能静默跳过；只有显式 `--overwrite` 才重建同一路径。

按用户要求，P1 的 model-input 全量实际文件哈希校验暂缓。本次只绑定外部站点条件输入，不重估因果标签，不修改冻结输入，不改变表示损失。

## 数据与结果如何处理

| 产物 | 处理方式 | 是否重训 |
|---|---|---|
| 房价、POI、VIIRS、人口与影像原始数据及年度聚合 | 这四项修复不要求重新采集或聚合 | 否 |
| 既有未条件化模型的 embedding、检索指标、幅度 RMSE | 模型输入和训练目标未改变，可保留对应冻结数据口径的结果 | 否 |
| `evaluation_report.json` 中的 `predictive_transfer` 方向 AUC | 用原 checkpoint 和原划分重算 embeddings/raw 的 AUC；其他指标无需因本次修复而重算 | 否 |
| 城市五折 controls/fusion 的 ridge 幅度和方向准确率 | 已有训练侧中心化，未调用有缺陷的 AUC 分类器；可以沿用 | 否 |
| 站点属性源文件 | 原内容正确时无需重建；保存训练实际使用的版本 | 否 |
| 旧 station-conditioned checkpoint 与导出 | 如能从训练日志/归档核实原属性版本，保留原权重并做有记录的来源迁移；未知来源只能显式未验证导出，不能补写当前文件哈希冒充旧来源 | 无法核实来源且需正式可复现结果时，重训该条件化模型 |
| 新 station-conditioned 训练 | 使用当前代码，自动保存属性绑定信息 | 按原实验计划 |
| 旧 `data/active/causal/transit_snapshots/` | 本地存在 482 个旧快照；按相同开通月和城市重建到新目录，比较 6 个特征后再作新版本发布 | 此阶段不重训 |
| 接入修正轨道特征的 model-input | 新建数据集版本，重新做折内预处理和训练，禁止覆盖旧发布 | 是 |
| 使用旧快照进行静态控制匹配的设计与下游因果产物 | 先核查服务器实际来源；若使用了旧快照，修正版本需要重新进行控制设计、资格诊断、受影响标签估计和发布，再构建下游 model-input | 若更新标签或特征，则重训受影响模型 |

### 轨道快照也被因果控制设计读取

Python `src/urban_intervention/causal/gpu/control_design.py::_static_features` 和 R `scripts/causal_r/grid_control_design_lib.R::read_city_static_blocks` 都会读取旧快照的六个变量。因此，“轨道快照没有直接进入表示学习特征”不能推出“已有因果标签完全无影响”。

本地冻结队列和快照不能证明服务器生产所用版本。应核对服务器控制设计规格、特征列、来源指纹与 donor 选择产物。若确认未使用旧变量，因果标签无需因本次修改重算；若使用过，旧标签保留作原版本研究记录，修正因果版本需走完整资格与发布流程。当前冻结标签保持原样，是否建设修正因果版本由来源核查结果决定。

### 历史邻接数据的限制

现有 `wikidata_adjacency.parquet` 只有当前邻接关系，没有边的有效起止日期。代码已排除未来节点对最短路径的影响；它计算的是“当前邻接表在当时可用站点上的子图”。这仍不等于已验证的历史网络，例如后插入站点可能改变旧站之间的邻接关系。

代码支持可选的 `valid_from` / `valid_to` 边日期（起始包含、终止不包含）。要把 closeness 作为严格的处理前网络特征接入正式新实验，需要核查/重建历史有效边，并审计站点 `lines` 的历史归属；不要将无日期的当前拓扑代理宣称为真实历史网络。sidecar 明确记录边日期是否可用。

## 重建命令

方向 AUC 可以独立重评，不重新训练、不覆盖原 `evaluation_report.json`：

```bash
conda run -n mit python scripts/analysis/reevaluate_direction_auc.py \
  outputs/representation/variance-default/seed_1/best_model.pt \
  data/active/model_inputs/viirs-mg-design \
  --output-dir outputs/representation_reevaluation/viirs/seed_1
```

该入口沿用原 `training_config.json` 的 ridge 与最少观测数（缺失时分别为 1 和 16），使用原城市划分及训练掩码，输出验证/测试的 embeddings/raw AUC，并另存推理嵌入池。其他 seed 与 POI 用各自对应 checkpoint 和数据集运行。旧 station-conditioned 模型需提供原属性文件，并遵守上文未验证兼容规则。

从项目根目录执行，默认输出位于 `outputs`，不覆盖旧生产快照：

```bash
conda run -n mit python scripts/analysis/build_transit_snapshots.py --city all
```

也可以先只处理一个城市，例如：

```bash
conda run -n mit python scripts/analysis/build_transit_snapshots.py --city beijing \
  --output-dir outputs/transit_snapshots_v2_review
```

新快照用于复核距离、站点数量和历史子图计算；当前控制设计仍读取旧生产目录。本次没有切换该目录、重跑队列或覆盖任何标签。只有核查、比较和版本发布完成后，才迁移正式消费者。

## 需要补充获取的数据

1. 历史站间连接表：城市、线路、起终站身份、连接生效日期，以及发生取消/改线时的失效日期。两站都已开通不能证明当时相邻；当前邻接表中的后来插站、延伸与改线需要历史证据。
2. 站点—线路历史归属：站点身份、线路身份、该站在该线路上开始服务的日期，以及适用的终止日期。站点首次开通和线路首次开通日期不能替代这项信息；它影响历史线路数、换乘和终点属性。
3. 坐标或开通日期的缺失/无效项：核验后只补缺失或有争议的记录。现有完整记录可以继续用于球面距离计算。

历史表保留来源链接/文档、获取日期和匹配到 canonical station 的规则。缺失历史日期明确标记未知，不自动用现代关系或任意早期日期补齐。先覆盖实际实验的城市和 cutoff。当前快照构建器支持已有邻接行的 valid_from / valid_to；要使用新收集的历史连接行，需经身份映射后提供给邻接输入。站点—线路历史表尚需单独接入消费者，不能只新增文件就宣称生效。

房价、POI、VIIRS、人口、影像无需因为本次修复重新获取。可以先继续原冻结标签口径的融合评估；构建修正的因果/历史网络版本前再完成上述来源核查。

## 服务器只读核验

将 `scripts/analysis/audit_transit_input_lineage.py` 同步到服务器后，从实际生产项目根目录执行。默认抽查每个目录前 50 份记录；`--max-records 0` 扫描全部。该脚本只读输入，JSON 通过标准输出返回。

```bash
cd /home/nas/lsr/MIT_summer
mkdir -p outputs/transit_lineage_audit
conda run --no-capture-output -n mit python scripts/analysis/audit_transit_input_lineage.py \
  --root . \
  --control-root outputs/control_design/tasks \
  --control-root outputs/causal_labels/tasks \
  --max-records 50 \
  > outputs/transit_lineage_audit/inventory.json
```

**目录必须与实际发布来源一致。** 上述是仓库标准目录；如果发布用的是 robust_m7、cross-city 或其他生产任务根，把它们作为 `--control-root` 传入。不要用最近更新时间或当前队列猜测发布版本。应先从发布 manifest、运行命令、日志/资格回执找到实际任务根、队列和任务身份，再将核验记录的 treatment_order、scope、control_unit_key 等与发布来源对上。

判读：

- `reference_inputs.wikidata_adjacency.parquet.history_status = missing_edge_dates`：现有输入没有历史连接日期。
- `model_inputs[].transit_feature_columns`：检查轨道变量是否直接进入模型输入；空列表不能排除其通过上游控制匹配影响标签。
- `control_design[].records[].documented_transit_features`：active_static_features 或 feature_balance 中确实保存的轨道变量，是对应控制设计使用轨道特征的证据。
- `not_documented_not_proof_of_absence` / `unknown`：证据不足，需要继续查 R 的 control_design.rds、旧代码、运行配置与来源记录。
- `sample_complete = false`：仅抽查；没有发现正例不能作为整体未使用的结论。零条记录也不能作未使用结论。

`transit_documented` 证明对应任务有轨道使用记录，不自动证明它被当前标签发布采用。确认实际发布引用该任务、且任务当时读入的是旧快照后，再判断受影响控制设计及下游标签的重算范围。若缺少快照内容指纹，文件路径相同也不足以证明版本相同。
