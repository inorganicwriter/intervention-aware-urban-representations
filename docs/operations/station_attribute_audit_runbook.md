# 现有站点属性：服务器核查

日期：2026-10-07。先核验现有数据，保留冻结因果标签，不启动神经训练。

## 需要同步哪些代码

- **只读核查**：只同步 `scripts/analysis/audit_station_attributes.py`，依赖服务器现有的 numpy、pandas、pyarrow。它独立读取 Parquet，不调用模型包。
- **方向 AUC 重评、新训练或 embedding 导出**：同步修复后的 `src/urban_intervention/representation/`。重评还需 `scripts/analysis/reevaluate_direction_auc.py`。旧训练不会因同步代码自动变化。
- `build_transit_snapshots.py` 的修复属于另一条网络特征流程。核查现有站点属性不要求先重建网络快照。

历史代码压缩包不再作为同步入口。同步时直接使用当前工作区中上述源码；需要训练或导出时，同步 `src/urban_intervention/representation/` 及相关依赖到服务器候选代码目录。若服务器有独立修改，先比较再合并，并记录同步文件的 SHA-256。正式数据留在既有资产目录。

## 跑两族核查

从项目根目录执行；如果服务器属性文件放在别处，替换 `--attributes`。

```bash
cd /home/nas/lsr/MIT_summer
for ds in viirs-mg-design poi-mg-design; do
  conda run --no-capture-output -n mit python scripts/analysis/audit_station_attributes.py \
    --attributes outputs/causal_labels/station_attributes/station_attributes.parquet \
    --dataset "data/active/model_inputs/$ds" \
    --output-dir "outputs/station_attribute_audit/$ds"
done
```

每族输出两份文件：

- `station_attribute_audit.json`：实际属性文件 SHA-256、有效训练网格覆盖率、重复键、字段取值、城市/划分缺失率、身份与开通月对齐情况。
- `station_attribute_review.csv`：按四个属性的正/负取值抽样，并加入缺失或未覆盖网格，供人工核查。使用固定抽样种子，便于复核。

脚本只写审计目录，不修改属性文件、model-input 或因果标签。本地已用隔离合成数据通过两项测试，服务器真实属性表尚待运行核验。

## 如何判读

1. `duplicate_attribute_orders` 应为空，`invalid_attribute_orders` 应为 0。重复身份会被标记为未覆盖，不会静默取第一行。非 treated 的空 treatment_order 不算非法键。
2. `covered_units` 应等于 `eligible_units`；否则先核对 `uncovered_orders`，再决定补齐身份映射或记录排除范围。
3. 四个布尔字段应存在，已观测值只能为 0/1。检查 `invalid` 和每个城市/划分的缺失情况；终点属性未知可以保留未知，需要明确建模方式。
4. `identity_checks` 中城市、站点身份、开通月的 mismatch 应为 0，同时查看 compared 数量。没有相应身份列或大量缺失时，不能由“零 mismatch”推断映射已验证。
5. 数值字段应为有限、非负整数；再人工判断范围是否合理。`n_lines_at_opening` 等于 0 的记录需要查源。

## 人工核查开通时的含义

从样本 CSV 中选约 20–30 个站点，覆盖多个城市、换乘/非换乘、新线/延伸、终点/非终点/未知。逐项核对：网格绑定的是哪个站点与哪次开通事件；日期是否对应这次事件；当时是否换乘或为终点。

优先使用现有的开通公告、历史线路图或已归档来源。当前网络图只能描述当前状态。`*_at_opening` 字段名和结构检查通过均不能代替历史来源核验，因此报告保留 `temporal_status=requires_source_review`。

现有构建脚本利用当前线路归属、邻接关系与开通日期推断部分属性。发现疑点时先核查相关记录，不要求重新获取全部数据。把“终点站”定义为当时线路端点；若研究的是列车始发/终到服务，需要另一个变量。新线与延伸、换乘与终点也不必机械地互斥。

## 核查后的实验边界

当前 station 条件输入使用换乘、新线、延伸、终点四个布尔属性，并为终点未知状态保留编码。线路数、同期开通数量等数值字段没有自动进入这一输入路径。现有五折 controls/fusion 脚本也没有站点属性增强入口。

核查通过后，固定属性文件版本与 SHA-256，再实现同一城市折下 raw 与 raw+站点属性的最小对照。预处理与选参仍仅用训练/内层验证城市。这一步只新增表示学习输入，不需要因新增这些属性而重算冻结因果标签。网络快照来源审计与是否建立新因果版本属于独立问题。

把两族 JSON 和人工核查记录汇总后，再决定哪些字段进入正式实验。
