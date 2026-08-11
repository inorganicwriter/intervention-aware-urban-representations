# 地铁站点决议与研究事件

更新时间：2026-07-23

## 当前口径

原始规范站点事件保存在：

```text
data/active/reference/transit/canonical_station_events.parquet
```

人工审核结论保存在：

```text
data/active/catalog/quality/station_issue_resolution_v1.csv
```

决议应用器不会覆盖原始站点表。它要求每条人工决议恰好命中一次，并输出：

```text
data/active/reference/transit/canonical_station_events_resolved.parquet
data/active/reference/transit/competing_transit_events.parquet
data/active/reference/transit/excluded_station_events.csv
data/active/reference/transit/station_resolution_manifest.json
```

- `canonical_station_events_resolved.parquet`：合并同一物理站点别名、修正城市归属后的主地铁事件。
- `competing_transit_events.parquet`：云巴、有轨电车等独立模式事件，用于污染、删失或敏感性分析。
- `excluded_station_events.csv`：被合并的重复记录、真实同格多站和研究范围外事件及其原因。
- `station_resolution_manifest.json`：输入/输出哈希、决议版本、行数与动作计数。

## 已冻结处理原则

- 同一物理换乘站的不同名称或线路记录合并为一个事件。
- 同一网格内的两个独立地铁站保留在空间暴露宇宙，但不进入“唯一站点事件”的主处理设计。
- 璧山等地铁与其他制式换乘的情形，只把正式地铁作为主处理；其他制式单独记录为竞争干预。
- 城市字段分错但坐标可信的记录修正归属；不属于 44 城主研究范围的事件不进入主处理清单。
- 下游一律读取 resolved 产品，不能重新根据名称模糊合并。

## 重建与验证

安装项目后运行：

```powershell
urban-resolve-stations
conda run -n mit python -m pytest tests/unit/test_station_resolution.py -q
```

随后重新运行空间 donor 审计。站点决议表、resolved 站点表和 manifest 哈希发生变化时，既有处理清单与控制设计必须视为过期并重建。

## 数据边界

站点产品已经完成身份与空间口径清理，但这不等于 5,048 个处理网格都能产生因果标签。每个网格仍需通过处理前支持、控制匹配或 GSC 拟合、结果可用性和质量门禁。
