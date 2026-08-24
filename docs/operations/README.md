# Operations

本目录只保存当前仍有效的操作说明：

- [`server_deployment.md`](server_deployment.md)：服务器部署、上传清单、生产运行序列与监控。
- [`current_project_status.md`](current_project_status.md)：固定资产、队列状态、已完成与未完成边界。
- [`station_data.md`](station_data.md)：站点人工决议及三个可追溯产品。
- [`housing_price_collection.md`](housing_price_collection.md)：房价采集、开放数据导入与验收。
- [`viirs_monthly.md`](viirs_monthly.md)：月度 VIIRS 缓存和网格聚合。
- [`poi_panel.md`](poi_panel.md)：POI 年度面板构建和审计。

处理网格与 1km donor 空间审计的冻结口径见
[`../research/decisions/DDR-001_spatial_treatment_and_donor_exclusion.md`](../research/decisions/DDR-001_spatial_treatment_and_donor_exclusion.md)。
被取代的阶段性文档统一归档于 [`../archive/`](../archive/README.md)。

## 支持的入口

```powershell
conda run -n mit python scripts/data_management/validate_registry.py
conda run -n mit python scripts/analysis/audit_housing_acquisition.py
conda run -n mit python scripts/analysis/audit_housing_panel.py
conda run -n mit python scripts/analysis/audit_poi_panel.py
conda run -n mit python scripts/analysis/audit_viirs_monthly_downloads.py
urban-resolve-stations --help
urban-spatial-donor-audit --help
```

正式反事实生产不从 `analysis/` 启动。Python/GPU 资格按
[`causal/gpu/README.md`](../../src/urban_intervention/causal/gpu/README.md) 执行，
并行部署与 R 参考环境按 [`scripts/causal_r/README.md`](../../scripts/causal_r/README.md)
执行。
