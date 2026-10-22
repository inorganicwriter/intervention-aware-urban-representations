# 独立卫星影像与角色输入表

日期：2026-10-13。独立卫星影像、街景及角色输入资产已完成验收并在服务器正式入库与绑定；数据准备阶段完成，下一步为模型接入和小规模训练验证。

## 影像口径

每个冻结处理事件使用其正式 500m × 500m 网格多边形。用户已确认逐年补齐开通前 6 年至开通后 3 年，包含开通当年：`event_time=-6,-5,-4,-3,-2,-1,0,1,2,3`。5,048 个事件对应 50,480 条网格—年份记录；截至 2026-10-13，48,470 条对应已结束的日历年，2,010 条对应尚未结束的年度，后者保留 `calendar_year_not_complete`，不下载部分年度充当全年观测。

影像在每个目标日历年内构造清晰像素的逐波段中位数。状态输入仅消费 `-6..-1`，共 30,288 条时序记录；开通年与开通后 `0..3` 留在全量档案，`state_temporal_allowed=false`。不会使用事后影像填补开通前缺口。年度合成不是某一天的原始照片。

| 目标年份 | 首选产品 | 原生分辨率与导出尺寸 |
|---|---|---|
| 2018 年及以后 | Sentinel-2 SR Harmonized | RGB/NIR 10m，SWIR 20m；50 × 50 像素 |
| 2013–2017 | Landsat 8 Collection 2 Tier 1 Level 2 | 30m；17 × 17 像素 |
| 2012 | Landsat 7 Collection 2 Tier 1 Level 2 | 30m；17 × 17 像素 |
| 2004–2011 | Landsat 5 Collection 2 Tier 1 Level 2 | 30m；17 × 17 像素 |

Sentinel-2 历史覆盖或有效像素不足时使用同一目标年的 Landsat 8；Landsat 5/8 可回退到同年 Landsat 7。产品不会在同一 patch 中混合。Landsat 的 17 像素精确覆盖 500 米，输出像元间距约为 29.41m，原生信息仍为 30m。Sentinel-2 的 SWIR 以最近邻重采样至导出网格，原生信息仍为 20m。传感器差异与早期影像较粗的限制应在后续实验中记录，并检查传感器/时期分层表现。

官方依据：[Sentinel-2 SR Harmonized](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)、[Landsat 8](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2)、[Landsat 5](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2)、[Landsat 7](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2)。

S2 使用 SCL 剔除无数据、饱和、阴影、云、卷云和雪冰；不依赖历史上部分年份缺失的 QA60。Landsat 使用 QA_PIXEL 位 0–5 与 QA_RADSAT。所有六个波段必须有共同有效观测。S2 反射率缩放为 0.0001；Landsat 缩放为 0.0000275 并减 0.2。未观测像素为 NaN，不当作零反射率。

导出用城市 UTM 投影和正式网格的精确仿射变换，显式 clip 正式多边形。`getDownloadURL` 指定投影与仿射变换时 `region` 被忽略，不能只传一个经纬度范围而认为影像已按网格裁切。[Earth Engine 下载接口](https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl)

## 文件

完整年度时序根目录：`data/active/satellite_event_sequences_v2/`。全量采集与逐张验收已完成：50,480 条记录、48,470 张实际影像；48,465 张可用，5 张低覆盖保留 mask，2,010 条未结束年度无影像。逐张审计错误数为 0。开通前六年状态索引已发布到 `multimodal_inputs_v1/satellite_state_sequences.parquet`。详见 [补齐与验收报告](input_asset_completion_2026-10-13.md)。

此前的 `data/active/satellite_grid_images_v1/` 仅含开通前一年，5,048 张已完成逐张验收，错误数为 0。该目录保留为基线缓存；时序采集校验后通过硬链接或复制复用同一年度资产。不能把这 5,048 张基线影像称作完整历史窗口。

- `images/{city}/{order}_{year}_{sensor}.tif`：八个 float32 波段，顺序为 red、green、blue、nir、swir1、swir2、valid_mask、observation_count。
- 对应 `_rgb.png`：固定反射率范围 0–0.3 的原始尺寸预览；`_mask.png`：有效像素掩码。模型应消费反射率 TIFF 和掩码，不能只凭 PNG 黑色判断缺失。
- `sources/{city}/...json`：空间相交的候选源景 ID 和实际日期。它描述合成的候选场景，不声称每一景对每个像素都有贡献。
- `records/{city}/{order}_{year}.json`：空间哈希、时间与用途、传感器、原生分辨率、输出尺寸、有效比例、文件校验和、回退记录。
- `satellite_sequence_queue.parquet`：用户确认的完整 50,480 条采集队列。
- `satellite_sequence_index.parquet`：每个事件每年一行，保留不可用年度与未来年度。
- `satellite_sequence_audit.json`：运行状态和覆盖计数。中途索引可能尚未收录正在处理的批次；实际已下载数量可从 records 目录核对。

有效像素比例至少 0.8 时状态模态可用；不达标也保留真实影像和事件，用模态 mask 控制使用。重新执行采集会校验契约、网格、影像校验和及预览文件后续传。

下一日历年只改变“已结束年度”界限时，原有已核验的历史帧会复用并保留实际采集时间；新结束的年度重新获取。若波段、掩码、合成或质量契约改变，不复用旧契约帧。当前未来年度占位行没有影像文件，届时不会被当成缓存成功记录。

```powershell
& D:/Anaconda/envs/mit/python.exe scripts/collection/download_satellite_event_sequences.py --plan-only
& D:/Anaconda/envs/mit/python.exe scripts/collection/download_satellite_event_sequences.py --workers 16 --batch-size 32
& D:/Anaconda/envs/mit/python.exe scripts/analysis/audit_satellite_grid_images.py --sequence
```

指定 `--orders` 或 `--years` 的试采应使用独立 `--output-dir`，避免子集覆盖正式索引。全量索引只在 `run_finished=true`、覆盖全部事件的十个年度且逐张审计通过后接入角色表。

审计以不可变索引快照核对每张 TIFF、预览和源景记录，使用独立本地进程并行验收；网格边界误差容限为 1mm。`--allow-incomplete` 可用于检查采集中间快照，报告保留 `full_coverage_verified=false` 和缺失覆盖说明，不能用于正式发布。2026-10-13 首个中间快照含 15,270 条记录，14,586 张实际影像逐张审计错误数为 0。

批量请求临时把多个已按原始投影采样的 patch 平移到传输条带，下载后立即拆分并恢复每张原始仿射变换；条带不作为地理资产保存。多传感器试采中的 12 张同年样本与已有单张下载的八个波段及仿射变换完全一致，比较记录保存在 `outputs/satellite_sequence_pilot/`。正式全量发布验收后，试采目录中的临时影像、采集记录分片与来源文件已清理，比较报告、试采索引和审计明细仍保留；正式年度影像没有清理。每张最终 TIFF 仍须独立审计。

## 角色输入表

`data/active/multimodal_inputs_v1/` 保存未经训练集拟合预处理的输入表。它是资产发布层；当前模型尚未迁移完成，不能把该目录当作已训练模型或包含监督标签的正式训练发布包。

| 文件 | 内容 |
|---|---|
| `state_inputs.parquet` | POI 四个基线字段与 VIIRS，开通前 1/2/3 年精确年度观测及 masks；卫星基线引用、六年时序可用计数；服务器绑定 3,778 个可用事件的完整街景，本地保留 190 个事件的北京试验 |
| `satellite_state_sequences.parquet` | 审计通过后发布的开通前六年影像索引，仅含 `event_time=-6..-1`，每年有独立可用性与像素 mask |
| `context_inputs.parquet` | 房价、人口的开通前 1/2/3 年观测；完整 AlphaEarth64 及 mask；城市网格统计代理和覆盖信息 |
| `intervention_inputs.parquet` | 七个干预字段及对应 masks，历史来源审阅状态 |
| `city_background_proxies.parquet` | 各城市/年的全部源网格汇总与覆盖数；不是处理网格的样本均值 |
| `manifest.json` | 输入分工、时间规则、来源 SHA256、覆盖数、街景和模型迁移状态 |

城市背景字段为全体已观测参考网格的 POI 总量对数、人口总量对数、平均夜光的 asinh、平均房价对数。它们是已有资产支持的网格统计代理，不是官方城市年鉴；房价均值只针对有观测的网格。保留各产品的观测网格数及占正式城市格网的比例，不能据此直接宣称全市人口总量或具有代表性的全市房价。各事件使用开通前一年城市值。未接入 GDP 等没有核验的外部统计。

原始年度表逐批核对 city/grid/year 唯一性和正式网格归属。缺少指定年份或观测时保留 mask；不借后续年份填充。对数/asinh 是固定变换，没有在全部城市上拟合归一化或降维参数。后续标准化仅由训练城市拟合。

干预输入通过冻结清单的城市、站点事件和日期核对后发布。783 个原始终点未知保留缺失；另一个线路数为 0 的事件 `treatment_order=3537` 屏蔽线路数与四个相关类型字段，最终终点字段可用 4,264 个。结构可用不等于历史线路归属已独立核验；所有重建属性保留 `requires_source_review` 标记。

```powershell
& D:/Anaconda/envs/mit/python.exe scripts/analysis/build_station_attributes.py
& D:/Anaconda/envs/mit/python.exe scripts/analysis/build_intervention_inputs.py
& D:/Anaconda/envs/mit/python.exe scripts/analysis/build_role_input_tables.py
& D:/Anaconda/envs/mit/python.exe scripts/analysis/build_role_input_tables.py --refresh-sequences
```

模型迁移遵循 [状态、干预与背景输入契约](../research/state_context_input_contract_2026-10-12.md)：状态编码器消费 POI/VIIRS/街景/独立卫星影像，背景编码器消费 AlphaEarth/房价/人口/城市代理，干预独立编码。完整街景与 −6 至 +3 年卫星档案均已在服务器项目正式存储；开通当年及事后影像不进入处理前状态。本轮不加入路网。服务器路径、绑定回执和临时缓存清理状态见 [输入同步记录](input_server_sync_2026-10-13.md)。
