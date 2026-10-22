# 街景服务器重建与项目内存储：2026-10-13

44 城市的元数据重建、开通前和事后精选图片核验及项目内复制已完成，合计 18,397 张图片、5,151,538,800 字节。开通前和事后正式清单均已发布，最终验收以目标目录 `publication_verification.json` 为准。按照用户最新要求，完整街景资产直接复制到 `/home/nas/lsr/MIT_summer/data/active/streetview_grid_v2/`，不再下载图片或全量数据表到 Windows。

## 已核验的覆盖

| 项目 | 数量或规则 |
|---|---|
| 冻结事件与城市 | 5,048 个事件，44 城；全部保留 |
| 原始成功元数据 | 505 个 CSV，81,508,576 行 |
| 正式处理网格内的去重元数据候选 | 2,531,781 行；候选图片尚未逐张核验 |
| 有合法开通前图片的事件 | 3,778，约 74.84% |
| 本次选用的图片 | 7,543 张，2,088,675,020 字节；每事件最多两张 |
| 没有选到合法开通前图片的事件 | 1,270；保留 mask 和原因 |
| 实际选用图片的解码、尺寸、校验和错误 | 0 |

6,500 次失败的路径查找或解码尝试不等于 6,500 张损坏图片：原始下载按全局 panoid 去重，同一图片可能存在于另一县目录。程序按精确文件名查找，失败后尝试同城其他县或更早的合法候选。实际选用的 7,543 张均已逐张完整解码，并记录 SHA256。

本次是原始街景档案的重新入格、筛选和项目内复制，没有重新抓取百度街景，也不声称每个事件具有连续年度街景。最多两张是当前状态输入的缓存配置；完整历史元数据候选另外保存，可用于后续扩充。

用户补充要求开通后的街景也应保存。事后档案按开通年 0 至 +3 年建立全部 20,192 条事件年度记录，每个事件每年最多选择两张经解码及校验和核验的图片，逐年记录缺失。月精度要求 `capture_month > opening_month`；开通当月的时间关系为 unknown，候选数单独记录。所有事后记录 `state_allowed=false`，用于变化分析或结果核验。事后图片已完成核验和项目内复制，发布复用经过解码及 SHA256 核验的复制回执，继续核对路径、字节数、元数据绑定和 masks。

开通前的 1,270 个缺失事件中，775 个只存在开通当月或之后的网格内拍摄，491 个正式网格内没有全景元数据，4 个有开通前候选但没有核验通过的图片。正式覆盖表保留全部事件，不把缺失解释为效果为零。

| 事件年 | 全部事件记录 | 有核验通过图片的记录 |
|---|---:|---:|
| 0（仅严格开通后月份） | 5,048 | 504 |
| +1 | 5,048 | 1,744 |
| +2 | 5,048 | 1,688 |
| +3 | 5,048 | 1,553 |
| 合计 | 20,192 | 5,489 |

共有 2,914 个事件至少一年有事后图片，精选 10,854 张，3,062,863,780 字节。年度缺失状态包括：11,241 条目标年没有网格内街景元数据、2,010 条尚未结束年度、1,239 条开通年仅有开通前拍摄、198 条仅有开通当月不确定候选、15 条有合法事后候选但没有核验通过的图片。缺失不是因果效应为零，图片不填补缺失年份。原始档案最晚月份因城市而异，多数截至 2021–2023 年；连续年度街景覆盖明显低于独立卫星影像。

独立卫星原有 −6 至 +3 年档案已包含开通当年和后三年，事后四个年度共 18,182 帧可用、2,010 条未结束年度占位。开通当年的卫星全年合成同时包含处理前后观测，作为事件年档案保存；不能称作严格开通后采样，也不能进入处理前状态输入。

## 空间与时间校正

正式单元沿用原因果分析的城市固定 500m × 500m 网格多边形，不以站点为中心另划网格。街景通过真实拍摄位置唯一入格，不把完整视野视为网格裁切影像。

原始成功 CSV 中 `wgs_lon/wgs_lat` 是查询点，拍摄位置必须从 `ret_bd_x/ret_bd_y` 转换。检查发现返回 BD09MC 坐标放大 100 倍，而旧 `step5_post_process_coords.py` 未消除倍数，导致转换目录出现越界经纬度。本轮使用原始成功 CSV，检查两个坐标轴的单位，重新执行 BD09MC → BD09 → GCJ02 → WGS84；保留查询坐标、原始返回值和单位倍数。WGS84 是坐标转换估计，未经过独立 GPS 地面校准。共享边界的多格歧义不绑定。

采集器既支持年月，又支持年月日，但成功 CSV 没有日期精度字段。本轮统一按月核验，要求 `capture_month < opening_month`，排除开通当月与之后的图片。没有用查询点、当前图片或开通后图片填补缺口。

冻结处理清单 SHA256：`8525e80c20a76b9e99b9e7592a9c6e4b1203a2f183513c2a4235992e952c8fd1`。

事件和正式网格合同 SHA256：`adcda30b90403b8cd7f90261dbe2cfd394eb515d97c1fda8166b60d237013950`。服务器项目中的处理清单与该合同一致。

## 服务器资产位置

正式输出位于 `/home/nas/lsr/MIT_summer/data/active/streetview_grid_v2/`：

- `images/{city_key}/{panoid}.jpg`：项目内复制的原始图片，复制后核对字节数、SHA256 和实际解码尺寸。
- `streetview_state_assets.parquet`：按事件绑定的真实开通前图片及时间、坐标、几何哈希、来源和校验和。
- `streetview_index.parquet`：已复制并核验的精选图片索引。
- `event_coverage.parquet`：全部 5,048 个事件、图片路径列表、图片数量、可用性 mask 和缺失原因；可按 `treatment_order` 与服务器其他输入表连接。
- `streetview_server_index.parquet`：正式处理网格内的全量元数据候选。`asset_bytes` 缺失，`image_verification_status=metadata_candidate_image_unverified`，不能把候选数当作真实可用图片数。
- `copy_receipt.parquet`、`server_audit.json`、`binding_audit.json`、`frozen_grid_contract.json`、`manifest.json`：复制与绑定证据、冻结身份和最终文件校验和。
- 事后档案另存 `streetview_post_assets.parquet`、`streetview_post_event_coverage.parquet`、`post_copy_receipt.parquet`、`post_server_audit.json`、`post_binding_audit.json` 和 `post_manifest.json`。开通前选择表保持独立的时间门禁；事后图片也复制到项目的 `images/`。

原始代码仍位于 `/home/nas/lsr/BaiduSvs_history/BaiduSV_National_Collector`，原始图片与成功元数据仍位于 `/home/nas/lsr/BaiduSvs_history/output`。原始档案没有改写。重建时曾使用 NAS 中间目录 `/home/nas/lsr/BaiduSvs_history/mit_grid_alignment_20261013` 和本机暂存 `/tmp/mit_streetview_grid_20261013`；正式发布、输入绑定和验收完成后，这两个重复暂存已按用户要求清理。重要日志、逐城审计和选择证据保存在项目 `outputs/input_sync_20261013_v1/cache_cleanup_evidence/`。最终资产和复现代码保留在项目目录。

Windows 上在用户修改存储要求之前已下载的北京试验缓存有 380 张图片，对应 190 个可用事件；它是局部验收材料，不能代替服务器全城发布，也不能称作全城训练资产。用户修改要求之后没有继续将图片或全量表下载到客户端。

## 运行与实现边界

重建代码：[rebuild_remote_streetview_index.py](../../scripts/collection/rebuild_remote_streetview_index.py)、[streetview_coordinates.py](../../scripts/collection/streetview_coordinates.py)。项目内复制与发布：[publish_server_streetview_inputs.py](../../scripts/analysis/publish_server_streetview_inputs.py)、[publish_server_streetview_post_archive.py](../../scripts/analysis/publish_server_streetview_post_archive.py)。最终验收：[audit_server_streetview_release.py](../../scripts/analysis/audit_server_streetview_release.py)。服务器正式目录另保存 `provenance/` 脚本副本、运行代码 SHA256 和本报告，复现不依赖客户端图片缓存。

北京时间 2026-10-13 19:44，最新 AlphaEarth、独立年度卫星和 S/T/C 也在同一服务器项目完成正式入库与绑定。269,738 个文件全部通过 SHA256 校验，输入绑定 37 项与路径验证 17 项全部通过。服务器状态表已连接完整的开通前街景与卫星路径；数据准备阶段完成，可以进入模型接入和小规模训练验证。历史缺失及未结束年度保留 masks，干预历史属性仍需来源复核。新模型接口、年度卫星编码和训练效果尚待验证，详见 [输入同步记录](input_server_sync_2026-10-13.md)。本目录原发布清单与 15:57 的历史验收记录保持不变，当前完整输入状态以同步回执为准。

此前相关回归 26 项通过；新增复制和事后门禁校验后，受影响的街景测试文件 9 项全部通过，Ruff 通过。复制使用校验和确认实际字节一致，并拒绝目录越界、改变的源文件或与核验结果不一致的现有目标图片。精选表与全量索引一条纬度出现 CSV 重复读写的一个浮点末位差异（约 3.55×10⁻¹⁵ 度）；数值比较仅允许四倍机器精度的相对误差，实际位置改动仍拒绝，完整网格和时间检查保持有效。全量几何绑定 141 项检查已经通过。

最终服务器发布验收在 2026-10-13 15:57（Asia/Shanghai）完成，45 项检查全部通过，物理图片清单确认 18,397 张。核对了正式文件 SHA256、冻结事件和网格合同、前后时间门禁、年度与事件 masks、复制回执、图片路径和全部绑定检查。客户端仅保存 [服务器验收摘要](../../outputs/streetview_server_audit/2026-10-13/server_publication_receipt.json)，完整图片及全量表都在服务器；服务器目录的 `publication_verification.json` 保存正式检查明细。
