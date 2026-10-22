# 最新输入资产上传与服务器绑定：2026-10-13

用户授权将本地新增输入资产上传到服务器，与现有街景及冻结因果数据共同保存在 `/home/nas/lsr/MIT_summer`。正式复制、目录发布、输入绑定和路径读取验证均已完成，最终验证时间为北京时间 2026-10-13 19:44:49。

完成证据：269,738 个文件、1,872,585,974 字节全部写入正式 NAS 项目并逐文件核验 SHA256，五个新增目录已发布到 `data/active/`。服务器输入绑定 37 项检查通过，后续路径验证 17 项检查通过，44 个城市的卫星 TIFF、mask、来源 JSON 及可用街景抽查均可读取。年度和状态序列的全量路径对应关系也通过检查。原冻结处理清单和已有街景发布清单哈希保持不变。

NAS 维护期间曾出现 I/O 等待；重新登录后确认后台任务存活且持续推进，随后完成发布。复制、绑定及验证进程均已正常结束。正式回执位于服务器 `outputs/input_sync_20261013_v1/`，包括 `transfer_receipt.json`、`integration_receipt.json` 和 `runtime_verification.json`；原始角色元数据、完整来源清单及执行脚本也持久保存在同一运行目录。服务器目录清单为 `data/active/catalog/inventories/input_bundle_server_20261013.json`。本地仅保存这些小型回执；完整街景没有下载到本地。

服务器角色清单 SHA256：`96b465b8c380b69b94d4e1ceec1aa75e2ae3c8d298844143adbcf8349691abe6`。本地角色表保持原始数值和北京试验街景计数，仅增加已核验服务器版本的引用；服务器角色表的完整街景计数为 3,778 个可用事件。

数据准备阶段完成，下一步为模型输入接入和小规模训练、保存、导出验证。历史缺失和未结束年度保留 masks；干预历史属性在正式实验前应完成来源复核，或限定使用已确认字段及其 masks。

## 临时缓存清理

用户授权后，服务器清理在北京时间 2026-10-13 19:58 完成：删除 `/tmp/mit_input_sync_20261013`、`/tmp/mit_streetview_grid_20261013` 和 `/home/nas/lsr/BaiduSvs_history/mit_grid_alignment_20261013` 三个本轮专用暂存目录。移除目录原占用 8,622,089,967 字节；先归档 74 份日志、逐城审计、原始选择证据等共 20,921,444 字节，净释放约 8.60 GB。

本地移除上传压缩包、卫星试采目录的 `images/records/sources` 载荷、单张诊断预览和 `src/scripts/tests` 的 Python 字节缓存，共 1,017 个文件、1,586,675,747 字节（约 1.59 GB）。试采比较报告、试采索引、审计明细、完整传输清单和回执保留；各试采目录记录了载荷已清理。正式年度卫星、基线复用目录、本地北京状态输入图片、历史 AlphaEarth 对照和 VIIRS 缓存保持不变。

清理前后核对正式角色表及街景主/事后清单哈希，结果不变。原始百度档案和原始角色元数据备份仍存在。当前 `transfer_receipt.json` 的 `source_manifest_path` 已改为持久的项目 `outputs/input_sync_20261013_v1/source_transfer_manifest.json`；旧暂存路径只作为历史来源记录保留。

服务器清理回执：`outputs/input_sync_20261013_v1/cache_cleanup_receipt.json`；清理前方案及保存证据：同目录 `cache_cleanup_plan.json`、`cache_cleanup_evidence/`。本地回执：[服务器清理摘要](../../outputs/input_server_sync/2026-10-13/server_cache_cleanup_receipt.json)、[本地清理摘要](../../outputs/input_server_sync/2026-10-13/local_cache_cleanup_receipt.json)。正式图片与数据没有下载到本地。

## 上传范围

| 目录 | 内容 |
|---|---|
| `data/active/alphaearth_grid_v2/` | 按正式网格重提取的完整 64 维背景嵌入和来源审计 |
| `data/active/satellite_grid_images_v1/` | 开通前一年基线缓存及源景、mask、预览 |
| `data/active/satellite_event_sequences_v2/` | 每个事件 −6 至 +3 年完整记录、实际影像及源景资料 |
| `data/active/intervention_grid_v1/` | 七字段干预输入、逐字段 masks、历史来源审阅状态 |
| `data/active/multimodal_inputs_v1/` | 状态、背景、干预原始表、城市背景代理及开通前六年影像表 |

包中还包括冻结处理清单的校验副本、既有卫星逐帧验收证据、AlphaEarth 空间审计和站点属性来源快照。共 269,738 个文件，1,872,585,974 字节；压缩包 1,573,733,940 字节。

压缩包 SHA256：`b7a1c9fe4106c106690aff2f64bd95e6de0d159e5b1f55665481b3e81166e7d3`。

文件清单 SHA256：`24a6f1e3eb15275ead0061fbebec5ccf32a6c67a1d547b982f337e27286e28ec`。

未重新上传已在服务器核验的 18,397 张街景，也未改写既有因果队列、处理清单、响应标签或原始街景档案。原有服务器数据继续使用已有版本；本轮新增五个输入目录。

## 传输与发布

所有认证信息只在会话内使用，不写入文件。SSH 连接使用此前核对的服务器公钥指纹。SFTP 支持中断续传，正式接收前核对压缩包和完整文件清单的 SHA256。

服务器先在 `/tmp/mit_input_sync_20261013/native_assets/` 解包并核验原始内容，再按城市和目录分散复制到项目内的 `outputs/input_sync_20261013_v1/incoming/`，复核每个目标文件的 SHA256。该方式解决 NAS 对顺序写入同一目录中大量小文件的延迟，保持普通文件结构，不引入压缩包路径或虚拟文件系统。

所有文件通过校验后，再将五个新目录发布到 `data/active/`。目标存在另一版本时拒绝覆盖。正式服务器输入表的更新仅改变运行时图片路径和街景绑定；原始角色元数据副本保存在项目内，源文件清单同样持久保存。最终绑定先按传输清单校验并保存原始元数据，重复执行时从已保存的原始版本读取，避免中断后的重试用部分更新版本覆盖原始证据。

## 服务器输入表调整

服务器角色表已引用实际服务器绝对路径，并接入街景全城正式选择表和缺失 masks。年度卫星完整索引保留原来的可移植相对路径和源字节，开通前六年状态表单独绑定为可直接读取的服务器路径。

POI、VIIRS、房价、人口、城市背景代理、AlphaEarth 向量和干预数值保持原样；不拟合预处理，不改变冻结事件身份，不加入响应监督。开通当年及之后的影像保存在观察档案，禁止作为处理前状态输入。

## 覆盖及仍保留的缺失

- 5,048 个事件、44 个城市全部保留。
- AlphaEarth：3,297 个合法背景向量；1,751 个历史产品年份缺失。
- 年度卫星：50,480 条完整记录，48,470 帧实际影像，48,465 帧可用；5 帧低覆盖、2,010 条未结束年度。
- 状态卫星：30,288 条开通前六年记录，其中 30,283 帧可用。
- 街景：开通前 7,543 张、3,778 个事件可用；事后 10,854 张、5,489 个事件年度组合可用。
- 干预：七字段及其 masks 已提供；历史属性继续标记 `requires_source_review`。

这些缺失保留原因和 masks，不等于记录遗漏或响应为零。新 S/T/C 模型接口、年度影像编码、正式训练和效果评价仍未由本次传输任务完成。

实现：[打包](../../scripts/analysis/package_input_assets_for_server.py)、[上传](../../scripts/collection/upload_input_asset_package.py)、[接收与校验](../../scripts/analysis/receive_server_input_assets.py)、[服务器输入绑定](../../scripts/analysis/integrate_server_input_assets.py)。内容校验、路径约束和中断重试相关测试 8 项通过，Ruff 通过。
