# 街景全景图与跨视角视觉接入流程

2026-10-13 更新：独立卫星每事件 −6 至 +3 年时序已采集并验收。44 城街景已按同一正式网格重建，3,778 个事件选出 7,543 张合法开通前图片。按用户最新要求，完整街景只在服务器项目中发布。下一轮输入角色以 `docs/research/state_context_input_contract_2026-10-12.md` 为准：街景和独立卫星进入状态，完整 AlphaEarth64 仅进入背景。下文 VIGOR 及其 AlphaEarth 融合命令属于旧实现对照，不是新 S/T/C 模型的训练入口。

同日最新 S/T/C、AlphaEarth 和年度卫星也已完成服务器入库与完整街景绑定，数据准备阶段完成。正式输入绑定 37 项及路径验证 17 项全部通过；开通后街景单独归档，禁止作为处理前状态。重复重建暂存与传输缓存已清理，正式图像、原始百度档案、来源清单和审计证据仍保留。当前阶段及回执见 [输入同步记录](input_server_sync_2026-10-13.md)，下一步为模型接入与小规模训练验收。

## 当前实际位置与网格重建

原始采集代码位于 `/home/nas/lsr/BaiduSvs_history/BaiduSV_National_Collector`，原始资产位于 `/home/nas/lsr/BaiduSvs_history/output`。认证信息不写入代码、数据清单或文档。

原始成功元数据：`metadata/success/{province}/{city}/{county}.csv`。图片使用原始精确路径 `images/{province}/{city}/{county}/{panoid[-2]}/{panoid[-1]}/{panoid}.jpg`。下载阶段全局按 panoid 去重，所以同一全景可能保存在邻县目录，索引在城市内部检查精确文件名；不使用前缀模糊匹配。

需要纠正的两个坐标问题：

- `wgs_lon/wgs_lat` 是查询点，实际全景拍摄位置来自 `ret_bd_x/ret_bd_y`。
- 原始返回值相对于 BD09MC 的单位放大了 100 倍，已有 `step5_post_process_coords.py` 没有消除该倍数，`metadata/converted/` 中因此出现越界经纬度。重建使用原始成功元数据，校验两个轴的单位后重新转换，不复用越界结果或用查询点填补拍摄位置。

`streetview_coordinates.py` 完成 BD09MC → BD09 → GCJ02 → WGS84，保留 query 坐标、原始返回值与转换倍数。其 WGS84 结果是坐标转换估计，未经过独立 GPS 地面校准。入格使用冻结正式多边形；共享边界无法唯一归属的全景不绑定。冻结事件及响应标签保持原状。

采集器会将 `YYYYMM` 补为当月 1 日，也会保留 `YYYYMMDD`，但成功 CSV 没有日期精度字段，不能把每个 1 日都视为真实拍摄日。本轮统一采用月精度门禁：状态要求 `capture_month < opening_month`，排除开通当月和之后的图像。原始日期保留在带校验和的源 CSV 中。全量服务器索引可包含事后档案，模型只能读取单独的开通前选择表。

当前发布入口：

```text
frozen_grid_contract.json（全部 5,048 事件及对应原始网格 WKT/哈希）
  -> rebuild_remote_streetview_index.py（服务器，44 城市；原始档案不改动）
  -> streetview_index.csv / streetview_pre_selected.csv / audit.json
  -> publish_server_streetview_inputs.py（服务器内复制、逐张解码、校验和复核）
  -> /home/nas/lsr/MIT_summer/data/active/streetview_grid_v2/
```

事后街景使用 `rebuild_remote_streetview_post_archive.py`，从同一全量元数据候选按事件年 0 至 +3 逐年选择，再由 `publish_server_streetview_post_archive.py` 复制和核验。每个事件每年最多两张；覆盖表保留全部 20,192 条记录。门禁为 `capture_month > opening_month`，开通当月只记不确定候选，事后档案统一 `state_allowed=false`。正式输出为同目录的 `streetview_post_assets.parquet`、`streetview_post_event_coverage.parquet` 和 `post_manifest.json`。年度卫星已有 −6 至 +3 年完整记录，开通年合成属于混合观测档案，开通前状态仅允许 −6 至 −1 年。

正式资产目录为服务器的 `/home/nas/lsr/MIT_summer/data/active/streetview_grid_v2/`。全量元数据候选索引与已复制的精选图片索引分别保存，`streetview_state_assets.parquet` 按 `treatment_order` 绑定真实开通前图片。`event_coverage.parquet` 保留全部 5,048 个事件、可用性 mask、缺失原因和图片路径，供服务器输入表连接。当前最多两张是状态缓存的选择配置，不表示原始档案只存在两期历史数据，也不代表年度街景完整。旧 SFTP 下载工具和 Windows 北京试验缓存只保留为先前局部验收材料；后续完整资产不下载到客户端。

NAS 的大量零散文件查询存在明显延迟，正式重建采用 `--metadata-only-index`：先发布坐标、日期与网格已核对的历史元数据候选，再逐张解码实际选用的开通前图片。候选档案表中的图片状态为 `metadata_candidate_image_unverified`，文件大小保留缺失；不能把候选行数当成已确认图片数。训练选择遇到缺文件或损坏图片时继续尝试较早的合法图片，直到选足最多两张或没有剩余候选。训练选择表、服务器项目内图片和缺失 mask 单独验收。

最终数量与验收以 [街景服务器重建报告](streetview_server_alignment_2026-10-13.md) 为准。旧 `build_streetview_index.py` 适用于已经正确平铺、且输入坐标确为拍摄位置的资产；不能直接对本次原始查询坐标运行后称作新的正式索引。

街景数据在本项目中升级为**地面 360° 全景图（Panoramas）**，作为高维感知输入的核心组成部分。
街景全景图与高空卫星连续场（Google AlphaEarth 64-d）通过 **VIGOR (CVPR 2021) 跨视角协同架构**
融合，克服低维表格特征的“维度倒挂”瓶颈。

完整数据与训练链路：

```text
服务器本地 360° 街景全景图归档
  -> scripts/collection/build_streetview_index.py        data/active/streetview/streetview_index.parquet
                                                         (严格执行 capture_date < opening_month 防泄漏)
  -> scripts/causal_r/build_pretraining_dataset.py       unit_features.parquet (绑定全景图路径与时间戳)
  -> scripts/collection/extract_alphaearth_embeddings.py data/active/alphaearth_grid_v2/{city}_alphaearth.parquet
                                                         (GEE 提取开通前 lag 1 年度的 64-d 连续场)
  -> urban-train-representation --use-multimodal         VIGOR Cross-View Synergy + 软因果针刺损失
```

## 1. 服务器全景图资产与元数据

服务器已归档 44 个城市的 360° 全景影像与元数据 CSV：
- 元数据核心字段：`panoid, province, city, county, wgs_lon, wgs_lat, capture_date, status`
- 仅保留 `status == 成功` 且坐标、拍摄日期有效之记录；
- 图像为等距柱状投影（Equirectangular Projection）的 360° 全景图文件。

## 2. 构建全景图空间与时序索引

```bash
python scripts/collection/build_streetview_index.py --city all
# 输出：data/active/streetview/streetview_index.parquet
#       data/active/streetview/streetview_index.audit.json
```

索引构建核心规则：
1. **500m 网格空间绑定**：使用冻结参考网格的 `geometry_wkt` 多边形，将全景图拍摄位置映射到唯一所属 `grid_id`。共享边界的多格歧义记录暂不绑定；不再以经纬度 bbox 分带代替实际投影网格；
2. **时序防泄漏门禁（Pre-treatment Temporal Barrier）**：若全景图归属于处理网格，其拍摄时间必须满足：
   $$\text{capture\_date} < \text{opening\_month}$$
   开通之后拍摄的全景图坚决丢弃，防止反事实因果效应被未来视觉信号污染；
3. **同网格多图排序**：同一网格内若存在多期历史全景图，按拍摄日期升序排序，保留最接近开通时点但严格早于开通期的影像（训练时按 `--max-images` 截取）。

## 3. 全景图编码与 VIGOR 跨视角协同

在模型前向传播中：
1. **全景图特征提取**：
   - 方案 A（全图编码）：360° 全景图通过冻结的 DINOv2 (`dinov2_vits14`) 提取 Patch-level Token 序列；
   - 方案 B（多视角切片）：等距柱状全景图投影切片为 4 个主要罗盘朝向（东、南、西、北）透视图像，分别输入 DINOv2 提取特征后拼接为地面视角 Token 序列：
     $$Z_{\text{pano}} \in \mathbb{R}^{K \times d_v}$$
2. **高空卫星场关联**：
   - 读取 Google AlphaEarth 64-d 密集嵌入场 $H_{\text{sat}} \in \mathbb{R}^{S \times d_s}$；
3. **VIGOR 跨视角交叉注意力（Cross-View Synergy）**：
   - 以地面视角 Token 为 Query，高空卫星网格特征为 Key 和 Value 进行交互：
     $$Z_{\text{cross}} = \text{Softmax}\left(\frac{Q_{\text{pano}} K_{\text{sat}}^T}{\sqrt{d}}\right) V_{\text{sat}}$$
   - 辅以表格先验（27-d）与站点属性 Token（5-bit）进行特征调制，投射至 128 维潜在空间。

## 4. 构建多模态预训练数据集

针对严格限定的合法监督集（VIIRS 6-d 与 POI 12-d，同城 Matching + GSC）：

```bash
python scripts/causal_r/build_pretraining_dataset.py \
  --response-release data/active/causal/releases/production \
  --dataset-id viirs-mg-design-multimodal --families viirs --methods matched,gsc --weight-alpha 0 \
  --streetview-index data/active/streetview/streetview_index.parquet \
  --satellite-source data/active/alphaearth_grid_v2

python scripts/causal_r/build_pretraining_dataset.py \
  --response-release data/active/causal/releases/production \
  --dataset-id poi-mg-design-multimodal --families poi --methods matched,gsc --weight-alpha 0 \
  --streetview-index data/active/streetview/streetview_index.parquet \
  --satellite-source data/active/alphaearth_grid_v2
```

## 5. 训练执行

```bash
conda run -n mit urban-train-representation data/active/model_inputs/viirs-mg-design-multimodal \
  --output outputs/representation/viirs-mg-multimodal \
  --use-multimodal --max-images 2 \
  --loss-anchor-weight 0.85 --loss-align-weight 0.15 \
  --batch-size 32 --device cuda --epochs 100 --seeds 1 2 3
```

## 6. 注意事项与保障

- **显存与 IO 优化**：全景图尺寸较大，可对全景图进行下采样或离线预提取 DINOv2 序列特征缓存至 `.h5` / `.pt`；
- **缺失稳健性**：部分偏远网格若缺失全景图，模型支持 `visual_mask`，退化为卫星 AlphaEarth 连续场 + 表格先验分支，确保训练不崩溃；
- **评估基线对照**：评测套件自动对照四类基线：Raw Tabular Ridge、Frozen DINOv2 Baseline、Satellite-only Baseline 以及 Multimodal Synergistic Model。
