# 街景接入流程

街景是可选输入，通过数据集里的 `streetview_assets` 进入表示学习；训练前必须先
构建好索引与数据集。完整链路：

```text
百度街景原始数据
  -> scripts/collection/package_streetview.py        outputs/streetview/<city>/{metadata,images}
  -> scripts/collection/build_streetview_index.py    data/active/streetview/streetview_index.parquet
  -> build_pretraining_dataset.py --streetview-index  dataset 的 streetview_assets
  -> urban-train-representation --use-images          DINOv2 冻结编码器
```

## 1. 打包原始影像（如尚未打包）

```bash
python scripts/collection/package_streetview.py --city all
# 输出：outputs/streetview/<city>/metadata/*.csv 与 images/
```

## 2. 构建索引

元数据列：`province, city, county, wgs_lon, wgs_lat, panoid, capture_date,
status`（多余列忽略；编码自动尝试 utf-8-sig / gbk / utf-8）。仅保留
`status == 成功` 且经纬度、日期、panoid 有效的记录，按 panoid 去重。

```bash
python scripts/collection/build_streetview_index.py --city all
# 输出：data/active/streetview/streetview_index.parquet
#       data/active/streetview/streetview_index.audit.json
```

索引列：`city_key, grid_id, capture_date, asset_path, panoid`。网格归属用参考
网格的 500m 点阵精确匹配（纬度选行 + 行内经度二分），落在点阵外的点丢弃；
`asset_path` 为解析后的绝对路径，文件缺失的行计入 audit 的
`missing_image_rows` 并跳过。

本地烟测：`--city beijing --limit 20000`；只看审计不写库时可先小批量跑。

## 3. 构建带街景的预训练数据集

```bash
python scripts/causal_r/build_pretraining_dataset.py \
  --response-release data/active/causal/releases/production \
  --dataset-id viirs-poi-design-sv --families viirs,poi --weight-alpha 0 \
  --streetview-index data/active/streetview/streetview_index.parquet
```

`unit_features.parquet` 会多出 `streetview_assets`（JSON 路径列表）与
`streetview_available`。拍摄日期严格早于开通月的图片才会写入；同格多图按日期
升序保存，训练时按 `--max-images` 截断。覆盖率检查：

```bash
python - <<'PY'
import pandas as pd
u = pd.read_parquet("data/active/model_inputs/viirs-poi-design-sv/unit_features.parquet")
ok = u["streetview_assets"].ne("[]")
print("coverage:", int(ok.sum()), "/", len(u))
print(u.loc[ok, "streetview_assets"].head(2).tolist())
PY
```

## 4. 训练

```bash
conda run -n mit urban-train-representation data/active/model_inputs/viirs-poi-design-sv \
  --output outputs/representation/viirs-poi-design-sv \
  --use-images --max-images 4 --image-pooling max \
  --batch-size 64 --device cuda --epochs 100 --seeds 1 2 3
```

## 5. 注意事项

- 街景是**数据集属性**：旧数据集不能临时加 `--use-images`，必须用
  `--streetview-index` 重新构建数据集。
- `asset_path` 必须对训练机可读；索引只存路径，不复制图片。
- DINOv2（`dinov2_vits14`，冻结）首次使用需要 hub 权重；离线环境先缓存并设置
  `TORCH_HUB` 指向缓存目录。
- `--use-images` 明显增加显存与 IO；先以 `--batch-size 32` 跑一个 epoch 冒烟。
- 导出嵌入时必须带相同的 `--use-images`，工具会校验 checkpoint 的开关。
- 评估报告含**冻结 DINOv2 基线**，用于对照图像信息是否被响应监督有效利用。
