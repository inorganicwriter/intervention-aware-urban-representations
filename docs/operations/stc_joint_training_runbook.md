# 共享 STC 联合训练运行约定

## 模型与数据

新模型为 `stc_joint_v1`：z128/t32/b96，预测输入256，两头6/12。有效AlphaEarth64保持原始数值；背景表格含mask与AlphaEarth可用性，编码32。缺失AlphaEarth使用可学习64维token。b不整体L2归一化。

主比较始终使用同一联合发布包。单族对照通过任务loss mask选择监督，不分别重建不同schema的数据包。样本资格采用两任务有效标签并集；各任务按自己的cell mask参与损失。训练/验证/测试城市互斥，标准化只由训练城市计算。

```bash
python scripts/causal_r/build_pretraining_dataset.py \
  --response-release data/active/causal/releases/production \
  --dataset-id stc-joint-mg-design-v1 --stc \
  --stc-inputs-dir data/active/multimodal_inputs_v1 \
  --families viirs,poi --methods matched,gsc --scope-view same_city --weight-alpha 0
```

## 单次正式训练

```bash
python -m urban_intervention.representation.cli \
  data/active/model_inputs/stc-joint-mg-design-v1 \
  --output outputs/stc_joint/formal_joint_seed42 \
  --training-mode joint --embedding-dim 128 --use-images --use-satellite \
  --epochs 100 --batch-size 32 --seed 42 \
  --max-grad-norm 1.0 --gradient-diagnostics-interval 1
```

更换单族监督只修改 `--training-mode viirs_only` / `poi_only` 和独立输出目录。新模式固定维度和顶层固定任务权重；历史STC单族包和旧检查点继续走历史路径。`--max-grad-norm 0`关闭裁剪，诊断频率0关闭任务诊断。正向影像缺失保留mask，不能用require-multimodal筛选替代缺失处理后再与全样本模型比较。

DINOv2加载优先使用Torch Hub本地缓存`facebookresearch_dinov2_main`，适合服务器离线运行；可通过`TORCH_HOME`指定缓存根目录（其中源码和checkpoint分别位于`hub/`及`hub/checkpoints/`）。

## 三组主比较与科学消融

```bash
python -m urban_intervention.representation.joint_experiments \
  data/active/model_inputs/stc-joint-mg-design-v1 \
  --output outputs/stc_joint/formal_comparison_seed42 \
  --experiments viirs_only poi_only joint \
  --epochs 100 --batch-size 32 --seed 42 --max-grad-norm 1.0

python -m urban_intervention.representation.joint_experiments \
  data/active/model_inputs/stc-joint-mg-design-v1 \
  --output outputs/stc_joint/formal_ablations_seed42 \
  --experiments anchor_only tc_only z_only no_streetview no_satellite tabular_only \
  --epochs 100 --batch-size 32 --seed 42
```

默认拒绝非strict-production输入，防止fixture分数误作正式结果。每组保存独立checkpoint、配置、loss历史、梯度诊断、测试指标、报告、runs记录及128维表示/双头预测导出。比较器核验各族测试cell支持指纹、城市划分与输入bundle哈希；未训练的头标记trained=false，其误差不能当作单任务模型的另一任务能力。

共享表示的聚类与UMAP/PCA坐标分析只读取128个emb列，不读取fused列：

```bash
python -m urban_intervention.representation.embedding_analysis \
  outputs/stc_joint/formal_comparison_seed42/joint/embeddings.parquet \
  --output outputs/stc_joint/formal_comparison_seed42/joint/embedding_analysis.json
```

未安装可选umap-learn时，输出明确标记pca_fallback；不能称为已运行UMAP。`--visualize`也可在训练报告中生成仅基于z的流形坐标。

`comparison_status.json`记录pending/running/completed/failed及错误。输出目录不可覆盖。测试仅最终评估，验证总损失选模；均值基线只使用训练城市有效标签。按族报告RMSE/MAE、训练均值MSE提升、近邻响应相关、检索相对随机邻居指标、按未见城市的预测/检索与训练城市拟合的线性probe。逐族任务指标与置换检验使用相同布局。

## 工程 smoke（不得作为科学性能证据）

```bash
python scripts/representation/smoke_joint_training.py \
  --output outputs/stc_joint/smoke_new_run --events 24 --epochs 2 \
  --experiments viirs_only poi_only joint anchor_only tc_only z_only no_streetview no_satellite tabular_only
```

该命令构造明确标记为non-production的合成标签、街景和六波段TIFF，读取已有本地DINOv2代码/预训练权重并冻结，不发起下载。合成数据固定seed。证据文件保存backbone权重SHA256。旧模型和正式资产均不覆盖。

## 验收核查

- 两族均有效时Anchor/Alignment分别各0.5；独立归一化，单有效任务为1，全无监督batch连AdamW更新及BatchNorm前向也跳过。
- 梯度诊断来自两族对实际总loss的贡献；不写.grad。记录共享State范数、两族余弦及冲突、联合裁剪前后范数、AlphaEarth/表格尺度与梯度。
- 导出应有128个emb列、6+12个预测列、身份、split、质量和每族mask；metadata记录架构/维度/数据哈希。
- 数据/代码版本与历史检查点分开保存。功能smoke通过不等于联合模型优于单族模型，正式数据对照和多seed实验须单独执行并报告。
