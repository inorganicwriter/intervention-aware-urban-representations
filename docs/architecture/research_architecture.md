# 项目核心研究架构

状态：有效  
更新日期：2026-10-12

## 1. 研究目标

项目研究不同地点对同类城市干预（地铁站正式开通）的响应相似性。
核心任务是构建 **Intervention-Conditioned Urban Representation**（干预条件化城市表征），
实现从处理前多模态感知数据预测微观空间单元（500m × 500m 网格）因果响应轨迹，并支持跨城市迁移。

## 2. 两层解耦研究系统

项目在方法论上严格解耦为两层系统：

1. **反事实因果估计层（已封版冻结）**：
   - 目标：为每个处理网格寻找或构造可信的未处理反事实，生成局部因果响应轨迹；
   - 状态：Phase 1 与 Phase 2 正式运行完毕，生产发布包 `data/active/causal/releases/production`（136,296 行）已严格封版冻结；
   - 计量边界：经过多族事件研究与平行趋势严密审计（`outputs/event_study_final/family_selection.md`），下游监督标签**严格限定为同城 Matching 与 GSC 估计的 VIIRS（6期）与 POI（12期：4维度×3年）**。房价与人口因平行趋势未过或数据缺失严重而排除，MC 与跨城估计排除。

2. **多模态干预表征学习层（当前主攻）**：
   - 目标：利用处理前高维感知输入（街景全景图、卫星连续嵌入场、表格先验、站点属性）学习 128 维潜在空间；
   - 跨视角协同：基于 **VIGOR (CVPR 2021)** 范式，地面 360° 全景图（DINOv2）与高空卫星连续场（Google AlphaEarth 64-d）进行交叉注意力融合；
   - 目标函数：以**软因果针刺损失（Soft Causal Anchoring，Huber 损失，~85%）**为主驱动力，弱对比对齐损失（~15%）为正则化；
   - 评估体系：贯穿 Level 1（轨迹重构与排序）、Level 2（规划语义分类：激活/位移/无响应）、Level 3（城市外 5-fold CV 泛化）到 Level 4（UMAP 流形与反事实可解释性）。

## 3. 核心数据流

```text
[不可变原始数据与来源记录]
        ↓
[空间资产基线: 500m 网格 / 5,048 处理网格 / 377 万 Donor / 1km 空间污染排除]
        ↓
========================= 第 1 层：因果反事实估计（已封版） =========================
只使用处理前信息的控制设计（同城 Matching: M=5 候选 + 静态精炼）
        ↓ 门禁未通过
Xu generalized synthetic control (GSC, 200 步 bootstrap)
        ↓
生产发布包: data/active/causal/releases/production (136,296 行, 冻结)
        ↓
严格筛选合法监督集:
  - 变量限定: VIIRS (6-d) & POI (12-d)
  - 方法限定: same-city Matching & GSC
  - 权重方案: W-design (设计可靠性评分)
        ↓
========================= 第 2 层：多模态表征学习（正在实施） =========================
多模态高维感知输入 (严守开通前时空防泄漏屏障):
  ├─ 地面视角: 360° 全景街景 (DINOv2 特征序列, capture_date < opening_month)
  ├─ 高空视角: Google AlphaEarth Foundations (64-d 连续场, lag 1)
  ├─ 规划先验: 27-d 处理前表格特征 (区位/网络/历史趋势)
  └─ 干预调制: 5-bit 站点属性离散 Token
        ↓
VIGOR 跨视角交叉注意力编码器 (Cross-View Visual Synergy)
        ↓
干预属性调制与特征拼接 (128 维空间)
        ↓
多任务因果学习目标:
  L_total = 0.85 * L_anchor (设计加权 Huber 轨迹拟合) + 0.15 * L_align (对比正则)
        ↓
四级评测与下游任务验证:
  Level 1 (连续轨迹检索) / Level 2 (规划语义分类) / Level 3 (城市外迁移) / Level 4 (流形可视化)
```

## 4. 模块职责与边界

- `data/archive/raw/`：不可变原始材料；
- `data/archive/staging/`：解析后、尚未发布的中间数据；
- `data/active/curated/`：标准化协变量、站点与空间产品；
- `data/active/causal/releases/production/`：**已冻结的因果响应生产发布包（只读）**；
- `data/active/streetview/`：街景全景图索引与元数据；
- `data/active/alphaearth/`：Google AlphaEarth 64-d 网格嵌入缓存；
- `outputs/`：审计报告、实验结果、训练 Checkpoints 与流形分析；
- `src/urban_intervention/causal/`：因果估计与发布逻辑（已验证）；
- `src/urban_intervention/representation/`：多模态编码器（`CrossViewVisualEncoder`）、数据集、软因果损失与评估套件；
- `scripts/collection/`：外部数据采集（`extract_alphaearth_embeddings.py`、`build_streetview_index.py`）；
- `tests/`：数据合同、信息边界与模型单元测试。

## 5. 固定研究约束

1. **时空单元与干预**：
   - 空间单元严格固定为 500m × 500m UTM 网格；
   - 处理事件为网格内正式地铁站开通；主空间污染排除半径为 1km。
2. **时序防泄漏屏障（Pre-treatment Temporal Barrier）**：
   - 街景全景图必须严格满足 `capture_date < opening_month`；
   - 卫星 AlphaEarth 嵌入必须严格采用开通前一年的年度合成（`lag 1`）；
   - 表格特征与控制选择仅读取开通前观测，绝对杜绝未来信息穿越。
3. **因果监督范围约束**：
   - 遵从 `outputs/event_study_final/family_selection.md` 计量审查结论；
   - 监督标签仅接纳 **VIIRS（6期）与 POI（12期）同城 Matching 和 GSC**；
   - 房价与人口因数据缺失或平行趋势检验未过，不得作为监督标签；
   - 矩阵补全（MC）与跨城匹配因统计效力不足与前趋势检验被拒，不得作为监督标签。
4. **反事实发布包只读性**：
   - `data/active/causal/releases/production` 为不可变事实基线，表征学习代码不得尝试覆写或修改该目录下的任何文件。

## 6. 权威文档导航

- 项目总览与权威层级：`docs/PROJECT_GUIDE.md`；
- 多模态表征学习详案：`docs/research/representation_learning.md`；
- 表征诊断与失效分析：`docs/research/representation_learning_audit.md`；
- 因果设计与响应标签：`docs/research/counterfactual_response_label_design.md`；
- 运行状态与待办清单：`docs/operations/current_project_status.md`；
- 街景全景图接入流程：`docs/operations/streetview_pipeline.md`；
- 数据资产总览：`docs/data/data_inventory.md`。
