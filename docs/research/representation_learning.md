# 干预条件化城市表示学习（Representation Learning）

更新日期：2026-10-12

状态：多模态空天地协同架构、严格前置时间边界、软因果锚定与四级评估规范（已贯通主训练与端到端验证）

> **当前输入与表示口径（2026-10-19）**：S 包括 POI/VIIRS 历史、全部合法开通前街景和独立卫星影像；T 包括开通时间、站点类型和规模字段；C 包括完整 AlphaEarth 64 维、人口与城市背景（房价字段已移除）；不输入路网。参考《研究思路.docx》的 `z_i=f_theta(x_i)`、`tau_hat_i=g_phi(z_i,T_i,C_i)`，当前项目将 `x_i` 明确为开通前状态 S：S 的多模态输入共同学习唯一状态 embedding `z`；T 与 C 作为预测条件，不并入 z。因果响应标签监督响应预测，响应相似性监督对齐 z。下文包含较早的架构与结果记录，现行字段和实现边界以 [状态、干预与背景输入契约](state_context_input_contract_2026-10-12.md) 和 [`stc_code_readiness.md`](../operations/stc_code_readiness.md) 为准。

本文档描述表示学习模块的研究设计、数据契约、模型架构、评估协议与操作指南，是 [counterfactual_response_label_design.md](counterfactual_response_label_design.md) 的下游核心消费层。

---

## 1. 研究问题与核心定位

表示学习的目标是刻画不同空间单元对同类重大城市干预（地铁站正式开通）的响应异质性与相似性。
表示学习模块把这一科学任务形式化为：

> 学习一个映射 $f: X \to \mathbb{R}^d$（$X$ 为**开通前严格隔离的多模态城市感知与统计特征**），使得潜嵌入空间中的几何近邻关系 $\text{sim}_\text{emb}(f(x_i), f(x_j))$ 与准实验因果响应相似性 $\text{sim}_\text{resp}(r_i, r_j)$ 一致，并且该一致性能够**稳定迁移到训练中未见过的城市**，从而支撑对未来规划站点因果响应轨迹的外推预测。

### 1.1 因果响应监督设定：带噪因果锚点（Noisy Causal Anchors）
- **客观设定**：微观网格的因果响应标签 $r_i = Y_{i, \text{post}}^{\text{obs}} - \hat{Y}_{i, \text{post}}^{\text{counterfactual}}$ 混合了真实政策拉动、微观经济波动、TOD提前开发以及反事实估计误差。研究不设定“先消除全部估计噪声或获得不可观测的绝对因果真值”作为启动前置，而是将准实验标签作为**带有可靠性权重的带噪因果锚点**（Noisy Causal Anchors）。
- **监督范围收敛**：
  - 生产版本因果标签（`data/active/causal/releases/production`，共 136,296 行）提供经验基础；
  - 正式表征监督重点使用通过了平行趋势检验和可信度诊断的子集：**VIIRS（6期月度动态曲线）与 POI（12维年度商业重塑矩阵）的同城（same-city）Matching 与 GSC 标签**；
  - 排除交易稀疏的 Housing（同城聚类边际）与宏观插值的人口 Population，排除跨城 MC 外推。详细按 outcome 诊断见第 6 节。

---

## 2. 多模态模型与架构

针对纯表格表示学习中存在的“27维统计摘要无流形压缩空间（伪降维）”问题，现行架构升级为**以高维感知特征为主体压缩、以低维时序表格为宏观先验**的层次化多模态网络。

### 2.1 输入模态与严格时间边界（Pre-treatment Boundary）
所有进入模型主预测实验的特征必须严格先于干预发生时点（$t < T_{\text{open}}$），杜绝同期或事后信息泄露：

1. **天基卫星特征（Overhead Satellite Field, 64维）**：
   - 数据源：Google DeepMind **AlphaEarth Foundations**（GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`），包含综合物理地表特征（光学、SAR雷达与高程嵌入）。
   - **时间边界核实与掩码机制**：
     - AlphaEarth 全球产品自 2017 年起提供年度汇总。在处理网格总数 5,048 个事件（44 城）中：
       - **3,297 个事件（65.31%）** 的 `target_year < opening_year`（开通年份 $\ge 2018$），属于严格事前影像（Valid Pre-treatment）；
       - **1,751 个事件（34.69%）** 开通年份 $\le 2017$，提取器回退至 2017 年影像，属于时间代理（Temporal Proxy，包含同期或事后信息）。
     - **执行策略**：在主预测实验中，默认启用严格时间隔离（`strict_pretreatment=True` / `allow_satellite_proxy=False`）。**不剔除早期事件**，而是将其卫星分支掩码屏蔽（`satellite_available = False`，向量置零），保留其合法的表格及历史街景输入继续参与多模态训练；事后代理实验仅作为明确标注的回顾性敏感性对照（`--allow-satellite-proxy`）。
2. **地基人本全景街景（Ground Street-View Panoramas）**：
   - 百度街景 $360^\circ$ 全景图，严格按拍摄日期过滤 `capture_date < opening_month`；
   - 经冻结的 **DINOv2**（`dinov2_vits14`）抽取微观建筑立面与商业界面特征。
3. **全局时序表格先验（Tabular Priors, 27维）**：
   - 开通前 1~3 年的标准化指标（房价、POI计数/商业占比/交通可达/类别熵、夜光、人口、NDVI/NDBI），仅在训练集城市上拟合标准化参数（`normalization.json`）。卫星波段 `A00`..`A63` 严格与表格输入隔离。
4. **站点干预条件化 Token（$T_i, C_i$）**：
   - `opening_year` 分桶嵌入（30 桶）与站点类型 5-bit 编码（换乘/新线/延伸/终点/缺失，32 类）。

### 2.2 跨视角协同模块（Phase 1：Pooled-Feature Multimodal Fusion）
第一阶段完整实现采用**池化多模态融合模型**：
- **天基表征**：网格内 10 米像素经平均池化为 64 维向量，送入 `SatelliteAlphaEarthEncoder` 映射至 128 维空间并执行 L2 正则化；
- **地基表征**：全景图经 DINOv2 抽取并经均值/最大池化整合为 128 维人本特征；
- **跨视角交互**：`CrossViewSynergyModule` 在两个 128 维视角间执行多头注意力交互（协同头数 $H=4$），支持任一模态缺失时的注意力掩码优雅退化；
- **表格先验与多任务头**：`TabularEncoder`（27 维 $\to$ 128 维）与干预条件 Token 通过残差及门控注入，最终通过 `MultiHeadPredictor` 解耦预测 VIIRS（6维）、POI（12维）、Housing（6维）及 Population（3维）的多任务因果响应。

> **架构演进说明**：当前池化多模态融合作为第一阶段完整模型，代码与描述完全一致。密集空间 Token 场（$H \times W \times 64$）与透视全景切片（Perspective Slices）作为后续增强项，不在第一阶段作为阻碍主实验的前置条件。

### 2.3 损失函数：Soft Causal Anchoring 正式主干
针对此前对比损失（InfoNCE）与距离均方误差之间的几何张力，主训练目标正式切换为**因果响应预测为主、弱对比为辅**的 Soft Causal Anchoring 损失（`multimodal_causal`）：

$$L_{\text{multimodal}} = w_{\text{anchor}} \cdot L_{\text{anchor}} + w_{\text{align}} \cdot L_{\text{align}}$$

其中默认配置：$w_{\text{anchor}} = 0.85$，$w_{\text{align}} = 0.15$。

1. **主损失：置信度加权 Huber 锚定损失（Soft Causal Anchoring）**：
   $$L_{\text{anchor}} = \frac{1}{\sum_{(i,f)} m_{i,f} w_{i,f}} \sum_{i} \sum_{f \in \mathcal{F}} m_{i,f} w_{i,f} \cdot \operatorname{Huber}\left(\hat{\tau}_{i,f} - \tau_{i,f}^{\text{QE}}, \delta=1.0\right)$$
   - Huber 阈值 $\delta=1.0$ 在小误差下等价于 L2 损失、在大残差下转为线性 L1 损失，有效抑制极值离群点；
   - 权重 $w_{i,f}$ 结合了 GSC 预处理 RMSPE 与 Matching 标准误的精度收缩估计（Precision Weights）。
2. **辅助损失：弱对比对齐（$L_{\text{align}}$）**：
   - 软拉近因果响应趋势相似的单元对，雕刻跨城嵌入流形。
3. **对照基线**：旧版的 `total_loss`（MSE + InfoNCE，`--loss-type contrastive_legacy`）作为明确标注的对照实验保留。

---

## 3. 四级评估体系（Level 1 ~ Level 4）

评估体系严格遵循多层级因果外推验证标准，涵盖连续轨迹预测、规划语义分类及跨城市外推迁移：

| 验证等级 | 核心指标与内容 | 回答的科学问题 |
|---|---|---|
| **Level 1: 连续因果轨迹预测**<br>(Trajectory Prediction) | - 未来 1~24 个月 VIIRS/POI 动态响应的 `mean_mse` 与 `pearson_r`<br>- `directional_accuracy`（预测变化方向的正负符号准确率）<br>- 公平基线比较：训练均值基线、折内 Raw Ridge、单模态模型 | 嵌入模型能否准确预测未见站点开通后的动态响应曲线？ |
| **Level 2: 规划语义响应分类**<br>(Planning Semantic Classification) | - **正向响应（Positive / Activation）**：平均响应 $> +0.05$<br>- **弱或不确定响应（Neutral or Uncertain / No Response）**：$|\text{平均响应}| \le 0.05$<br>- **负向响应（Negative / Displacement）**：平均响应 $< -0.05$<br>- 评估指标：`accuracy`, `macro_f1`, `cohens_kappa`, `class_f1` | 模型能否直接指导规划实践，识别潜力激活区与钝化/分流区？ |
| **Level 3: 严格跨城外推迁移**<br>(Cross-city Transfer) | - **44 城市隔离 5 折交叉验证**（测试城市从未参与训练）<br>- `transfer.per_city`：未见城市的近邻因果相关性 `nn_corr@k`<br>- `transfer.few_shot_probe`：给定目标城 4/8/16/32 个已知站点后的线性探针适应曲线 | 在已有地铁城市训练的表征，能否直接迁移到新规划地铁城市？ |
| **Level 4: 表征流形与可解释性**<br>(Manifold Interpretation) | - 全量网格 $z_i$ 的 UMAP / SVD-PCA 全局流形降维<br>- 检验不同城市是否存在共同的响应流形（Response Manifold） | 空间表征是否学到了本质的政策响应结构，而非仅记忆地理位置？ |

---

## 4. 实验比较阶梯（公平消融对照）

为确认多模态表征是否带来真实增益，实验严格保持样本、城市划分、时间截止和评估掩码一致，按以下阶梯逐级对比：
1. **训练均值基线（Training Mean Baseline）**：纯机会水平；
2. **表格 Ridge 回归（Raw Tabular Ridge）**：仅使用 27 维开通前表格特征的线性回归；
3. **表格 + 卫星简单模型（Tabular + Satellite Early Fusion）**：加入 AlphaEarth 64 维嵌入；
4. **表格 + 卫星 + 街景多模态融合模型（Multimodal Synergy）**：完整池化协同网络；
5. **复杂表示学习与对比微调**：检验 Soft Causal Anchoring 带来的额外因果对齐增益。

---

## 5. 核心代码模块与运行状态

| 模块文件 | 角色与职责 | 当前状态 |
|---|---|---|
| `src/urban_intervention/causal/pretraining_dataset.py` | AlphaEarth 卫星时间边界校验、代理标记审计与数据集发布 | **完成** (15/15 单元测试通过) |
| `src/urban_intervention/representation/dataset.py` | 多模态输入绑定、时间代理掩码与特征隔离 | **完成** (14/14 单元测试通过) |
| `src/urban_intervention/representation/loss.py` | Soft Causal Anchoring Huber 损失与多模态因果损失 | **完成** (单元测试通过) |
| `src/urban_intervention/representation/model.py` | 卫星编码器、街景编码器、跨视角协同与预测头 | **完成** (9/9 单元测试通过) |
| `src/urban_intervention/representation/trainer.py` & `training/*.py` | 主训练器、Loss 分支调度、Checkpoint 配置保存与评估集成 | **完成** (17/17 对齐测试通过) |
| `src/urban_intervention/representation/evaluation.py` | Level 1 轨迹预测、Level 2 语义分类与 Level 4 UMAP 流形 | **完成** (端到端验证通过) |
| `tests/integration/test_multimodal_causal_e2e.py` | 端到端完整小规模集成流水线（加载→训练→保存→重载→评估） | **完成** (3/3 集成测试通过) |

---

## 6. POI 诊断矩阵与监督适用范围（Method $\times$ Outcome）

基于 `outputs/event_study_final/pretrend_summary.csv` 记录的实证诊断检验结果，对所有 outcome 进行严格梳理：

### 6.1 同城（Same-City）实证检验矩阵

| 指标族 | 分析方法 | 具体 Outcome | 检验类型 | 聚类维度 | 统计量 / $t$ 值 | $p$ 值 | 5% 水平诊断 | 监督适用范围定位 |
|---|---|---|---|---|---|---|---|---|
| **viirs** | Matching | `viirs_avg_asinh` | 联合 Wald | City | $F = 2.288$ | **0.0631** | 未拒绝（10%拒绝） | 主分析核心指标 |
| **viirs** | GSC | `viirs_avg_asinh` | 平均前置 Gap=0 | City | $t = -1.102$ | **0.2775** | **未拒绝** | **主分析最高信度指标** |
| **poi** | Matching | `poi_count_log` | 联合 Wald | City | $F = 2.201$ | **0.0717** | 未拒绝（10%拒绝） | 主分析核心指标 |
| **poi** | GSC | `poi_count_log` | 平均前置 Gap=0 | City | $t = -1.259$ | **0.2165** | **未拒绝** | **主分析最高信度指标** |
| **poi** | GSC | `poi_category_entropy` | 平均前置 Gap=0 | City | $t = -2.049$ | **0.0482** | 显著拒绝 | 次要/多任务敏感性分析 |
| **poi** | GSC | `poi_commercial_share` | 平均前置 Gap=0 | City | $t = +3.654$ | **0.00086** | 显著拒绝 | 次要/多任务敏感性分析 |
| **poi** | GSC | `poi_transport_access_log` | 平均前置 Gap=0 | City | $t = +3.752$ | **0.00065** | 显著拒绝 | 次要/多任务敏感性分析 |
| **poi** | MC | `poi_category_entropy` | 平均前置 Gap=0 | City | $t = -2.777$ | 0.0095 | 显著拒绝 | 仅供探索对照 |
| **poi** | MC | `poi_commercial_share` | 平均前置 Gap=0 | City | $t = -1.728$ | 0.0947 | 未拒绝 | 仅供探索对照 |
| **poi** | MC | `poi_count_log` | 平均前置 Gap=0 | City | $t = -4.711$ | $5.67 \times 10^{-5}$ | 显著拒绝 | 仅供探索对照 |
| **poi** | MC | `poi_transport_access_log` | 平均前置 Gap=0 | City | $t = -0.810$ | 0.4247 | 未拒绝 | 仅供探索对照 |

### 6.2 监督适用范围与方法学原则
1. **主结果指标**：
   - 同城 GSC 下的 `viirs_avg_asinh`（$p = 0.278$）与 `poi_count_log`（$p = 0.217$）在 5% 水平下完全接受“处理前平均 Gap 为 0”的原假设，无处理前趋势不平衡，作为 Level 1 连续预测与 Level 3 跨城迁移的**核心主结果**。
2. **次要多任务与敏感性分析指标**：
   - `poi_category_entropy`、`poi_commercial_share` 与 `poi_transport_access_log` 在同城 GSC 呈现显著的处理前偏差。
3. **方法学铁律（Methodological Integrity）**：
   - **严禁仅凭 p 值自动删除标签**，也**严禁为了追求显著结果修改因果估计**；
   - 保持 12 维 POI 向量的多任务结构，在损失计算中赋予置信度收缩权重（$w_{i,f}$）；
   - 在汇报中明确区分主分析与敏感性分析，完整保留筛选与对照记录。

---

## 7. 服务器验证清单与生产数据指纹

当前本地已完成代码实现、单元测试与端到端小规模闭环。以下事项涉及仅存在于远程服务器（4×RTX 4090）的资产，需在服务器运行时独立核查：

1. **轨道快照与生产标签溯源核查**：
   - 核查远程 `data/active/causal/releases/production`（136,296 行）构建时的 Git Commit 哈希与输入指纹；
   - 确认 2026-10-07 上游控制设计（历史轨道邻接缺少有效边日期）的修正是否影响该发布；如需更新，一律构建独立新目录（例如 `data/active/causal/releases/production_v2_audited`），**严禁覆盖已有生产标签**。
2. **街景全景图合格样本覆盖核查**：
   - 核查远程服务器 `data/active/streetview/streetview_index.parquet` 中实际覆盖的网格数；
   - 检验满足 `capture_date < opening_month` 的街景网格与 3,297 个时间合法 AlphaEarth 卫星事件的实际交集规模。
3. **GPU 吞吐与全量训练验证**：
   - 在服务器 4×4090 上执行全量多模态训练，验证 DINOv2 特征缓存读取与 CrossViewSynergy 模块在真实批次下的显存与吞吐表现。

---

## 8. 操作与复现命令

### 8.1 本地端到端小规模验证（已通过）
无需远程服务器大文件，在本地直接运行端到端测试，验证“数据加载 $\to$ 时间掩码 $\to$ 软因果锚定训练 $\to$ Checkpoint保存 $\to$ 模型重载 $\to$ Level 1/2 评估报告”完整链路：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
pytest tests/integration/test_multimodal_causal_e2e.py -v
```

### 8.2 服务器全量生产环境训练命令（4×RTX 4090）

#### 步骤 1：构建严格前置预训练数据集（屏蔽事后卫星代理）
```bash
python scripts/causal_r/build_pretraining_dataset.py \
  --response-release data/active/causal/releases/production \
  --dataset-id viirs_poi_multimodal_strict \
  --families viirs,poi \
  --methods matched,gsc \
  --alphaearth-embeddings data/active/alphaearth/alphaearth_treated_embeddings.parquet \
  --weight-alpha 0.5
```
*(注：默认 `--allow-satellite-proxy` 为 False，自动对开通年份 $\le 2017$ 的 1,751 个事件施加卫星掩码，保留合法表格与街景)*

#### 步骤 2：启动多模态 Soft Causal Anchoring 正式训练
```bash
python src/urban_intervention/representation/cli.py \
  data/active/model_inputs/viirs_poi_multimodal_strict \
  --output outputs/representation/multimodal_causal_v1 \
  --epochs 100 \
  --batch-size 32 \
  --embedding-dim 128 \
  --use-satellite \
  --use-cross-view-synergy \
  --loss-type multimodal_causal \
  --loss-anchor-weight 0.85 \
  --loss-align-weight 0.15 \
  --loss-huber-delta 1.0 \
  --seeds 1 2 3
```

#### 步骤 3：启动旧版 InfoNCE 对照基线（用于消融对比 Table 1）
```bash
python src/urban_intervention/representation/cli.py \
  data/active/model_inputs/viirs_poi_multimodal_strict \
  --output outputs/representation/baseline_contrastive_legacy \
  --epochs 100 \
  --batch-size 32 \
  --embedding-dim 128 \
  --loss-type contrastive_legacy \
  --seeds 1 2 3
```

#### 步骤 4：生成评估对比表与模型卡
```bash
python -m urban_intervention.representation.summary \
  outputs/representation/multimodal_causal_v1/seed_* \
  outputs/representation/baseline_contrastive_legacy/seed_* \
  --output outputs/representation/comparison_table.csv

python -m urban_intervention.representation.model_card \
  outputs/representation/multimodal_causal_v1/seed_1
```
