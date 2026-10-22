# 响应敏感城市表示学习：文献证据与模型设计建议

调研日期：2026-10-20（北京时间）。

状态：研究建议，尚未替换现行 S/T/C 输入契约，尚未实施模型修改或开展本次建议对应的性能实验。

## 1. 建议先确定的研究目标

本项目学习一个从严格开通前观测生成、能够迁移到未见城市的空间表示，并用准实验估计的政策响应来塑造这个表示。开通后的真实影像不作为预测输入或视觉重建目标。

依据《研究思路.docx》的 `z_i=f_theta(x_i)` 与 `tau_hat_i=g_phi(z_i,T_i,C_i)`，建议主路线继续明确区分：

```text
空间表示：z_i = f_theta(S_i)
预测条件：t_i = E_T(T_i), b_i = E_C(C_i)
响应预测：tau_hat_i = g_phi(z_i, t_i, b_i)
```

这里的“干预感知”由训练目标实现：响应预测的梯度会更新生成 z 的编码器。编码器无需把本次干预属性 T 混进 z，才算干预感知。T 不作为编码器输入，也不意味着要求 z 与 T 在统计上独立；干预选址与开通前状态本来可能相关。

这一安排与 TARNet 将协变量表示和处理条件分开的架构思想相符，但本项目已有准实验响应标签，监督层的 T 是干预描述而非 treated/control 指示变量，不能照搬其识别假设、处理组平衡损失或潜在结果估计流程。[Shalit et al., ICML 2017](https://proceedings.mlr.press/v70/shalit17a.html)

另设 `z_SC=f(S,C)` 的比较方案，可以检验将 AlphaEarth、人口和房价等事前背景纳入空间表示是否改善迁移。这是值得比较的研究选择，现有文献并不能证明项目当前的 S/C 分工最优。对比时保持字段来源和时间门禁，避免同一来源被重复计为新增模态。T 继续条件化响应预测。若输出依赖 T 的 `h_i=H(z_i,t_i,b_i)`，应将其命名为“事件条件表示”，与地点状态表示 z 分别报告。

**不建议仅为把 416 凑成 512，再增加一个表示层。也不建议将“S/C/T 都是模型输入”直接等同于“唯一导出的地点 embedding 必须直接依赖 S/C/T 全部字段”。**这两个设计都需要实验依据。

## 2. 调研范围和证据强度

这是针对当前模型决策的定向文献调研，不是系统综述。检索覆盖多模态城市表示、遥感基础表示、变长观测与时间编码、处理效应表示、连续响应度量学习和表示维度选择。重点核对以下 17 篇论文及 Google 官方产品说明，正文区分文献原始结论与本项目的设计推断。

已使用 arXiv、CVF、PMLR、NeurIPS、OpenReview、Nature 和作者官方仓库等原始来源核对。当前会话未提供技能中列出的 academic-search MCP；其 OpenAlex 后备检索脚本因本地网络访问错误 WinError 10013 失败，随后通过可用网页工具访问上述原始来源。未使用未验证的引用次数排名。预印本不视作已经同行评审的证据。

| 编号 | 文献及来源 | 已核对的结论 | 对本项目的适用范围 |
|---|---|---|---|
| 1 | Brown et al. (2025), [AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data](https://arxiv.org/html/2507.22291v1)，arXiv:2507.22291 | 多源、带时间条件的紧凑表示；补充材料 S21.3 实际比较瓶颈容量与维数，效果依任务而变。 | 借鉴表示与时间建模及评估方法；不能据其64维直接决定本项目维数，也不能从本项目数千事件复刻其基础训练规模。 |
| 2 | Gong et al. (2026), [Earth Embeddings Reveal Diverse Urban Signals from Space](https://arxiv.org/html/2604.03456v1)，arXiv:2604.03456，预印本 | 比较城市指标上的 AlphaEarth、Prithvi、Clay；紧凑表示仍可有竞争力，任务和城市间差异明显。 | 支持保留强预训练基线和比较维数。其 Methods 使用对应设置内随机80/20划分，不能将 city-wise 结果等同于严格留城外推。 |
| 3 | Wang, Li & Rajagopal (2020), [Urban2Vec: Incorporating Street View Imagery and POIs for Multi-Modal Urban Neighborhood Embedding](https://arxiv.org/abs/2001.11101)，AAAI 2020 | 街景和 POI 可以联合学习邻里表示，并用于多个下游任务。 | 支持异质城市信息融合；其空间邻近关系和观测指标目标不直接保证政策响应相似性。 |
| 4 | Hao et al. (2025), [UrbanVLP: Multi-Granularity Vision-Language Pretraining for Urban Socioeconomic Indicator Prediction](https://arxiv.org/html/2403.16831v3)，AAAI 2025，arXiv:2403.16831 | 街景与卫星提供不同粒度信息；预训练后采用冻结编码器和轻量预测器。 | 支持多粒度融合及有限标签下的轻量训练；其文本生成和静态指标目标不是本项目必须复制的模块。 |
| 5 | Klemmer et al. (2025), [SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery](https://arxiv.org/html/2311.17179v3)，AAAI 2025，初稿2023 | 图像与地点的对齐产生可迁移表示；正文附录使用冻结图像编码器和256维投影。 | 256是可参考的工程起点，不是我们任务的最优维度证据。位置编码还应检查地理捷径。 |
| 6 | Astruc et al. (2025), [AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html)，CVPR 2025 | 用模态相关编码及共同模型处理不同传感器、尺度与分辨率的观测。 | 支持先处理来源和尺度差异再融合；不能默认现有六波段年度影像与其权重接口直接兼容。 |
| 7 | Tseng et al. (2025), [Galileo: Learning Global & Local Features of Many Remote Sensing Modalities](https://arxiv.org/html/2502.09356v3)，所读为arXiv稿 | 区分空间、时间及模态结构，组合全局与局部目标；模型宽度存在128、192、768等配置。 | 共享表示不要求维数一律为2的幂；局部与总体信息可能互补。训练数据和目标与政策响应不同。 |
| 8 | Li et al. (2025), [Learning Street View Representations with Spatiotemporal Contrast](https://arxiv.org/html/2502.04638v1)，arXiv:2502.04638，所读为预印本 | 不同街景空间/时间对比目标偏向不同的动态、静态或邻里信息。 | 不能不加判断地把同一地点跨年的变化全部当作应消除的噪声；施工、商业更新可能与响应有关。 |
| 9 | Ilse, Tomczak & Welling (2018), [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html)，ICML 2018 | 以注意力对变长实例集合做置换不变聚合。 | 支持多张街景到一个地点向量；原文不提供地铁时间衰减规律，注意力权重也不等于因果贡献。 |
| 10 | Shukla & Marlin (2021), [Multi-Time Attention Networks for Irregularly Sampled Time Series](https://openreview.net/forum?id=4c0J6lwQ4_)，ICLR 2021 | 连续时间编码及注意力将不规则观测变为定长表示。 | 支持保留拍摄时间与间隔；不能据此认定12个月是统一压缩尺度或某个衰减函数最优。 |
| 11 | Shalit, Johansson & Sontag (2017), [Estimating individual treatment effect: generalization bounds and algorithms](https://proceedings.mlr.press/v70/shalit17a.html)，ICML 2017 | 从协变量生成表示、在处理条件下预测潜在结果；讨论处理组平衡与误差界。 | 可借鉴表示与处理条件分工；本项目的准实验标签监督不自动满足其设定，处理描述T不等同处理组指示。 |
| 12 | Curth & van der Schaar (2021), [Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms](https://proceedings.mlr.press/v130/curth21a.html)，AISTATS 2021 | 区分插件估计与伪结果回归，学习效果依估计策略和条件而变。 | 支持明确上游因果估计与下游学习两个阶段；并未证明本项目GSC/Matching标签的误差性质。 |
| 13 | Zha et al. (2023), [Rank-N-Contrast: Learning Continuous Representations for Regression](https://papers.nips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)，NeurIPS 2023 | 利用连续目标的距离排序塑造表示几何，并验证分布变化下的效果。 | 比任意把连续响应分箱更贴近本项目目标；多维、带mask、带估计误差的因果响应需要另行适配和消融。 |
| 14 | El Bouchattaoui et al. (2024), [Causal Contrastive Learning for Counterfactual Regression Over Time](https://proceedings.neurips.cc/paper_files/paper/2024/hash/02cef2ae63853724eb99e70721d3bc65-Abstract-Conference.html)，NeurIPS 2024 | 在纵向反事实回归中结合RNN、预测编码及信息保留目标。 | 表明对比学习可服务反事实任务；其时变处理和混杂设定与一次开通事件不同，不要求我们开展图像转移训练。 |
| 15 | Kusupati et al. (2022), [Matryoshka Representation Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html)，NeurIPS 2022 | 在多个维数上显式训练可截取表示，比较维数与任务表现。 | 支持把维数作为性能/成本选择；未经这种训练，不能任意截取普通embedding前64维宣称语义保留。 |
| 16 | Rolf et al. (2021), [A generalizable and accessible approach to machine learning with global satellite imagery](https://www.nature.com/articles/s41467-021-24638-z)，Nature Communications 12:4392，DOI:10.1038/s41467-021-24638-z | 固定遥感特征与简单回归能在多个任务上取得有竞争力的结果。 | 要保留冻结特征+Ridge/树模型基线，不能以深网络复杂程度代替增益证明。 |
| 17 | Zhang, Bellot & van der Schaar (2020), [Learning Overlapping Representations for the Estimation of Individualized Treatment Effects](https://proceedings.mlr.press/v108/zhang20c.html)，AISTATS 2020 | 讨论域不变表示在处理效应估计中的限制及信息可逆性/支持集问题。 | 不应为了跨城外推盲目消除所有城市背景，或声称高维数本身就保证信息保留；理论不直接给出本项目最优融合结构。 |

另核对 [Google Satellite Embedding 官方说明](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)：64个轴共同表示一个坐标，应完整使用；这不是禁止下游网络学习投影的规定，也不意味着64个坐标各有独立物理含义。

## 3. 对前述维度建议的修正

### 3.1 原始输入宽度不等于最终表示宽度

当前 C 含82个数值：AlphaEarth64与18个表格值；另有18个字段mask与1个AlphaEarth可用性mask。若全部直接拼接，共101个槽位。mask不等于额外19个实体背景属性，也不强制必须逐位原样写入最终表示。当前实现的AlphaEarth可用性标记用于选择缺失token，并非实际与64维向量直接拼接。

学习表示通常会改变维度。“输入含AlphaEarth64，再加属性，所以最终embedding数学上必然大于64”并不成立。前面把“不能压缩”作为一般结论过于绝对。更合适的要求是：完整输入、不预先任意删维、明确瓶颈位置，并用下游及迁移实验验证信息是否足够。AlphaEarth本身的维数和噪声瓶颈也是经过任务比较选择的，而非所有来源维数相加。[AlphaEarth，S21.3](https://arxiv.org/html/2507.22291v1#S21.SS3)

### 3.2 不为凑整数扩大联合向量

本项目可以偏好128、256、512等宽度，便于配置和比较，但不存在“维数必须是2的幂”的表示学习原则。相比宽度本身，任务信号、时间编码、信息保留、正则化和外推验证更重要。Galileo使用192和768等配置也是直接反例。[Galileo，模型配置](https://arxiv.org/html/2502.09356v3)

把416维拼接向量映射为512维，可以形成新的非线性特征，却不会自动产生额外独立观测信息。因此不把它作为主模型必须包含的模块。

### 3.3 建议的起始宽度

下表是根据当前任务规模与模态结构提出的工程候选，**不是文献证明的最优配置**。

| 对象 | 建议起点 | 含义 |
|---|---:|---|
| S中的POI/VIIRS历史编码 | 64 | 原始15值与15mask的非线性时序摘要 |
| S中的街景集合编码 | 128 | 逐张384维特征结合时间编码，再做可变长聚合 |
| S中的卫星时序编码 | 128 | 各年独立图像特征及年度位置/有效mask，再做时序聚合 |
| 主空间embedding z | **256** | 融合上述状态分支；三子分支拼接320维后学习256维表示 |
| C中的AlphaEarth块 | **64，保留完整向量** | 合法年份且可用时直接保留一个完整输入路径；缺失有显式标记 |
| C中的其余背景编码 | 64 | 18值、18mask及AlphaEarth可用性标记，共37个输入槽位到64维 |
| C的条件表示 b | **128** | 完整AlphaEarth64与其余背景64拼接；主方案不先将AlphaEarth64压到32 |
| T的条件表示 t | **32** | 7值与7mask到32维，保持轻量 |
| 响应预测器拼接输入 | **416** | `concat(z256,b128,t32)`，其宽度无须是2的幂 |
| 预测器隐藏层 | 128或256 | 由训练/验证城市选择，不作为唯一空间embedding解释 |
| 输出监督 | VIIRS 6；POI 12 | 预测两族冻结维度的准实验响应，不把其维数当成z的维数 |

S的320到256以及街景/卫星观测到定长表示仍是学习到的摘要，不能声称全流程无损。C保留原始AlphaEarth路径只是减少不必要的早期瓶颈；不能由“输出宽度不小于输入宽度”推导网络保持所有信息。

主维度比较：固定C/T和输入资产，比较 `dim(z)={128,256,512}`；已有64维运行作为历史和工程对照。只用验证城市选择最终宽度，随后锁定配置在测试城市评估。如果128和256表现接近，优先较小者。维度搜索期间保持训练预算、早停规则和基线尽可能一致，并记录参数量，避免把更多参数的收益误写成“表示维度”的独立收益。

## 4. 推荐的训练路线

### 4.1 复用预训练特征，小规模学习融合与响应

冻结街景DINOv2为起点，保留逐张384维特征缓存，训练时间编码、集合聚合、模态投影、状态融合及响应头。比较已验收的小型六波段CNN与适配后确实兼容的遥感预训练编码器；不先以更换基础模型作为启动条件。[UrbanVLP](https://arxiv.org/html/2403.16831v3)、[MOSAIKS](https://www.nature.com/articles/s41467-021-24638-z)支持冻结/固定特征配轻量学习器作为强起点。

当前输入有5,048个候选事件，但这不等于5,048个都具备完整有效响应；更不等于图片、年份或逐维标签数那么多个独立事件。监督量应按有效事件、城市和结果族重新计数。相邻网格、同线同月开通及同城事件还可能相关。如此规模不足以单凭“借鉴AlphaEarth”就决定从头训练大型基础模型。

若后续要做自监督适配，可用合法事前观测训练模态一致性或掩码特征预测，随后加入响应监督，并比较其真实增益。严格归纳外推实验中，自监督也仅使用训练城市；使用未标注测试城市的方案另列为transductive设定。不是必须增加未来图像解码或事前/事后视觉配对。

### 4.2 时间信息进入每张图、每个年份

街景：对每张合法图片保留真实拍摄月、距开通月的间隔和有效性；图像特征与时间编码共同决定聚合。当前 `d/(d+12)` 可以保留为候选，但12是人为尺度，单一单调变换也不证明“越近必然越有用”。比较现有压缩时间、对数间隔和连续时间编码时，不改变图片支持集。数据确实有质量指标时再输入，不虚构质量分数。[Attention MIL](https://proceedings.mlr.press/v80/ilse18a.html)、[mTAN](https://openreview.net/forum?id=4c0J6lwQ4_)

卫星：保持 -6至-1年位置，缺年不挤掉对应槽位；先提每年特征，再做masked GRU或轻量时间注意力。保留相对年份和像素/年度有效mask。现有传感器来自Sentinel-2和多个Landsat系列，不能把六个相同名称波段就视作完全同分布。传感器和原生分辨率信息是已有来源元数据的适配候选，不允许借机引入未来观测。[AnySat](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html)

### 4.3 先验证响应预测，再检验度量学习的增益

主目标使用逐维mask和已有可靠性/设计权重的Huber响应损失，按结果族和有效权重归一化。不同指标的尺度只在训练城市拟合；最终误差还原到可解释的原始标签尺度报告。标签可靠性权重不能只由数值大小代替，也不能把上游质量权重不经检查重复乘入。

```text
L_anchor = family_mean[ sum(m * w * Huber(tau_hat - tau_QE)) / sum(m * w) ]
L = L_anchor + lambda_align * L_response_metric
```

`lambda_align=0` 是必须存在的基线；例如0、0.05、0.15可作为候选，实际大小同时受两项损失数值尺度影响。这些候选是项目建议，并非论文确定的统一比例。

响应相似性项可借鉴连续目标的距离排序方法，或比较当前soft InfoNCE。响应是连续、带正负幅度、带mask的向量，不能只把“方向相同”当成等价响应，更不能把近零响应的余弦距离当成稳定相似性指标。应以共同有效维度计算尺度明确的加权差异，设置最小可比较支持，并报告幅度与轨迹形状各自效果。[Rank-N-Contrast](https://papers.nips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)

**需要新增的适用性判断**：响应相似性同时受S、T和C影响。如果直接要求任意两事件的z距离等于观察到的响应距离，可能把城市背景或干预差异错误压到状态几何里。主比较先包括无对比项；增强项在有共同监督维度、可比较干预条件的事件之间计算，并对背景可比性做敏感性分析。用T/C相似度加权或分层只是待验证的项目适配，不能说RNC已经证明它对带噪准实验标签有效。也不把未经验证的模型预测响应再当真值生成监督。

现有上游是准实验反事实估计。对它们进行监督不会自动消除未观测混杂、提前反应或估计偏差；论文结论应称为响应估计的预测与迁移，而非已观测个体真实处理效应的恢复。[Curth & van der Schaar](https://proceedings.mlr.press/v130/curth21a.html)

### 4.4 一套共享表示，两个结果族的头

目标是得到同一个响应敏感空间坐标系，因此建议在同一事件身份表上共享编码器，分别预测VIIRS6维和POI12维，缺哪个响应单元就屏蔽哪个单元。按族归一化损失，避免POI仅因维数较多而占优；每族的时间窗口、质量权重及字段含义保持冻结。

分别训练VIIRS、POI的工程闭环不证明已经学得同一空间。两个独立训练模型的z坐标通常可以任意旋转，不能把其余弦距离直接放到一起比较。现有模型/数据代码存在多族路径，但此次已核对的服务器验收是两族分别运行；下一次研究实验要专门核对共享模型的输出映射、mask、损失和导出语义。这里没有断言现有多族接口全部不可用。

## 5. 比单独扩大维度更值得优先检查的实现位置

1. **早期背景瓶颈**：当前ContextEncoder先把AlphaEarth64映射到32。建议以完整64路径为主方案，与压缩路径消融；不根据官方“全维输入”要求断言一切学习投影都错误。
2. **卫星空间信息**：当前SatelliteSixBandCNN经64通道与全局平均池化得到64维，再线性映射到目标宽度。把末层输出改成256/512不能恢复此前池化丢掉的布局。
3. **输入空间采样**：当前dataset默认把年度影像统一到17×17；采集的Sentinel-2资产可为50×50。对原生采样、重采样与传感器适配的比较，比单纯末层扩宽更能检验视觉信息利用程度。
4. **导出对象**：当前STCResponseModel导出z来自S，forward同时返回拼接向量。修改后必须明确哪个是空间embedding、哪个是预测器输入，不能将拼接宽度称为已验证的语义维数。
5. **缺失与城市/年份相关**：AlphaEarth较早年份不可用、街景覆盖和房价覆盖不均。mask有助于正确处理缺失，也可能携带城市与年份捷径；必须增加仅mask/观测年龄基线及分覆盖组评估。

本节依据本地源码读取提出；并未对模型行为完成新的因果或性能验证。对应文件：`src/urban_intervention/representation/encoder.py`、`dataset.py`、`model.py`、`loss.py`、`training/runner.py`。

## 6. 用什么实验作出最终决定

所有比较固定事件支持集、严格事前门禁、响应质量mask及正式城市划分；不能通过只保留全模态事件使不同模型比较失去可比性。模型选择只用训练/验证城市，测试城市锁定。主结果按城市汇总，并以城市为聚类单位估计不确定性，不能把同城大量标签单元视作独立城市样本。

| 比较 | 要回答的问题 |
|---|---|
| 训练均值；仅mask/观测年龄；T+C（含正常mask）；T+C+全部合法原始表格 | 背景、干预及缺失机制本身能预测多少响应？ |
| AlphaEarth64+原始表格+T，Ridge/树模型；冻结多模态特征+同类轻量模型 | 新网络是否超过强预训练和简单监督基线？ |
| z(S)+T+C；z(S,C)+T | AlphaEarth和其他背景应作为预测条件，还是也进入可复用空间embedding？ |
| 预测损失；预测+现有对比；预测+连续响应度量适配 | 响应几何学习是否有独立、稳定的迁移增益？ |
| z为128/256/512；C保留64路径或投影 | 宽度与早期瓶颈的收益是否超过随机种子波动与参数量差异？ |
| 去街景；去独立卫星；去AlphaEarth；打乱观测时间 | 模态和时间信息是否提供了可验证的增量？ |
| 冻结z后线性/Ridge探针、条件近邻检索、跨结果族/时间窗口探针 | 表示本身是否有用，而非只由强响应头拟合标签？ |

报告逐维/逐族误差、城市平均表现、方向与幅度表现、有效监督量、不同覆盖和观测年龄组，以及训练预算/参数量。主目标是未见城市响应预测；UMAP图不作为主要证据。近邻检索时同时报告干预/背景条件可比性，不能只把混合事件下的响应相似当成状态相似。

准实验响应并非无噪真值，测试误差仍是相对这些标签的误差。可用不同已合法发布估计来源的一致性做敏感性分析，不改写上游生产标签来迎合模型。既有服务器短训练证明工程闭环，不证明本次任何模型候选的研究性能。

## 7. 结论及下一阶段范围

建议主路线是：**冻结或轻量适配的多模态编码器 + 时间感知聚合 + 可复用的256维状态表示 + 完整AlphaEarth输入路径和独立干预/背景条件 + 带可靠性权重的多结果响应监督**。响应对比项作为需要证明增益的增强，而非将其固定为不可质疑的核心比例。

512作为维数候选保留；不通过把S/C/T硬融合成512来定义研究贡献。S-only与S+C的空间表示范围由同口径实验决定；不擅自改动当前已发布的角色契约。若用户随后要求实施，优先把分支宽度/导出语义和共享多族监督改成明确可配置的研究接口，再开展上述小规模候选比较及严格跨城主实验。

本次工作只新增本备忘录；没有修改训练源码、运行生产训练、同步服务器或替换验收证据。
