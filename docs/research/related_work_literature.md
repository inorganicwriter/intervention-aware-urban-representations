# 相关工作文献综述

## Intervention-aware Urban Representation Learning

---

## 1. Urban Representation Learning / 城市空间表征学习

### 1.1 多模态城市嵌入 (Multimodal Urban Embeddings)

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Zhang, F. et al. (2021).** *Uncovering inconspicuous places using social media check-ins and street view images.* Computers, Environment and Urban Systems. | 利用街景+社交媒体签到学习place embedding | 多模态融合范式参考 |
| **Yao, Y. et al. (2017).** *Sensing spatial distribution of urban land use by integrating points-of-interest and Google Word2Vec model.* IJGIS. | POI + Word2Vec 学习地块功能表征 | POI语义编码方法 |
| **Mai, G. et al. (2023).** *On the opportunities and challenges of foundation models for geospatial artificial intelligence.* arXiv. | 地理空间基础模型的综述 | 整体框架定位 |
| **Yan, B. et al. (2017).** *A spatially explicit reinforcement learning model for geographic knowledge graph summarization.* Transactions in GIS. | 空间知识图谱表征学习 | 空间结构编码 |
| **Jenkins, P. et al. (2023).** *Urban-GAN: Procedural urban layout generation with generative adversarial networks.* | 生成式城市空间表示 | Embedding空间结构化 |
| **Zhai, W. et al. (2022).** *A scale-sensitive framework for spatially explicit GeoAI applications.* | GeoAI的空间尺度敏感性 | 500m grid尺度选择依据 |

### 1.2 街景与卫星影像的城市感知

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Naik, N. et al. (2017).** *Computer vision uncovers predictors of physical urban change.* PNAS. | 使用街景时序数据预测城市物理变化 | **最相关** — street view time series + urban change |
| **Gebru, T. et al. (2017).** *Using deep learning and Google Street View to estimate the demographic makeup of neighborhoods.* PNAS. | 街景→人口统计推断 | 街景编码器选型依据 |
| **Jean, N. et al. (2016).** *Combining satellite imagery and machine learning to predict poverty.* Science. | 卫星影像+CNN预测贫困 | 遥感经济推断范式 |
| **Dubey, A. et al. (2016).** *Measuring human-perceived similarity in cities using street view images.* | 城市视觉相似度度量 | Embedding空间距离度量 |
| **Mittal, P. et al. (2023).** *Geographical distance matters: A multi-view representation learning approach for human mobility.* | 多视图地理表征学习 | 空间-语义联合embedding |
| **Liu, Y. et al. (2023).** *Urban visual intelligence: Studying cities with AI and street-level imagery.* AAAG. | 街景+AI的城市研究综述 | 方法论全景 |

### 1.3 视觉-语言模型与城市理解

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Radford, A. et al. (2021).** *Learning transferable visual models from natural language supervision.* ICML (CLIP). | CLIP模型 | VLM backbone选型 |
| **Oquab, M. et al. (2023).** *DINOv2: Learning robust visual features without supervision.* | 自监督视觉特征学习 | 卫星/街景encoder |
| **Mai, G. et al. (2024).** *CSP: Self-supervised contrastive spatial pre-training for geospatial-visual representations.* | 地理空间对比学习预训练 | **高度相关** — geospatial contrastive learning |
| **Zhou, Y. et al. (2024).** *GeoGPT: Understanding and reasoning with geospatial large models.* | 地理空间大语言模型 | Foundation model for geo |
| **Cong, Y. et al. (2023).** *SatCLIP: Global, general-purpose location embeddings with satellite imagery.* | 卫星影像的全球位置嵌入 | 卫星影像→embedding范式 |
| **Buyukdemircioglu, M. et al. (2024).** *RemoteCLIP: A vision language foundation model for remote sensing.* | 遥感视觉语言基础模型 | Remote sensing VLM |

---

## 2. Causal Inference for Urban Policy / 城市政策的因果推断

### 2.1 准实验方法基础

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Abadie, A. et al. (2010).** *Synthetic control methods for comparative case studies: Estimating the effect of California's tobacco control program.* JASA. | 合成控制法 | SCM方法基础 |
| **Abadie, A. et al. (2021).** *Sampling-based versus design-based uncertainty in regression analysis.* Econometrica. | 设计型vs抽样型推断 | 研究设计理论 |
| **Athey, S. & Imbens, G. (2022).** *Design-based analysis in difference-in-differences settings with staggered adoption.* Journal of Econometrics. | 交错采纳DiD | 地铁分期开通的DiD设计 |
| **Sun, L. & Abraham, S. (2021).** *Estimating dynamic treatment effects in event studies with heterogeneous treatment effects.* Journal of Econometrics. | 异质性处理效应的event study | Event study方法 |
| **Callaway, B. & Sant'Anna, P. (2021).** *Difference-in-differences with multiple time periods.* Journal of Econometrics. | 多期DiD | 多次treatment窗口处理 |
| **Athey, S. et al. (2021).** *Matrix completion methods for causal panel data models.* JASA. | 矩阵补全用于因果面板 | Matrix Completion 作为第三层回退的方法依据 |
| **Ratledge, N. et al. (2022).** *Using machine learning to assess the livelihood impact of electricity access.* Nature. | ML 预测 + MC/SC-EN 因果推断 | MC 在数据稀疏环境下的可靠性验证 |
| **Xu, Y. (2017).** *Generalized synthetic control method: Causal inference with interactive fixed effects models.* Political Analysis. | 广义合成控制法 | GSC 回退的方法依据 |

### 2.2 轨道交通与城市发展的因果推断

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Gonzalez-Navarro, M. & Turner, M. (2018).** *Subways and urban growth: Evidence from Earth.* Journal of Urban Economics. | 地铁与城市增长的因果证据 | **核心参照** |
| **Donaldson, D. (2018).** *Railroads of the Raj: Estimating the impact of transportation infrastructure.* AER. | 铁路对经济发展的影响 | 基础设施因果推断范式 |
| **Baum-Snow, N. (2007).** *Did highways cause suburbanization?* QJE. | 高速公路与郊区化 | 交通设施→空间重构 |
| **Mayer, T. & Trevien, C. (2017).** *The impact of urban public transportation: Evidence from the Paris region.* Journal of Urban Economics. | 巴黎公共交通的影响估计 | 地铁站异质性效应 |
| **Mulalic, I. et al. (2023).** *The impact of metro expansion on housing prices and population density.* | 地铁扩展对房价和人口的影响 | housing作为control variable |
| **Bollinger, C. & Ihlanfeldt, K. (1997).** *The impact of rapid rail transit on economic development: The case of Atlanta's MARTA.* JUE. | 轨道交通对经济活动的影响 | 经典TOD实证 |

### 2.2.1 轨道网络可达性的测度（匹配协变量构建的方法论依据）

| 文献 | 方法 | 本项目采用 |
|------|------|-----------|
| **Smersh, G.T. & Smith, M.T. (2000).** *Accessibility changes and urban house price appreciation.* Journal of Housing Economics. | 到轨道交通设施距离作为可达性测度 | **距离变量**（dist_nearest_station_m） |
| **Debrezion, G., Pels, E. & Rietveld, P. (2007).** *The impact of railway stations on residential and commercial property value: A meta-analysis.* Journal of Real Estate Finance and Economics. | 站距对房价影响的元分析，确认500m/800m/1km缓冲区与连续距离两类主流测度 | **缓冲区站点数**（stations_500m/800m/1500m） |
| **Yang, L. et al. (2021).** *Place-varying impact of metro accessibility on property prices.* (Springer, Property Price Impacts of Environment-Friendly Transport Accessibility). | 到最近地铁站距离与步行缓冲区的时空变系数处理 | **最近站距离**（连续） |
| **To, W.M. (2015).** *Centrality of an urban rail system.* Urban Rail Transit 1(3):155-162. | 站点为节点、轨道段为边，closeness centrality 测度轨道网络可达性；确认站间直线距离近似轨道长度的误差可忽略 | **网络 closeness**（同线路相邻站连边） |
| **Gao, Z. & Wang, Y. (2026).** *Subway network accessibility and carbon balance: evidence from China's dynamic centrality analysis.* Carbon Balance and Management. | 动态网络中心性（逐年快照）测度地铁网络可达性 | **处理前快照**（动态网络） |
| **Wu, Q. et al. (2022).** *Regional impact of urban rail transit network accessibility on residential property price.* ICRT 2021. | 网络可达性对房价的区域影响 | **网络 closeness → 网格赋值** |
| **Zhang, X. et al. (2018).** *Urban rail transit network vulnerability measurement based on complex network theory.* DEStech (Chongqing case). | 复杂网络理论（度/介数/接近中心性）在轨道网络的应用 | 中心性指标选型参考 |

**测度结构**：
1. `dist_nearest_station_m`：到最近已开通地铁站的直线距离（Smersh & Smith 2000；Yang et al. 2021）
2. `stations_500m/800m/1500m`：处理前各缓冲区半径内已开通站点数（Debrezion et al. 2007 元分析确认的主流口径）
3. `lines_in_1500m`：处理前 1.5km 缓冲区内经过的线路条数（Debrezion et al. 2007 多站点/多线路维度）
4. `network_closeness`：处理前网络快照的站点 closeness centrality（To 2015；Gao & Wang 2026 动态快照），网络拓扑用 **Wikidata P197 相邻站关系**（真实线路拓扑），网格取最近站值（Wu et al. 2022）
5. 站点属性：换乘站（多线路）、终点站（P197 网络度≤1）、新线/延长线（线路首开年推断）、同期开通规模（同月同城开通站数）

### 2.3 Gentrification 与 Displacement

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Zuk, M. et al. (2018).** *Gentrification, displacement, and the role of public investment.* Journal of Planning Literature. | 绅士化与公共投资的综述 | 区分activation vs displacement |
| **Ellen, I. & O'Regan, K. (2011).** *How low-income neighborhoods change: Entry, exit, and enhancement.* Regional Science and Urban Economics. | 低收入社区的变化机制 | POI进入/退出追踪的理论基础 |
| **Chapple, K. et al. (2017).** *Developing a new methodology for analyzing potential displacement.* | 潜在displacement的分析方法 | displacement测量 |
| **Gupta, A. et al. (2022).** *The anatomy of gentrification-induced displacement.* | 绅士化引发displacement的解剖 | **高度相关** |
| **Glaeser, E. et al. (2018).** *Urban gentrification and the spatial structure of cities.* | 绅士化与城市空间结构 | 空间异质性 |
| **Delmelle, E. (2021).** *Transit-induced gentrification and displacement: The state of the practice.* | 交通引发的绅士化综述 | **核心文献** — TOD gentrification |

---

## 3. Urban Remote Sensing / 城市遥感

### 3.1 夜间灯光

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Henderson, V. et al. (2012).** *Measuring economic growth from outer space.* AER. | 夜间灯光测量经济增长 | **经典参考** |
| **Chen, X. & Nordhaus, W. (2011).** *Using luminosity data as a proxy for economic statistics.* PNAS. | 光度作为经济统计代理变量 | VIIRS经济学基础 |
| **Elvidge, C. et al. (2017).** *VIIRS night-time lights.* International Journal of Remote Sensing. | VIIRS夜间灯光技术综述 | 数据技术细节 |
| **Zhao, M. et al. (2023).** *A global dataset of annual urban extents (1992-2020) from harmonized nighttime lights.* | 全球城市范围数据集 | 城市边界识别 |
| **Zhang, F. et al. (2024).** *Fragile booms in cities.* (your MACRO paper) | 夜间灯光+多模态的城市脆弱性 | **自有工作** — VIIRS作为fast signal |

### 3.2 卫星影像城市变化检测

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Huang, X. et al. (2021).** *Mapping urban land use by combining social media and satellite images.* | 社交媒体+卫星影像的土地利用制图 | 多源融合 |
| **Li, X. et al. (2020).** *Mapping global urban boundaries from the global artificial impervious area (GAIA) data.* | 全球城市不透水面制图 | 建成区提取 |
| **He, C. et al. (2023).** *Global urban land expansion and its driving forces.* | 全球城市土地扩张驱动力 | 建成区变化趋势 |
| **Zhu, X. et al. (2022).** *Deep learning meets time series analysis for urban change detection from satellite images.* | 深度学习+时间序列城市变化检测 | Sentinel时序处理 |
| **Gong, P. et al. (2020).** *Annual maps of global artificial impervious area (GAIA) between 1985 and 2018.* RSE. | 全球逐年不透水面 | 建成区年度变化 |

### 3.3 NO2作为经济活动代理变量

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Cui, Y. et al. (2022).** *Can NO2 column measurements from space track global GDP growth?* | NO2柱浓度追踪GDP增长 | NO2作为经济代理的验证 |
| **de Foy, B. et al. (2016).** *Satellite NO2 retrievals suggest China's economy grew as emissions decreased during COVID-19.* | COVID-19期间NO2与经济变化 | NO2+经济事件 |
| **Fowlie, M. et al. (2019).** *Bringing satellite-based air quality estimates down to Earth.* AER P&P. | 卫星空气污染数据的地面验证 | 数据质量讨论 |

---

## 4. Urban Functional Zones & POI Dynamics / 城市功能区与POI动态

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Yuan, J. et al. (2012).** *Discovering regions of different functions in a city using human mobility and POIs.* KDD. | POI+人类移动性发现城市功能区 | 经典方法 |
| **Crooks, A. et al. (2015).** *Crowdsourcing urban form and function.* | 众包城市形态与功能 | OSM-POI的可靠性 |
| **McKenzie, G. et al. (2018).** *A measure of social and geographical category similarity between POIs.* | POI间社会-地理类别相似度 | POI语义编码 |
| **Jiang, S. et al. (2015).** *The timegeo modeling framework for urban mobility without travel surveys.* | 时间地理建模框架 | POI时间模式 |
| **Berjisian, A. et al. (2023).** *Neighborhood change and the role of new businesses: Evidence from Yelp data.* | 新商户与社区变化 | POI entry/exit → 社区变化 |
| **Glaeser, E. et al. (2014).** *Entrepreneurship and urban growth: An empirical assessment with historical mines.* | 创业与城市增长 | 商业活力与城市发展 |

---

## 5. Transit-Oriented Development / 公交导向开发

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Cervero, R. & Kockelman, K. (1997).** *Travel demand and the 3Ds: Density, diversity, and design.* Transportation Research D. | TOD的3D框架 | 控制变量设计 |
| **Ewing, R. & Cervero, R. (2010).** *Travel and the built environment: A meta-analysis.* JAPA. | 建成环境与出行的元分析 | density/design指标 |
| **Debrezion, G. et al. (2007).** *The impact of railway stations on residential and commercial property value: A meta-analysis.* JRES. | 铁路站点对房地产价值影响元分析 | **高度相关** |
| **Bartholomew, K. & Ewing, R. (2011).** *Hedonic price effects of pedestrian- and transit-oriented development.* Journal of Planning Literature. | TOD对房价的影响 | property value channel |
| **Cao, X. & Porter-Nelson, D. (2016).** *Real estate development in anticipation of the Green Line light rail transit in St. Paul.* | 轻轨预期效应下的房地产 | anticipation effects |
| **Diao, M. (2019).** *Towards sustainable urban transport in Singapore: Policy instruments and mobility trends.* Transport Policy. | 新加坡可持续交通政策 | 亚洲TOD案例 |
| **Padeiro, M. et al. (2019).** *Transit-oriented development and gentrification: A systematic review.* | TOD与绅士化系统综述 | displacement identification |

---

## 6. Contrastive Learning for Spatiotemporal Data / 时空对比学习

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Chen, T. et al. (2020).** *A simple framework for contrastive learning of visual representations (SimCLR).* ICML. | 对比学习框架 | 方法基础 |
| **He, K. et al. (2020).** *Momentum contrast for unsupervised visual representation learning (MoCo).* CVPR. | 动量对比学习 | Encoder设计 |
| **Zhang, Y. et al. (2022).** *Spatio-temporal contrastive learning for urban computing.* | 时空对比学习用于城市计算 | **高度相关** |
| **Rao, J. et al. (2023).** *Contrastive representation learning for geospatial entities.* | 地理实体的对比表征学习 | Geospatial CL |
| **Wang, Z. et al. (2023).** *Self-supervised learning for spatio-temporal forecasting: A contrastive learning approach.* | 自监督时空预测 | response prediction的预训练 |
| **Liang, Y. et al. (2023).** *When do contrastive representations generalize? Understanding the dynamics of contrastive pre-training.* | 对比预训练泛化理论 | 理论支撑 |

---

## 7. Urban Foundation Models / 城市基础模型

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Li, Z. et al. (2024).** *Urban foundation models: A survey.* | 城市基础模型的全面综述 | 定位你的工作 |
| **Zhao, Y. et al. (2024).** *UrbanDiT: A diffusion transformer for city-scale urban dynamics.* | 扩散Transformer城市动态 | foundation model范式 |
| **Xue, H. et al. (2023).** *UrbanGPT: Spatio-temporal large language models.* | 时空大语言模型 | LLM for urban |
| **Manvi, R. et al. (2024).** *GeoLLM: Extracting geospatial knowledge from large language models.* | 从LLM提取地理空间知识 | LLM+Geo |
| **Zhang, W. et al. (2024).** *City foundation models for a generalizable understanding of urban dynamics.* | 城市基础模型综述 | 整体方法框架 |

---

## 8. 经济学理论基础

| 文献 | 核心贡献 | 关联 |
|------|----------|------|
| **Duranton, G. & Puga, D. (2004).** *Micro-foundations of urban agglomeration economies.* Handbook of Urban and Regional Economics. | 城市集聚经济的微观基础 | 集聚→经济增长 |
| **Glaeser, E. & Gottlieb, J. (2009).** *The wealth of cities: Agglomeration economies and spatial equilibrium in the United States.* JEL. | 集聚经济与空间均衡 | displacement机制 |
| **Moretti, E. (2013).** *Real wage inequality.* American Economic Journal: Applied Economics. | 城市工资差异 | 人力资本channel |
| **Combes, P. et al. (2012).** *The productivity advantages of large cities: Distinguishing agglomeration from firm selection.* Econometrica. | 大城市生产率优势 | 集聚vs选择效应 |
| **Rosenthal, S. & Strange, W. (2004).** *Evidence on the nature and sources of agglomeration economies.* Handbook. | 集聚经济来源 | sharing/matching/learning |
| **Hidalgo, C. et al. (2007).** *The product space conditions the development of nations.* Science. | 产品空间与经济发展 | 产业结构演变 |
| **Batty, M. (2013).** *The new science of cities.* MIT Press. | 城市科学的系统视角 | 多时间尺度理论基础 |

---

## 总结：文献定位

你提出的 **Intervention-aware Urban Representation Learning** 位于以下交叉点：

1. **Urban Representation Learning** (zone embedding)：现有工作主要学习“空间长什么样”。
2. **Causal ML + Policy Evaluation** (DiD / SCM / Matrix Completion)：因果推断通常独立于表征学习。
3. **TOD Impact** (地铁站效应)：现有研究多依赖手工特征，而非学习到的 embedding。

你的核心创新：**学到的是"空间对policy如何响应"**，即embedding空间的相似度 = 干预响应路径的相似度。这实质上是将 contrastive learning 的相似度定义从视觉/语义相似重构为**因果响应相似**。

最需要密切关注的竞品工作：
- Mai et al. (2024) CSP — geospatial contrastive pre-training (对比目标不同)
- Naik et al. (2017) — street view time series for physical change (无causal embedding)
- Zhang et al. (2022) — spatio-temporal contrastive learning (需区分因果响应 vs 时序预测)
- Cong et al. (2023) SatCLIP — location embeddings from satellite (无policy conditioning)
