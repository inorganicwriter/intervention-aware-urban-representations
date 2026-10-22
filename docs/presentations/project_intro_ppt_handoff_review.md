# 手动制作英文 PPT：逻辑审阅与页面建议

审阅日期：2026-10-05。已阅读现有 26 页 PDF，并检查全部页面的图形排版。当前主讲内容 21 页、参考文献 5 页；15 分钟英文讲稿约 1,431 词。本文件为重做 PPT 的建议，现有 PDF 保持原样。

## 总体判断

主线已经顺畅：现实问题 → 相关工作 → 输入与任务 → 因果方法 → 动态结果 → 响应学习 → 实验结果 → 结论。

需要加强的是听众的理解过程：先让他们看到一个响应签名，再解释怎样估计它；说明标签如何进入学习；最后将原始特征的信息与神经嵌入的收益分别讲清楚。现有正文适合作为内容底稿，手动 PPT 建议压为 18 页，每页承担一个明确任务。

贯穿汇报的一句话：**将站点开通的动态响应组织成学习目标，检验处理前城市特征及其嵌入能否支持未见城市的响应比较。**

## 优先调整的内容

| 优先级 | 当前页码 | 现状与可能的理解障碍 | 手动 PPT 的调整 |
|---|---|---|---|
| 必做 | 5–6 | response signature 先以符号出现，听众到14页才看到形状定义 | 在问题/总流程页给一个网格、一条时间轴和响应向量示意。展示 VIIRS 的6个时间点；保留“处理前特征进入编码器，估计响应提供监督”。示意图注明 conceptual，数值例子用真实标签 |
| 必做 | 7–10 | 路由很清楚，匹配对照如何形成、为何可作反事实讲得较少 | 匹配页优先讲处理/对照、处理前匹配依据、对照资格与事件窗口；具体规则按项目已实现设计填入。用时间轴突出−18至−7基线、提前期与post窗口，再讲公式 |
| 必做 | 18–20 | 都使用nn_corr@5，但18页为城市CV，19–20页为历史划分，raw数值差很多 | 图上方直接标“5-fold city CV: features and ridge”与“Earlier city split: trained embeddings”。把协议区别放在主要视觉位置。两组数值分别用于各自的基线比较 |
| 必做 | 19–20 | 柱图与增量图复用同一批汇总结果；queue的含义只在模型脚注出现 | 合为一页主结果：性能对照为主图，默认增量为醒目标注；queue-off作为小区域或一次动画补充。若保留queue对照，先用一句话解释FIFO队列：扩展训练中的跨batch候选样本 |
| 必做 | 21 | 结论页的新融合公式占据主要区域，听众可能把尚待评估的方案当成已完成成果 | 先给三张发现卡片，再用一句话讲下一步“Preserve raw geometry and add predicted responses”。融合公式留给口头技术讨论 |
| 建议 | 3–4 | 文献和方法名称多，15分钟容易变成点名式综述 | 合为一页三列：Transport evaluation / Urban representations / Our connection。每列2–3个代表引用，保留Introduction后接Related Work的顺序 |
| 建议 | 11–13 | 图较小，方法支持与统计细节大多在长脚注中 | 加大图，将本轮主要讲述的matching面板作为视觉重点。保留GSC/MC支持信息；5%与10%的诊断差异、城市/网格聚类差异用短注释，不用绿色“通过”标记 |
| 建议 | 14–16 | 目标、模型、损失连续三页；optional组件与当前实测模型混在脚注中 | 模型与损失合为一页：上方编码器，下方“shape supervision”和“response prediction”。当前主实验突出tabular输入；可选图像/条件化组件用于讨论时说明 |

## 技术与结果表述应保持的口径

- **统计单位**：设计清单有5,048行、44城市，station_event_id有5,048个唯一值。可称5,048个站点开通事件。它们映射到网格，不能直接等同于5,048个唯一网格或有效学习标签。
- **两个响应对象**：总体事件研究系数β描述支持样本的动态模式；学习标签r是逐网格响应估计。两者在图示中使用不同符号。
- **预趋势**：在5%水平下，城市聚类matching检验对四族均未拒绝；住房、VIIRS、POI数量在10%水平下拒绝。GSC/MC检验平均pre-gap，与matching联合Wald不同。建议直接显示检验、p值和支持，口头结合识别假设解释。
- **不同方法曲线**：各方法的支持样本和纵轴有差异，曲线展示各自样本的结果；不把它们表述为同样本三方法的稳健性一致结论。
- **目标一致性**：VIIRS的拆分相似度结构较一致；POI合成目标较弱，可能涉及outcome异质性及噪声。该指标描述目标结构，因果有效性由估计设计与诊断说明。
- **两个实验协议**：第18页是5折城市CV的特征/直接预测对照；第19–20页是历史城市划分的神经表示。当前没有同协议神经城市CV结果可展示。
- **误差条**：监督模型使用3个seed的SD，AE使用此前报告的范围。把这条说明放在图边上；不从是否跨零判断统计显著性。
- **指标单位**：nn_corr@5是相关得分，增量用+0.0114等绝对数值；方向准确率的差异可用percentage points。RMSE相对下降用3.2%/4.3%，不称正式R²。
- **已训练表示**：VIIRS默认相对raw均值增量+0.0114，queue-off为+0.0214；POI分别−0.0134/−0.0083。可表述初步VIIRS收益，保持POI的相反结果可见。
- **queue-off**：观察到均值更高、seed波动更大；只报告该对照结果，具体机制与稳定性继续检验。
- **当前模型**：处理前表格特征进入编码器，估计的开通响应定义监督。intervention-conditioned在这里围绕固定的站点开通干预和响应目标使用。

## 推荐的18页主讲结构

| PPT页 | 建议英文标题 | 当前PDF来源 | 这一页应让听众记住什么 | 建议画面 |
|---|---|---|---|---|
| 1 | Learning How Cities Respond | 1 | 研究站点开通响应与城市学习 | 简洁标题，副标题和汇报人 |
| 2 | The Same Intervention Can Lead to Different Responses | 2 | 地方条件使响应异质 | 一张站点/街区图＋三条作用机制 |
| 3 | Connecting Transport Evaluation and Urban Learning | 3–4 | 两类研究在响应监督处连接 | 三列研究定位，少量代表引用 |
| 4 | Can Features Reveal Similar Responses in New Cities? | 5 | X→embedding，响应标签定义学习任务 | 两条路径的总流程＋响应向量示意 |
| 5 | Events, Grids and Response Windows | 6 | 44城、5048事件、6期VIIRS/12-cell POI | 样本数字＋时间轴；年份柱图可缩小 |
| 6 | Building Traceable Response Labels | 7 | 支持驱动路由，每族每事件一种方法 | 简化路由＋“response/mask/method”输出 |
| 7 | Matching Constructs a Local Counterfactual | 8 | 差分设计、对照与清洁基线 | 处理/对照示意＋时间轴＋一个公式 |
| 8 | GSC and MC Use Shared Panel Structure | 9 | 两种反事实模型的共同点与差异 | 并排两模型，公式简短 |
| 9 | Event Studies Summarize Dynamic Patterns | 10 | pooled β与grid r分别用于展示和监督 | 动态系数公式＋两对象关系 |
| 10 | Pre-treatment Diagnostics Depend on Method and Support | 11 | 看检验、拟合与支持 | 放大诊断图，强调城市层面和阈值 |
| 11 | Nighttime Lights Show a Positive Matched Response | 12 | 支持样本下的持续正向路径 | 放大matching，标6/24月效应与CI |
| 12 | POI Count Also Rises After Opening | 13 | 商业活动代理的正向动态响应 | 放大matching，标3年估计和支持变化 |
| 13 | Response Shape Defines the Learning Target | 14 | 形状、幅度、方向分别评价 | 响应形状示意＋小型一致性表 |
| 14 | Responses Supervise the Embedding | 15–16 | X编码，响应约束几何与预测 | 模型图＋三项损失作用，权重小字标出 |
| 15 | Evaluation Separates Shape, Magnitude and Direction | 17 | 什么增量、相对谁、在哪个协议 | 三任务表＋两套协议标识 |
| 16 | Features Carry a Small Cross-city Response Signal | 18 | 城市外可预测成分存在 | 三指标图，醒目标出方向+1.2/+5.0pp |
| 17 | Supervised Retrieval Gains Vary by Outcome Family | 19–20 | VIIRS初步增益，POI raw更强 | 主要性能图＋默认Δ；queue补充区域 |
| 18 | From Response Evidence to Cross-city Comparison | 21 | 已有证据和下一步如何连接 | 三条发现＋一句融合检验，避免新公式占主位 |

参考文献保留在文件末尾供查阅，不逐页讲述。这里的合并保留背景、Related Work和两部分技术主线；无需新增附录。

## 15分钟节奏

| 内容 | 新PPT页 | 目标时间 |
|---|---|---|
| 开场、背景、Related Work与问题 | 1–4 | 2.5分钟 |
| 数据与因果方法、诊断 | 5–10 | 4分钟 |
| VIIRS与POI因果结果 | 11–12 | 2分钟 |
| 学习目标、模型和评价 | 13–15 | 2.5分钟 |
| 两类学习结果 | 16–17 | 2分钟 |
| 结论 | 18 | 0.75分钟 |
| 停顿、翻页、看图余量 | — | 1.25分钟 |

可在现有约1430词讲稿基础上压至约1200–1300词，为新示意图的解释留时间。若15分钟含提问，需要进一步缩短主讲。

## 手动制作时的视觉原则

- 每页用“主张型标题”，并在图边放一句结论；研究任务页使用明确的问题标题。
- 一页一个主图。支撑数字和统计定义贴近图，较长方法说明进入讲稿。
- 正文建议24pt左右，图内刻度/图例以投影可读为准，通常争取18pt左右；图表字体继续Times New Roman，PPT正文沿用模板。
- 颜色沿用原图：Matching深蓝、GSC青绿、MC橙色；模型对比的颜色沿用模型定义，不在同张图中改变含义。
- 复制图时保留坐标轴、单位、样本量和误差条定义。不同尺度的VIIRS/POI柱图中，继续比较同族模型。
- 所有实测图使用项目现有矢量或高分辨率资产；概念示意明确标Illustration，避免合成数值被误读为实验数据。

## 结束时建议留下的三句话

1. We organize station-opening effects into traceable response signatures.
2. Pre-treatment features carry a small predictive signal across held-out cities.
3. Supervised embeddings show an initial VIIRS gain, while POI favors raw features.

这三句分别对应因果资产、直接预测、已训练表示，完整覆盖当前已有证据。
