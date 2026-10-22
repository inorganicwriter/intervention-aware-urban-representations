# 项目介绍图表：数据口径与展示约定

当前交付为英文 15 分钟汇报，21 页主讲内容与 5 页参考文献。正文沿用 Beamer 默认字体；图表、流程图和图注使用 Times New Roman。定量图以冻结因果汇总表及既有服务器表示学习摘要为依据，保留估计值、区间、样本、变换尺度、标记和面板顺序。Python/matplotlib 导出 PDF、可编辑文字 SVG 与 300 dpi PNG；英文资产位于 figures/en，中文资产保留。项目根目录 seminar_materials 汇总当前 PDF、21页讲稿和18张图的三种格式，图表索引标出9张正文图。

英文绘图复用现有数据与绘图逻辑，仅转换文字和字体。中英文 intro_chart_audit.json 完全一致；正文使用的9张图经 PDF 字体资源检查，均嵌入 TimesNewRoman 字体。英文 source/PDF/SVG 均通过中文残留检查。静态审查器只接受 sans-serif，因此其字体规则以用户指定的 Times New Roman 覆盖，并通过实际嵌入字体验证；300 dpi PNG 为汇报预览，最终 PDF 使用矢量图，无需投稿 TIFF 或期刊栏宽。

## 图表契约

| 图 | 需要支持的判断 | 数据、口径与布局 |
|---|---|---|
| 年份分布 | 样本覆盖多个开通年份 | 处理清单全部5048行按 opening_month 年份计数；单柱图，移除重复累计曲线 |
| 可用性 | 不同方法/结果的覆盖不同 | 全部4族×6方法/对照范围；matching=availability_treated_events，GSC/MC=max_supported_treated_grids；数值不能相加，不能解释为final_training_mask人数 |
| 同城支持 | 同城样本支持存在显著差异 | 沿用上列计数，4族分行、3方法并列点图；零支持标为无数据，不取对数零值 |
| 预趋势 | 不同检验回答不同零假设 | 先固定 canonical outcome，再按 joint_wald_pretrend 与 cluster_mean_pre_gap_zero 分面；p轴0–1，标0.05参考线但不设“通过”区域 |
| POI诊断 | 四outcome的pre-gap诊断不一致 | GSC/MC四outcome完整保留；matching只有count的总体Wald结果，另页给出，绝不补造另外三个结果 |
| 事件研究 | 已发布的总体曲线及支持边界 | 同城、预注册窗口：月频-24..24，年频-6..3；固定canonical outcome，逐方法独立坐标；城市簇>=3且区间有效时展示原95%区间，不足3簇的点保留为叉号、区间不作推断解释 |
| 历史检索 | 单划分下尚无稳定表示优势 | 两族分面点图，所有五类摘要保留；AE为范围，监督模型为3 seeds均值±SD，raw/chance无误差条；不是修正后CV结果，不是置信区间 |
| 城市级最小对照 | 处理前特征对冻结响应标签具有弱的城市外预测信号 | 使用用户提供的服务器5折汇总；3个点图分别显示原始形状检索vs随机、ridge RMSE vs训练均值、ridge方向vs训练多数符号；保留全部12个数值，不推算CI或正式R² |
| 权重曲线 | 序数设计分数影响相对权重 | 函数示意而非实测：exp(-gamma D)与p^alpha；明确归一化前的分量，不解释为因果正确概率 |
| 流程图 | 信息流、方法路由、训练/推断边界 | 概念示意；不画虚构实验点；默认与可选输入分开；标签只连损失，不直接连encoder输入 |
| 决策图 | 下一步实验区分解释 | 所有分支箭头连接实际节点；“未发现增量”不等于“无可预测信号” |

## 历史检索摘要的来源

`data/representation_history.csv` 原样迁移既有介绍绘图脚本中的历史摘要数值；真实训练产物在服务器，本地没有重新提取或核验seed级文件。AE中心和上下界沿用既有汇总，seed数未知，故不补写seed数。`seed_sd`的low/high字段存SD，`range`字段存区间端点。图上显式区分两类误差条，均非CI。

## 过滤与信息保留

- 因果曲线选择同城和canonical outcome是既定正文比较；跨城范围和其他POI outcome不在该曲线中混合。可用性图仍保留全部范围，POI诊断保留四outcome。
- 事件研究保留规范窗口内全部估计点，不按显著性筛选、不抽样。参考期按manifest的h=-1归一化为0：匹配表省略该行，仅在展示层补空心点；GSC/MC已有参考行也统一画空心点，不赋予CI或估计含义。真正缺期不连线；簇不足时保留点并标注，不画有推断含义的CI。冻结源表与下游标签不变。
- 每次执行导出`intro_chart_audit.json`，记录各图字段映射、输入与筛选行数、缺失与簇不足数量；静态检查与实际PDF页面检查均需通过。

## 本次结构修正

GSC是对照全时段未处理观测估计共同因子、处理单元pre观测估计载荷；MC低秩残差之外有双向固定效应；生产标签按cascade路由、每个网格×结果族使用被分配方法。总体TWFE、逐网格标签及下游迁移证据分别表述。GPT-image提示词只用于概念图，定量图继续使用数值源表。

## Seminar 正文与验证状态

按用户要求恢复原稿Background全部13页和Related Work全部5页。主线为Background → Related Work → 方法与资产 → 表示学习 → 结果与发现 → 展望；取消附录章节。关键内容按所属问题进入正文：TWFE与标签区别、权重构造和恒定horizon权重的抵消进入方法；检索与不确定性进入评估；住房、人口、预趋势及POI四outcome诊断进入结果；标签结构与嵌入诊断接在表示结果之后。覆盖图已保留同城支持数量，不再为相同计数重复占页。保留28条原参考文献；个别表述与代码或原文不符的地方修正，例如GSC/MC级联、MC共享低秩结构以及跨城市交通文献的范围。

静态图表预检：零FAIL。TIFF/600dpi/89或183mm宽度提示属于投稿规范，本次为宽屏演示，使用300dpi PNG与矢量PDF/SVG。内置LaTeX编译器仍返回平台标准目录错误；使用本机现有XeLaTeX编译同一源文件。恢复结构的版本共48页（含封面正文44页，参考文献4页），无Overfull、缺字或字体警告。独立图数据值与冻结因果资产不变，历史表示结果仍按原有服务器摘要报告。

## 城市对照与英文讲稿更新（2026-10-04）

新增`representation_city_controls.csv`保存用户提供的服务器最小对照摘要，绘制`intro_city_controls`三项指标图。形状检索原始/随机：VIIRS 0.0509/0.0085，POI 0.0595/0.0144；ridge/训练均值RMSE：0.3627/0.3748、0.4442/0.4642；ridge/训练多数符号方向：55.7%/54.5%、56.9%/51.9%。来源为对话中的服务器汇总，本地没有逐折产物、逐城支持或原始CI，图上保留汇总精度，不绘制虚构误差条。主叙事为弱的城市外预测信号；kNN对照暂未提供额外收益。神经旧划分与新城市CV结果分别展示，不跨协议直接比较数值，不将RMSE比值称为解释方差。

融合页描述锁定的评估设计：fit训练、val选α与块版本/λ、fit+val重拟合、outer评价；完整候选池的逐anchor增量按城市聚合，另报告排除同城候选及逐outcome结果。融合指标正在统一口径，尚无可发布的融合结果。本次不修改实验代码、标签或服务器产物。

中文PDF新增城市对照页，正文连同封面共45页，参考文献4页，共49页。英文口头讲稿单独保存为`project_intro_speaker_notes_en.md`，按这版PDF页码逐页对应，包含固定术语表与20–30分钟的讲述节奏。

Related Work 前移至背景与研究问题之后：背景为第2–9页，Related Work为第10–14页，研究框架为第15–19页。原背景中的工具比较、核心思路、学习任务、流程图及贡献页全部保留并归入研究框架。文献页围绕既有工作的重点与本项目的连接组织，定位表按研究方向比较。英文讲稿同步分节、移动页码并重写Related Work的口头衔接；第20页以后的页码保持不变，总计49页。

最终按论文式叙事整合：Introduction为第2–14页，融合现实动机、研究问题、研究缺口、核心思路、学习任务、整体流程与贡献；随后Related Work为第15–19页，第20页直接进入方法。取消独立Background与研究框架分节，保留原有内容并重写衔接。工具比较页改为“研究缺口：从反事实估计到响应迁移”，说明估计结果怎样连接监督任务；讲稿也按同一论证链衔接Intro、Related Work和方法。最终总页数仍为49页。


## 问题定义集中与已有结果量化（2026-10-04）

当前结构为 Introduction（2–10）→ Related Work（11–15）→ 问题定义与方法概览（16–19）→ 因果方法（20–29）→ 表示学习（30–33）→ 结果（34–43）。问题定义、核心思路、研究任务和总流程集中在 Related Work 后。原有背景与文献内容保留，总计49页；英文讲稿逐页重排并更新过渡。

VIIRS 同城匹配 6/12/18/24 月估计为0.0446/0.0500/0.0575/0.0705，43城市，24月城市层面95%区间[0.0475,0.0935]；四个展示期支持2255–2264单元。POI数量第1/2/3年为0.0803/0.1063/0.1132，44城市，第3年区间[0.0701,0.1564]；各年支持2476/2242/1947。数值直接来自冻结 event_study_panel.csv 的同城 matching 行，保留变换尺度，不转换为百分比效应，也不将支持变化下的估计增幅解释为同一固定样本的增长。

结果汇总增加已有服务器报告的成对响应相似度矩阵折半信度（Spearman–Brown修正）：matched VIIRS/POI为0.806/0.285，GSC为0.887/0.156；其来源为用户此前给出的诊断简报。城市CV的ridge相对RMSE下降为1−ridge RMSE/训练均值基线RMSE，约3.2%/4.3%，不是R²。历史神经结果仍按旧划分单独展示。融合数值尚未提供，保留为下一步实验设计，不生成实验数值或误差区间。


## 本地论文稿对齐（2026-10-04）

标题以 manuscript_draft_intro_related_work_v0.md 与 project_abstract.md 的共同标题为准：Learning How Cities Respond: Intervention-Conditioned Urban Representations；中文为“城市如何响应？干预条件下的城市表示学习”。封面保留中文并加入正式英文标题，英文讲稿同步。Introduction 按“因果评估与城市表示的连接→可追溯响应证据→城市外评价”组织；Related Work 保留因果估计基础，补入交通案例、城市表示/迁移以及因果方法与城市学习的连接。问题定义与任务仍在第16–19页，实测结果不变。

正式术语：response signature 是跨结果与时间的估计效应向量，response profile 是单结果动态路径，causal response supervision 是监督机制。intervention-conditioned 表示学习目标围绕站点开通响应构建，当前不新增干预状态输入；早期 abstract 的多模态设想不作为已完成实现或实测成果。

新增核心文献的原文/作者来源：Urban2Vec https://ojs.aaai.org/index.php/AAAI/article/view/5450；RegionDCL https://zhengwang125.github.io/paper/RegionDCL.pdf；FLORAL https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/Transfer_Knowledge_between_cities_Zheng.pdf；Weather2vec https://ojs.aaai.org/index.php/AAAI/article/view/26696；CauOD https://fi.ee.tsinghua.edu.cn/~dingjingtao/papers/CauOD-CIKM22.pdf；CGNet-PIE https://journals.sagepub.com/doi/pdf/10.1177/27541231251318348；Ahuja https://proceedings.mlr.press/v202/ahuja23a.html。交通案例参照出版社页面、作者发表清单及本地稿件的已核对引用。展示内容仅概述研究对象和方法，不复制文献中的效果数值或宣称本项目已优于这些方法。

对齐后已编译并检查封面、第9–19页及全部参考文献页；共50页，正文连封面45页，参考文献5页。讲稿对应45页正文，约3600个英文词，页名匹配；无超过30词的句子。编译日志无Overfull、缺字或字体警告。


## 论证顺序重整与表示结果判读（2026-10-04）

正文改为40页（含封面）：Intro6页、Related Work5页、问题与数据4页、因果方法7页、因果证据7页、响应学习与结果8页、结论与下一步2页。机制和异质性、规划问题和缺口、问题定义和任务、设计权重和分数分别合并；样本覆盖图移至数据，冻结发布并入路由。背景与Related Work的主要内容和因果/表示关键技术均在正文，无附录。

叙事由“全部方法→全部结果”改为“因果构建→因果证据→学习对象/评价/模型→学习结果”。学习结果显式分开城市CV直接预测与历史神经嵌入：默认监督对raw均值Δ为VIIRS +0.0114、POI −0.0134，直接从 representation_history.csv 计算；原始数值、误差条、支持和来源不变。不将不同协议比较，不将ridge增益表述为神经表示增益。新增 project_intro_outline.md 记录各段论证任务、可展示数值和待检验结论；英文讲稿同序重排。


## 表示学习常用结果图（2026-10-05）

契约：性能柱图使用汇总表全部10行；消融图选两族默认与queue-off共4行，raw作为固定参考。比较模型检索得分及相对raw的增量。两图为quantitative grid，Python/matplotlib、Times New Roman，输出矢量PDF/可编辑SVG/300 dpi PNG。柱图纵轴均从0开始，分族采用独立尺度；增量图横轴共用绝对尺度与零线。

增量定义为supervised mean−raw mean；固定raw被减去后，已有seed SD保持不变。VIIRS为+0.0114/+0.0214，POI为−0.0134/−0.0083；SD分别为0.0067/0.0184、0.0025/0.0174。AE的误差条是先前报告范围，非seed SD。所有值沿用representation_history.csv；不估计CI、p值或逐seed值，不绘制合成嵌入。不以seed SD区间是否跨零判显著性。静态字体规则的Times New Roman例外沿用用户指定；汇报宽度与300 dpi预览例外不适用期刊投稿要求。
