# project_intro：GPT-image 流程图提示词

2026-10-05更新：以下保留历史中文提示词。当前21页英文汇报的实际英文提示词和已生成的三张PNG位于 `seminar_materials/gpt_image_flowcharts/`，入口为 `prompts_en.md`。当前英文LaTeX未自动接入这些PNG，原PDF保持原样。

前三组提示词对应当前中文稿中的三个流程图替换槽；第四组为展望图备选，本版不单独占页。只用于流程示意，不用生成式模型重画效应曲线、p 值、历史检索或误差条。定量图的 PDF / SVG / PNG 由项目绘图脚本生成。

## 通用要求与使用方式

- 16:9 横版，白底，扁平矢量式外观，无照片、无立体材质、无装饰性城市背景。深蓝表示数据与模型，青绿表示因果响应与监督，少量橙色表示决策；避免红绿表示“识别通过/失败”。
- 无需画页标题、页脚和文献。节点文字少而大，箭头有明确起止，留足边距。中文标签保持原样，不添加不存在的结果、变量、公式或数值。
- 若新模板已有配色，先将下面的色值替换成模板配色。流程图的结构约束应保留。
- 推荐先用“无文字版”：将每条提示词最后加上第 5 节附加指令，后期在 PowerPoint 中覆盖中文标签。GPT-image 输出是位图；矢量式外观并不代表可编辑 SVG。保留无字底图和文本层有利于换模板。
- 带字版务必人工复核中文和箭头。已在 LaTeX 中预留自动替换路径；如果生成图仍有错误，先不要放入替换槽。

| 图 | 对应页 | 审核后保存路径（项目根目录起） |
|---|---|---|
| 项目总流程 | 项目流程 | `docs/presentations/figures/intro_pipeline_gpt.png` |
| 因果标签路由 | 生产标签采用级联路由 | `docs/presentations/figures/intro_routing_gpt.png` |
| 表示训练结构 | 当前表示模型：标签只用于训练监督 | `docs/presentations/figures/intro_model_gpt.png` |
| 实验决策 | 展望备选；本版不自动接入 | `docs/presentations/figures/intro_decision_gpt.png` |

## 1. 项目总流程：因果估计与表示学习的两条信息流

直接粘贴：

```text
Create a publication-quality scientific workflow diagram for a Chinese research presentation, landscape 16:9, white background, clean flat vector-like style. Use navy #234E70 for data/model nodes, teal #168A85 for causal estimation/supervision nodes, pale fills, thin consistent arrows, rounded rectangles, generous whitespace. Large legible Chinese text, no title, no footer, no photos, no 3D graphics.

Arrange six nodes in two lanes. At far left, a shared source node centered vertically: “站点开通与网格面板” with a smaller second line “5,048 事件 · 44 城市”. It branches upward to “处理前特征 X” / “训练城市内标准化”, and downward to “因果响应估计” / “诊断 · 权重 · 冻结”. The top lane continues right to “表格编码器” / “归一化嵌入 e”. The lower lane continues right to “响应监督” / “形状相似度 + 预测”. Draw one downward arrow from the embedding node to the supervision node labeled “训练时计算损失”. The embedding node also connects right to a final node “未见城市评估” / “相对 raw / 外观基线”. Under that final node place a short annotation “推断仅需要 X”.

Scientific constraints: estimated response labels enter training losses only; there must be NO arrow from response labels to the encoder input. Do not imply labels are observed causal ground truth. Do not imply supervised representation already outperforms raw features. Do not place a generic causal estimator after the encoder. Do not add invented numerical results. Arrow endpoints must meet node boundaries. Render exactly these Chinese labels.
```

## 2. 因果标签：精确级联顺序与方法分配

直接粘贴：

```text
Create a precise scientific cascade-routing diagram in landscape 16:9 for a Chinese causal-inference presentation. White background, flat vector-like style, navy #234E70 and teal #168A85, pale fills, generous whitespace, large Chinese typography. No title/footer, no decorative imagery, no fake data plots.

The exact ordered sequence is:
“同城匹配” → “同城 GSC” → “同城 MC” → “跨城匹配” → “跨城 GSC” → “跨城 MC” → “跳过”.
Lay out seven equally sized nodes in a readable snake layout: the first four left-to-right on the upper row; then turn downward at the fourth node, and the last three continue right-to-left on the lower row. All arrows in this sequence mean “不可用则继续”. Keep this label in one shared legend instead of repeating it in tiny text.

Along the bottom, show a separate teal output bar “保留该方法的响应标签、掩码与诊断”. Each of the six estimator nodes has a short teal branch toward this shared output, with a common legend “可用即保留，停止后续尝试”. Route these branches through a clear collector line outside the cascade; do not cross or obscure the main sequence. The final “跳过” node has NO retention arrow. Add a short top annotation “每个网格 × 结果族分配一种方法”.

Scientific constraints: this is a sequential assignment, not an ensemble, not majority voting, not six parallel estimates on every grid. Do not add a p>0.05 decision rule. Availability follows the pipeline's existing support/design rules, not visual significance selection. Do not suggest GSC or MC always improves a failed matched estimate. If retention branches make the diagram too crowded, remove the branches and keep one clear annotation “任一步可用即保留；否则按箭头继续”, while preserving the exact seven-node sequence.
```

## 3. 表示学习：训练损失与推断边界

直接粘贴：

```text
Create a clear scientific neural-model workflow diagram, landscape 16:9, white background, flat vector-like appearance. Chinese presentation for an urban-studies and causal-inference audience. Use navy #234E70 for model/data nodes and teal #168A85 for frozen labels/loss nodes; use pale fills and large readable labels. No title/footer, no photos, no 3D, no fictitious performance claims.

Top row, left to right, four nodes:
1 “处理前表格 X” / “年度特征 · 1–3 年 lag”
2 “MLP 编码器” / “256 → 128”
3 “投影头” / “128 维 · L2 归一化”
4 “嵌入 e” / “用于检索”
Connect them sequentially with navy arrows. Add a small navy branch from “嵌入 e” to a separate “按结果族预测头” node.

Lower row, clearly labeled as training-only supervision:
A “冻结响应 r” / “有效掩码 · 设计权重 · SE”
B “成对响应相似度”
C “对齐损失 + 距离损失”
D “逐 cell 预测损失”
Arrows A→B→C. A also connects to D via a clearly routed separate line. The embedding e connects down to C. The prediction-head node connects down to D. Both loss nodes feed a compact total-loss box “0.25 对齐 + 0.25 距离 + 0.50 预测”.

Use a divider to show that lower-row labels/losses are needed during training, while inference uses only the top-row feature-to-embedding path. Add the concise annotation “未见城市推断仅需要 X”. Optional image or station/year conditioning branches should be omitted from this main figure; they are not enabled in all historical experiments.

Scientific constraints: labels supervise losses, NEVER feed the encoder as input; prediction heads read the final normalized embedding; do not draw a separate direct feature-to-response bypass; there is no causal graph discovery, no counterfactual image generator, no data augmentation in this workflow. Do not label the target as causal ground truth. Do not invent exact predictor output dimensions for a combined dataset. Preserve the exact loss coefficients and arrow logic.
```

## 4. 最小对照之后：按证据分支

直接粘贴：

```text
Create a compact evidence-based experiment decision flowchart, landscape 16:9, white background, clean vector-like style. Navy #234E70 process boxes, a muted orange #D28436 decision diamond, teal #168A85 next-step boxes. Large Chinese text, short labels, clear arrow endpoints, no title/footer or ornamental graphics.

At left, process box “城市 CV 最小对照” / “raw → ridge / 小型回归”. Connect to central decision diamond “城市外预测是否出现稳定增量？”.
The upper branch labeled “有” connects to a right-side process box “检查表示是否保留信号” / “直接回归 vs 当前表示 vs raw+响应块”.
The lower branch labeled “未发现” connects to a right-side process box “审查支持、标签与特征” / “补充时点安全变量，再评估边界”.
At the bottom, a short independent annotation “未发现增量 ≠ 已证明无信号”.

Scientific constraints: these are planned evidence-dependent decisions, not completed positive results. A null result must not be labeled “数据不可学” or “表示失败已证明”. Do not imply that adding variables guarantees an improvement. Do not invent p-values, accuracies or confidence intervals. All branches must terminate at the named process boxes.
```

## 5. 无文字版附加指令与标签映射

在任意一条提示词最后追加：

```text
Override text rendering: produce a text-free diagram. Do not render Chinese, English, letters, numbers or pseudo-text anywhere. Keep all node rectangles, lane positions, arrow routing, colors and whitespace exactly as described. Leave generous blank regions inside each node and beside decision arrows for manually adding editable labels afterward. Use the node ordering in the description as the placement map; do not add placeholder glyphs or node IDs.
```

标签映射直接沿用各提示词列出的节点和顺序。总流程是六节点；级联是七节点加一个保留输出；模型是四主节点加预测头、四监督节点与总损失；决策是左侧对照、中央问题、右上/右下动作。不要让无字图中的形状数或箭头方向与映射不一致。

## 6. 放入替换槽前的检查

检查四点：标签不进 encoder；路由顺序完整；所有分支连接真实节点；没有新增成功结论或数值。前三张图放入表中的 PNG 路径后重新编译 `project_intro.tex`，会替换对应后备 PDF 流程图。第四张为备选，须明确放入某一页后才会展示。新模板若调整图的占比，请先重新检查投影时的字号。
