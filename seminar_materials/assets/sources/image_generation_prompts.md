# 图像生成提示词与历史记录

整理日期：2026-10-06。以下保留生成与修正过程；原文件名和历史页码用于追溯，当前文件及最终 PPT 页码以 ../README.md 为准。已替换的简单文字流程图、早期 Matching 图和早期 problem setup 图已清理。

## 保留图片路径对照

| 原文件 | 当前文件 |
|---|---|
| concept_figures_v8/gsc_mc_intuition_v8.png | diagrams/gsc_mc_intuition.png |
| concept_figures_v8/problem_setup_v8_corrected.png | diagrams/problem_setup.png |
| concept_figures_v8/related_work_bridge_v8.png | diagrams/related_work.png |
| concept_figures_v8/takeaways_visual_v8.png | diagrams/takeaways.png |
| concept_figures_v9/matching_parallel_trends_v9.png | diagrams/matching_parallel_trends.png |
| gpt_image_flowcharts/intro_model_gpt_illustrated_en.png | diagrams/model_architecture.png |
| gpt_image_flowcharts/intro_pipeline_gpt_illustrated_en.png | diagrams/full_pipeline.png |
| gpt_image_flowcharts/intro_routing_gpt_icons_v3.png | diagrams/causal_routing.png |

---

## 原记录：background_images/prompts_and_usage.md

# Background visuals for the 8-minute seminar

Generated with the built-in GPT-image tool. The diagrams are conceptual illustrations; the curves contain no empirical values.

## Used on slide 2

- `rail_expansion_tod.png`: station opening, access, and potential urban response channels.
- `local_conditions_response_paths.png`: contrasting local conditions and possible response shapes.

## Alternative scene illustrations

- `station_opening.png`
- `access_to_jobs_services.png`
- `housing_local_businesses.png`

## Diagram prompts

### 1

Use case: scientific-educational. Create a clear icon-rich conceptual diagram for an English urban research seminar. Portrait-ish 3:4 image, pure white background, clean Times New Roman style serif text, navy outlines, muted teal and warm orange accents, refined urban illustration, no title at top because PowerPoint supplies it. Readable at 390 CSS pixels wide by 260 pixels high, use few large labels. Three vertically stacked linked stages with arrows pointing down. Top: a metro entrance and train icon, exact label "Station opening". Middle: station connected to office building, clinic and homes, exact label "Access to jobs and services". Bottom: three small tangible icons, apartment building, cafe storefront, walking residents; exact shared label "Housing, businesses and population". The arrows represent potential mechanisms, not measured causal effects. This is an explanatory mechanism diagram of rail expansion and transit-oriented development. Avoid charts, numbers, empirical claims, photographs, logos, floating UI cards, pill badges and decorative textures. Prioritize a immediately readable causal story with bold icons and readable large labels. Use a landscape 3:2 overall canvas if that improves readability, with the three stages still stacked vertically.

### 2

Use case: scientific-educational. Create a clear icon-rich conceptual diagram explaining why the same station opening can produce different local responses. Landscape 3:2 overall composition, pure white background, crisp navy outlines, muted teal and warm orange, Times New Roman style serif labels large enough to read at 390 CSS pixels wide. No title at top because PowerPoint supplies it. A metro station icon at top center labelled exactly "Same opening" branches downward with two arrows to TWO contrasting neighborhood illustrations side by side: on left dense mid-rise buildings, mixed-use shops, existing transit connections labelled "City center"; on right lower-density homes, open land, sparse connections labelled "City edge". Below these two neighborhoods put one shared concise condition line "Land use, network links, local demand". Then two arrows lead down to two tiny visibly different response trajectories, labelled together "Different response paths". These are schematic squiggles only with NO numeric axes, ticks, confidence bands, effect size labels, clusters or measurements. Do not imply city center always has a higher effect than city edge: different curves illustrate possible shapes only, neither labeled larger, better or positive. Bottom tiny note "Conceptual illustration". This diagram explains local conditions and response heterogeneity. No photographs, no logos, no extra wording, no UI cards or badges. All labels must be exact and legible; keep composition open and visually balanced.

## Scene prompts

### 1

Use case: scientific-educational. An attractive urban research seminar illustration. Wide 4:3 composition, pure white backdrop, professional hand-drawn architectural editorial illustration with clean navy outlines, gentle pale blue, muted teal and restrained warm orange highlights, no text, no labels, no numbers, no logos, no watermark. Balanced full composition with generous white margins. Clear enough to display at 250 CSS pixels wide. Conceptual illustration only, no data, no numeric axes or scientific results. Keep buildings recognizable and pedestrian scale realistic. Show a newly opened urban metro station entrance in a neighborhood: elegant simple entrance canopy and stairway, a small metro train visible on a subtle cutaway track, a few people entering and walking through the plaza, adjacent mid-rise blocks. Convey the real-world intervention of opening a station. Focus on the station and its immediate surroundings, without crowds or dense decoration.

### 2

Use case: scientific-educational. An attractive urban research seminar illustration. Wide 4:3 composition, pure white backdrop, professional hand-drawn architectural editorial illustration with clean navy outlines, gentle pale blue, muted teal and restrained warm orange highlights, no text, no labels, no numbers, no logos, no watermark. Balanced full composition with generous white margins. Clear enough to display at 250 CSS pixels wide. Conceptual illustration only, no data, no numeric axes or scientific results. Keep buildings recognizable and pedestrian scale realistic. Show an isometric compact city map illustrating access to jobs and services: homes at lower left, a metro station at center, office building at upper right, a small clinic and shops nearby. Draw a few teal connecting routes and pedestrian links connecting homes through the station to destinations. Recognizable building pictograms and small people, no circular UI badges. Convey easier connections across the city, not predicted impacts. No clocks, distances, or invented travel times.

### 3

Use case: scientific-educational. An attractive urban research seminar illustration. Wide 4:3 composition, pure white backdrop, professional hand-drawn architectural editorial illustration with clean navy outlines, gentle pale blue, muted teal and restrained warm orange highlights, no text, no labels, no numbers, no logos, no watermark. Balanced full composition with generous white margins. Clear enough to display at 250 CSS pixels wide. Conceptual illustration only, no data, no numeric axes or scientific results. Keep buildings recognizable and pedestrian scale realistic. Show an attractive urban street with ground-floor shops, cafes, and apartments above, together with a few pedestrians and cyclists. Include homes and a business frontage so housing demand and local business activity are both tangible. Clear small warm details such as shop awnings, plants, tables. A lived-in mixed-use neighborhood, not a spectacular skyline. No before-after comparison, no crowds implying a measured increase.



---

## 原记录：concept_figures_v8/prompts_and_usage.md

# Concept figures — V8

Generated with the built-in GPT-image tool. These are conceptual illustrations; numerical findings remain in the original result charts.

## related_work_bridge_v8.png

Slide: 3

Create a polished scientific seminar infographic on pure white, wide 16:7 landscape, Times New Roman labels, navy teal and muted burgundy, clear architectural icons. This is a conceptual diagram, no real data. Two balanced research streams converge toward one central research question. Left: metro station, neighborhood streets, and a schematic effect-over-time curve, exact heading 'Transport effects', small label 'Estimate changes after opening'. Right: a city map, satellite/building icons, and abstract embedding points, exact heading 'Urban representations', small label 'Compare places using local features'. Both streams point toward a central lower motif of a city grid linked to response-path sketches, exact text 'Learn to compare places by their responses'. Professional publication figure, substantial detailed illustrations and readable large labels, a single unified composition, no cards or dashboard, no numerical results, no logos, no other text. The sketches depict concepts, no numeric axes.

## problem_setup_v8.png

Slide: 4

Generate a precise conceptual scientific diagram for a seminar slide on pure white, wide landscape 16:7 aspect, elegant clean icons, navy teal muted orange. All labels large Times New Roman, low text density. TWO clear horizontal stages arranged top and bottom with arrows. TOP stage heading 'Estimate local responses': a cluster of blue urban grid cells with a metro station icon labelled exactly 'Treated group', and separate grey urban grid cells with neighborhood icons labelled 'Control group'. Their comparison points toward a simple schematic response-over-time curve labelled 'Estimated response path'. IMPORTANT: these are groups of city grids, control grids are eligible comparison areas, show no invented values or empirical measurements. BOTTOM stage heading 'Learn to compare places': housing, POI, lights and population icon bundle labelled 'Before-opening features' points to a compact vector/point motif labelled 'Learned representation', then points to a city grid with linked highlighted neighbors labelled 'Similar response paths'. A single thin vertical arrow from the top estimated response path to the bottom learned representation is labelled 'Training supervision'. A small simple footer under the bottom feature-to-representation pathway reads 'New city: features only'. Never connect observed post-opening outcomes to new-city inference inputs. No extra prose, no logos, no boxes or cards, clean uncluttered arrows and generous readable spacing. Scientific schematic, not measured results.

## matching_groups_v8.png

Slide: 7

Create a wide landscape 16:5 scientific educational diagram, white background, Times New Roman labels. An icon-rich explanation of Matching, using city grid maps and paired lines, navy for 'Treated group', teal for 'Control group'. Left: two city grid thumbnails, treated group has a new metro station icon, control group has eligible comparable grid cells and no intervention depicted, with a thin matching arrow. Center: two separate schematic mini time paths, labelled exactly 'Change in treated group' and 'Change in control group', each clearly shows a before marker and after marker linked by a line. Right: their difference represented by a subtraction sign and a single response-path icon, labelled 'Difference in changes'. These are illustrative paths, no numeric coordinates or actual findings, no implied universal positive effect. Exact top labels only 'Treated group' and 'Control group' plus the three labels above. Subtle professional map/building/train icons, very legible labels, no logos, no giant statistics, no extra text. Conceptual instructional figure.

## gsc_mc_intuition_v8.png

Slide: 8

Draw a wide 16:5 landscape scientific conceptual diagram for GSC and MC, white background, navy teal and orange, large Times New Roman labels, two clear horizontal halves. LEFT half: a bundle of simple grey control-group time curves feeds into one dark teal common pattern, labelled 'GSC: learn common trends'; below a navy treated-group pre-opening path followed by a teal dashed forecast labelled 'Predict the no-opening path'. Use a visible opening divider and dashed forecast, NO numeric axes, no empirical result. RIGHT half: a city-by-time matrix of small square cells, some pale blue observed cells and a clearly contiguous missing block on the lower right, followed by an arrow to the same matrix with that block filled in teal. Label 'MC: fill missing outcomes', and beneath 'Use shared patterns across places and time'. Above left source curves exact small label 'Control group'; next to the left target curve exact small label 'Treated group'. This diagram illustrates counterfactual estimation, not observed effects. No invented measurements, no extra text, no logos, no card borders, clean flat scientific layout, readable mechanism motifs and few words.

## takeaways_visual_v8.png

Slide: 20

Create a visually rich yet clean academic seminar summary illustration on pure white, landscape 16:7, Times New Roman lettering, navy teal muted burgundy with small orange accents. A single coherent left-to-right visual composition with THREE stages, no cards or UI. LEFT motif: a metro opening in a neighborhood with night lights, shop and service icons, heading exactly 'Estimate local effects', caption 'Positive effects in matched samples'. MIDDLE motif: buildings, population, lights and POI icons feeding a simple prediction line motif, heading 'Use earlier local information', caption 'Small prediction gains across cities'. RIGHT motif: urban map and learned vector/neighbor points with separate moon/light icon and shop/POI icon, heading 'Learn to compare places', two short captions 'Lights: higher mean score' and 'POI: original features stronger'. Below all stages a clear simple illustration joins original-feature icons and predicted-response icons into one combined vector, exact caption 'Next: combine features and predicted responses'. No actual charts or numerical data, no invented results, no logos. Icons should be detailed enough to explain the concepts; avoid large empty spaces, tiny labels, decorative clutter or too much text. Results are limited to these exact supplied qualitative statements.



---

## 原记录：concept_figures_v8/correction_prompts.md

# Corrections

Built-in GPT-image edits. Corrected variants are used on slides 4 and 7.

## problem_setup_v8_corrected.png

Edit this conceptual figure with minimal changes. Preserve every layout, label, icon, color and all the bottom training/inference pathway exactly. In the top-right plot labelled 'Estimated response path', replace its two plotted curves by ONE single solid teal effect curve: at zero before the orange opening divider, rising smoothly after opening. Add a subtle grey horizontal zero reference. Keep existing Response and Time labels. An estimated response is the difference between observed and no-opening outcomes; the upper plot must show that difference, not the two outcome curves. No other changes, no numerical data.

## matching_groups_v8_corrected.png

Edit this conceptual Matching figure, preserve the maps, group names, all labels and the three schematic plots. Correct the math connectors in the three-plot sequence: place a clear minus sign BETWEEN 'Change in treated group' and 'Change in control group'; replace the current minus sign BETWEEN 'Change in control group' and 'Difference in changes' with an equals sign. Thus treated change minus control change equals difference in changes. Preserve everything else. No additional text, no invented measurements. Ensure the signs are legible and sit midway between plots rather than overlap any axis.



---

## 原记录：concept_figures_v9/prompts_and_usage.md

# Matching：平行趋势示意图

用于当前 PPT 第 7 页。图为概念示意，使用内置 GPT-image 生成；地图也为示意。保留 Matching 原公式与页脚。

最终素材：matching_parallel_trends_v9.png。

两组开通前都用水平线，直观表达变化平行、水平可以不同。灰色虚线从 treated group 的开通时点出发，表示其未开通时的路径；Estimated effect 是 observed treated 与该虚线的竖直差。平行趋势也允许两组有相同的非零斜率；水平线只是最简示例。

## 初始提示词

Edit the attached Matching instructional infographic. Preserve the two detailed city-map thumbnails on the LEFT, their metro icon, matching arrow, navy/teal palette and exact labels 'Treated group' and 'Control group'. Completely replace ALL THREE mini plots on the RIGHT, the subtraction sign, equals sign, headings and their shaded bands with ONE spacious shared-axis difference-in-differences time plot. Wide 3:1 landscape image, pure white, Times New Roman large legible labels. Right plot header: 'Compare changes over time'. Vertical y-axis: 'Outcome'; horizontal x-axis: 'Time'. A vertical thin orange dashed divider about halfway across the time range labelled 'Station opening' separates at least THREE clearly visible pre-opening time points from the post-opening region. TWO SOLID lines before opening, navy treated group above teal control group with a constant vertical gap, equal slopes, visibly parallel throughout the entire pre-opening interval. They need not have equal outcome levels. After opening, teal control group continues its earlier gentle slope; solid navy treated group rises more steeply. From the treated group's exact opening-time level, add a grey DASHED post-opening no-opening path continuing the same gentle slope as control, keeping the pre-opening vertical gap constant. At the final time, mark a single vertical bracket ONLY between solid navy and grey dashed endpoints, labelled 'Estimated effect'. Label the grey dashed line 'Treated group without opening'. Small bottom region labels 'Before opening' and 'After opening'; small annotation above PRE-opening segment 'Parallel trends'. Navy/teal group color labels near their lines OR a compact legend, matching the maps. No numerical ticks, no uncertainty bands, no third solid purple series, no extra prose, no logos. Ensure the dashed counterfactual begins exactly at the navy solid line on the station-opening divider, and does not originate at the control line. These are conceptual lines, not empirical estimates. Keep map proportions and coherent overall layout, use the entire horizontal canvas. Scientific correctness is essential.

## 平行趋势精修提示词

Edit this Matching infographic with a surgical correction of the RIGHT plot only. Preserve all LEFT maps/icons/text and overall 3:1 layout, navy treated and teal control colors, pure white background, Times New Roman font. For crystal clear mathematical parallel trends, make BOTH pre-opening solid lines perfectly HORIZONTAL: navy treated at a higher constant outcome level and teal control at a lower constant outcome level. Both have at least 4 dots, identical zero slope and constant vertical distance. Navy opening-time point stays at the same level as its 4 earlier dots. After the orange Station opening divider, teal control continues perfectly HORIZONTAL at its original lower level, and GREY DASHED Treated group without opening path continues perfectly HORIZONTAL from the EXACT navy opening-time point at its original upper level. Solid navy treated group rises after opening with 4 post-opening dots. Estimated effect is the final vertical gap between navy observed endpoint and horizontal grey dashed counterfactual endpoint, never the control line. Preserve header Compare changes over time, axes Outcome and Time, annotation Parallel trends, Before opening / After opening. These horizontal lines are a stylized conceptual example; no numerical ticks, no confidence bands. Make text concise, legible, no extra labels or additions.

## 图像来源

最终原始输出：C:/Users/HCOOH/.codex/generated_images/01a0f86a-ce69-7842-add5-4315ef4c3360/exec-f17cc811-aa72-4fa8-ac3e-fa40778e867f.png。
初始参考：seminar_materials/concept_figures_v8/matching_groups_v8_corrected.png。


---

## 原记录：gpt_image_flowcharts/prompts_en.md

# GPT-image flowcharts for the 21-slide English seminar

## 文件与页面对应

| 图 | 文件 | 建议使用页 |
|---|---|---|
| 项目总流程 | [intro_pipeline_gpt_en.png](intro_pipeline_gpt_en.png) | 第5页问题与任务 |
| 因果标签路由 | [intro_routing_gpt_en.png](intro_routing_gpt_en.png) | 第7页因果标签 |
| 表示学习结构 | [intro_model_gpt_en.png](intro_model_gpt_en.png) | 第15页模型，与第16页损失结合讲述 |

三张图均已检查标签、路由顺序和模型监督方向。模型图交叉处的小白色间隔表示两条线经过而不连接。项目总流程使用宽画幅，可在PPT正文区域居中；路由和模型图为接近16:9画幅。文字为生成式位图中的Times New Roman风格。

原有确定性矢量流程图位于上一级 `figures/svg/` 和 `figures/pdf/`。本次生成图另存为PNG，尚未替换PDF中的图。

生成方式：内置 image_gen；英文概念流程图，白底，Times New Roman 风格文字。对应当前21页正文：总流程可用于第5页；路由用于第7页；模型用于第15页，三项损失与第16页对应。保持原PDF不变，生成图用于手动PPT。

以下为本次实际使用的完整提示词。GPT-image生成的是位图，节点与文字不是独立可编辑图层。原有PDF/SVG流程图继续保留。

## 1. Project workflow

```text
Use case: scientific-educational. Asset type: flowchart image for an English 16:9 academic seminar slide. Create ONE high-resolution landscape 16:9 diagram on a pure white background. Flat vector-like appearance, precise thin arrows, rounded rectangular nodes, pale fills, generous whitespace and margins. All typography should look like Times New Roman, large and readable on projection, dark navy text. Navy #234E70 for features and models, teal #168A85 for response labels and supervision, muted orange #D28436 only where a decision needs emphasis. No slide title, no footer, no photographic imagery, no 3D, no decorative skyline, no watermark. Use only the exact English labels specified below. This is a conceptual diagram: do not invent numbers, response curves, performance scores or claims. Keep arrows connected to actual node boundaries, avoid line crossings and tiny text. PROJECT WORKFLOW. Six main nodes in two lanes. Far-left shared source node, centered vertically, exact text "Station openings and grid panels", second line "5,048 events | 44 cities". It branches upward to "Pre-treatment features X", second line "Training-side preprocessing"; and downward to "Causal response estimation", second line "Support, diagnostics, frozen labels". The top lane continues right to "Tabular encoder", second line "Normalized embedding e". The bottom lane continues right to "Response supervision", second line "Shape similarity and prediction". Draw a downward arrow from Tabular encoder to Response supervision, arrow label "Training losses". Draw a lower-lane arrow from Causal response estimation to Response supervision. Draw an upper-lane arrow from Tabular encoder to the final far-right node "Held-out city evaluation", second line "Compare with raw features and autoencoder". Below the final node, the annotation "Inference uses X only". Maintain six main nodes, clear two-lane structure. Estimated labels enter supervision losses only. There is no arrow from response estimation or labels to encoder input. Show no supervision arrow to held-out evaluation. Do not label estimated responses as ground truth. No success claim.
```

## 2. Causal label routing

```text
Use case: scientific-educational. Asset type: flowchart image for an English 16:9 academic seminar slide. Create ONE high-resolution landscape 16:9 diagram on a pure white background. Flat vector-like appearance, precise thin arrows, rounded rectangular nodes, pale fills, generous whitespace and margins. All typography should look like Times New Roman, large and readable on projection, dark navy text. Navy #234E70 for features and models, teal #168A85 for response labels and supervision, muted orange #D28436 only where a decision needs emphasis. No slide title, no footer, no photographic imagery, no 3D, no decorative skyline, no watermark. Use only the exact English labels specified below. This is a conceptual diagram: do not invent numbers, response curves, performance scores or claims. Keep arrows connected to actual node boundaries, avoid line crossings and tiny text. CAUSAL LABEL ROUTING. Exact ordered seven-stage cascade: "Same-city matching" -> "Same-city GSC" -> "Same-city MC" -> "Cross-city matching" -> "Cross-city GSC" -> "Cross-city MC" -> "Skip". Place first three on top row left-to-right. Arrow from top-right Same-city MC vertically down to Cross-city matching at bottom-right. Bottom row proceeds right-to-left to Cross-city GSC then Cross-city MC. Final arrow from Cross-city MC downward to a small Skip node. Navy first three stages, teal next three stages, neutral gray Skip. Each estimator node must be the same size and have its exact complete stage name, legible line breaks allowed. Top annotation "One assigned method per grid and outcome family". In an open area below the cascade to the right of Skip, place a shared teal output card with exact text "Keep response labels, masks and diagnostics". Put the shared rule immediately above this card: "If available: keep the method and stop". Put the shared fallback rule in a separate clearly visible legend: "If unavailable: continue along arrows". The output card is explained by the shared rule; do not add six crossing retention branches. Skip has no arrow to the output card. The cascade is sequential assignment according to support/design availability, not an ensemble and not a p-value filter. Do not add a significance gate or claims that one method repairs another.
```

## 3. Representation learning

```text
Use case: scientific-educational. Asset type: flowchart image for an English 16:9 academic seminar slide. Create ONE high-resolution landscape 16:9 diagram on a pure white background. Flat vector-like appearance, precise thin arrows, rounded rectangular nodes, pale fills, generous whitespace and margins. All typography should look like Times New Roman, large and readable on projection, dark navy text. Navy #234E70 for features and models, teal #168A85 for response labels and supervision, muted orange #D28436 only where a decision needs emphasis. No slide title, no footer, no photographic imagery, no 3D, no decorative skyline, no watermark. Use only the exact English labels specified below. This is a conceptual diagram: do not invent numbers, response curves, performance scores or claims. Keep arrows connected to actual node boundaries, avoid line crossings and tiny text. REPRESENTATION LEARNING MODEL. Top row has four main navy nodes left-to-right: "Pre-treatment features X" with second line "Annual features | 1-3 year lags"; "MLP encoder" with second line "256 -> 128"; "Projection head" with second line "128 dimensions | L2 normalization"; "Embedding e" with second line "Response retrieval". Connect all four sequentially with navy arrows. A navy branch from Embedding e goes to a separate right-side node "Outcome-family prediction heads". Head reads the final normalized embedding. Upper-row section label "Feature-to-embedding path".
Below a subtle divider, lower region section label "Training supervision". Four teal nodes: "Frozen response estimates r" with second line "Masks, design weights, SE"; "Pairwise response similarity"; "Alignment + distance losses"; "Cell-wise prediction loss". Connect Frozen response estimates r -> Pairwise response similarity -> Alignment + distance losses. Connect Embedding e down to Alignment + distance losses. Connect prediction heads down to Cell-wise prediction loss. Connect Frozen response estimates r separately to Cell-wise prediction loss using a clearly routed elbow line around the lower edge, avoiding crossings and arrows to model input. Two loss nodes connect to a bottom total-loss card with exact text "Total loss" and second line "0.25 alignment + 0.25 distance + 0.50 prediction". Place final short annotation "Held-out city inference uses X only". Estimated response labels supervise losses only and never feed the encoder or feature path. No direct X-to-prediction-head bypass. Omit optional image encoders and conditioning. No augmentation, counterfactual generator or graph discovery. Ensure all exact coefficients and graph directions are preserved.
```



---

## 原记录：gpt_image_flowcharts/illustrated_prompts_en.md

# Illustrated GPT-image diagrams

## Final local correction for the model diagram

```text
Make exactly TWO small wiring changes to this image, preserve every panel, illustration, all text, fonts, colors, layout and every other arrow. FIRST: erase the tiny horizontal black right-pointing arrow between the right edge of panel C "Alignment + distance losses" and the left edge of panel D "Cell-wise prediction loss", at approximately the image's 66.5% width and 70% height. Replace that little arrow with clean white background. C and D are parallel losses and must have NO direct arrow between them. Keep B -> C and keep both C -> E and D -> E. SECOND: at the crossing of the long horizontal "Targets r" wire with the vertical C -> E wire just above Total loss, insert a small white bridge gap in the horizontal target wire so it clearly crosses without joining the vertical wire. The target arrow still ends upward at D. Do not change anything else. No added text, no new arrows.
```


参考：用户提供的多面板方法图，仅参考视觉结构。采用英文Times New Roman风格，增加功能图标和模块内部示意；所有小曲线、矩阵与地图均为conceptual illustration。

本次为增强版，保留第一版简图与原PDF。以下记录生成及技术核对后使用的局部修正提示词。

## 1. Project overview

### Generation

```text
Use case: scientific-educational. Create an illustrated scientific METHODS OVERVIEW for a real urban research project, an English academic seminar. The attached image is a STYLE AND COMPOSITION REFERENCE ONLY: borrow its numbered large panels, internal mini-diagrams, simple recognizable icons, carefully aligned subcomponents, white background, clear black outlines and clear outer arrows. Do not copy its subject, text, photographs, numerical results or logos. Make a polished high-resolution 16:9 landscape image with crisp legible Times New Roman style serif typography throughout, navy #234E70 and teal #168A85 highlights, light muted fills, restrained orange accents, thin elegant outlines, even spacing. This must look like a detailed research graphical abstract, NOT a list of text boxes. Illustrate technical operations inside each panel with specific icons and explanatory miniature diagrams. Use short exact English labels, no long prose, no ornamental 3D, no watermark. Any tiny matrix or trajectory is a conceptual illustration; give it no numeric axes or empirical error bars, and include the bottom note "Conceptual illustration". Do not fabricate measurements, sample counts, embedding clusters, accuracy or significance. Large readable main text; content must fit without clipping. PROJECT METHODS OVERVIEW. Six numbered panels in a 2-by-3 grid, connected in a snake: upper row 1 -> 2 -> 3, down to lower-right 4, then left to 5 then left to 6.
Panel 1 exact header "1. Urban data and station openings". Draw a recognizable metro train/station icon and a city-grid map with station pin, plus four compact illustrated data tiles labelled "Housing", "VIIRS", "POI", "Population" (house, satellite/night-light, shop, people icons). At bottom show the factual badge "5,048 events | 44 cities", and "500 m grids".
Panel 2 header "2. Time-aligned inputs". Inside split two short branches: small feature table icon labelled "Pre-treatment features X" and city-panel icon labelled "Outcome panels Y". Include opening timeline with "Before opening" and "After opening", a clear opening marker, and "Training-side preprocessing". This panel organizes both features and outcomes.
Panel 3 header "3. Causal response estimation". Show three mini-method pictograms with exact captions "Matching", "GSC", "MC": paired treated/control grids; shared factor nodes; low-rank matrix with missing post cells. One concise note "Support-based routing". Illustrative trajectory without numeric axes labelled "Observed" and "Counterfactual"; the difference is a bracket marked "Estimated response".
Panel 4 lower-right header "4. Frozen response signatures". Use a document/lock icon, a response-vector strip, a mask grid and a method-source tag, exact labels "Response r", "Mask", "Method", "Diagnostics". Include "VIIRS: 6 horizons" and "POI: 4 outcomes x 3 years". The arrow from this panel to panel5 labelled "Training targets".
Panel 5 lower-middle header "5. Response-supervised learning". Inside show feature table "X" -> network glyph "MLP" -> vector "Embedding e". Below it show compact teal loss box "Shape + prediction losses", with arrows from embedding and a response-label document marked "r" into this loss box. There is never a response-label arrow into MLP input. A small branch from embedding to "Prediction heads" feeds losses. No hypothetical image encoder. Add a thin explicitly labelled auxiliary line from panel2's feature branch into panel5's X input, routed in the gap between rows; this is labelled "Features X". Panel4's supervision path goes to losses only.
Panel 6 lower-left header "6. Held-out city evaluation". Illustrate a separate city outline with a pin, a query-vector icon linked to five generic neighbor-grid tiles, and three metric icons captioned "Shape retrieval", "Magnitude", "Direction". Include "Raw features / Autoencoder / Supervised" as baseline legend and "Inference uses X only". Show neighbor tiles rather than a t-SNE scatter; no invented clusters or results.
Arrows between numbered panels indicate workflow organization. The internal model graph distinguishes features as inputs and estimated responses as supervision. Keep this graph accurate, avoid overlaid arrows through content.
```

### Scientific correction

```text
Edit this illustrated six-panel overview locally. Preserve the six numbered panels, all illustrations, colors, visual quality, typography and content unless specifically changed below. Correct Panel 5 wiring: remove the line running from the Features X table directly down into Shape + prediction losses. Features X should connect ONLY to MLP. Keep MLP -> Embedding e. Keep Embedding e -> Prediction heads and Embedding e -> losses. Add a clear arrow from Prediction heads to Shape + prediction losses; all head predictions feed that loss box. Keep r -> losses, and never feed r into MLP. Correct Panel 6: replace "Nearest neighbors (in feature space)" with exact text "Nearest neighbors (embedding space)". Replace the compass/geographical arrow graphic in the Direction metric tile with a clean pair of bars above and below a zero line, annotated "+" and "-", to convey the SIGN OF RESPONSE, not direction in physical space. Keep tile title "Direction". The five neighbors may be suggested by three visible tiles plus ellipsis. No other content changes. Inference uses X only. Keep the footer "Conceptual illustration".
```

## 2. Causal methods and routing

### Generation

```text
Use case: scientific-educational. Create an illustrated scientific METHODS OVERVIEW for a real urban research project, an English academic seminar. The attached image is a STYLE AND COMPOSITION REFERENCE ONLY: borrow its numbered large panels, internal mini-diagrams, simple recognizable icons, carefully aligned subcomponents, white background, clear black outlines and clear outer arrows. Do not copy its subject, text, photographs, numerical results or logos. Make a polished high-resolution 16:9 landscape image with crisp legible Times New Roman style serif typography throughout, navy #234E70 and teal #168A85 highlights, light muted fills, restrained orange accents, thin elegant outlines, even spacing. This must look like a detailed research graphical abstract, NOT a list of text boxes. Illustrate technical operations inside each panel with specific icons and explanatory miniature diagrams. Use short exact English labels, no long prose, no ornamental 3D, no watermark. Any tiny matrix or trajectory is a conceptual illustration; give it no numeric axes or empirical error bars, and include the bottom note "Conceptual illustration". Do not fabricate measurements, sample counts, embedding clusters, accuracy or significance. Large readable main text; content must fit without clipping. CAUSAL PIPELINE DETAIL. Rich illustrated methods diagram in three horizontal bands. This should be more detailed than a simple routing flowchart.
Top band contains 3 compact modules. Module "Grid and outcome panels": city map with station pin and table glyph. Module "Event windows": small timeline with exact labels "Baseline: -18 to -7 months", "Anticipation: -6 to -1", "Post-opening"; small note "Annual baseline: previous year". Module "Support and design": simple data availability matrix icon and control-grid icon, exact short labels "Coverage", "Controls", "Pre-period support". Do not add approval checkmarks or p-value gates.
Middle band is a clear six-stage cascade with small method-specific icons inside every stage: matching uses two linked grids, GSC uses a factor-network icon, MC uses a low-rank matrix icon. The ordered sequence is "Same-city matching" -> "Same-city GSC" -> "Same-city MC" -> "Cross-city matching" -> "Cross-city GSC" -> "Cross-city MC". First three appear top subrow left-to-right, cross-city three below right-to-left. The same-city MC node connects downward to cross-city matching. Cross-city MC connects to a small gray "Skip" node. Put a compact shared arrow legend "Unavailable: continue" and shared retention annotation "Available: retain and stop". Do not mix estimation methods or compute an ensemble; each grid and outcome family gets one assigned method. Use map icons to distinguish within-city and across-city stages, but keep all six stage labels readable.
Bottom band contains three output objects as icon-rich cards: "Assigned response" with illustrative observed/counterfactual difference and r vector; "Design diagnostics" with pre-period comparison and coverage glyphs; "Frozen learning dataset" with lock, mask grid, method tag and response-strip icons. Explicit note "Responses, masks, methods and diagnostics". The bottom band describes the retained estimator outputs. Do not draw an arrow from Skip to retained outputs. Put note "One method per grid and outcome family". Do not show all six methods estimating the same grid simultaneously. Charts are labelled "Illustration" without numeric axes. Freeze outputs only after method assignment; do not imply diagnostic p-values determine route.
```

### Scientific correction

```text
Edit this causal pipeline diagram locally, preserving all panel positions, icons, data counts, palette, typography and the cascade order. Fix the Event windows timeline: the dashed "Opening date" marker must sit EXACTLY at the boundary where orange anticipation ends and teal post-opening begins. Label intervals stay "Baseline: -18 to -7 months", "Anticipation: -6 to -1", "Post-opening". Move the dashed marker to this boundary; it must not divide the middle of the anticipation period. In the top-right panel, remove the unrequested list of specific control variables (Built environment, Demographics, Accessibility, Socioeconomic, Additional factors). Replace it with two compact labelled pictograms: "Eligible controls" (paired grid icon) and "Pre-period support" (a small temporal coverage matrix with pre-period columns). Do not use a map buffer as an illustration of temporal pre-period support. Correct the cascade legend: all arrows joining six methods and the final Skip are fallback arrows. Use ONE shared arrow legend "Unavailable: continue along arrows"; use a separate text-only retention annotation "Available: retain and stop". Do not use a solid-arrow legend labelled Available when the actual method-to-method arrows are solid. Every method-to-method arrow means unavailable, and output retention is described in the note. Preserve the exact ordered stages, bottom response/diagnostic/dataset modules, illustrations, and footnote. No new variables or claims.
```

## 3. Response learning

### Generation

```text
Use case: scientific-educational. Create an illustrated scientific METHODS OVERVIEW for a real urban research project, an English academic seminar. The attached image is a STYLE AND COMPOSITION REFERENCE ONLY: borrow its numbered large panels, internal mini-diagrams, simple recognizable icons, carefully aligned subcomponents, white background, clear black outlines and clear outer arrows. Do not copy its subject, text, photographs, numerical results or logos. Make a polished high-resolution 16:9 landscape image with crisp legible Times New Roman style serif typography throughout, navy #234E70 and teal #168A85 highlights, light muted fills, restrained orange accents, thin elegant outlines, even spacing. This must look like a detailed research graphical abstract, NOT a list of text boxes. Illustrate technical operations inside each panel with specific icons and explanatory miniature diagrams. Use short exact English labels, no long prose, no ornamental 3D, no watermark. Any tiny matrix or trajectory is a conceptual illustration; give it no numeric axes or empirical error bars, and include the bottom note "Conceptual illustration". Do not fabricate measurements, sample counts, embedding clusters, accuracy or significance. Large readable main text; content must fit without clipping. RESPONSE LEARNING METHODS OVERVIEW. Detailed illustrated neural architecture with separate model and supervision regions, large academic icons and exact graph. Upper region occupies 45 percent height and is labelled "Feature-to-embedding path". Four navy stages arranged left-to-right: "Pre-treatment features X" containing a feature-table pictogram and small house/shop/satellite icons plus caption "1-3 year lags"; "MLP encoder" containing a clean small node-and-edge neural-network pictogram caption "256 -> 128"; "Projection head" containing compression/vector-bar glyph caption "128 dimensions | L2 normalization"; "Embedding e" containing a vector strip linked to 5 generic neighbor-grid tiles caption "Response retrieval". Model arrows sequential X -> MLP -> projection -> embedding. A separate branch from embedding to an "Outcome-family prediction heads" module, containing a branching-head icon; heads read normalized e.
Lower region labelled "Training supervision", separated by thin dashed divider. Place a teal "Frozen response estimates r" module with lock/document icon, small time-profile illustration and mask grid, captions "Masks | Weights | SE". Draw its first arrow to "Pairwise response similarity" with a small symmetric matrix pictogram. From pairwise similarity draw arrow to "Alignment + distance losses" with a cosine/paired-vector illustration. Draw a model-input arrow from embedding e to this alignment-loss module. Separately show "Cell-wise prediction loss" with predicted-value and observed-value strips. Prediction heads feed this loss, and frozen response r also feeds this loss with a separate clearly routed label-target arrow. The only two incoming arrows into a bottom-center "Total loss" box originate from Alignment + distance losses and Cell-wise prediction loss. Exact coefficient text inside Total loss: "0.25 alignment + 0.25 distance + 0.50 prediction". Frozen r must NEVER connect directly to total loss, encoder, projection or embedding. Prevent shared bidirectional wires. Use small bridge gaps when two lines cross so there is no junction. Add a small held-out-city map icon at far right alongside note "Inference uses X only". No generic cluster scatter, no images-as-model-inputs, no augmentation, no performance statistics. Icons illustrate computation, not measured results. Keep the model graph unambiguous and make the architecture visually rich, balanced, and attractive.
```

### Scientific correction

```text
Edit this richly illustrated model diagram locally. Preserve the attractive panel layout, colors, icons, neural drawings, Times-style typography, coefficients and all correct top model arrows. Make these mandatory scientific corrections:
1. In the upper-left feature table use column headers "Unit", "Housing", "VIIRS", "POI", "Population", "..." instead of Built Env./Mobility/Socioeconomic. These are the project's actual feature categories.
2. In Outcome-family prediction heads replace the three boxes Housing/Mobility/Environment by exactly two output boxes "VIIRS" and "POI"; preserve branches from normalized e to both.
3. Replace lower-section subtitle "Use external responses and self-supervision to train representations" with "Frozen estimated responses supervise training". The project has response supervision, no self-supervision claim.
4. In frozen responses panel replace "External estimates r (frozen)" with "Estimated responses r (frozen)".
5. Remove the arrow C (Alignment + distance losses) -> D (Cell-wise prediction loss). These are parallel losses.
6. REMOVE the arrow from F (Inference) into D and its "Label targets (from r)" label. Inference supplies no labels and has no arrow into a training loss. Leave F as an independent explanatory city icon with "Inference uses X only".
7. Add a separate target arrow from A (Frozen response estimates r) into D (Cell-wise prediction loss), labelled "Targets r". Route it under the B and C panels through the whitespace above Total loss, then upward into D's lower boundary. If it crosses C -> Total loss, use an unambiguous white bridge gap, with no junction. Prediction heads -> D remains a separate input.
8. In D use strip labels "Predicted responses" and "Response targets r", replacing observed y wording. Keep the comparison glyph between strips.
9. Replace the incorrect unweighted 1 - cos graphic and maximize-similarity labels inside C with two small vector-pair glyphs labelled "Embedding similarity" and "Response similarity", linked by "Fit similarities". Do not put an invented distance formula there.
10. The ONLY two incoming arrows to E Total loss originate in C and D. Exact text "0.25 alignment + 0.25 distance + 0.50 prediction" stays unchanged.
Exact graph: X -> MLP -> projection -> e; e -> prediction heads; r -> pairwise response similarity -> alignment + distance losses; e -> alignment + distance losses; r -> cell-wise prediction loss; prediction heads -> cell-wise prediction loss; both parallel losses -> total loss. No other loss edges. Keep "Conceptual illustration".
```



---

## 原记录：gpt_image_flowcharts/routing_icons_v3_prompt.md

# Causal estimator routing — icon version

Generated with the built-in GPT-image tool. Final asset: `intro_routing_gpt_icons_v3.png`. Used on slide 7. The diagram is conceptual; it contains no empirical results.

## Prompt

Use case: scientific-educational. Create a polished icon-rich academic causal-estimator ROUTING diagram for an English seminar PowerPoint. Wide landscape composition approximately 2.3:1, pure white background, crisp dark navy outlines, restrained teal and pale blue fills, small orange station pins, Times New Roman style serif labels. This diagram must be readable when displayed at about 818 CSS pixels wide. Large labels, generous spacing, mostly icons and small explanatory miniature diagrams. Six method stages in a two-row serpentine cascade: top row left to right "1. Within-city matching", "2. Within-city GSC", "3. Within-city MC"; bottom row right to left "4. Cross-city matching", "5. Cross-city GSC", "6. Cross-city MC". Arrows connect 1 -> 2 -> 3 -> downward 4 -> leftward 5 -> leftward 6 -> a small gray "Skip" at far left. Every one of these arrows means fallback when support is unavailable. Put a single shared readable annotation "Unavailable: continue along arrows". Put a distinct annotation "Available: retain and stop". Method 1 has icon of two neighboring colored grid cells with a station pin and paired link. Method 2 has a small factor-network pictogram and stacked spatial/temporal matrix layers. Method 3 has an observed matrix and completed matrix with subtly filled missing cells. Method 4 has two tiny city silhouettes and matching grid pair with orange and teal pins. Method 5 has two city silhouettes alongside the factor-network pictogram. Method 6 has two city silhouettes and a low-rank matrix pictogram. Distinguish within-city modules with a pale blue band and cross-city with pale teal. Keep all six exact stage labels large; no explanatory sentences inside nodes. A slim bottom band titled "Retained estimator outputs" contains a response-profile pictogram labelled "Response vector", a small observed-cell grid labelled "Mask and method", and a document-and-lock icon labelled "Frozen dataset". This bottom band explains retained outputs, not an additional fallback stage; no arrow from Skip into retained outputs. Bottom note "One assigned method per grid and outcome family". Tiny trajectories are conceptual illustrations, with no numerical axes or measured results. NO fake data, p-values, percentages, gate checkmarks, ensemble averaging, formulas, extra methods or text-heavy cards. No logos, no 3D, no shadow, no watermark. The six stages, arrows, and rich method-specific icons are the focal point; this must look like a refined scientific figure, not a generic text flowchart.


---

## 原记录：gpt_image_flowcharts/README.md

# GPT-image英文方法图

推荐使用下列**增强版**，已按用户提供的多面板参考图重做：模块中加入城市网格、轨道站点、数据表、共享因子、矩阵补全、响应向量与神经网络等功能示意。

| 图 | PNG | 建议放置 |
|---|---|---|
| 项目方法总览 | [intro_pipeline_gpt_illustrated_en.png](intro_pipeline_gpt_illustrated_en.png) | 第5页Problem setup，作为贯穿全场的总览 |
| 因果估计与标签路由 | [intro_routing_gpt_illustrated_en.png](intro_routing_gpt_illustrated_en.png) | 第7页Causal labels，内部细节可在第8–9页放大讲 |
| 表示学习结构与监督 | [intro_model_gpt_illustrated_en.png](intro_model_gpt_illustrated_en.png) | 第15页Model，第16页可放大下半部讲损失 |

完整生成及修改提示词见 [illustrated_prompts_en.md](illustrated_prompts_en.md)。生成方式为内置image_gen；参考图仅用于视觉结构。前一版简单文字流程图也保留在本目录，提示词见 [prompts_en.md](prompts_en.md)。

## 讲述和排版

- 总览图先讲“数据→估计响应→监督学习→城市外比较”，细节留给方法页。
- 因果图虚线方法箭头表示不可用时继续；可用时保留该方法并停止。每个网格和结果族只分配一种方法。
- 模型图上半部是特征编码，下半部是训练监督。两组损失并行进入总损失；交叉线的白色间隔表示不连接。VIIRS/POI展示可用的结果族预测头，当前两族实验分别运行。
- 所有地图、轨迹、矩阵、向量色块和邻居网格均为概念示意，图上已标Conceptual illustration。实测结果继续使用现有定量图。
- 文字为位图中生成的Times New Roman风格；PNG的文字与图标不构成可编辑文本层。原有矢量图继续保留。
- 三张增强图信息量较大，建议整页显示；演讲时可按模块出现或逐步放大。实际插入PPT后检查投影字号。

本次没有替换当前PDF中的图，也没有改变21页正文顺序。原PDF和讲稿仍可直接使用。

