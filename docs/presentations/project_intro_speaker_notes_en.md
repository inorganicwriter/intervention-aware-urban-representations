# 12 分钟讲稿：20 页正文

当前讲稿约 1262 个英文词。按每分钟 110–115 词，口头读稿约 11.0–11.5 分钟，再留出指图、翻页和停顿时间。目标约 12 分钟，实际以排练为准。第 21–25 页为参考文献，留作讨论。

| 页码 | 讲述重点 | 建议用时 | 累计时间目标 |
|---|---|---|---|
| 1–3 | 开场、背景与相关研究 | 1:20 | 1:20 |
| 4–5 | 两步任务与数据 | 0:45 | 2:05 |
| 6–10 | 因果方法、event study 与前期检查 | 2:15 | 4:20 |
| 11–12 | 灯光与本地活动：读图、发现、样本口径 | 1:55 | 6:15 |
| 13–16 | 学习目标、模型与评估 | 2:00 | 8:15 |
| 17–19 | 原始特征、学习表示、训练波动 | 3:00 | 11:15 |
| 20 | 结论与下一步 | 0:45 | 12:00 |

结果页口头正文约 524 词。按“怎样读图 → 主要发现 → 对下一步的意义”讲，不逐项念图中的数字。

- 第 11 页：先指 Matching，再指零线和开通后路径。MC 在灯光中的城市支持很少，方法面板各有自己的支持样本。
- 第 12 页：解释 POI count 是本地活动代理。后期估计较大，但支持样本随年份变化。平均动态与逐区域训练标签分别说明。
- 第 17 页：依次指形状、误差、方向三个面板。口头数字重点是 POI 每 100 次预测约多判断对 5 次，属于 ridge 相对训练多数方向基线。
- 第 18 页：说明高柱代表更相似的响应邻居。appearance autoencoder 在首次口头出现时直接说全称，不说 AE。灯光和 POI 分别解释。
- 第 19 页：指零线、左右方向和横向误差条。误差条为不同训练 seed 的标准差，表示训练波动；不按置信区间或显著性解释。
- 第 20 页：融合评估是下一步，当前材料尚无其服务器结果。

公式讲其作用，不逐个读符号。representation learning 指训练方法，learned representation / embedding 指输出。GSC、MC 在第 6 页先说全称，POI 在第 5 页先说全称。正文用陈述句衔接。第 1 页保留用户原文。

8 分钟版保留在 speaker_notes_en_8min.md；当前 PPT 备注与 12 分钟版同步。

---

# Learning How Cities Respond: 12-minute speaker script

## 01 | Learning How Cities Respond

Hello everyone. I'm Sirun Li from Peking University and My mentor is Professor Yunhan Zheng, this summer, We study how places respond to a new rail station, and use these responses to train the guide comparisons across cities by representations learning. Let me start with a familiar example.

## 02 | Why do places respond differently?

A station opens near your home. Jobs become easier to reach, shops gain customers, and housing demand may change. These changes depend on local conditions. A city center and the city edge may respond differently. Our goal is to use earlier local information to find places with similar responses.

## 03 | Related Work: causal effects and representation learning

Two research areas inform this goal. Transport studies estimate infrastructure effects. Urban representation learning turns local information into numerical descriptions of places. Our project connects them: estimated response paths guide how we learn those descriptions.

## 04 | Problem setup: learning from local responses

We connect them in two steps. The treated group contains areas near new stations; the control group supplies comparison areas. First, we estimate local response paths. Then these paths guide a model that summarizes earlier information. In a new city, it uses earlier information alone.

## 05 | Data: station openings across 44 cities

For these two steps, we combine station openings with housing, nighttime lights, shops, services, and population. The learning tests focus on lights and points of interest, or POI. All model inputs come from before opening.

## 06 | Causal labels: method routing and a frozen dataset

With these data, we estimate what would happen without opening. We try Matching first, then Generalized Synthetic Control, or GSC, and Matrix Completion, or MC, to extend coverage. The first method that meets the checks supplies each area's training labels. Here is how Matching works.

## 07 | Matching: compare changes in the two groups

Matching pairs each treated area with a similar control area. In this sketch, their earlier trends are parallel, though their levels differ. The dashed line shows the treated group's path without opening. Its gap from the observed path is the estimated effect: the difference between the two groups' changes.

## 08 | GSC and MC: predict the path without opening

When Matching has too little support, GSC learns shared control group trends and fits the treated group's earlier pattern. MC fills missing no-opening outcomes using patterns across areas and time. Both estimate what would happen without opening.

## 09 | Event study: pre-opening trends and dynamic effects

The event study serves two purposes. Before opening, it checks whether the treated group and control group follow similar trends. This helps assess the parallel trends assumption. After opening, it shows how the estimated effect changes over time. Let us first look at the pre-opening checks.

## 10 | Pre-treatment diagnostics: evidence by method

Here are the pre-opening checks. Matching's city-level tests show little evidence of different earlier trends, though some outcomes are near the cutoff. For GSC and MC, we also examine earlier fit and supporting data. These checks help us judge how credible the later comparison is. We now turn to the estimated effects.

## 11 | Nighttime lights: positive effects after opening

The first result is nighttime lights. The horizontal axis shows months before and after opening. Zero on the vertical axis means no estimated effect. Focus on Matching on the left: after opening, the estimates move above zero and stay positive over the main follow-up period. The intervals at the main reported horizons also sit above zero. This is evidence of higher nighttime brightness in the matched areas. The other panels use different supported samples. For lights, MC covers very few cities, so Matching provides the clearest evidence here. We can now look at another measure of local activity.

## 12 | Local activity: more recorded places after opening

POI gives another view through recorded places such as shops and services. Here, time is measured in years. Matching shows positive effects after opening, with larger estimates at later horizons. The main intervals sit above zero. This suggests more recorded places in the treated group relative to its matched control group. POI count is a proxy for local activity. Coverage changes across years, and each method uses its own supported sample. These charts describe average dynamics. Separate area-level estimates provide the response paths for training. Those paths guide which places should be close in the learned representation.

## 13 | Learning target: response shape and consistency

To compare places using these paths, we mainly compare how they rise and fall together. Two areas can have a similar pattern even when their response sizes differ. We therefore check size and direction separately. The label checks show a clearer similarity structure for lights. POI combines several outcomes, so its patterns are less consistent. This makes outcome-level evaluation useful when interpreting the learning results.

## 14 | Representation learning: features to embeddings

Representation learning trains the model to turn earlier local information into a short numerical summary, called a learned representation, or embedding. Response paths guide the training. In a new city, the model uses earlier information alone. We train it with three related goals.

## 15 | Joint objective: neighbors, similarity and response values

To learn this representation, the model has three jobs: bring places with similar responses closer, match their similarity scores, and predict response values. We then check whether the learned representation helps in new cities.

## 16 | Evaluation: shape, magnitude and direction

We check similar paths, prediction error, and the direction of change. For retrieval, we take the nearest places in the embedding and compare their response paths. Higher similarity means better neighbors. Prediction error measures how well we recover response values. Direction measures how often we get the sign right. Direct predictions use city cross-validation; learned representations use an earlier city split. Each comparison has its own baseline. We start with the original features.

## 17 | Result 1: do local features help predict responses?

The original features carry a small signal. Read the three panels from left to right. On the left, areas that are close in their original features have more similar responses than random neighbors. In the middle, ridge, a simple linear prediction model, reduces error compared with using the training average. On the right, it also improves direction prediction. For POI, it gets direction right about five more times per hundred predictions than choosing the training set's most common direction. These tests keep entire cities out of model fitting. Together, they show that information available before opening carries some predictive signal across cities. This gives us a useful baseline for judging the added value of representation learning.

## 18 | Result 2: do learned representations find better neighbors?

Next, we compare learned representations with the original features. Taller bars mean that the nearest places have more similar response paths. We also include an appearance autoencoder, which learns to reconstruct the original features. It gives us a comparison with a model trained without response supervision. For lights, the supervised models have higher average scores than the original features. For POI, the original features and appearance baseline do better. The two response families therefore lead to different results. The error bars for supervised models show variation across training runs. They help us see how much the average score depends on training. The next plot makes the gains and their variation easier to compare.

## 19 | Training settings: do the gains hold up?

Here, zero is the original-feature baseline, and moving right means improvement. The model can keep examples from earlier training batches to provide more candidates for comparison. We compare training with and without those stored examples. For lights, removing them raises the average score, but the longer horizontal bar shows more variation across runs. For POI, both settings stay below the original-feature baseline on average. So average performance and stability both matter when choosing a training setting. These results point toward keeping the useful original feature structure while adding response information. That is the next test shown on the final slide.

## 20 | Takeaways: causal evidence and representation learning

These results connect causal evidence with representation learning. Matching shows positive effects on lights and local activity. Earlier information predicts a small part of the responses across cities. Learned representations have a higher mean retrieval score for lights, with variation across training runs. POI favors original features. Our next test combines original features with predicted responses. Validation cities choose the mix, and separate test cities measure the gain. The goal is to use response information while keeping useful local structure. Thank you.

