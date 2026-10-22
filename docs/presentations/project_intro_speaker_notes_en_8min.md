# Learning How Cities Respond: 8-minute speaker script

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

Here are the pre-opening checks. Matching's city-level tests show little evidence of different earlier trends, though some outcomes are near the cutoff. For GSC and MC, we also examine earlier fit and supporting data.

## 11 | Nighttime lights: positive effects after opening

The first result is nighttime lights. Look at Matching on the left: the treated group has a positive estimated response relative to its matched control group. The effect grows later, and the main intervals sit above zero.

## 12 | Local activity: more recorded places after opening

POI gives another view of local activity. Matching shows higher counts relative to the matched control group after opening. Coverage varies across years. These local response paths now guide representation learning.

## 13 | Learning target: response shape and consistency

To compare places using these paths, we mainly compare how they rise and fall together. Size and direction receive separate checks. Light responses are more consistent. POI combines several outcomes, so each outcome also needs a closer look.

## 14 | Representation learning: features to embeddings

Representation learning trains the model to turn earlier local information into a short numerical summary, called a learned representation, or embedding. Response paths guide the training. In a new city, the model uses earlier information alone. We train it with three related goals.

## 15 | Joint objective: neighbors, similarity and response values

To learn this representation, the model has three jobs: bring places with similar responses closer, match their similarity scores, and predict response values. We then check whether the learned representation helps in new cities.

## 16 | Evaluation: shape, magnitude and direction

We check similar paths, prediction error, and the direction of change. Direct predictions use city cross-validation; learned representations use an earlier city split. Each comparison has its own baseline. We start with the original features as a baseline.

## 17 | Result 1: do local features help predict responses?

The original features carry a small signal: similar-looking areas have more similar responses than random neighbors. A simple prediction model also helps. For POI, it gets direction right about five more times per hundred predictions than choosing the training set's most common direction. This gives us a baseline for testing learned representations.

## 18 | Result 2: do learned representations find better neighbors?

Next, we compare learned representations with the original features. For lights, they find neighbors with a higher average response-similarity score. For POI, the original features do better. The error bars show variation across training runs, which the next plot makes easier to see.

## 19 | Training settings: do the gains hold up?

Here, zero is the original-feature baseline; right means improvement. Removing stored examples from training raises the average light score and its variation. POI stays below the baseline on average. The gains depend on the outcome and training setting.

## 20 | Takeaways: causal evidence and representation learning

These results connect causal evidence with representation learning. Matching shows positive effects on lights and local activity. Earlier information predicts a small part of the responses. Learned representations have a higher mean retrieval score for lights; POI favors original features. Our next test combines original features with predicted responses. Thank you.

