# Learning How Cities Respond: Intervention-Conditioned Urban Representations

## 01 | Learning How Cities Respond

Hello everyone. Today I will share a project on how places respond to rail station openings. We estimate local changes after an opening, then use those response patterns to guide urban representation learning. The planning question is whether evidence from past openings can help us compare places in a new city. I will start with the causal evidence, then show what the learning experiments tell us.

## 02 | Why do places respond differently?

A new station changes access to jobs and services. That change can enter housing prices, bring businesses closer together, and influence where people live. These mechanisms connect rail expansion with transit-oriented development, or TOD.

Their strength depends on the place. Network position, land supply, and local demand all matter. We therefore want to understand both the size of a response and how it develops over time. This question brings together transport evaluation and urban learning.

## 03 | Related Work: transport effects and causal methods

Transport studies link rail access with firms, employment, housing, and population. They also highlight anticipation before opening and spillovers into nearby areas. This motivates looking at a response path rather than one post-opening value.

Matching, generalized synthetic control, and matrix completion offer different ways to estimate the outcome without an opening. Event-study research helps us assess timing and pre-treatment trends. We bring these methods together in a dataset that keeps each response's source, support, and diagnostics.

## 04 | Related Work: urban representations and response learning

The learning literature offers a way to compare places through embeddings. Urban2Vec uses street views and POIs. RegionDCL uses buildings and POIs. FLORAL explores transfer to cities with limited data, while Weather2vec, CauOD, and CGNet-PIE connect spatial learning with causal questions or intervention effects.

Our link is response supervision. Pre-treatment features describe a place, and estimated opening responses guide which places should be close in the embedding space. That leads to the following problem setup.

## 05 | Problem setup: inputs, targets and tasks

For each grid-level opening event, we have pre-treatment features, a response signature, and a mask showing which response values are available. A signature records estimated effects at several post-opening horizons.

The encoder maps features into an embedding. Our main task asks whether its five nearest neighbors have similar response shapes. We also evaluate response magnitude and direction. In a held-out city, the encoder uses features alone; response labels provide the evaluation target. Here is the data behind these objects.

## 06 | Data: station openings across 44 cities

The dataset covers about five thousand station openings across forty-four cities, linked through fixed 500-meter grids. The causal analysis covers housing, nighttime lights, points of interest, or POI, and population. Current learning experiments focus on VIIRS nighttime lights and POI.

Lights capture monthly responses, while POI captures annual changes across several outcomes. We use lagged annual features and fit preprocessing on training cities. Coverage and design support determine which grid responses we can estimate.

## 07 | Causal labels: method routing and a frozen dataset

To estimate a response, we need a counterfactual: the outcome we would expect without the opening. The pipeline first tries within-city matching, then generalized synthetic control, or GSC, and matrix completion, or MC. Cross-city routes extend coverage when within-city support is insufficient.

We keep the first supported route. Each grid and outcome family receives one assigned estimator, together with its response values, observation masks, and diagnostics. We freeze this dataset before learning. Matching is the first route in that process.

## 08 | Matching: a grid-level difference-in-differences label

The formula compares two changes from baseline: one for the treated grid and one for its fixed matched control. Their difference gives the grid's response at each horizon. Repeating this comparison produces a dynamic profile.

For monthly outcomes, we leave an anticipation window before opening and use an earlier baseline. Annual outcomes use the previous year. Comparable untreated trends and suitable controls support the interpretation. Anticipation and spillovers inform these design choices.

## 09 | GSC and MC: shared structure for counterfactuals

GSC and MC use patterns shared across grids and time. In GSC, untreated controls identify common time factors. A treated grid's pre-period observations identify its loadings and intercept. Cross-validation selects the rank before we project its post-opening counterfactual.

MC fits observed untreated cells with fixed effects and a low-rank residual matrix. A nuclear norm penalty controls that structure. It then fills treated post-period entries. Both methods subtract the counterfactual from the observed outcome to obtain a response.

## 10 | Event study: pooled dynamics and grid-level labels

These estimators give grid-level profiles. The event study helps us see the broader pattern around opening. Unit and time fixed effects absorb stable local differences and common changes. Matched controls follow their treated event's clock, with the previous period as the zero reference.

The beta coefficients summarize pooled dynamics; the grid responses become learning labels. The released matching curves use two-way fixed effects, or TWFE. Staggered openings and heterogeneous effects shape their interpretation. The pre-opening periods provide our first diagnostic.

## 11 | Pre-treatment diagnostics: evidence by method

Matching uses a joint test of pre-treatment coefficients. At the usual five-percent threshold, the test retains the zero-pre-trend hypothesis for all outcomes. Housing, lights, and POI count sit closer to the boundary.

GSC and MC test mean pre-treatment gaps. GSC retains the zero-gap hypothesis for lights and POI count. Population is flagged in both panel methods, and MC flags POI count. We read these diagnostics alongside pre-treatment fit and sample support before examining the response curves.

## 12 | VIIRS: a sustained positive matched response

For nighttime lights, focus on the matching panel on the left. The response stays positive across the main post-opening horizons, and the later estimates are larger. Their city-level confidence intervals all lie above zero.

These effects use the asinh scale. Each method panel has its own sample and axis, so the comparisons need that context. The matched profile shows a sustained positive response over two years. POI lets us examine local activity over a longer annual window.

## 13 | POI count: positive activity responses after opening

POI count provides an activity proxy. The matched response is positive over the first three years, with all post-opening city-level confidence intervals above zero on the log count scale.

Support changes across horizons, so coverage also matters when reading the rising estimates. The light and POI results illustrate the response paths we want to compare. The learning step asks how to turn those paths into a similarity target.

## 14 | Learning target: response shape and consistency

We use weighted Pearson correlation over jointly observed response values. It compares the pattern across a vector. Shifts and positive rescaling preserve that pattern, which is why magnitude and direction need separate evaluation.

VIIRS describes one outcome over time, while the current POI target pools several outcomes. The split-half diagnostic finds much stronger similarity consistency for VIIRS. This motivates outcome-level POI analysis and gives context for the model results.

## 15 | Model: response supervision for pre-treatment features

The upper path takes pre-treatment features through a small multilayer perceptron, or MLP. A projection head produces a compact, unit-normalized embedding. Prediction heads read that embedding to reconstruct response values.

The lower path shows how frozen responses supervise training. They define pairwise similarity targets and cell-level prediction targets. At inference, a held-out city's features follow the upper path. The optional FIFO queue expands candidate pairs across batches. The historical main comparisons use tabular features.

## 16 | Joint objective: neighbors, similarity and response values

The three loss terms give this architecture its response supervision. Alignment assigns soft positive weights using response similarity. Distance fitting brings embedding cosine similarity toward the numerical response-similarity target. Prediction reconstructs available response values using masks and label weights.

Prediction receives half the total weight, with the pairwise terms sharing the other half. Constant within-grid weights cancel in Pearson correlation but still affect prediction. We judge the value of this supervision through held-out city retrieval and prediction.

## 17 | Evaluation: shape, magnitude and direction

For shape retrieval, we find five embedding neighbors and average their response similarity. Raw features, an appearance autoencoder, and random neighbors provide comparisons. RMSE measures magnitude error, and sign accuracy measures direction.

The direct controls use five-fold city cross-validation, with preprocessing and tuning inside each fold. The embeddings use an earlier city split, multiple training seeds, and a shared candidate pool. Each result uses its baseline under the same protocol. We start with the signal already present in the features.

## 18 | Result 1: features carry a held-out city response signal

The left panel shows that raw feature neighbors share more response similarity than random neighbors for both families.

In the middle, ridge reduces magnitude error relative to the training mean. On the right, it also improves direction accuracy. The clearest gain is for POI, at about five percentage points. These controls show a small predictive signal across cities. The embedding experiments ask whether response supervision can turn that signal into better retrieval.

## 19 | Result 2: retrieval with trained embeddings

These bars show the embedding results under the earlier city split. The appearance autoencoder learns from features alone; the supervised models also use response labels.

For VIIRS, supervised embeddings have a higher mean retrieval score than raw features. For POI, raw features remain stronger. The error bars show variation across training seeds for supervised models and reported ranges for the autoencoder. The pattern differs by response family, which is clearer when we plot gains relative to raw.

## 20 | Ablation: gains over raw feature retrieval

This plot subtracts the raw feature score. VIIRS shows a positive mean gain. Removing the queue raises that mean and increases seed variability. Both POI means remain below raw.

These are changes in correlation score; the earlier direction gains use percentage points. A higher mean can come with more training variation. This makes a paired city-level comparison the next step for assessing stability.

## 21 | Takeaways: causal evidence and cross-city learning

The project connects estimated local responses with comparisons across cities. Matched samples show positive VIIRS and POI count dynamics. Pre-treatment features carry a small predictive signal. Trained embeddings show an initial VIIRS gain, while POI favors raw geometry.

Our next test keeps raw features and adds predicted responses. Inner cities select the fusion weight; outer cities measure paired gains. Raw geometry remains an available choice. The evaluation is implemented, with results pending. Thank you, and I look forward to your questions.
