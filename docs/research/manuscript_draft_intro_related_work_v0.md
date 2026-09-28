# Learning How Cities Respond: Intervention-Conditioned Urban Representations

> Draft scope: Introduction and Related Work. Citations have been added from an initial targeted search and bibliographic details checked against publisher, conference, or author pages. The search is not exhaustive. Claims about final label coverage, model performance, and cross-city gains are intentionally not included until the formal response release is complete.

## 1. Introduction

Metro investments can reshape housing markets, business activity, and population distribution, with effects that vary across urban areas (Mayer & Trevien, 2017; Diao et al., 2017; Bardaka et al., 2018). This variation matters when planners use evidence from existing networks to assess proposed stations. They need to understand which neighborhoods may respond similarly and how far evidence from another city applies. Comparing these responses requires both credible estimates of intervention effects and a way to relate them to local urban characteristics.

Causal evaluation and urban representation learning offer complementary approaches to this problem. Transport studies estimate effects on specific outcomes, including housing values, employment, and urban spatial structure (Diao et al., 2017; Gonzalez-Navarro & Turner, 2018). Urban embeddings summarize place characteristics and support neighborhood comparison and prediction (Wang et al., 2020; Li et al., 2023). Cross-city learning extends these capabilities to settings with limited data (Wei et al., 2016; Yang et al., 2025). However, similarity in urban characteristics does not necessarily imply similarity in policy response. The resulting challenge is to connect estimated intervention effects with representations that support comparisons across cities.

This connection requires careful treatment of the evidence used for learning. Counterfactual estimation depends on identifying assumptions, adequate comparison units, and sufficient data before the intervention (Xu, 2017; Athey et al., 2021). Applying these requirements across thousands of urban units entails aligning intervention dates, assembling outcome histories, and assessing the support for each estimate. Differences in data coverage further complicate comparisons across cities and outcomes. A useful representation must therefore draw on response estimates whose construction and diagnostic results can be examined.

We propose a framework for learning intervention-conditioned urban representations from metro station openings in 44 Chinese cities. The study design comprises 5,048 treated grids, each measuring 500 m by 500 m. The label pipeline estimates intervention effects by comparing observed outcomes with estimated untreated counterfactuals, subject to data support and diagnostic checks. Outcomes are measured using housing prices, nighttime lights, points of interest, and population data. A response signature collects these estimated effects across outcomes and time horizons. The encoder takes pre-treatment urban features as inputs and uses response signatures as supervision.

The framework defines urban similarity through estimated responses to a common intervention. It records the estimator, data support, and diagnostic results associated with each response label, allowing differences in label quality to remain visible. Its evaluation protocol holds out entire cities to test generalization beyond training locations. Together, these components address whether evidence from past metro openings can support comparisons of likely responses in other urban areas.

## 2. Related Work and Research Gap

### 2.1 Causal evaluation of urban transport interventions

Research on urban transport has examined effects at several spatial scales. Mayer and Trevien (2017) study how rail access affects employment, firm location, and population in the Paris region. Gonzalez-Navarro and Turner (2018) examine subway networks, population growth, and urban spatial structure across cities worldwide. At the neighborhood scale, Diao et al. (2017) account for spatial dependence and anticipation when estimating housing price effects in Singapore. Bardaka et al. (2018) examine neighborhood socioeconomic change and spatial spillovers around Denver light rail stations. These studies establish the importance of outcome choice, spatial context, and intervention timing in interpreting transport effects.

Callaway and Sant’Anna (2021) and Sun and Abraham (2021) address heterogeneous effects under staggered treatment adoption. Matching methods construct comparison sets using treatment histories and covariates (Imai et al., 2023). Synthetic control constructs a weighted comparison unit using information from before treatment (Abadie et al., 2010). Generalized synthetic control estimates untreated outcomes through interactive fixed effects (Xu, 2017), while matrix completion reconstructs missing counterfactual outcomes using low-rank structure (Athey et al., 2021).

These methods require different assumptions and forms of data support. Difference-in-differences identification relies on parallel trends, potentially conditional on observed covariates (Callaway & Sant’Anna, 2021). Generalized synthetic control and matrix completion instead place restrictions on the structure used to reconstruct untreated outcomes (Xu, 2017; Athey et al., 2021). Pre-treatment fit and balance diagnostics help assess the suitability of a design, although they cannot establish the validity of its identifying assumptions. For comparisons across many locations, researchers must also account for differences in available histories, eligible controls, and exposure to nearby interventions. These requirements motivate explicit documentation of how each response estimate was obtained before using it as a learning target.

### 2.2 Urban representation learning and cross-city transfer

Urban representation methods encode information about the structure, functions, and activities of places. Yao et al. (2017) integrate points of interest (POIs) with word embeddings to characterize urban land use. Urban2Vec combines street-view imagery and POI information to learn neighborhood embeddings for prediction and similarity analysis (Wang et al., 2020). Zhang et al. (2020) combine social media check-ins and street-view images to characterize places and differences in their use. Li et al. (2023) extend region representation learning to OpenStreetMap building footprints. These approaches illustrate how different data sources and learning objectives define the information preserved in urban representations.

Cross-city learning seeks to make knowledge from one urban setting useful in another. Wei et al. (2016) develop a multimodal transfer method for cities with limited data and demonstrate it through air quality prediction. Yang et al. (2025) review applications and challenges across urban prediction, detection, and deployment tasks. Such work supports the use of shared representations while drawing attention to differences in data availability and urban context.

Transfer performance nevertheless depends on the target task and the information captured during training. Embeddings trained to describe land use or visual characteristics organize places according to those characteristics. The usefulness of these embeddings for comparing intervention responses requires separate evaluation. For metro planning, the relevant comparison concerns how neighborhoods change following station openings relative to their estimated untreated trajectories. This motivates a learning objective that explicitly incorporates estimated intervention responses.

### 2.3 From causal response estimates to intervention-conditioned representations

Several studies already connect causal methods with representation learning or urban prediction. Weather2vec learns spatial representations to adjust for non-local confounding in environmental causal analyses (Tec et al., 2023). CauOD incorporates causal learning into origin–destination prediction for urban planning (Zeng et al., 2022). CGNet-PIE combines generative modeling with a potential-outcomes framework to estimate urban planning intervention effects (Fang et al., 2025). In the broader literature, Ahuja et al. (2023) investigate identification of latent causal factors from interventional data. These studies provide relevant foundations for connecting urban features, interventions, and outcomes.

Our study focuses on learning spatial similarity from multidimensional response signatures estimated for actual metro openings. An intervention-conditioned urban representation encodes pre-treatment features under supervision from these signatures. Its training objective encourages nearby embeddings for urban units with similar estimated responses. The distinction lies in using response signatures to organize urban space and testing whether that organization transfers across cities.

This design also requires a clear separation between encoder inputs and supervision. Post-treatment outcomes contribute to response estimation, while the encoder receives only features available before station opening. The resulting representation can therefore be computed for a target area using its pre-treatment information. Estimated responses remain subject to the assumptions and uncertainty of their counterfactual designs. Evaluation must consequently consider both the support for the labels and performance in cities excluded from representation training.

## 3. Provisional positioning statement

The central positioning of this paper is:

> **Causal estimation provides intervention-response supervision, while representation learning makes that response knowledge reusable across urban space and transferable across cities.**

This framing distinguishes the study from both conventional causal evaluation and generic urban embedding. The paper does not claim that representation learning establishes causal identification on its own. Instead, it asks whether carefully constructed causal response labels can be transformed into a transferable representation of how cities respond.

## References

Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies: Estimating the effect of California’s tobacco control program. *Journal of the American Statistical Association, 105*(490), 493–505. https://doi.org/10.1198/jasa.2009.ap08746

Ahuja, K., Mahajan, D., Wang, Y., & Bengio, Y. (2023). Interventional causal representation learning. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 372–407). PMLR. https://proceedings.mlr.press/v202/ahuja23a.html

Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021). Matrix completion methods for causal panel data models. *Journal of the American Statistical Association, 116*(536), 1716–1730. https://doi.org/10.1080/01621459.2021.1891924

Bardaka, E., Delgado, M. S., & Florax, R. J. G. M. (2018). Causal identification of transit-induced gentrification and spatial spillover effects: The case of the Denver light rail. *Journal of Transport Geography, 71*, 15–31. https://doi.org/10.1016/j.jtrangeo.2018.06.025

Callaway, B., & Sant’Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics, 225*(2), 200–230. https://doi.org/10.1016/j.jeconom.2020.12.001

Diao, M., Leonard, D., & Sing, T. F. (2017). Spatial-difference-in-differences models for impact of new mass rapid transit line on private housing values. *Regional Science and Urban Economics, 67*, 64–77. https://doi.org/10.1016/j.regsciurbeco.2017.08.006

Fang, Z., Zheng, Q., Zheng, S., & Zhao, L. (2025). CGNet-PIE: A causal generative approach for urban planning intervention effect estimation. *Transactions in Urban Data, Science, and Technology, 4*(1), 3–21. https://doi.org/10.1177/27541231251318348

Gonzalez-Navarro, M., & Turner, M. A. (2018). Subways and urban growth: Evidence from earth. *Journal of Urban Economics, 108*, 85–106. https://doi.org/10.1016/j.jue.2018.09.002

Imai, K., Kim, I. S., & Wang, E. (2023). Matching methods for causal inference with time-series cross-sectional data. *American Journal of Political Science, 67*(3), 587–605. https://doi.org/10.1111/ajps.12685

Li, Y., Huang, W., Cong, G., Wang, H., & Wang, Z. (2023). Urban region representation learning with OpenStreetMap building footprints. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 1363–1373). ACM. https://doi.org/10.1145/3580305.3599538

Mayer, T., & Trevien, C. (2017). The impact of urban public transportation: Evidence from the Paris region. *Journal of Urban Economics, 102*, 1–21. https://doi.org/10.1016/j.jue.2017.07.003

Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics, 225*(2), 175–199. https://doi.org/10.1016/j.jeconom.2020.09.006

Tec, M., Scott, J. G., & Zigler, C. M. (2023). Weather2vec: Representation learning for causal inference with non-local confounding in air pollution and climate studies. *Proceedings of the AAAI Conference on Artificial Intelligence, 37*(12), 14504–14513. https://doi.org/10.1609/aaai.v37i12.26696

Wang, Z., Li, H., & Rajagopal, R. (2020). Urban2Vec: Incorporating street view imagery and POIs for multi-modal urban neighborhood embedding. *Proceedings of the AAAI Conference on Artificial Intelligence, 34*(1), 1013–1020. https://doi.org/10.1609/aaai.v34i01.5450

Wei, Y., Zheng, Y., & Yang, Q. (2016). Transfer knowledge between cities. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1905–1914). ACM. https://doi.org/10.1145/2939672.2939830

Xu, Y. (2017). Generalized synthetic control method: Causal inference with interactive fixed effects models. *Political Analysis, 25*(1), 57–76. https://doi.org/10.1017/pan.2016.2

Yang, Y., Zhan, J., Liu, Y., & Wang, Q. (2025). Cross-city transfer learning: Applications and challenges for smart cities and sustainable transportation. *Communications in Transportation Research, 5*(3), 100206. https://doi.org/10.1016/j.commtr.2025.100206

Yao, Y., Li, X., Liu, X., Liu, P., Liang, Z., Zhang, J., & Mai, K. (2017). Sensing spatial distribution of urban land use by integrating points-of-interest and Google Word2Vec model. *International Journal of Geographical Information Science, 31*(4), 825–848. https://doi.org/10.1080/13658816.2016.1244608

Zeng, J., Zhang, G., Rong, C., Ding, J., Yuan, J., & Li, Y. (2022). Causal learning empowered OD prediction for urban planning. In *Proceedings of the 31st ACM International Conference on Information & Knowledge Management* (pp. 2455–2464). ACM. https://doi.org/10.1145/3511808.3557255

Zhang, F., Zu, J., Hu, M., Zhu, D., Kang, Y., Gao, S., Zhang, Y., & Huang, Z. (2020). Uncovering inconspicuous places using social media check-ins and street view images. *Computers, Environment and Urban Systems, 81*, 101478. https://doi.org/10.1016/j.compenvurbsys.2020.101478
