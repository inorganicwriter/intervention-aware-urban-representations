suppressPackageStartupMessages({
  library(data.table)
  library(PanelMatch)
  library(Matching)
  library(gsynth)
})

source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

set.seed(20260722)
units <- 1:36
times <- 1:14
panel <- CJ(unit_id = units, time_id = times)
unit_x1 <- rnorm(length(units), sd = 0.20)
unit_x2 <- rnorm(length(units), sd = 0.20)
unit_x1[1:10] <- seq(-0.4, 0.4, length.out = 10L)
unit_x2[1:10] <- seq(0.4, -0.4, length.out = 10L)
unit_x1[11:36] <- seq(-2, 2, length.out = 26L)
unit_x2[11:36] <- seq(2, -2, length.out = 26L)
panel[, x1 := unit_x1[unit_id] + sin(time_id / 3) + rnorm(.N, sd = 0.02)]
panel[, x2 := unit_x2[unit_id] + cos(time_id / 4) + rnorm(.N, sd = 0.02)]
panel[, D := as.integer(unit_id %in% 1:10 & time_id >= 9L)]
loading <- rnorm(length(units))
factor <- cos(times / 2)
panel[, Y0 := loading[unit_id] * factor[time_id] + 0.3 * x1 - 0.2 * x2 +
  0.01 * unit_id + rnorm(.N, sd = 0.03)]
panel[, Y := Y0 + 0.6 * D]

# Complete Imai-Kim-Wang official workflow.
panel_data <- PanelData(
  panel.data = as.data.frame(panel), unit.id = "unit_id", time.id = "time_id",
  treatment = "D", outcome = "Y"
)
pm <- PanelMatch(
  panel.data = panel_data, lag = 3L, refinement.method = "mahalanobis",
  qoi = "att", size.match = 1L, match.missing = FALSE,
  covs.formula = ~ I(lag(x1, 1:3)) + I(lag(x2, 1:3)),
  lead = 0:2, forbid.treatment.reversal = TRUE, matching = TRUE,
  listwise.delete = TRUE, use.diagonal.variance.matrix = FALSE,
  placebo.test = TRUE
)
stopifnot(length(pm$att) == 10L)
balance <- get_covariate_balance(
  pm, panel.data = panel_data, covariates = c("x1", "x2"),
  include.unrefined = TRUE
)
effects <- lapply(0:2, function(lead) {
  get_set_treatment_effects(pm.obj = pm, panel.data = panel_data, lead = lead)
})
pm_estimate <- PanelEstimate(
  sets = pm, panel.data = panel_data, number.iterations = 50L,
  se.method = "bootstrap", include.placebo.test = TRUE, parallel = FALSE
)
stopifnot(!is.null(balance), length(effects) == 3L, inherits(pm_estimate, "PanelEstimate"))

# Complete Abadie-Imbens bias-corrected ATT with analytic variance.
wide <- panel[time_id %in% c(6L, 7L, 8L, 11L), .(
  Y_pre = Y[time_id == 8L], Y_post = Y[time_id == 11L],
  x1_l1 = x1[time_id == 8L], x1_l2 = x1[time_id == 7L], x1_l3 = x1[time_id == 6L],
  x2_l1 = x2[time_id == 8L], x2_l2 = x2[time_id == 7L], x2_l3 = x2[time_id == 6L]
), by = unit_id]
wide[, `:=`(delta = Y_post - Y_pre, Tr = as.integer(unit_id %in% 1:10))]
X <- as.matrix(wide[, .(x1_l1, x1_l2, x1_l3, x2_l1, x2_l2, x2_l3)])
stopifnot(
  nrow(wide) == 36L, sum(wide$Tr) == 10L, ncol(X) == 6L,
  all(is.finite(X)), all(is.finite(wide$delta))
)
ai <- Match(
  Y = wide$delta, Tr = wide$Tr, X = X, Z = X, estimand = "ATT", M = 1L,
  BiasAdjust = TRUE, replace = TRUE, ties = TRUE, CommonSupport = TRUE,
  Weight = 2L, Var.calc = 1L
)
stopifnot(length(unique(ai$index.treated)) > 2L, length(unique(ai$index.control)) > 2L)
balance_text <- capture.output(ai_balance <- MatchBalance(
  Tr ~ x1_l1 + x1_l2 + x1_l3 + x2_l1 + x2_l2 + x2_l3,
  data = as.data.frame(wide), match.out = ai, ks = FALSE, nboots = 50L,
  paired = FALSE, print.level = 1
))
stopifnot(
  length(ai$index.treated) > 0L, is.finite(ai$est), is.finite(ai$se),
  is.finite(ai$se.standard),
  length(balance_text) > 0L, !is.null(ai_balance)
)

# Complete Xu official gsynth workflow with CV and parametric uncertainty.
gsc <- gsynth(
  Y ~ D, data = panel, index = c("unit_id", "time_id"),
  estimator = "gsynth", force = "two-way", CV = TRUE, criterion = "mspe",
  r = 0:3, min.T0 = 5L, se = TRUE, inference = "parametric", nboots = 20L,
  parallel = FALSE, normalize = TRUE, seed = 20260722
)
stopifnot(
  inherits(gsc, "gsynth"), is.finite(gsc$r.cv), !is.null(gsc$Y.ct),
  all(is.finite(gsc$Y.ct))
)

# Complete Athey et al. matrix-completion workflow with lambda CV. Treated
# post-period outcomes must not alter the fitted untreated counterfactual.
mc_args <- list(
  formula = Y ~ D, data = NULL, index = c("unit_id", "time_id"),
  method = "mc", force = "two-way", CV = TRUE, criterion = "mspe",
  nlambda = 8L, min.T0 = 1L, se = FALSE, parallel = FALSE,
  cv.method = "rolling", cv.nobs = 1L, cv.donut = 0L, cv.buffer = 0L,
  normalize = FALSE, seed = 20260725
)
mc_input <- copy(panel)
mc_input[D == 1L, Y := mean(mc_input[unit_id == 1L & D == 0L, Y])]
# Fit once with the observed treated post-period values and once with the
# runner's pre-treatment-only replacement. The counterfactual must be
# invariant to the observed treatment response.
mc_args$data <- panel
set.seed(20260725)
mc_observed <- do.call(fect::fect, mc_args)
mc_args$data <- mc_input
set.seed(20260725)
mc <- do.call(fect::fect, mc_args)
panel_shifted <- copy(panel)
panel_shifted[D == 1L, Y := Y + 100]
mc_args$data <- panel_shifted
set.seed(20260725)
mc_shifted <- do.call(fect::fect, mc_args)
mc_final_args <- mc_args
mc_final_args$CV <- FALSE
mc_final_args$lambda <- mc$lambda.cv
mc_final_args$se <- FALSE
mc_final <- do.call(fect::fect, mc_final_args)
stopifnot(
  inherits(mc, "fect"), identical(mc$method, "mc"),
  length(mc$lambda.cv) == 1L, is.finite(mc$lambda.cv), mc$lambda.cv > 0,
  !is.null(mc$CV.out.mc), "MSPE" %in% colnames(mc$CV.out.mc),
  inherits(mc_final, "fect"), identical(mc_final$method, "mc"),
  all(is.finite(mc_final$Y.ct)),
  all(is.finite(mc$Y.ct)),
  isTRUE(all.equal(mc_observed$Y.ct, mc$Y.ct, tolerance = 1e-10)),
  isTRUE(all.equal(mc$Y.ct, mc_shifted$Y.ct, tolerance = 1e-10))
)

# The formal runners must retain the registered full settings.
registered <- complete_estimator_spec()
stopifnot(
  registered$panelmatch$number.iterations == 1000L,
  registered$panelmatch$placebo.test,
  registered$abadie_imbens$BiasAdjust,
  registered$abadie_imbens$Var.calc == 1L,
  registered$xu_gsc$nboots == 200L,
  identical(registered$xu_gsc$r, 0:5),
  identical(registered$mc$min.T0, 1L),
  identical(registered$mc$backend, "fect"),
  identical(registered$mc$cv_nobs, 1L),
  identical(registered$mc$cv_donut, 0L),
  identical(registered$mc$cv_buffer, 0L),
  isTRUE(registered$mc$two_stage_cv_inference),
  identical(registered$mc$CV, TRUE),
  identical(registered$mc$criterion, "mspe"),
  identical(registered$mc$inference, "jackknife")
)

cat("All four complete published-estimator workflows passed synthetic tests.\n")
