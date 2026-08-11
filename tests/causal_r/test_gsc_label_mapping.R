suppressPackageStartupMessages(library(data.table))
source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

treated <- data.table(
  gsc_unit_id = c(2L, 4L), treatment_order = c(10L, 20L),
  city_key = c("a", "b"), grid_id = c("g1", "g2")
)
panel <- CJ(gsc_unit_id = c(2L, 4L), time_id = 1:5)
panel[, value := 10 * gsc_unit_id + time_id]
fit <- list(
  id = c(4L, 2L),
  r.cv = 1L,
  Y.ct = cbind(
    `4` = panel[gsc_unit_id == 4L][order(time_id), value] - c(0, 0, 0, 2, 3),
    `2` = panel[gsc_unit_id == 2L][order(time_id), value] - c(0, 0, 0, 5, 7)
  )
)
paths <- normalize_gsc_labels(
  fit, panel, treated, pre_periods = 1:3, post_periods = 4:5,
  outcome_family = "viirs", outcome = "viirs_avg_asinh"
)
stopifnot(
  paths[treatment_order == 10L & event_time == 1L, causal_response_label] == 5,
  paths[treatment_order == 10L & event_time == 2L, causal_response_label] == 7,
  paths[treatment_order == 20L & event_time == 1L, causal_response_label] == 2,
  identical(sort(unique(paths$event_time)), c(-3L, -2L, -1L, 1L, 2L)),
  all(paths$label_available)
)

fit$est.att <- cbind(
  ATT = paths$causal_response_label,
  `S.E.` = rep(0.25, nrow(paths)),
  CI.lower = paths$causal_response_label - 0.5,
  CI.upper = paths$causal_response_label + 0.5,
  p.value = rep(0.10, nrow(paths)), count = rep(1, nrow(paths))
)
one_target <- paths[treatment_order == 10L]
single_fit <- fit
single_fit$est.att <- fit$est.att[paths$treatment_order == 10L, , drop = FALSE]
# The production code stamps uncertainty_source from class(fit)[[1L]]; mimic a
# real gsynth object so the fixture exercises the production contract.
class(single_fit) <- "gsynth"
with_uncertainty <- attach_single_target_gsc_uncertainty(
  one_target, single_fit, nboots = 200L, effect_scale = 2
)
stopifnot(
  all(with_uncertainty$standard_error == 0.50),
  all(with_uncertainty$confidence_lower ==
      single_fit$est.att[, "CI.lower"] * 2),
  all(with_uncertainty$bootstrap_repetitions == 200L),
  all(with_uncertainty$uncertainty_source == "gsynth_parametric_bootstrap")
)

cat("GSC counterfactual columns map to treatment order and event time correctly.\n")
