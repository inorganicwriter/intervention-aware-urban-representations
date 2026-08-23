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
  outcome_family = "viirs", outcome = "viirs_avg_asinh",
  pre_event_time = c(-3L, -2L, -1L)
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
# gsynth identifies inference rows through row names containing event_time.
# A default 1:n row index is unsafe and must not be used for alignment.
rownames(single_fit$est.att) <- as.character(one_target$event_time)
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

stopifnot(identical(
  event_time_from_period(
    as.IDate(c("2016-06-01", "2016-07-01", "2016-08-01")),
    as.IDate("2019-12-01"), "monthly"
  ),
  c(-42L, -41L, -40L)
))

# Estimator event-time rows use the contiguous model index.  They must still
# align to a calendar-offset path when the anticipation gap is excluded.
calendar_paths <- normalize_gsc_labels(
  fit, panel, treated[1L], pre_periods = 1:3, post_periods = 4:5,
  outcome_family = "viirs", outcome = "viirs_avg_asinh",
  pre_event_time = c(-42L, -41L, -40L)
)
calendar_fit <- single_fit
calendar_fit$est.att <- rbind(
  single_fit$est.att[2:3, , drop = FALSE],
  single_fit$est.att[3L, , drop = FALSE],
  single_fit$est.att[4:5, , drop = FALSE]
)
rownames(calendar_fit$est.att) <- c("-2", "-1", "0", "1", "2")
calendar_uncertainty <- attach_single_target_gsc_uncertainty(
  calendar_paths, calendar_fit, nboots = 200L, effect_scale = 2
)
stopifnot(
  is.na(calendar_uncertainty$standard_error[1L]),
  all(calendar_uncertainty$standard_error[2:5] == 0.50)
)

cat("GSC counterfactual columns map to treatment order and event time correctly.\n")
