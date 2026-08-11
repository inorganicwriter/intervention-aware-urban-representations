suppressPackageStartupMessages({
  library(data.table)
  library(Matching)
})
source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

set.seed(20260723)
unit_map <- data.table(
  unit_id = 1:9,
  unit_key = paste0("u", 1:9),
  grid_id = paste0("g", 1:9),
  role = ifelse(1:9 == 5L, "treated", "donor")
)
panel <- CJ(unit_id = 1:9, time_id = 1:6)
panel <- merge(panel, unit_map, by = "unit_id")
panel[, x := unit_id / 10 + time_id / 100 + rnorm(.N, sd = 0.015)]
panel[, y := x + rnorm(.N, sd = 0.01)]
design <- list(
  panel = panel,
  unit_map = unit_map,
  covariates = "x",
  covariate_lags = 1:3,
  lag = 3L
)

prepared <- make_preonly_matching_frame(design)
selected_before <- select_preonly_pairs(prepared)
outcome_before <- attach_prepost_outcome(prepared$frame, design, "y", 2L)

# Neither changing post-treatment values nor changing post-treatment missingness
# may alter the pre-treatment design or selected controls.
mutated <- design
mutated$panel[time_id > 3L, y := y + 1000 * unit_id]
mutated$panel[unit_id == selected_before$pairs$control_unit_id[[1L]] & time_id == 5L, y := NA_real_]
prepared_after <- make_preonly_matching_frame(mutated)
selected_after <- select_preonly_pairs(prepared_after)
outcome_after <- attach_prepost_outcome(prepared_after$frame, mutated, "y", 2L)

stopifnot(
  identical(prepared$frame, prepared_after$frame),
  identical(
    selected_before$pairs[, .(treated_unit_id, control_unit_id)],
    selected_after$pairs[, .(treated_unit_id, control_unit_id)]
  ),
  any(outcome_before$delta_outcome != outcome_after$delta_outcome, na.rm = TRUE),
  anyNA(outcome_after$delta_outcome)
)

calendar <- monthly_event_calendar("2019-12", anticipation_months = 6L)
stopifnot(
  calendar$clean_pre_end == as.IDate("2019-05-01"),
  calendar$first_treated_month == as.IDate("2020-01-01"),
  identical(as.character(calendar$excluded_months),
            as.character(seq(as.IDate("2019-06-01"), as.IDate("2019-12-01"), by = "1 month"))),
  !calendar$opening_month %in% calendar$model_months
)

cat("Pre-only matching leakage and event-time contracts passed.\n")
