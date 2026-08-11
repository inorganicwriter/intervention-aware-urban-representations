suppressPackageStartupMessages({
  library(data.table)
  library(PanelMatch)
})

source(file.path("scripts", "causal_r", "formal_matching_lib.R"))

panel <- CJ(unit_id = 1:11, time_id = 1:4)
panel[, D := as.integer(unit_id == 1L & time_id >= 4L)]
panel[, Y := 0]
panel[, x1 := ifelse(unit_id == 1L, 1, 1 + (unit_id - 2L) * 0.7) + time_id * 0.03]
panel[, x2 := ifelse(unit_id == 1L, 2, 2 + (unit_id - 2L) * 0.4) - time_id * 0.02]
panel[unit_id == 2L, `:=`(x1 = 1.01 + time_id * 0.03, x2 = 2.01 - time_id * 0.02)]

panel_data <- PanelData(
  panel.data = as.data.frame(panel),
  unit.id = "unit_id",
  time.id = "time_id",
  treatment = "D",
  outcome = "Y"
)
panelmatch_result <- PanelMatch(
  panel.data = panel_data,
  lag = 3,
  refinement.method = "mahalanobis",
  qoi = "att",
  size.match = 1,
  match.missing = FALSE,
  covs.formula = ~ I(lag(x1, 1:3)) + I(lag(x2, 1:3)),
  lead = 0,
  forbid.treatment.reversal = TRUE,
  matching = TRUE,
  listwise.delete = TRUE,
  use.diagonal.variance.matrix = FALSE
)
panelmatch_weights <- attr(panelmatch_result$att[[1L]], "weights")
panelmatch_control <- as.integer(names(panelmatch_weights)[panelmatch_weights > 0][[1L]])

feature_row <- function(unit) {
  values <- panel[unit_id == unit & time_id %in% 1:3]
  c(rev(values$x1), rev(values$x2))
}
target <- feature_row(1L)
controls <- t(vapply(2:11, feature_row, numeric(6)))
colnames(controls) <- c(
  "x1__lag1", "x1__lag2", "x1__lag3",
  "x2__lag1", "x2__lag2", "x2__lag3"
)
formal_result <- abadie_imbens_nearest(
  target = target,
  controls = controls,
  control_ids = as.character(2:11),
  matches = 1L
)

stopifnot(as.integer(formal_result$selected_ids[[1L]]) == panelmatch_control)
stopifnot(formal_matching_spec()$matches_per_treated == 1L)
stopifnot(!formal_matching_spec()$post_treatment_data_used_for_matching)

cat("PanelMatch/Abadie-Imbens equivalence test passed.\n")
