suppressPackageStartupMessages({
  library(data.table)
  library(PanelMatch)
})

source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

# This is an integration gate, not a formal estimate. The production runner never
# applies this 300-donor test fixture restriction.
design <- build_annual_estimator_panel(
  "xiamen", 2019L, "population", "population+viirs", leads = 1:3
)
needed_columns <- c("population_log", design$covariates)
complete_ids <- design$panel[, .(
  complete = all(complete.cases(.SD))
), by = .(unit_id, role), .SDcols = needed_columns]
treated_id <- complete_ids[role == "treated" & complete, unit_id][[1L]]
donor_ids <- head(complete_ids[role == "donor" & complete, unit_id], 300L)
stopifnot(length(treated_id) == 1L, length(donor_ids) == 300L)

panel <- design$panel[unit_id %in% c(treated_id, donor_ids), .(
  original_unit_id = unit_id, time_id, D, population_log,
  viirs_avg_asinh
)]
map <- data.table(original_unit_id = sort(unique(panel$original_unit_id)))
map[, unit_id := seq_len(.N)]
panel <- merge(panel, map, by = "original_unit_id")
setorder(panel, unit_id, time_id)
panel[, Y := population_log]

panel_data <- PanelData(
  panel.data = as.data.frame(panel), unit.id = "unit_id", time.id = "time_id",
  treatment = "D", outcome = "Y"
)
pm <- PanelMatch(
  panel.data = panel_data, lag = 3L, refinement.method = "mahalanobis",
  qoi = "att", size.match = 1L, match.missing = FALSE,
  covs.formula = ~ I(lag(population_log, 1:3)) +
    I(lag(viirs_avg_asinh, 1:3)),
  # PanelMatch lead 0 is the first treated period, which is research
  # event_time 1 because the partial opening year is excluded.
  lead = design$leads - 1L,
  forbid.treatment.reversal = TRUE, matching = TRUE,
  listwise.delete = TRUE, use.diagonal.variance.matrix = FALSE,
  placebo.test = TRUE
)
balance <- get_covariate_balance(
  pm, panel.data = panel_data,
  covariates = c("population_log", "viirs_avg_asinh"),
  include.unrefined = TRUE
)
effects <- lapply(design$leads - 1L, function(lead) {
  get_set_treatment_effects(pm.obj = pm, panel.data = panel_data, lead = lead)
})
estimate <- PanelEstimate(
  sets = pm, panel.data = panel_data, number.iterations = 50L,
  se.method = "bootstrap", include.placebo.test = TRUE, parallel = FALSE
)
stopifnot(
  length(pm$att) == 1L, !is.null(balance), length(effects) == 3L,
  inherits(estimate, "PanelEstimate")
)

output <- file.path(
  project_root(), "outputs", "complete_estimators", "validation",
  "real_panelmatch_gate"
)
dir.create(output, recursive = TRUE, showWarnings = FALSE)
saveRDS(
  list(pm = pm, balance = balance, effects = effects, estimate = estimate),
  file.path(output, "gate_objects.rds"), compress = "xz"
)
fwrite(data.table(
  status = "passed", city_key = "xiamen", cohort = 2019L,
  treated_units = 1L, test_fixture_donors = 300L,
  production_donors = nrow(design$donors),
  bootstrap_iterations_test_only = 50L,
  formal_estimate = FALSE,
  note = paste(
    "Integration gate only. Production runner uses all same-city eligible donors",
    "and 1000 bootstrap iterations."
  )
), file.path(output, "gate_manifest.csv"), bom = TRUE)

cat("Real-data PanelMatch integration gate passed; no formal estimate was created.\n")
