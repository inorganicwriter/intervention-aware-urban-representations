suppressPackageStartupMessages({
  library(data.table)
  if (!requireNamespace("PanelMatch", quietly = TRUE)) {
    stop("PanelMatch package is not installed. Run: install.packages('PanelMatch')")
  }
  library(PanelMatch)
})

source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L || length(args) > 5L) {
  stop(paste(
    "Usage: run_complete_panelmatch.R CITY COHORT OUTCOME_FAMILY",
    "[SIGNATURE=auto] [FREQUENCY=annual]"
  ))
}
city_key <- args[[1L]]
cohort <- args[[2L]]
outcome_family <- args[[3L]]
signature <- if (length(args) >= 4L) args[[4L]] else "auto"
frequency <- if (length(args) >= 5L) args[[5L]] else "annual"
assert_choice(frequency, c("annual", "monthly"), "frequency")
if (frequency == "monthly" && !outcome_family %in% c("housing", "viirs")) {
  stop("Monthly PanelMatch is defined for housing and VIIRS")
}

spec <- complete_estimator_spec()
design <- if (frequency == "annual") {
  build_annual_estimator_panel(
    city_key, as.integer(cohort), outcome_family, signature,
    leads = spec$annual$leads
  )
} else {
  build_monthly_estimator_panel(
    city_key, cohort, outcome_family, signature, leads = spec$monthly$leads
  )
}

extract_matched_sets <- function(sets, unit_map) {
  set_list <- sets$att
  if (is.null(set_list) || !length(set_list)) return(data.table())
  rows <- rbindlist(lapply(seq_along(set_list), function(index) {
    matched_set <- set_list[[index]]
    weights <- attr(matched_set, "weights")
    if (is.null(weights)) {
      control_ids <- as.character(unlist(matched_set, use.names = FALSE))
      selected_weights <- rep(1 / length(control_ids), length(control_ids))
    } else {
      keep <- is.finite(weights) & weights > 0
      control_ids <- names(weights)[keep]
      selected_weights <- as.numeric(weights[keep])
    }
    label <- names(set_list)[[index]]
    pieces <- strsplit(label, ".", fixed = TRUE)[[1L]]
    data.table(
      matched_set = label,
      treated_unit_id = suppressWarnings(as.integer(pieces[[1L]])),
      treatment_time_id = suppressWarnings(as.integer(tail(pieces, 1L))),
      control_unit_id = as.integer(control_ids),
      weight = selected_weights
    )
  }), fill = TRUE)
  treated_map <- unit_map[, .(
    treated_unit_id = unit_id, treated_unit_key = unit_key,
    treated_grid_id = grid_id
  )]
  control_map <- unit_map[, .(
    control_unit_id = unit_id, control_unit_key = unit_key,
    control_grid_id = grid_id
  )]
  merge(merge(rows, treated_map, by = "treated_unit_id", all.x = TRUE),
        control_map, by = "control_unit_id", all.x = TRUE)
}

run_one_outcome <- function(outcome) {
  needed <- unique(c("unit_id", "time_id", "D", outcome, design$covariates))
  panel <- design$panel[, ..needed]
  if (all(is.na(panel[[outcome]]))) stop("Outcome is entirely missing: ", outcome)
  panel[, Y := get(outcome)]
  outcome_panel_data <- PanelData(
    panel.data = as.data.frame(panel),
    unit.id = "unit_id", time.id = "time_id", treatment = "D", outcome = "Y"
  )
  # Freeze matched sets with a placeholder outcome. PanelMatch refinement uses
  # treatment histories and pre-treatment covariates; post outcomes are read
  # only after matched-set identities have been persisted.
  design_panel <- copy(panel)
  design_panel[, Y := 0]
  design_panel_data <- PanelData(
    panel.data = as.data.frame(design_panel),
    unit.id = "unit_id", time.id = "time_id", treatment = "D", outcome = "Y"
  )
  covariate_formula <- panelmatch_covariate_formula(
    design$covariates, design$covariate_lags
  )
  pm <- PanelMatch(
    panel.data = design_panel_data,
    lag = design$lag,
    refinement.method = spec$panelmatch$refinement.method,
    qoi = spec$panelmatch$qoi,
    size.match = spec$panelmatch$size.match,
    match.missing = spec$panelmatch$match.missing,
    covs.formula = covariate_formula,
    lead = design$leads - 1L,
    verbose = TRUE,
    forbid.treatment.reversal = spec$panelmatch$forbid.treatment.reversal,
    matching = spec$panelmatch$matching,
    listwise.delete = spec$panelmatch$listwise.delete,
    use.diagonal.variance.matrix = spec$panelmatch$use.diagonal.variance.matrix,
    placebo.test = spec$panelmatch$placebo.test
  )
  if (is.null(pm$att) || !length(pm$att)) stop("PanelMatch produced no ATT matched sets")
  balance <- get_covariate_balance(
    pm, panel.data = design_panel_data, covariates = design$covariates,
    include.unrefined = TRUE
  )
  set_effects <- setNames(lapply(design$leads - 1L, function(lead) {
    get_set_treatment_effects(pm.obj = pm, panel.data = outcome_panel_data, lead = lead)
  }), paste0("event_time_", design$leads))
  estimate <- PanelEstimate(
    sets = pm,
    panel.data = outcome_panel_data,
    number.iterations = spec$panelmatch$number.iterations,
    se.method = spec$panelmatch$se.method,
    include.placebo.test = spec$panelmatch$placebo.test,
    parallel = FALSE
  )

  output <- estimator_output_dir(
    "panelmatch", city_key, cohort, outcome, design$signature
  )
  saveRDS(pm, file.path(output, "panelmatch_object.rds"), compress = "xz")
  saveRDS(estimate, file.path(output, "panelestimate_object.rds"), compress = "xz")
  saveRDS(balance, file.path(output, "covariate_balance.rds"), compress = "xz")
  saveRDS(set_effects, file.path(output, "matched_set_effects.rds"), compress = "xz")
  saveRDS(design_panel_data, file.path(output, "preonly_paneldata_object.rds"), compress = "xz")
  saveRDS(outcome_panel_data, file.path(output, "outcome_paneldata_object.rds"), compress = "xz")
  write_parquet(panel, file.path(output, "estimation_panel.parquet"), compression = "zstd")
  fwrite(design$unit_map, file.path(output, "unit_map.csv"), bom = TRUE)
  matched <- extract_matched_sets(pm, design$unit_map)
  fwrite(matched, file.path(output, "matched_sets.csv"), bom = TRUE)
  writeLines(capture.output(print(balance)), file.path(output, "covariate_balance.txt"))
  writeLines(capture.output(print(summary(estimate))), file.path(output, "panelestimate_summary.txt"))
  write_run_manifest(output, list(
    schema = spec$schema, method = "Imai-Kim-Wang PanelMatch",
    package = paste0("PanelMatch ", packageVersion("PanelMatch")),
    city_key = city_key, cohort = cohort, frequency = frequency,
    outcome_family = outcome_family, outcome = outcome,
    signature = design$signature, treatment_units = nrow(design$treated),
    eligible_same_city_donors = nrow(design$donors),
    matched_sets = length(pm$att), lag = design$lag,
    covariate_lags = design$covariate_lags, event_time = design$leads,
    panelmatch_zero_based_leads = design$leads - 1L,
    control_selection_uses_post_outcome = FALSE,
    refinement_method = spec$panelmatch$refinement.method,
    size_match = spec$panelmatch$size.match,
    bootstrap_iterations = spec$panelmatch$number.iterations,
    placebo_test = spec$panelmatch$placebo.test,
    annual_anticipation_years = if (frequency == "annual") design$annual_anticipation_years else NA_integer_,
    formal_queue_written = FALSE
  ))
  cat("Completed official PanelMatch for", outcome, "at", output, "\n")
}

invisible(lapply(design$outcomes, run_one_outcome))
