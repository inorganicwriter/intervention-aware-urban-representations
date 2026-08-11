suppressPackageStartupMessages({
  library(data.table)
  library(Matching)
})

source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L || length(args) > 8L) {
  stop(paste(
    "Usage: run_complete_abadie_imbens.R CITY COHORT OUTCOME_FAMILY HORIZON",
    "[SIGNATURE=auto] [FREQUENCY=annual] [ANTICIPATION_MONTHS=6] [TREATMENT_ORDER]"
  ))
}
city_key <- args[[1L]]
cohort <- args[[2L]]
outcome_family <- args[[3L]]
horizon <- as.integer(args[[4L]])
signature <- if (length(args) >= 5L) args[[5L]] else "auto"
frequency <- if (length(args) >= 6L) args[[6L]] else "annual"
anticipation_months <- if (length(args) >= 7L) as.integer(args[[7L]]) else NULL
treatment_order <- if (length(args) >= 8L) as.integer(args[[8L]]) else NULL
assert_choice(frequency, c("annual", "monthly"), "frequency")
if (!is.finite(horizon) || horizon < 1L) stop("HORIZON must be a positive integer")
if (frequency == "monthly") {
  assert_choice(outcome_family, c("housing", "viirs"), "monthly outcome_family")
}

spec <- complete_estimator_spec()
design <- if (frequency == "annual") {
  build_annual_estimator_panel(
    city_key, as.integer(cohort), outcome_family, signature,
    leads = seq_len(horizon),
    treatment_order = treatment_order
  )
} else {
  build_monthly_estimator_panel(
    city_key, cohort, outcome_family, signature, leads = seq_len(horizon),
    anticipation_months = anticipation_months, treatment_order = treatment_order
  )
}

run_one_outcome <- function(outcome) {
  # Stage 1 is immutable with respect to every outcome at and after treatment.
  prepared <- make_preonly_matching_frame(design)
  feature_split <- split_holdout_features(prepared$features, holdout_lags = 1L)
  selected <- select_preonly_pairs(
    prepared, spec$abadie_imbens, match_features = feature_split$training
  )
  frame <- selected$frame
  Tr <- frame$Tr

  outcome_tag <- paste0(
    outcome, "_h", horizon,
    if (!is.null(treatment_order)) sprintf("_t%05d", treatment_order) else ""
  )
  output <- estimator_output_dir(
    "abadie_imbens", city_key, cohort,
    outcome_tag, design$signature
  )
  saveRDS(selected$fit, file.path(output, "preonly_matching_object.rds"), compress = "xz")
  write_parquet(
    frame, file.path(output, "preonly_design_cross_section.parquet"),
    compression = "zstd"
  )
  fwrite(selected$pairs, file.path(output, "preonly_matched_pairs.csv"), bom = TRUE)
  diagnostics <- pair_preonly_diagnostics(
    selected$pairs, frame, selected$active_features
  )
  write_parquet(
    diagnostics$long, file.path(output, "preonly_balance_long.parquet"),
    compression = "zstd"
  )
  fwrite(diagnostics$summary, file.path(output, "preonly_balance_summary.csv"), bom = TRUE)
  calibration <- calibrate_preonly_placebos(
    frame, selected$active_features, feature_split$holdout,
    sample_n = 200L, quantile_probability = 0.95
  )
  quality_pass <- TRUE
  for (i in seq_len(nrow(selected$pairs))) {
    q <- evaluate_preonly_pair_quality(
      selected$pairs[i, ], frame, selected$active_features,
      feature_split$holdout, calibration
    )
    if (!isTRUE(q$accepted)) { quality_pass <- FALSE; break }
  }
  quality <- q
  fwrite(calibration$placebo, file.path(output, "preonly_placebo_calibration.csv"), bom = TRUE)
  fwrite(quality, file.path(output, "preonly_quality_gate.csv"), bom = TRUE)

  # Stage 2 reads outcomes only after control identities have been frozen.
  outcome_frame <- attach_prepost_outcome(frame, design, outcome, horizon)
  labels <- pair_change_labels(selected$pairs, outcome_frame, outcome, horizon)
  write_parquet(labels, file.path(output, "causal_response_labels.parquet"), compression = "zstd")

  full_outcome_available <- all(is.finite(outcome_frame$delta_outcome))
  fit <- NULL
  balance <- NULL
  matching_warnings <- character()
  estimator_status <- "not_run_incomplete_post_outcomes"
  identification_failure <- NA_character_

  # The complete Abadie-Imbens cohort ATT is estimated only when the entire
  # pre-selected risk set has an observed change. We never remove candidates
  # based on post-treatment availability, because doing so would change design.
  if (full_outcome_available) {
    if (sum(Tr) <= ncol(selected$X)) {
      identification_failure <- paste0(
        "bias_adjustment_not_identified_treated_", sum(Tr),
        "_covariates_", ncol(selected$X)
      )
    } else if (sum(1L - Tr) <= spec$abadie_imbens$Var.calc) {
      identification_failure <- "analytic_variance_not_identified_in_preselected_donors"
    }
  }
  if (full_outcome_available && is.na(identification_failure)) {
    fit <- withCallingHandlers(
      Match(
        Y = outcome_frame$delta_outcome, Tr = Tr,
        X = selected$X, Z = selected$X,
        estimand = spec$abadie_imbens$estimand, M = spec$abadie_imbens$M,
        BiasAdjust = spec$abadie_imbens$BiasAdjust,
        replace = spec$abadie_imbens$replace, ties = spec$abadie_imbens$ties,
        # Common support was enforced explicitly on the pre-treatment X matrix.
        CommonSupport = FALSE,
        Weight = spec$abadie_imbens$Weight, Var.calc = spec$abadie_imbens$Var.calc
      ),
      warning = function(condition) {
        matching_warnings <<- c(matching_warnings, conditionMessage(condition))
        invokeRestart("muffleWarning")
      }
    )
    downgrade <- grepl("BiasAdjust set to FALSE|Var\\.calc.*reset to 0", matching_warnings)
    if (any(downgrade)) {
      stop(
        "Matching package attempted a forbidden estimator downgrade: ",
        paste(matching_warnings[downgrade], collapse = " | ")
      )
    }
    outcome_pairs <- data.table(
      treated_row = as.integer(fit$index.treated),
      control_row = as.integer(fit$index.control)
    )
    design_pairs <- selected$pairs[, .(treated_row, control_row)]
    if (!identical(outcome_pairs, design_pairs)) {
      stop("Post-outcome Match changed the frozen pre-treatment matched pairs")
    }
    balance_formula <- as.formula(
      paste("Tr ~", paste(selected$active_features, collapse = " + "))
    )
    balance_text <- capture.output(
      balance <- MatchBalance(
        balance_formula, data = as.data.frame(frame), match.out = fit,
        ks = FALSE, nboots = 500, paired = FALSE, print.level = 1
      )
    )
    saveRDS(fit, file.path(output, "matching_object.rds"), compress = "xz")
    saveRDS(balance, file.path(output, "match_balance_object.rds"), compress = "xz")
    writeLines(balance_text, file.path(output, "match_balance.txt"))
    writeLines(capture.output(summary(fit)), file.path(output, "matching_summary.txt"))
    fwrite(data.table(
      outcome = outcome, horizon = horizon,
      estimate = as.numeric(fit$est),
      analytic_standard_error = as.numeric(fit$se.standard),
      analytic_t = as.numeric(fit$est / fit$se.standard),
      treated_observations = sum(Tr), control_observations = sum(1L - Tr),
      matched_rows = length(fit$index.treated)
    ), file.path(output, "estimate.csv"), bom = TRUE)
    estimator_status <- "complete_abadie_imbens_estimated"
  } else if (full_outcome_available) {
    estimator_status <- paste0("not_run_", identification_failure)
  }

  write_run_manifest(output, list(
    schema = spec$schema, method = "pre-only nearest-neighbor design plus Abadie-Imbens ATT",
    package = paste0("Matching ", packageVersion("Matching")),
    city_key = city_key, cohort = cohort, frequency = frequency,
    treatment_order = treatment_order,
    outcome_family = outcome_family, outcome = outcome, horizon = horizon,
    estimand = spec$abadie_imbens$estimand, M = spec$abadie_imbens$M,
    BiasAdjust = spec$abadie_imbens$BiasAdjust,
    replace = spec$abadie_imbens$replace, Weight = spec$abadie_imbens$Weight,
    Var_calc = spec$abadie_imbens$Var.calc,
    control_selection_uses_post_outcome = FALSE,
    preonly_matched_pairs = nrow(selected$pairs),
    explicit_common_support = TRUE,
    unsupported_treated_units = selected$unsupported_treated_unit_ids,
    available_pair_labels = sum(labels$label_available),
    quality_gate_status = if (quality_pass) "accepted" else "route_to_gsc",
    quality_calibration = "deterministic donor-donor pseudo-treatment q95",
    heldout_lag = 1L,
    full_estimator_status = estimator_status,
    full_estimator_identification_failure = identification_failure,
    ordinary_estimator_bootstrap_used = FALSE,
    signature = design$signature, treatment_units = nrow(design$treated),
    eligible_same_city_donors = nrow(design$donors),
    preonly_complete_treated = sum(Tr), preonly_complete_donors = sum(1L - Tr),
    active_covariates = selected$active_features,
    dropped_zero_variance_covariates = selected$dropped_features,
    package_warnings = matching_warnings,
    anticipation_months = if (frequency == "monthly") {
      design$event_calendar$anticipation_months
    } else NA_integer_,
    annual_anticipation_years = if (frequency == "annual") design$annual_anticipation_years else NA_integer_,
    opening_month_excluded = frequency == "monthly",
    formal_queue_written = FALSE
  ))
  cat("Completed pre-only matching and label construction for", outcome, "at", output, "\n")
}

invisible(lapply(design$outcomes, run_one_outcome))
