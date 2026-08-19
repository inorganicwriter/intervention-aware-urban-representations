suppressPackageStartupMessages({
  library(data.table)
  library(fect)
})

source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
causal_run_id <- Sys.getenv("MIT_CAUSAL_RUN_ID", unset = "")
if (length(args) < 3L || length(args) > 10L) {
  stop(paste(
    "Usage: run_complete_mc.R CITY COHORT OUTCOME_FAMILY",
    "[SIGNATURE=auto] [FREQUENCY=annual] [ANTICIPATION_MONTHS=6]",
    "[TREATMENT_ORDER] [DONOR_SCOPE=same_city] [RUN_MODE=production]",
    "[PRICE_MEASURE=median]"
  ))
}
city_key <- args[[1L]]
cohort <- args[[2L]]
outcome_family <- args[[3L]]
signature <- if (length(args) >= 4L) args[[4L]] else "auto"
frequency <- if (length(args) >= 5L) args[[5L]] else "annual"
anticipation_months <- if (length(args) >= 6L) as.integer(args[[6L]]) else NULL
treatment_order <- if (length(args) >= 7L) as.integer(args[[7L]]) else NULL
requested_treatment_order <- treatment_order
donor_scope <- if (length(args) >= 8L) args[[8L]] else "same_city"
run_mode <- if (length(args) >= 9L) args[[9L]] else "production"
price_measure <- if (length(args) >= 10L) args[[10L]] else "median"
assert_choice(frequency, c("annual", "monthly"), "frequency")
assert_choice(donor_scope, c("same_city", "all_city_standardized"), "donor_scope")
assert_choice(run_mode, c("production", "smoke_test"), "run_mode")
assert_choice(price_measure, c("median", "hedonic"), "price_measure")
if (frequency == "monthly") {
  assert_choice(outcome_family, c("housing", "viirs"), "monthly outcome_family")
}

spec <- complete_estimator_spec()
run_nboots <- if (run_mode == "smoke_test") 20L else spec$mc$nboots
if (isTRUE(spec$mc$parallel)) {
  options(future.globals.maxSize = 2 * 1024^3)
}

scope_donors <- function() {
  if (donor_scope == "same_city") return(read_city_donors(city_key))
  as.data.table(read_parquet(
    file.path(
      project_root(), "data", "active", "causal", "formal_matching_inputs",
      "eligible_never_treated_donors.parquet"
    ), col_select = c("city_key", "grid_id", "unit_id")
  ))
}

scope_cities <- function(donors) sort(unique(donors$city_key))

scope_annual_outcomes <- function(donors) {
  if (donor_scope == "same_city") {
    return(read_city_annual_family(city_key, outcome_family))
  }
  rbindlist(lapply(scope_cities(donors), function(city) {
    read_city_annual_family(city, outcome_family)
  }), use.names = TRUE)
}

scope_monthly_outcomes <- function(donors, months) {
  if (donor_scope == "same_city") {
    return(read_city_monthly_outcome(city_key, outcome_family, months,
                             price_measure = price_measure,
                             strict = outcome_family != "viirs"))
  }
  rbindlist(lapply(scope_cities(donors), function(city) {
    read_city_monthly_outcome(city, outcome_family, months,
                          price_measure = price_measure,
                          strict = outcome_family != "viirs")
  }), use.names = TRUE)
}

build_mc_design <- function() {
  requested_city <- city_key
  if (frequency == "annual") {
    treated <- read_treatments()[
      city_key == requested_city & opening_year == as.integer(cohort)
    ]
    if (!is.null(requested_treatment_order)) {
      treated <- treated[treatment_order == requested_treatment_order]
    }
    if (!nrow(treated)) stop("No treated units for requested annual MC target")
    if (nrow(treated) != 1L) {
      stop("Annual MC target selection must resolve to exactly one treatment")
    }
    selected_signature <- if (donor_scope == "same_city") {
      "outcome_only_prepath_mc"
    } else "outcome_only_prepath_mc_all_city"
    donors <- scope_donors()
    outcomes <- scope_annual_outcomes(donors)
    available <- sort(unique(outcomes$year))
    annual_anticipation <- spec$timing$annual_anticipation_years
    pre <- available[available < as.integer(cohort) - annual_anticipation]
    post <- available[available > as.integer(cohort) & available <= as.integer(cohort) + 3L]
    if (length(pre) < spec$mc$min.T0) stop("Insufficient pre-treatment periods for MC")
    if (!length(post)) stop("No post-treatment outcome")
    times <- c(pre, post)
    setnames(outcomes, "year", "period")
    opening_period_excluded <- as.integer(cohort)
    first_treated_period <- min(post)
  } else {
    all_treated <- read_treatments()
    cohort_date <- as.IDate(paste0(substr(cohort, 1L, 7L), "-01"))
    treated <- all_treated[city_key == requested_city & opening_month_date == cohort_date]
    if (!is.null(requested_treatment_order)) {
      treated <- treated[treatment_order == requested_treatment_order]
    }
    if (!nrow(treated)) stop("No treated units for requested monthly MC target")
    if (nrow(treated) != 1L) {
      stop("Monthly MC target selection must resolve to exactly one treatment")
    }
    selected_signature <- if (donor_scope == "same_city") {
      "outcome_only_prepath_mc"
    } else "outcome_only_prepath_mc_all_city"
    donors <- scope_donors()
    calendar <- monthly_event_calendar(
      cohort_date, lag = spec$monthly$lag, leads = spec$monthly$leads,
      anticipation_months = anticipation_months
    )
    if (outcome_family == "viirs") {
      pre <- calendar$pre_months[calendar$pre_months >= as.IDate("2012-01-01")]
      outcomes <- scope_monthly_outcomes(donors, c(pre, calendar$post_months))
    } else {
      outcomes <- scope_monthly_outcomes(donors, calendar$model_months)
      available_months <- sort(unique(outcomes$month))
      pre <- calendar$pre_months[calendar$pre_months %in% available_months]
      outcomes <- outcomes[month %in% c(pre, calendar$post_months)]
    }
    if (length(pre) < spec$mc$min.T0) {
      stop("Insufficient pre-treatment monthly periods for MC")
    }
    post <- calendar$post_months
    times <- c(pre, post)
    setnames(outcomes, "month", "period")
    opening_period_excluded <- calendar$opening_month
    first_treated_period <- calendar$first_treated_month
  }
  list(
    treated = treated, donors = donors, outcomes = outcomes,
    times = times, pre = pre, post = post,
    outcome_names = spec$families[[outcome_family]],
    signature = selected_signature,
    opening_period_excluded = opening_period_excluded,
    first_treated_period = first_treated_period,
    anticipation_months = if (frequency == "monthly") calendar$anticipation_months else NA_integer_,
    annual_anticipation_years = if (frequency == "annual") spec$timing$annual_anticipation_years else NA_integer_,
    donor_scope = donor_scope
  )
}

design <- build_mc_design()
if (nrow(design$treated) != 1L) {
  stop("MC estimation requires exactly one treated unit per fit")
}

run_one_outcome <- function(outcome) {
  units <- rbindlist(list(
    design$treated[, .(city_key, grid_id, role = "treated", treatment_order)],
    design$donors[, .(city_key, grid_id, role = "donor", treatment_order = NA_integer_)]
  ), use.names = TRUE)
  units <- unique(units, by = c("city_key", "grid_id"))
  setorder(units, role, grid_id)

  values <- design$outcomes[, .(city_key, grid_id, period, value = get(outcome))]
  values <- values[period %in% design$times]

  # MC does not require complete pre-treatment paths; it handles missing entries.
  # Treated unit must have at least min.T0 finite pre-period observations.
  treated_keys <- units[role == "treated", .(city_key, grid_id)]
  treated_pre_finite <- merge(
    values[period %in% design$pre & is.finite(value), .(pre_finite_count = .N), by = .(city_key, grid_id)],
    treated_keys, by = c("city_key", "grid_id")
  )
  if (!nrow(treated_pre_finite) || treated_pre_finite$pre_finite_count[[1L]] < spec$mc$min.T0) {
    stop("Treated unit lacks enough finite pre-treatment observations for MC")
  }

  # Donors: admit all with at least min.T0 finite pre-periods (not necessarily complete).
  # MC with very large donor pools can exhaust memory; cap at a manageable size.
  donor_keys <- units[role == "donor", .(city_key, grid_id)]
  donor_pre_finite <- merge(
    values[period %in% design$pre & is.finite(value), .(pre_finite_count = .N), by = .(city_key, grid_id)],
    donor_keys, by = c("city_key", "grid_id")
  )
  donor_pre_finite <- donor_pre_finite[pre_finite_count >= spec$mc$min.T0]
  if (!nrow(donor_pre_finite)) stop("No donor has enough finite pre-treatment observations for MC")
  donor_capped <- nrow(donor_pre_finite) > spec$mc$max_donors
  if (donor_capped) {
    setorder(donor_pre_finite, -pre_finite_count, city_key, grid_id)
    donor_pre_finite <- donor_pre_finite[seq_len(spec$mc$max_donors)]
  }
  units <- rbind(
    units[role == "treated"],
    merge(donor_pre_finite[, .(city_key, grid_id)], units[role == "donor"],
          by = c("city_key", "grid_id"))
  )
  if (!any(units$role == "treated")) stop("Treated unit has no observations")
  if (!any(units$role == "donor")) stop("No donor has any observations")
  setorder(units, role, grid_id)
  units[, mc_unit_id := seq_len(.N)]

  panel <- CJ(mc_unit_id = units$mc_unit_id, period = design$times, sorted = TRUE)
  panel <- merge(
    panel, units[, .(mc_unit_id, city_key, grid_id, role, treatment_order)],
    by = "mc_unit_id"
  )
  panel <- merge(panel, values, by = c("city_key", "grid_id", "period"), all.x = TRUE)
  panel[, time_id := match(period, design$times)]
  pre_count <- length(design$pre)
  panel[, D := as.integer(role == "treated" & time_id > pre_count)]

  target_scale <- 1
  target_center <- 0
  if (design$donor_scope == "all_city_standardized") {
    city_stats <- panel[role == "donor" & time_id <= pre_count & is.finite(value), .(
      pre_center = mean(value), pre_scale = stats::sd(value)
    ), by = city_key]
    city_stats[!is.finite(pre_scale) | pre_scale <= sqrt(.Machine$double.eps), pre_scale := 1]
    panel <- merge(panel, city_stats, by = "city_key", all.x = TRUE)
    if (any(!is.finite(panel$pre_center)) || any(!is.finite(panel$pre_scale))) {
      stop("MC cross-city lacks finite pre-only city scaling parameters")
    }
    panel[, model_value := (value - pre_center) / pre_scale]
    target_city_stats <- unique(panel[role == "treated", .(pre_center, pre_scale)])
    if (nrow(target_city_stats) != 1L) stop("Target city scaling is not unique")
    target_center <- target_city_stats$pre_center[[1L]]
    target_scale <- target_city_stats$pre_scale[[1L]]
  } else {
    panel[, model_value := value]
  }

  estimation_data <- panel[, .(mc_unit_id, time_id, Y = model_value, D)]
  # The treated post-period outcome is needed only after estimation to form the
  # response label. Mask it before lambda construction/CV and model fitting so
  # no feature of the observed treatment response can alter its counterfactual.
  target_mc_id <- units[role == "treated", mc_unit_id][[1L]]
  treated_pre_fill <- mean(
    estimation_data[mc_unit_id == target_mc_id & D == 0L, Y], na.rm = TRUE
  )
  if (!is.finite(treated_pre_fill)) stop("MC treated pre-period fill value is unavailable")
  target_rows <- estimation_data[, which(mc_unit_id == target_mc_id & D == 1L)]
  if (!length(target_rows)) {
    stop("MC estimation data lacks treated post-period rows to mask")
  }
  # Mask the treated post-period rows, then verify the mask actually applied:
  # every masked row must now equal the fill value.  Checking the values
  # captured before assignment would both false-positive when an original
  # outcome coincidentally equals the fill and silently miss a real mask
  # failure.
  estimation_data[D == 1L, Y := treated_pre_fill]
  masked_values <- estimation_data$Y[target_rows]
  if (any(masked_values != treated_pre_fill, na.rm = TRUE)) {
    stop("MC estimation data failed to mask treated post-period outcomes")
  }

  # Lambda selection (CV) depends almost entirely on the donor pool: one
  # treated unit out of up to 2,001 rows shifts the chosen lambda negligibly.
  # Cache the selected lambda per (city, cohort, family, scope) so a cohort
  # with many treated grids pays the CV cost once instead of per unit.
  .staging_root <- .resolve_path(
    "OUTPUT_COMPLETE_STAGING_DIR", project_root(),
    "outputs", "complete_estimators", "staging"
  )
  lambda_cache_dir <- file.path(.staging_root, "mc_lambda_cache")
  dir.create(lambda_cache_dir, recursive = TRUE, showWarnings = FALSE)
  lambda_cache_path <- file.path(
    lambda_cache_dir,
    paste0(city_key, "__", cohort, "__", outcome_family, "__", donor_scope, ".rds")
  )
  selected_lambda <- NULL
  if (file.exists(lambda_cache_path)) {
    cached <- readRDS(lambda_cache_path)
    if (is.numeric(cached) && length(cached) == 1L && is.finite(cached) && cached > 0) {
      selected_lambda <- cached
    }
  }
  if (is.null(selected_lambda)) {
    selection_fit <- fect(
      Y ~ D, data = estimation_data, index = c("mc_unit_id", "time_id"),
      method = spec$mc$estimator,
      force = spec$mc$force, CV = spec$mc$CV,
      criterion = spec$mc$criterion, nlambda = spec$mc$nlambda,
      cv.method = spec$mc$cv_method, cv.nobs = spec$mc$cv_nobs,
      cv.donut = spec$mc$cv_donut, cv.buffer = spec$mc$cv_buffer,
      se = FALSE,
      parallel = spec$mc$parallel, cores = spec$mc$cores,
      min.T0 = spec$mc$min.T0, normalize = FALSE,
      seed = 20260725
    )
    if (!inherits(selection_fit, "fect") ||
        !identical(selection_fit$method, "mc") ||
        length(selection_fit$lambda.cv) != 1L ||
        !is.finite(selection_fit$lambda.cv) || selection_fit$lambda.cv <= 0) {
      stop("MC selection stage did not choose one finite positive lambda")
    }
    selected_lambda <- as.numeric(selection_fit$lambda.cv)
    saveRDS(selected_lambda, lambda_cache_path)
  }
  fit <- fect(
    Y ~ D, data = estimation_data, index = c("mc_unit_id", "time_id"),
    method = spec$mc$estimator, force = spec$mc$force,
    CV = FALSE, lambda = selected_lambda,
    se = spec$mc$se, nboots = run_nboots,
    vartype = spec$mc$inference,
    parallel = spec$mc$parallel, cores = spec$mc$cores,
    min.T0 = spec$mc$min.T0, normalize = FALSE,
    seed = 20260725
  )
  if (!inherits(fit, "fect") || is.null(fit$Y.ct)) {
    stop("MC did not return a valid counterfactual model")
  }
  if (!identical(fit$method, "mc")) {
    stop("Requested MC estimator returned a different fitted method")
  }
  if (!isTRUE(spec$mc$CV) || is.null(selection_fit$CV.out.mc) ||
      !is.matrix(selection_fit$CV.out.mc) ||
      !"MSPE" %in% colnames(selection_fit$CV.out.mc) ||
      !any(is.finite(selection_fit$CV.out.mc[, "MSPE"]))) {
    stop("MC cross-validation diagnostics are missing or invalid")
  }
  mc_cv_mspe <- min(selection_fit$CV.out.mc[, "MSPE"], na.rm = TRUE)

  # Never guess the target column: a wrong Y.ct column silently corrupts labels.
  if (is.null(fit$id)) stop("MC fit does not expose unit identities for Y.ct")
  target_column <- match(
    as.character(units[role == "treated", mc_unit_id][[1L]]),
    as.character(fit$id)
  )
  if (is.na(target_column)) stop("MC target counterfactual column is missing")
  if (design$donor_scope == "all_city_standardized") {
    fit$Y.ct[, target_column] <- fit$Y.ct[, target_column] * target_scale + target_center
  }

  treated_units <- units[role == "treated"]
  target_panel <- panel[mc_unit_id == treated_units$mc_unit_id[[1L]]][order(time_id)]
  model_periods <- c(design$pre, design$post)
  pre_count <- length(design$pre)
  post_event_time <- if (frequency == "annual") {
    as.integer(design$post) - as.integer(cohort)
  } else seq_along(design$post)
  counterfactual <- as.numeric(fit$Y.ct[, target_column])
  if (length(counterfactual) != length(model_periods) ||
      !all(is.finite(counterfactual))) {
    stop("MC target counterfactual path is incomplete or non-finite")
  }
  paths <- data.table(
    treatment_order = treated_units$treatment_order[[1L]],
    city_key = treated_units$city_key[[1L]],
    grid_id = treated_units$grid_id[[1L]],
    outcome_family = outcome_family,
    outcome = outcome,
    period = model_periods,
    event_time = c(seq.int(-pre_count, -1L), as.integer(post_event_time)),
    observed = target_panel$value,
    counterfactual = counterfactual
  )
  paths[, `:=`(
    causal_response_label = observed - counterfactual,
    label_available = is.finite(observed) & is.finite(counterfactual),
    method = "athey_2021_mc",
    selected_factors = NA_integer_,
    mc_lambda = selected_lambda,
    mc_cv_mspe = as.numeric(mc_cv_mspe)
  )]
  if (is.null(fit$est.att)) {
    paths[, `:=`(
      standard_error = NA_real_, confidence_lower = NA_real_,
      confidence_upper = NA_real_, p_value = NA_real_,
      bootstrap_repetitions = as.integer(run_nboots),
      uncertainty_source = "mc_nonparametric_bootstrap_unavailable"
    )]
  } else {
    paths <- attach_single_target_gsc_uncertainty(
      paths, fit, run_nboots, effect_scale = target_scale
    )
    paths[, uncertainty_source := "mc_nonparametric_bootstrap"]
  }
  prefit <- paths[event_time < 0L, .(
    pre_rmspe = sqrt(mean(causal_response_label^2, na.rm = TRUE)),
    pre_observed_periods = sum(is.finite(observed))
  ), by = .(treatment_order, outcome_family, outcome)]
  paths <- merge(paths, prefit, by = c("treatment_order", "outcome_family", "outcome"), all.x = TRUE)

  outcome_tag <- paste0(
    outcome,
    if (!is.null(treatment_order)) sprintf("_t%05d", treatment_order) else ""
  )
  output_signature <- if (run_mode == "smoke_test") {
    paste0(design$signature, "_smoke_test")
  } else design$signature
  output <- estimator_output_dir(
    "matrix_completion", city_key, cohort, outcome_tag, output_signature
  )
  saveRDS(fit, file.path(output, "mc_object.rds"), compress = "xz")
  saveRDS(selection_fit, file.path(output, "mc_cv_object.rds"), compress = "xz")
  write_parquet(panel, file.path(output, "estimation_panel.parquet"), compression = "zstd")
  write_parquet(paths, file.path(output, "causal_response_labels.parquet"), compression = "zstd")
  fwrite(units, file.path(output, "unit_map.csv"), bom = TRUE)
  writeLines(capture.output(print(fit)), file.path(output, "mc_print.txt"))
  writeLines(capture.output(summary(fit)), file.path(output, "mc_summary.txt"))
  fwrite(data.table(
    clean_pre_periods = pre_count,
    post_periods = length(design$post),
    preonly_complete_treated = sum(units$role == "treated"),
    preonly_donors_used = sum(units$role == "donor"),
    available_post_labels = paths[event_time > 0L, sum(label_available)],
    estimator = "mc"
  ), file.path(output, "diagnostics.csv"), bom = TRUE)
  write_run_manifest(output, list(
    schema = spec$schema, method = "Matrix completion (Athey et al. 2021)",
    run_id = causal_run_id,
    package = paste0("fect ", packageVersion("fect")),
    city_key = city_key, cohort = cohort, frequency = frequency,
    treatment_order = treatment_order,
    outcome_family = outcome_family, outcome = outcome,
    signature = design$signature, estimator = "mc", backend = spec$mc$backend,
    fitted_method = fit$method,
    CV = spec$mc$CV, criterion = spec$mc$criterion,
    nlambda = spec$mc$nlambda, selected_lambda = selected_lambda,
    cv_method = spec$mc$cv_method, cv_nobs = spec$mc$cv_nobs,
    cv_donut = spec$mc$cv_donut, cv_buffer = spec$mc$cv_buffer,
    two_stage_cv_inference = spec$mc$two_stage_cv_inference,
    inference_fit_CV = FALSE,
    cv_min_mspe = mc_cv_mspe,
    force = spec$mc$force, min_T0 = spec$mc$min.T0,
    se = spec$mc$se, inference = spec$mc$inference, nboots = run_nboots,
    run_mode = run_mode,
    production_eligible = run_mode == "production",
    price_measure = price_measure,
    donor_admission_uses_post_outcome = FALSE,
    treated_post_outcome_mask = "treated_pre_mean_before_mc_cv_and_fit",
    opening_period_excluded = as.character(design$opening_period_excluded),
    first_treated_period = as.character(design$first_treated_period),
    anticipation_months = design$anticipation_months,
    annual_anticipation_years = design$annual_anticipation_years,
    donor_scope = design$donor_scope,
    eligible_scope_donors = nrow(design$donors),
    donors_used = sum(units$role == "donor"),
    cross_city_scaling = if (design$donor_scope == "all_city_standardized") {
      "city donor pre-period mean/sd; post-period information excluded"
    } else "none",
    target_effect_scale_to_original_units = target_scale,
    donor_cap = if (donor_capped) {
      paste0("top_", spec$mc$max_donors, "_by_pre_finite_count")
    } else "none",
    formal_queue_written = FALSE
  ))
  cat("Completed MC and normalized causal labels for", outcome, "at", output, "\n")
}

run_status <- rbindlist(lapply(design$outcome_names, function(outcome) {
  tryCatch({
    run_one_outcome(outcome)
    data.table(outcome = outcome, status = "success", failure_reason = NA_character_)
  }, error = function(error) {
    message("MC outcome failed [", outcome, "]: ", conditionMessage(error))
    data.table(
      outcome = outcome, status = "failed",
      failure_reason = conditionMessage(error)
    )
  })
}), fill = TRUE)
run_status[, run_id := causal_run_id]
family_tag <- paste0(
  outcome_family,
  if (!is.null(treatment_order)) sprintf("_t%05d", treatment_order) else ""
)
family_signature <- if (run_mode == "smoke_test") {
  paste0(design$signature, "_smoke_test")
} else design$signature
family_output <- estimator_output_dir(
  "matrix_completion_runs", city_key, cohort, family_tag, family_signature
)
temporary_status <- file.path(family_output, "outcome_status.csv.tmp")
fwrite(run_status, temporary_status, bom = TRUE)
published_status <- file.path(family_output, "outcome_status.csv")
if (!file.copy(temporary_status, published_status, overwrite = TRUE)) {
  stop("Unable to publish MC family outcome status")
}
file.remove(temporary_status)
if (!any(run_status$status == "success")) {
  stop("MC failed for every requested outcome")
}
