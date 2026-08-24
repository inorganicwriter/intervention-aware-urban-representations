suppressPackageStartupMessages({
  library(data.table)
  library(fect)
  library(gsynth)
})

source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
causal_run_id <- Sys.getenv("MIT_CAUSAL_RUN_ID", unset = "")
specification_fingerprint <- Sys.getenv("MIT_SPECIFICATION_FINGERPRINT", unset = "")
if (length(args) < 3L || length(args) > 11L) {
  stop(paste(
    "Usage: run_complete_xu_gsc.R CITY COHORT OUTCOME_FAMILY",
    paste(
      "[SIGNATURE=auto] [FREQUENCY=annual] [ANTICIPATION_MONTHS=6]",
       "[TREATMENT_ORDER] [DONOR_SCOPE=same_city] [RUN_MODE=production]",
       "[PRICE_MEASURE=median] [OBSERVATION_WINDOW=1]"
    )
  ))
}
city_key <- args[[1L]]
cohort <- args[[2L]]
outcome_family <- args[[3L]]
signature <- if (length(args) >= 4L) args[[4L]] else "auto"
frequency <- if (length(args) >= 5L) args[[5L]] else "annual"
anticipation_months <- if (length(args) >= 6L) as.integer(args[[6L]]) else NULL
treatment_order <- if (length(args) >= 7L) as.integer(args[[7L]]) else NULL
donor_scope <- if (length(args) >= 8L) args[[8L]] else "same_city"
run_mode <- if (length(args) >= 9L) args[[9L]] else "production"
price_measure <- if (length(args) >= 10L) args[[10L]] else "median"
observation_window <- if (length(args) >= 11L) as.integer(args[[11L]]) else 1L
assert_choice(frequency, c("annual", "monthly"), "frequency")
assert_choice(donor_scope, c("same_city", "all_city_standardized"), "donor_scope")
assert_choice(run_mode, c("production", "preview", "smoke_test", "gpu_export"), "run_mode")
assert_choice(price_measure, c("median", "hedonic"), "price_measure")
if (observation_window < 1L || observation_window > 6L) {
  stop("observation_window must be in 1..6 months")
}
if (frequency == "monthly") {
  assert_choice(outcome_family, c("housing", "viirs"), "monthly outcome_family")
  if (price_measure == "hedonic") assert_choice(outcome_family, "housing", "hedonic outcome_family")
}
if (!nzchar(specification_fingerprint)) {
  specification_fingerprint <- paste0(
    "main_a6_r1km__a6__w", observation_window, "__price_", price_measure
  )
}

spec <- complete_estimator_spec()

# gsynth 1.4.0 is a thin fect wrapper but does not expose fect's CV geometry.
# Call fect explicitly so package-default changes cannot silently alter the
# formal specification, then preserve the gsynth-compatible result contract.
fit_gsynth_explicit <- function(formula, data, index = c("gsc_unit_id", "time_id"),
                                force, CV, r, criterion,
                                estimator, se, parallel = FALSE, cores = 1L,
                                min.T0, normalize, seed, nboots = 200L,
                                inference = "parametric") {
  set.seed(as.integer(seed))
  output <- fect::fect(
    formula = formula, data = data, method = estimator,
    index = index, force = force,
    CV = CV, r = r, criterion = criterion,
    k = spec$xu_gsc$k, cv.method = spec$xu_gsc$cv_method,
    cv.prop = spec$xu_gsc$cv_prop, cv.nobs = spec$xu_gsc$cv_nobs,
    cv.buffer = spec$xu_gsc$cv_buffer, cv.rule = spec$xu_gsc$cv_rule,
    se = se, nboots = nboots, vartype = inference,
    parallel = parallel, cores = cores, min.T0 = min.T0,
    normalize = normalize, seed = seed, keep.sims = TRUE,
    tol = spec$xu_gsc$tol, max.iteration = spec$xu_gsc$max.iteration
  )
  output$data <- data
  class(output) <- "gsynth"
  output
}

build_gsc_gpu_cv_contract <- function(estimation_data, seed) {
  time_ids <- sort(unique(estimation_data$time_id))
  control_ids <- sort(unique(estimation_data[D == 0L, gsc_unit_id]))
  treated_ids <- unique(estimation_data[D == 1L, gsc_unit_id])
  control_ids <- setdiff(control_ids, treated_ids)
  TT <- length(time_ids)
  Nco <- length(control_ids)
  II <- matrix(0L, nrow = TT, ncol = Nco)
  cells <- estimation_data[gsc_unit_id %in% control_ids]
  row_index <- match(cells$time_id, time_ids)
  column_index <- match(cells$gsc_unit_id, control_ids)
  II[cbind(row_index, column_index)] <- as.integer(is.finite(cells$Y))
  set.seed(as.integer(seed))
  folds <- fect:::.build_cv_mask_rolling(
    II = II, D = matrix(0L, nrow = TT, ncol = Nco),
    k = spec$xu_gsc$k, cv.nobs = spec$xu_gsc$cv_nobs,
    cv.buffer = spec$xu_gsc$cv_buffer, cv.prop = spec$xu_gsc$cv_prop,
    min.T0 = spec$xu_gsc$min.T0, seed = NULL
  )
  rbindlist(lapply(seq_along(folds), function(fold_id) {
    masked <- folds[[fold_id]]$cv.id
    scored <- folds[[fold_id]]$est.id
    unit_position <- ((masked - 1L) %/% TT) + 1L
    time_position <- ((masked - 1L) %% TT) + 1L
    data.table(
      fold_id = as.integer(fold_id),
      gsc_unit_id = as.integer(control_ids[unit_position]),
      time_id = as.integer(time_ids[time_position]),
      scored = masked %in% scored
    )
  }))
}
run_nboots <- if (run_mode == "production") {
  spec$xu_gsc$nboots
} else if (run_mode == "smoke_test") {
  20L
} else {
  0L
}
if (isTRUE(spec$xu_gsc$parallel_cv) || isTRUE(spec$xu_gsc$parallel_bootstrap)) {
  # gsynth's parallel parametric bootstrap exports large closures
  # (draw.error/FUN can each exceed 2 GiB on wide donor panels); the default
  # 2 GiB future.globals.maxSize then aborts every bootstrap.  Raise the cap
  # well above observed peak object sizes instead of disabling parallelism.
  options(future.globals.maxSize = 16 * 1024^3)
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

build_gsc_design <- function() {
  requested_city <- city_key
  requested_signature <- signature
  if (frequency == "annual") {
    treated <- read_treatments()[
      city_key == requested_city & opening_year == as.integer(cohort)
    ]
    if (!is.null(treatment_order)) {
      requested_order <- as.integer(treatment_order)
      treated <- treated[treatment_order == requested_order]
    }
    if (!nrow(treated)) stop("No treated units for requested annual GSC target")
    selected_signature <- if (donor_scope == "same_city") {
      "outcome_only_prepath"
    } else "outcome_only_prepath_all_city_standardized"
    donors <- scope_donors()
    outcomes <- scope_annual_outcomes(donors)
    available <- sort(unique(outcomes$year))
    annual_anticipation <- spec$timing$annual_anticipation_years
    pre <- available[available < as.integer(cohort) - annual_anticipation]
    post <- available[available > as.integer(cohort) & available <= as.integer(cohort) + 3L]
    if (length(pre) < spec$xu_gsc$min.T0) stop("Insufficient clean pre-treatment annual periods")
    if (length(post) < length(spec$annual$leads)) {
      stop("Insufficient clean post-treatment annual periods")
    }
    times <- c(pre, post)
    setnames(outcomes, "year", "period")
    opening_period_excluded <- as.integer(cohort)
    first_treated_period <- min(post)
  } else {
    all_treated <- read_treatments()
    cohort_date <- as.IDate(paste0(substr(cohort, 1L, 7L), "-01"))
    treated <- all_treated[city_key == requested_city & opening_month_date == cohort_date]
    if (!is.null(treatment_order)) {
      requested_order <- as.integer(treatment_order)
      treated <- treated[treatment_order == requested_order]
    }
    if (!nrow(treated)) stop("No treated units for requested monthly cohort")
    selected_signature <- if (donor_scope == "same_city") {
      "outcome_only_prepath"
    } else "outcome_only_prepath_all_city_standardized"
    donors <- scope_donors()
    calendar <- monthly_event_calendar(
      cohort_date, lag = spec$monthly$lag, leads = spec$monthly$leads,
      anticipation_months = anticipation_months
    )
    pre <- calendar$pre_months
    if (outcome_family == "viirs") {
      pre <- pre[pre >= as.IDate("2012-01-01")]
      outcomes <- scope_monthly_outcomes(donors, c(pre, calendar$post_months))
    } else {
      outcomes <- scope_monthly_outcomes(donors, calendar$model_months)
    }
    available_months <- sort(unique(outcomes$month))
    missing_pre <- setdiff(as.character(pre), as.character(available_months))
    missing_post <- setdiff(
      as.character(calendar$post_months), as.character(available_months)
    )
    if (length(missing_pre)) {
      stop(
        "Insufficient clean pre-treatment monthly periods for gsynth; missing ",
        length(missing_pre), " month(s)"
      )
    }
    if (length(missing_post)) {
      stop(
        "Insufficient post-treatment monthly periods for gsynth; missing ",
        length(missing_post), " month(s)"
      )
    }
    outcomes <- outcomes[month %in% c(pre, calendar$post_months)]
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

design <- build_gsc_design()
if (nrow(design$treated) != 1L) {
  stop("GSC estimation requires exactly one treated unit per fit")
}

cross_city_masked_placebo <- function(panel, units, pre_count, target_unit_id) {
  holdout_target <- if (frequency == "monthly") 12L else 1L
  holdout_n <- min(holdout_target, pre_count - spec$xu_gsc$min.T0)
  if (holdout_n < 1L) {
    stop("Cross-city masked placebo requires more than min.T0 clean pre-periods")
  }
  low_power_placebo <- frequency == "annual" && holdout_n == 1L
  train_end <- pre_count - holdout_n
  donor_ids <- units[role == "donor", gsc_unit_id]
  if (length(donor_ids) < 20L) stop("Cross-city masked placebo requires at least 20 donors")
  set.seed(20260723L)
  sampled_rows <- unique(sample.int(length(donor_ids), size = min(20L, length(donor_ids))))
  placebo_ids <- donor_ids[sampled_rows]
  pseudo_ids <- c(target_unit_id, placebo_ids)
  masked <- panel[time_id <= pre_count, .(
    gsc_unit_id, time_id, Y = model_value
  )]
  masked[, D := as.integer(gsc_unit_id %in% pseudo_ids & time_id > train_end)]
  max_r <- min(max(spec$xu_gsc$r), floor((train_end - 1L) / 2L))
  r_values <- 0:max(0L, max_r)
  fit <- fit_gsynth_explicit(
    Y ~ D, data = masked, index = c("gsc_unit_id", "time_id"),
    force = spec$xu_gsc$force, CV = TRUE, r = r_values,
    criterion = spec$xu_gsc$criterion, estimator = spec$xu_gsc$estimator,
    se = FALSE, parallel = spec$xu_gsc$parallel_cv,
    cores = spec$xu_gsc$cv_cores, min.T0 = train_end,
    normalize = spec$xu_gsc$normalize, seed = 20260724
  )
  if (!inherits(fit, "gsynth") || is.null(fit$Y.ct)) {
    stop("Cross-city masked placebo gsynth fit failed")
  }
  holdout_times <- (train_end + 1L):pre_count
  quality <- rbindlist(lapply(pseudo_ids, function(unit_id) {
    column <- match(as.character(unit_id), as.character(fit$id))
    if (is.na(column)) stop("Masked placebo counterfactual column is missing")
    observed <- masked[
      gsc_unit_id == unit_id & time_id %in% holdout_times,
      Y
    ]
    predicted <- fit$Y.ct[holdout_times, column]
    data.table(
      gsc_unit_id = unit_id,
      placebo_role = if (unit_id == target_unit_id) "target" else "donor_placebo",
      masked_periods = holdout_n,
      masked_rmspe = sqrt(mean((observed - predicted)^2, na.rm = TRUE))
    )
  }))
  threshold <- as.numeric(stats::quantile(
    quality[placebo_role == "donor_placebo", masked_rmspe],
    0.95, names = FALSE, type = 8
  ))
  target_rmspe <- quality[placebo_role == "target", masked_rmspe][[1L]]
  quality[, `:=`(
    donor_placebo_q95 = threshold,
    target_accepted = is.finite(target_rmspe) & target_rmspe <= threshold
  )]
  list(
    accepted = isTRUE(quality$target_accepted[[1L]]),
    target_rmspe = target_rmspe, threshold = threshold,
    quality = quality, fit = fit,
    low_power_placebo = low_power_placebo
  )
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
  pre_support <- values[period %in% design$pre, .(
    pre_periods = uniqueN(period[is.finite(value)]),
    pre_finite = all(is.finite(value))
  ), by = .(city_key, grid_id)]
  pre_support <- pre_support[pre_finite & pre_periods == length(design$pre)]
  units <- merge(units, pre_support[, .(city_key, grid_id)], by = c("city_key", "grid_id"))
  if (!any(units$role == "treated")) stop("No treated unit has a complete clean pre-treatment path")
  if (!any(units$role == "donor")) stop("No donor has a complete clean pre-treatment path")
  target <- units[role == "treated", .(city_key, grid_id)]
  target_post_periods <- values[
    city_key == target$city_key[[1L]] &
      grid_id == target$grid_id[[1L]] &
      period %in% design$post & is.finite(value),
    uniqueN(period)
  ]
  if (target_post_periods != length(design$post)) {
    stop("No complete post-treatment outcome for treated unit")
  }
  setorder(units, role, grid_id)
  units[, gsc_unit_id := seq_len(.N)]

  panel <- CJ(gsc_unit_id = units$gsc_unit_id, period = design$times, sorted = TRUE)
  panel <- merge(
    panel, units[, .(gsc_unit_id, city_key, grid_id, role, treatment_order)],
    by = "gsc_unit_id"
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
      stop("Cross-city GSC lacks finite pre-only city scaling parameters")
    }
    panel[, model_value := (value - pre_center) / pre_scale]
    target_city_stats <- unique(panel[role == "treated", .(pre_center, pre_scale)])
    if (nrow(target_city_stats) != 1L) stop("Target city scaling is not unique")
    target_center <- target_city_stats$pre_center[[1L]]
    target_scale <- target_city_stats$pre_scale[[1L]]
  } else {
    panel[, model_value := value]
  }
  estimation_data <- panel[, .(gsc_unit_id, time_id, Y = model_value, D)]

  if (run_mode == "gpu_export") {
    outcome_tag <- paste0(
      outcome,
      if (!is.null(treatment_order)) sprintf("_t%05d", treatment_order) else ""
    )
    output <- estimator_output_dir(
      "xu_gsc", city_key, cohort, outcome_tag,
      paste0(design$signature, "_gpu_input")
    )
    write_parquet(panel, file.path(output, "estimation_panel.parquet"), compression = "zstd")
    cv_contract <- build_gsc_gpu_cv_contract(estimation_data, 20260723L)
    write_parquet(
      cv_contract, file.path(output, "gsc_cv_folds.parquet"), compression = "zstd"
    )
    fwrite(units, file.path(output, "unit_map.csv"), bom = TRUE)
    write_run_manifest(output, list(
      schema = "causal_gpu_input_v1", method = "Xu GSC GPU input export",
      run_id = causal_run_id, city_key = city_key, cohort = cohort,
      frequency = frequency, treatment_order = treatment_order,
      outcome_family = outcome_family, outcome = outcome,
      donor_scope = design$donor_scope, run_mode = run_mode,
      cv_method = spec$xu_gsc$cv_method, cv_folds = spec$xu_gsc$k,
      cv_nobs = spec$xu_gsc$cv_nobs, cv_buffer = spec$xu_gsc$cv_buffer,
      cv_prop = spec$xu_gsc$cv_prop, cv_rule = spec$xu_gsc$cv_rule,
      tol = spec$xu_gsc$tol, max_iteration = spec$xu_gsc$max.iteration,
      cv_seed = 20260723L, fect_version = as.character(packageVersion("fect")),
      production_eligible = FALSE, selected_factors = NA_integer_,
      inference = "not_run", donor_admission_uses_post_outcome = FALSE
    ))
    cat("Exported Xu GSC GPU input for", outcome, "at", output, "\n")
    return(invisible(output))
  }

  selection_fit <- fit_gsynth_explicit(
    Y ~ D, data = estimation_data, index = c("gsc_unit_id", "time_id"),
    force = spec$xu_gsc$force, CV = spec$xu_gsc$CV, r = spec$xu_gsc$r,
    criterion = spec$xu_gsc$criterion, estimator = spec$xu_gsc$estimator,
    se = FALSE,
    parallel = spec$xu_gsc$parallel_cv, cores = spec$xu_gsc$cv_cores,
    min.T0 = spec$xu_gsc$min.T0, normalize = spec$xu_gsc$normalize,
    seed = 20260723
  )
  if (!inherits(selection_fit, "gsynth") || !is.finite(selection_fit$r.cv)) {
    stop("gsynth did not return a valid cross-validated factor selection")
  }
  selected_r <- as.integer(selection_fit$r.cv)
  masked_placebo <- NULL
  if (design$donor_scope == "all_city_standardized") {
    # This gate is deliberately evaluated before the formal bootstrap.  A
    # failed cross-city placebo should not consume 200 bootstrap refits.
    target_unit_id <- units[role == "treated", gsc_unit_id][[1L]]
    masked_placebo <- cross_city_masked_placebo(
      panel, units, pre_count, target_unit_id
    )
    if (!masked_placebo$accepted) {
      stop(
        "Cross-city masked-placebo gate failed: target RMSPE ",
        masked_placebo$target_rmspe, " > donor q95 ", masked_placebo$threshold
      )
    }
  }
  fit_args <- list(
    formula = Y ~ D, data = estimation_data,
    index = c("gsc_unit_id", "time_id"),
    force = spec$xu_gsc$force, CV = FALSE, r = selected_r,
    criterion = spec$xu_gsc$criterion, estimator = spec$xu_gsc$estimator,
    se = run_mode != "preview",
    min.T0 = spec$xu_gsc$min.T0, normalize = spec$xu_gsc$normalize,
    seed = 20260723
  )
  if (run_mode == "preview") {
    fit_args$parallel <- FALSE
    fit_args$cores <- 1L
  } else {
    fit_args$nboots <- run_nboots
    fit_args$inference <- spec$xu_gsc$inference
    fit_args$parallel <- spec$xu_gsc$parallel_bootstrap
    fit_args$cores <- spec$xu_gsc$bootstrap_cores
  }
  fit <- do.call(fit_gsynth_explicit, fit_args)
  if (!inherits(fit, "gsynth") || is.null(fit$Y.ct)) {
    stop("gsynth did not return a valid counterfactual model")
  }
  treated_unit_ids <- units[role == "treated", gsc_unit_id]
  target_columns <- match(as.character(treated_unit_ids), as.character(fit$id))
  if (anyNA(target_columns)) {
    stop("gsynth counterfactual is missing a treated target column")
  }
  target_counterfactual <- fit$Y.ct[, target_columns, drop = FALSE]
  if (!is.numeric(target_counterfactual) || !all(is.finite(target_counterfactual))) {
    stop("gsynth produced non-finite treated-target counterfactual estimates")
  }
  fit$r.cv <- selected_r
  fit$CV.out <- selection_fit$CV.out
  if (design$donor_scope == "all_city_standardized") {
    for (uid in treated_unit_ids) {
      target_column <- match(as.character(uid), as.character(fit$id))
      if (is.na(target_column)) stop("Cross-city target counterfactual column is missing for unit: ", uid)
      fit$Y.ct[, target_column] <- fit$Y.ct[, target_column] * target_scale + target_center
    }
  }

  treated_units <- units[role == "treated"]
  paths <- normalize_gsc_labels(
    fit, panel, treated_units, design$pre, design$post,
    outcome_family, outcome,
    post_event_time = if (frequency == "annual") {
      as.integer(design$post) - as.integer(cohort)
    } else seq_along(design$post),
    pre_event_time = event_time_from_period(
      design$pre, design$opening_period_excluded, frequency
    )
  )
  if (run_mode == "preview") {
    paths[, `:=`(
      standard_error = NA_real_, confidence_lower = NA_real_,
      confidence_upper = NA_real_, p_value = NA_real_,
      bootstrap_repetitions = 0L,
      uncertainty_source = "preview_point_estimate"
    )]
  } else {
    paths <- attach_single_target_gsc_uncertainty(
      paths, fit, run_nboots, effect_scale = target_scale
    )
  }
  if (frequency == "monthly") {
    paths <- apply_post_observation_window(paths, observation_window)
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
  output_signature <- if (run_mode %in% c("smoke_test", "preview")) {
    paste0(design$signature, "_", run_mode)
  } else design$signature
  output <- estimator_output_dir(
    "xu_gsc", city_key, cohort, outcome_tag, output_signature
  )
  saveRDS(fit, file.path(output, "gsynth_object.rds"), compress = "xz")
  saveRDS(
    selection_fit, file.path(output, "gsynth_factor_selection_object.rds"),
    compress = "xz"
  )
  write_parquet(panel, file.path(output, "estimation_panel.parquet"), compression = "zstd")
  write_parquet(paths, file.path(output, "causal_response_labels.parquet"), compression = "zstd")
  fwrite(units, file.path(output, "unit_map.csv"), bom = TRUE)
  if (!is.null(masked_placebo)) {
    fwrite(
      masked_placebo$quality,
      file.path(output, "cross_city_masked_placebo.csv"), bom = TRUE
    )
    saveRDS(
      masked_placebo$fit,
      file.path(output, "cross_city_masked_placebo_model.rds"), compress = "xz"
    )
  }
  writeLines(capture.output(print(fit)), file.path(output, "gsynth_print.txt"))
  writeLines(capture.output(summary(fit)), file.path(output, "gsynth_summary.txt"))
  if (!is.null(fit$att)) fwrite(as.data.table(as.table(fit$att)), file.path(output, "att.csv"), bom = TRUE)
  fwrite(data.table(
    selected_factors = fit$r.cv,
    clean_pre_periods = pre_count,
    post_periods = length(design$post),
    preonly_complete_treated = sum(units$role == "treated"),
    preonly_complete_donors = sum(units$role == "donor"),
    available_post_labels = paths[event_time > 0L, sum(label_available)]
  ), file.path(output, "diagnostics.csv"), bom = TRUE)
  write_run_manifest(output, list(
    schema = spec$schema, method = "Xu generalized synthetic control",
    run_id = causal_run_id,
    package = paste0("fect ", packageVersion("fect")),
    implementation_backend = "fect::fect",
    fect_version = as.character(packageVersion("fect")),
    compatibility_interface = "gsynth-compatible result class",
    gsynth_version = as.character(packageVersion("gsynth")),
    city_key = city_key, cohort = cohort, frequency = frequency,
    treatment_order = treatment_order,
    specification_fingerprint = specification_fingerprint,
    outcome_family = outcome_family, outcome = outcome,
    signature = design$signature, estimator = spec$xu_gsc$estimator,
    force = spec$xu_gsc$force, CV = spec$xu_gsc$CV,
    criterion = spec$xu_gsc$criterion, factor_candidates = spec$xu_gsc$r,
    cv_method = spec$xu_gsc$cv_method, cv_folds = spec$xu_gsc$k,
    cv_prop = spec$xu_gsc$cv_prop, cv_nobs = spec$xu_gsc$cv_nobs,
    cv_buffer = spec$xu_gsc$cv_buffer, cv_rule = spec$xu_gsc$cv_rule,
    tol = spec$xu_gsc$tol, max_iteration = spec$xu_gsc$max.iteration,
    min_T0 = spec$xu_gsc$min.T0, se = spec$xu_gsc$se && run_mode != "preview",
    inference = spec$xu_gsc$inference, nboots = run_nboots,
    run_mode = run_mode,
    production_eligible = run_mode == "production",
    price_measure = price_measure,
    observation_window = if (frequency == "monthly") observation_window else 1L,
    factor_selection_parallel = spec$xu_gsc$parallel_cv,
    factor_selection_cores = spec$xu_gsc$cv_cores,
    bootstrap_parallel = spec$xu_gsc$parallel_bootstrap,
    bootstrap_cores = spec$xu_gsc$bootstrap_cores,
    future_globals_max_size_bytes = if (spec$xu_gsc$parallel_cv) {
      getOption("future.globals.maxSize")
    } else NA_real_,
    donor_admission_uses_post_outcome = FALSE,
    opening_period_excluded = as.character(design$opening_period_excluded),
    first_treated_period = as.character(design$first_treated_period),
    anticipation_months = design$anticipation_months,
    annual_anticipation_years = design$annual_anticipation_years,
    donor_scope = design$donor_scope,
    eligible_scope_donors = nrow(design$donors),
    preonly_complete_scope_donors_used = sum(units$role == "donor"),
    cross_city_scaling = if (design$donor_scope == "all_city_standardized") {
      "city donor pre-period mean/sd; post-period information excluded"
    } else "none",
    target_effect_scale_to_original_units = target_scale,
    cross_city_masked_placebo = if (is.null(masked_placebo)) {
      "not_applicable"
    } else "target versus 20 deterministic donor placebos; q95 gate",
    cross_city_masked_target_rmspe = if (is.null(masked_placebo)) {
      NA_real_
    } else masked_placebo$target_rmspe,
    cross_city_masked_donor_q95 = if (is.null(masked_placebo)) {
      NA_real_
    } else masked_placebo$threshold,
    low_power_placebo = if (is.null(masked_placebo)) {
      NA
    } else isTRUE(masked_placebo$low_power_placebo),
    donor_cap = "none", selected_factors = fit$r.cv,
    formal_queue_written = FALSE
  ))
  cat("Completed", run_mode, "Xu gsynth and normalized causal labels for", outcome, "at", output, "\n")
}

invisible(lapply(design$outcome_names, run_one_outcome))
