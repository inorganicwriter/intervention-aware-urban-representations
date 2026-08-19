suppressPackageStartupMessages({
  library(data.table)
  library(gsynth)
})

source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
causal_run_id <- Sys.getenv("MIT_CAUSAL_RUN_ID", unset = "")
if (length(args) < 3L || length(args) > 10L) {
  stop(paste(
    "Usage: run_complete_xu_gsc.R CITY COHORT OUTCOME_FAMILY",
    paste(
      "[SIGNATURE=auto] [FREQUENCY=annual] [ANTICIPATION_MONTHS=6]",
      "[TREATMENT_ORDER] [DONOR_SCOPE=same_city] [RUN_MODE=production]",
      "[PRICE_MEASURE=median]"
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
assert_choice(frequency, c("annual", "monthly"), "frequency")
assert_choice(donor_scope, c("same_city", "all_city_standardized"), "donor_scope")
assert_choice(run_mode, c("production", "smoke_test"), "run_mode")
assert_choice(price_measure, c("median", "hedonic"), "price_measure")
if (frequency == "monthly") {
  assert_choice(outcome_family, c("housing", "viirs"), "monthly outcome_family")
  if (price_measure == "hedonic") assert_choice(outcome_family, "housing", "hedonic outcome_family")
}

spec <- complete_estimator_spec()
run_nboots <- if (run_mode == "smoke_test") 20L else spec$xu_gsc$nboots
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
    if (!length(post)) stop("No post-treatment full-year outcome")
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
    if (outcome_family == "viirs") {
      pre <- calendar$pre_months[calendar$pre_months >= as.IDate("2012-01-01")]
      outcomes <- scope_monthly_outcomes(donors, c(pre, calendar$post_months))
    } else {
      outcomes <- scope_monthly_outcomes(donors, calendar$model_months)
      available_months <- sort(unique(outcomes$month))
      pre <- calendar$pre_months[calendar$pre_months %in% available_months]
      outcomes <- outcomes[month %in% c(pre, calendar$post_months)]
    }
    if (length(pre) < spec$xu_gsc$min.T0) {
      stop("Insufficient clean pre-treatment monthly periods for gsynth")
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
  fit <- gsynth(
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

  selection_fit <- gsynth(
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
  fit <- gsynth(
    Y ~ D, data = estimation_data, index = c("gsc_unit_id", "time_id"),
    force = spec$xu_gsc$force, CV = FALSE, r = selected_r,
    criterion = spec$xu_gsc$criterion, estimator = spec$xu_gsc$estimator,
    se = spec$xu_gsc$se, nboots = run_nboots,
    inference = spec$xu_gsc$inference,
    parallel = spec$xu_gsc$parallel_bootstrap,
    cores = spec$xu_gsc$bootstrap_cores,
    min.T0 = spec$xu_gsc$min.T0, normalize = spec$xu_gsc$normalize,
    seed = 20260723
  )
  if (!inherits(fit, "gsynth") || is.null(fit$Y.ct)) {
    stop("gsynth did not return a valid bootstrapped counterfactual model")
  }
  if (!is.numeric(fit$Y.ct) || !all(is.finite(fit$Y.ct))) {
    stop("gsynth produced non-finite counterfactual estimates")
  }
  fit$r.cv <- selected_r
  fit$CV.out <- selection_fit$CV.out
  masked_placebo <- NULL
  if (design$donor_scope == "all_city_standardized") {
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
  if (design$donor_scope == "all_city_standardized") {
    treated_unit_ids <- units[role == "treated", gsc_unit_id]
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
    } else seq_along(design$post)
  )
  paths <- attach_single_target_gsc_uncertainty(
    paths, fit, run_nboots, effect_scale = target_scale
  )
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
    package = paste0("gsynth ", packageVersion("gsynth")),
    city_key = city_key, cohort = cohort, frequency = frequency,
    treatment_order = treatment_order,
    outcome_family = outcome_family, outcome = outcome,
    signature = design$signature, estimator = spec$xu_gsc$estimator,
    force = spec$xu_gsc$force, CV = spec$xu_gsc$CV,
    criterion = spec$xu_gsc$criterion, factor_candidates = spec$xu_gsc$r,
    min_T0 = spec$xu_gsc$min.T0, se = spec$xu_gsc$se,
    inference = spec$xu_gsc$inference, nboots = run_nboots,
    run_mode = run_mode,
    production_eligible = run_mode == "production",
    price_measure = price_measure,
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
  cat("Completed official Xu gsynth and normalized causal labels for", outcome, "at", output, "\n")
}

invisible(lapply(design$outcome_names, run_one_outcome))
