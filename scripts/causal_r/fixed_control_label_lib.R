suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

finite_mean_or_na <- function(value) {
  finite <- value[is.finite(value)]
  if (length(finite)) mean(finite) else NA_real_
}

finite_mean_with_minimum <- function(value, minimum_observations) {
  finite <- value[is.finite(value)]
  if (length(finite) < minimum_observations) return(NA_real_)
  mean(finite)
}

did_regression <- function(treated_values, control_values, pre_count) {
  pre_len <- pre_count
  if (pre_len < 2L || !all(is.finite(treated_values)) || !all(is.finite(control_values))) {
    return(data.table(
      regression_beta = NA_real_, regression_se = NA_real_,
      regression_p = NA_real_, regression_nobs = NA_integer_
    ))
  }
  treated <- data.table(unit = 1L, value = treated_values)
  control <- data.table(unit = 2L, value = control_values)
  treated[, `:=`(Treat = 1L, Post = as.integer(seq_len(.N) > pre_len))]
  control[,  `:=`(Treat = 0L, Post = as.integer(seq_len(.N) > pre_len))]
  panel <- rbind(treated, control)
  panel[, TreatPost := Treat * Post]
  model <- lm(value ~ Treat + Post + TreatPost, data = panel)
  s <- summary(model)$coefficients
  if (!"TreatPost" %in% rownames(s)) {
    return(data.table(
      regression_beta = NA_real_, regression_se = NA_real_,
      regression_p = NA_real_, regression_nobs = as.integer(nrow(panel))
    ))
  }
  data.table(
    regression_beta  = as.numeric(s["TreatPost", "Estimate"]),
    regression_se    = as.numeric(s["TreatPost", "Std. Error"]),
    regression_p     = as.numeric(s["TreatPost", "Pr(>|t|)"]),
    regression_nobs  = as.integer(nrow(panel))
  )
}

monthly_fixed_control_labels <- function(target, control_city_key, control_grid_id,
                                         family, root = project_root()) {
  horizons <- c(1L, 3L, 6L, 12L, 18L, 24L)
  calendar <- monthly_event_calendar(
    target$opening_month_date, lag = 36L, leads = horizons,
    anticipation_months = complete_estimator_spec()$timing$main_anticipation_months
  )
  baseline_months <- tail(calendar$pre_months, 12L)
  months <- c(baseline_months, calendar$post_months)
  treated <- read_city_monthly_outcome(target$city_key, family, months, root)[
    grid_id == target$grid_id
  ]
  control <- read_city_monthly_outcome(control_city_key, family, months, root)[
    grid_id == control_grid_id
  ]
  outcome <- complete_estimator_spec()$families[[family]][[1L]]
  minimum_baseline <- if (family == "viirs") 12L else 1L
  treated_baseline <- finite_mean_with_minimum(
    treated[month %in% baseline_months, get(outcome)], minimum_baseline
  )
  control_baseline <- finite_mean_with_minimum(
    control[month %in% baseline_months, get(outcome)], minimum_baseline
  )
  result <- data.table(
    outcome = outcome, event_time = horizons,
    period = calendar$post_months
  )
  result <- merge(
    result,
    treated[, .(period = month, treated_post = get(outcome))],
    by = "period", all.x = TRUE
  )
  result <- merge(
    result,
    control[, .(period = month, control_post = get(outcome))],
    by = "period", all.x = TRUE
  )
  result[, `:=`(
    treated_baseline = treated_baseline,
    control_baseline = control_baseline,
    treated_change = treated_post - treated_baseline,
    control_change = control_post - control_baseline
  )]
  result[, `:=`(
    observed = treated_post,
    counterfactual = treated_baseline + control_change,
    causal_response_label = treated_change - control_change,
    label_available = is.finite(treated_change) & is.finite(control_change)
  )]

  # Regression-based DiD for each horizon.  Sparse monthly panels (housing in
  # particular) omit intermediate months entirely, so the outcome vectors must
  # be aligned to the full calendar sequence with NA placeholders before any
  # positional indexing; otherwise regression_beta/se/p silently shift across
  # months.
  reg_all_months <- c(baseline_months, calendar$post_months)
  calendar_frame <- data.table(month = reg_all_months, reg_position = seq_along(reg_all_months))
  treated_full <- merge(
    calendar_frame,
    read_city_monthly_outcome(target$city_key, family, reg_all_months, root)[
      grid_id == target$grid_id
    ],
    by = "month", all.x = TRUE
  )[order(reg_position)]
  control_full <- merge(
    calendar_frame,
    read_city_monthly_outcome(control_city_key, family, reg_all_months, root)[
      grid_id == control_grid_id
    ],
    by = "month", all.x = TRUE
  )[order(reg_position)]
  tv <- treated_full[, get(outcome)]
  cv <- control_full[, get(outcome)]
  pre_n <- length(baseline_months)
  post_idx <- (pre_n + 1L):length(reg_all_months)
  result[, c("regression_beta", "regression_se", "regression_p", "regression_nobs") := {
    post_i <- match(period, reg_all_months)
    if (is.finite(tv[pre_n]) && is.finite(cv[pre_n]) && is.finite(tv[post_i]) && is.finite(cv[post_i])) {
      did_regression(tv[1:post_i], cv[1:post_i], pre_n)
    } else {
      data.table(regression_beta = NA_real_, regression_se = NA_real_,
                 regression_p = NA_real_, regression_nobs = NA_integer_)
    }
  }, by = .(event_time)]

  result[]
}

annual_fixed_control_labels <- function(target, control_city_key, control_grid_id,
                                        family, root = project_root()) {
  horizons <- 1:3
  baseline_year <- target$opening_year - 1L
  post_years <- target$opening_year + horizons
  variables <- complete_estimator_spec()$families[[family]]
  treated <- read_city_annual_family(target$city_key, family, root)[
    grid_id == target$grid_id & year %in% c(baseline_year, post_years)
  ]
  control <- read_city_annual_family(control_city_key, family, root)[
    grid_id == control_grid_id & year %in% c(baseline_year, post_years)
  ]
  rbindlist(lapply(variables, function(outcome) {
    treated_baseline <- finite_mean_or_na(treated[year == baseline_year, get(outcome)])
    control_baseline <- finite_mean_or_na(control[year == baseline_year, get(outcome)])
    result <- data.table(outcome = outcome, event_time = horizons, year = post_years)
    result <- merge(
      result, treated[, .(year, treated_post = get(outcome))], by = "year", all.x = TRUE
    )
    result <- merge(
      result, control[, .(year, control_post = get(outcome))], by = "year", all.x = TRUE
    )
    result[, `:=`(
      treated_baseline = treated_baseline,
      control_baseline = control_baseline,
      treated_change = treated_post - treated_baseline,
      control_change = control_post - control_baseline
    )]
    result[, `:=`(
      observed = treated_post,
      counterfactual = treated_baseline + control_change,
      causal_response_label = treated_change - control_change,
      label_available = is.finite(treated_change) & is.finite(control_change)
    )]

    # Regression-based DiD for each horizon
    pre_years <- seq.int(
      baseline_year - 3L, baseline_year
    )
    all_years <- c(pre_years, post_years)
    treated_full <- read_city_annual_family(target$city_key, family, root)[
      grid_id == target$grid_id & year %in% all_years
    ][order(year)]
    control_full <- read_city_annual_family(control_city_key, family, root)[
      grid_id == control_grid_id & year %in% all_years
    ][order(year)]
    tv <- treated_full[, get(outcome)]
    cv <- control_full[, get(outcome)]
    pre_n <- length(pre_years)
    post_idx <- seq.int(pre_n + 1L, length(all_years))
    result[, c("regression_beta", "regression_se", "regression_p", "regression_nobs") := {
      post_i <- match(year, all_years)
      if (all(is.finite(tv[1:post_i])) && all(is.finite(cv[1:post_i]))) {
        did_regression(tv[1:post_i], cv[1:post_i], pre_n)
      } else {
        data.table(regression_beta = NA_real_, regression_se = NA_real_,
                   regression_p = NA_real_, regression_nobs = NA_integer_)
      }
    }, by = .(year)]

    result
  }), use.names = TRUE)
}

fixed_control_labels <- function(treatment_order, control_city_key,
                                 control_grid_id, family,
                                 root = project_root()) {
  requested_order <- as.integer(treatment_order)
  target <- read_treatments(root)[treatment_order == requested_order]
  if (nrow(target) != 1L) stop("Treatment order is not unique")
  assert_choice(family, names(complete_estimator_spec()$families), "family")
  result <- if (family %in% c("housing", "viirs")) {
    monthly_fixed_control_labels(target, control_city_key, control_grid_id, family, root)
  } else {
    annual_fixed_control_labels(target, control_city_key, control_grid_id, family, root)
  }
  result[, `:=`(
    treatment_order = target$treatment_order,
    city_key = target$city_key,
    grid_id = target$grid_id,
    opening_month = target$opening_month,
    outcome_family = family,
    control_city_key = control_city_key,
    control_grid_id = control_grid_id,
    control_unit_key = paste(control_city_key, control_grid_id, sep = "::"),
    method = "frozen_matched_change_12m_baseline",
    specification_id = "main_a6_r1km"
  )]
  result[, `:=`(
    standard_error = NA_real_,
    confidence_lower = NA_real_,
    confidence_upper = NA_real_,
    p_value = NA_real_,
    bootstrap_repetitions = 0L,
    uncertainty_source = "preonly_match_design_diagnostics"
  )]
  setcolorder(result, c(
    "treatment_order", "city_key", "grid_id", "opening_month",
    "outcome_family", "outcome", "event_time", "specification_id",
    "observed", "counterfactual", "causal_response_label", "label_available",
    "treated_baseline", "control_baseline", "treated_change", "control_change",
    "control_city_key", "control_grid_id", "control_unit_key", "method",
    setdiff(names(result), c(
      "treatment_order", "city_key", "grid_id", "opening_month",
      "outcome_family", "outcome", "event_time", "specification_id",
      "observed", "counterfactual", "causal_response_label", "label_available",
      "treated_baseline", "control_baseline", "treated_change", "control_change",
      "control_city_key", "control_grid_id", "control_unit_key", "method"
    ))
  ))
  result[]
}
