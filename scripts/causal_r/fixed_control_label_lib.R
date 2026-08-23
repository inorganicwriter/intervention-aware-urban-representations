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
                                         family, root = project_root(),
                                         window = 1L, price_measure = "median") {
  window <- as.integer(window)
  if (window < 1L || window > 6L) stop("window must be in 1..6 months")
  horizons <- c(1L, 3L, 6L, 12L, 18L, 24L)
  # Observation-window semantics: the event-period value at horizon h is the
  # mean over months [max(1, h-window+1), h] after opening (window=1 keeps
  # the single-month specification; window=3/6 are the W=3/W=6 robustness
  # views).  The window never reaches into the pre-opening months.
  # Sparse windows are admitted when both treated and control have at least
  # one finite value; effective counts are retained for downstream auditing.
  window_starts <- pmax(1L, horizons - window + 1L)
  post_leads <- sort(unique(unlist(Map(
    function(start, end) seq.int(start, end), window_starts, horizons
  ))))
  calendar <- monthly_event_calendar(
    target$opening_month_date, lag = 36L, leads = post_leads,
    anticipation_months = complete_estimator_spec()$timing$main_anticipation_months
  )
  baseline_months <- tail(calendar$pre_months, 12L)
  months <- c(baseline_months, calendar$post_months)
  strict_viirs <- family != "viirs"
  treated <- read_city_monthly_outcome(target$city_key, family, months, root,
                                       price_measure = price_measure,
                                       strict = strict_viirs)[
    grid_id == target$grid_id
  ]
  control <- read_city_monthly_outcome(control_city_key, family, months, root,
                                       price_measure = price_measure,
                                       strict = strict_viirs)[
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
  lead_to_month <- setNames(
    as.IDate(calendar$post_months), as.character(post_leads)
  )
  result <- data.table(
    outcome = outcome, event_time = horizons,
    period = as.IDate(lead_to_month[as.character(horizons)])
  )
  window_stats <- function(month_table, leads) {
    months_in <- as.IDate(lead_to_month[as.character(sort(unique(leads)))])
    values <- month_table[month %in% months_in, get(outcome)]
    values <- values[is.finite(values)]
    list(
      mean = if (!length(values)) NA_real_ else mean(values),
      n = as.integer(length(values))
    )
  }
  treated_stats <- lapply(
    horizons,
    function(h) window_stats(treated, max(1L, h - window + 1L):h)
  )
  control_stats <- lapply(
    horizons,
    function(h) window_stats(control, max(1L, h - window + 1L):h)
  )
  result[, `:=`(
    treated_post = vapply(treated_stats, function(x) x$mean, numeric(1L)),
    control_post = vapply(control_stats, function(x) x$mean, numeric(1L)),
    effective_n_treated = vapply(treated_stats, function(x) x$n, integer(1L)),
    effective_n_control = vapply(control_stats, function(x) x$n, integer(1L)),
    # Require a complete post-treatment observation window.  At horizon 1
    # only one post month exists; from horizon 3 onward W=3 requires all
    # three calendar months rather than accepting a sparse partial mean.
    minimum_window_n = pmin(as.integer(window), horizons)
  )]
  result[, `:=`(
    treated_baseline = treated_baseline,
    control_baseline = control_baseline,
    treated_change = treated_post - treated_baseline,
    control_change = control_post - control_baseline
  )]
  result[, `:=`(
    observed = treated_post,
    counterfactual = treated_baseline + control_change,
    window_supported = effective_n_treated >= minimum_window_n &
      effective_n_control >= minimum_window_n
  )]
  # Keep this as a separate assignment: data.table does not guarantee that a
  # column created in the same := call is visible to another RHS expression.
  result[, label_available := window_supported &
    is.finite(treated_change) & is.finite(control_change)]
  result[, causal_response_label := ifelse(
    label_available,
    treated_change - control_change,
    NA_real_
  )]

  # Regression-based DiD for each horizon.  Sparse monthly panels (housing in
  # particular) omit intermediate months entirely, so the outcome vectors must
  # be aligned to the full calendar sequence with NA placeholders before any
  # positional indexing; otherwise regression_beta/se/p silently shift across
  # months.  The regression diagnostics keep single-month alignment at each
  # horizon endpoint; only the causal label uses the observation window.
  reg_all_months <- c(baseline_months, calendar$post_months)
  calendar_frame <- data.table(month = reg_all_months, reg_position = seq_along(reg_all_months))
  treated_full <- merge(
    calendar_frame,
    read_city_monthly_outcome(target$city_key, family, reg_all_months, root,
                              price_measure = price_measure, strict = strict_viirs)[
      grid_id == target$grid_id
    ],
    by = "month", all.x = TRUE
  )[order(reg_position)]
  control_full <- merge(
    calendar_frame,
    read_city_monthly_outcome(control_city_key, family, reg_all_months, root,
                              price_measure = price_measure, strict = strict_viirs)[
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

    # Regression-based DiD for each horizon.  Like the monthly branch above,
    # the annual vectors must be aligned to the full calendar-year skeleton
    # with NA placeholders: if a year in pre_years/post_years is absent from
    # the annual panel, positional indexing would silently shift the window
    # (e.g. opening-4 missing but opening+1 present), and the is.finite guard
    # would pass on misaligned data.
    pre_years <- seq.int(
      baseline_year - 3L, baseline_year
    )
    all_years <- c(pre_years, post_years)
    year_frame <- data.table(year = all_years, reg_position = seq_along(all_years))
    treated_full <- merge(
      year_frame,
      read_city_annual_family(target$city_key, family, root)[
        grid_id == target$grid_id & year %in% all_years
      ],
      by = "year", all.x = TRUE
    )[order(reg_position)]
    control_full <- merge(
      year_frame,
      read_city_annual_family(control_city_key, family, root)[
        grid_id == control_grid_id & year %in% all_years
      ],
      by = "year", all.x = TRUE
    )[order(reg_position)]
    tv <- treated_full[, get(outcome)]
    cv <- control_full[, get(outcome)]
    pre_n <- length(pre_years)
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
                                 root = project_root(), window = 1L,
                                 price_measure = "median") {
  requested_order <- as.integer(treatment_order)
  target <- read_treatments(root)[treatment_order == requested_order]
  if (nrow(target) != 1L) stop("Treatment order is not unique")
  assert_choice(family, names(complete_estimator_spec()$families), "family")
  result <- if (family %in% c("housing", "viirs")) {
    monthly_fixed_control_labels(
      target, control_city_key, control_grid_id, family, root,
      window = window, price_measure = price_measure
    )
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
