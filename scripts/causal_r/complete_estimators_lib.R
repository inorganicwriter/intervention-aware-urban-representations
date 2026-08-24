suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path(dirname(sys.frame(1)$ofile), "paths.R"), chdir = TRUE)
.PATHS <- tryCatch(
  load_project_paths(),
  error = function(e) {
    message("paths.json not available, falling back to relative paths: ", e$message)
    NULL
  }
)

.resolve_path <- function(var_name, root = project_root(), ...) {
  # Global shortcuts (loaded from paths.json via load_project_paths) are only
  # authoritative for the project root they were generated from.  An explicit
  # different root must win, otherwise callers pointing the library at an
  # alternate data root (e.g. tests with temp dirs) are silently redirected
  # back to the real project data.
  global_value <- if (exists(var_name, envir = .GlobalEnv, inherits = FALSE)) {
    get(var_name, envir = .GlobalEnv)
  } else {
    NULL
  }
  if (!is.null(global_value) && identical(root, project_root())) {
    global_value
  } else {
    file.path(root, ...)
  }
}

.estimator_env_integer <- function(name, default) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(as.integer(default))
  value <- suppressWarnings(as.integer(raw))
  if (is.na(value) || value < 1L) {
    stop(name, " must be a positive integer; received: ", raw)
  }
  value
}

complete_estimator_spec <- function() {
  list(
    schema = "complete_published_estimators_v3_explicit_deterministic_contracts",
    timing = list(
      opening_month_is_partial = TRUE,
      first_treated_month_offset = 1L,
      main_anticipation_months = 6L,
      sensitivity_anticipation_months = c(0L, 12L),
      annual_first_treated_year_offset = 1L,
      annual_anticipation_years = 0L
    ),
    annual = list(lag = 3L, leads = 1:3),
    monthly = list(lag = 36L, covariate_lags = c(13L, 25L, 37L), leads = 1:24),
    # Kept as an alias for old read-only callers. New code uses $monthly.
    monthly_housing = list(lag = 36L, covariate_lags = c(13L, 25L, 37L), leads = 1:24),
    panelmatch = list(
      refinement.method = "mahalanobis", qoi = "att", size.match = 1L,
      match.missing = FALSE, forbid.treatment.reversal = TRUE,
      matching = TRUE, listwise.delete = TRUE,
      use.diagonal.variance.matrix = FALSE, placebo.test = TRUE,
      number.iterations = 1000L, se.method = "bootstrap"
    ),
    abadie_imbens = list(
      estimand = "ATT", M = 1L, BiasAdjust = TRUE, replace = TRUE,
      ties = TRUE, CommonSupport = TRUE, Weight = 2L, Var.calc = 1L,
      # Matching's non-zero default expands the squared-distance boundary
      # and can randomly discard genuine nearest neighbours when ties=FALSE.
      distance.tolerance = 0
    ),
    xu_gsc = list(
      estimator = "gsynth", force = "two-way", CV = TRUE,
      criterion = "mspe", r = 0:5, min.T0 = 5L, se = TRUE,
      nboots = 200L, inference = "parametric", normalize = TRUE,
      k = 5L, cv_method = "rolling", cv_prop = 0.1,
      cv_nobs = 3L, cv_buffer = 1L, cv_rule = "1se",
      tol = 1e-5, max.iteration = 5000L,
      cv_cores = .estimator_env_integer("MIT_GSC_CV_CORES", 1L),
      parallel_cv = .estimator_env_integer("MIT_GSC_CV_CORES", 1L) > 1L,
      bootstrap_cores = .estimator_env_integer("MIT_GSC_BOOTSTRAP_CORES", 1L),
      parallel_bootstrap = .estimator_env_integer("MIT_GSC_BOOTSTRAP_CORES", 1L) > 1L
    ),
    mc = list(
      estimator = "mc", backend = "fect", force = "two-way",
      CV = TRUE, criterion = "mspe", nlambda = 20L,
      k = 20L, cv_prop = 0.1, cv_rule = "1se",
      cv_method = "rolling", cv_nobs = 1L, cv_donut = 0L, cv_buffer = 0L,
      tol = 1e-5, max.iteration = 5000L,
      two_stage_cv_inference = TRUE,
      min.T0 = 1L, max_donors = 2000L, se = TRUE, nboots = 200L,
      # The formal one-treated-unit MC specification uses jackknife
      # inference. Unit-level bootstrap can be unstable when the treated
      # unit is absent from a resample; jackknife leave-one-out refits are
      # the reproducible inference path used here. Parametric inference is
      # not available for method "mc".
      inference = "jackknife",
      cores = .estimator_env_integer("MIT_MC_CORES", 1L),
      parallel = .estimator_env_integer("MIT_MC_CORES", 1L) > 1L
    ),
    families = list(
      housing = "housing_log_price",
      poi = c(
        "poi_count_log", "poi_category_entropy", "poi_commercial_share",
        "poi_transport_access_log"
      ),
      viirs = "viirs_avg_asinh",
      population = "population_log"
    )
  )
}

shift_month <- function(month, n) {
  month <- as.IDate(format(as.IDate(month), "%Y-%m-01"))
  as.IDate(seq(month, by = paste(as.integer(n), "months"), length.out = 2L)[2L])
}

event_time_from_period <- function(periods, opening_period, frequency) {
  frequency <- as.character(frequency)
  if (length(frequency) != 1L || !frequency %in% c("annual", "monthly")) {
    stop("frequency must be annual or monthly")
  }
  if (frequency == "annual") {
    return(as.integer(periods) - as.integer(opening_period))
  }
  periods <- as.IDate(format(as.IDate(periods), "%Y-%m-01"))
  opening_period <- as.IDate(format(as.IDate(opening_period), "%Y-%m-01"))
  period_index <- as.integer(format(periods, "%Y")) * 12L +
    as.integer(format(periods, "%m"))
  opening_index <- as.integer(format(opening_period, "%Y")) * 12L +
    as.integer(format(opening_period, "%m"))
  period_index - opening_index
}

monthly_event_calendar <- function(opening_month, lag = 36L, leads = 1:24,
                                   anticipation_months = NULL) {
  spec <- complete_estimator_spec()
  if (is.null(anticipation_months)) {
    anticipation_months <- spec$timing$main_anticipation_months
  }
  if (!length(leads)) stop("leads must contain at least one horizon")
  opening_month <- as.IDate(paste0(substr(as.character(opening_month), 1L, 7L), "-01"))
  anticipation_months <- as.integer(anticipation_months)
  if (!is.finite(anticipation_months) || anticipation_months < 0L) {
    stop("anticipation_months must be a non-negative integer")
  }
  clean_pre_end <- shift_month(opening_month, -(anticipation_months + 1L))
  first_treated_month <- shift_month(
    opening_month, spec$timing$first_treated_month_offset
  )
  pre_months <- seq(clean_pre_end, by = "-1 month", length.out = lag)
  pre_months <- sort(as.IDate(pre_months))
  post_months <- vapply(
    as.integer(leads), function(lead) as.character(shift_month(opening_month, lead)),
    character(1L)
  )
  post_months <- as.IDate(post_months)
  excluded_months <- seq(
    shift_month(clean_pre_end, 1L), opening_month, by = "1 month"
  )
  list(
    opening_month = opening_month,
    clean_pre_end = clean_pre_end,
    first_treated_month = first_treated_month,
    pre_months = pre_months,
    post_months = post_months,
    model_months = c(pre_months, post_months),
    excluded_months = as.IDate(excluded_months),
    anticipation_months = anticipation_months
  )
}

project_root <- function() {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

assert_choice <- function(value, choices, label) {
  if (length(value) != 1L || is.na(value) || !value %in% choices) {
    stop(label, " must be one of: ", paste(choices, collapse = ", "))
  }
  invisible(value)
}

parse_family_signature <- function(signature) {
  families <- names(complete_estimator_spec()$families)
  pieces <- sort(unique(strsplit(signature, "+", fixed = TRUE)[[1L]]))
  pieces <- pieces[nzchar(pieces)]
  if (!length(pieces) || !all(pieces %in% families)) {
    stop("Invalid family signature: ", signature)
  }
  pieces
}

signature_variables <- function(signature) {
  unname(unlist(complete_estimator_spec()$families[parse_family_signature(signature)]))
}

read_treatments <- function(root = project_root()) {
  .path <- .resolve_path("TREATMENT_UNIT_LIST", root, "data", "active", "causal", "treatment_unit_list.parquet")
  x <- as.data.table(read_parquet(.path))
  x[, `:=`(
    opening_year = as.integer(substr(opening_month, 1L, 4L)),
    opening_month_date = as.IDate(paste0(opening_month, "-01")),
    unit_key = paste(city_key, grid_id, sep = "::")
  )]
  if (nrow(x) != 5048L || anyDuplicated(x$unit_key)) {
    stop("Treatment list is not the immutable 5,048-unit list")
  }
  x[]
}

read_city_donors <- function(city_key, root = project_root()) {
  requested_city <- city_key
  .path <- .resolve_path("ELIGIBLE_DONORS", root,
                         "data", "active", "causal", "formal_matching_inputs",
                         "eligible_never_treated_donors.parquet")
  x <- as.data.table(read_parquet(.path, col_select = c("city_key", "grid_id", "unit_id")))
  x <- x[city_key == requested_city]
  if (!nrow(x) || anyDuplicated(x$grid_id)) stop("No unique donors for ", city_key)
  x[]
}

read_city_annual_family <- function(city_key, family, root = project_root()) {
  assert_choice(family, names(complete_estimator_spec()$families), "family")
  if (family == "housing") {
    .path <- .resolve_path("HOUSING_ANNUAL_DIR", root,
                           "data", "active", "causal", "formal_matching_inputs", "housing_annual")
    path <- file.path(.path, paste0(city_key, ".parquet"))
    return(as.data.table(read_parquet(
      path, col_select = c("city_key", "grid_id", "year", "housing_log_price")
    )))
  }
  if (family == "poi") {
    .dir <- .resolve_path("POI_DIR", root, "data", "active", "curated", "poi")
    path <- file.path(.dir, paste0(city_key, "_poi_grid_yearly.parquet"))
    x <- as.data.table(read_parquet(path, col_select = c(
      "city", "grid_id", "year", "poi_count", "poi_category_entropy",
      "poi_commercial_share", "poi_transport_access_count"
    )))
    setnames(x, "city", "city_key")
    x[, `:=`(
      poi_count_log = log1p(pmax(poi_count, 0)),
      poi_transport_access_log = log1p(pmax(poi_transport_access_count, 0))
    )]
    return(x[, .(
      city_key, grid_id, year, poi_count_log, poi_category_entropy,
      poi_commercial_share, poi_transport_access_log
    )])
  }
  if (family == "viirs") {
    .dir <- .resolve_path("VIIRS_ANNUAL_DIR", root, "data", "active", "curated", "viirs_annual_aggregated")
    path <- file.path(.dir, paste0(city_key, "_viirs_annual.parquet"))
    x <- as.data.table(read_parquet(
      path, col_select = c("city_key", "grid_id", "year", "avg_rad")
    ))
    # Negative finite radiance is valid in the VIIRS product and must not be clipped.
    x[, viirs_avg_asinh := asinh(avg_rad)]
    return(x[, .(city_key, grid_id, year, viirs_avg_asinh)])
  }
  .dir <- .resolve_path("POPULATION_DIR", root, "data", "active", "curated", "population")
  path <- file.path(.dir, paste0(city_key, "_pop.parquet"))
  x <- as.data.table(read_parquet(path, col_select = c("city", "grid_id", "year", "pop_count")))
  setnames(x, "city", "city_key")
  x[, population_log := log1p(pmax(pop_count, 0))]
  x[, .(population_log = mean(population_log, na.rm = TRUE)), by = .(city_key, grid_id, year)]
}

read_city_annual_features <- function(city_key, families, root = project_root()) {
  families <- sort(unique(families))
  parts <- lapply(families, function(family) {
    read_city_annual_family(city_key, family, root)
  })
  result <- Reduce(function(left, right) {
    merge(left, right, by = c("city_key", "grid_id", "year"), all = TRUE)
  }, parts)
  setorder(result, grid_id, year)
  result[]
}

read_city_monthly_housing <- function(city_key, root = project_root(),
                                      price_measure = "median") {
  assert_choice(price_measure, c("median", "hedonic"), "price_measure")
  if (price_measure == "hedonic") {
    # Hedonic quality-adjusted panel (Lianjia 22 cities), built by
    # scripts/labels/build_housing_hedonic.py into the rebuildable outputs
    # tree; never a data/active asset.
    .hed_dir <- .resolve_path(
      "HOUSING_HEDONIC_DIR", root,
      "outputs", "causal_labels", "housing_hedonic"
    )
    path <- file.path(.hed_dir, paste0(city_key, "_monthly.parquet"))
    if (!file.exists(path)) {
      stop(
        "Hedonic housing panel missing for ", city_key, ": ", path,
        ". Run build_housing_hedonic.py first (Lianjia cities only)."
      )
    }
    x <- as.data.table(read_parquet(path, col_select = c(
      "city_key", "grid_id", "observed_month", "adjusted_price_median"
    )))
    x[, month := as.IDate(format(as.IDate(observed_month), "%Y-%m-01"))]
    x[is.finite(adjusted_price_median) & adjusted_price_median > 0,
      housing_log_price := log(adjusted_price_median)]
    return(x[!is.na(housing_log_price), .(
      housing_log_price = median(housing_log_price, na.rm = TRUE)
    ), by = .(city_key, grid_id, month)])
  }
  .dir <- .resolve_path("PANEL_HOUSING_MONTHLY_DIR", root, "data", "active", "panels", "housing_grid_month")
  path <- file.path(.dir, paste0(city_key, ".parquet"))
  x <- as.data.table(read_parquet(path, col_select = c(
    "city_key", "grid_id", "observed_month", "log_price_raw_median"
  )))
  x[, month := as.IDate(format(as.IDate(observed_month), "%Y-%m-01"))]
  x[is.finite(log_price_raw_median), .(
    housing_log_price = median(log_price_raw_median)
  ), by = .(city_key, grid_id, month)]
}

read_city_monthly_viirs <- function(city_key, months, root = project_root(),
                                    strict = TRUE) {
  months <- sort(unique(as.IDate(format(as.IDate(months), "%Y-%m-01"))))
  parts <- lapply(months, function(period) {
    year <- as.integer(format(period, "%Y"))
    month_number <- as.integer(format(period, "%m"))
    .monthly_dir <- .resolve_path("VIIRS_MONTHLY_DIR", root, "data", "active", "curated", "viirs", "monthly")
    path <- file.path(
      .monthly_dir,
      paste0("city_key=", city_key), paste0("year=", year),
      sprintf("month=%02d", month_number), "part.parquet"
    )
    if (!file.exists(path)) {
      if (strict) {
        stop(
          "Missing VIIRS monthly cache partition: ", path,
          ". Run ensure_viirs_monthly_cache.py before the R estimator."
        )
      }
      # Non-strict (label-path) reads: a missing partition means the post
      # horizon extends past the 2012-2024 cache ceiling (openings >= 2023).
      # Skip the month; the caller's panel merge turns it into NA and the
      # label availability per horizon is decided downstream, so late
      # openings keep the horizons the cache can cover.
      return(NULL)
    }
    x <- as.data.table(read_parquet(
      path,
      col_select = c("grid_id", "avg_rad", "valid_days_mean", "source_point_count")
    ))
    x[, `:=`(
      city_key = city_key,
      month = period,
      viirs_avg_asinh = asinh(avg_rad)
    )]
    x[, .(
      city_key, grid_id, month, viirs_avg_asinh,
      viirs_valid_days_mean = valid_days_mean,
      viirs_source_point_count = source_point_count
    )]
  })
  result <- rbindlist(parts, use.names = TRUE)
  if (any(grepl("^V[0-9]+$", names(result)))) {
    warning("Some parts produced unnamed columns during rbindlist; check schema alignment")
  }
  result
}

read_city_monthly_outcome <- function(city_key, family, months,
                                      root = project_root(),
                                      price_measure = "median",
                                      strict = TRUE) {
  assert_choice(family, c("housing", "viirs"), "monthly family")
  if (family == "housing") {
    return(read_city_monthly_housing(city_key, root, price_measure)[month %in% months])
  }
  read_city_monthly_viirs(city_key, months, root, strict = strict)
}

treatment_signatures <- function(root = project_root()) {
  .dir <- .resolve_path("FORMAL_MATCHING_DIR", root, "data", "active", "causal", "formal_matching_inputs")
  path <- file.path(.dir, "formal_target_support.parquet")
  x <- as.data.table(read_parquet(path))
  families <- names(complete_estimator_spec()$families)
  complete_columns <- paste0(families, "_complete")
  x[, signature := apply(.SD, 1L, function(row) {
    present <- families[as.logical(row)]
    paste(sort(present), collapse = "+")
  }), .SDcols = complete_columns]
  x[]
}

select_treated_group <- function(city_key, cohort, signature, treatment_order = NULL,
                                 root = project_root()) {
  requested_city <- city_key
  requested_signature <- signature
  treatments <- read_treatments(root)
  support <- treatment_signatures(root)[, .(treatment_order, signature)]
  x <- merge(treatments, support, by = "treatment_order", all.x = TRUE)
  x <- x[city_key == requested_city & opening_year == as.integer(cohort)]
  if (!is.null(treatment_order)) {
    requested_order <- as.integer(treatment_order)
    if (is.na(requested_order)) stop("treatment_order must not be NA")
    x <- x[treatment_order == requested_order]
  }
  if (requested_signature != "auto") x <- x[signature == requested_signature]
  if (!nrow(x)) stop("No treated units for requested city/cohort/signature")
  if (requested_signature == "auto") {
    selected <- x[nzchar(signature), .N, by = signature][order(-N, signature)]$signature[[1L]]
    x <- x[signature == selected]
  }
  attr(x, "signature") <- unique(x$signature)[[1L]]
  x[]
}

make_unit_map <- function(treated, donors) {
  units <- rbindlist(list(
    treated[, .(city_key, grid_id, role = "treated")],
    donors[, .(city_key, grid_id, role = "donor")]
  ), use.names = TRUE)
  units <- unique(units, by = c("city_key", "grid_id"))
  setorder(units, role, grid_id)
  units[, unit_id := seq_len(.N)]
  units[, unit_key := paste(city_key, grid_id, sep = "::")]
  units[]
}

build_annual_estimator_panel <- function(
    city_key, cohort, outcome_family, signature = "auto", leads = 1:3,
    treatment_order = NULL, root = project_root()) {
  spec <- complete_estimator_spec()
  assert_choice(outcome_family, names(spec$families), "outcome_family")
  treated <- select_treated_group(
    city_key, cohort, signature, treatment_order = treatment_order, root = root
  )
  signature <- attr(treated, "signature")
  if (!is.null(treatment_order) && nrow(treated) != 1L) {
    stop("Requested treatment_order is not in the selected annual group")
  }
  covariate_families <- parse_family_signature(signature)
  needed_families <- sort(unique(c(covariate_families, outcome_family)))
  features <- read_city_annual_features(city_key, needed_families, root)
  donors <- read_city_donors(city_key, root)
  unit_map <- make_unit_map(treated, donors)
  if (spec$timing$annual_anticipation_years >= spec$annual$lag) {
    stop("annual_anticipation_years must be strictly less than annual lag")
  }
  pre_years <- seq.int(
    as.integer(cohort) - spec$annual$lag,
    as.integer(cohort) - 1L - spec$timing$annual_anticipation_years
  )
  post_years <- as.integer(cohort) + as.integer(leads)
  years <- c(pre_years, post_years)
  panel <- CJ(unit_id = unit_map$unit_id, year = years, sorted = TRUE)
  panel <- merge(panel, unit_map[, .(unit_id, city_key, grid_id, role, unit_key)], by = "unit_id")
  panel <- merge(panel, features, by = c("city_key", "grid_id", "year"), all.x = TRUE)
  missing_years <- setdiff(years, unique(panel$year))
  if (length(missing_years)) {
    stop("Requested years are not present in the feature data: ",
         paste(sort(missing_years), collapse = ", "))
  }
  first_treated_year <- as.integer(cohort) + spec$timing$annual_first_treated_year_offset
  panel[, time_id := match(year, years)]
  panel[, D := as.integer(role == "treated" & year >= first_treated_year)]
  setorder(panel, unit_id, time_id)
  list(
    panel = panel, unit_map = unit_map, treated = treated, donors = donors,
    signature = signature, covariates = signature_variables(signature),
    outcomes = spec$families[[outcome_family]], lag = spec$annual$lag,
    covariate_lags = 1:spec$annual$lag, leads = as.integer(leads),
    frequency = "annual", cohort = as.integer(cohort), city_key = city_key,
    outcome_family = outcome_family,
    annual_anticipation_years = spec$timing$annual_anticipation_years
  )
}

build_monthly_estimator_panel <- function(
    city_key, cohort_month, outcome_family = "housing", signature = "auto",
    leads = 1:24, anticipation_months = NULL, treatment_order = NULL,
    price_measure = "median", root = project_root()) {
  spec <- complete_estimator_spec()
  assert_choice(outcome_family, c("housing", "viirs"), "monthly outcome_family")
  requested_city <- city_key
  requested_signature <- signature
  cohort_month <- as.IDate(paste0(substr(cohort_month, 1L, 7L), "-01"))
  cohort_year <- as.integer(format(cohort_month, "%Y"))
  all_treated <- read_treatments(root)
  support <- treatment_signatures(root)[, .(treatment_order, signature)]
  treated <- merge(all_treated, support, by = "treatment_order", all.x = TRUE)
  treated <- treated[city_key == requested_city & opening_month_date == cohort_month]
  if (!is.null(treatment_order)) {
    requested_order <- as.integer(treatment_order)
    treated <- treated[treatment_order == requested_order]
  }
  if (requested_signature != "auto") treated <- treated[signature == requested_signature]
  if (!nrow(treated)) stop("No treated units for requested city/month/signature")
  if (requested_signature == "auto") {
    selected <- treated[nzchar(signature), .N, by = signature][order(-N, signature)]$signature[[1L]]
    treated <- treated[signature == selected]
    signature <- selected
  }
  if (!is.null(treatment_order) && nrow(treated) != 1L) {
    stop("Requested treatment_order is not in the selected monthly group")
  }
  covariate_families <- parse_family_signature(signature)
  # When the monthly outcome family is also a matching family, its monthly
  # path supplies that covariate. Merging the annual version would create
  # .x/.y columns and silently remove the registered variable name.
  annual_families <- setdiff(covariate_families, outcome_family)
  annual <- if (length(annual_families)) {
    read_city_annual_features(city_key, annual_families, root)
  } else NULL
  donors <- read_city_donors(city_key, root)
  unit_map <- make_unit_map(treated, donors)
  calendar <- monthly_event_calendar(
    cohort_month, lag = spec$monthly$lag, leads = leads,
    anticipation_months = anticipation_months
  )
  months <- calendar$model_months
  outcomes <- read_city_monthly_outcome(
    city_key, outcome_family, months, root,
    price_measure = if (outcome_family == "housing") price_measure else "median"
  )
  panel <- CJ(unit_id = unit_map$unit_id, month = as.IDate(months), sorted = TRUE)
  panel <- merge(panel, unit_map[, .(unit_id, city_key, grid_id, role, unit_key)], by = "unit_id")
  panel <- merge(panel, outcomes, by = c("city_key", "grid_id", "month"), all.x = TRUE)
  panel[, year := as.integer(format(month, "%Y"))]
  if (!is.null(annual)) {
    panel <- merge(panel, annual, by = c("city_key", "grid_id", "year"), all.x = TRUE)
  }
  # The partial opening month and anticipation window are absent from model time.
  # The first treated observation is the first full calendar month after opening.
  panel[, time_id := match(month, calendar$model_months)]
  panel[, D := as.integer(role == "treated" & time_id > spec$monthly$lag)]
  setorder(panel, unit_id, time_id)
  list(
    panel = panel, unit_map = unit_map, treated = treated, donors = donors,
    signature = signature, covariates = signature_variables(signature),
    outcomes = spec$families[[outcome_family]], lag = spec$monthly$lag,
    covariate_lags = spec$monthly$covariate_lags,
    leads = as.integer(leads), frequency = "monthly", cohort = as.character(cohort_month),
    city_key = city_key, outcome_family = outcome_family, event_calendar = calendar
  )
}


build_monthly_housing_estimator_panel <- function(
    city_key, cohort_month, signature = "auto", leads = 1:24,
    anticipation_months = NULL, treatment_order = NULL, price_measure = "median",
    root = project_root()) {
  build_monthly_estimator_panel(
    city_key = city_key, cohort_month = cohort_month, outcome_family = "housing",
    signature = signature, leads = leads,
    anticipation_months = anticipation_months, treatment_order = treatment_order,
    price_measure = price_measure, root = root
  )
}

make_preonly_matching_frame <- function(design) {
  treatment_time <- design$lag + 1L
  if (identical(design$frequency, "monthly")) {
    covariate_rows <- design$panel[
      time_id <= design$lag,
      c("unit_id", "time_id", design$covariates), with = FALSE
    ]
    # Three non-overlapping 12-month blocks: lag1 is the nearest clean year
    # and is held out; lag2/lag3 identify the control from older trajectories.
    n_blocks <- design$lag %/% 12L
    covariate_rows[, lag := n_blocks - ((time_id - 1L) %/% 12L)]
    covariate_rows <- covariate_rows[, lapply(.SD, function(value) {
      finite <- value[is.finite(value)]
      if (length(finite)) mean(finite) else NA_real_
    }), by = .(unit_id, lag), .SDcols = design$covariates]
  } else {
    covariate_rows <- design$panel[
      time_id %in% (treatment_time - design$covariate_lags),
      c("unit_id", "time_id", design$covariates), with = FALSE
    ]
    covariate_rows[, lag := treatment_time - time_id]
  }
  long <- melt(
    covariate_rows, id.vars = c("unit_id", "lag"),
    measure.vars = design$covariates,
    variable.name = "covariate", value.name = "value"
  )
  long[, feature := paste0(covariate, "__lag", lag)]
  covariate_wide <- dcast(long, unit_id ~ feature, value.var = "value")
  feature_columns <- setdiff(names(covariate_wide), "unit_id")
  frame <- merge(
    design$unit_map[, .(unit_id, unit_key, grid_id, role)],
    covariate_wide, by = "unit_id"
  )
  frame <- frame[complete.cases(frame[, ..feature_columns])]
  frame[, Tr := as.integer(role == "treated")]
  if (!any(frame$Tr == 1L) || !any(frame$Tr == 0L)) {
    stop("No treated/control comparison with complete pre-treatment covariates")
  }
  list(frame = frame, features = feature_columns)
}

attach_prepost_outcome <- function(preonly_frame, design, outcome, horizon) {
  baseline_time <- design$lag
  post_time <- design$lag + as.integer(horizon)
  if (horizon < 1L || post_time > max(design$panel$time_id)) {
    stop("horizon must identify an available post-treatment period")
  }
  values <- dcast(
    design$panel[time_id %in% c(baseline_time, post_time),
                 .(unit_id, time_id, value = get(outcome))],
    unit_id ~ time_id, value.var = "value"
  )
  values[, delta_outcome := get(as.character(post_time)) - get(as.character(baseline_time))]
  merge(preonly_frame, values[, .(unit_id, delta_outcome)], by = "unit_id", all.x = TRUE)
}

active_matching_matrix <- function(prepared) {
  X <- as.matrix(prepared$frame[, prepared$features, with = FALSE])
  storage.mode(X) <- "double"
  if (any(!is.finite(X))) stop("Pre-treatment matching matrix contains non-finite values")
  feature_sd <- apply(X, 2L, stats::sd)
  active <- is.finite(feature_sd) & feature_sd > sqrt(.Machine$double.eps)
  if (!any(active)) stop("All pre-treatment matching covariates have zero variance")
  list(
    X = X[, active, drop = FALSE],
    active_features = prepared$features[active],
    dropped_features = prepared$features[!active]
  )
}

select_preonly_pairs <- function(prepared, matching_spec = complete_estimator_spec()$abadie_imbens,
                                 match_features = NULL, support_features = NULL) {
  if (!requireNamespace("Matching", quietly = TRUE)) stop("Matching package is required")
  if (is.null(match_features)) match_features <- prepared$features
  if (is.null(support_features)) support_features <- match_features
  unknown <- setdiff(unique(c(match_features, support_features)), prepared$features)
  if (length(unknown)) stop("Unknown matching features: ", paste(unknown, collapse = ", "))
  matching_prepared <- list(frame = prepared$frame, features = match_features)
  matrix_info <- active_matching_matrix(matching_prepared)
  frame <- prepared$frame
  donor_rows <- frame$Tr == 0L
  support_active <- matrix_info$active_features[
    matrix_info$active_features %in% support_features
  ]
  if (length(support_active)) {
    lower <- apply(matrix_info$X[donor_rows, support_active, drop = FALSE], 2L, min)
    upper <- apply(matrix_info$X[donor_rows, support_active, drop = FALSE], 2L, max)
    inside <- sweep(matrix_info$X[, support_active, drop = FALSE], 2L, lower, ">=") &
      sweep(matrix_info$X[, support_active, drop = FALSE], 2L, upper, "<=")
    supported_treated <- frame$Tr == 1L & apply(inside, 1L, all)
  } else {
    supported_treated <- rep(TRUE, nrow(frame))
  }
  unsupported_ids <- frame$unit_id[frame$Tr == 1L & !supported_treated]
  keep <- frame$Tr == 0L | supported_treated
  frame <- frame[keep]
  matrix_info$X <- matrix_info$X[keep, , drop = FALSE]
  if (!any(frame$Tr == 1L)) stop("No treated unit is inside explicit donor common support")
  deterministic_exact <- !isTRUE(matching_spec$ties) &&
    identical(as.numeric(matching_spec$distance.tolerance), 0) &&
    identical(as.integer(matching_spec$Weight), 2L)
  if (deterministic_exact) {
    # Matching::Match randomly samples controls when an exact kth-distance tie
    # remains and ties=FALSE.  The GPU contract instead freezes donor-row order;
    # calculate the same Weight=2 Mahalanobis ranking directly on this design
    # path so R references and GPU input-only runs have identical semantics.
    treated_rows <- which(frame$Tr == 1L)
    donor_rows <- which(frame$Tr == 0L)
    inverse <- stable_covariance_inverse(matrix_info$X)$inverse
    deterministic_pairs <- rbindlist(lapply(treated_rows, function(treated_row) {
      delta <- sweep(
        matrix_info$X[donor_rows, , drop = FALSE],
        2L, matrix_info$X[treated_row, ], "-"
      )
      distance <- sqrt(pmax(rowSums((delta %*% inverse) * delta), 0))
      selected <- head(
        order(distance, donor_rows, method = "radix"),
        min(as.integer(matching_spec$M), length(donor_rows))
      )
      data.table(
        treated_row = rep.int(as.integer(treated_row), length(selected)),
        control_row = as.integer(donor_rows[selected]),
        pair_weight = rep.int(1 / length(selected), length(selected))
      )
    }))
    fit <- list(
      index.treated = deterministic_pairs$treated_row,
      index.control = deterministic_pairs$control_row,
      weights = deterministic_pairs$pair_weight,
      deterministic_exact_ties = TRUE
    )
  } else {
    fit <- Matching::Match(
      Y = NULL, Tr = frame$Tr, X = matrix_info$X,
      estimand = matching_spec$estimand,
      M = matching_spec$M, replace = matching_spec$replace,
      # Matching 4.10-15 fails internally for a one-treated-unit risk set when
      # CommonSupport=TRUE. Support is applied explicitly above using only X.
      ties = matching_spec$ties, CommonSupport = FALSE,
      Weight = matching_spec$Weight, BiasAdjust = FALSE, Var.calc = 0L,
      distance.tolerance = matching_spec$distance.tolerance
    )
  }
  if (!length(fit$index.treated) || !length(fit$index.control)) {
    stop("Matching::Match produced no pre-treatment pairs")
  }
  pairs <- data.table(
    treated_row = as.integer(fit$index.treated),
    control_row = as.integer(fit$index.control),
    pair_weight = as.numeric(fit$weights)
  )
  unit_key <- if ("unit_key" %in% names(frame)) frame$unit_key else rep(NA_character_, nrow(frame))
  grid_id <- if ("grid_id" %in% names(frame)) frame$grid_id else rep(NA_character_, nrow(frame))
  pairs[, `:=`(
    treated_unit_id = frame$unit_id[treated_row],
    treated_unit_key = unit_key[treated_row],
    treated_grid_id = grid_id[treated_row],
    control_unit_id = frame$unit_id[control_row],
    control_unit_key = unit_key[control_row],
    control_grid_id = grid_id[control_row]
  )]
  list(
    fit = fit, pairs = pairs, X = matrix_info$X, frame = frame,
    unsupported_treated_unit_ids = unsupported_ids,
    active_features = matrix_info$active_features,
    dropped_features = matrix_info$dropped_features,
    support_features = support_active
  )
}

split_holdout_features <- function(features, holdout_lags = 1L) {
  lag <- suppressWarnings(as.integer(sub("^.*__lag", "", features)))
  holdout <- features[is.finite(lag) & lag %in% holdout_lags]
  training <- setdiff(features, holdout)
  if (!length(training) || !length(holdout)) {
    stop("Pre-only design requires non-empty training and held-out features")
  }
  list(training = training, holdout = holdout)
}

stable_covariance_inverse <- function(X) {
  covariance <- stats::cov(X)
  if (ncol(X) == 1L) covariance <- matrix(covariance, 1L, 1L)
  eig <- eigen((covariance + t(covariance)) / 2, symmetric = TRUE)
  cutoff <- max(abs(eig$values)) * sqrt(.Machine$double.eps)
  keep <- eig$values > cutoff
  if (!any(keep)) stop("Training covariance has zero numerical rank")
  vectors <- eig$vectors[, keep, drop = FALSE]
  inverse <- vectors %*% diag(1 / eig$values[keep], nrow = sum(keep)) %*% t(vectors)
  list(inverse = inverse, rank = sum(keep))
}

# Two-stage refinement: among the outcome-history matched candidate controls
# (rows of `pairs`), pick per treated unit the candidate with the smallest
# donor-covariance Mahalanobis distance on the time-invariant covariates.
# This lets location/transit covariates shape control choice without letting
# them dominate the outcome-history distance (which would collapse the
# matched path; treated grids are by construction closer to the transit
# network than the 1km-excluded donors).
static_balance_refine <- function(pairs, frame, static_features) {
  if (!length(static_features)) return(copy(pairs[1L]))
  donor <- frame[Tr == 0L]
  X <- as.matrix(donor[, ..static_features])
  storage.mode(X) <- "double"
  donor_sd <- apply(X, 2L, stats::sd, na.rm = TRUE)
  active <- static_features[is.finite(donor_sd) & donor_sd > sqrt(.Machine$double.eps)]
  if (!length(active)) return(copy(pairs[1L]))
  X <- X[, active, drop = FALSE]
  inverse <- stable_covariance_inverse(X)$inverse
  treated_ids <- unique(pairs$treated_unit_id)
  rbindlist(lapply(treated_ids, function(tid) {
    candidates <- pairs[treated_unit_id == tid]
    treated <- as.numeric(frame[candidates$treated_row[[1L]], ..active])
    static_distance <- vapply(seq_len(nrow(candidates)), function(j) {
      delta <- treated - as.numeric(frame[candidates$control_row[[j]], ..active])
      sqrt(max(as.numeric(t(delta) %*% inverse %*% delta), 0))
    }, numeric(1L))
    candidates[which.min(static_distance)]
  }))
}

calibrate_preonly_placebos <- function(frame, training_features, holdout_features,
                                       sample_n = 200L, quantile_probability = 0.95,
                                       static_features = NULL, M = 1L) {
  donors <- frame[Tr == 0L]
  if (nrow(donors) < 3L) stop("At least three donors are required for placebo calibration")
  training <- as.matrix(donors[, ..training_features])
  holdout <- as.matrix(donors[, ..holdout_features])
  storage.mode(training) <- "double"
  storage.mode(holdout) <- "double"
  inverse <- stable_covariance_inverse(training)
  holdout_sd <- apply(holdout, 2L, stats::sd)
  informative <- holdout_sd > sqrt(.Machine$double.eps)
  if (!any(informative)) {
    stop("All holdout features have near-zero variance across donors")
  }
  static_inverse <- NULL
  static_active <- character()
  if (!is.null(static_features) && length(static_features)) {
    static_matrix <- as.matrix(donors[, ..static_features])
    storage.mode(static_matrix) <- "double"
    static_sd <- apply(static_matrix, 2L, stats::sd, na.rm = TRUE)
    static_active <- static_features[
      is.finite(static_sd) & static_sd > sqrt(.Machine$double.eps)
    ]
    if (length(static_active)) {
      static_inverse <- stable_covariance_inverse(
        static_matrix[, static_active, drop = FALSE]
      )$inverse
    }
  }
  M <- max(1L, min(as.integer(M), nrow(donors)))
  sample_n <- min(as.integer(sample_n), nrow(donors))
  sampled <- unique(as.integer(round(seq(1, nrow(donors), length.out = sample_n))))
  placebo <- rbindlist(lapply(sampled, function(row) {
    delta <- sweep(training, 2L, training[row, ], "-")
    distance <- sqrt(pmax(rowSums((delta %*% inverse$inverse) * delta), 0))
    distance[row] <- Inf
    candidates <- head(order(distance, method = "radix"), M)
    chosen <- candidates[[1L]]
    if (length(static_active)) {
      target_static <- as.numeric(static_matrix[row, static_active, drop = FALSE])
      static_distance <- vapply(candidates, function(cid) {
        delta_static <- target_static -
          as.numeric(static_matrix[cid, static_active, drop = FALSE])
        sqrt(max(as.numeric(t(delta_static) %*% static_inverse %*% delta_static), 0))
      }, numeric(1L))
      chosen <- candidates[which.min(static_distance)]
    }
    holdout_gap <- (holdout[row, ] - holdout[chosen, ]) / holdout_sd
    holdout_gap[!is.finite(holdout_gap)] <- NA_real_
    data.table(
      pseudo_treated_unit_id = donors$unit_id[row],
      pseudo_control_unit_id = donors$unit_id[chosen],
      training_distance = distance[chosen],
      holdout_rms_standardized_gap = sqrt(mean(holdout_gap^2, na.rm = TRUE)),
      holdout_max_abs_standardized_gap = max(abs(holdout_gap), na.rm = TRUE)
    )
  }))
  thresholds <- placebo[, .(
    training_distance_threshold = as.numeric(stats::quantile(
      training_distance, quantile_probability, names = FALSE, type = 8
    )),
    holdout_rms_threshold = as.numeric(stats::quantile(
      holdout_rms_standardized_gap, quantile_probability, names = FALSE, type = 8
    )),
    holdout_max_abs_threshold = as.numeric(stats::quantile(
      holdout_max_abs_standardized_gap, quantile_probability, names = FALSE, type = 8
    )),
    calibration_pairs = .N,
    quantile_probability = quantile_probability,
    covariance_rank = inverse$rank
  )]
  list(placebo = placebo, thresholds = thresholds, inverse = inverse$inverse,
       holdout_sd = holdout_sd)
}

evaluate_preonly_pair_quality <- function(pair, frame, training_features,
                                          holdout_features, calibration) {
  treated <- frame[pair$treated_row]
  control <- frame[pair$control_row]
  training_gap <- as.numeric(treated[, ..training_features]) -
    as.numeric(control[, ..training_features])
  training_distance <- sqrt(max(as.numeric(
    t(training_gap) %*% calibration$inverse %*% training_gap
  ), 0))
  holdout_gap <- (
    as.numeric(treated[, ..holdout_features]) -
      as.numeric(control[, ..holdout_features])
  ) / calibration$holdout_sd
  holdout_gap[!is.finite(holdout_gap)] <- NA_real_
  metrics <- data.table(
    training_distance = training_distance,
    holdout_rms_standardized_gap = sqrt(mean(holdout_gap^2, na.rm = TRUE)),
    holdout_max_abs_standardized_gap = max(abs(holdout_gap), na.rm = TRUE)
  )
  thresholds <- calibration$thresholds
  metrics[, accepted :=
    training_distance <= thresholds$training_distance_threshold &
    holdout_rms_standardized_gap <= thresholds$holdout_rms_threshold &
    holdout_max_abs_standardized_gap <= thresholds$holdout_max_abs_threshold]
  cbind(metrics, thresholds)
}

pair_change_labels <- function(pairs, outcome_frame, outcome, horizon) {
  changes <- outcome_frame[, .(unit_id, delta_outcome)]
  result <- merge(
    pairs, changes,
    by.x = "treated_unit_id", by.y = "unit_id", all.x = TRUE
  )
  setnames(result, "delta_outcome", "treated_change")
  result <- merge(
    result, changes,
    by.x = "control_unit_id", by.y = "unit_id", all.x = TRUE
  )
  setnames(result, "delta_outcome", "counterfactual_change")
  result[, `:=`(
    outcome = outcome,
    horizon = as.integer(horizon),
    causal_response_label = treated_change - counterfactual_change,
    label_available = is.finite(treated_change) & is.finite(counterfactual_change)
  )]
  result[]
}

pair_preonly_diagnostics <- function(pairs, frame, features) {
  donor_sd <- vapply(frame[Tr == 0L, ..features], stats::sd, numeric(1L), na.rm = TRUE)
  long <- rbindlist(lapply(seq_len(nrow(pairs)), function(index) {
    pair <- pairs[index]
    treated <- as.numeric(frame[pair$treated_row, ..features])
    control <- as.numeric(frame[pair$control_row, ..features])
    gap <- treated - control
    standardized <- gap / donor_sd
    standardized[!is.finite(standardized)] <- NA_real_
    data.table(
      pair_index = index,
      treated_unit_id = pair$treated_unit_id,
      control_unit_id = pair$control_unit_id,
      feature = features,
      treated_value = treated,
      control_value = control,
      raw_gap = gap,
      standardized_gap = standardized
    )
  }))
  summary <- long[, .(
    preonly_rms_standardized_gap = sqrt(mean(standardized_gap^2, na.rm = TRUE)),
    preonly_max_abs_standardized_gap = max(abs(standardized_gap), na.rm = TRUE),
    active_feature_count = sum(is.finite(standardized_gap))
  ), by = .(pair_index, treated_unit_id, control_unit_id)]
  list(long = long, summary = summary)
}

normalize_gsc_labels <- function(fit, panel, treated_units, pre_periods, post_periods,
                                  outcome_family, outcome, post_event_time = NULL,
                                  pre_event_time = NULL) {
  if (is.null(fit$Y.ct) || is.null(fit$id)) stop("gsynth object lacks counterfactual paths")
  model_periods <- c(pre_periods, post_periods)
  if (is.null(post_event_time)) post_event_time <- seq_along(post_periods)
  if (is.null(pre_event_time)) {
    stop("pre_event_time is required; use actual calendar offsets")
  }
  if (length(post_event_time) != length(post_periods)) {
    stop("post_event_time length does not match post periods")
  }
  if (length(pre_event_time) != length(pre_periods)) {
    stop("pre_event_time length does not match pre periods")
  }
  rbindlist(lapply(seq_len(nrow(treated_units)), function(index) {
    target <- treated_units[index]
    fitted_column <- match(as.character(target$gsc_unit_id), as.character(fit$id))
    if (is.na(fitted_column)) {
      stop("gsynth counterfactual is missing treated unit ", target$gsc_unit_id)
    }
    target_panel <- panel[gsc_unit_id == target$gsc_unit_id][order(time_id)]
    if (nrow(target_panel) != length(model_periods)) stop("Treated outcome path has wrong length")
    result <- data.table(
      treatment_order = target$treatment_order,
      city_key = target$city_key,
      grid_id = target$grid_id,
      outcome_family = outcome_family,
      outcome = outcome,
      period = model_periods,
      event_time = c(as.integer(pre_event_time), as.integer(post_event_time)),
      observed = target_panel$value,
      counterfactual = as.numeric(fit$Y.ct[, fitted_column])
    )
    result[, `:=`(
      causal_response_label = observed - counterfactual,
      label_available = is.finite(observed) & is.finite(counterfactual),
      method = "xu_2017_gsynth",
      selected_factors = as.integer(fit$r.cv)
    )]
    result
  }))
}

# Apply the monthly observation-window semantics after the estimator produces
# monthly observed/counterfactual paths. Pre-treatment paths remain monthly for
# event-study diagnostics; post-treatment horizons use the mean over
# [max(1, h-window+1), h].  The frozen label contract requires every month in
# the requested window to have finite observed and counterfactual values;
# effective counts are persisted so unsupported windows remain visible in
# downstream quality and distribution reports.
apply_post_observation_window <- function(paths, window = 1L) {
  window <- as.integer(window)
  if (window < 1L || window > 6L) stop("window must be in 1..6 months")
  if (window == 1L || !nrow(paths) || !any(paths$event_time > 0L)) {
    return(paths)
  }
  paths <- as.data.table(paths)
  pre <- paths[event_time < 0L]
  post <- paths[event_time > 0L]
  horizons <- sort(unique(as.integer(post$event_time)))
  windowed <- rbindlist(lapply(horizons, function(horizon) {
    start <- max(1L, horizon - window + 1L)
    part <- post[event_time >= start & event_time <= horizon]
    if (!nrow(part)) return(NULL)
    result <- copy(part[which.max(event_time)])
    # Early horizons contain fewer than `window` months; later horizons contain
    # exactly `window` months.  Partial windows are not valid W-month labels.
    minimum_window_n <- min(window, as.integer(horizon))
    finite_mean <- function(value) {
      value <- value[is.finite(value)]
      if (!length(value)) NA_real_ else mean(value)
    }
    n_observed <- sum(is.finite(part$observed))
    n_counterfactual <- sum(is.finite(part$counterfactual))
    observed_mean <- finite_mean(part$observed)
    counterfactual_mean <- finite_mean(part$counterfactual)
    window_supported <- (
      nrow(part) == minimum_window_n &&
        n_observed == minimum_window_n &&
        n_counterfactual == minimum_window_n
    )
    result[, `:=`(
      observed = observed_mean,
      counterfactual = counterfactual_mean,
      causal_response_label = if (window_supported &&
                                  is.finite(observed_mean) &&
                                  is.finite(counterfactual_mean)) {
        observed_mean - counterfactual_mean
      } else NA_real_,
      minimum_window_n = as.integer(minimum_window_n),
      effective_n_observed = as.integer(n_observed),
      effective_n_counterfactual = as.integer(n_counterfactual),
      window_supported = window_supported
    )]
    result[, label_available := window_supported &
      is.finite(observed) & is.finite(counterfactual) &
      is.finite(causal_response_label)]
    if ("standard_error" %in% names(part)) {
      se_values <- part$standard_error
      result[, standard_error := if (all(is.finite(se_values))) {
        sqrt(sum(se_values^2)) / nrow(part)
      } else NA_real_]
      result[, confidence_lower := if (is.finite(standard_error)) {
        causal_response_label - 1.96 * standard_error
      } else NA_real_]
      result[, confidence_upper := if (is.finite(standard_error)) {
        causal_response_label + 1.96 * standard_error
      } else NA_real_]
      result[, p_value := if (is.finite(standard_error) && standard_error > 0) {
        2 * stats::pnorm(-abs(causal_response_label / standard_error))
      } else NA_real_]
    }
    result[, uncertainty_source := paste0(
      as.character(uncertainty_source[[1L]]), "_window", window
    )]
    result
  }), use.names = TRUE, fill = TRUE)
  setorder(windowed, event_time)
  rbindlist(list(pre, windowed), use.names = TRUE, fill = TRUE)
}

attach_single_target_gsc_uncertainty <- function(paths, fit, nboots,
                                                  effect_scale = 1) {
  paths <- copy(as.data.table(paths))
  paths[, `:=`(
    standard_error = NA_real_, confidence_lower = NA_real_,
    confidence_upper = NA_real_, p_value = NA_real_,
    bootstrap_repetitions = as.integer(nboots),
    uncertainty_source = paste0(class(fit)[[1L]], "_parametric_bootstrap")
  )]
  if (uniqueN(paths$treatment_order) != 1L) {
    stop("Grid-label production requires one treated target per fit")
  }
  if (is.null(fit$est.att)) stop(class(fit)[[1L]], " fit lacks bootstrap ATT uncertainty")
  effect_scale <- as.numeric(effect_scale)
  if (length(effect_scale) != 1L || !is.finite(effect_scale) || effect_scale <= 0) {
    stop("GSC uncertainty effect_scale must be one finite positive value")
  }
  est_att <- as.data.frame(fit$est.att)
  uncertainty <- if ("event_time" %in% names(est_att)) {
    as.data.table(est_att)
  } else {
    row_names <- rownames(est_att)
    if (is.null(row_names) || !length(row_names) ||
        identical(as.character(row_names), as.character(seq_len(nrow(est_att))))) {
      stop(
        "Estimator uncertainty output lacks explicit event-time identifiers; ",
        "refusing to align uncertainty by default row position"
      )
    }
    as.data.table(est_att, keep.rownames = "event_time")
  }
  required <- c("S.E.", "CI.lower", "CI.upper", "p.value")
  if (!all(required %in% names(uncertainty))) {
    stop("gsynth uncertainty output lacks required inference columns")
  }
  if ("event_time" %in% names(fit$est.att)) {
    uncertainty[, event_time := as.integer(fit$est.att[["event_time"]])]
  } else {
    uncertainty[, event_time := suppressWarnings(as.integer(as.character(event_time)))]
  }
  if (anyNA(uncertainty$event_time)) {
    stop("gsynth uncertainty output lacks usable event-time identifiers")
  }
  # gsynth/fect report the treatment/opening period as event_time == 0.  The
  # project specification excludes that partial month/year.  Positive
  # horizons are aligned by their formal event_time.  Negative estimator
  # event_time values are model-index offsets, however, and therefore cannot
  # be compared directly with the calendar offsets used in the normalized
  # monthly path (e.g. -35:-1 versus -42:-7).  Align negative rows by their
  # ordered pre-treatment paths and retain the earliest path row as NA when
  # the estimator omits it.
  uncertainty <- uncertainty[event_time != 0L]
  if (anyDuplicated(uncertainty$event_time)) {
    stop("Estimator uncertainty output has duplicated event-time identifiers")
  }
  path_event_time <- as.integer(paths$event_time)
  if (anyNA(path_event_time) || anyDuplicated(path_event_time)) {
    stop("Normalized target path has invalid or duplicated event-time identifiers")
  }
  post_event_time <- path_event_time[path_event_time > 0L]
  missing_post <- setdiff(post_event_time, uncertainty$event_time)
  if (length(missing_post)) {
    stop(
      "Estimator uncertainty output lacks formal post-treatment event times: ",
      paste(sort(missing_post), collapse = ", ")
    )
  }
  matched <- rep(NA_integer_, length(path_event_time))
  post_index <- which(path_event_time > 0L)
  matched[post_index] <- match(
    path_event_time[post_index], uncertainty$event_time
  )
  pre_index <- which(path_event_time < 0L)
  uncertainty_pre_index <- which(uncertainty$event_time < 0L)
  if (length(pre_index) && length(uncertainty_pre_index)) {
    direct_pre <- match(
      path_event_time[pre_index], uncertainty$event_time
    )
    if (all(!is.na(direct_pre))) {
      matched[pre_index] <- direct_pre
    } else {
      ordered_uncertainty_pre <- uncertainty_pre_index[
        order(uncertainty$event_time[uncertainty_pre_index])
      ]
      if (length(ordered_uncertainty_pre) == length(pre_index) - 1L) {
        matched[pre_index[-1L]] <- ordered_uncertainty_pre
      } else if (length(ordered_uncertainty_pre) == length(pre_index)) {
        matched[pre_index] <- ordered_uncertainty_pre
      } else {
        stop(
          "Estimator pre-treatment uncertainty cannot be aligned to the ",
          "normalized target path"
        )
      }
    }
  }
  paths[, `:=`(
    standard_error = NA_real_, confidence_lower = NA_real_,
    confidence_upper = NA_real_, p_value = NA_real_
  )]
  present <- !is.na(matched)
  paths[present, `:=`(
    standard_error = as.numeric(uncertainty[["S.E."]][matched[present]]) * effect_scale,
    confidence_lower = as.numeric(uncertainty[["CI.lower"]][matched[present]]) * effect_scale,
    confidence_upper = as.numeric(uncertainty[["CI.upper"]][matched[present]]) * effect_scale,
    p_value = as.numeric(uncertainty[["p.value"]][matched[present]])
  )]
  paths[]
}

panelmatch_covariate_formula <- function(covariates, lags) {
  # PanelMatch resolves a bare `lag()` inside its formula environment but does
  # NOT export it from its namespace (verified against the locked 3.1.3), so
  # `PanelMatch::lag()` would fail with "not an exported object" at refinement.
  terms <- vapply(covariates, function(variable) {
    paste0("I(lag(", variable, ", c(", paste(lags, collapse = ","), ")))")
  }, character(1L))
  stats::as.formula(paste("~", paste(terms, collapse = " + ")))
}

complete_unit_ids <- function(panel, columns, times) {
  panel[time_id %in% times, .(
    complete = all(stats::complete.cases(.SD))
  ), by = unit_id, .SDcols = columns][complete, unit_id]
}

estimator_output_dir <- function(method, city_key, cohort, outcome, signature,
                                 root = project_root()) {
  safe_signature <- gsub("[^A-Za-z0-9+_-]", "_", signature)
  safe_cohort <- gsub("[^0-9-]", "_", as.character(cohort))
  .staging_dir <- .resolve_path("COMPLETE_ESTIMATORS_STAGING_DIR", root, "outputs", "complete_estimators", "staging")
  path <- file.path(
    .staging_dir, method, city_key,
    safe_cohort, outcome, safe_signature
  )
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}

write_run_manifest <- function(path, values) {
  values <- c(
    list(created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)),
    values
  )
  manifest <- rbindlist(lapply(names(values), function(name) {
    data.table(field = name, value = paste(values[[name]], collapse = ";"))
  }))
  fwrite(manifest, file.path(path, "manifest.csv"), bom = TRUE)
  invisible(manifest)
}
