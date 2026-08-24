suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(Matching)
})

source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

.grid_control_cache <- new.env(parent = emptyenv())
.viirs_cache_contract <- "complete_44_city_2012_2024_monthly_v1"

grid_control_spec <- function() {
  list(
    schema = "grid_control_design_v3_exact_stable_ties",
    minimum_families = 1L,
    anticipation_months = 6L,
    monthly_blocks = 3L,
    months_per_block = 12L,
    placebo_sample = 200L,
    placebo_quantile = 0.95,
    matching_candidates = 5L,
    scopes = c("same_city", "all_city_standardized"),
    matching = list(
      estimand = "ATT", M = 1L, replace = TRUE, ties = FALSE,
      Weight = 2L, BiasAdjust = FALSE, Var.calc = 0L,
      distance.tolerance = 0
    )
  )
}

family_feature_names <- function(family) {
  variables <- complete_estimator_spec()$families[[family]]
  as.vector(outer(variables, 1:3, function(variable, lag) {
    paste0(variable, "__lag", lag)
  }))
}

# Time-invariant pre-treatment covariates merged into every matching frame.
# Location comes from data/active/curated/location_features/; transit comes
# from the per-opening-month pre-treatment snapshot
# (data/active/causal/transit_snapshots/{city}/{opening_month}.parquet), so
# donors are aligned to the same pre-treatment time point as the target.
static_covariate_names <- function() {
  c(
    "loc_dist_main_km",
    "loc_dist_nearest_subcentre_km",
    "loc_dist_nearest_centre_km",
    "transit_dist_nearest_station_m",
    "transit_stations_500m",
    "transit_stations_800m",
    "transit_stations_1500m",
    "transit_lines_in_1500m",
    "transit_network_closeness"
  )
}

read_city_static_blocks <- function(city_key, target, root = project_root()) {
  cache_key <- paste(
    normalizePath(root, winslash = "/", mustWork = TRUE), city_key,
    target$opening_month[[1L]], "static", sep = "|"
  )
  if (exists(cache_key, envir = .grid_control_cache, inherits = FALSE)) {
    return(get(cache_key, envir = .grid_control_cache, inherits = FALSE))
  }
  result <- data.table()
  location_path <- file.path(
    root, "data", "active", "curated", "location_features",
    paste0(city_key, "_location.parquet")
  )
  snapshot_path <- file.path(
    root, "data", "active", "causal", "transit_snapshots",
    city_key, paste0(target$opening_month[[1L]], ".parquet")
  )
  if (file.exists(location_path) && file.exists(snapshot_path)) {
    location <- as.data.table(read_parquet(location_path, col_select = c(
      "grid_id", "dist_main_km", "dist_nearest_subcentre_km", "dist_nearest_centre_km"
    )))
    setnames(
      location,
      c("dist_main_km", "dist_nearest_subcentre_km", "dist_nearest_centre_km"),
      c("loc_dist_main_km", "loc_dist_nearest_subcentre_km", "loc_dist_nearest_centre_km")
    )
    snapshot <- as.data.table(read_parquet(snapshot_path, col_select = c(
      "grid_id", "dist_nearest_station_m", "stations_500m", "stations_800m",
      "stations_1500m", "lines_in_1500m", "network_closeness"
    )))
    setnames(
      snapshot,
      c(
        "dist_nearest_station_m", "stations_500m", "stations_800m",
        "stations_1500m", "lines_in_1500m", "network_closeness"
      ),
      c(
        "transit_dist_nearest_station_m", "transit_stations_500m", "transit_stations_800m",
        "transit_stations_1500m", "transit_lines_in_1500m", "transit_network_closeness"
      )
    )
    result <- merge(location, snapshot, by = "grid_id", all = TRUE)
    result[, city_key := city_key]
  }
  assign(cache_key, result, envir = .grid_control_cache)
  result[]
}

finite_block_mean <- function(value, minimum_observations) {
  finite <- value[is.finite(value)]
  if (length(finite) < minimum_observations) return(NA_real_)
  mean(finite)
}

monthly_block_features <- function(x, calendar, variables, minimum_observations) {
  if (!nrow(x)) return(data.table())
  month_map <- data.table(month = calendar$pre_months)
  n_months <- nrow(month_map)
  n_blocks <- n_months %/% 12L
  month_map[, lag := n_blocks - ((seq_len(.N) - 1L) %/% 12L)]
  x <- merge(x, month_map, by = "month", all = FALSE)
  if (!nrow(x)) return(data.table())
  aggregated <- x[, lapply(.SD, finite_block_mean,
                           minimum_observations = minimum_observations),
                  by = .(city_key, grid_id, lag), .SDcols = variables]
  long <- melt(
    aggregated, id.vars = c("city_key", "grid_id", "lag"),
    measure.vars = variables, variable.name = "variable", value.name = "value"
  )
  long[, feature := paste0(variable, "__lag", lag)]
  dcast(long, city_key + grid_id ~ feature, value.var = "value")
}

annual_lag_features <- function(x, opening_year, variables) {
  if (!nrow(x)) return(data.table())
  x <- x[year %in% (opening_year - 1:3)]
  if (!nrow(x)) return(data.table())
  x[, lag := opening_year - year]
  long <- melt(
    x, id.vars = c("city_key", "grid_id", "lag"),
    measure.vars = variables, variable.name = "variable", value.name = "value"
  )
  long[, feature := paste0(variable, "__lag", lag)]
  dcast(long, city_key + grid_id ~ feature, value.var = "value")
}

read_city_family_blocks <- function(city_key, target, family,
                                    root = project_root()) {
  cache_key <- paste(
    normalizePath(root, winslash = "/", mustWork = TRUE), city_key,
    target$opening_month[[1L]], family, sep = "|"
  )
  if (exists(cache_key, envir = .grid_control_cache, inherits = FALSE)) {
    return(get(cache_key, envir = .grid_control_cache, inherits = FALSE))
  }
  opening_month <- as.IDate(paste0(target$opening_month, "-01"))
  opening_year <- as.integer(substr(target$opening_month, 1L, 4L))
  calendar <- monthly_event_calendar(
    opening_month, lag = 36L, leads = 1L,
    anticipation_months = grid_control_spec()$anticipation_months
  )
  variables <- complete_estimator_spec()$families[[family]]
  if (family == "housing") {
    x <- read_city_monthly_housing(city_key, root)[month %in% calendar$pre_months]
    result <- monthly_block_features(x, calendar, variables, 1L)
    assign(cache_key, result, envir = .grid_control_cache)
    return(result)
  }
  if (family == "viirs") {
    x <- read_city_monthly_viirs(city_key, calendar$pre_months, root)
    result <- monthly_block_features(x, calendar, variables, 12L)
    assign(cache_key, result, envir = .grid_control_cache)
    return(result)
  }
  x <- read_city_annual_family(city_key, family, root)
  result <- annual_lag_features(x, opening_year, variables)
  assign(cache_key, result, envir = .grid_control_cache)
  result
}

clear_grid_control_cache <- function() {
  rm(list = ls(envir = .grid_control_cache, all.names = TRUE),
     envir = .grid_control_cache)
  invisible(NULL)
}

missing_control_viirs_cache <- function(cities, periods, root = project_root()) {
  expected <- CJ(city_key = cities, month = periods, sorted = TRUE)
  expected[, `:=`(
    parquet = file.path(
      root, "data", "active", "curated", "viirs", "monthly",
      paste0("city_key=", city_key), paste0("year=", format(month, "%Y")),
      paste0("month=", format(month, "%m")), "part.parquet"
    ),
    audit = file.path(
      root, "outputs", "viirs_monthly", "partition_audits", city_key,
      paste0(format(month, "%Y-%m"), ".json")
    )
  )]
  expected[!file.exists(parquet) | !file.exists(audit)]
}

assert_complete_control_viirs_cache <- function(root = project_root()) {
  cache_key <- paste0(
    "viirs_contract|", normalizePath(root, winslash = "/", mustWork = TRUE)
  )
  if (exists(cache_key, envir = .grid_control_cache, inherits = FALSE)) {
    return(invisible(TRUE))
  }
  cities <- sort(unique(all_donors(root)$city_key))
  if (length(cities) != 44L) {
    stop("Monthly VIIRS cache contract expected 44 donor cities, found ", length(cities))
  }
  periods <- seq(as.IDate("2012-01-01"), as.IDate("2024-12-01"), by = "month")
  missing <- missing_control_viirs_cache(cities, periods, root)
  if (nrow(missing)) {
    preview <- paste(
      paste0(head(missing$city_key, 8L), ":", format(head(missing$month, 8L), "%Y-%m")),
      collapse = ", "
    )
    stop(
      "Control-design production requires all 6,864 monthly VIIRS ",
      "Parquet+audit partitions; missing ", nrow(missing), " (", preview, ")."
    )
  }
  assign(cache_key, TRUE, envir = .grid_control_cache)
  invisible(TRUE)
}

merge_family_blocks <- function(parts) {
  parts <- parts[vapply(parts, nrow, integer(1L)) > 0L]
  if (!length(parts)) return(data.table())
  Reduce(function(left, right) {
    merge(left, right, by = c("city_key", "grid_id"), all = TRUE)
  }, parts)
}

read_city_control_features <- function(city_key, target, families,
                                       root = project_root(), strict = TRUE) {
  parts <- lapply(families, function(family) {
    tryCatch(
      read_city_family_blocks(city_key, target, family, root),
      error = function(error) {
        if (strict) stop(error)
        structure(data.table(), family_error = conditionMessage(error))
      }
    )
  })
  names(parts) <- families
  result <- merge_family_blocks(parts)
  static <- read_city_static_blocks(city_key, target, root)
  if (nrow(static)) {
    if (nrow(result) && all(c("city_key", "grid_id") %in% names(result))) {
      result <- merge(result, static, by = c("city_key", "grid_id"), all.x = TRUE)
    } else {
      # No pre-treatment outcome support in this city/time window (e.g. very
      # early openings before VIIRS/POI start): keep only the static
      # covariates.  Family-feature checks upstream then route the target to
      # gsc_pending instead of crashing the merge.
      result <- copy(static)
    }
  }
  attr(result, "unavailable_families") <- families[vapply(parts, nrow, integer(1L)) == 0L]
  result
}

target_active_families <- function(target, target_features) {
  if (!nrow(target_features) || !all(c("city_key", "grid_id") %in% names(target_features))) {
    return(character())
  }
  row <- target_features[
    city_key == target$city_key & grid_id == target$grid_id
  ]
  if (nrow(row) != 1L) return(character())
  families <- names(complete_estimator_spec()$families)
  families[vapply(families, function(family) {
    columns <- family_feature_names(family)
    all(columns %in% names(row)) && all(is.finite(as.numeric(row[, ..columns])))
  }, logical(1L))]
}

all_donors <- function(root = project_root()) {
  as.data.table(read_parquet(
    file.path(
      root, "data", "active", "causal", "formal_matching_inputs",
      "eligible_never_treated_donors.parquet"
    ),
    col_select = c("city_key", "grid_id", "unit_id")
  ))
}

robust_city_standardize <- function(frame, features) {
  center <- frame[role == "donor", c(
    list(city_center_n = .N),
    lapply(.SD, function(value) {
      finite <- value[is.finite(value)]
      if (length(finite)) stats::median(finite) else NA_real_
    })
  ), by = city_key, .SDcols = features]
  scale <- frame[role == "donor", lapply(.SD, function(value) {
      finite <- value[is.finite(value)]
      result <- if (length(finite) > 1L) stats::mad(finite) else NA_real_
      if (!is.finite(result) || result <= sqrt(.Machine$double.eps)) {
        result <- if (length(finite) > 1L) stats::sd(finite) else NA_real_
      }
      if (!is.finite(result) || result <= sqrt(.Machine$double.eps)) 1 else result
    }), by = city_key, .SDcols = features]
  setnames(center, features, paste0(features, "__center"))
  setnames(scale, features, paste0(features, "__scale"))
  result <- merge(frame, center, by = "city_key", all.x = TRUE)
  result <- merge(result, scale, by = "city_key", all.x = TRUE)
  for (feature in features) {
    result[, (feature) := (
      get(feature) - get(paste0(feature, "__center"))
    ) / get(paste0(feature, "__scale"))]
  }
  result[, c(paste0(features, "__center"), paste0(features, "__scale")) := NULL]
  result
}

build_scope_frame <- function(target, active_families, scope,
                               root = project_root()) {
  assert_choice(scope, grid_control_spec()$scopes, "scope")
  donors <- all_donors(root)
  cities <- if (scope == "same_city") target$city_key else sort(unique(donors$city_key))
  city_parts <- lapply(cities, function(city) {
    read_city_control_features(
      city, target, active_families, root,
      strict = identical(scope, "same_city")
    )
  })
  names(city_parts) <- cities
  features <- c(
    unlist(lapply(active_families, family_feature_names), use.names = FALSE),
    static_covariate_names()
  )
  city_parts <- city_parts[vapply(city_parts, function(x) {
    nrow(x) && all(features %in% names(x))
  }, logical(1L))]
  if (!length(city_parts)) stop("No city has the required pre-treatment feature blocks")
  feature_data <- rbindlist(city_parts, use.names = TRUE, fill = TRUE)
  donor_features <- merge(
    donors, feature_data, by = c("city_key", "grid_id"), all = FALSE
  )
  target_features <- read_city_control_features(
    target$city_key, target, active_families, root, strict = TRUE
  )[
    city_key == target$city_key & grid_id == target$grid_id
  ]
  if (nrow(target_features) != 1L) stop("Target pre-treatment feature row is unavailable")
  target_features[, unit_id := paste(city_key, grid_id, sep = "::")]
  target_features[, role := "treated"]
  donor_features[, role := "donor"]
  frame <- rbindlist(list(
    target_features[, c("city_key", "grid_id", "unit_id", "role", features), with = FALSE],
    donor_features[, c("city_key", "grid_id", "unit_id", "role", features), with = FALSE]
  ), use.names = TRUE)
  frame <- frame[complete.cases(frame[, ..features])]
  if (scope == "same_city") frame <- frame[city_key == target$city_key]
  if (sum(frame$role == "treated") != 1L || sum(frame$role == "donor") < 3L) {
    stop("Fewer than three complete donors or missing treated feature row")
  }
  if (scope == "all_city_standardized") {
    frame <- robust_city_standardize(frame, features)
  }
  setorder(frame, role, city_key, grid_id)
  frame[, numeric_unit_id := seq_len(.N)]
  frame[, unit_key := paste(city_key, grid_id, sep = "::")]
  list(
    frame = frame,
    features = features,
    included_cities = sort(unique(frame[role == "donor", city_key])),
    candidate_count = sum(frame$role == "donor")
  )
}

attempt_grid_match <- function(target, active_families, scope,
                               root = project_root()) {
  built <- build_scope_frame(target, active_families, scope, root)
  features <- built$features
  training <- features[grepl("__lag[23]$", features)]
  holdout <- features[grepl("__lag1$", features)]
  static <- static_covariate_names()[static_covariate_names() %in% features]
  # Two-stage control choice: (1) match on outcome-history features (lag2/3)
  # with M candidate controls; (2) refine to the candidate with the best
  # balance on the time-invariant location/transit covariates.  Common
  # support is enforced on outcome-history features only: treated grids are
  # by construction closer to the transit network than the 1km-excluded
  # donors, so closed-range support on the static covariates would make the
  # matched path infeasible.  Static covariates are reported in the SMD
  # diagnostics.
  frame <- copy(built$frame)
  prepared <- frame[, c("numeric_unit_id", "unit_key", "grid_id", "role", features), with = FALSE]
  setnames(prepared, "numeric_unit_id", "unit_id")
  prepared[, Tr := as.integer(role == "treated")]
  matching_spec <- complete_estimator_spec()$abadie_imbens
  matching_spec$ties <- FALSE
  matching_spec$M <- grid_control_spec()$matching_candidates
  matching_spec$distance.tolerance <- grid_control_spec()$matching$distance.tolerance
  selected <- select_preonly_pairs(
    list(frame = prepared, features = features), matching_spec,
    match_features = training, support_features = training
  )
  pair <- static_balance_refine(selected$pairs, selected$frame, static)
  calibration <- calibrate_preonly_placebos(
    selected$frame, selected$active_features, holdout,
    sample_n = grid_control_spec()$placebo_sample,
    quantile_probability = grid_control_spec()$placebo_quantile,
    static_features = static, M = grid_control_spec()$matching_candidates
  )
  quality <- evaluate_preonly_pair_quality(
    pair, selected$frame, selected$active_features,
    holdout, calibration
  )
  control_key <- pair$control_unit_key[[1L]]
  control_source <- frame[unit_key == control_key][1L]
  diagnostics <- pair_preonly_diagnostics(
    pair, selected$frame, c(selected$active_features, static, holdout)
  )
  list(
    accepted = isTRUE(quality$accepted[[1L]]),
    pair = pair,
    control_city_key = control_source$city_key,
    control_grid_id = control_source$grid_id,
    quality = quality,
    placebo = calibration$placebo,
    diagnostics = diagnostics,
    candidate_count = built$candidate_count,
    included_cities = built$included_cities,
    training_features = selected$active_features,
    static_features = static[static %in% names(selected$frame)],
    holdout_features = holdout,
    dropped_features = selected$dropped_features
  )
}

design_one_grid_control <- function(treatment_order, root = project_root()) {
  assert_complete_control_viirs_cache(root)
  requested_order <- as.integer(treatment_order)
  treatments <- read_treatments(root)
  target <- treatments[treatment_order == requested_order]
  if (nrow(target) != 1L) stop("Treatment order is not unique")
  families <- names(complete_estimator_spec()$families)
  target_all <- read_city_control_features(
    target$city_key, target, families, root, strict = FALSE
  )
  active <- target_active_families(target, target_all)
  if (length(active) < grid_control_spec()$minimum_families) {
    return(list(
      status = "gsc_pending", target = target,
      active_families = active,
      failure_reason = paste0("fewer_than_", grid_control_spec()$minimum_families, "_complete_pre_treatment_families"),
      attempts = list()
    ))
  }
  # 6-round routing (DDR): round 1 = same-city matching only.  Cross-city
  # matching is round 4, invoked on demand from the label queue after
  # same-city GSC/MC fail.
  attempt <- tryCatch(
    attempt_grid_match(target, active, "same_city", root),
    error = function(error) structure(
      list(message = conditionMessage(error)), class = "grid_match_error"
    )
  )
  if (inherits(attempt, "grid_match_error")) {
    return(list(
      status = "gsc_pending", target = target, active_families = active,
      failure_reason = paste0("same_city:", attempt$message),
      attempts = list(same_city = attempt)
    ))
  }
  if (!attempt$accepted) {
    return(list(
      status = "gsc_pending", target = target, active_families = active,
      failure_reason = "same_city:preonly_placebo_quality_gate_failed",
      attempts = list(same_city = attempt)
    ))
  }
  list(
    status = "matched", target = target, active_families = active,
    selected_scope = "same_city", selected = attempt,
    failure_reason = NA_character_,
    attempts = list(same_city = attempt)
  )
}

# Round 4 of the 6-round routing: cross-city matching.  Called on demand
# from the label queue after same-city GSC and MC have both failed.
design_cross_city_control <- function(treatment_order, root = project_root()) {
  requested_order <- as.integer(treatment_order)
  treatments <- read_treatments(root)
  target <- treatments[treatment_order == requested_order]
  if (nrow(target) != 1L) stop("Treatment order is not unique")
  families <- names(complete_estimator_spec()$families)
  target_all <- read_city_control_features(
    target$city_key, target, families, root, strict = FALSE
  )
  active <- target_active_families(target, target_all)
  if (length(active) < grid_control_spec()$minimum_families) {
    return(list(
      status = "not_matched", target = target, active_families = active,
      failure_reason = paste0("fewer_than_", grid_control_spec()$minimum_families, "_complete_pre_treatment_families"),
      attempts = list()
    ))
  }
  attempt <- tryCatch(
    attempt_grid_match(target, active, "all_city_standardized", root),
    error = function(error) structure(
      list(message = conditionMessage(error)), class = "grid_match_error"
    )
  )
  if (inherits(attempt, "grid_match_error")) {
    return(list(
      status = "not_matched", target = target, active_families = active,
      failure_reason = paste0("all_city_standardized:", attempt$message),
      attempts = list(cross_city = attempt)
    ))
  }
  if (!attempt$accepted) {
    return(list(
      status = "not_matched", target = target, active_families = active,
      failure_reason = "all_city_standardized:preonly_placebo_quality_gate_failed",
      attempts = list(cross_city = attempt)
    ))
  }
  list(
    status = "matched", target = target, active_families = active,
    selected_scope = "all_city_standardized", selected = attempt,
    failure_reason = NA_character_,
    attempts = list(cross_city = attempt)
  )
}

control_design_record <- function(result) {
  target <- result$target
  base <- data.table(
    schema = grid_control_spec()$schema,
    implementation_version = "r-reference-grid-v3",
    backend = "r_matching",
    viirs_cache_contract = .viirs_cache_contract,
    treatment_order = target$treatment_order,
    city_key = target$city_key,
    grid_id = target$grid_id,
    station_event_id = if ("station_event_id" %in% names(target)) target$station_event_id else NA_character_,
    opening_month = target$opening_month,
    status = result$status,
    active_families = paste(sort(result$active_families), collapse = "+"),
    selected_method = if (result$status == "matched") {
      "Matching::Match_M5_static_refine"
    } else NA_character_,
    donor_scope = if (result$status == "matched") result$selected_scope else NA_character_,
    control_city_key = NA_character_, control_grid_id = NA_character_,
    control_unit_key = NA_character_, candidate_count = NA_integer_,
    candidate_city_count = NA_integer_, training_feature_count = NA_integer_,
    holdout_feature_count = NA_integer_, training_distance = NA_real_,
    holdout_rms_standardized_gap = NA_real_,
    holdout_max_abs_standardized_gap = NA_real_,
    training_distance_threshold = NA_real_, holdout_rms_threshold = NA_real_,
    holdout_max_abs_threshold = NA_real_,
    control_selection_uses_post_outcome = FALSE,
    failure_reason = result$failure_reason
  )
  if (result$status != "matched") return(base)
  selected <- result$selected
  quality <- selected$quality[1L]
  base[, `:=`(
    control_city_key = selected$control_city_key,
    control_grid_id = selected$control_grid_id,
    control_unit_key = paste(selected$control_city_key, selected$control_grid_id, sep = "::"),
    candidate_count = selected$candidate_count,
    candidate_city_count = length(selected$included_cities),
    training_feature_count = length(selected$training_features),
    holdout_feature_count = length(selected$holdout_features),
    training_distance = quality$training_distance,
    holdout_rms_standardized_gap = quality$holdout_rms_standardized_gap,
    holdout_max_abs_standardized_gap = quality$holdout_max_abs_standardized_gap,
    training_distance_threshold = quality$training_distance_threshold,
    holdout_rms_threshold = quality$holdout_rms_threshold,
    holdout_max_abs_threshold = quality$holdout_max_abs_threshold
  )]
  base
}

write_grid_control_result <- function(result, output) {
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  record <- control_design_record(result)
  fwrite(record, file.path(output, "control_record.csv"), bom = TRUE)
  saveRDS(result, file.path(output, "control_design.rds"), compress = "xz")
  if (identical(result$status, "matched")) {
    selected <- result$selected
    fwrite(selected$placebo, file.path(output, "placebo_calibration.csv"), bom = TRUE)
    write_parquet(
      selected$diagnostics$long,
      file.path(output, "feature_balance.parquet"), compression = "zstd"
    )
    fwrite(
      selected$diagnostics$summary,
      file.path(output, "feature_balance_summary.csv"), bom = TRUE
    )
  }
  record
}
