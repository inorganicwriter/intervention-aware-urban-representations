suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path("scripts", "causal_r", "paths.R"))
root <- normalizePath(getwd(), winslash = "/")
tryCatch(load_project_paths(root), error = function(e) message("Paths manifest unavailable: ", e$message))
source(file.path("scripts", "causal_r", "formal_matching_lib.R"))
source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

if (exists("TREATMENT_UNIT_LIST", envir = .GlobalEnv)) {
  treatments <- as.data.table(read_parquet(TREATMENT_UNIT_LIST))
} else {
  treatments <- as.data.table(read_parquet(file.path(root, "data", "active", "causal", "treatment_unit_list.parquet")))
}
treatments[, opening_year := as.integer(substr(opening_month, 1L, 4L))]

family_paths <- function(family) {
  switch(
    family,
    housing = list.files(
      if (exists("HOUSING_ANNUAL_DIR")) HOUSING_ANNUAL_DIR
      else file.path(root, "data", "active", "causal", "formal_matching_inputs", "housing_annual"),
      "^[a-z]+\\.parquet$", full.names = TRUE),
    poi = list.files(
      if (exists("POI_DIR")) POI_DIR else file.path(root, "data", "active", "curated", "poi"),
      "_poi_grid_yearly\\.parquet$", full.names = TRUE),
    viirs = list.files(
      if (exists("VIIRS_ANNUAL_DIR")) VIIRS_ANNUAL_DIR
      else file.path(root, "data", "active", "curated", "viirs_annual_aggregated"),
      "_viirs_annual\\.parquet$", full.names = TRUE),
    population = list.files(
      if (exists("POPULATION_DIR")) POPULATION_DIR else file.path(root, "data", "active", "curated", "population"),
      "_pop\\.parquet$", full.names = TRUE)
  )
}

read_city_family <- function(path, family) {
  if (family == "housing") {
    return(as.data.table(read_parquet(path, col_select = c(
      "city_key", "grid_id", "year", "housing_log_price"
    ))))
  }
  if (family == "poi") {
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
    x <- as.data.table(read_parquet(path, col_select = c(
      "city_key", "grid_id", "year", "avg_rad", "avg_rad_median"
    )))
    x[, viirs_avg_asinh := asinh(avg_rad)]
    return(x[, .(city_key, grid_id, year, viirs_avg_asinh)])
  }
  x <- as.data.table(read_parquet(path, col_select = c(
    "city", "grid_id", "year", "pop_count"
  )))
  setnames(x, "city", "city_key")
  x[, population_log := log1p(pmax(pop_count, 0))]
  x[, .(population_log = mean(population_log, na.rm = TRUE)),
    by = .(city_key, grid_id, year)]
}

audit_family <- function(family) {
  variables <- formal_matching_spec()$families[[family]]
  spec <- complete_estimator_spec()
  rows <- rbindlist(lapply(family_paths(family), function(path) {
    city_name <- sub("(_poi_grid_yearly|_viirs_annual|_pop)?\\.parquet$", "", basename(path))
    target_city <- treatments[city_key == city_name]
    if (nrow(target_city) == 0L) return(NULL)
    x <- read_city_family(path, family)
    x <- x[grid_id %in% target_city$grid_id]
    merge(
      target_city[, .(treatment_order, city_key, grid_id, opening_year)],
      x,
      by = c("city_key", "grid_id"),
      all.x = TRUE,
      allow.cartesian = TRUE
    )
  }), fill = TRUE)
  rows[, lag := opening_year - year]
  post_rows <- rows[lag %in% seq_len(spec$annual$lag)]
  pre_rows <- rows[year < opening_year - spec$timing$annual_anticipation_years]
  post_result <- post_rows[, .(
    complete_years = uniqueN(year[complete.cases(.SD)]),
    observed_years = uniqueN(year[!is.na(year)])
  ), by = treatment_order, .SDcols = variables]
  pre_result <- pre_rows[, .(
    gsc_pre_complete_years = uniqueN(year[complete.cases(.SD)]),
    gsc_pre_observed_years = uniqueN(year[!is.na(year)])
  ), by = treatment_order, .SDcols = variables]
  result <- merge(post_result, pre_result, by = "treatment_order", all = TRUE)
  for (column in c(
    "complete_years", "observed_years", "gsc_pre_complete_years",
    "gsc_pre_observed_years"
  )) {
    result[is.na(get(column)), (column) := 0L]
  }
  result[, complete := complete_years == spec$annual$lag]
  # GSC's annual estimator requires min.T0 clean pre-treatment years and the
  # full annual post-treatment horizon.  These are distinct from the
  # three-year matching admission criterion.
  result[, gsc_ready := (
    complete & gsc_pre_complete_years >= spec$xu_gsc$min.T0
  )]
  setnames(
    result,
    c(
      "complete_years", "observed_years", "gsc_pre_complete_years",
      "gsc_pre_observed_years", "complete", "gsc_ready"
    ),
    paste0(
      family,
      c(
        "_complete_years", "_observed_years", "_gsc_pre_complete_years",
        "_gsc_pre_observed_years", "_complete", "_gsc_ready"
      )
    )
  )
  result
}

families <- names(formal_matching_spec()$families)
minimum_complete_families <- formal_matching_spec()$minimum_complete_families
audit <- treatments[, .(treatment_order, city_key, grid_id, opening_month)]
for (family in families) {
  audit <- merge(audit, audit_family(family), by = "treatment_order", all.x = TRUE)
}
complete_columns <- paste0(families, "_complete")
for (column in complete_columns) {
  audit[is.na(get(column)), (column) := FALSE]
}
audit[, complete_families := rowSums(.SD), .SDcols = complete_columns]
gsc_ready_columns <- paste0(families, "_gsc_ready")
for (column in gsc_ready_columns) {
  audit[is.na(get(column)), (column) := FALSE]
}
audit[, gsc_ready_families := rowSums(.SD), .SDcols = gsc_ready_columns]
setorder(audit, treatment_order)
write_parquet(
  audit,
  file.path(
    if (exists("FORMAL_MATCHING_DIR")) FORMAL_MATCHING_DIR
    else file.path(root, "data", "active", "causal", "formal_matching_inputs"),
    "formal_target_support.parquet"
  ),
  compression = "zstd"
)
fwrite(
  audit[, .N, by = complete_families][order(complete_families)],
  if (exists("FORMAL_MATCHING_DIR")) file.path(FORMAL_MATCHING_DIR, "formal_target_support_summary.csv")
  else file.path(root, "data", "active", "causal", "formal_matching_inputs", "formal_target_support_summary.csv"),
  bom = TRUE
)
print(audit[, .N, by = complete_families][order(complete_families)])
cat(sprintf(
  "First target meeting the %d-family matching boundary:\n",
  minimum_complete_families
))
print(audit[complete_families >= minimum_complete_families][1L])
