suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path("scripts", "causal_r", "formal_matching_lib.R"))
source(file.path("scripts", "causal_r", "paths.R"))

root <- normalizePath(getwd(), winslash = "/")
tryCatch(load_project_paths(root), error = function(e) message(e$message))

.resolve_path <- function(var_name, root, ...) {
  # Global shortcuts (from paths.json) are authoritative only for the project
  # root they were generated from; an explicit different root must win.
  global_value <- if (exists(var_name, envir = .GlobalEnv, inherits = FALSE)) {
    get(var_name, envir = .GlobalEnv)
  } else {
    NULL
  }
  if (!is.null(global_value) && identical(root, normalizePath(getwd(), winslash = "/"))) {
    global_value
  } else {
    file.path(root, ...)
  }
}

causal_dir <- .resolve_path("CAUSAL_DIR", root, "data", "active", "causal")
input_dir <- .resolve_path("FORMAL_MATCHING_DIR", root, "data", "active", "causal", "formal_matching_inputs")
housing_dir <- .resolve_path("HOUSING_ANNUAL_DIR", root, "data", "active", "causal", "formal_matching_inputs", "housing_annual")
dir.create(housing_dir, recursive = TRUE, showWarnings = FALSE)

universe_files <- list.files(
  .resolve_path("GRID_UNIVERSE_DIR", root, "data", "active", "causal", "grid_universe"),
  pattern = "^[a-z]+\\.parquet$",
  full.names = TRUE
)
if (length(universe_files) != 44L) {
  stop("Expected 44 grid-universe files, found ", length(universe_files))
}

donors <- rbindlist(lapply(universe_files, function(path) {
  x <- as.data.table(read_parquet(
    path,
    col_select = c(
      "city_key", "grid_id", "is_nonexperimental_grid",
      "known_station_contamination", "primary_spatial_exclusion_reason"
    )
  ))
  x[
    is_nonexperimental_grid &
      !known_station_contamination &
      primary_spatial_exclusion_reason == "eligible_spatial_donor",
    .(city_key, grid_id)
  ]
}), use.names = TRUE)
donors[, unit_id := paste(city_key, grid_id, sep = "::")]
if (anyDuplicated(donors$unit_id)) stop("Formal donor universe is not unique")
setorder(donors, city_key, grid_id)
write_parquet(
  donors,
  file.path(input_dir, "eligible_never_treated_donors.parquet"),
  compression = "zstd"
)

housing_files <- list.files(
  .resolve_path("PANEL_HOUSING_MONTHLY_DIR", root, "data", "active", "panels", "housing_grid_month"),
  pattern = "^[a-z]+\\.parquet$",
  full.names = TRUE
)
if (length(housing_files) != 44L) {
  stop("Expected 44 housing panel files, found ", length(housing_files))
}

housing_summary <- rbindlist(lapply(housing_files, function(path) {
  city_name <- sub("\\.parquet$", "", basename(path))
  x <- as.data.table(read_parquet(
    path,
    col_select = c(
      "city_key", "grid_id", "observed_month", "log_price_raw_median",
      "n_observations"
    )
  ))
  x <- x[!is.na(log_price_raw_median)]
  x[, year := as.integer(format(observed_month, "%Y"))]
  annual <- x[, .(
    housing_log_price = median(log_price_raw_median, na.rm = TRUE),
    housing_observed_months = uniqueN(observed_month),
    housing_observations = sum(n_observations, na.rm = TRUE)
  ), by = .(city_key, grid_id, year)]
  setorder(annual, grid_id, year)
  write_parquet(
    annual,
    file.path(housing_dir, paste0(city_name, ".parquet")),
    compression = "zstd"
  )
  annual[, .(
    city_key = city_name,
    rows = .N,
    grids = uniqueN(grid_id),
    first_year = min(year),
    last_year = max(year)
  )]
}), use.names = TRUE)
fwrite(
  housing_summary,
  file.path(input_dir, "housing_annual_coverage.csv"),
  bom = TRUE
)

spec <- formal_matching_spec()
writeLines(
  capture.output(dput(spec)),
  file.path(input_dir, "formal_matching_spec.dput"),
  useBytes = TRUE
)

metadata <- data.table(
  schema = spec$schema,
  formal_donors = nrow(donors),
  cities = uniqueN(donors$city_key),
  housing_annual_rows = sum(housing_summary$rows),
  created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
fwrite(metadata, file.path(input_dir, "build_metadata.csv"), bom = TRUE)
print(metadata)
