# paths.R — Canonical filesystem paths for R estimator scripts.
#
# Source this file at the top of any R script that needs access to the
# project data/output paths. It reads the JSON manifest produced by
# `urban_intervention.data.paths.export_paths_json()`.
#
# Usage:
#   source(file.path(project_root, "scripts", "causal_r", "paths.R"))
#   paths <- load_project_paths(project_root)
#   paths[["TREATMENT_UNIT_LIST"]]
#   paths[["PANEL_HOUSING_MONTHLY_DIR"]]
#
# After calling load_project_paths(), individual path shortcuts are
# available as top-level variables (e.g. TREATMENT_UNIT_LIST, CAUSAL_DIR).

load_project_paths <- function(root = NULL) {
  if (is.null(root)) {
    root <- Sys.getenv("MIT_PROJECT_ROOT", unset = NA_character_)
    if (is.na(root) || !nzchar(root)) {
      stop("MIT_PROJECT_ROOT is not set and no root argument provided")
    }
  }
  manifest <- file.path(root, ".runtime", "paths.json")
  if (!file.exists(manifest)) {
    stop("paths.json not found at ", manifest,
         " — run: python -c 'from urban_intervention.data.paths import export_paths_json; export_paths_json()'")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("The jsonlite package is required. Install it with: install.packages('jsonlite')")
  }
  paths <- jsonlite::fromJSON(manifest, simplifyVector = FALSE)
  paths <- lapply(paths, function(x) gsub("\\", "/", x, fixed = TRUE))
  manifest_root <- paths[["PROJECT_ROOT"]]
  canonical_root <- normalizePath(root, winslash = "/", mustWork = FALSE)
  canonical_manifest_root <- normalizePath(manifest_root, winslash = "/", mustWork = FALSE)
  if (.Platform$OS.type == "windows") {
    canonical_root <- tolower(canonical_root)
    canonical_manifest_root <- tolower(canonical_manifest_root)
  }
  if (!identical(canonical_root, canonical_manifest_root)) {
    stop(
      "paths.json belongs to a different project root: ", canonical_manifest_root,
      "; current root is ", canonical_root,
      ". Regenerate it with export_paths_json()."
    )
  }
  list2env(paths, envir = .GlobalEnv)
  invisible(paths)
}

# Helper to construct city-specific paths (mirrors Python factory functions)
housing_monthly_panel_path <- function(city) {
  file.path(PANEL_HOUSING_MONTHLY_DIR, paste0(city, ".parquet"))
}

housing_annual_path <- function(city) {
  file.path(HOUSING_ANNUAL_DIR, paste0(city, ".parquet"))
}

poi_annual_path <- function(city) {
  file.path(POI_DIR, paste0(city, "_poi_grid_yearly.parquet"))
}

viirs_annual_path <- function(city) {
  file.path(VIIRS_ANNUAL_DIR, paste0(city, "_viirs_annual.parquet"))
}

population_data_path <- function(city) {
  file.path(POPULATION_DIR, paste0(city, "_pop.parquet"))
}
