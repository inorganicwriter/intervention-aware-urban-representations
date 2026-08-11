suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

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
treatment_path <- .resolve_path("TREATMENT_UNIT_LIST", root, "data", "active", "causal", "treatment_unit_list.parquet")

if (!file.exists(treatment_path)) {
  stop("Frozen treatment list not found: ", treatment_path)
}

treatments <- as.data.table(read_parquet(treatment_path))
required <- c(
  "treatment_order", "city_key", "grid_id", "station_event_id", "opening_month"
)
if (!all(required %in% names(treatments))) {
  stop("Frozen treatment list is missing required columns")
}
if (nrow(treatments) != 5048L) {
  stop("Expected 5,048 frozen treatment units, found ", nrow(treatments))
}
if (anyDuplicated(treatments$treatment_order) ||
    anyDuplicated(paste(treatments$city_key, treatments$grid_id, sep = "::"))) {
  stop("Frozen treatment list is not unique")
}

queue <- treatments[, ..required]
queue[, `:=`(
  status = "pending",
  selected_method = NA_character_,
  selected_control_grid_id = NA_character_,
  failure_reason = NA_character_
)]
setorder(queue, treatment_order)

family_queue <- queue[, ..required][
  , .(outcome_family = c("housing", "poi", "viirs", "population")),
  by = required
]
family_queue[, `:=`(
  status = "pending",
  selected_method = NA_character_,
  failure_reason = NA_character_
)]
setorder(family_queue, treatment_order, outcome_family)

control_queue <- queue[, ..required]
control_queue[, `:=`(
  status = "pending",
  active_families = NA_character_, selected_method = NA_character_,
  donor_scope = NA_character_, control_city_key = NA_character_,
  control_grid_id = NA_character_, control_unit_key = NA_character_,
  candidate_count = NA_integer_, candidate_city_count = NA_integer_,
  training_feature_count = NA_integer_, holdout_feature_count = NA_integer_,
  training_distance = NA_real_, holdout_rms_standardized_gap = NA_real_,
  holdout_max_abs_standardized_gap = NA_real_,
  training_distance_threshold = NA_real_, holdout_rms_threshold = NA_real_,
  holdout_max_abs_threshold = NA_real_,
  control_selection_uses_post_outcome = FALSE,
  failure_reason = NA_character_
)]
setorder(control_queue, treatment_order)

fwrite(
  queue,
  file.path(causal_dir, "counterfactual_work_queue.csv"),
  bom = TRUE
)
fwrite(
  family_queue,
  file.path(causal_dir, "outcome_family_work_queue.csv"),
  bom = TRUE
)
fwrite(
  control_queue,
  file.path(causal_dir, "control_design_queue.csv"),
  bom = TRUE
)

cat(sprintf(
  paste(
    "Reset %d treatment units, %d frozen-control rows,",
    "and %d treatment-family rows to pending.\n"
  ),
  nrow(queue), nrow(control_queue), nrow(family_queue)
))
