suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path("scripts", "causal_r", "grid_control_design_lib.R"))
source(file.path("scripts", "causal_r", "fixed_control_label_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L || length(args) > 4L) {
  stop(paste(
    "Usage: export_matching_reference.R TREATMENT_ORDER OUTPUT_DIR",
    "[SCOPE] [MODE=reference|gpu_input]"
  ))
}

order <- as.integer(args[[1L]])
output <- args[[2L]]
scope <- if (length(args) >= 3L) args[[3L]] else "same_city"
mode <- if (length(args) >= 4L) args[[4L]] else "reference"
if (!is.finite(order)) stop("TREATMENT_ORDER must be an integer")
assert_choice(scope, grid_control_spec()$scopes, "scope")
assert_choice(mode, c("reference", "gpu_input"), "mode")

root <- project_root()
target <- read_treatments(root)[treatment_order == order]
if (nrow(target) != 1L) stop("Treatment order is not unique")
families <- names(complete_estimator_spec()$families)
target_all <- read_city_control_features(
  target$city_key, target, families, root, strict = FALSE
)
active <- target_active_families(target, target_all)
if (length(active) < grid_control_spec()$minimum_families) {
  stop("Target has no complete pre-treatment family")
}

built <- build_scope_frame(target, active, scope, root)
features <- built$features
training <- features[grepl("__lag[23]$", features)]
holdout <- features[grepl("__lag1$", features)]
static <- static_covariate_names()[static_covariate_names() %in% features]
frame <- copy(built$frame)
prepared <- frame[, c(
  "numeric_unit_id", "unit_key", "grid_id", "role", features
), with = FALSE]
setnames(prepared, "numeric_unit_id", "unit_id")
prepared[, Tr := as.integer(role == "treated")]

if (mode == "gpu_input") {
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  write_parquet(prepared, file.path(output, "matching_input.parquet"), compression = "zstd")
  fwrite(data.table(
    schema = "causal_gpu_matching_input_exact_stable_ties",
    treatment_order = order,
    scope = scope,
    active_families = paste(active, collapse = "+"),
    training_features = paste(training, collapse = "|"),
    static_features = paste(static[static %in% names(prepared)], collapse = "|"),
    holdout_features = paste(holdout, collapse = "|"),
    matching_candidates = grid_control_spec()$matching_candidates,
    placebo_sample = grid_control_spec()$placebo_sample,
    placebo_quantile = grid_control_spec()$placebo_quantile,
    distance_tolerance = 0,
    tie_policy = "distance_then_original_donor_index",
    candidate_count = built$candidate_count
  ), file.path(output, "metadata.csv"), bom = TRUE)
  cat(
    "Exported matching GPU input", order, scope,
    "with", built$candidate_count, "donors to", output, "\n"
  )
  quit(save = "no", status = 0L)
}

matching_spec <- complete_estimator_spec()$abadie_imbens
matching_spec$ties <- FALSE
matching_spec$M <- grid_control_spec()$matching_candidates
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
  pair, selected$frame, selected$active_features, holdout, calibration
)

dir.create(output, recursive = TRUE, showWarnings = FALSE)
write_parquet(
  selected$frame,
  file.path(output, "matching_input.parquet"),
  compression = "zstd"
)
fwrite(selected$pairs, file.path(output, "reference_candidates.csv"), bom = TRUE)
fwrite(cbind(pair, quality), file.path(output, "reference_selection.csv"), bom = TRUE)
fwrite(calibration$placebo, file.path(output, "reference_placebos.csv"), bom = TRUE)
control_key <- as.character(pair$control_unit_key[[1L]])
control_parts <- strsplit(control_key, "::", fixed = TRUE)[[1L]]
if (length(control_parts) != 2L || any(!nzchar(control_parts))) {
  stop("Selected matching control lacks city::grid identity")
}
reference_labels <- rbindlist(lapply(active, function(family) {
  fixed_control_labels(
    order, control_parts[[1L]], control_parts[[2L]], family, root,
    window = 1L, price_measure = "median"
  )
}), use.names = TRUE, fill = TRUE)
write_parquet(
  reference_labels,
  file.path(output, "reference_labels.parquet"),
  compression = "zstd"
)
fwrite(data.table(
  schema = "causal_gpu_matching_reference_final_labels",
  treatment_order = order,
  scope = scope,
  active_families = paste(active, collapse = "+"),
  training_features = paste(selected$active_features, collapse = "|"),
  static_features = paste(static[static %in% names(selected$frame)], collapse = "|"),
  holdout_features = paste(holdout, collapse = "|"),
  matching_candidates = grid_control_spec()$matching_candidates,
  placebo_sample = grid_control_spec()$placebo_sample,
  placebo_quantile = grid_control_spec()$placebo_quantile,
  distance_tolerance = 0,
  tie_policy = "distance_then_original_donor_index",
  candidate_count = built$candidate_count,
  reference_label_window = 1L,
  reference_label_price_measure = "median",
  reference_label_rows = nrow(reference_labels)
), file.path(output, "metadata.csv"), bom = TRUE)

cat(
  "Exported matching reference", order, scope,
  "with", built$candidate_count, "donors to", output, "\n"
)
