suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path("scripts", "causal_r", "fixed_control_label_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
causal_run_id <- Sys.getenv("MIT_CAUSAL_RUN_ID", unset = "")
if (length(args) < 4L || length(args) > 5L) {
  stop(paste(
    "Usage: run_fixed_control_labels.R TREATMENT_ORDER",
    "CONTROL_CITY CONTROL_GRID OUTCOME_FAMILY [OUTPUT_DIR]"
  ))
}
treatment_order <- as.integer(args[[1L]])
control_city_key <- args[[2L]]
control_grid_id <- args[[3L]]
family <- args[[4L]]
root <- project_root()
output <- if (length(args) == 5L) args[[5L]] else file.path(
  root, "outputs", "causal_labels", "fixed_control_staging",
  sprintf("%05d", treatment_order), family
)
dir.create(output, recursive = TRUE, showWarnings = FALSE)

labels <- fixed_control_labels(
  treatment_order, control_city_key, control_grid_id, family, root
)
write_parquet(labels, file.path(output, "causal_response_labels.parquet"), compression = "zstd")
fwrite(labels[, .(
  outcome, event_time, label_available, treated_baseline, control_baseline
)], file.path(output, "label_availability.csv"), bom = TRUE)
write_run_manifest(output, list(
  schema = "fixed_control_labels_v1",
  run_id = causal_run_id,
  estimator = "frozen_matched_change",
  treatment_order = treatment_order,
  outcome_family = family,
  control_city_key = control_city_key,
  control_grid_id = control_grid_id,
  run_mode = "production",
  production_eligible = TRUE
))
cat("Fixed-control labels", treatment_order, family, "available", sum(labels$label_available),
    "of", nrow(labels), "at", output, "\n")
