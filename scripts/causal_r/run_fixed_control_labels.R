suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path("scripts", "causal_r", "fixed_control_label_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
causal_run_id <- Sys.getenv("MIT_CAUSAL_RUN_ID", unset = "")
specification_fingerprint <- Sys.getenv("MIT_SPECIFICATION_FINGERPRINT", unset = "")
run_mode <- Sys.getenv("MIT_CAUSAL_RUN_MODE", unset = "production")
if (!run_mode %in% c("production", "preview")) {
  stop("MIT_CAUSAL_RUN_MODE must be production or preview")
}
if (length(args) < 4L || length(args) > 7L) {
  stop(paste(
    "Usage: run_fixed_control_labels.R TREATMENT_ORDER",
    "CONTROL_CITY CONTROL_GRID OUTCOME_FAMILY [OUTPUT_DIR]",
    "[WINDOW=1] [PRICE_MEASURE=median]"
  ))
}
treatment_order <- as.integer(args[[1L]])
control_city_key <- args[[2L]]
control_grid_id <- args[[3L]]
family <- args[[4L]]
root <- project_root()
output <- if (length(args) >= 5L && nzchar(args[[5L]])) args[[5L]] else file.path(
  root, "outputs", "causal_labels", "fixed_control_staging",
  sprintf("%05d", treatment_order), family
)
window <- if (length(args) >= 6L) as.integer(args[[6L]]) else 1L
price_measure <- if (length(args) >= 7L) args[[7L]] else "median"
if (!nzchar(specification_fingerprint)) {
  specification_fingerprint <- paste0(
    "main_a6_r1km__a6__w", window, "__price_", price_measure
  )
}
dir.create(output, recursive = TRUE, showWarnings = FALSE)

labels <- fixed_control_labels(
  treatment_order, control_city_key, control_grid_id, family, root,
  window = window, price_measure = price_measure
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
  specification_fingerprint = specification_fingerprint,
  outcome_family = family,
  control_city_key = control_city_key,
  control_grid_id = control_grid_id,
  run_mode = run_mode,
  production_eligible = identical(run_mode, "production"),
  window = window,
  price_measure = price_measure
))
cat("Fixed-control labels", treatment_order, family, "available", sum(labels$label_available),
    "of", nrow(labels), "at", output, "\n")
