# run_event_study_aggregation.R — aggregate event-study series and run
# parallel-trends validation across all admitted production estimator outputs.
#
# Usage:
#   Rscript run_event_study_aggregation.R [STAGING_ROOT] [OUTPUT_DIR]
#       [STRATUM_ATTRIBUTE] [STRATUM_VALUES]
#
# Defaults: staging root = outputs/complete_estimators/staging,
# output dir = outputs/event_study. Only outputs with production manifests
# (run_mode=production, production_eligible=TRUE) are admitted.  An optional
# station-attribute stratification restricts labels to treatment orders whose
# attribute (e.g. is_transfer_at_opening) is in STRATUM_VALUES (comma
# separated, numeric or 0/1).

suppressPackageStartupMessages({
  library(data.table)
})

source(file.path("scripts", "causal_r", "event_study_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
root <- project_root()
staging_root <- if (length(args) >= 1L && nzchar(args[[1L]])) args[[1L]] else file.path(
  root, "outputs", "complete_estimators", "staging"
)
output_dir <- if (length(args) >= 2L && nzchar(args[[2L]])) args[[2L]] else file.path(
  root, "outputs", "event_study"
)
stratum_attribute <- if (length(args) >= 3L && nzchar(args[[3L]])) args[[3L]] else NULL
stratum_values <- NULL
if (length(args) >= 4L && nzchar(args[[4L]])) {
  raw_values <- strsplit(args[[4L]], ",", fixed = TRUE)[[1L]]
  numeric_values <- suppressWarnings(as.numeric(raw_values))
  stratum_values <- ifelse(is.na(numeric_values), raw_values, numeric_values)
}

if (!dir.exists(staging_root)) {
  stop("Staging root does not exist: ", staging_root)
}

result <- run_event_study_aggregation(
  staging_root, output_dir,
  stratum_attribute = stratum_attribute,
  stratum_values = stratum_values
)
cat("Event-study aggregation complete at", output_dir, "\n")
cat("- Admitted production tasks:", uniqueN(result$labels$treatment_order),
    "grids;", nrow(result$labels), "label rows\n")
if (nrow(result$joint_tests) > 0L) {
  cat("- Joint zero-pre-trend tests:\n")
  print(result$joint_tests[, .(outcome_family, outcome, n_grids,
                               mean_grid_mean, t_statistic, p_value, reject_5pct)])
} else {
  cat("- No pre-period labels yet; run the label queues first.\n")
}
cat("- Figures:", length(result$figures), "written\n")
