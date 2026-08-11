# run_event_study_aggregation.R — aggregate event-study series and run
# parallel-trends validation across all admitted production estimator outputs.
#
# Usage:
#   Rscript run_event_study_aggregation.R [STAGING_ROOT] [OUTPUT_DIR]
#
# Defaults: staging root = outputs/complete_estimators/staging,
# output dir = outputs/event_study. Only outputs with production manifests
# (run_mode=production, production_eligible=TRUE) are admitted.

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

if (!dir.exists(staging_root)) {
  stop("Staging root does not exist: ", staging_root)
}

result <- run_event_study_aggregation(staging_root, output_dir)
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
