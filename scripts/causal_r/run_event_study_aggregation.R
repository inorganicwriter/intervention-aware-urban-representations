# run_event_study_aggregation.R — aggregate event-study series and run
# parallel-trends validation across all admitted production estimator outputs.
#
# Usage:
#   Rscript run_event_study_aggregation.R [STAGING_ROOT] [OUTPUT_DIR]
#       [STRATUM_ATTRIBUTE] [STRATUM_VALUES]
#       [--orders-file FILE] [--specification-fingerprint REGEX]
#       [--frequency annual|monthly]
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
option_value <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index >= length(args)) return(NULL)
  args[[index + 1L]]
}
option_flags <- which(grepl("^--", args))
positional <- if (length(option_flags)) args[seq_len(min(option_flags) - 1L)] else args
root <- project_root()
staging_root <- if (length(positional) >= 1L && nzchar(positional[[1L]])) positional[[1L]] else file.path(
  root, "outputs", "complete_estimators", "staging"
)
output_dir <- if (length(positional) >= 2L && nzchar(positional[[2L]])) positional[[2L]] else file.path(
  root, "outputs", "event_study"
)
stratum_attribute <- if (length(positional) >= 3L && nzchar(positional[[3L]])) positional[[3L]] else NULL
stratum_values <- NULL
if (length(positional) >= 4L && nzchar(positional[[4L]])) {
  raw_values <- strsplit(positional[[4L]], ",", fixed = TRUE)[[1L]]
  numeric_values <- suppressWarnings(as.numeric(raw_values))
  stratum_values <- ifelse(is.na(numeric_values), raw_values, numeric_values)
}
orders_file <- option_value("--orders-file")
specification_fingerprint <- option_value("--specification-fingerprint")
frequency <- option_value("--frequency")
if (!is.null(frequency) && !frequency %in% c("annual", "monthly")) {
  stop("--frequency must be annual or monthly when supplied")
}
# The aggregation must not silently mix old sensitivity runs with the current
# main specification.  The option is a regex so a caller can still select one
# exact fingerprint when needed.
if (is.null(specification_fingerprint)) {
  specification_fingerprint <- "^main_a6_r1km__a6__w3__price_main$"
}
orders <- NULL
if (!is.null(orders_file)) {
  if (!file.exists(orders_file)) stop("Orders file does not exist: ", orders_file)
  order_frame <- fread(orders_file, colClasses = "character")
  if (!"treatment_order" %in% names(order_frame)) {
    stop("Orders file lacks treatment_order: ", orders_file)
  }
  orders <- as.integer(order_frame$treatment_order)
  if (anyNA(orders) || anyDuplicated(orders) || length(orders) != 400L) {
    stop("Orders file must contain exactly 400 unique numeric treatment orders")
  }
}

if (!dir.exists(staging_root)) {
  stop("Staging root does not exist: ", staging_root)
}

result <- run_event_study_aggregation(
  staging_root, output_dir,
  stratum_attribute = stratum_attribute,
  stratum_values = stratum_values,
  orders = orders,
  specification_fingerprint = specification_fingerprint,
  frequency = frequency
)
cat("Event-study aggregation complete at", output_dir, "\n")
cat("- Admitted production tasks:", uniqueN(result$labels$treatment_order),
    "grids;", nrow(result$labels), "label rows\n")
if (nrow(result$joint_tests) > 0L) {
  cat("- Joint zero-pre-trend tests:\n")
  print(result$joint_tests[, .(frequency, outcome_family, outcome, n_grids,
                               mean_grid_mean, t_statistic, p_value, reject_5pct)])
} else {
  cat("- No pre-period labels yet; run the label queues first.\n")
}
cat("- Figures:", length(result$figures), "written\n")
