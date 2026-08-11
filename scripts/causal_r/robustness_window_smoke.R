# Robustness smoke: pre-treatment window length (24/36/48 months).
# Verifies the lag parameter flows through monthly_event_calendar.
#
# Usage: Rscript robustness_window_smoke.R LAG_MONTHS

suppressPackageStartupMessages({
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
lag_months <- if (length(args) >= 1L) as.integer(args[[1L]]) else 36L
root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
source(file.path(root, "scripts", "causal_r", "complete_estimators_lib.R"))

opening <- as.IDate("2019-06-01")
calendar <- monthly_event_calendar(opening, lag = lag_months, leads = 1L,
                                   anticipation_months = 6L)
n_pre <- length(calendar$pre_months)
cat(sprintf("lag=%d pre months=%d post months=%d\n",
            lag_months, n_pre, length(calendar$post_months)))
stopifnot(n_pre == lag_months)
cat("OK window smoke\n")
