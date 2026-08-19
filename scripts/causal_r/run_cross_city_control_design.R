suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path("scripts", "causal_r", "grid_control_design_lib.R"))

# Round 4 of the 6-round routing: cross-city matching, invoked on demand by
# the label queue after same-city GSC and MC have both failed.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L) {
  stop("Usage: run_cross_city_control_design.R ORDER [TASK_ROOT]")
}
orders <- as.integer(strsplit(args[[1L]], ",", fixed = TRUE)[[1L]])
if (!length(orders) || any(!is.finite(orders))) stop("All treatment orders must be integers")
root <- project_root()
task_root <- if (length(args) == 2L) args[[2L]] else file.path(
  root, "outputs", "control_design", "tasks"
)

for (order in orders) {
  output <- file.path(task_root, sprintf("%05d", order))
  result <- tryCatch(
    design_cross_city_control(order, root),
    error = function(error) {
      target <- read_treatments(root)[treatment_order == order]
      list(
        status = "error", target = target, active_families = character(),
        failure_reason = conditionMessage(error), attempts = list()
      )
    }
  )
  if (identical(result$status, "matched")) {
    write_grid_control_result(result, output)
  } else {
    # Round-4 failure: never overwrite the same-city durable
    # control_record.csv.  A later phase-1 re-run with reuse_durable would
    # otherwise resurrect this failure into the control queue.  The
    # cross-city attempt is recorded separately for audit.
    record <- control_design_record(result)
    record[, donor_scope := "all_city_standardized"]
    dir.create(output, recursive = TRUE, showWarnings = FALSE)
    fwrite(record, file.path(output, "cross_city_attempt.csv"), bom = TRUE)
  }
  cat("Cross-city control design", order, "status", result$status, "\n")
  flush.console()
}
