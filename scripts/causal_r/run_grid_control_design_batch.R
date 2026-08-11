suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path("scripts", "causal_r", "grid_control_design_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L) {
  stop("Usage: run_grid_control_design_batch.R ORDER1,ORDER2,... [TASK_ROOT]")
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
    design_one_grid_control(order, root),
    error = function(error) {
      target <- read_treatments(root)[treatment_order == order]
      list(
        status = "error", target = target, active_families = character(),
        failure_reason = conditionMessage(error), attempts = list()
      )
    }
  )
  write_grid_control_result(result, output)
  cat("Grid control design", order, "status", result$status, "\n")
  flush.console()
}
