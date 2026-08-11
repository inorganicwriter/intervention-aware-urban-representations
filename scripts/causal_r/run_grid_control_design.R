suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path("scripts", "causal_r", "grid_control_design_lib.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L) {
  stop("Usage: run_grid_control_design.R TREATMENT_ORDER [OUTPUT_DIR]")
}
treatment_order <- as.integer(args[[1L]])
if (!is.finite(treatment_order)) stop("TREATMENT_ORDER must be an integer")
root <- project_root()
output <- if (length(args) == 2L) args[[2L]] else file.path(
  root, "outputs", "control_design", "tasks", sprintf("%05d", treatment_order)
)
dir.create(output, recursive = TRUE, showWarnings = FALSE)

result <- design_one_grid_control(treatment_order, root)
write_grid_control_result(result, output)

cat("Grid control design", treatment_order, "status", result$status, "at", output, "\n")
