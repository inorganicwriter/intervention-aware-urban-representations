suppressPackageStartupMessages({
  library(data.table)
  library(gsynth)
})

source(file.path("scripts", "causal_r", "formal_matching_lib.R"))

set.seed(20260722)
units <- 1:30
times <- 1:12
panel <- CJ(unit_id = units, time_id = times)
unit_loading <- rnorm(length(units))
time_factor <- sin(times / 2)
panel[, outcome :=
  unit_loading[unit_id] * time_factor[time_id] +
    0.05 * unit_id + 0.03 * time_id + rnorm(.N, sd = 0.02)]
panel[, D := as.integer(unit_id == 1L & time_id >= 9L)]
panel[D == 1L, outcome := outcome + 0.5]

fit <- gsynth(
  outcome ~ D,
  data = panel,
  index = c("unit_id", "time_id"),
  force = "two-way",
  CV = TRUE,
  r = 0:3,
  criterion = "mspe",
  se = TRUE,
  nboots = 20,
  inference = "parametric",
  parallel = FALSE,
  min.T0 = 5,
  normalize = TRUE,
  seed = 20260722
)

stopifnot(inherits(fit, "gsynth"))
stopifnot(is.finite(fit$r.cv))
stopifnot(all(is.finite(fit$Y.ct[, match("1", as.character(fit$id))])))
stopifnot(formal_matching_spec()$gsc$bootstrap_replications == 200L)

cat("Xu (2017) gsynth smoke test passed.\n")
