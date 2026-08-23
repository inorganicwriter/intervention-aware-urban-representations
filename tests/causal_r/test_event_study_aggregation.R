# test_event_study_aggregation.R — contract tests for the event-study
# aggregation and parallel-trends validation.

suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

source(file.path("scripts", "causal_r", "event_study_lib.R"))

write_estimator_manifest <- function(dir_path, fields) {
  manifest <- data.table(field = names(fields), value = as.character(fields))
  fwrite(manifest, file.path(dir_path, "manifest.csv"), bom = TRUE)
}

make_task_labels <- function(order, city, event_times, values, family = "population",
                             outcome = "population_log") {
  data.table(
    treatment_order = order, outcome_family = family, outcome = outcome,
    city_key = city, grid_id = sprintf("g%d", order), period = event_times,
    event_time = event_times, observed = values, counterfactual = values,
    causal_response_label = values, label_available = TRUE,
    method = "test", standard_error = NA_real_
  )
}

write_task <- function(root, tag, labels, production = TRUE) {
  dir_path <- file.path(root, "xu_gsc", "test_city", tag)
  dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  family_name <- unique(as.character(labels$outcome_family))
  stopifnot(length(family_name) == 1L)
  write_parquet(as.data.frame(labels), file.path(dir_path, "causal_response_labels.parquet"),
                compression = "zstd")
  write_estimator_manifest(dir_path, list(
    schema = "test_v1", run_id = paste0("run_", tag),
    estimator = "gsynth", run_mode = if (production) "production" else "smoke",
    production_eligible = if (production) "TRUE" else "FALSE",
    frequency = if (family_name %in% c("housing", "viirs")) "monthly" else "annual",
    specification_fingerprint = "main_a6_r1km__a6__w3__price_main"
  ))
}

set.seed(20260804)

# --- Case 1: flat pre-period labels (parallel trends holds) ---
staging <- tempfile("event_study_staging_")
dir.create(staging, recursive = TRUE)

pre_times <- -3:-1
post_times <- 1:3
for (order in 1001:1002) {
  city <- sprintf("city_%d", order)
  pre_values <- rnorm(length(pre_times), mean = 0, sd = 0.02)
  post_values <- rnorm(length(post_times), mean = 0.2, sd = 0.02)
  write_task(
    staging, sprintf("outcome_t%05d", order),
    make_task_labels(order, city, c(pre_times, post_times),
                     c(pre_values, post_values))
  )
}
# --- Smoke task that must be excluded ---
write_task(staging, "outcome_t9000",
           make_task_labels(9000, "smoke_city", pre_times, rep(0.5, 3)),
           production = FALSE)

# --- Case 2: trending pre-period labels (parallel trends violated) ---
staging2 <- tempfile("event_study_staging_trend_")
dir.create(staging2, recursive = TRUE)
trend_sets <- list(
  c(-0.20, -0.10, 0.00, 0.20, 0.25, 0.30),
  c(-0.21, -0.11, -0.01, 0.22, 0.27, 0.32)
)
for (i in seq_along(trend_sets)) {
  order <- 2000L + i
  write_task(staging2, sprintf("outcome_t%05d", order),
             make_task_labels(order, sprintf("trend_city_%d", i),
                              c(pre_times, post_times), trend_sets[[i]]))
}

output <- tempfile("event_study_out_")
result <- run_event_study_aggregation(staging, output)

# --- Series contract ---
series <- fread(file.path(output, "event_study_series.csv"))
stopifnot(
  "population" %in% series$outcome_family,
  nrow(series) == 6L,
  setequal(series$event_time, c(-3L, -2L, -1L, 1L, 2L, 3L)),
  all(series$n_units == 2L),           # smoke task excluded
  all(series$ci_lower <= series$mean_label),
  all(series$mean_label <= series$ci_upper),
  all(series$se_label >= 0)
)

# --- Joint test: flat pre-period labels must NOT reject ---
joint <- fread(file.path(output, "event_study_joint_tests.csv"))
stopifnot(nrow(joint) == 1L)
stopifnot(
  joint$n_grids == 2L,
  joint$n_pre_observations == 6L,
  abs(joint$mean_grid_mean) < 0.05,
  isTRUE(joint$p_value > 0.05),
  isFALSE(joint$reject_5pct)
)

# --- Case 2 must reject at 5% ---
output2 <- tempfile("event_study_out_trend_")
result2 <- run_event_study_aggregation(staging2, output2)
joint2 <- fread(file.path(output2, "event_study_joint_tests.csv"))
stopifnot(
  nrow(joint2) == 1L,
  isTRUE(joint2$reject_5pct),
  isTRUE(joint2$p_value < 0.05)
)

# --- Smoke-only staging produces no series ---
output3 <- tempfile("event_study_out_smoke_")
staging3 <- tempfile("event_study_staging_smoke_")
dir.create(staging3, recursive = TRUE)
write_task(staging3, "outcome_t8000",
           make_task_labels(8000, "smoke_city", pre_times, rep(0.5, 3)),
           production = FALSE)
result3 <- run_event_study_aggregation(staging3, output3)
stopifnot(nrow(fread(file.path(output3, "event_study_series.csv"))) == 0L,
          nrow(fread(file.path(output3, "event_study_joint_tests.csv"))) == 0L)

# --- Artifacts ---
report <- readLines(file.path(output, "event_study_report.md"), warn = FALSE)
stopifnot(any(grepl("Joint zero-pre-trend tests", report)),
          any(grepl("population", report)),
          file.exists(file.path(output, "figures",
                                "event_study_annual__population__population_log.png")))

cat(paste0(
  "Event-study aggregation tests passed: admission filter, series, CI, ",
  "joint tests (flat vs trending), figures and report.\n"
))
