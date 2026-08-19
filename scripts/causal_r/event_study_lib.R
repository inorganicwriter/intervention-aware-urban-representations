# event_study_lib.R — cross-grid event-study aggregation and parallel-trends tests.
#
# The GSC and MC runners write per-task causal_response_labels.parquet files
# whose event_time spans negative (pre-period counterfactual gaps) and positive
# (post-treatment horizons). Those pre-period rows are the event-study
# coefficients: they are the direct, pre-registered evidence for the
# parallel-trends assumption. The label queue publishes only post horizons,
# so this library re-reads the estimator staging outputs and aggregates:
#
#   1. a per-family/outcome event-time series (mean label, SD, SE, 95% CI);
#   2. a grid-level joint test that all pre-period labels are jointly zero
#      (one-sample t-test on per-grid mean pre-period labels, which clusters
#      within grid automatically);
#   3. event-study figures (mean + 95% CI vs event time).
#
# Only outputs whose manifest proves a production run
# (run_mode=production, production_eligible=TRUE) are admitted; smoke and
# canary artifacts are excluded.

suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

project_root <- function() {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

event_study_manifest_fields <- c("field", "value")

read_estimator_manifest_key_value <- function(path) {
  manifest <- fread(path, colClasses = "character")
  if (!setequal(names(manifest), event_study_manifest_fields) ||
      anyDuplicated(manifest$field)) {
    stop("Malformed estimator manifest: ", path)
  }
  setNames(manifest$value, manifest$field)
}

# Collect admitted production label files under a staging root.
collect_production_label_files <- function(staging_root) {
  parquet_files <- list.files(
    staging_root, pattern = "causal_response_labels\\.parquet$",
    recursive = TRUE, full.names = TRUE
  )
  admitted <- character(0)
  for (path in parquet_files) {
    manifest_path <- file.path(dirname(path), "manifest.csv")
    if (!file.exists(manifest_path)) next
    manifest <- tryCatch(
      read_estimator_manifest_key_value(manifest_path),
      error = function(e) NULL
    )
    if (is.null(manifest)) next
    if (!isTRUE(tolower(manifest[["run_mode"]]) == "production")) next
    if (!isTRUE(toupper(manifest[["production_eligible"]]) == "TRUE")) next
    admitted <- c(admitted, path)
  }
  admitted
}

# Read admitted label files into one long data.table with method provenance.
# With ``stratum_attribute`` and ``stratum_values`` the labels are restricted
# to treatment orders whose station attribute (from
# outputs/causal_labels/station_attributes/) is in the given value set; this
# powers per-attribute event studies (transfer / new-line / terminal /
# same-month-opening size).
read_event_study_labels <- function(staging_root,
                                    stratum_attribute = NULL,
                                    stratum_values = NULL) {
  files <- collect_production_label_files(staging_root)
  if (length(files) == 0L) {
    return(data.table(
      treatment_order = integer(0), city_key = character(0),
      outcome_family = character(0), outcome = character(0),
      event_time = integer(0), causal_response_label = numeric(0),
      label_available = logical(0), method = character(0)
    ))
  }
  parts <- lapply(files, function(path) {
    labels <- as.data.table(read_parquet(path))
    required <- c(
      "treatment_order", "outcome_family", "outcome", "event_time",
      "causal_response_label", "label_available"
    )
    missing <- setdiff(required, names(labels))
    if (length(missing) > 0L) {
      warning("Skipping ", path, " (missing columns: ", paste(missing, collapse = ", "), ")")
      return(NULL)
    }
    manifest <- read_estimator_manifest_key_value(file.path(dirname(path), "manifest.csv"))
    estimator <- as.character(manifest[["estimator"]])
    labels[, method := estimator]
    labels[, staging_file := basename(dirname(path))]
    labels[, .(
      treatment_order = as.integer(treatment_order),
      city_key = as.character(city_key),
      outcome_family = as.character(outcome_family),
      outcome = as.character(outcome),
      event_time = as.integer(event_time),
      causal_response_label = as.numeric(causal_response_label),
      label_available = as.logical(label_available),
      method = method
    )]
  })
  parts <- Filter(Negate(is.null), parts)
  if (length(parts) == 0L) {
    return(data.table(
      treatment_order = integer(0), city_key = character(0),
      outcome_family = character(0), outcome = character(0),
      event_time = integer(0), causal_response_label = numeric(0),
      label_available = logical(0), method = character(0)
    ))
  }
  labels <- rbindlist(parts, use.names = TRUE, fill = TRUE)
  if (!is.null(stratum_attribute)) {
    attributes_path <- file.path(
      project_root(), "outputs", "causal_labels", "station_attributes",
      "station_attributes.parquet"
    )
    if (!file.exists(attributes_path)) {
      stop("Station attributes missing for stratification: ", attributes_path)
    }
    attributes <- as.data.table(read_parquet(
      attributes_path,
      col_select = c("treatment_order", stratum_attribute)
    ))
    labels <- merge(labels, attributes, by = "treatment_order", all.x = TRUE)
    if (!is.null(stratum_values)) {
      labels <- labels[get(stratum_attribute) %in% stratum_values]
    }
    labels[, (stratum_attribute) := NULL]
  }
  labels
}

# Aggregate the event-time series per family/outcome over available labels.
# When per-task bootstrap standard errors are available (GSC/MC paths), the
# aggregated SE combines within-task bootstrap variance and between-grid
# variance (random-effects style).  For tasks without SE (matched path), a
# naive sd/sqrt(n) is used as a fallback.
aggregate_event_study_series <- function(labels) {
  observed <- labels[label_available == TRUE & is.finite(causal_response_label)]
  if (nrow(observed) == 0L) {
    return(data.table(
      outcome_family = character(0), outcome = character(0),
      event_time = integer(0), n_units = integer(0),
      mean_label = numeric(0), sd_label = numeric(0), se_label = numeric(0),
      ci_lower = numeric(0), ci_upper = numeric(0), methods = character(0)
    ))
  }
  observed[, event_time := as.integer(event_time)]
  if ("standard_error" %in% names(observed)) {
    observed[!is.finite(standard_error), standard_error := NA_real_]
  } else {
    observed[, standard_error := NA_real_]
  }
  series <- observed[, .(
    n_units = uniqueN(treatment_order),
    mean_label = mean(causal_response_label),
    sd_label = sd(causal_response_label),
    mean_se = mean(standard_error, na.rm = TRUE),
    se_available = sum(is.finite(standard_error)),
    methods = paste(sort(unique(method)), collapse = "|")
  ), by = .(outcome_family, outcome, event_time)]
  series[is.na(sd_label), sd_label := 0]
  # Between-grid variance of grid means
  grid_means <- observed[, .(grid_mean = mean(causal_response_label)),
                         by = .(outcome_family, outcome, event_time, treatment_order)]
  between <- grid_means[, .(n_grids = .N, between_var = var(grid_mean)),
                        by = .(outcome_family, outcome, event_time)]
  series <- merge(series, between, by = c("outcome_family", "outcome", "event_time"), all.x = TRUE)
  series[is.na(between_var), between_var := 0]
  # Within variance from bootstrap SEs: mean SE^2 / n_units
  series[, within_var := ifelse(se_available > 0L,
                                mean_se^2 / n_units, NA_real_)]
  series[, se_label := sqrt(ifelse(
    is.finite(within_var) & within_var > 0,
    within_var + between_var / n_units,
    sd_label^2 / n_units
  ))]
  series[!is.finite(se_label) | se_label <= 0, se_label := sd_label / sqrt(n_units)]
  series[, ci_lower := mean_label - 1.96 * se_label]
  series[, ci_upper := mean_label + 1.96 * se_label]
  series[order(outcome_family, outcome, event_time)]
}

# Joint zero-pre-trend test: one-sample t-test on per-grid mean pre labels.
# Aggregating to grid-level means clusters within-grid correlation without
# requiring a covariance-package dependency.
joint_pretrend_tests <- function(labels, min_pre_event_time = -5L) {
  pre <- labels[
    label_available == TRUE & is.finite(causal_response_label) & event_time < 0L &
      event_time >= min_pre_event_time
  ]
  if (nrow(pre) == 0L) {
    return(data.table(
      outcome_family = character(0), outcome = character(0),
      n_grids = integer(0), n_pre_observations = integer(0),
      mean_grid_mean = numeric(0), sd_grid_mean = numeric(0),
      t_statistic = numeric(0), p_value = numeric(0), reject_5pct = logical(0)
    ))
  }
  grid_level <- pre[, .(
    n_pre_observations = .N,
    grid_mean = mean(causal_response_label)
  ), by = .(outcome_family, outcome, treatment_order)]
  grid_level[, .(
    n_grids = .N,
    n_pre_observations = sum(n_pre_observations),
    mean_grid_mean = mean(grid_mean),
    sd_grid_mean = sd(grid_mean)
  ), by = .(outcome_family, outcome)]
}

# One-sample t-test on grid-level pre means (the main joint test).
finalize_joint_tests <- function(grid_level_summary) {
  if (nrow(grid_level_summary) == 0L) {
    return(grid_level_summary[, .(outcome_family, outcome, n_grids, n_pre_observations,
                                  mean_grid_mean, sd_grid_mean, t_statistic = NA_real_,
                                  p_value = NA_real_, reject_5pct = NA)])
  }
  grid_level_summary[, t_statistic := mean_grid_mean / (sd_grid_mean / sqrt(n_grids))]
  grid_level_summary[
    n_grids < 2L | !is.finite(sd_grid_mean) | sd_grid_mean <= 0,
    t_statistic := NA_real_
  ]
  grid_level_summary[, p_value := 2 * pt(-abs(t_statistic), df = pmax(n_grids - 1L, 1L))]
  grid_level_summary[is.na(t_statistic), p_value := NA_real_]
  grid_level_summary[, reject_5pct := !is.na(p_value) & p_value < 0.05]
  grid_level_summary
}

# City-clustered joint pre-trend test (Abadie et al. 2023, QJE): aggregate
# pre labels to city-level means first, then a one-sample t-test across
# cities.  Metro openings are city-level policies; grids within a city share
# city shocks, so the city-level test is the conservative contrast to the
# grid-level test.
joint_pretrend_tests_city <- function(labels, min_pre_event_time = -5L) {
  pre <- labels[
    label_available == TRUE & is.finite(causal_response_label) & event_time < 0L &
      event_time >= min_pre_event_time
  ]
  if (nrow(pre) == 0L || !"city_key" %in% names(pre)) {
    return(data.table(
      outcome_family = character(0), outcome = character(0),
      n_cities = integer(0), n_pre_observations = integer(0),
      mean_city_mean = numeric(0), sd_city_mean = numeric(0),
      t_statistic = numeric(0), p_value = numeric(0), reject_5pct = logical(0)
    ))
  }
  city_level <- pre[, .(
    n_pre_observations = .N,
    city_mean = mean(causal_response_label)
  ), by = .(outcome_family, outcome, city_key)]
  summary <- city_level[, .(
    n_cities = .N,
    n_pre_observations = sum(n_pre_observations),
    mean_city_mean = mean(city_mean),
    sd_city_mean = sd(city_mean)
  ), by = .(outcome_family, outcome)]
  summary[, t_statistic := mean_city_mean / (sd_city_mean / sqrt(n_cities))]
  summary[
    n_cities < 2L | !is.finite(sd_city_mean) | sd_city_mean <= 0,
    t_statistic := NA_real_
  ]
  summary[, p_value := 2 * pt(-abs(t_statistic), df = pmax(n_cities - 1L, 1L))]
  summary[is.na(t_statistic), p_value := NA_real_]
  summary[, reject_5pct := !is.na(p_value) & p_value < 0.05]
  summary
}

# Render one event-study figure (mean + 95% CI vs event time) per family/outcome.
write_event_study_figures <- function(series, figure_dir) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  if (nrow(series) == 0L) return(character(0))
  written <- character(0)
  for (key in unique(series[, paste0(outcome_family, "__", outcome)])) {
    part <- series[outcome_family == sub("__.*$", "", key) & outcome == sub("^.*__", "", key)]
    if (nrow(part) < 2L) next
    png(
      file.path(figure_dir, paste0("event_study_", key, ".png")),
      width = 900, height = 600, res = 130
    )
    x_range <- range(part$event_time)
    y_max <- max(part$ci_upper, na.rm = TRUE)
    y_min <- min(part$ci_lower, 0, na.rm = TRUE)
    plot(
      part$event_time, part$mean_label, type = "n",
      xlim = x_range, ylim = c(y_min, y_max),
      xlab = "Event time", ylab = "Mean causal response label",
      main = paste0("Event study: ", key, " (n units per period shown)")
    )
    abline(h = 0, lty = 2, col = "grey50")
    abline(v = 0, lty = 2, col = "grey50")
    segments(
      part$event_time, part$ci_lower, part$event_time, part$ci_upper,
      col = "steelblue", lwd = 2
    )
    points(part$event_time, part$mean_label, pch = 19, col = "steelblue")
    text(part$event_time, part$mean_label,
         labels = paste0("n=", part$n_units), pos = 3, cex = 0.6)
    dev.off()
    written <- c(written, file.path(figure_dir, paste0("event_study_", key, ".png")))
  }
  written
}

# Full pipeline: staging root + output directory -> summary files.
run_event_study_aggregation <- function(staging_root, output_dir,
                                        min_pre_event_time = -5L,
                                        stratum_attribute = NULL,
                                        stratum_values = NULL) {
  labels <- read_event_study_labels(
    staging_root,
    stratum_attribute = stratum_attribute,
    stratum_values = stratum_values
  )
  series <- aggregate_event_study_series(labels)
  grid_summary <- joint_pretrend_tests(labels, min_pre_event_time)
  joint_tests <- finalize_joint_tests(grid_summary)
  city_tests <- joint_pretrend_tests_city(labels, min_pre_event_time)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  fwrite(series, file.path(output_dir, "event_study_series.csv"), bom = TRUE)
  fwrite(joint_tests, file.path(output_dir, "event_study_joint_tests.csv"), bom = TRUE)
  fwrite(city_tests, file.path(output_dir, "event_study_joint_tests_city_cluster.csv"), bom = TRUE)

  figure_dir <- file.path(output_dir, "figures")
  figures <- write_event_study_figures(series, figure_dir)

  report <- render_event_study_report(labels, series, joint_tests, figures,
                                      staging_root, output_dir)
  writeLines(report, file.path(output_dir, "event_study_report.md"))

  list(
    labels = labels, series = series, joint_tests = joint_tests,
    figures = figures, report_path = file.path(output_dir, "event_study_report.md")
  )
}

render_event_study_report <- function(labels, series, joint_tests, figures,
                                      staging_root, output_dir) {
  lines <- c("# Event Study & Parallel-Trends Validation", "")
  lines <- c(lines, paste0(
    "- Staging root: `", staging_root, "`"
  ))
  lines <- c(lines, paste0(
    "- Admitted production tasks: ", uniqueN(labels$treatment_order),
    " grids, ", nrow(labels), " label rows (pre+post)"
  ))
  lines <- c(lines, paste0(
    "- Pre-period rows (event_time < 0): ", nrow(labels[event_time < 0L]),
    "; post-period rows: ", nrow(labels[event_time > 0L]), ""
  ))
  lines <- c(lines, "")
  lines <- c(lines, "## Joint zero-pre-trend tests (grid-level means, one-sample t)", "")
  if (nrow(joint_tests) == 0L) {
    lines <- c(lines, "No production estimator outputs found yet; run the label queues first.")
  } else {
    lines <- c(lines, "| family | outcome | n_grids | mean | sd | t | p | reject 5% |")
    lines <- c(lines, "|---|---|---|---|---|---|---|---|")
    for (i in seq_len(nrow(joint_tests))) {
      row <- joint_tests[i]
      lines <- c(lines, sprintf(
        "| %s | %s | %d | %.4g | %.4g | %s | %s | %s |",
        row$outcome_family, row$outcome, row$n_grids, row$mean_grid_mean,
        row$sd_grid_mean,
        ifelse(is.na(row$t_statistic), "—", sprintf("%.3f", row$t_statistic)),
        ifelse(is.na(row$p_value), "—", sprintf("%.4f", row$p_value)),
        ifelse(is.na(row$reject_5pct), "—", ifelse(row$reject_5pct, "yes", "no"))
      ))
    }
  }
  lines <- c(lines, "")
  lines <- c(lines, "## Event-time series", "")
  lines <- c(lines, paste0("- `event_study_series.csv`: mean/SD/SE/95% CI per family x outcome x event time"))
  lines <- c(lines, paste0("- `event_study_joint_tests.csv`: machine-readable joint tests"))
  if (length(figures) > 0L) {
    lines <- c(lines, "- Figures:")
    for (figure in figures) {
      lines <- c(lines, paste0("  - `", figure, "`"))
    }
  }
  lines <- c(lines, "")
  lines <- c(lines, "## Reading guide")
  lines <- c(lines,
    "- Pre-period means near zero are consistent with the parallel-trends assumption;",
    "  rejection of the joint zero-pre-trend test (p < 0.05) is a red flag.",
    "- The test cannot prove parallel trends (Roth 2022); it complements the",
    "  selection-stage holdout/placebo gates and anticipation sensitivity.",
    "- Pre-period support varies across grids (min.T0 differs by family and path);",
    "  the series table reports n per event time, and the joint test uses all",
    "  available pre rows per grid."
  )
  paste(lines, collapse = "\n")
}
