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

empty_event_study_labels <- function() {
  data.table(
    treatment_order = integer(0), city_key = character(0),
    outcome_family = character(0), outcome = character(0),
    frequency = character(0), event_time = integer(0),
    causal_response_label = numeric(0),
    label_available = logical(0), method = character(0),
    donor_scope = character(0), run_id = character(0),
    specification_fingerprint = character(0), price_measure = character(0),
    observation_window = character(0), standard_error = numeric(0)
  )
}

read_estimator_manifest_key_value <- function(path) {
  manifest <- fread(path, colClasses = "character")
  if (!setequal(names(manifest), event_study_manifest_fields) ||
      anyDuplicated(manifest$field)) {
    stop("Malformed estimator manifest: ", path)
  }
  setNames(manifest$value, manifest$field)
}

estimator_manifest_value <- function(manifest, field) {
  if (!field %in% names(manifest)) return(NA_character_)
  value <- manifest[[field]]
  if (!length(value) || is.na(value[[1L]])) return(NA_character_)
  as.character(value[[1L]])
}

# Collect admitted production label files under a staging root.
collect_production_label_files <- function(staging_root,
                                           specification_fingerprint = NULL,
                                           frequency = NULL) {
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
    manifest_run_mode <- estimator_manifest_value(manifest, "run_mode")
    manifest_production <- estimator_manifest_value(manifest, "production_eligible")
    manifest_frequency <- estimator_manifest_value(manifest, "frequency")
    if (!identical(tolower(manifest_run_mode), "production")) next
    if (!identical(toupper(manifest_production), "TRUE")) next
    if (is.na(manifest_frequency) ||
        !manifest_frequency %in% c("annual", "monthly")) next
    if (!is.null(frequency) &&
        !identical(manifest_frequency, as.character(frequency))) next
    if (!is.null(specification_fingerprint)) {
      manifest_spec <- estimator_manifest_value(manifest, "specification_fingerprint")
      if (is.na(manifest_spec) ||
          !grepl(specification_fingerprint, manifest_spec)) next
    }
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
                                    stratum_values = NULL,
                                    orders = NULL,
                                    specification_fingerprint = NULL,
                                    frequency = NULL) {
  files <- collect_production_label_files(
    staging_root,
    specification_fingerprint = specification_fingerprint,
    frequency = frequency
  )
  if (length(files) == 0L) {
    return(empty_event_study_labels())
  }
  parts <- lapply(files, function(path) {
    labels <- as.data.table(read_parquet(path))
    required <- c(
      "treatment_order", "city_key", "outcome_family", "outcome", "event_time",
      "causal_response_label", "label_available"
    )
    missing <- setdiff(required, names(labels))
    if (length(missing) > 0L) {
      warning("Skipping ", path, " (missing columns: ", paste(missing, collapse = ", "), ")")
      return(NULL)
    }
    parsed_event_time <- suppressWarnings(as.integer(labels$event_time))
    if (anyNA(parsed_event_time)) {
      warning("Skipping ", path, " (event_time contains non-integer values)")
      return(NULL)
    }
    if (!is.null(orders)) {
      labels <- labels[treatment_order %in% orders]
      if (nrow(labels) == 0L) return(NULL)
    }
    manifest <- read_estimator_manifest_key_value(file.path(dirname(path), "manifest.csv"))
    manifest_value <- function(name) estimator_manifest_value(manifest, name)
    labels[, method := manifest_value("estimator")]
    labels[, donor_scope := manifest_value("donor_scope")]
    labels[, run_id := manifest_value("run_id")]
    labels[, specification_fingerprint := manifest_value("specification_fingerprint")]
    labels[, price_measure := manifest_value("price_measure")]
    labels[, observation_window := manifest_value("observation_window")]
    labels[, staging_file := basename(dirname(path))]
    if (!"standard_error" %in% names(labels)) labels[, standard_error := NA_real_]
    labels[, .(
      treatment_order = as.integer(treatment_order),
      city_key = as.character(city_key),
      outcome_family = as.character(outcome_family),
      outcome = as.character(outcome),
      frequency = as.character(manifest_value("frequency")),
      event_time = as.integer(event_time),
      causal_response_label = as.numeric(causal_response_label),
      label_available = as.logical(label_available),
      method = method,
      donor_scope = donor_scope,
      run_id = run_id,
      specification_fingerprint = specification_fingerprint,
      price_measure = price_measure,
      observation_window = observation_window,
      standard_error = as.numeric(standard_error)
    )]
  })
  parts <- Filter(Negate(is.null), parts)
  if (length(parts) == 0L) {
    return(empty_event_study_labels())
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
# When per-task estimator standard errors are available (GSC/MC paths), the
# aggregated SE combines within-task estimator variance and between-grid
# variance (random-effects style).  For tasks without SE (matched path), a
# naive sd/sqrt(n) is used as a fallback.
aggregate_event_study_series <- function(labels) {
  observed <- labels[label_available == TRUE & is.finite(causal_response_label)]
  if (nrow(observed) == 0L) {
    return(data.table(
      frequency = character(0), outcome_family = character(0), outcome = character(0),
      event_time = integer(0), n_units = integer(0),
      mean_label = numeric(0), sd_label = numeric(0), se_label = numeric(0),
      ci_lower = numeric(0), ci_upper = numeric(0), methods = character(0),
      donor_scopes = character(0), specifications = character(0),
      mean_se2 = numeric(0), se_available = integer(0), n_grids = integer(0),
      between_var = numeric(0), within_var = numeric(0)
    ))
  }
  observed[, event_time := as.integer(event_time)]
  duplicate_key <- c("treatment_order", "frequency", "outcome_family",
                     "outcome", "event_time")
  duplicate_rows <- observed[, .N, by = duplicate_key][N > 1L]
  if (nrow(duplicate_rows) > 0L) {
    examples <- paste(
      head(duplicate_rows, 5L)[["treatment_order"]],
      head(duplicate_rows, 5L)[["frequency"]],
      head(duplicate_rows, 5L)[["outcome_family"]],
      head(duplicate_rows, 5L)[["outcome"]],
      head(duplicate_rows, 5L)[["event_time"]],
      sep = "/"
    )
    stop(
      "Duplicate admitted event-study rows for the same grid/frequency/family/outcome/event_time. ",
      "Filter to one final estimator specification before plotting. Examples: ",
      paste(examples, collapse = ", ")
    )
  }
  if ("standard_error" %in% names(observed)) {
    observed[!is.finite(standard_error), standard_error := NA_real_]
  } else {
    observed[, standard_error := NA_real_]
  }
  series <- observed[, .(
    n_units = .N,
    mean_label = mean(causal_response_label),
    sd_label = sd(causal_response_label),
    mean_se2 = if (any(is.finite(standard_error))) {
      mean(standard_error[is.finite(standard_error)]^2)
    } else NA_real_,
    se_available = sum(is.finite(standard_error)),
    methods = paste(sort(unique(method)), collapse = "|"),
    donor_scopes = paste(sort(unique(na.omit(donor_scope))), collapse = "|"),
    specifications = paste(sort(unique(na.omit(specification_fingerprint))), collapse = "|")
  ), by = .(frequency, outcome_family, outcome, event_time)]
  series[is.na(sd_label), sd_label := 0]
  # Between-grid variance of grid means
  grid_means <- observed[, .(grid_mean = mean(causal_response_label)),
                         by = .(frequency, outcome_family, outcome, event_time, treatment_order)]
  between <- grid_means[, .(n_grids = .N, between_var = var(grid_mean)),
                        by = .(frequency, outcome_family, outcome, event_time)]
  series <- merge(
    series, between,
    by = c("frequency", "outcome_family", "outcome", "event_time"),
    all.x = TRUE
  )
  series[is.na(between_var), between_var := 0]
  # Within variance from bootstrap SEs: mean(SE^2) / n_units.  Squaring the
  # mean SE would understate variance when task-level SEs are heterogeneous.
  series[, within_var := ifelse(se_available > 0L,
                                mean_se2 / n_units, NA_real_)]
  series[, se_label := sqrt(ifelse(
    is.finite(within_var) & within_var > 0,
    within_var + between_var / n_units,
    sd_label^2 / n_units
  ))]
  series[!is.finite(se_label) | se_label <= 0, se_label := sd_label / sqrt(n_units)]
  series[, ci_lower := mean_label - 1.96 * se_label]
  series[, ci_upper := mean_label + 1.96 * se_label]
  series[order(frequency, outcome_family, outcome, event_time)]
}

# Select the latest clean pre-periods for the joint test.  When event_time is
# a true calendar offset, the latest five clean periods in the main monthly
# specification are -11:-7, not -5:-1 (which are anticipation periods).
select_pretrend_rows <- function(labels, min_pre_event_time = NULL,
                                  latest_n = 5L) {
  pre <- labels[
    label_available == TRUE & is.finite(causal_response_label) & event_time < 0L
  ]
  if (is.null(min_pre_event_time)) {
    pre[, .pre_rank := frank(-event_time, ties.method = "dense"),
        by = .(frequency, outcome_family, outcome, treatment_order)]
    pre <- pre[.pre_rank <= as.integer(latest_n)]
    pre[, .pre_rank := NULL]
  } else {
    min_pre_event_time <- as.integer(min_pre_event_time)
    if (length(min_pre_event_time) != 1L || is.na(min_pre_event_time) ||
        min_pre_event_time >= 0L) {
      stop("min_pre_event_time must be a finite negative integer")
    }
    pre <- pre[event_time >= min_pre_event_time]
  }
  pre
}

# Joint zero-pre-trend test: one-sample t-test on per-grid mean pre labels.
# Aggregating to grid-level means clusters within-grid correlation without
# requiring a covariance-package dependency.
joint_pretrend_tests <- function(labels, min_pre_event_time = NULL) {
  pre <- select_pretrend_rows(labels, min_pre_event_time)
  if (nrow(pre) == 0L) {
    return(data.table(
      frequency = character(0), outcome_family = character(0), outcome = character(0),
      n_grids = integer(0), n_pre_observations = integer(0),
      mean_grid_mean = numeric(0), sd_grid_mean = numeric(0),
      t_statistic = numeric(0), p_value = numeric(0), reject_5pct = logical(0)
    ))
  }
  grid_level <- pre[, .(
    n_pre_observations = .N,
    grid_mean = mean(causal_response_label)
  ), by = .(frequency, outcome_family, outcome, treatment_order)]
  grid_level[, .(
    n_grids = .N,
    n_pre_observations = sum(n_pre_observations),
    mean_grid_mean = mean(grid_mean),
    sd_grid_mean = sd(grid_mean)
  ), by = .(frequency, outcome_family, outcome)]
}

# One-sample t-test on grid-level pre means (the main joint test).
finalize_joint_tests <- function(grid_level_summary) {
  if (nrow(grid_level_summary) == 0L) {
    return(grid_level_summary[, .(frequency, outcome_family, outcome, n_grids, n_pre_observations,
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
joint_pretrend_tests_city <- function(labels, min_pre_event_time = NULL) {
  pre <- select_pretrend_rows(labels, min_pre_event_time)
  if (nrow(pre) == 0L || !"city_key" %in% names(pre)) {
    return(data.table(
      frequency = character(0), outcome_family = character(0), outcome = character(0),
      n_cities = integer(0), n_pre_observations = integer(0),
      mean_city_mean = numeric(0), sd_city_mean = numeric(0),
      t_statistic = numeric(0), p_value = numeric(0), reject_5pct = logical(0)
    ))
  }
  city_level <- pre[, .(
    n_pre_observations = .N,
    city_mean = mean(causal_response_label)
  ), by = .(frequency, outcome_family, outcome, city_key)]
  summary <- city_level[, .(
    n_cities = .N,
    n_pre_observations = sum(n_pre_observations),
    mean_city_mean = mean(city_mean),
    sd_city_mean = sd(city_mean)
  ), by = .(frequency, outcome_family, outcome)]
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

# Render standard event-study figures: pre/post shading, event-time zero line,
# point estimates with 95% CI, and a compact pre-trend test annotation.  The
# labels are direct response paths rather than regression coefficients, so all
# available pre-periods remain visible; t=0 is the treatment boundary, not an
# omitted regression base period.
draw_event_study_plot <- function(part, title, test_row = NULL, show_legend = TRUE) {
  part <- part[is.finite(event_time)][order(event_time)]
  if (!nrow(part)) return(invisible(FALSE))
  finite_values <- c(part$ci_lower, part$ci_upper, part$mean_label, 0)
  finite_values <- finite_values[is.finite(finite_values)]
  if (!length(finite_values)) return(invisible(FALSE))
  x_points <- sort(unique(as.integer(part$event_time)))
  x_range <- range(c(x_points, 0))
  if (diff(x_range) == 0) x_range <- x_range + c(-1, 1)
  y_range <- range(finite_values)
  y_pad <- max(diff(y_range) * 0.10, 0.05)
  ylim <- y_range + c(-y_pad, y_pad)
  if (ylim[[1L]] > 0) ylim[[1L]] <- 0
  if (ylim[[2L]] < 0) ylim[[2L]] <- 0
  p_note <- if (!is.null(test_row) && nrow(test_row) &&
                is.finite(test_row$p_value[[1L]])) {
    n_note <- if ("n_grids" %in% names(test_row)) {
      sprintf("; n = %d", test_row$n_grids[[1L]])
    } else ""
    sprintf("Pre-trend joint p = %.3f%s", test_row$p_value[[1L]], n_note)
  } else "Pre-trend joint p = NA"

  plot(
    part$event_time, part$mean_label, type = "n",
    xlim = x_range + c(-0.5, 0.5), ylim = ylim,
    xaxt = "n", xlab = "Event time", ylab = "Causal response label",
    main = title, sub = p_note
  )
  usr <- par("usr")
  rect(usr[[1L]], usr[[3L]], -0.5, usr[[4L]],
       col = "#F2F2F2", border = NA)
  rect(-0.5, usr[[3L]], usr[[2L]], usr[[4L]],
       col = "#EAF2F8", border = NA)
  abline(h = 0, lty = 2, lwd = 1, col = "#666666")
  abline(v = 0, lty = 1, lwd = 1.2, col = "#222222")
  x_ticks <- if (length(x_points) <= 13L) {
    x_points
  } else {
    step <- if (diff(x_range) > 48L) 12L else if (diff(x_range) > 24L) 6L else 3L
    sort(unique(c(seq(x_range[[1L]], x_range[[2L]], by = step), 0L)))
  }
  axis(1, at = x_ticks)
  axis(2)
  box()

  pre <- part[event_time < 0L]
  post <- part[event_time >= 0L]
  pre_col <- "#4D4D4D"
  post_col <- "#1F78B4"
  for (piece in list(pre, post)) {
    if (!nrow(piece)) next
    col <- if (piece$event_time[[1L]] < 0L) pre_col else post_col
    ci_ok <- is.finite(piece$ci_lower) & is.finite(piece$ci_upper)
    if (any(ci_ok)) {
      segments(
        piece$event_time[ci_ok], piece$ci_lower[ci_ok],
        piece$event_time[ci_ok], piece$ci_upper[ci_ok],
        col = col, lwd = 1.4
      )
      points(piece$event_time[ci_ok], piece$ci_lower[ci_ok], pch = "_", col = col)
      points(piece$event_time[ci_ok], piece$ci_upper[ci_ok], pch = "_", col = col)
    }
    ok <- is.finite(piece$mean_label)
    if (any(ok)) {
      ok_index <- which(ok)
      runs <- split(
        ok_index,
        cumsum(c(TRUE, diff(piece$event_time[ok_index]) != 1L))
      )
      for (run in runs) {
        if (length(run) >= 2L) {
          lines(piece$event_time[run], piece$mean_label[run], col = col, lwd = 2)
        }
        points(piece$event_time[run], piece$mean_label[run], col = col, pch = 16)
      }
    }
  }
  if (show_legend) {
    legend(
      "topleft", legend = c("Pre-treatment", "Post-treatment"),
      col = c(pre_col, post_col), lwd = 2, pch = 16,
      bty = "n", cex = 0.85
    )
  }
  invisible(TRUE)
}

figure_font_family <- "Times New Roman"

write_plot_bundle <- function(base_path, draw_fun, width = 7.5, height = 5) {
  paths <- c(
    png = paste0(base_path, ".png"),
    pdf = paste0(base_path, ".pdf"),
    svg = paste0(base_path, ".svg")
  )
  draw_with_font <- function() {
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par), add = TRUE)
    par(family = figure_font_family)
    draw_fun()
  }
  grDevices::png(paths[["png"]], width = round(width * 300),
                 height = round(height * 300), res = 300, type = "cairo")
  tryCatch(draw_with_font(), finally = grDevices::dev.off())
  grDevices::cairo_pdf(paths[["pdf"]], width = width, height = height,
                       family = figure_font_family)
  tryCatch(draw_with_font(), finally = grDevices::dev.off())
  grDevices::svg(paths[["svg"]], width = width, height = height,
                 family = figure_font_family)
  tryCatch(draw_with_font(), finally = grDevices::dev.off())
  unname(paths)
}

write_event_study_figures <- function(series, figure_dir, joint_tests = NULL) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  if (nrow(series) == 0L) return(character(0))
  written <- character(0)
  keys <- unique(series[, .(frequency, outcome_family, outcome)])
  for (index in seq_len(nrow(keys))) {
    frequency_name <- keys$frequency[[index]]
    family <- keys$outcome_family[[index]]
    outcome_name <- keys$outcome[[index]]
    part <- series[
      frequency == frequency_name & outcome_family == family & outcome == outcome_name
    ]
    if (nrow(part) < 2L) next
    test_row <- if (!is.null(joint_tests)) {
      joint_tests[
        frequency == frequency_name & outcome_family == family & outcome == outcome_name
      ]
    } else NULL
    safe_key <- gsub(
      "[^A-Za-z0-9_.-]+", "_",
      paste0(frequency_name, "__", family, "__", outcome_name)
    )
    base_path <- file.path(figure_dir, paste0("event_study_", safe_key))
    draw_one <- function() {
      draw_event_study_plot(
        part, paste0("Event study: ", frequency_name, " / ", family,
                     " / ", outcome_name), test_row
      )
    }
    written <- c(written, write_plot_bundle(base_path, draw_one))
  }

  # One readable family-level figure is more useful than a single raw pooled
  # curve when a family has multiple outcomes on different scales (notably
  # POI). Each facet remains on its native transformed outcome scale.
  for (frequency_name in unique(series$frequency)) {
    for (family in unique(series[frequency == frequency_name, outcome_family])) {
      outcomes <- unique(series[
        frequency == frequency_name & outcome_family == family, outcome
      ])
    outcomes <- outcomes[vapply(outcomes, function(value) {
      nrow(series[
        frequency == frequency_name & outcome_family == family & outcome == value
      ]) >= 2L
    }, logical(1L))]
    if (!length(outcomes)) next
    n_panel_cols <- if (length(outcomes) == 1L) 1L else 2L
    n_panel_rows <- ceiling(length(outcomes) / n_panel_cols)
    family_base <- file.path(
      figure_dir,
      paste0("event_study_family_", frequency_name, "_", family)
    )
    draw_family <- function() {
      old_par <- par(no.readonly = TRUE)
      on.exit(par(old_par), add = TRUE)
      par(mfrow = c(n_panel_rows, n_panel_cols), mar = c(4, 4, 3, 1),
          oma = c(0, 0, 2, 0))
      for (panel_index in seq_along(outcomes)) {
        outcome_name <- outcomes[[panel_index]]
        part <- series[
          frequency == frequency_name & outcome_family == family & outcome == outcome_name
        ]
        test_row <- if (!is.null(joint_tests)) {
          joint_tests[
            frequency == frequency_name & outcome_family == family & outcome == outcome_name
          ]
        } else NULL
        draw_event_study_plot(part, outcome_name, test_row,
                              show_legend = panel_index == 1L)
      }
      mtext(paste0("Pooled event-study overview: ", frequency_name, " / ", family),
            outer = TRUE, cex = 1.2)
    }
      written <- c(written, write_plot_bundle(
        family_base, draw_family,
        width = if (n_panel_cols == 1L) 7.5 else 12,
        height = max(4.5 * n_panel_rows, 5)
      ))
    }
  }

  # A single four-family overview is the primary hand-off figure. Panels keep
  # native outcome scales, while sharing the same event-time convention,
  # treatment boundary, CI styling, and pre-trend annotation.
  overview_keys <- unique(series[, .(frequency, outcome_family, outcome)])
  n_panel_cols <- if (nrow(overview_keys) <= 1L) 1L else 2L
  n_panel_rows <- ceiling(nrow(overview_keys) / n_panel_cols)
  overview_base <- file.path(figure_dir, "event_study_overview")
  draw_overview <- function() {
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par), add = TRUE)
    par(mfrow = c(n_panel_rows, n_panel_cols), mar = c(4, 4, 3, 1),
        oma = c(0, 0, 2, 0))
    for (panel_index in seq_len(nrow(overview_keys))) {
      frequency_name <- overview_keys$frequency[[panel_index]]
      family <- overview_keys$outcome_family[[panel_index]]
      outcome_name <- overview_keys$outcome[[panel_index]]
      part <- series[
        frequency == frequency_name & outcome_family == family & outcome == outcome_name
      ]
      test_row <- if (!is.null(joint_tests)) {
        joint_tests[
          frequency == frequency_name & outcome_family == family & outcome == outcome_name
        ]
      } else NULL
      draw_event_study_plot(
        part, paste0(frequency_name, " / ", family, " / ", outcome_name), test_row,
        show_legend = panel_index == 1L
      )
    }
    mtext("Pooled event-study overview across outcome families", outer = TRUE, cex = 1.2)
  }
  written <- c(written, write_plot_bundle(
    overview_base, draw_overview, width = 12, height = max(4.5 * n_panel_rows, 5)
  ))
  written
}

# Full pipeline: staging root + output directory -> summary files.
run_event_study_aggregation <- function(staging_root, output_dir,
                                        min_pre_event_time = NULL,
                                        stratum_attribute = NULL,
                                        stratum_values = NULL,
                                        orders = NULL,
                                        specification_fingerprint =
                                          "^main_a6_r1km__a6__w3__price_main$",
                                        frequency = NULL) {
  if (!is.null(min_pre_event_time) &&
      (!is.finite(min_pre_event_time) || min_pre_event_time >= 0L)) {
    stop("min_pre_event_time must be a finite negative integer")
  }
  if (!is.null(frequency) && !frequency %in% c("annual", "monthly")) {
    stop("frequency must be annual or monthly when supplied")
  }
  labels <- read_event_study_labels(
    staging_root,
    stratum_attribute = stratum_attribute,
    stratum_values = stratum_values,
    orders = orders,
    specification_fingerprint = specification_fingerprint,
    frequency = frequency
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
  figures <- write_event_study_figures(series, figure_dir, joint_tests)

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
    "; treatment/post rows (event_time >= 0): ", nrow(labels[event_time >= 0L]), ""
  ))
  if ("specification_fingerprint" %in% names(labels)) {
    specs <- sort(unique(na.omit(labels$specification_fingerprint)))
    lines <- c(lines, paste0(
      "- Specification fingerprint(s): ",
      if (length(specs)) paste(specs, collapse = ", ") else "NA"
    ))
  }
  if ("method" %in% names(labels)) {
    methods <- sort(unique(na.omit(labels$method)))
    lines <- c(lines, paste0(
      "- Estimator method(s): ",
      if (length(methods)) paste(methods, collapse = ", ") else "NA"
    ))
  }
  lines <- c(lines, "")
  lines <- c(lines, "## Joint zero-pre-trend tests (grid-level means, one-sample t)", "")
  if (nrow(joint_tests) == 0L) {
    lines <- c(lines, "No production estimator outputs found yet; run the label queues first.")
  } else {
    lines <- c(lines, "| frequency | family | outcome | n_grids | mean | sd | t | p | reject 5% |")
    lines <- c(lines, "|---|---|---|---|---|---|---|---|---|")
    for (i in seq_len(nrow(joint_tests))) {
      row <- joint_tests[i]
      lines <- c(lines, sprintf(
        "| %s | %s | %s | %d | %.4g | %.4g | %s | %s | %s |",
        row$frequency, row$outcome_family, row$outcome, row$n_grids,
        row$mean_grid_mean,
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
