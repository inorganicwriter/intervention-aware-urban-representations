suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(fixest)
})

# Standard two-way fixed-effects event-study regression on the matched
# sample (treated grids + their frozen matched control grids).
#
#   Y_it = sum_{k in K} beta_k * D_it^k + alpha_i + gamma_t + eps_it
#
# - Y_it: outcome level (housing log price, asinh VIIRS radiance)
# - D_it^k: event-time dummies, baseline is the last clean pre-period
# - alpha_i: grid fixed effects; gamma_t: calendar-month fixed effects
# - SEs clustered by grid
#
# Parallel-trends check: joint Wald test of H0: beta_k = 0 for all k < 0.
#
# Usage:
#   Rscript run_event_study_matching.R OUTCOME_FAMILY [MIN_PRE] [MAX_POST]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  stop("Usage: run_event_study_matching.R OUTCOME_FAMILY [MIN_PRE] [MAX_POST]")
}
outcome_family <- args[[1L]]
if (!outcome_family %in% c("housing", "viirs", "poi", "population")) {
  stop("Event-study matching supports housing | viirs | poi | population")
}
is_annual <- outcome_family %in% c("poi", "population")
anticipation_months <- if (is_annual) 0L else 6L
lag_periods <- if (is_annual) 3L else 36L
clean_pre_end_event_time <- -(anticipation_months + 1L)
default_min_pre <- if (is_annual) -4L else -(lag_periods + anticipation_months)
min_pre <- if (length(args) >= 2L) as.integer(args[[2L]]) else default_min_pre
max_post <- if (length(args) >= 3L) as.integer(args[[3L]]) else if (is_annual) 3L else 24L
reference_event_time <- clean_pre_end_event_time
if (is.na(min_pre) || is.na(max_post) || min_pre > reference_event_time ||
    max_post < 0L) {
  stop("Event-study range must satisfy MIN_PRE < 0 <= MAX_POST")
}
figure_font_family <- "Times New Roman"

save_recorded_vector_plot <- function(plot_record, base_path, width = 8, height = 5.2) {
  grDevices::cairo_pdf(
    paste0(base_path, ".pdf"), width = width, height = height,
    family = figure_font_family
  )
  replayPlot(plot_record)
  grDevices::dev.off()
  grDevices::svg(
    paste0(base_path, ".svg"), width = width, height = height,
    family = figure_font_family
  )
  replayPlot(plot_record)
  grDevices::dev.off()
}

root <- project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
source(file.path(root, "scripts", "causal_r", "paths.R"))
load_project_paths(root)

control_queue <- fread(file.path(CAUSAL_DIR, "control_design_queue.csv"))
matched <- control_queue[status == "matched" & !is.na(control_city_key) & !is.na(control_grid_id)]
if (nrow(matched) == 0L) {
  stop("No matched control designs yet; run the grid-control queue first")
}
cat(sprintf("Matched pairs: %d\n", nrow(matched)))

# ---- Build event-time aligned panel ------------------------------------
# Monthly VIIRS partitions are per city-year-month; read each distinct
# city/opening window once into a memory cache. The opening month is part of
# the key because one city can contain treatment events at different dates.
viirs_cache <- new.env()
read_viirs_window <- function(city_key, opening_month, months_back = 36L, months_ahead = 24L) {
  opening_key <- format(as.IDate(format(as.IDate(opening_month), "%Y-%m-01")), "%Y-%m")
  cache_key <- paste(city_key, opening_key, months_back, months_ahead, sep = "::")
  if (!exists(cache_key, envir = viirs_cache)) {
    # IDate arithmetic must stay integral: a float day offset would turn the
    # sequence into numeric and break seq(by = "month").  31-day bounds only
    # widen the window; months are normalised to the 1st below.
    window <- seq(
      as.IDate(opening_month) - months_back * 31L,
      as.IDate(opening_month) + months_ahead * 31L,
      by = "month"
    )
    parts <- lapply(window, function(period) {
      year <- as.integer(format(period, "%Y"))
      month_number <- as.integer(format(period, "%m"))
      path <- file.path(
        VIIRS_MONTHLY_DIR, paste0("city_key=", city_key),
        paste0("year=", year), sprintf("month=%02d", month_number), "part.parquet"
      )
      if (!file.exists(path)) return(NULL)
      x <- as.data.table(read_parquet(
        path, col_select = c("grid_id", "avg_rad")
      ))
      x[, `:=`(city_key = city_key, month = as.IDate(format(period, "%Y-%m-01")))]
      x
    })
    parts <- parts[!vapply(parts, is.null, logical(1L))]
    assign(cache_key, rbindlist(parts, use.names = TRUE), envir = viirs_cache)
  }
  get(cache_key, envir = viirs_cache)
}

read_panel <- function(city_key, opening_month) {
  if (outcome_family == "housing") {
    path <- file.path(PANEL_HOUSING_MONTHLY_DIR, paste0(city_key, ".parquet"))
    if (!file.exists(path)) return(NULL)
    x <- as.data.table(read_parquet(
      path, col_select = c("grid_id", "observed_month", "log_price_raw_median")
    ))
    x[, city_key := city_key]
    x[, month := as.IDate(format(as.IDate(observed_month), "%Y-%m-01"))]
    x[, outcome := log_price_raw_median]
    x[, .(city_key, grid_id, month, outcome)]
  } else if (is_annual) {
    if (outcome_family == "poi") {
      path <- file.path(POI_DIR, paste0(city_key, "_poi_grid_yearly.parquet"))
    } else {
      path <- file.path(POPULATION_DIR, paste0(city_key, "_pop.parquet"))
    }
    if (!file.exists(path)) return(NULL)
    x <- as.data.table(read_parquet(path))
    count_var <- if (outcome_family == "poi") "poi_count" else "pop_count"
    if (!nrow(x) || !count_var %in% names(x)) return(NULL)
    x[, city_key := city_key]
    x[, month := as.IDate(paste0(year, "-01-01"))]
    # log transform mirrors the estimator families (housing_log_price,
    # poi_count_log, population_log are all log-scale outcomes).
    x[, outcome := log1p(pmax(get(count_var), 0))]
    x[, .(city_key, grid_id, month, outcome)]
  } else {
    x <- read_viirs_window(city_key, opening_month)
    if (is.null(x) || !nrow(x)) return(NULL)
    x[, outcome := asinh(avg_rad)]
    x[, .(city_key, grid_id, month, outcome)]
  }
}

# Treated and control grids by city.
treated_units <- unique(matched[, .(treatment_order, city_key, grid_id, control_city_key, control_grid_id, opening_month)])
# Same-month opening size per city (spillover heterogeneity), loaded lazily.
acc_cache <- new.env()
get_same_month_size <- function(city_key, target_grid_id) {
  if (!exists(city_key, envir = acc_cache)) {
    acc_path <- file.path(CAUSAL_DIR, "accessibility_features",
                          paste0(city_key, "_accessibility.parquet"))
    if (file.exists(acc_path)) {
      acc_dt <- as.data.table(read_parquet(
        acc_path,
        col_select = c("grid_id", "stations_opened_same_month")
      ))
      assign(city_key, acc_dt, envir = acc_cache)
    } else {
      assign(city_key, NULL, envir = acc_cache)
    }
  }
  acc_dt <- get(city_key, envir = acc_cache)
  if (is.null(acc_dt)) return(NA_integer_)
  hit <- acc_dt[grid_id == target_grid_id, stations_opened_same_month]
  if (length(hit) == 0L) NA_integer_ else as.integer(hit[1L])
}
panel_parts <- list()
for (i in seq_len(nrow(treated_units))) {
  row <- treated_units[i]
  opening <- as.IDate(paste0(row$opening_month, "-01"))
  t_panel <- read_panel(row$city_key, opening)
  c_panel <- read_panel(row$control_city_key, opening)
  if (is.null(t_panel) || is.null(c_panel)) next
  t_part <- t_panel[grid_id == row$grid_id]
  c_part <- c_panel[grid_id == row$control_grid_id]
  if (nrow(t_part) == 0L || nrow(c_part) == 0L) next
  t_part[, role := "treated"]
  c_part[, role := "control"]
  combined <- rbind(t_part, c_part, use.names = TRUE, fill = TRUE)
  combined[, treatment_order := row$treatment_order]
  combined[, event_time := if (is_annual) {
    as.integer(format(month, "%Y")) - as.integer(format(opening, "%Y"))
  } else {
    (as.integer(format(month, "%Y")) * 12L + as.integer(format(month, "%m"))) -
      (as.integer(format(opening, "%Y")) * 12L + as.integer(format(opening, "%m")))
  }]
  combined[, grid_id := paste0(row$city_key, "::", grid_id)]
  combined[, unit := paste0(role, "_", treatment_order, "_", grid_id)]
  # Treatment time (month index) for the Sun-Abraham estimator: treated
  # units are treated at their own opening month; control units are never
  # treated (treatment time = +Inf).  The unit-level panel time axis is the
  # calendar month; sunab() needs a treatment-period variable per unit.
  combined[, treatment_time := ifelse(role == "treated",
                                      as.integer(format(opening, if (is_annual) "%Y" else "%Y%m")),
                                      as.integer(NA))]
  # Same-month opening size: spillover / network-effect heterogeneity
  combined[, same_month_openings := ifelse(
    role == "treated",
    get_same_month_size(row$city_key, row$grid_id),
    NA_integer_
  )]
  panel_parts[[length(panel_parts) + 1L]] <- combined
}
if (!length(panel_parts)) {
  stop("No treated/control outcome panels could be read for the matched queue")
}
panel <- rbindlist(panel_parts, use.names = TRUE, fill = TRUE)
# The opening month/year is a partial-exposure boundary and is not a model
# period.  The anticipation window is also excluded from this parallel-trends
# sample. The first complete month/year after opening is event_time = 1.
panel <- panel[
  event_time != 0L &
    event_time >= min_pre &
    event_time <= max_post &
    (event_time <= clean_pre_end_event_time | event_time >= 1L)
]
panel <- panel[is.finite(outcome)]
if (!nrow(panel)) {
  stop("Matched event-study panel is empty after event-time and finite-outcome filters")
}
if (anyNA(panel[role == "treated", treatment_time])) {
  stop("Treated event-study units have missing treatment_time")
}
cat(sprintf("Panel rows: %d (units: %d, treated events: %d)\n",
            nrow(panel), uniqueN(panel$unit), uniqueN(panel$treatment_order)))

# ---- Event-study regression (TWFE, grid + month FE, cluster by grid) ----
# Use i(event_time, ref=last clean pre-period): fixest expands numeric event-time dummies
# robustly (factor-based manual dummies mis-parse negative levels).  Controls
# are coded as the omitted reference period rather than NA so that fixest
# cannot drop the entire control panel during RHS missing-value handling.
panel[, et_dummy := as.integer(ifelse(
  role == "treated", event_time, reference_event_time
))]
if (anyNA(panel$et_dummy) || any(panel[
  role == "control", et_dummy != reference_event_time
])) {
  stop("Invalid event-time coding for matched controls")
}

formula_text <- sprintf(
  "outcome ~ i(et_dummy, ref = %dL) | unit + month", reference_event_time
)
fit <- tryCatch(
  feols(as.formula(formula_text), data = panel,
        cluster = ~ grid_id),
  error = function(e) NULL
)
fit_city <- tryCatch(
  feols(as.formula(formula_text), data = panel,
        cluster = ~ city_key),
  error = function(e) NULL
)
if (is.null(fit)) {
  cat("TWFE grid-cluster estimation failed (sparse panel); writing empty coefficients.\n")
  coef_table <- data.table(term = character(0), event_time = integer(0))
} else {
  coef_table <- as.data.table(coeftable(fit), keep.rownames = TRUE)
  setnames(coef_table, "rn", "term")
  coef_table[, event_time := as.integer(sub("et_dummy::", "", term))]
  if (!is.null(fit_city)) {
    coef_city <- as.data.table(coeftable(fit_city), keep.rownames = TRUE)
    setnames(coef_city, "rn", "term")
    coef_city[, event_time := as.integer(sub("et_dummy::", "", term))]
    setnames(coef_city, "Std. Error", "se_city")
    coef_table <- merge(coef_table, coef_city[, .(term, se_city)],
                        by = "term", all.x = TRUE)
  }
}

# ---- Joint parallel-trends Wald test (all pre-period betas = 0) ---------
  pre_terms <- coef_table[event_time < 0L & event_time != reference_event_time]$term
wald <- NULL
wald_city <- NULL
if (length(pre_terms) > 0L && !is.null(fit)) {
  wald <- tryCatch(
    as.data.table(wald_test(fit, keep = pre_terms))[1L],
    error = function(e) NULL
  )
}
if (length(pre_terms) > 0L && !is.null(fit_city)) {
  wald_city <- tryCatch(
    as.data.table(wald_test(fit_city, keep = pre_terms))[1L],
    error = function(e) NULL
  )
}

# ---- Output -------------------------------------------------------------
out_dir <- file.path(OUTPUT_DIR, "event_study", "matching", outcome_family)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fwrite(coef_table, file.path(out_dir, "event_study_coefficients.csv"), bom = TRUE)
if (!is.null(wald)) {
  fwrite(wald, file.path(out_dir, "parallel_trends_wald.csv"), bom = TRUE)
}
if (!is.null(wald_city)) {
  fwrite(wald_city, file.path(out_dir, "parallel_trends_wald_city_cluster.csv"), bom = TRUE)
}
writeLines(
  if (is.null(fit)) "TWFE grid-cluster estimation failed (sparse panel)"
  else capture.output(summary(fit)),
  file.path(out_dir, "event_study_fit_summary.txt")
)
writeLines(
  if (is.null(fit_city)) "TWFE city-cluster estimation failed (sparse panel)"
  else capture.output(summary(fit_city)),
  file.path(out_dir, "event_study_fit_summary_city_cluster.txt")
)

# ---- Treatment-attribute stratification ---------------------------------
# Station-level attributes (transfer / new-line / terminal / same-month
# opening size) are treatment attributes: they cannot enter treated-control
# matching but stratify the response.  Strata with fewer than 3 treated
# events report trends only (no Wald power claim).
attributes_path <- file.path(
  OUTPUT_DIR, "causal_labels", "station_attributes", "station_attributes.parquet"
)
if (file.exists(attributes_path)) {
  attributes <- as.data.table(read_parquet(attributes_path, col_select = c(
    "treatment_order", "is_transfer_at_opening", "is_new_line_opening",
    "is_extension_opening", "is_terminal_at_opening", "same_month_openings"
  )))
  # The panel already carries a same_month_openings column (accessibility
  # source); rename the attribute version to avoid data.table's .x/.y clash.
  setnames(attributes, "same_month_openings", "attr_same_month_openings")
  panel <- merge(panel, attributes, by = "treatment_order", all.x = TRUE)
  strata <- list(
    transfer = c("is_transfer_at_opening" = 1L),
    non_transfer = c("is_transfer_at_opening" = 0L),
    new_line = c("is_new_line_opening" = 1L),
    existing_line = c("is_new_line_opening" = 0L),
    terminal = c("is_terminal_at_opening" = 1L),
    non_terminal = c("is_terminal_at_opening" = 0L)
  )
  same_month_median <- stats::median(
    panel[role == "treated" & !is.na(attr_same_month_openings), attr_same_month_openings],
    na.rm = TRUE
  )
  if (is.finite(same_month_median)) {
    panel[, large_batch := as.integer(attr_same_month_openings > same_month_median)]
    strata$small_batch <- c("large_batch" = 0L)
    strata$large_batch <- c("large_batch" = 1L)
  }
  for (stratum_name in names(strata)) {
    condition <- strata[[stratum_name]]
    keep <- rep(TRUE, nrow(panel))
    for (column in names(condition)) {
      keep <- keep & (panel[[column]] == condition[[column]] | panel$role == "control")
    }
    strat_panel <- panel[keep]
    strat_events <- uniqueN(strat_panel[role == "treated"]$treatment_order)
    if (strat_events < 3L) next
    fit_stratum <- tryCatch(
      feols(as.formula(formula_text), data = strat_panel,
            cluster = ~ grid_id),
      error = function(e) NULL
    )
    if (is.null(fit_stratum)) next
    coef_stratum <- as.data.table(coeftable(fit_stratum), keep.rownames = TRUE)
    setnames(coef_stratum, "rn", "term")
    coef_stratum[, event_time := as.integer(sub("et_dummy::", "", term))]
    coef_stratum[, stratum := stratum_name]
    coef_stratum[, n_treated_events := strat_events]
    pre_terms_stratum <- coef_stratum[
      event_time < 0L & event_time != reference_event_time
    ]$term
    wald_stratum <- NULL
    if (length(pre_terms_stratum) > 0L) {
      wald_stratum <- tryCatch(
        as.data.table(wald_test(fit_stratum, keep = pre_terms_stratum))[1L],
        error = function(e) NULL
      )
    }
    fwrite(
      coef_stratum,
      file.path(out_dir, paste0("stratum_", stratum_name, "_coefficients.csv")),
      bom = TRUE
    )
    if (!is.null(wald_stratum)) {
      fwrite(
        wald_stratum[, stratum := stratum_name],
        file.path(out_dir, paste0("stratum_", stratum_name, "_wald.csv")),
        bom = TRUE
      )
    }
    cat(sprintf("Stratum %s: %d treated events, coefficients written\n",
                stratum_name, strat_events))
  }
}
cat("\nSaved to", out_dir, "\n")

# ---- Figure -------------------------------------------------------------
if (!is.null(fit) && nrow(coef_table) > 0L) {
  png(file.path(out_dir, "event_study_matching.png"), width = 2400, height = 1560,
      res = 300, type = "cairo")
  par(family = figure_font_family)
  ci <- tryCatch(confint(fit), error = function(e) NULL)
  if (is.null(ci)) {
    dev.off()
  } else {
    ci_dt <- as.data.table(ci, keep.rownames = TRUE)
    setnames(ci_dt, c("term", "ci_lower", "ci_upper"))
    plot_data <- merge(coef_table[, .(term, event_time, estimate = Estimate)],
                       ci_dt[, .(term, ci_lower, ci_upper)], by = "term", all.x = TRUE)
    plot_data[, event_time := as.integer(sub("et_dummy::", "", term))]
    plot_data <- plot_data[order(event_time)]
    x_range <- range(plot_data$event_time)
    # Include the full CI range in the axis so uncertainty is never clipped.
    # The event-study figure should show the inferential object, not only the
    # point estimates.
    y_span <- range(c(plot_data$estimate, plot_data$ci_lower,
                      plot_data$ci_upper, 0), na.rm = TRUE)
    y_pad <- 0.15 * max(diff(y_span), 0.1)
    y_min <- y_span[1L] - y_pad
    y_max <- y_span[2L] + y_pad
    plot(plot_data$event_time, plot_data$estimate, type = "n",
         xlim = x_range, ylim = c(y_min, y_max),
         xlab = "Event time (months)", ylab = "Coefficient (beta_k)",
         main = paste0("Event study (matched sample): ", outcome_family))
    abline(h = 0, lty = 2, col = "grey50")
    abline(v = 0, lty = 3, col = "grey70")
    # Error bars are safer than a ribbon here: event times can be missing in
    # sparse matched panels, and a polygon would imply unsupported values
    # between non-adjacent periods.
    segments(plot_data$event_time, plot_data$ci_lower,
             plot_data$event_time, plot_data$ci_upper, col = "steelblue", lwd = 1)
    lines(plot_data$event_time, plot_data$estimate, col = "steelblue", lwd = 1.5)
    points(plot_data$event_time, plot_data$estimate, pch = 19, col = "steelblue")
    plot_record <- recordPlot()
    dev.off()
    save_recorded_vector_plot(
      plot_record, file.path(out_dir, "event_study_matching")
    )
    cat("Figure written.\n")
  }
}

# ---- Sun-Abraham (2021) heterogeneity-robust event study ---------------
# Interaction-weighted estimator robust to staggered-treatment bias in TWFE
# (Goodman-Bacon 2021; Sun & Abraham 2021).  Both arguments of sunab() must
# use the same calendar scale: treatment_time is the absolute opening month
# (YYYYMM) and calendar_month is the absolute panel month (YYYYMM); control
# units carry treatment_time = NA, which fixest treats as never-treated and
# keeps as the clean comparison group.
sunab_fit <- NULL
sunab_coef <- NULL
sunab_wald <- NULL
tryCatch(
  {
    panel[, calendar_month := if (is_annual) {
      as.integer(format(month, "%Y"))
    } else {
      as.integer(format(as.IDate(month), "%Y%m"))
    }]
    sunab_fit <- feols(
      outcome ~ sunab(
        treatment_time, calendar_month, ref.c = reference_event_time
      ) | unit + month,
      data = panel,
      cluster = ~ grid_id
    )
    sunab_coef <- as.data.table(coeftable(sunab_fit), keep.rownames = TRUE)
    setnames(sunab_coef, "rn", "term")
    sunab_coef[, event_time := as.integer(sub("calendar_month::([-0-9]+)(:cohort::.*)?$", "\\1", term))]
    fwrite(sunab_coef,
           file.path(out_dir, "event_study_sun_abraham_coefficients.csv"),
           bom = TRUE)
    # Keep the study-window relative periods only for the Wald test and the
    # figure; the full coefficient table is archived above for audit.
    sunab_window <- sunab_coef[event_time %in% seq.int(min_pre, max_post)]
    pre_terms_sa <- sunab_window[
      event_time < 0L & event_time != reference_event_time
    ]$term
    if (length(pre_terms_sa) > 0L) {
      sunab_wald <- tryCatch(
        as.data.table(wald_test(sunab_fit, keep = pre_terms_sa))[1L],
        error = function(e) NULL
      )
    }
    if (!is.null(sunab_wald)) {
      fwrite(sunab_wald,
             file.path(out_dir, "parallel_trends_wald_sun_abraham.csv"),
             bom = TRUE)
    }
    writeLines(
      capture.output(summary(sunab_fit)),
      file.path(out_dir, "event_study_sun_abraham_fit_summary.txt")
    )
    # Figure: Sun-Abraham coefficients vs event time
    png(file.path(out_dir, "event_study_matching_sun_abraham.png"),
        width = 2400, height = 1560, res = 300, type = "cairo")
    par(family = figure_font_family)
    ci_sa <- confint(sunab_fit)
    ci_sa_dt <- as.data.table(ci_sa, keep.rownames = TRUE)
    setnames(ci_sa_dt, c("term", "ci_lower", "ci_upper"))
    plot_sa <- merge(sunab_window[, .(term, event_time, estimate = Estimate)],
                     ci_sa_dt[, .(term, ci_lower, ci_upper)],
                     by = "term", all.x = TRUE)
    plot_sa <- plot_sa[order(event_time)]
    sa_span <- range(c(plot_sa$estimate, plot_sa$ci_lower,
                       plot_sa$ci_upper, 0), na.rm = TRUE)
    sa_pad <- 0.15 * max(diff(sa_span), 0.1)
    plot(plot_sa$event_time, plot_sa$estimate, type = "n",
         xlim = range(plot_sa$event_time),
         ylim = c(sa_span[1L] - sa_pad, sa_span[2L] + sa_pad),
         xlab = "Event time (months)", ylab = "IW coefficient",
         main = paste0("Sun-Abraham event study (matched): ", outcome_family))
    abline(h = 0, lty = 2, col = "grey50")
    abline(v = 0, lty = 3, col = "grey70")
    segments(plot_sa$event_time, plot_sa$ci_lower,
             plot_sa$event_time, plot_sa$ci_upper, col = "darkorange", lwd = 1)
    lines(plot_sa$event_time, plot_sa$estimate, col = "darkorange", lwd = 1.5)
    points(plot_sa$event_time, plot_sa$estimate, pch = 19, col = "darkorange")
    plot_record <- recordPlot()
    dev.off()
    save_recorded_vector_plot(
      plot_record, file.path(out_dir, "event_study_matching_sun_abraham")
    )
    cat("Sun-Abraham event study written.\n")
  },
  error = function(e) {
    cat("Sun-Abraham estimation skipped:", conditionMessage(e), "\n")
  }
)

# ---- Spillover / network-effect heterogeneity --------------------------
# Split treated events by same-month opening size (median) and estimate the
# Sun-Abraham event study within each stratum.  Large simultaneous openings
# proxy for network effects (Yu et al. 2013, JTG): if the post-treatment
# response differs across strata, spillover/network effects matter.
same_month_vals <- panel[role == "treated" & !is.na(same_month_openings),
                         same_month_openings]
if (length(same_month_vals) > 4L) {
  med <- median(same_month_vals, na.rm = TRUE)
  panel[, stratum := ifelse(role == "treated" & !is.na(same_month_openings),
                            ifelse(same_month_openings > med, "large", "small"),
                            NA_character_)]
  for (strat in c("small", "large")) {
    strat_panel <- panel[stratum == strat | role == "control"]
    if (nrow(strat_panel[stratum == strat]) < 3L) next
    fit_strat <- tryCatch(
      feols(
        outcome ~ sunab(
          treatment_time, calendar_month, ref.c = reference_event_time
        ) | unit + month,
        data = strat_panel, cluster = ~ grid_id
      ),
      error = function(e) NULL
    )
    if (is.null(fit_strat)) {
      cat(sprintf("Spillover stratum %s skipped\n", strat))
      next
    }
    coef_strat <- as.data.table(coeftable(fit_strat), keep.rownames = TRUE)
    setnames(coef_strat, "rn", "term")
    coef_strat[, event_time := as.integer(sub("calendar_month::([-0-9]+)(:cohort::.*)?$", "\\1", term))]
    coef_strat[, stratum := strat]
    fwrite(coef_strat,
           file.path(out_dir, sprintf("spillover_%s_coefficients.csv", strat)),
           bom = TRUE)
    if (nrow(coef_strat) > 0L && all(is.finite(coef_strat$event_time))) {
      png(file.path(out_dir, sprintf("spillover_%s_event_study.png", strat)),
          width = 2100, height = 1400, res = 300, type = "cairo")
      par(family = figure_font_family)
      ci_s <- confint(fit_strat)
      ci_s_dt <- as.data.table(ci_s, keep.rownames = TRUE)
      setnames(ci_s_dt, c("term", "ci_lower", "ci_upper"))
      plot_s <- merge(coef_strat[, .(term, event_time, estimate = Estimate)],
                      ci_s_dt[, .(term, ci_lower, ci_upper)],
                      by = "term", all.x = TRUE)[order(event_time)]
      sp_span <- range(c(plot_s$estimate, plot_s$ci_lower,
                         plot_s$ci_upper, 0), na.rm = TRUE)
      sp_pad <- 0.15 * max(diff(sp_span), 0.1)
      plot(plot_s$event_time, plot_s$estimate, type = "n",
           xlim = range(plot_s$event_time),
           ylim = c(sp_span[1L] - sp_pad, sp_span[2L] + sp_pad),
           xlab = "Event time", ylab = "IW coefficient",
           main = sprintf("Spillover stratum: %s (same-month openings)", strat))
      abline(h = 0, lty = 2, col = "grey50")
      abline(v = 0, lty = 3, col = "grey70")
      segments(plot_s$event_time, plot_s$ci_lower,
               plot_s$event_time, plot_s$ci_upper, col = "darkgreen", lwd = 1)
      lines(plot_s$event_time, plot_s$estimate, col = "darkgreen", lwd = 1.5)
      points(plot_s$event_time, plot_s$estimate, pch = 19, col = "darkgreen")
      plot_record <- recordPlot()
      dev.off()
      save_recorded_vector_plot(
        plot_record, file.path(out_dir, sprintf("spillover_%s_event_study", strat))
      )
    }
    cat(sprintf("Spillover stratum %s written (median=%d)\n", strat, med))
  }
} else {
  cat("Spillover heterogeneity skipped: too few treated with same-month size\n")
}
