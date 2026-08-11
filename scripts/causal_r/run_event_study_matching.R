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
# - D_it^k: event-time dummies, baseline k = -1 omitted
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
min_pre <- if (length(args) >= 2L) as.integer(args[[2L]]) else -36L
max_post <- if (length(args) >= 3L) as.integer(args[[3L]]) else 24L

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
outcome_col <- if (outcome_family == "housing") "log_price_raw_median" else "avg_rad"
panel_files <- if (outcome_family == "housing") {
  list.files(file.path(PANEL_HOUSING_MONTHLY_DIR), pattern = "*.parquet$",
             full.names = TRUE)
} else {
  stop("Event-study matching currently supports outcome_family = housing")
}

read_panel <- function(city_key) {
  path <- file.path(PANEL_HOUSING_MONTHLY_DIR, paste0(city_key, ".parquet"))
  if (!file.exists(path)) return(NULL)
  x <- as.data.table(read_parquet(
    path, col_select = c("grid_id", "observed_month", outcome_col)
  ))
  x[, city_key := city_key]
  x[, month := as.IDate(format(as.IDate(observed_month), "%Y-%m-01"))]
  x[, outcome := get(outcome_col)]
  x[, .(city_key, grid_id, month, outcome)]
}

# Treated and control grids by city.
treated_units <- unique(matched[, .(city_key, grid_id, control_city_key, control_grid_id, opening_month)])
# Same-month opening size per city (spillover heterogeneity), loaded lazily.
acc_cache <- new.env()
get_same_month_size <- function(city_key, grid_id) {
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
  hit <- acc_dt[grid_id == grid_id, stations_opened_same_month]
  if (length(hit) == 0L) NA_integer_ else as.integer(hit[1L])
}
panel_parts <- list()
for (i in seq_len(nrow(treated_units))) {
  row <- treated_units[i]
  t_panel <- read_panel(row$city_key)
  c_panel <- read_panel(row$control_city_key)
  if (is.null(t_panel) || is.null(c_panel)) next
  opening <- as.IDate(paste0(row$opening_month, "-01"))
  t_part <- t_panel[grid_id == row$grid_id]
  c_part <- c_panel[grid_id == row$control_grid_id]
  if (nrow(t_part) == 0L || nrow(c_part) == 0L) next
  t_part[, role := "treated"]
  c_part[, role := "control"]
  combined <- rbind(t_part, c_part, use.names = TRUE, fill = TRUE)
  combined[, treatment_order := row$treatment_order]
  combined[, event_time := as.integer(round(as.numeric(difftime(month, opening, units = "days")) / 30.44))]
  combined[, grid_id := paste0(row$city_key, "::", grid_id)]
  combined[, unit := paste0(role, "_", treatment_order, "_", grid_id)]
  # Treatment time (month index) for the Sun-Abraham estimator: treated
  # units are treated at their own opening month; control units are never
  # treated (treatment time = +Inf).  The unit-level panel time axis is the
  # calendar month; sunab() needs a treatment-period variable per unit.
  combined[, treatment_time := ifelse(role == "treated", as.integer(format(opening, "%Y%m")), as.integer(NA))]
  # Same-month opening size: spillover / network-effect heterogeneity
  combined[, same_month_openings := ifelse(
    role == "treated",
    get_same_month_size(row$city_key, row$grid_id),
    NA_integer_
  )]
  panel_parts[[length(panel_parts) + 1L]] <- combined
}
panel <- rbindlist(panel_parts, use.names = TRUE, fill = TRUE)
panel <- panel[event_time >= min_pre & event_time <= max_post]
panel <- panel[is.finite(outcome)]
cat(sprintf("Panel rows: %d (units: %d, treated events: %d)\n",
            nrow(panel), uniqueN(panel$unit), uniqueN(panel$treatment_order)))

# ---- Event-study regression (TWFE, grid + month FE, cluster by grid) ----
# Use i(event_time, ref=-1): fixest expands numeric event-time dummies
# robustly (factor-based manual dummies mis-parse negative levels).
panel[, et_dummy := ifelse(role == "treated", event_time, NA_integer_)]

formula_text <- "outcome ~ i(et_dummy, ref = -1L) | unit + month"
fit <- feols(as.formula(formula_text), data = panel,
             cluster = ~ grid_id)

coef_table <- as.data.table(coeftable(fit), keep.rownames = TRUE)
setnames(coef_table, "rn", "term")
coef_table[, event_time := as.integer(sub("et_dummy::", "", term))]

# ---- City-clustered robustness (Abadie et al. 2023, QJE) ----------------
# Metro openings are city-level policies; grids within a city share city
# shocks, so grid-level clustering may understate SEs.  Re-estimate with
# clustering at the city level as the primary robustness contrast.
fit_city <- feols(as.formula(formula_text), data = panel,
                  cluster = ~ city_key)
coef_city <- as.data.table(coeftable(fit_city), keep.rownames = TRUE)
setnames(coef_city, "rn", "term")
coef_city[, event_time := as.integer(sub("et_dummy::", "", term))]
setnames(coef_city, "Std. Error", "se_city")
coef_table <- merge(coef_table, coef_city[, .(term, se_city)],
                    by = "term", all.x = TRUE)

# ---- Joint parallel-trends Wald test (all pre-period betas = 0) ---------
pre_terms <- coef_table[event_time < 0L & event_time != -1L]$term
wald <- NULL
if (length(pre_terms) > 0L) {
  wald <- tryCatch(
    as.data.table(wald_test(fit, keep = pre_terms))[1L],
    error = function(e) NULL
  )
}
# City-clustered Wald (Abadie et al. 2023)
wald_city <- NULL
if (length(pre_terms) > 0L) {
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
  capture.output(summary(fit)),
  file.path(out_dir, "event_study_fit_summary.txt")
)
writeLines(
  capture.output(summary(fit_city)),
  file.path(out_dir, "event_study_fit_summary_city_cluster.txt")
)
cat("\nSaved to", out_dir, "\n")

# ---- Figure -------------------------------------------------------------
png(file.path(out_dir, "event_study_matching.png"), width = 1000, height = 650, res = 130)
ci <- confint(fit)
ci_dt <- as.data.table(ci, keep.rownames = TRUE)
setnames(ci_dt, c("term", "ci_lower", "ci_upper"))
plot_data <- merge(coef_table[, .(event_time, estimate = Estimate)],
                   ci_dt[, .(term, ci_lower, ci_upper)], by = "term", all.x = TRUE)
plot_data[, event_time := as.integer(sub("et_dummy::", "", term))]
plot_data <- plot_data[order(event_time)]
x_range <- range(plot_data$event_time)
y_max <- max(plot_data$ci_upper, na.rm = TRUE)
y_min <- min(plot_data$ci_lower, 0, na.rm = TRUE)
plot(plot_data$event_time, plot_data$estimate, type = "n",
     xlim = x_range, ylim = c(y_min, y_max),
     xlab = "Event time (months)", ylab = "Coefficient (beta_k)",
     main = paste0("Event study (matched sample): ", outcome_family))
abline(h = 0, lty = 2, col = "grey50")
abline(v = -1, lty = 3, col = "grey70")
segments(plot_data$event_time, plot_data$ci_lower,
         plot_data$event_time, plot_data$ci_upper, col = "steelblue", lwd = 2)
points(plot_data$event_time, plot_data$estimate, pch = 19, col = "steelblue")
dev.off()
cat("Figure written.\n")

# ---- Sun-Abraham (2021) heterogeneity-robust event study ---------------
# Interaction-weighted estimator robust to staggered-treatment bias in TWFE
# (Goodman-Bacon 2021; Sun & Abraham 2021).  Treatment time is set only for
# treated units (their opening month); never-treated controls are the clean
# comparison group.  fixest::sunab estimates the IW coefficients directly.
sunab_fit <- NULL
sunab_coef <- NULL
sunab_wald <- NULL
tryCatch(
  {
    panel[, treatment_month := as.integer(treatment_time)]
    sunab_fit <- feols(
      outcome ~ sunab(treatment_month, event_time, ref.c = -1L) | unit + month,
      data = panel[role == "treated" | !is.na(treatment_time)],
      cluster = ~ grid_id
    )
    sunab_coef <- as.data.table(coeftable(sunab_fit), keep.rownames = TRUE)
    setnames(sunab_coef, "rn", "term")
    sunab_coef[, event_time := as.integer(sub("event_time::", "", term))]
    pre_terms_sa <- sunab_coef[event_time < 0L & event_time != -1L]$term
    if (length(pre_terms_sa) > 0L) {
      sunab_wald <- tryCatch(
        as.data.table(wald_test(sunab_fit, keep = pre_terms_sa))[1L],
        error = function(e) NULL
      )
    }
    fwrite(sunab_coef,
           file.path(out_dir, "event_study_sun_abraham_coefficients.csv"),
           bom = TRUE)
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
        width = 1000, height = 650, res = 130)
    ci_sa <- confint(sunab_fit)
    ci_sa_dt <- as.data.table(ci_sa, keep.rownames = TRUE)
    setnames(ci_sa_dt, c("term", "ci_lower", "ci_upper"))
    plot_sa <- merge(sunab_coef[, .(event_time, estimate = Estimate)],
                     ci_sa_dt[, .(term, ci_lower, ci_upper)],
                     by = "term", all.x = TRUE)
    plot_sa <- plot_sa[order(event_time)]
    plot(plot_sa$event_time, plot_sa$estimate, type = "n",
         xlim = range(plot_sa$event_time),
         ylim = c(min(plot_sa$ci_lower, 0, na.rm = TRUE),
                  max(plot_sa$ci_upper, na.rm = TRUE)),
         xlab = "Event time (months)", ylab = "IW coefficient",
         main = paste0("Sun-Abraham event study (matched): ", outcome_family))
    abline(h = 0, lty = 2, col = "grey50")
    abline(v = -1, lty = 3, col = "grey70")
    segments(plot_sa$event_time, plot_sa$ci_lower,
             plot_sa$event_time, plot_sa$ci_upper, col = "darkorange", lwd = 2)
    points(plot_sa$event_time, plot_sa$estimate, pch = 19, col = "darkorange")
    dev.off()
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
        outcome ~ sunab(treatment_time, event_time, ref.c = -1L) | unit + month,
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
    coef_strat[, event_time := as.integer(sub("event_time::", "", term))]
    coef_strat[, stratum := strat]
    fwrite(coef_strat,
           file.path(out_dir, sprintf("spillover_%s_coefficients.csv", strat)),
           bom = TRUE)
    png(file.path(out_dir, sprintf("spillover_%s_event_study.png", strat)),
        width = 900, height = 600, res = 130)
    ci_s <- confint(fit_strat)
    ci_s_dt <- as.data.table(ci_s, keep.rownames = TRUE)
    setnames(ci_s_dt, c("term", "ci_lower", "ci_upper"))
    plot_s <- merge(coef_strat[, .(event_time, estimate = Estimate)],
                    ci_s_dt[, .(term, ci_lower, ci_upper)],
                    by = "term", all.x = TRUE)[order(event_time)]
    plot(plot_s$event_time, plot_s$estimate, type = "n",
         xlim = range(plot_s$event_time),
         ylim = c(min(plot_s$ci_lower, 0, na.rm = TRUE),
                  max(plot_s$ci_upper, na.rm = TRUE)),
         xlab = "Event time", ylab = "IW coefficient",
         main = sprintf("Spillover stratum: %s (same-month openings)", strat))
    abline(h = 0, lty = 2, col = "grey50")
    abline(v = -1, lty = 3, col = "grey70")
    segments(plot_s$event_time, plot_s$ci_lower,
             plot_s$event_time, plot_s$ci_upper, col = "darkgreen", lwd = 2)
    points(plot_s$event_time, plot_s$estimate, pch = 19, col = "darkgreen")
    dev.off()
    cat(sprintf("Spillover stratum %s written (median=%d)\n", strat, med))
  }
} else {
  cat("Spillover heterogeneity skipped: too few treated with same-month size\n")
}
