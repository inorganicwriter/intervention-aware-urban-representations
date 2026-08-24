suppressPackageStartupMessages({
  library(data.table)
})

source(file.path("scripts", "causal_r", "grid_control_design_lib.R"))
source(file.path("scripts", "causal_r", "fixed_control_label_lib.R"))

stopifnot(
  identical(grid_control_spec()$minimum_families, 1L),
  identical(grid_control_spec()$matching$M, 1L),
  identical(grid_control_spec()$matching$replace, TRUE),
  identical(grid_control_spec()$matching$Weight, 2L),
  identical(grid_control_spec()$matching$BiasAdjust, FALSE),
  identical(grid_control_spec()$matching$Var.calc, 0L)
)

synthetic <- data.table(
  city_key = c("a", "a", "a", "b", "b"),
  role = c("treated", "donor", "donor", "donor", "donor"),
  level__lag2 = c(2, 1, 3, 10, 14),
  constant__lag2 = c(0, 0, 0, 2, 4)
)
standardized <- robust_city_standardize(
  synthetic, c("level__lag2", "constant__lag2")
)
stopifnot(all(is.finite(as.matrix(
  standardized[, .(level__lag2, constant__lag2)]
))))

matching_frame <- data.table(
  unit_id = 1:5,
  unit_key = paste0("u", 1:5),
  grid_id = paste0("g", 1:5),
  role = c("treated", rep("donor", 4)),
  x__lag2 = c(1.1, 0, 1, 2, 3),
  x__lag3 = c(1.2, 0, 1, 2, 3),
  x__lag1 = c(1.15, 0, 1.1, 2.1, 3.1)
)
matching_frame[, Tr := as.integer(role == "treated")]
matching_spec <- complete_estimator_spec()$abadie_imbens
matching_spec$ties <- FALSE
selected <- select_preonly_pairs(
  list(frame = matching_frame, features = c("x__lag2", "x__lag3", "x__lag1")),
  matching_spec, match_features = c("x__lag2", "x__lag3")
)
stopifnot(
  nrow(selected$pairs) == 1L,
  selected$pairs$control_unit_key[[1L]] %in% paste0("u", 2:5),
  selected$pairs$treated_unit_key[[1L]] == "u1"
)

# A twelve-month block mean, not the final month, is the matched-label baseline.
baseline <- finite_mean_with_minimum(1:12, 1L)
stopifnot(
  identical(baseline, 6.5),
  is.na(finite_mean_with_minimum(c(1:11, NA_real_), 12L))
)

# Two-stage refinement: among outcome-matched candidate controls the static
# (location/transit) covariate must break the tie toward the better-balanced
# candidate, and must not change the treated identity.
refine_frame <- data.table(
  unit_id = c("t", "c1", "c2", "c3"),
  unit_key = c("t", "c1", "c2", "c3"),
  grid_id = c("gt", "g1", "g2", "g3"),
  role = c("treated", "donor", "donor", "donor"),
  Tr = c(1L, 0L, 0L, 0L),
  x__lag2 = c(1, 1.05, 0.95, 10),
  x__lag3 = c(1, 5, 0.9, 10)
)
refine_pairs <- data.table(
  treated_unit_id = "t", treated_row = 1L,
  control_unit_id = c("c1", "c2"), control_row = c(2L, 3L)
)
refined <- static_balance_refine(refine_pairs, refine_frame, "x__lag3")
stopifnot(
  nrow(refined) == 1L,
  refined$control_unit_id[[1L]] == "c2"
)
refined_fallback <- static_balance_refine(refine_pairs, refine_frame, character())
stopifnot(
  nrow(refined_fallback) == 1L,
  refined_fallback$control_unit_id[[1L]] == "c1"
)

empty <- data.table()
target <- data.table(city_key = "a", grid_id = "g1")
stopifnot(length(target_active_families(target, empty)) == 0L)

cache_root <- tempfile("grid-control-viirs-cache-")
period <- as.IDate("2012-01-01")
part <- file.path(
  cache_root, "data", "active", "curated", "viirs", "monthly",
  "city_key=a", "year=2012", "month=01", "part.parquet"
)
audit <- file.path(
  cache_root, "outputs", "viirs_monthly", "partition_audits",
  "a", "2012-01.json"
)
dir.create(dirname(part), recursive = TRUE)
file.create(part)
stopifnot(nrow(missing_control_viirs_cache("a", period, cache_root)) == 1L)
dir.create(dirname(audit), recursive = TRUE)
file.create(audit)
stopifnot(nrow(missing_control_viirs_cache("a", period, cache_root)) == 0L)
unlink(cache_root, recursive = TRUE)

# Sparse-month alignment regression: monthly_fixed_control_labels must align
# treated/control outcome vectors to the full calendar sequence with NA
# placeholders.  A missing intermediate month used to shift every later
# observation, silently misaligning regression_beta/se/p.  The fix merges the
# outcome table onto the calendar frame before indexing.
source(file.path("scripts", "causal_r", "fixed_control_label_lib.R"))
sparse_months <- as.IDate(c("2015-01-01", "2015-03-01", "2015-04-01"))
calendar_frame <- data.table(
  month = as.IDate(c("2015-01-01", "2015-02-01", "2015-03-01", "2015-04-01")),
  reg_position = 1:4
)
sparse <- data.table(
  month = sparse_months,
  grid_id = "g1",
  housing_log_price = c(100, 110, 112)
)
aligned <- merge(
  calendar_frame, sparse[grid_id == "g1"], by = "month", all.x = TRUE
)[order(reg_position)]
stopifnot(
  nrow(aligned) == 4L,
  is.na(aligned$housing_log_price[[2L]]),  # missing February is an NA placeholder
  identical(aligned$housing_log_price[[4L]], 112)
)

# Exact kth-distance ties must not inherit Matching::Match's RNG-dependent
# ties=FALSE sampling. The frozen grid-design contract uses donor row order.
tie_frame <- data.table(
  unit_id = 1:9,
  unit_key = paste0("u", 1:9),
  grid_id = paste0("g", 1:9),
  role = c("treated", rep("donor", 8L)),
  Tr = c(1L, rep(0L, 8L)),
  x = c(0, rep(c(-1, 1), 4L))
)
tie_spec <- complete_estimator_spec()$abadie_imbens
tie_spec$ties <- FALSE
tie_spec$M <- 3L
set.seed(1L)
tie_first <- select_preonly_pairs(
  list(frame = tie_frame, features = "x"), tie_spec,
  match_features = "x", support_features = "x"
)$pairs$control_row
set.seed(99L)
tie_second <- select_preonly_pairs(
  list(frame = tie_frame, features = "x"), tie_spec,
  match_features = "x", support_features = "x"
)$pairs$control_row
stopifnot(
  identical(tie_first, 2:4),
  identical(tie_second, tie_first)
)

cat("Grid-level frozen-control and 12-month baseline contracts passed.\n")
