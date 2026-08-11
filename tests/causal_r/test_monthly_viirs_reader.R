suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})
source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

root <- tempfile("viirs-reader-")
path <- file.path(
  root, "data", "active", "curated", "viirs", "monthly",
  "city_key=test_city", "year=2020", "month=01"
)
dir.create(path, recursive = TRUE)
write_parquet(data.table(
  grid_id = c("g1", "g2"), avg_rad = c(-0.5, 2),
  valid_days_mean = c(20, 21), source_point_count = c(1L, 2L)
), file.path(path, "part.parquet"))

x <- read_city_monthly_viirs("test_city", as.IDate("2020-01-01"), root)
stopifnot(
  nrow(x) == 2L,
  isTRUE(all.equal(x[grid_id == "g1", viirs_avg_asinh], asinh(-0.5))),
  x[grid_id == "g1", viirs_avg_asinh] < 0,
  identical(x$viirs_source_point_count, c(1L, 2L))
)

unlink(root, recursive = TRUE)
cat("Monthly VIIRS reader preserves negative radiance with asinh transform.\n")
