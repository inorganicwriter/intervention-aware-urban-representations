suppressPackageStartupMessages(library(data.table))
source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

design <- build_monthly_housing_estimator_panel(
  "xiamen", "2019-12", "auto", leads = 1:24
)
prepared <- make_preonly_matching_frame(design)
stopifnot(
  design$lag == 36L,
  # Production monthly spec (complete_estimators_lib.R) uses 13/25/37:
  # covariates at annual spacing with a 6-month anticipation buffer.
  identical(design$covariate_lags, c(13L, 25L, 37L)),
  identical(design$leads, 1:24),
  all(design$covariates %in% names(design$panel)),
  !any(grepl("\\.[xy]$", names(design$panel))),
  nrow(design$treated) == 26L,
  nrow(design$donors) == 16514L,
  nrow(design$panel) == nrow(design$unit_map) * 60L,
  design$event_calendar$clean_pre_end == as.IDate("2019-05-01"),
  design$event_calendar$first_treated_month == as.IDate("2020-01-01"),
  !as.IDate("2019-12-01") %in% design$panel$month,
  all(design$panel[role == "treated" & time_id <= 36L, D] == 0L),
  all(design$panel[role == "treated" & time_id > 36L, D] == 1L),
  all(design$panel[role == "donor", D] == 0L)
  , any(prepared$frame$Tr == 1L)
  , identical(sort(unique(sub("^.*__lag", "", prepared$features))), c("1", "2", "3"))
)

cat("Monthly treatment timing and lag construction passed real-data tests.\n")
