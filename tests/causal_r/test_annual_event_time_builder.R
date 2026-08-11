suppressPackageStartupMessages(library(data.table))
source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

design <- build_annual_estimator_panel(
  "xiamen", 2019L, "population", "population+viirs",
  leads = 1:3, treatment_order = 2370L
)
treated <- design$panel[role == "treated"]
stopifnot(
  identical(sort(unique(design$panel$year)), c(2016L, 2017L, 2018L, 2020L, 2021L, 2022L)),
  !2019L %in% design$panel$year,
  identical(treated[D == 1L, year], 2020:2022),
  identical(treated[D == 1L, time_id], 4:6)
)

# Every requested horizon must map to an actually present post period.
for (horizon in 1:3) {
  frame <- make_preonly_matching_frame(design)
  outcome <- attach_prepost_outcome(frame$frame, design, "population_log", horizon)
  stopifnot(nrow(outcome) == nrow(frame$frame))
}

cat("Annual opening-year exclusion and event horizons 1-3 passed.\n")
