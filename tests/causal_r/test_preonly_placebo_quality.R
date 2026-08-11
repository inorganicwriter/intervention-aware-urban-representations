suppressPackageStartupMessages(library(data.table))
source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))

set.seed(20260723)
donors <- data.table(
  unit_id = 1:300, Tr = 0L,
  x__lag2 = rnorm(300), x__lag3 = rnorm(300), x__lag1 = rnorm(300)
)
target_good <- donors[100]
target_good[, `:=`(unit_id = 301L, Tr = 1L)]
frame_good <- rbind(donors, target_good)
prepared_good <- list(frame = frame_good, features = c("x__lag1", "x__lag2", "x__lag3"))
selected_good <- select_preonly_pairs(
  prepared_good, match_features = c("x__lag2", "x__lag3")
)
calibration <- calibrate_preonly_placebos(
  selected_good$frame, c("x__lag2", "x__lag3"), "x__lag1", sample_n = 100L
)
quality_good <- evaluate_preonly_pair_quality(
  selected_good$pairs[1L], selected_good$frame,
  c("x__lag2", "x__lag3"), "x__lag1", calibration
)
stopifnot(quality_good$accepted)

target_bad <- copy(target_good)
target_bad[, x__lag1 := 100]
frame_bad <- rbind(donors, target_bad)
prepared_bad <- list(frame = frame_bad, features = c("x__lag1", "x__lag2", "x__lag3"))
selected_bad <- select_preonly_pairs(
  prepared_bad, match_features = c("x__lag2", "x__lag3")
)
quality_bad <- evaluate_preonly_pair_quality(
  selected_bad$pairs[1L], selected_bad$frame,
  c("x__lag2", "x__lag3"), "x__lag1", calibration
)
stopifnot(!quality_bad$accepted)

cat("Donor-donor placebo calibration accepts a supported path and rejects held-out divergence.\n")
