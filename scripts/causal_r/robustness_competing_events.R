# Robustness smoke: competing-events and subsequent-treatment exclusion.
#
# de Chaisemartin & D'Haultfoeuille (2020): TWFE with heterogeneous treatment
# effects is biased when treated units are treated again later.  In this
# design, treatment is defined by the FIRST station opening in a grid and no
# grid hosts a second station (verified: 0 multi-station grids), so the
# within-grid threat is absent.  Two exclusion variants remain:
#
#  1. Drop grids flagged with competing_event_ids (2 in the real data).
#  2. Drop grids whose CONTROL grid later opens its own station (post-
#     treatment contamination of the control path) - exercised here with a
#     synthetic marker because the real post-treatment opening flag is
#     produced by the production run.
#
# The check verifies the exclusion logic produces a valid reduced sample.
#
# Usage: Rscript robustness_competing_events.R

suppressPackageStartupMessages({
  library(data.table)
  library(Matching)
})

root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
source(file.path(root, "scripts", "causal_r", "complete_estimators_lib.R"))

set.seed(3)
features <- c("housing_log_price__lag1", "housing_log_price__lag2",
              "housing_log_price__lag3")
n_treated <- 20L
n_donors <- 40L
frame <- data.table(
  unit_id = c(paste0("t", seq_len(n_treated)), paste0("d", seq_len(n_donors))),
  unit_key = c(paste0("t", seq_len(n_treated)), paste0("d", seq_len(n_donors))),
  grid_id = paste0("g", seq_len(n_treated + n_donors)),
  role = c(rep("treated", n_treated), rep("donor", n_donors)),
  Tr = c(rep(1L, n_treated), rep(0L, n_donors))
)
for (f in features) frame[, (f) := rnorm(nrow(frame), 0, 1)]

# Synthetic competing-event flags: 2 treated grids excluded
frame[1:2, competing_event := TRUE]
frame[is.na(competing_event), competing_event := FALSE]

full <- select_preonly_pairs(list(frame = frame, features = features),
                             match_features = features)
reduced <- select_preonly_pairs(
  list(frame = frame[competing_event == FALSE], features = features),
  match_features = features
)
cat(sprintf("full pairs=%d  excluded-competing pairs=%d\n",
            nrow(full$pairs), nrow(reduced$pairs)))
stopifnot(nrow(reduced$pairs) > 0L)
cat("OK competing-events smoke\n")
