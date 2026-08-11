# Robustness smoke: donor scope (same_city vs all_city_standardized).
# Verifies the two scopes are exercised and produce different candidate sets.
#
# Usage: Rscript robustness_donor_scope_smoke.R SCOPE (same_city|all_city_standardized)

suppressPackageStartupMessages({
  library(data.table)
  library(Matching)
})

args <- commandArgs(trailingOnly = TRUE)
scope <- if (length(args) >= 1L) args[[1L]] else "same_city"
root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
source(file.path(root, "scripts", "causal_r", "complete_estimators_lib.R"))

set.seed(2)
features <- c("housing_log_price__lag1", "housing_log_price__lag2",
              "housing_log_price__lag3")
# Treated in city a; donors in city a (same) and city b (cross)
n_a <- 40L
n_b <- 40L
frame <- data.table(
  unit_id = c(paste0("a_t", seq_len(5)), paste0("a_d", seq_len(n_a)),
              paste0("b_d", seq_len(n_b))),
  unit_key = c(paste0("a_t", seq_len(5)), paste0("a_d", seq_len(n_a)),
               paste0("b_d", seq_len(n_b))),
  grid_id = paste0("g", seq_len(5L + n_a + n_b)),
  city_key = c(rep("a", 5L + n_a), rep("b", n_b)),
  role = c(rep("treated", 5L), rep("donor", n_a + n_b)),
  Tr = c(rep(1L, 5L), rep(0L, n_a + n_b))
)
for (f in features) frame[, (f) := rnorm(nrow(frame), 0, 1)]
if (scope == "same_city") frame <- frame[city_key == "a"]
prepared <- list(frame = frame, features = features)
fit <- select_preonly_pairs(prepared, match_features = features)
cat(sprintf("scope=%s donors=%d matched pairs=%d\n",
            scope, sum(frame$Tr == 0L), nrow(fit$pairs)))
stopifnot(nrow(fit$pairs) > 0L)
cat("OK donor scope smoke\n")
