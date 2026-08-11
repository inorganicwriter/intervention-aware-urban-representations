# Robustness smoke: covariate set variants (full vs no-transit).
# Verifies select_preonly_pairs accepts an explicit feature subset, which is
# how covariate-set sensitivity is exercised.
#
# Usage: Rscript robustness_covariate_smoke.R VARIANT  (full | no_transit)

suppressPackageStartupMessages({
  library(data.table)
  library(Matching)
})

args <- commandArgs(trailingOnly = TRUE)
variant <- if (length(args) >= 1L) args[[1L]] else "full"
root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
source(file.path(root, "scripts", "causal_r", "complete_estimators_lib.R"))

set.seed(1)
features_all <- c("housing_log_price__lag1", "housing_log_price__lag2",
                  "housing_log_price__lag3", "poi_count_log__lag1",
                  "poi_count_log__lag2", "poi_count_log__lag3",
                  "network_closeness", "dist_main_km")
features <- if (variant == "no_transit") {
  setdiff(features_all, c("network_closeness", "dist_main_km"))
} else {
  features_all
}
n <- 60L
frame <- data.table(
  unit_id = paste0("u", seq_len(n)),
  unit_key = paste0("u", seq_len(n)),
  grid_id = paste0("g", seq_len(n)),
  role = ifelse(seq_len(n) <= 6L, "treated", "donor"),
  Tr = ifelse(seq_len(n) <= 6L, 1L, 0L)
)
for (f in features_all) {
  frame[, (f) := rnorm(n, mean = if (grepl("dist_main", f)) 20 else 0,
                       sd = 1)]
}
prepared <- list(frame = frame, features = features_all)
fit <- select_preonly_pairs(prepared,
                            match_features = features)
cat(sprintf("variant=%s active features=%d matched pairs=%d\n",
            variant, length(fit$active_features), nrow(fit$pairs)))
stopifnot(nrow(fit$pairs) > 0L)
cat("OK covariate smoke\n")
