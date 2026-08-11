formal_matching_spec <- function() {
  list(
    schema = "formal_counterfactual_design_v1",
    treatment_time = paste(
      "opening month/year excluded as partial exposure; first full month/year",
      "after opening is event_time 1"
    ),
    treatment_history_lag_years = 3L,
    pre_year_lags = 1:3,
    minimum_complete_families = 1L,
    matching_with_replacement = TRUE,
    matches_per_treated = 1L,
    distance = paste(
      "PanelMatch period-specific full-covariance Mahalanobis distances",
      "averaged over lag years; Moore-Penrose inverse when singular"
    ),
    common_support = "treated value inside closed donor range for every active feature",
    balance_diagnostic = "absolute mean standardized difference <= 0.10",
    balance_threshold = 0.10,
    balance_is_unit_selection_rule = FALSE,
    post_treatment_data_used_for_matching = FALSE,
    donor_rule = paste(
      "nonexperimental grid; eligible_spatial_donor; no known station",
      "contamination; identical untreated lag history"
    ),
    citations = c(
      panelmatch = paste(
        "Imai, Kim, and Wang (2023), Matching Methods for Causal Inference",
        "with Time-Series Cross-Sectional Data, AJPS 67(3):587-605"
      ),
      nearest_neighbor = paste(
        "Abadie and Imbens (2006), Large Sample Properties of Matching",
        "Estimators for Average Treatment Effects, Econometrica 74(1):235-267"
      ),
      generalized_synthetic_control = paste(
        "Xu (2017), Generalized Synthetic Control Method,",
        "Political Analysis 25(1):57-76"
      )
    ),
    families = list(
      housing = c("housing_log_price"),
      poi = c(
        "poi_count_log", "poi_category_entropy", "poi_commercial_share",
        "poi_transport_access_log"
      ),
      viirs = c("viirs_avg_asinh"),
      population = c("population_log")
    ),
    gsc = list(
      estimator = "gsynth interactive fixed effects",
      force = "two-way",
      factor_candidates = 0:5,
      factor_selection = "pre-treatment cross-validation",
      minimum_pre_periods = 5L,
      standard_errors = TRUE,
      inference = "parametric",
      bootstrap_replications = 200L,
      minimum_complete_donors = 20L,
      maximum_complete_donors = Inf,
      donor_screen = "none in the main specification; all pre-only admissible donors"
    )
  )
}

assert_numeric_matrix <- function(x, label) {
  if (!is.matrix(x) || !is.numeric(x)) {
    stop(label, " must be a numeric matrix")
  }
  if (nrow(x) == 0L || ncol(x) == 0L) {
    stop(label, " must have at least one row and one column")
  }
  if (any(!is.finite(x))) {
    stop(label, " contains non-finite values")
  }
  invisible(TRUE)
}

moore_penrose_inverse <- function(x, tolerance = sqrt(.Machine$double.eps)) {
  if (!is.matrix(x) || nrow(x) != ncol(x)) {
    stop("x must be a square matrix")
  }
  eig <- eigen((x + t(x)) / 2, symmetric = TRUE)
  cutoff <- max(abs(eig$values)) * tolerance
  keep <- eig$values > cutoff
  if (!any(keep)) stop("covariance matrix has zero numerical rank")
  vectors <- eig$vectors[, keep, drop = FALSE]
  inverse <- vectors %*% diag(1 / eig$values[keep], nrow = sum(keep)) %*% t(vectors)
  list(inverse = inverse, rank = sum(keep), eigenvalues = eig$values)
}

check_closed_range_support <- function(target, controls) {
  assert_numeric_matrix(controls, "controls")
  target <- as.numeric(target)
  if (length(target) != ncol(controls) || any(!is.finite(target))) {
    stop("target is incompatible with controls")
  }
  lower <- apply(controls, 2L, min)
  upper <- apply(controls, 2L, max)
  inside <- target >= lower & target <= upper
  list(
    supported = all(inside),
    inside = inside,
    lower = lower,
    upper = upper
  )
}

prepare_abadie_imbens_controls <- function(controls, control_ids) {
  assert_numeric_matrix(controls, "controls")
  if (length(control_ids) != nrow(controls)) {
    stop("control_ids length does not match controls")
  }
  feature_names <- colnames(controls)
  if (is.null(feature_names)) {
    feature_names <- paste0("feature", seq_len(ncol(controls)), "__lag1")
    colnames(controls) <- feature_names
  }
  lag_labels <- sub("^.*__", "", feature_names)
  lag_blocks <- lapply(unique(lag_labels), function(lag_label) {
    columns <- which(lag_labels == lag_label)
    covariance <- stats::cov(controls[, columns, drop = FALSE])
    if (length(columns) == 1L) covariance <- matrix(covariance, 1L, 1L)
    inverse <- moore_penrose_inverse(covariance)
    list(
      lag = lag_label,
      columns = columns,
      inverse_covariance = inverse$inverse,
      rank = inverse$rank,
      dimension = length(columns)
    )
  })
  list(
    controls = controls,
    control_ids = control_ids,
    lag_blocks = lag_blocks,
    covariance_rank = sum(vapply(lag_blocks, `[[`, integer(1), "rank")),
    covariance_dimension = ncol(controls),
    control_sd = apply(controls, 2L, stats::sd),
    lower = apply(controls, 2L, min),
    upper = apply(controls, 2L, max)
  )
}

abadie_imbens_nearest_prepared <- function(target, prepared, matches = 1L) {
  controls <- prepared$controls
  control_ids <- prepared$control_ids
  target <- as.numeric(target)
  if (length(target) != ncol(controls) || any(!is.finite(target))) {
    stop("target is incompatible with controls")
  }
  if (matches < 1L || matches > nrow(controls)) {
    stop("invalid number of matches")
  }
  delta <- sweep(controls, 2L, target, "-")
  lag_distance <- vapply(prepared$lag_blocks, function(block) {
    block_delta <- delta[, block$columns, drop = FALSE]
    squared <- rowSums(
      (block_delta %*% block$inverse_covariance) * block_delta
    )
    sqrt(pmax(squared, 0))
  }, numeric(nrow(controls)))
  if (is.null(dim(lag_distance))) lag_distance <- matrix(lag_distance, ncol = 1L)
  distance <- rowMeans(lag_distance)
  order_index <- order(distance, control_ids, method = "radix")
  selected <- head(order_index, matches)
  standardized_gap <- (target - controls[selected[1L], ]) / prepared$control_sd
  standardized_gap[!is.finite(standardized_gap)] <- NA_real_
  list(
    selected_ids = control_ids[selected],
    selected_rows = selected,
    distance = distance[selected],
    covariance_rank = prepared$covariance_rank,
    covariance_dimension = prepared$covariance_dimension,
    standardized_gap = standardized_gap,
    control_sd = prepared$control_sd
  )
}

abadie_imbens_nearest <- function(target, controls, control_ids, matches = 1L) {
  prepared <- prepare_abadie_imbens_controls(controls, control_ids)
  abadie_imbens_nearest_prepared(target, prepared, matches)
}

summarize_standardized_balance <- function(contributions, threshold = 0.10) {
  required <- c("feature", "standardized_gap")
  if (!all(required %in% names(contributions))) {
    stop("balance contributions are missing required columns")
  }
  result <- contributions[
    !is.na(standardized_gap),
    .(
      matched_pairs = .N,
      mean_standardized_difference = mean(standardized_gap),
      mean_absolute_pair_gap = mean(abs(standardized_gap))
    ),
    by = feature
  ]
  result[, passes_0_10_diagnostic := abs(mean_standardized_difference) <= threshold]
  result[]
}
