suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

causal_label_spec <- function() {
  list(
    schema = "causal_response_labels_v1",
    specification_id = "main_a6_r1km",
    monthly_horizons = c(1L, 3L, 6L, 12L, 18L, 24L),
    annual_horizons = 1:3,
    methods = c("matched_change", "xu_2017_gsynth", "athey_2021_mc"),
    valid_statuses = c(
      "pending", "matching_running", "matched_labelled", "gsc_pending",
      "gsc_running", "gsc_labelled", "mc_pending", "mc_running",
      "mc_labelled", "skipped"
    ),
    terminal_statuses = c("matched_labelled", "gsc_labelled", "mc_labelled", "skipped")
  )
}

label_key_columns <- function() {
  c("treatment_order", "outcome_family", "outcome", "event_time", "specification_id")
}

new_label_rows <- function(
    treatment_order, city_key, grid_id, opening_month, outcome_family,
    outcome, event_time, observed, counterfactual, method,
    specification_id = causal_label_spec()$specification_id,
    transformed_scale = TRUE) {
  x <- data.table(
    treatment_order = as.integer(treatment_order),
    city_key = as.character(city_key),
    grid_id = as.character(grid_id),
    opening_month = as.character(opening_month),
    outcome_family = as.character(outcome_family),
    outcome = as.character(outcome),
    event_time = as.integer(event_time),
    specification_id = as.character(specification_id),
    observed = as.numeric(observed),
    counterfactual = as.numeric(counterfactual),
    causal_response_label = as.numeric(observed) - as.numeric(counterfactual),
    transformed_scale = as.logical(transformed_scale),
    method = as.character(method)
  )
  x[, `:=`(
    label_available = is.finite(observed) & is.finite(counterfactual),
    standard_error = NA_real_,
    confidence_lower = NA_real_,
    confidence_upper = NA_real_,
    quality_grade = NA_character_,
    failure_reason = fifelse(
      is.finite(observed) & is.finite(counterfactual), NA_character_,
      "target_period_outcome_or_counterfactual_missing"
    )
  )]
  x[]
}

validate_causal_labels <- function(labels) {
  keys <- label_key_columns()
  required <- c(
    keys, "city_key", "grid_id", "opening_month", "observed",
    "counterfactual", "causal_response_label", "method", "label_available",
    "failure_reason"
  )
  missing <- setdiff(required, names(labels))
  if (length(missing)) stop("Label table is missing columns: ", paste(missing, collapse = ", "))
  if (anyDuplicated(labels[, ..keys])) stop("Causal label primary key is not unique")
  if (!all(labels$method %in% causal_label_spec()$methods)) stop("Unknown causal label method")
  available <- labels$label_available
  if (any(available & !is.finite(labels$causal_response_label))) {
    stop("Available labels must be finite")
  }
  expected <- labels$observed - labels$counterfactual
  if (any(available & abs(labels$causal_response_label - expected) > 1e-10)) {
    stop("Causal label is not observed minus counterfactual")
  }
  if (any(!available & is.na(labels$failure_reason))) {
    stop("Unavailable labels require a failure reason")
  }
  invisible(TRUE)
}

allowed_queue_transition <- function(from, to) {
  edges <- list(
    pending = c("matching_running"),
    matching_running = c("matched_labelled", "gsc_pending", "skipped"),
    gsc_pending = c("gsc_running"),
    gsc_running = c("gsc_labelled", "mc_pending", "skipped"),
    mc_pending = c("mc_running"),
    mc_running = c("mc_labelled", "skipped")
  )
  identical(from, to) || to %in% edges[[from]]
}

transition_queue_row <- function(queue, treatment_order, outcome_family, to,
                                 selected_method = NA_character_,
                                 failure_reason = NA_character_) {
  spec <- causal_label_spec()
  if (!to %in% spec$valid_statuses) stop("Unknown queue status: ", to)
  index <- which(
    queue$treatment_order == treatment_order & queue$outcome_family == outcome_family
  )
  if (length(index) != 1L) stop("Queue transition must target exactly one family row")
  from <- queue$status[index]
  if (!allowed_queue_transition(from, to)) {
    stop("Illegal queue transition: ", from, " -> ", to)
  }
  queue[index, `:=`(
    status = to,
    selected_method = selected_method,
    failure_reason = failure_reason
  )]
  queue
}

atomic_fwrite <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp")
  fwrite(x, temporary, bom = TRUE)
  for (attempt in 1:5) {
    if (file.exists(path)) {
      if (file.remove(path)) break
      Sys.sleep(0.2 * attempt)
    } else break
  }
  if (file.exists(path) && !file.remove(path)) stop("Cannot replace queue: ", path)
  if (!file.rename(temporary, path)) stop("Atomic queue rename failed: ", path)
  invisible(path)
}

atomic_write_parquet <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp.parquet")
  write_parquet(x, temporary, compression = "zstd")
  for (attempt in 1:5) {
    if (file.exists(path)) {
      if (file.remove(path)) break
      Sys.sleep(0.2 * attempt)
    } else break
  }
  if (file.exists(path) && !file.remove(path)) stop("Cannot replace labels: ", path)
  if (!file.rename(temporary, path)) stop("Atomic label rename failed: ", path)
  invisible(path)
}
