suppressPackageStartupMessages(library(data.table))
source(file.path("scripts", "causal_r", "causal_label_lib.R"))

labels <- new_label_rows(
  treatment_order = 1L, city_key = "city", grid_id = "grid",
  opening_month = "2020-01", outcome_family = "housing",
  outcome = "housing_log_price", event_time = c(1L, 3L),
  observed = c(5, NA), counterfactual = c(3.5, 4), method = "matched_change"
)
stopifnot(
  labels[event_time == 1L, causal_response_label] == 1.5,
  labels[event_time == 1L, label_available],
  !labels[event_time == 3L, label_available],
  !is.na(labels[event_time == 3L, failure_reason])
)
validate_causal_labels(labels)

queue <- data.table(
  treatment_order = 1L, outcome_family = "housing", status = "pending",
  selected_method = NA_character_, failure_reason = NA_character_
)
queue <- transition_queue_row(queue, 1L, "housing", "matching_running")
queue <- transition_queue_row(
  queue, 1L, "housing", "gsc_pending",
  failure_reason = "no_credible_single_control"
)
queue <- transition_queue_row(queue, 1L, "housing", "gsc_running")
queue <- transition_queue_row(
  queue, 1L, "housing", "gsc_labelled", selected_method = "xu_2017_gsynth"
)
stopifnot(queue$status == "gsc_labelled")
stopifnot(inherits(try(
  transition_queue_row(queue, 1L, "housing", "matching_running"), silent = TRUE
), "try-error"))

mc_queue <- data.table(
  treatment_order = 2L, outcome_family = "population", status = "gsc_running",
  selected_method = NA_character_, failure_reason = NA_character_
)
mc_queue <- transition_queue_row(
  mc_queue, 2L, "population", "mc_pending",
  failure_reason = "xu_gsc_failed"
)
mc_queue <- transition_queue_row(mc_queue, 2L, "population", "mc_running")
mc_queue <- transition_queue_row(
  mc_queue, 2L, "population", "mc_labelled",
  selected_method = "athey_2021_mc_same_city"
)
stopifnot(mc_queue$status == "mc_labelled")

path <- tempfile(fileext = ".csv")
atomic_fwrite(queue, path)
stopifnot(file.exists(path), fread(path)$status == "gsc_labelled")
unlink(path)

cat("Causal response label schema, formula, routing, and atomic queue write passed.\n")
