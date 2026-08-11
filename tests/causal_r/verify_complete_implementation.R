suppressPackageStartupMessages({
  library(data.table)
  library(PanelMatch)
  library(Matching)
  library(gsynth)
})

source(file.path("scripts", "causal_r", "complete_estimators_lib.R"))
root <- project_root()

stopifnot(
  as.character(packageVersion("PanelMatch")) == "3.1.3",
  as.character(packageVersion("Matching")) == "4.10.15",
  as.character(packageVersion("gsynth")) == "1.4.0"
)

counterfactual_queue <- fread(file.path(CAUSAL_DIR, "counterfactual_work_queue.csv"))
family_queue <- fread(file.path(CAUSAL_DIR, "outcome_family_work_queue.csv"))
control_queue <- fread(file.path(CAUSAL_DIR, "control_design_queue.csv"))
stopifnot(
  nrow(counterfactual_queue) == 5048L,
  nrow(control_queue) == 5048L,
  nrow(family_queue) == 20192L,
  all(counterfactual_queue$status %in% c(
    "pending", "labelled", "partially_labelled", "skipped"
  )),
  all(family_queue$status %in% c(
    "pending", "gsc_pending", "gsc_running", "mc_pending", "mc_running",
    "matched_labelled", "gsc_labelled", "mc_labelled", "skipped"
  )),
  all(control_queue$status %in% c(
    "pending", "matched", "gsc_pending", "error"
  )),
  !any(control_queue$control_selection_uses_post_outcome),
  !any(counterfactual_queue$status %in% c("matching_running", "gsc_running")),
  !any(family_queue$status %in% c("matching_running", "gsc_running", "mc_running"))
)

ai_path <- file.path(
  root, "outputs", "complete_estimators", "staging", "abadie_imbens",
  "xiamen", "2019", "population_log_h1", "poi+population+viirs"
)
ai <- readRDS(file.path(ai_path, "matching_object.rds"))
ai_manifest <- fread(file.path(ai_path, "manifest.csv"))
stopifnot(
  is.finite(ai$est), is.finite(ai$se.standard), is.finite(ai$est.noadj),
  nrow(ai_manifest[field == "BiasAdjust" & value == "TRUE"]) == 1L,
  nrow(ai_manifest[field == "Var_calc" & value == "1"]) == 1L,
  nrow(ai_manifest[field == "ordinary_estimator_bootstrap_used" & value == "FALSE"]) == 1L
)

gsc_path <- file.path(
  root, "outputs", "complete_estimators", "staging", "xu_gsc",
  "xiamen", "2019", "population_log", "population+viirs"
)
gsc_manifest <- fread(file.path(gsc_path, "manifest.csv"))
gsc_diagnostics <- fread(file.path(gsc_path, "diagnostics.csv"))
stopifnot(
  nrow(gsc_manifest[field == "factor_candidates" & value == "0;1;2;3;4;5"]) == 1L,
  nrow(gsc_manifest[field == "nboots" & value == "200"]) == 1L,
  nrow(gsc_manifest[field == "donor_cap" & value == "none"]) == 1L,
  gsc_diagnostics$pre_treatment_periods[[1L]] >= 5L,
  gsc_diagnostics$complete_donors[[1L]] == 16514L,
  file.size(file.path(gsc_path, "counterfactual_paths.csv")) > 0,
  file.size(file.path(gsc_path, "gsynth_object.rds")) > 0
)

gate <- fread(file.path(
  root, "outputs", "complete_estimators", "validation",
  "real_panelmatch_gate", "gate_manifest.csv"
))
stopifnot(
  gate$status[[1L]] == "passed", !gate$formal_estimate[[1L]],
  gate$production_donors[[1L]] == 16514L,
  gate$test_fixture_donors[[1L]] == 300L
)

cat("Complete-estimator implementation audit passed; queue states are valid and resumable.\n")
