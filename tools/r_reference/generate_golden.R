# Generate golden reference outputs from the original GLEAM R package.
#
# Usage (from the gleam-py repository root):
#   Rscript tools/r_reference/generate_golden.R <path-to-GLEAM-R-source> [out_dir] [case_regex]
#
# Runs every run_*() example shipped with the R package and writes each
# resulting table to <out_dir>/<case>/<table>.csv with numeric columns written
# at 17 significant digits, so the Python port can be checked for numerical
# parity. A case that errors in R is recorded in <out_dir>/<case>/ERROR.txt.

args <- commandArgs(trailingOnly = TRUE)
r_src <- if (length(args) >= 1) args[[1]] else "../GLEAM"
out_dir <- if (length(args) >= 2) args[[2]] else "tests/golden"
case_regex <- if (length(args) >= 3) args[[3]] else "."

suppressMessages(pkgload::load_all(r_src, quiet = TRUE, export_all = TRUE))
library(data.table)

ex_mod <- file.path(r_src, "inst/extdata/run_modules_examples")
ex_run <- file.path(r_src, "inst/extdata/run_gleam_examples")
rd <- function(dir, name) data.table::fread(file.path(dir, name))

write_table <- function(dt, path) {
  dt <- data.table::copy(data.table::as.data.table(dt))
  for (col in names(dt)) {
    v <- dt[[col]]
    if (is.double(v)) {
      s <- sprintf("%.17g", v)
      s[is.na(v)] <- NA_character_
      data.table::set(dt, j = col, value = s)
    }
  }
  data.table::fwrite(dt, path, na = "NA")
}

write_result <- function(case, result) {
  dir <- file.path(out_dir, case)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  unlink(list.files(dir, full.names = TRUE))
  flat <- list()
  flatten <- function(x, prefix) {
    if (data.table::is.data.table(x) || is.data.frame(x)) {
      flat[[prefix]] <<- x
    } else if (is.list(x)) {
      for (n in names(x)) flatten(x[[n]], if (prefix == "") n else paste0(prefix, "__", n))
    }
  }
  flatten(result, "")
  for (n in names(flat)) write_table(flat[[n]], file.path(dir, paste0(n, ".csv")))
  cat(sprintf("  %-55s %s\n", case, paste(names(flat), collapse = ", ")))
}

run_case <- function(case, expr) {
  if (!grepl(case_regex, case)) return(invisible(NULL))
  res <- tryCatch(expr, error = function(e) e)
  if (inherits(res, "error")) {
    dir <- file.path(out_dir, case)
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
    unlink(list.files(dir, full.names = TRUE))
    writeLines(conditionMessage(res), file.path(dir, "ERROR.txt"))
    cat(sprintf("  %-55s ERROR: %s\n", case, conditionMessage(res)))
  } else {
    write_result(case, res)
  }
}

q <- FALSE  # show_indicator

cat("Module examples\n")
run_case("weights_module", run_weights_module(
  rd(ex_mod, "weights_input_chrt_data.csv"), rd(ex_mod, "weights_input_hrd_data.csv"),
  show_indicator = q
))
run_case("ration_quality_module", list(result = run_ration_quality_module(
  rations_share = rd(ex_mod, "feed_rations_share_chrt.csv"),
  feed_params = rd(ex_mod, "feed_quality.csv"), show_indicator = q
)))
run_case("emissions_ration_module", list(result = run_emissions_ration_module(
  rations_share = rd(ex_mod, "feed_rations_share_chrt.csv"),
  feed_emissions = rd(ex_mod, "feed_emission_factors.csv"), show_indicator = q
)))
run_case("metabolic_energy_req_module", list(result = run_metabolic_energy_req_module(
  cohort_level_data = rd(ex_mod, "metabolic_energy_req_input_chrt_data.csv"),
  herd_level_data = rd(ex_mod, "metabolic_energy_req_input_hrd_data.csv"), show_indicator = q
)))
run_case("emissions_enteric_module", list(result = run_emissions_enteric_module(
  cohort_level_data = rd(ex_mod, "emissions_enteric_input_chrt_data.csv"), show_indicator = q
)))
run_case("nitrogen_balance_module", list(result = run_nitrogen_balance_module(
  cohort_level_data = rd(ex_mod, "nitrogen_balance_input_chrt_data.csv"),
  herd_level_data = rd(ex_mod, "nitrogen_balance_input_hrd_data.csv"), show_indicator = q
)))
run_case("emissions_manure_module", list(result = run_emissions_manure_module(
  cohort_level_data = rd(ex_mod, "emissions_manure_input_chrt_data.csv"),
  manure_management_system_fraction = rd(ex_mod, "manure_management_system_fraction.csv"),
  manure_management_system_factors = rd(ex_mod, "manure_management_system_factors.csv"),
  show_indicator = q
)))
run_case("production_module", list(result = run_production_module(
  cohort_level_data = rd(ex_mod, "production_input_chrt_data.csv"),
  herd_level_data = rd(ex_mod, "production_input_hrd_data.csv"),
  simulation_duration = 365, show_indicator = q
)))
run_case("allocation_module", run_allocation_module(
  cohort_level_data = rd(ex_mod, "allocation_input_chrt_data.csv"),
  herd_level_data = rd(ex_mod, "allocation_input_hrd_data.csv"), show_indicator = q
))
run_case("aggregation_module", run_aggregation_module(
  cohort_level_data = rd(ex_mod, "aggregation_input_chrt_data.csv"),
  allocation_herd_long = rd(ex_mod, "aggregation_allocation_input_data.csv"),
  simulation_duration = 365, global_warming_potential_set = "AR6", show_indicator = q
))
for (gwp in c("AR4", "AR5_excluding_carbon_feedback", "AR5_including_carbon_feedback")) {
  run_case(paste0("aggregation_module_", gwp), run_aggregation_module(
    cohort_level_data = rd(ex_mod, "aggregation_input_chrt_data.csv"),
    allocation_herd_long = rd(ex_mod, "aggregation_allocation_input_data.csv"),
    simulation_duration = 365, global_warming_potential_set = gwp, show_indicator = q
  ))
}
run_case("demographic_herd_module", run_demographic_herd_module(
  cohort_level_data = rd(ex_mod, "herd_simulation_input_chrt_data.csv"),
  herd_level_data = rd(ex_mod, "herd_simulation_input_hrd_data.csv"),
  simulation_duration = 365, show_indicator = q
))
run_case("nondemographic_herd_module", run_nondemographic_herd_module(
  cohort_level_data = rd(ex_mod, "nondemographic_herd_input_chrt_data.csv"),
  herd_level_data = rd(ex_mod, "nondemographic_herd_input_hrd_data.csv"),
  simulation_duration = 365, show_indicator = q
))
run_case("all_herd_module_demographic", run_all_herd_module(
  cohort_level_data = rd(ex_mod, "herd_simulation_input_chrt_data.csv"),
  herd_level_data = rd(ex_mod, "herd_simulation_input_hrd_data.csv"),
  run_demographic = TRUE, run_nondemographic = FALSE, show_indicator = q
))
run_case("all_herd_module_nondemographic", run_all_herd_module(
  cohort_level_data = rd(ex_mod, "nondemographic_herd_input_chrt_data.csv"),
  herd_level_data = rd(ex_mod, "nondemographic_herd_input_hrd_data.csv"),
  run_demographic = FALSE, run_nondemographic = TRUE, show_indicator = q
))
run_case("all_herd_module_both", run_all_herd_module(
  cohort_level_data = rd(ex_mod, "herd_all_input_chrt_data.csv"),
  herd_level_data = rd(ex_mod, "herd_all_input_hrd_data.csv"),
  run_demographic = TRUE, run_nondemographic = TRUE, show_indicator = q
))

cat("Full pipeline examples\n")
gleam_inputs <- function(filter = NULL) {
  f <- function(dt) if (is.null(filter)) dt else dt[filter(herd_id)]
  list(
    feed_rations = f(rd(ex_run, "feed_rations_share_chrt.csv")),
    feed_params = rd(ex_run, "feed_quality.csv"),
    feed_emissions = rd(ex_run, "feed_emission_factors.csv"),
    manure_management_system_fraction = f(rd(ex_run, "manure_management_system_fraction.csv")),
    manure_management_system_factors = f(rd(ex_run, "manure_management_system_factors.csv"))
  )
}
not_nondemo <- function(h) !h %in% c(14, 15)
only_nondemo <- function(h) h %in% c(14, 15)

run_case("run_gleam_mixed_no_structure", do.call(run_gleam, c(list(
  has_herd_structure = FALSE, run_demographic = TRUE, run_nondemographic = TRUE,
  cohort_level_data = rd(ex_run, "master_chrt_lvl_no_structure_mixed_data.csv")[not_nondemo(herd_id)],
  herd_level_data = rd(ex_run, "master_hrd_lvl_mixed_data.csv")[not_nondemo(herd_id)],
  simulation_duration = 365, show_indicator = q
), gleam_inputs(not_nondemo))))
run_case("run_gleam_nondemo_only", do.call(run_gleam, c(list(
  has_herd_structure = FALSE, run_demographic = FALSE, run_nondemographic = TRUE,
  cohort_level_data = rd(ex_run, "master_chrt_lvl_no_structure_nondemo_data.csv"),
  herd_level_data = rd(ex_run, "master_hrd_lvl_nondemo_data.csv"),
  simulation_duration = 365, show_indicator = q
), gleam_inputs(only_nondemo))))
for (gwp in c("AR6", "AR5_excluding_carbon_feedback", "AR5_including_carbon_feedback", "AR4")) {
  run_case(paste0("run_gleam_structure_", gwp), do.call(run_gleam, c(list(
    has_herd_structure = TRUE, run_demographic = FALSE, run_nondemographic = FALSE,
    cohort_level_data = rd(ex_run, "master_chrt_lvl_structure_data.csv"),
    herd_level_data = rd(ex_run, "master_hrd_lvl_structure_data.csv"),
    simulation_duration = 365, global_warming_potential_set = gwp, show_indicator = q
  ), gleam_inputs())))
}
run_case("run_gleam_mixed_no_structure_d180", do.call(run_gleam, c(list(
  has_herd_structure = FALSE, run_demographic = TRUE, run_nondemographic = TRUE,
  cohort_level_data = rd(ex_run, "master_chrt_lvl_no_structure_mixed_data.csv")[not_nondemo(herd_id)],
  herd_level_data = rd(ex_run, "master_hrd_lvl_mixed_data.csv")[not_nondemo(herd_id)],
  simulation_duration = 180, show_indicator = q
), gleam_inputs(not_nondemo))))

cat("Direct-emissions pipeline examples\n")
# The shared example tables also contain herd 13 (CHK, added on the
# new-species-chk branch) which the direct-emissions inputs do not cover, so
# they are restricted to the direct-emissions herd set.
direct_hrd <- rd(ex_mod, "emissions_direct_input_hrd_data.csv")
in_direct <- function(dt) dt[herd_id %in% direct_hrd$herd_id]
direct_common <- list(
  herd_level_data = direct_hrd,
  manure_management_system_fraction = in_direct(rd(ex_mod, "manure_management_system_fraction.csv")),
  manure_management_system_factors = in_direct(rd(ex_mod, "manure_management_system_factors.csv")),
  simulation_duration = 365, global_warming_potential_set = "AR6", show_indicator = q
)
feed_args <- list(
  feed_rations = in_direct(rd(ex_mod, "feed_rations_share_chrt.csv")),
  feed_params = rd(ex_mod, "feed_quality.csv")
)
run_case("emissions_direct_1a_no_structure", do.call(run_emissions_direct, c(list(
  has_herd_structure = FALSE,
  cohort_level_data = rd(ex_mod, "emissions_direct_input_chrt_no_structure_data.csv")
), feed_args, direct_common)))
run_case("emissions_direct_1b_structure", do.call(run_emissions_direct, c(list(
  has_herd_structure = TRUE,
  cohort_level_data = rd(ex_mod, "emissions_direct_input_chrt_structure_data.csv")
), feed_args, direct_common)))
run_case("emissions_direct_2a_no_structure_rq", do.call(run_emissions_direct, c(list(
  has_herd_structure = FALSE,
  cohort_level_data = rd(ex_mod, "emissions_direct_input_chrt_no_structure_ration_quality_data.csv")
), direct_common)))
run_case("emissions_direct_2b_structure_rq", do.call(run_emissions_direct, c(list(
  has_herd_structure = TRUE,
  cohort_level_data = rd(ex_mod, "emissions_direct_input_chrt_structure_ration_quality_data.csv")
), direct_common)))
run_case("emissions_direct_1b_structure_ef_only", do.call(run_emissions_direct, c(list(
  has_herd_structure = TRUE, emission_factors_only = TRUE,
  cohort_level_data = rd(ex_mod, "emissions_direct_input_chrt_structure_data.csv")
), feed_args, direct_common)))

cat("Done.\n")
