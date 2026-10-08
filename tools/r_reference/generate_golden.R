# Generate golden reference outputs from the original GLEAM R package.
#
# Usage (from the gleampy repository root):
#   Rscript tools/r_reference/generate_golden.R <path-to-GLEAM-R-source> <out_dir> [case_regex]
#
# Runs every run_*() example shipped with the R package and writes each
# resulting table to <out_dir>/<case>/<table>.csv with numeric columns written
# at 17 significant digits, so the Python port can be checked for numerical
# parity. A case that errors in R is recorded in <out_dir>/<case>/ERROR.txt.
# Files are written with LF line endings on every platform.
#
# Requires R >= 4.4 with pkgload and data.table (the package also imports cli);
# see tools/r_reference/README.md.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript tools/r_reference/generate_golden.R <R source> <out_dir> [case_regex]\n",
       "  out_dir is required: pass a new, empty directory (see tools/r_reference/README.md).",
       call. = FALSE)
}
r_src <- args[[1]]
out_dir <- args[[2]]
case_regex <- if (length(args) >= 3) args[[3]] else "."

# Locate this script, so tests/data and tests/golden are found from any cwd.
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1]]))) else "tools/r_reference"
same_dir <- function(a, b) {
  a <- normalizePath(a, winslash = "/", mustWork = FALSE)
  b <- normalizePath(b, winslash = "/", mustWork = FALSE)
  if (.Platform$OS.type == "windows") tolower(a) == tolower(b) else a == b
}
if (same_dir(out_dir, file.path(script_dir, "..", "..", "tests", "golden"))) {
  stop("refusing to write into tests/golden: pass a new, empty directory as out_dir, ",
       "compare it with tests/golden and copy only the cases you meant to change ",
       "(see tools/r_reference/README.md).", call. = FALSE)
}

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
  data.table::fwrite(dt, path, na = "NA", eol = "\n")
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
    con <- file(file.path(dir, "ERROR.txt"), "wb"); writeLines(conditionMessage(res), con); close(con)
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

# run_emissions_direct() on inputs with CHK and non-demographic (FN / MN)
# cohorts: the run_gleam examples rather than the direct-emissions examples,
# which hold only the six demographic cohorts of the ruminants and pigs.
run_case("emissions_direct_3a_no_structure_mixed", do.call(run_emissions_direct, list(
  has_herd_structure = FALSE, run_demographic = TRUE, run_nondemographic = TRUE,
  cohort_level_data = rd(ex_run, "master_chrt_lvl_no_structure_mixed_data.csv")[not_nondemo(herd_id)],
  herd_level_data = rd(ex_run, "master_hrd_lvl_mixed_data.csv")[not_nondemo(herd_id)],
  feed_rations = rd(ex_run, "feed_rations_share_chrt.csv")[not_nondemo(herd_id)],
  feed_params = rd(ex_run, "feed_quality.csv"),
  manure_management_system_fraction = rd(ex_run, "manure_management_system_fraction.csv")[not_nondemo(herd_id)],
  manure_management_system_factors = rd(ex_run, "manure_management_system_factors.csv")[not_nondemo(herd_id)],
  simulation_duration = 365, global_warming_potential_set = "AR6", show_indicator = q
)))
structure_direct_common <- function() list(
  has_herd_structure = TRUE,
  herd_level_data = rd(ex_run, "master_hrd_lvl_structure_data.csv"),
  manure_management_system_fraction = rd(ex_run, "manure_management_system_fraction.csv"),
  manure_management_system_factors = rd(ex_run, "manure_management_system_factors.csv"),
  simulation_duration = 365, global_warming_potential_set = "AR6", show_indicator = q
)
run_case("emissions_direct_3b_structure_chk_nondemo", do.call(run_emissions_direct, c(list(
  cohort_level_data = rd(ex_run, "master_chrt_lvl_structure_data.csv"),
  feed_rations = rd(ex_run, "feed_rations_share_chrt.csv"),
  feed_params = rd(ex_run, "feed_quality.csv")
), structure_direct_common())))

cat("Extra coverage cases (inputs built by tests/data/build_golden_inputs.py)\n")
data_dir <- normalizePath(file.path(script_dir, "..", "..", "tests", "data"), mustWork = FALSE)
rdd <- function(case, name) data.table::fread(file.path(data_dir, case, paste0(name, ".csv")))
case_inputs <- function(case, rations = "feed_rations") list(
  herd_level_data = rdd(case, "herd_level_data"),
  feed_rations = rdd(case, rations),
  feed_params = rd(ex_run, "feed_quality.csv"),
  feed_emissions = rd(ex_run, "feed_emission_factors.csv"),
  manure_management_system_fraction = rdd(case, "manure_management_system_fraction"),
  manure_management_system_factors = rdd(case, "manure_management_system_factors")
)

# run_emissions_direct() with primary ration quality supplied per cohort and
# non-demographic phase (no feed tables).
run_case("emissions_direct_3b_structure_chk_nondemo_rq", do.call(run_emissions_direct, c(list(
  cohort_level_data = rdd("direct_rq", "cohort_level_data")
), structure_direct_common())))

# BFL / SHP / GTS / CML / CTL with non-demographic FN and MN cohorts (SHP: FN
# only, SHP MN is rejected by R, see run_gleam_shp_mn_rejected).
run_case("run_gleam_ruminant_nondemo_no_structure", do.call(run_gleam, c(list(
  has_herd_structure = FALSE, run_demographic = TRUE, run_nondemographic = TRUE,
  cohort_level_data = rdd("ruminant_nondemo", "cohort_level_data"),
  simulation_duration = 365, global_warming_potential_set = "AR6", show_indicator = q
), case_inputs("ruminant_nondemo"))))
run_case("run_gleam_ruminant_nondemo_structure", do.call(run_gleam, c(list(
  has_herd_structure = TRUE, run_demographic = FALSE, run_nondemographic = FALSE,
  cohort_level_data = rdd("ruminant_nondemo", "cohort_level_data_structure"),
  simulation_duration = 365, global_warming_potential_set = "AR6", show_indicator = q
), case_inputs("ruminant_nondemo"))))
run_case("emissions_direct_3c_no_structure_ruminant_nondemo", do.call(run_emissions_direct, c(list(
  has_herd_structure = FALSE, run_demographic = TRUE, run_nondemographic = TRUE,
  cohort_level_data = rdd("ruminant_nondemo", "cohort_level_data"),
  simulation_duration = 365, global_warming_potential_set = "AR6", show_indicator = q
), case_inputs("ruminant_nondemo")[c(
  "herd_level_data", "feed_rations", "feed_params",
  "manure_management_system_fraction", "manure_management_system_factors"
)])))
# CHK whose adult females do not lay (is_egg_producing = FALSE on FA).
run_case("run_gleam_chk_nonlaying_no_structure", do.call(run_gleam, c(list(
  has_herd_structure = FALSE, run_demographic = TRUE, run_nondemographic = FALSE,
  cohort_level_data = rdd("chk_nonlaying", "cohort_level_data"),
  simulation_duration = 365, global_warming_potential_set = "AR6", show_indicator = q
), case_inputs("chk_nonlaying", "feed_rations_share_chrt"))))
run_case("run_gleam_chk_nonlaying_structure", do.call(run_gleam, c(list(
  has_herd_structure = TRUE, run_demographic = FALSE, run_nondemographic = FALSE,
  cohort_level_data = rdd("chk_nonlaying", "cohort_level_data_structure"),
  simulation_duration = 365, global_warming_potential_set = "AR6", show_indicator = q
), case_inputs("chk_nonlaying", "feed_rations_share_chrt"))))
# Adult CHK females gaining weight, laying and not laying: the growth
# coefficient of FA (0.0279, flag ignored in R) only shows when the daily
# weight gain is not 0, which the pipelines never produce for adults.
run_case("metabolic_energy_req_module_chk_fa_growth", list(result = run_metabolic_energy_req_module(
  cohort_level_data = rdd("mer_chk_growth", "cohort_level_data"),
  herd_level_data = rdd("mer_chk_growth", "herd_level_data"), show_indicator = q
)))
# Expected to fail (ERROR.txt): the maintenance validator requires
# offtake_rate < 1 for SHP males, but the non-demographic herd module sets
# offtake_rate = 1 on every MN row.
run_case("run_gleam_shp_mn_rejected", do.call(run_gleam, c(list(
  has_herd_structure = FALSE, run_demographic = TRUE, run_nondemographic = TRUE,
  cohort_level_data = rdd("shp_mn_rejected", "cohort_level_data"),
  simulation_duration = 365, global_warming_potential_set = "AR6", show_indicator = q
), case_inputs("shp_mn_rejected"))))

cat("Done.\n")
