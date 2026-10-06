# Run scenarios produced by make_scenarios.py through the R GLEAM package.
#
# Usage:
#   Rscript tools/parity/run_scenarios.R <path-to-GLEAM-R-source> <scenarios_dir>
#
# For each scenario directory, writes R outputs to <scenario>/r_out/<table>.csv
# (numeric columns at 17 significant digits) or <scenario>/r_out/ERROR.txt.

args <- commandArgs(trailingOnly = TRUE)
r_src <- args[[1]]
scen_dir <- args[[2]]

suppressMessages(pkgload::load_all(r_src, quiet = TRUE, export_all = TRUE))
suppressMessages(library(data.table))

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

flatten <- function(x, prefix = "") {
  out <- list()
  if (data.table::is.data.table(x) || is.data.frame(x)) {
    out[[prefix]] <- x
  } else if (is.list(x)) {
    for (n in names(x)) out <- c(out, flatten(x[[n]], if (prefix == "") n else paste0(prefix, "__", n)))
  }
  out
}

for (sd in sort(list.dirs(scen_dir, recursive = FALSE))) {
  args_txt <- paste(readLines(file.path(sd, "args.json")), collapse = "")
  get_arg <- function(key) {
    m <- regmatches(args_txt, regexec(sprintf('"%s": *("?[^",}]*"?)', key), args_txt))[[1]][2]
    m <- gsub('"', "", m)
    if (m %in% c("true", "false")) return(m == "true")
    if (!is.na(suppressWarnings(as.numeric(m)))) return(as.numeric(m))
    m
  }
  out <- file.path(sd, "r_out")
  dir.create(out, showWarnings = FALSE)
  unlink(list.files(out, full.names = TRUE))
  r_in <- file.path(sd, "r_in")
  dir.create(r_in, showWarnings = FALSE)
  # Record the inputs exactly as fread parsed them (hex floats), so Python can
  # be fed bit-identical inputs: fread is not correctly rounded on every
  # platform, and some GLEAM validations compare floats exactly.
  rd <- function(name) {
    dt <- data.table::fread(file.path(sd, paste0(name, ".csv")))
    hex <- data.table::copy(dt)
    for (col in names(hex)) {
      v <- hex[[col]]
      if (is.double(v)) {
        s <- sprintf("%a", v)
        s[is.na(v)] <- NA_character_
        data.table::set(hex, j = col, value = s)
      }
    }
    data.table::fwrite(hex, file.path(r_in, paste0(name, ".csv")), sep = "\t", na = "")
    dt
  }
  t0 <- Sys.time()
  res <- tryCatch(
    run_gleam(
      has_herd_structure = get_arg("has_herd_structure"),
      run_demographic = get_arg("run_demographic"),
      run_nondemographic = get_arg("run_nondemographic"),
      cohort_level_data = rd("cohort_level_data"),
      herd_level_data = rd("herd_level_data"),
      feed_rations = rd("feed_rations"),
      feed_params = rd("feed_params"),
      feed_emissions = rd("feed_emissions"),
      manure_management_system_fraction = rd("manure_management_system_fraction"),
      manure_management_system_factors = rd("manure_management_system_factors"),
      simulation_duration = get_arg("simulation_duration"),
      global_warming_potential_set = get_arg("global_warming_potential_set"),
      show_indicator = FALSE
    ),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    writeLines(conditionMessage(res), file.path(out, "ERROR.txt"))
    cat(sprintf("%s ERROR %s\n", basename(sd), conditionMessage(res)))
  } else {
    tabs <- flatten(res)
    for (n in names(tabs)) write_table(tabs[[n]], file.path(out, paste0(n, ".csv")))
    cat(sprintf("%s ok (%.1fs)\n", basename(sd), as.numeric(Sys.time() - t0, units = "secs")))
  }
}
