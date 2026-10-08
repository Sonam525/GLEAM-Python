# R to Python port notes

## Which R code was ported

The port does **not** target the released R package (`un-fao/GLEAM` 0.8.0,
branch `main`), which lacks chickens, the non-demographic herd model and
`run_emissions_direct()`. It targets the union of unreleased development
branches of [un-fao/GLEAM](https://github.com/un-fao/GLEAM), merged in this
order, plus one adaptation commit. The result is called
`integration/r-reference` below.

| Step | Source | Commit | Content |
|---|---|---|---|
| start | `feature/new-species-chk` | `50f3c39dd02fa4969714b1d2657b81b616291769` | chickens (CHK), egg production and allocation |
| merge | `feature/herd-nondemo` | `f3d125475af661389943838a97b2d1836aabd940` | non-demographic herd module, all-herd orchestrator |
| merge | `main` (0.8.0) | `90e416197e89093c4f3a347b263ba805d33d4aac` | released package, CRAN fixes |
| merge | `feature/run-direct-emissions-only` | `aa8915d9c5595ce78fc33db57ed0d34399212bc4` | `run_emissions_direct()` |
| patch | `tools/r_reference/patches/` | `eef5adba358059f1a133e58f52519266ef784dd4` | adaptation of `run_emissions_direct()` (below) |

`feature/optional-validation-rule` (`6a66d86`), which adds the
`validate_inputs` switch, was re-implemented rather than merged.

The merges are clean. One adaptation was needed (commit `eef5adb`, dated
2026-10-08): `run_emissions_direct()` was written against `main` and had to
call `run_all_herd_module()`, which gives it the arguments `run_demographic`
and `run_nondemographic`, add the optional non-demographic herd columns and
join ration quality on `nondemo_productive_phase_id`, exactly as
`run_gleam()` does on `feature/herd-nondemo`. Its example herd table gained
`prop_nondemo_fem_juv = prop_nondemo_mal_juv = 0` and its tests were pointed
at the direct-emissions example inputs. This commit is not part of
un-fao/GLEAM. It is shipped as a patch, and
`tools/r_reference/build_r_reference.sh` rebuilds the whole reference from
the public un-fao/GLEAM commits and checks its source tree hash. The finished
reference can also be browsed as the branch
[`integration/r-reference` of Sonam525/GLEAM](https://github.com/Sonam525/GLEAM/tree/integration/r-reference)
(same source tree). See
[tools/r_reference/README.md](../tools/r_reference/README.md).

The reference passes its own test suite: `testthat::test_dir()` on
`tests/testthat` reports 306 tests and 733 expectations (122 of them in
`test-run_emissions_direct.R`), 0 failures, 0 errors, and 4 expected warnings
("no non-demographic rows were found") from the direct-emissions tests.

Not ported:

* `feature/specie-cohort-specific-validation` only reorganises input
  validation and conflicts heavily with the chicken branch.
* `release/v1-pipeline` is an obsolete early pipeline.
* `copilot/*` contain only empty "Initial plan" commits.
* `cran_*` and `dommens-*` are fully merged into `main`.
* The `gleam/` folder on `main` is an accidentally committed, stale snapshot
  of `feature/herd-nondemo`.

## How parity is verified

1. **Golden outputs.** `tools/r_reference/generate_golden.R` runs the
   `run_*()` examples of the R reference and saves every output table in
   `tests/golden/<case>/`. The cases cover every module example, the
   `run_gleam()` examples (with all four GWP sets and a 180-day assessment),
   `run_emissions_direct()` variants, and extra inputs in `tests/data/` for
   branches the examples never reach (ruminant and camel non-demographic
   cohorts, non-laying CHK, CHK adult growth). A case where R is expected to
   fail stores the R error message instead, and the test checks that Python
   raises `GleamValidationError` with the same message once formatting is
   normalised. The pytest suite reproduces each call in Python and compares
   every table: same columns, same column order, same row order, equal
   logical and text values, and numbers within `rtol = 1e-9` relative with
   no absolute term, so a value R computes as exactly zero must be exactly
   zero in Python (see `tests/golden_utils.py`; its opt-in `zero_floor` rule
   for cancellation noise is only a diagnostic, used neither by the parity
   tests nor by the documented scenario run).
2. **Ported unit tests.** Every `testthat` block of the R package is ported
   to pytest with the same inputs and expected values. The one block that
   inspects source code instead of results, `test-no_duplicate_functions.R`,
   fails when a function is defined more than once in `R/`, whose files
   share one namespace, so the definition collated last silently wins.
   Python modules have separate namespaces; `tests/test_no_duplicates.py`
   adapts the test to the hazards that remain: no public function or class
   is defined more than once in the package, no name is registered twice
   in `gleampy/__init__.py`, every export comes from the module it is
   registered to, and every public `calc_*` / `run_*` function of
   `gleampy.core` and `gleampy.modules` is exported.
3. **Randomised scenarios.** `tools/parity/` perturbs the pipeline examples
   (weights, yields, rates, feed and manure parameters, ration and manure
   shares, assessment length, GWP set), runs each scenario through R and
   Python, and compares all outputs with the golden-test rule. When R
   rejects a scenario, Python must reject it too, with the same validation
   message (after normalising formatting) and the same list of violating
   rows; an R crash or any other Python exception counts as a failure.
   Python receives the inputs exactly as R parsed them (hex floats), so the
   comparison is bit-for-bit on inputs. The documented run:

   ```bash
   python tools/parity/make_scenarios.py OUT 10 20261006
   Rscript tools/parity/run_scenarios.R ../GLEAM-reference OUT
   python tools/parity/compare_scenarios.py OUT
   ```

   Result on these 30 scenarios (10 each from the herd-structure, mixed
   demographic + non-demographic and CHK non-demographic examples), all 30
   matching:
   * 23 ran in both implementations and matched on every output table;
   * the other 7 were rejected by both, with the same message and the same
     rows, by the ULP-sensitive weight check (see "Exact float comparison in
     the weights check" below).

   `compare_scenarios.py OUT --stats` also prints the agreement figures
   quoted below, and `--zero-floor 1e-12` turns on the opt-in near-zero
   rule to help diagnose a failing scenario.
4. **Bit-level checks.** While porting, each module was also compared with R
   on hex-dumped doubles. All arithmetic reproduces R's order of operations,
   including the plain left-to-right summation data.table uses for the
   sum-to-one checks. The remaining differences come from R's math library
   on the reference machine (Windows ARM64, R running as emulated x86_64):
   its `^`/`exp`/`log` are up to a few ULP from the correctly rounded result
   that numpy returns, and its `fread`/`sprintf` are not correctly rounded.

### How close the numbers are

The tests enforce `rtol = 1e-9`. The measured agreement is much closer:

* **Golden cases** (153,585 non-zero numbers in the 173 golden tables,
  120,779 of them in floating-point columns; the rest are whole numbers such
  as ids, durations and counts): 72 % are bit-identical to R (65 % of the
  floating-point values). Cases that do not simulate the demographic herd
  agree within 6e-15 relative, a few units in the last place. Cases that
  simulate it agree within about 1e-11; the largest differences are in
  `growth_rate_herd` (6.6e-12 on the demographic example, 8.3e-12 on the
  ruminant non-demographic case without herd structure) and in the
  quantities derived from the simulated stocks.
* **Randomised scenarios** (137,237 non-zero numbers in the output tables
  of the 23 scenarios that ran, 108,491 of them in floating-point columns):
  67 % are bit-identical (58 % of the floating-point values). Herd-structure
  and CHK non-demographic scenarios agree within 1.4e-14. Scenarios that
  simulate the demographic herd agree within 1.4e-11, except
  `growth_rate_herd` of a herd growing 0.15 % a year: 6.1e-11 relative,
  9e-14 absolute.

The demographic herd model amplifies 1-ULP library differences because some
of its formulas are ill-conditioned:

* `probability_growth = (s^(D-1) - s^D) / (1 - s^D)` cancels when the daily
  survival `s` is close to 1;
* `1 - exp(-(hazard_death + hazard_offtake))` loses digits when the net
  hazard is close to 0 (offtake close to minus the death rate);
* `growth_rate_herd = (FJ[t] / FJ[t-1])^365 - 1` has no relative accuracy
  when the herd is close to stationary: a 1-ULP change in the daily ratio
  changes it by about 8e-14 in absolute terms.

Comparisons of `run_demographic_herd_module` on random herds (inputs passed
to both sides as hex doubles) found: on herds with realistic parameters,
stocks and offtake within about 1e-12 relative; over the whole valid
parameter range, within about 1e-10; in rare extreme cases (net hazard near
0, herds that do not converge within `max_simulation_years`) about 1e-9,
which the parity tolerance does not cover. `growth_rate_herd` of a
near-stationary herd can differ by 1e-6 relative or more while its absolute
difference stays around 1e-13; compare it with an absolute tolerance. The
iteration counts (`days_to_steady_state`) are identical.

## Intentional differences from R

* **Vectorised core models.** R evaluates the `calc_*` functions row by row
  (`by = .I`) on scalars. Python evaluates them on whole numpy arrays, with
  one boolean mask per R `if` branch. Scalars in give a scalar out. Per-herd
  functions of the demographic herd model (`calc_transition_probabilities`,
  `calc_steady_state_structure`, `calc_projected_population_size`,
  `calc_summary_offtake`) keep R's per-herd signature (named vectors become
  dicts / Series); `run_demographic_herd_module` still evaluates all herds
  at once.
* **Never truncated to integer.** R's data.table assignments can truncate
  results when an input or output column is integer-typed (see "Integer
  columns truncate results" below). Python computes everything in float64,
  whatever the pandas dtype of the inputs (int64, nullable Int64, float64,
  object), and replaces existing output columns with float64 columns. Its
  results equal R's when R gets double-typed inputs.
* **Zero milk.** R's `validate_allocation_milk_inputs` skips the milk-fraction
  checks only when `identical(milk_production_fpcm_cohort, 0)`, which is
  `FALSE` for an integer `0L`. With an integer milk column, R therefore
  rejects NA or out-of-range standard milk fractions on rows without milk.
  Python treats integer and double zero alike and skips those checks; when R
  succeeds, the numbers are the same.
* **A `simulation_duration` column in the cohort table.** R's
  `run_production_module` (milk, egg and fibre output) and
  `run_allocation_module` (fibre and work energies) call their `calc_*`
  functions inside `cohort_level_data[, ... := calc_*(..., simulation_duration
  = simulation_duration), by = .I]`. data.table looks names up among the
  table's columns first, so a cohort column named `simulation_duration`
  silently replaces the function argument, row by row. In `run_gleam()` and
  `run_emissions_direct()`, which run production and allocation after the
  energy and emission modules, it changes the production outputs, the
  allocation shares and everything aggregated from them, but not the energy
  requirements or the emissions per cohort. Python always uses the
  argument. No other module is affected: aggregation melts the table first,
  and the herd modules do not evaluate the argument inside a cohort-table
  `j`. Without such a column the results are identical.
* **Validation order.** Checks run over all rows at once, category by
  category. When several rows are invalid for different reasons, the first
  error reported can differ from R's first row-wise error; vector messages
  name the failing element (`` `x`[3] = ... ``).
* **Missing columns.** Where R fails with an internal error (for example
  `object 'x' not found`) because a column that a row needs is missing,
  Python raises `GleamValidationError` ("Missing required columns in
  `` `arg` ``: ...") when validation is on, and follows R's rules for which
  rows need which columns (see "Columns that no validator checks"). Checks
  made in the module code, where R reads the column, also apply with
  `validate_inputs=False`: for example, `run_gleam()` and
  `run_emissions_direct()` reject a missing `live_weight_at_weaning` with or
  without validation, as they reject `run_demographic = FALSE` without
  FN/MN rows (see "`run_demographic = FALSE` without FN/MN rows crashes").
  With validation off, a column that only a validator checks (for example
  `herd_id` in `herd_level_data`) can instead fail later with a plain
  `KeyError`.
* **No side effects on inputs.** R's `run_demographic_herd_module` sorts the
  caller's `herd_level_data` by reference (`setkey`). Python never modifies
  the caller's tables. Columns that R's validators create and drop on their
  working copy (`activity_sum` in the energy module) are left alone in
  Python, so a user column of that name survives.
* **`validate_inputs=False`** (every `run_*` function) skips the input
  validators and warns once, as on `feature/optional-validation-rule`.
  Checks made where R itself would stop still apply, for example on a
  missing column that R reads (see "Missing columns").
* **Logical switches.** The pipeline switches follow R's `isTRUE()`. A
  Python `bool` and a `numpy.bool_` (for example from `Series.any()` or an
  element of a bool array) behave the same everywhere, including the step
  of `run_gleam()` and `run_emissions_direct()` that sets the
  non-demographic start weights from `live_weight_at_weaning`.
  `has_herd_structure`, `run_demographic`, `run_nondemographic` and
  `emission_factors_only` also accept a one-element bool array, the
  counterpart of R's length-1 logical. `validate_inputs` accepts only a
  `bool` or a `numpy.bool_`; any other value, arrays included, stops with
  "`validate_inputs` must be TRUE or FALSE." With validation on, a
  non-logical switch (`1`, `1.0`, `"TRUE"`, `None`, a list, a longer array)
  is rejected as in R, for example "`has_herd_structure` must be a single
  logical value (TRUE or FALSE)." With `validate_inputs=False` the switches
  are not checked and a non-logical value counts as FALSE:
  `has_herd_structure=1` simulates the herd, `emission_factors_only=1` runs
  production, allocation and aggregation, and `run_demographic=1` does not
  run the demographic module. R's own code tests `if (has_herd_structure)`
  and `if (!emission_factors_only)`, where a non-zero number counts as TRUE
  and a string stops R with an error. This cannot affect parity:
  `integration/r-reference` has no unchecked path, and its validators
  reject these values.
* **Undefined branches.** Where R crashes because no branch assigned a
  result (unknown species with validation off, `object '...' not found`),
  Python returns NaN or raises `GleamValidationError`.
* **Degenerate-herd stops raise `GleamValidationError`.** Some herds pass
  validation but still stop R with an internal error: "missing value where
  TRUE/FALSE needed" in the demographic steady-state simulation, and "$
  operator is invalid for atomic vectors" for a non-demographic phase 1 of
  0 days (see "Crashes on degenerate herds"). Python stops on the same
  herds with a `GleamValidationError` (a `ValueError`) that gives R's
  message, the cause and, in the run modules, the herd id, so a single
  `except GleamValidationError` catches every rejected input.
* **CSV reading.** `gleampy.io.read_csv` follows `data.table::fread`'s
  defaults for the files GLEAM uses (separator detection, white-space
  stripping, empty fields, NA and NaN tokens, all-empty columns as logical
  NA, encodings); the exact rules and the few parts of `fread` it does not
  reproduce are in the `gleampy.io` module documentation. One difference is
  deliberate: a whole-number column with missing values is float64, not
  integer, which has no effect because Python never truncates.
* **All-empty input columns.** `read_csv` reads a column without any value
  as a logical NA column (an object column of `None`), as `fread` does, so
  the run validators reject it where R does (an empty
  `phase1_nondemo_mal_duration_days` column stops both with
  "`herd_level_data$phase1_nondemo_mal_duration_days` must be numeric.").
  One case still differs: in the non-demographic herd module, an all-empty
  `cohort_duration_days` column makes data.table coerce the herd-level phase
  durations written into it to `TRUE` (with warnings), after which R stops
  with "`phase1_nondemo_duration` must be a single numeric value." Python
  writes the durations as float64 and runs, with the results R gives for a
  numeric NA column (see "Output columns that already exist" below).
* **Progress output** goes to stderr; `show_indicator=False` silences it.
* Error messages drop the `cli` markup (`{.arg x}` becomes `` `x` ``) and use
  `>=` / `<=` instead of `≥` / `≤`.

## Suspected bugs in the R package

Unless marked **not replicated**, these are reproduced faithfully in Python
(so results stay identical to R) and are listed here so they can be reported
upstream. Several matter when the model is driven by an optimiser or an RL
agent: they are marked **[RL]**.

### Integer columns truncate results (not replicated)

R writes almost every result with `DT[, col := f(...), by = .I]`. data.table
creates the new column with the type of the **first** row's result and
coerces every later row to that type. Assigning into a column that already
exists coerces to that column's type. `fread` reads a column whose values are
all whole numbers as integer, so:

* **Weights.** `run_weights_module` assigns the `calc_cohort_weights()`
  outputs, then the average and final weights and the daily gain, with
  `:=` and `by = .I`. When a herd weight column holds only whole numbers
  (typical for a file with one herd, or with whole-kilogram adult weights)
  and the first cohort row copies it (FA, MA, FJ or FS, or FN/MN with an
  integer start weight), the weight columns become integer. Later
  fractional weights are truncated with only a "truncated (precision
  lost)" warning: FJ birth weight 1.2 becomes 1, weaning weight 7.5
  becomes 7, slaughter weights 110.5 become 110, an interpolated
  non-demographic phase weight 46.88 becomes 46, a CHK FN daily gain of
  0.0286 kg becomes 0. Values beyond the 32-bit range become NA, and R
  then stops in the average-weight validation (for example
  "`live_weight_cohort_at_slaughter` must be a single numeric value.").
  The result depends on the order of the cohort rows. The
  error flows through energy, intake and emissions in `run_gleam()` and
  `run_emissions_direct()`: for example herd 9 of the examples on its own
  gives cohort values up to 17 % lower in R (FJ initial weight 1 kg instead
  of 1.2 kg) and a total CO2-eq 0.14 % lower. 8 of the 15 example herds are
  affected when each is read from its own file.
  Python returns the values R gives for double-typed inputs, whatever the
  input dtype (`tests/test_parity_weights.py`).
* **Non-demographic start weights.** When both herd modules run,
  `run_gleam()` and `run_emissions_direct()` overwrite
  `live_weight_{female,male}_nondemographic_start` with
  `live_weight_at_weaning` (`:=` on a subset of rows). If the herd table
  already has these columns as integer (whole numbers) or logical (empty),
  data.table coerces the weaning weight to that type: it is truncated, or
  becomes `TRUE`. Python writes float64 values.
* **Non-demographic phase durations.** In the bundled `run_gleam()` and
  all-herd inputs, `cohort_duration_days` holds whole numbers with blank
  FN/MN rows, so `fread` types it integer and R truncates fractional
  herd-level phase durations (`phase1_nondemo_fem_duration_days` etc.) to
  whole days. That changes the cycle geometry, stocks and offtake, and which
  inputs are accepted: a phase-2 duration of 0.5 days is truncated to 0 and
  accepted by R, but rejected by Python. Likewise, with the FN phases of
  herd 15 of the non-demographic `run_gleam()` example set to 19.25 and
  19.3 days (7 days of rest), R truncates both to 19 days and runs, while
  Python rejects the 0.6-day partial phase they leave (see "Sub-day partial
  phases").
* **Output columns that already exist** in the input as integer, logical or
  character (for example a module run again on its own output read back
  with `fread`, where all-zero columns come back as integer and all-NA ones
  as logical): integer values are truncated, logical ones become `TRUE` and
  then fail validation ("must be numeric"), character ones are coerced
  silently.
* **Integer overflow.** In `calc_meat_production`
  (`offtake_heads_assessment * live_weight_cohort_at_slaughter`) and
  `calc_milk_production` (`milk_yield_day * simulation_duration *
  cohort_stock_size`, multiplied left to right), integer inputs use integer
  arithmetic and products above 2^31 - 1 become NA with the warning "NAs
  produced by integer overflow". This affects `run_production_module()` and
  direct calls, and also `run_gleam()` and `run_emissions_direct()` (unless
  `emission_factors_only = TRUE`, which skips production) with
  `has_herd_structure = TRUE`, which take the head counts from the given
  herd structure. For example, herd 1 of the `run_gleam()` structure
  example, written with its head counts rounded to whole numbers and read
  back with `fread`, has an integer `offtake_heads_assessment`, and
  `live_weight_cohort_at_slaughter` is integer too, because its first
  cohort row (FA) copies the whole-number adult weight (see "Weights"
  above). FA meat is 4,251,321 head x 680 kg: R's live-weight meat becomes
  NA and `run_gleam()` stops in the allocation module with
  "`meat_production_live_weight_cohort` must be a single numeric value."
  Python runs and gives 2.89e9 kg. Milk overflows only when
  `simulation_duration` is an integer as well (for example `365L`; the
  default `365` is a double). With `has_herd_structure = FALSE` the herd
  modules compute `cohort_stock_size` and `offtake_heads_assessment` as
  doubles, even from integer herd inputs, so the pipelines do not
  overflow.

Python never truncates (see "Intentional differences"). To reproduce R
exactly, convert integer columns to double after `fread`, for example
`for (j in names(dt)) if (is.integer(dt[[j]])) data.table::set(dt, j = j, value = as.numeric(dt[[j]]))`.
Upstream fix: coerce numeric inputs with `as.numeric()` in the `run_*`
modules.

### Validation

* **Exact float comparison in the weights check.** The check
  `live_weight_cohort_initial <= live_weight_cohort_average <=
  live_weight_cohort_final` compares doubles exactly.
  `final = w*(1-r) + w*r` can be 1 ULP below `w`, so adult cohorts and CHK
  juveniles randomly fail for ordinary offtake rates (seen in randomised
  scenarios). **[RL]**
* **Sub-day partial phases.** `validate_nondemo_offtake_inputs` checks the
  partial phases left after the last full production cycle
  (`partial_phase1/2_nondemo_duration`, computed by
  `calc_nondemo_cycle_geometry`, not entered by the user) against the
  `cohort_duration_days` range [1, 8000] whenever they are above 0. With
  fractional phase or rest durations the remainder is often between 0 and
  1 day, and a valid herd is rejected with a message about a
  `cohort_duration_days` value the user never supplied (for example phase 1
  of 90.2 days and 1 day of rest leave 0.2 day). `validate_nondemo_phase_inputs`
  accepts the same values under the `simulation_duration` rule. With
  continuous random durations, roughly 1 to 4 % of draws per FN/MN block hit
  this (about one or two divided by the cycle length in days). Python
  rejects the same herds, also in direct calls of
  `calc_nondemo_offtake_total_horizon`, and its message adds that the value
  is `partial_phase1_nondemo_duration` or `partial_phase2_nondemo_duration`,
  the remainder of the 365-day horizon, not an input. **[RL]**
* **SHP non-demographic males are always rejected.** The maintenance
  validator checks `offtake_rate` in [-2, 1) for every SHP male cohort,
  MN included, but the non-demographic herd module (and the structured
  examples) set `offtake_rate = 1` on every FN/MN row, and SHP MN
  maintenance uses a fixed coefficient that never reads the offtake rate.
  Any SHP herd with MN cohorts therefore fails `run_gleam()` and
  `run_metabolic_energy_req_module()` with "`offtake_rate` = 1 is out of
  range" unless validation is off (golden case `run_gleam_shp_mn_rejected`).
  GTS and CTL MN are accepted. **[RL]**
* **`cohort_stock_size` of 0.** The run validator allows `>= 0`, but
  `calc_cohort_totals` requires a positive value, so any empty cohort aborts
  aggregation. Before that, in the energy module, a laying CHK cohort (FA,
  or FN with `is_egg_producing`) with 0 head gets infinite egg-deposition
  energy (`egg_output_human_consumption / 365 / cohort_stock_size`), and the
  `Inf` passes through `metabolic_energy_req_total` and `ration_intake`
  without any error: `validate_egg_inputs` only requires
  `cohort_stock_size >= 0`, and the total and intake validators accept
  `Inf`. `run_metabolic_energy_req_module()` and the core functions return
  the `Inf`; only `run_gleam()`'s aggregation step then stops. With an egg
  output of 0 as well, the egg energy is 0/0 = NaN and the module stops
  instead with "`metabolic_energy_req_egg_deposition` must be a single
  numeric value." Upstream fix: require a positive `cohort_stock_size` in
  `validate_egg_inputs`, or reject non-finite values in the total-energy
  and intake validators.
* **NA ration fractions.** An NA `feed_ration_fraction` makes the
  sum-to-one check pass silently (NA sum).
* **Missing digestible energy.** `calc_feed_digestibility_fraction`
  deliberately turns a missing `feed_digestible_energy_ruminant` /
  `_pigs` into a digestibility of 0. As a result the "Missing required
  digestibility inputs" check in `run_ration_quality_module()` can never
  fire, and a feed whose digestible energy is missing for the eating
  species silently adds 0 to `ration_digestibility_fraction`, which lowers
  enteric CH4 and manure volatile solids. A missing metabolizable or
  urinary energy raises an error instead.
* **Rest duration of 0.** `rest_between_nondemo_cycles_duration = 0` is
  rejected because the `duration` range starts at 1.
* **Truncated error messages.** `validate_run_aggregation_inputs.R` passes
  two strings to `cli_abort`, so the second half is dropped.
  `validate_run_emissions_manure_inputs.R` reports only the first invalid
  group.
* **Effectively required columns.**
  * `run_weights_module` reads 14 herd columns for every row (10 live
    weights and the 4 non-demographic phase durations), so the
    non-demographic columns are required even for demographic-only data.
  * Its validator also always requires `species_short`.
  * `nondemo_productive_phase_id` only works when absent thanks to lazy
    evaluation.
* **Columns that no validator checks.** R reads some columns without
  checking them first, then stops with a base error instead of a validation
  message (checked on herds 1 (CTL) and 13 (CHK) of the `run_gleam()`
  examples):
  * `is_egg_producing` when `herd_level_data` has a CHK herd: "object
    'is_egg_producing' not found" in `run_gleam()`,
    `run_emissions_direct()`, `run_metabolic_energy_req_module()`,
    `run_nitrogen_balance_module()` and `run_production_module()`. Only the
    allocation validator checks it. Without CHK herds the column is optional
    (R adds it as NA).
  * `cohort_stock_size` with `has_herd_structure = TRUE` and a CHK herd:
    "object 'cohort_stock_size' not found". Without CHK herds a module
    validator reports it.
  * `herd_id` in `herd_level_data` when that table has
    `prop_nondemo_fem_juv` and `prop_nondemo_mal_juv`:
    `validate_run_gleam_inputs()` itself stops with "object 'herd_id' not
    found". Without those two columns the validators report the missing
    columns.

  With validation on, Python raises `GleamValidationError` in all these
  cases ("Missing required columns in `cohort_level_data`:
  "is_egg_producing"", and likewise for the others). Only the error type and
  message differ; no result changes.
* **`run_demographic = FALSE` without FN/MN rows crashes.** With
  `has_herd_structure = FALSE`, `run_demographic = FALSE`,
  `run_nondemographic = TRUE` and a cohort table without FN or MN rows,
  `validate_run_all_herd_module_inputs` only warns ("run_nondemographic=TRUE
  but no non-demographic rows were found in cohort_level_data."). No herd
  module runs, and `run_all_herd_module()` returns an empty cohort table and
  a `NULL` herd table. `run_gleam()` and `run_emissions_direct()` then stop
  inside data.table at `gleam_hrd_data[, (missing_optional_nondemo_cols) :=
  NA_real_]` (`R/run_gleam.R:778`, `R/run_emissions_direct.R:537`) with
  "Check that is.data.table(DT) == TRUE. Otherwise, `:=` is defined for use
  in j, once only and in particular ways. ...", which does not name the
  cause. Python's `run_all_herd_module()` behaves as R does (same warning,
  empty cohort table, `herd_level_results` `None`). `run_gleam()` and
  `run_emissions_direct()` raise `GleamValidationError` "`run_demographic =
  FALSE` requires non-demographic (FN/MN) rows in `cohort_level_data`: no
  herd module ran, so there are no herd-level results.", with or without
  validation. An optimiser or RL agent that switches `run_demographic` off
  must keep FN/MN rows in the cohort table. **[RL]**
* **Fibre for non-demographic cohorts.** Fibre production and the fibre
  energy validation skip FN/MN, although the calculations use
  `fibre_yield_year` for them. A negative value passes and gives negative
  fibre.
* **Inconsistent milk checks.**
  * `validate_allocation_milk_inputs` uses `identical(x, 0)`, which is FALSE
    for integer `0L` (not replicated, see "Intentional differences").
  * The lactation validation checks `milk_fat_fraction` for camels, which
    don't use it.
* **Egg-laying flag.** The core functions gate eggs on
  `isTRUE(is_egg_producing)`, which is TRUE only for a logical `TRUE`, while
  the run validators test `is_egg_producing %in% TRUE`, which also matches
  `1` and `"TRUE"`. With validation on, non-logical flags are rejected
  ("must be logical (TRUE/FALSE)"). With validation off
  (`validate_inputs=False`, whose validators return early as on
  `feature/optional-validation-rule`), a flag column coded `1`/`0` or
  `"TRUE"`/`"FALSE"` silently gives no eggs, no egg energy, no egg nitrogen
  and no egg allocation. Python follows R in both places: the core
  functions follow `isTRUE()`, so only a logical `True` (Python `bool`,
  `numpy.bool_` or a `True` of a boolean column) counts, and the energy and
  nitrogen run validators use `%in% TRUE` to decide whether the egg columns
  are required, so a flag of `1` without those columns gets R's
  missing-columns message.

### Model logic

* **CHK ration quality uses pig parameters.** Chickens fall into the `else`
  (pig) branch of the ration digestibility, metabolizable-energy and
  urinary-energy calculations. CHK dry-matter intake, which is computed on
  a metabolizable-energy basis, and therefore all CHK emissions use
  `feed_metabolizable_energy_pigs`. The `feed_metabolizable_energy_chicken`
  column of the example feed table, listed as an input in R's `run_gleam()`
  and `run_emissions_direct()` documentation, is ignored, as are the
  `*_chicken` parameter-range rules.
* **CHK non-laying maintenance does not scale with body weight.** For CHK
  cohorts that are not laying (FJ, FS, MJ, MS, MN and non-laying FN),
  maintenance is `0.3866 + 0.0282 * (LCT - T)` below the lower critical
  temperature and `0.3866 + 0.0037 * (LCT - T)` above it, in MJ per head and
  day, without the `live_weight^0.75` factor that the laying branch and
  every other species use. A 40 g chick and a 1.8 kg pullet get the same
  0.38 MJ/day. The cited source (Sakomura 2004) appears to give these
  coefficients per kg^0.75, with the temperature term increasing above the
  lower critical temperature (`+0.0037 * (T - LCT)`), whereas R's formula
  decreases. **[RL]**
* **CHK `FA` egg energy.** Egg energy, adult maintenance and the 0.0279
  growth coefficient apply to CHK `FA` whatever `is_egg_producing` says,
  although the documentation ties egg energy to the flag.
* **Negative PGS lactation energy.** For PGS adult females,
  `metabolic_energy_req_lactation = litter_size * (1 - 0.5 *
  death_rate_juvenile) * (0.02059 * (lww - lwb) * 1000 / lactation_duration
  - 0.3766 / 0.67) * cadj` is negative whenever piglets gain less than about
  27.3 g/day (`(lww - lwb) * 1000 / lactation_duration`), for example with a
  long `lactation_duration`. No validator checks its sign; only the total
  energy must stay positive. The negative term lowers total energy, intake
  and every emission, so an optimiser can lower emissions with a physically
  meaningless lactation term. Constrain the action space to piglet gains of
  at least 27.3 g/day. **[RL]**
* **Signed convergence test.** The steady-state convergence test in the
  demographic herd model is `all(lambda_change < min_lambda_change)` (signed,
  not absolute as documented). A decreasing lambda always counts as
  converged.
* **Crashes on degenerate herds.**
  * A cohort that stays at zero animals gives `0/0` and R stops with
    "missing value where TRUE/FALSE needed".
  * A juvenile cohort (FJ or MJ) of 1 day passes validation (the range is
    [1, 8000]) but leaves a 0-day juvenile class after the 1-day birth
    class (`cohort_duration_days - 1`): `probability_growth` is
    `(s^-1 - s^0) / (1 - s^0) = Inf`, the stocks become +/-Inf, then NaN,
    and R stops with the same message. Juvenile durations strictly between
    1 and 2 days do not crash but silently give a growth probability above
    1, in R and in Python: with `D = duration - 1` and `h` the daily death
    plus offtake hazard, it is `(e^h - 1) / (e^(h D) - 1)`, at least
    `1 / D` and increasing with `h`. R spreads the juvenile death and
    offtake rates over the cohort's duration, so `h` is large for such
    short cohorts: on the 13 herds of the demographic herd example,
    FJ = 1.5 gives 2.05 to 2.62 (`1 / D` = 2) and FJ = 1.2 gives 5.2 to 8.6
    (`1 / D` = 5).
    Use juvenile durations of at least 2 days; CHK juveniles last 3 days in
    the examples. **[RL]**
  * A non-demographic phase 1 of 0 days returns a bare `0` from
    `calc_nondemo_start_sizes` and the run module fails with "$ operator is
    invalid for atomic vectors".

  Python stops on the same herds; its message also names the cause (a
  non-finite growth probability from a 1-day juvenile cohort, a cohort with
  zero animals, or a phase 1 of 0 days) and the herd (see "Degenerate-herd
  stops raise `GleamValidationError`").
* **Offtake clamped before validation.** Pregnancy energy clamps
  `offtake_rate` to [0, 1] before validation, while the allowed range is
  [-2, 1).
* **Silently dropped feeds.** Feeds missing from `feed_params` are dropped by
  the ration-quality join without renormalising the ration. Feeds missing
  from `feed_emissions` count as zero emissions when `feed_emissions` has no
  `feed_name` column; when it has one, the `feed_name` cross-check rejects
  them.
* **`ratio_m3CH4_to_kgCH4` is ignored.** It is documented as a column of the
  manure factors table, but the run module always uses the default 0.67.
* **Fragile commodity renaming.** `rename_map[commodity_name]` in the
  allocation module indexes by factor codes; it works only because the
  factor levels happen to be in the same order as the map.
* **Global-variable lookup.** `run_all_herd_module` uses
  `exists("herd_level_data_nondemo")`, which can pick up a global variable of
  that name.

### Documentation mismatches

* The `assign_allocation_shares` docs name the residual commodity "None";
  the code uses "Other".
* SHP/GTS FS pregnancy uses 0.077; the documented formula says 0.12 (the
  sentence above it says 0.077).
* SHP/GTS adult females with litter size below 1 silently get a pregnancy
  coefficient of 0.
* `run_gleam()` and `run_emissions_direct()` document a
  `feed_metabolizable_energy_chicken` input that no code reads (see "CHK
  ration quality uses pig parameters").
