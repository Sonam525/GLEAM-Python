# R to Python port notes

## Which R code was ported

The R package lives on several branches. The port targets their union, built
as the local branch `integration/r-reference` of the `Sonam525/GLEAM` fork:

| Source | Commit | Content |
|---|---|---|
| `feature/new-species-chk` | `50f3c39` | chickens (CHK), egg production and allocation |
| `feature/herd-nondemo` | `f3d1254` | non-demographic herd module, all-herd orchestrator |
| `main` (v0.8.0) | `90e4161` | released package, CRAN fixes |
| `feature/run-direct-emissions-only` | `aa8915d` | `run_emissions_direct()` |
| `feature/optional-validation-rule` | `6a66d86` | `validate_inputs` switch (re-implemented, not merged) |

The merge is clean. One adaptation was needed (commit `a6f5215` on the
integration branch): `run_emissions_direct()` was written against `main` and
had to call `run_all_herd_module()`, add the optional non-demographic herd
columns and join ration quality on `nondemo_productive_phase_id`, exactly as
`run_gleam()` does on `feature/herd-nondemo`. Its example herd table gained
`prop_nondemo_fem_juv = prop_nondemo_mal_juv = 0` and its tests were pointed
at the direct-emissions example inputs. The adapted R package passes all its
tests (627 + 122 expectations).

Not ported:

* `feature/specie-cohort-specific-validation` only reorganises input
  validation and conflicts heavily with the chicken branch.
* `release/v1-pipeline` is an obsolete early pipeline.
* `copilot/*` contain only empty "Initial plan" commits.
* `cran_*` and `dommens-*` are fully merged into `main`.
* The `gleam/` folder on `main` is an accidentally committed, stale snapshot
  of `feature/herd-nondemo`.

## How parity is verified

1. **Golden outputs.** `tools/r_reference/generate_golden.R` runs every
   `run_*` example of the R package and saves all output tables
   (`tests/golden/`). The pytest suite reproduces each call in Python and
   compares every table: same columns, same column order, same row order,
   numbers within `rtol = 1e-9`. That covers 30 cases: every module example,
   the three `run_gleam()` examples (plus all four GWP sets and a 180-day
   assessment) and five `run_emissions_direct()` variants.
2. **Ported unit tests.** Every `testthat` block of the R package is ported
   to pytest with the same inputs and expected values.
3. **Randomised scenarios.** `tools/parity/` perturbs the pipeline examples
   (weights, yields, rates, feed and manure parameters, ration and manure
   shares, assessment length, GWP set), runs each scenario through R and
   Python, and compares all outputs. R and Python must also fail on the
   same scenarios with the same validation error. Python receives the
   inputs exactly as R parsed them (hex floats), so the comparison is
   bit-for-bit on inputs. Result on 30 scenarios (10 each from the
   herd-structure, mixed demographic + non-demographic and CHK
   non-demographic examples):
   * 23 ran successfully and matched R on every output table;
   * the other 7 were rejected by both implementations, by the same
     ULP-sensitive weight check (see below) on the same rows.
4. **Bit-level checks.** While porting, each module was also compared with R
   on hex-dumped doubles. All arithmetic reproduces R's order of operations;
   most modules are bit-identical. The remaining ULP-level differences come
   from R's math library on the reference machine (Windows ARM64, R running
   as emulated x86_64): its `^`/`exp`/`log` are up to a few ULP from the
   correctly rounded result that numpy returns, and its `fread`/`sprintf`
   are not correctly rounded. The largest relative difference over the
   ~90,000 non-zero numeric outputs of the 12 pipeline golden cases is
   6.6e-12.

## Intentional differences from R

* **Vectorised core models.** R evaluates the `calc_*` functions row by row
  (`by = .I`) on scalars. Python evaluates them on whole numpy arrays, with
  one boolean mask per R `if` branch. Scalars in give a scalar out. Per-herd
  functions of the demographic herd model keep R's per-herd signature
  (named vectors become dicts / Series).
* **Validation order.** Checks run over all rows at once, category by
  category. When several rows are invalid for different reasons, the first
  error reported can differ from R's first row-wise error; vector messages
  name the failing element (`` `x`[3] = ... ``).
* **No side effects on inputs.** R modifies some input data.tables by
  reference (adds `is_egg_producing`, sorts `herd_level_data`, adds NA
  columns to `herd_level_data`). Python never mutates the caller's tables.
* **`validate_inputs=False`** (every `run_*` function) skips all checks and
  warns once, as on `feature/optional-validation-rule`.
* **Undefined branches.** Where R crashes because no branch assigned a
  result (unknown species with validation off, `object '...' not found`),
  Python returns NaN or raises `GleamValidationError`.
* **Progress output** goes to stderr; `show_indicator=False` silences it.
* Error messages drop the `cli` markup (`{.arg x}` becomes `` `x` ``) and use
  `>=` / `<=` instead of `≥` / `≤`.

## Suspected bugs in the R package

These are reproduced faithfully in Python (so results stay identical) and
are listed here so they can be reported upstream.

### Validation

* **Exact float comparison in the weights check.** The check
  `live_weight_cohort_initial <= live_weight_cohort_average <=
  live_weight_cohort_final` compares doubles exactly.
  `final = w*(1-r) + w*r` can be 1 ULP below `w`, so adult cohorts and CHK
  juveniles randomly fail for ordinary offtake rates (seen in randomised
  scenarios).
* **`cohort_stock_size` of 0.** The run validator allows `>= 0`, but
  `calc_cohort_totals` requires a positive value, so any empty cohort aborts
  aggregation.
* **NA ration fractions.** An NA `feed_ration_fraction` makes the
  sum-to-one check pass silently (NA sum).
* **Rest duration of 0.** `rest_between_nondemo_cycles_duration = 0` is
  rejected because the `duration` range starts at 1.
* **Truncated error messages.** `validate_run_aggregation_inputs.R` passes
  two strings to `cli_abort`, so the second half is dropped.
  `validate_run_emissions_manure_inputs.R` reports only the first invalid
  group.
* **Effectively required columns.**
  * `run_weights_module` reads all 14 herd weight columns unconditionally,
    so non-demographic columns are required even for demographic-only data.
  * Its validator also always requires `species_short`.
  * `nondemo_productive_phase_id` only works when absent thanks to lazy
    evaluation.
* **Fibre for non-demographic cohorts.** Fibre production and the fibre
  energy validation skip FN/MN, although the calculations use
  `fibre_yield_year` for them. A negative value passes and gives negative
  fibre.
* **Inconsistent milk checks.**
  * `validate_allocation_milk_inputs` uses `identical(x, 0)`, which is FALSE
    for integer `0L`.
  * The lactation validation checks `milk_fat_fraction` for camels, which
    don't use it.

### Model logic

* **CHK ration quality uses pig parameters.** Chickens fall into the `else`
  branch of the ration-quality digestibility and urinary-energy
  calculations. The `*_chicken` parameter-range rules are never used.
* **CHK `FA` egg energy.** Egg energy, adult maintenance and the 0.0279
  growth coefficient apply to CHK `FA` whatever `is_egg_producing` says,
  although the documentation ties egg energy to the flag.
* **Signed convergence test.** The steady-state convergence test in the
  demographic herd model is `all(lambda_change < min_lambda_change)` (signed,
  not absolute as documented). A decreasing lambda always counts as
  converged.
* **Crashes on degenerate herds.**
  * A cohort that stays at zero animals gives `0/0` and R stops with
    "missing value where TRUE/FALSE needed".
  * A non-demographic phase 1 of 0 days returns a bare `0` from
    `calc_nondemo_start_sizes` and the run module fails with "$ operator is
    invalid for atomic vectors".
* **Offtake clamped before validation.** Pregnancy energy clamps
  `offtake_rate` to [0, 1] before validation, while the allowed range is
  [-2, 1).
* **Silently dropped feeds.** Feeds missing from `feed_params` are dropped by
  the ration-quality join without renormalising the ration. Feeds missing
  from `feed_emissions` count as zero emissions.
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
* SHP/GTS FS pregnancy uses 0.077 while the docs say 0.12.
* SHP/GTS adult females with litter size below 1 silently get a pregnancy
  coefficient of 0.
