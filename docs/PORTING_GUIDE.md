# Porting guide: GLEAM R package -> `gleam` Python package

This document defines the conventions used to translate the R package
(`un-fao/GLEAM`) into Python. Every module follows it so the port is uniform,
reviewable, and numerically identical to R.

## Reference R source

* R source tree: `../GLEAM` (local clone of `Sonam525/GLEAM`), branch
  `integration/r-reference` = `feature/new-species-chk` + `feature/herd-nondemo`
  + `main` + `feature/run-direct-emissions-only` (with `run_emissions_direct`
  adapted to the merged herd API). This is the **single source of truth**.
* Do not modify `../GLEAM` or `tests/golden/`.
* R 4.6 is installed: `"$LOCALAPPDATA/Programs/R/R-latest/bin/Rscript.exe"`.
  To inspect intermediate values:
  `Rscript -e 'suppressMessages(pkgload::load_all("../GLEAM", quiet=TRUE, export_all=TRUE)); <code>'`
  (`load_all` takes ~10 s; R is slow because it evaluates row by row).

## File mapping

| R file | Python file |
|---|---|
| `R/core_model_<m>.R` | `src/gleam/core/<m>.py` |
| `R/validate_<m>_core_model.R` | `src/gleam/validation/<m>_core.py` |
| `R/run_<m>_module.R` | `src/gleam/modules/<m>.py` |
| `R/validate_run_<m>_inputs.R` | `src/gleam/validation/<m>_run.py` |
| `tests/testthat/test-<m>_core.R` | `tests/test_<m>_core.py` |
| golden outputs `tests/golden/<case>/` | `tests/test_parity_<m>.py` |

Shared, already-written files (do **not** edit; work around and report gaps):
`src/gleam/_utils.py`, `src/gleam/validation/_shared.py`,
`src/gleam/constants.py`, `src/gleam/io.py`, `src/gleam/__init__.py`,
`tests/golden_utils.py`. Private helpers needed by one module go in that
module's own file.

## Core-model functions (`calc_*`)

* Same function name, argument names, argument order and defaults as R.
  `NA_real_` -> `np.nan`, `NA_character_` -> `None`, `NULL` -> `None`.
* **Vectorised**: R calls these row by row (`by = .I`) with scalars; Python
  accepts scalars, lists, numpy arrays or pandas Series and broadcasts them.
  Convert with `as_float` / `as_str` / `as_bool` from `gleam._utils`.
* Every scalar `if / else if / else` branch in R becomes a boolean mask
  (`np.select`, `np.where`) that reproduces the branch **for each element**.
  Branch on string codes with `==` on object arrays or `isin`.
* Return type: if all inputs are scalars return a Python `float` (use
  `finalize(out, all_scalar(...))`), otherwise a 1-d `np.ndarray`. An R
  function returning a named `list(...)` returns a `dict` with the same keys
  in the same order (use `finalize_dict`).
* Keep the arithmetic in the **same order** as R where practical (floating-
  point associativity); parity tolerance is `rtol=1e-9`.
* Use `np.errstate(divide="ignore", invalid="ignore")` around divisions: R
  gives `Inf` / `NaN` silently.
* Keep the R docstring content as a concise numpy-style docstring (purpose,
  parameters with units, returns, formula). Do not copy the long roxygen
  verbatim; keep the essential scientific description and equations.

## Validation

* The R `validate_*` functions called inside `calc_*` become vectorised
  functions in `validation/<m>_core.py` that check all elements at once and
  raise `GleamValidationError` (a `ValueError`) with the R message (minus cli
  markup; use backticks for argument names, e.g. "`offtake_rate` must be ...").
* Every validation function must be a no-op when `validation_enabled()` is
  false (decorate with `@validator` from `gleam.validation._shared`).
* Use the shared helpers (`validate_param_range`, `validate_scalar_numeric`,
  `check_required_columns`, ...) exactly where R uses them.
* R warnings (`cli::cli_warn`) -> `warn(...)` (category `GleamWarning`).

## Module runners (`run_*`)

* Signature = R signature plus `validate_inputs: bool = True` as the last
  argument (from `feature/optional-validation-rule`). The body runs inside
  `with setup_validation(validate_inputs):`.
* `show_indicator=True` prints progress via `gleam._utils.Progress`.
* Inputs are pandas DataFrames; **never mutate the caller's objects**
  (`copy_frame`). Outputs are DataFrames / dicts of DataFrames with the same
  keys as the R list.
* Output tables must have the **same columns, in the same order, and the
  same row order** as R. Rules that reproduce data.table behaviour:
  * `DT[, col := expr]` adds the column at the end (or overwrites in place),
    row order unchanged;
  * `merge(x, y, by = ...)` -> `merge_dt(x, y, by=...)` (inner join, sorted by
    keys with NA first, `.x`/`.y` suffixes, NA keys match);
    `all.x = TRUE` -> `all_x=True`; `all = TRUE` -> `all_x=True, all_y=True`;
  * `herd_level_data[.SD, on = "herd_id", x.col]` -> `lookup(cohort, herd, "col")`;
  * `DT[i, j, by = g]` aggregations keep groups in **first-appearance order**
    (`groupby(..., sort=False)`), `keyby` sorts;
  * `rbindlist(..., fill = TRUE)` -> `rbind_fill`;
  * `unique()` keeps first occurrences; `setorder`/`setkey` -> `order_by`;
  * `melt` / `dcast`: reproduce data.table's column order and naming.
* Missing values: R `sum(x)` is `NA` if any `NA` (pandas skips NaN — use
  `r_sum` or `min_count`/`skipna=False`); `sum(x, na.rm = TRUE)` of all-NA
  is `0`. `NA > 0` is `NA` and `ifelse(NA, a, b)` is `NA` (`na_cmp`, `ifelse`).
  In data.table `DT[cond, x := v]` rows where `cond` is `NA` are **not**
  assigned.
* Logical columns are object arrays of `True`/`False`/`None`; use `is_true`
  for `isTRUE()` semantics.
* Integer-valued columns that R stores as integer may be int or float in
  pandas; parity tests compare numerically, so either is fine.

## Tests

* Port every `test_that(...)` block of the corresponding testthat file into
  pytest (`def test_<descriptive_name>():`), keeping the same inputs and
  expected values. `expect_equal(a, b)` -> `pytest.approx(b, rel=1.5e-8)`
  (testthat's tolerance), `expect_error(expr, "pattern")` ->
  `pytest.raises(GleamValidationError, match="pattern")` (adapt the pattern if
  R cli markup differs, but keep its intent).
* Add golden parity tests (`tests/test_parity_<m>.py`) that run the module on
  the bundled example inputs exactly as `tools/r_reference/generate_golden.R`
  does and compare every output table with
  `golden_utils.assert_matches_golden(df, case, table)` (default `rtol=1e-9`,
  exact column order and row order). Loosening a check requires a comment
  explaining why.
* Run tests with `.venv/Scripts/python -m pytest tests/<file> -q`.
