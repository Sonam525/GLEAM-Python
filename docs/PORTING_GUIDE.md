# Contributor guide: conventions for the R-to-Python port

gleampy is a translation of the GLEAM R package that must stay numerically
faithful to it, down to R's order of floating-point operations. This guide gives the conventions every module
follows, so changes stay uniform, reviewable and identical to R in their
results. [R_PORT_NOTES.md](R_PORT_NOTES.md) lists the intentional
differences and the R bugs that are, or deliberately are not, reproduced.

## Setting up

```bash
git clone https://github.com/Sonam525/GLEAM-Python.git
cd GLEAM-Python
python -m venv .venv            # then activate it
pip install -e ".[dev]"         # the package plus pytest and ruff
python -m pytest                # unit tests + parity with the R golden outputs
ruff check src tools tests --select E9,F
```

### The R reference

The R code that is the source of truth is not a released version of the R
package. Rebuild it next to this repository (needs git and bash):

```bash
bash tools/r_reference/build_r_reference.sh ../GLEAM-reference
```

[tools/r_reference/README.md](../tools/r_reference/README.md) explains what
it contains, which R packages it needs (R >= 4.4 with data.table, cli,
pkgload; testthat for its tests) and how to regenerate golden outputs
**into a scratch directory**. Never edit `tests/golden/` by hand, and never
point the golden generator at it directly.

To look at intermediate R values, put the code in a script and run it with
`Rscript file.R`:

```r
suppressMessages(pkgload::load_all("../GLEAM-reference", quiet = TRUE, export_all = TRUE))
dt <- data.table::fread("some_input.csv")
print(calc_cohort_weights(...), digits = 17)
```

`export_all = TRUE` also exposes R's internal functions. Loading takes a few
seconds, and the R code runs row by row, so keep inputs small. To compare
numbers bit for bit, write doubles with `sprintf("%a", x)` (from a script,
not from `Rscript -e`) and read them in Python with `float.fromhex`.

## File mapping

| R file | Python file |
|---|---|
| `R/core_model_<m>.R` | `src/gleampy/core/<m>.py` |
| `R/validate_<m>_core_model.R` | `src/gleampy/validation/<m>_core.py` |
| `R/run_<m>_module.R` | `src/gleampy/modules/<m>.py` |
| `R/validate_run_<m>_inputs.R` | `src/gleampy/validation/<m>_run.py` |
| `R/gleam_constants.R` | `src/gleampy/constants.py` |
| `tests/testthat/test-<m>_core.R` | `tests/test_<m>_core.py` |
| golden outputs `tests/golden/<case>/` | `tests/test_parity_<m>.py` |

Shared helpers live in `src/gleampy/_utils.py` (type conversion, R-like `NA`
handling, data.table equivalents) and `src/gleampy/validation/_shared.py`
(validation primitives, `GleamValidationError`, `GleamWarning`, the
validation switch). Use them rather than re-implementing their logic, and
extend them when a module needs something new. A public function is
exported by registering it in `src/gleampy/__init__.py` (`_register(...)`),
under the same name as in R.

## Core-model functions (`calc_*`)

* Same function name, argument names, argument order and defaults as R.
  `NA_real_` -> `np.nan`, `NA_character_` -> `None`, `NULL` -> `None`.
* **Vectorised.** R calls these row by row (`by = .I`) with scalars; Python
  accepts scalars, lists, numpy arrays or pandas Series and broadcasts them.
  Convert inputs with `as_float` / `as_str` / `as_bool` from
  `gleampy._utils`.
* Every scalar `if / else if / else` branch in R becomes a boolean mask
  (`np.select`, `np.where`) that reproduces the branch **for each element**.
  Branch on string codes with `==` on object arrays or `isin`.
* Return type: if all inputs are scalars return a Python `float` (use
  `finalize(out, all_scalar(...))`), otherwise a 1-d `np.ndarray`. An R
  function returning a named `list(...)` returns a `dict` with the same keys
  in the same order (use `finalize_dict`).
* Keep the arithmetic in the **same order** as R (floating-point addition and
  multiplication are not associative), and write `x^2` as `x * x` where R's
  `^` does (`R_pow` squares exactly).
* Use `np.errstate(divide="ignore", invalid="ignore")` around divisions: R
  gives `Inf` / `NaN` silently.
* Docstrings are numpy style: a short description with the equation, a
  **Parameters** section with units (kg, MJ/day, days, fraction, ...), and
  **Returns**. Keep the essential scientific content of the R roxygen
  documentation, not its full text.

## Numbers and types

* **Always compute in float64**, whatever the dtype of the input column
  (int64, nullable `Int64`/`Float64`, float64, object, categorical);
  `as_float` does this. Never truncate to integer. In R, data.table's
  `:=` with `by = .I` types each output column from its first row, so
  integer-typed inputs (from `fread`) can truncate later results. That is
  an R bug the port deliberately does not reproduce; results must not
  depend on pandas dtypes either.
* Logical flags follow R's `isTRUE()`: use `is_true` (element-wise) or
  `is_true_scalar` (pipeline switches). Only a logical `True` counts (Python
  `bool`, `numpy.bool_`, `True` in a boolean column); `1`, `1.0` and
  `"TRUE"` do not. Validation rejects non-logical flags, as R does. Where R
  tests `x %in% TRUE` instead (the nitrogen-balance and metabolic-energy
  run validators decide whether the egg columns are required with
  `any(is_egg_producing %in% TRUE)`), use `in_true`, which, like `%in%`,
  also matches `1` and `"TRUE"`.
* Pandas compatibility: the package supports pandas 2.2+ and 3.x and numpy
  1.26+ and 2.x. Group by plain values (convert categorical keys first, or
  pass `observed=True`), and handle nullable extension dtypes with missing
  values in every conversion.

## Validation

* The R `validate_*` functions called inside `calc_*` become vectorised
  functions in `validation/<m>_core.py` that check all elements at once and
  raise `GleamValidationError` (a `ValueError`) with the R message, minus cli
  markup: use backticks for argument names ("`` `offtake_rate` `` must be
  ...").
* Every validation function must be a no-op when `validation_enabled()` is
  false (decorate with `@validator` from `gleampy.validation._shared`).
* Use the shared helpers (`validate_param_range`, `validate_scalar_numeric`,
  `check_required_columns`, ...) exactly where R uses them. A scalar
  argument in R (for example `simulation_duration`) must be rejected when a
  vector is passed.
* While validation is on, a missing input column that a row needs raises
  `GleamValidationError` ("Missing required columns in `` `arg` ``: ..."),
  never a bare `KeyError`, and is never silently replaced by NaN. Keep R's
  rules for which rows need which columns. A check made in the module code,
  where R reads the column, also runs with validation off (R stops there
  too), so it calls `abort` directly: `check_required_columns` does nothing
  while validation is off.
* Group sums that are compared with a tolerance (ration fractions sum to 1
  within 1e-6, manure fractions within 1e-8) use `group_sum`: plain
  left-to-right double addition in row order, like data.table's `gsum`, not
  pandas' compensated sums.
* R warnings (`cli::cli_warn`) -> `warn(...)` (category `GleamWarning`).

## Module runners (`run_*`)

* Signature = R signature plus `validate_inputs: bool = True` as the last
  argument (from `feature/optional-validation-rule`). The body runs inside
  `with setup_validation(validate_inputs):`.
* `show_indicator=True` prints progress via `gleampy._utils.Progress`.
* Inputs are pandas DataFrames; **never mutate the caller's objects**
  (`copy_frame`). Outputs are DataFrames / dicts of DataFrames with the same
  keys as the R list.
* Output tables must have the **same columns, in the same order, and the
  same row order** as R. Rules that reproduce data.table behaviour:
  * `DT[, col := expr]` adds the column at the end (or replaces it in place,
    as a float64 column), row order unchanged;
  * `merge(x, y, by = ...)` -> `merge_dt(x, y, by=...)` (inner join, sorted by
    keys with NA first, `.x`/`.y` suffixes, NA keys match);
    `all.x = TRUE` -> `all_x=True`; `all = TRUE` -> `all_x=True, all_y=True`;
  * `herd_level_data[.SD, on = "herd_id", x.col]` -> `lookup(cohort, herd, "col")`;
  * `DT[i, j, by = g]` aggregations keep groups in **first-appearance order**
    (`groupby(..., sort=False)`), `keyby` sorts;
  * `rbindlist(..., fill = TRUE)` -> `rbind_fill`;
  * `unique()` keeps first occurrences; `setorder`/`setkey` -> `order_by`;
  * `melt` / `dcast`: reproduce data.table's column order and naming.
* Missing values: R `sum(x)` is `NA` if any `NA` (pandas skips NaN; use
  `r_sum` or `min_count`/`skipna=False`); `sum(x, na.rm = TRUE)` of all-NA
  is `0`. `NA > 0` is `NA` and `ifelse(NA, a, b)` is `NA` (`na_cmp`,
  `ifelse`). In data.table `DT[cond, x := v]` rows where `cond` is `NA` are
  **not** assigned.
* Logical columns are object arrays of `True` / `False` / `None`.
* The dtype of an output column that R stores as integer (for example
  counts or ids) may be int or float in pandas; the parity tests compare
  values. This is about how results are stored, never about computing them
  (see "Numbers and types").

## Performance changes

A change made only for speed must leave every result bit-identical. Besides
the golden tests, compare the outputs of a large replicated run before and
after the change:

```bash
python tools/benchmark.py 100 --save before.pkl     # on the old code
python tools/benchmark.py 100 --compare before.pkl  # on the new code
```

## Tests

* Port every `test_that(...)` block of the corresponding testthat file to
  pytest (`def test_<descriptive_name>():`), keeping the same inputs and
  expected values. `expect_equal(a, b)` -> `pytest.approx(b, rel=1.5e-8)`
  (testthat's tolerance); `expect_error(expr, "pattern")` ->
  `pytest.raises(GleamValidationError, match="pattern")` (adapt the pattern
  if the R cli markup differs, but keep its intent).
* Add golden parity tests (`tests/test_parity_<m>.py`) that run the module
  on the same inputs as `tools/r_reference/generate_golden.R` and compare
  every output table with `golden_utils.assert_matches_golden(df, case,
  table)`: exact column and row order, `rtol = 1e-9` relative, no absolute
  tolerance (a value R computes as exactly zero must be exactly zero).
  Loosening a check needs a comment explaining why.
* A behaviour the bundled examples never reach needs its own golden case:
  build its inputs in `tests/data/` (`tests/data/build_golden_inputs.py`),
  add the case to `generate_golden.R`, regenerate it into a scratch
  directory and copy only that case into `tests/golden/`.
* Run one file with `python -m pytest tests/<file> -q`.
