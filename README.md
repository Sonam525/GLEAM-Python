# gleampy

A Python port of FAO's **Global Livestock Environmental Assessment Model**
(GLEAM-X), translated from the R package
[`un-fao/GLEAM`](https://github.com/un-fao/GLEAM) and checked against it
numerically.

GLEAM quantifies greenhouse-gas emissions from livestock with an IPCC Tier 2,
life-cycle approach. It simulates herd dynamics, computes energy requirements,
feed intake and nitrogen balance, estimates CH4 and N2O from enteric
fermentation and manure, adds feed-production emissions, and allocates
everything to milk, meat, fibre, eggs and draught work. It covers cattle
(CTL), buffalo (BFL), sheep (SHP), goats (GTS), pigs (PGS), camels (CML) and
chickens (CHK).

> **Which R version?** The port targets the union of **unreleased**
> development branches of `un-fao/GLEAM` (chickens, the non-demographic herd
> model and `run_emissions_direct()`) plus one small adaptation, **not** the
> released R package 0.8.0. Released 0.8.0 lacks 12 of the functions ported
> here, and several of its other functions return different tables.
> [tools/r_reference/README.md](tools/r_reference/README.md) lists the exact
> commits and rebuilds that R reference with one script; the result can also
> be browsed as the branch
> [`integration/r-reference`](https://github.com/Sonam525/GLEAM/tree/integration/r-reference)
> of Sonam525/GLEAM. gleampy's version number, 0.8.0, is the `Version` field
> of that reference's DESCRIPTION; it does not mean the port matches the
> released 0.8.0.

This is an independent translation, not an official FAO product.

## Why a Python port

* **Fast and batchable.** The R package evaluates its formulas row by row.
  Here the equation-level core functions are vectorised over numpy arrays,
  and every module runner processes all herds at once. Run times of the full
  `run_gleam()` pipeline (`tools/benchmark.py`, which replicates the bundled
  example herds; one laptop, Snapdragon X Plus, Python 3.12, numpy 2.5,
  pandas 3.0):

  | Herd structure | Herds | Validated | `validate_inputs=False` |
  |---|---|---|---|
  | given (`has_herd_structure=True`) | 15 | 0.12 s | 0.09 s |
  | given | 1,500 | 1.0 s | 0.6 s |
  | simulated (`has_herd_structure=False`, demographic + non-demographic) | 13 | 0.35 s | 0.3 s |
  | simulated | 1,300 | 2.6 s | 2.3 s |

  Timings vary with machine load (up to about a factor of two between
  runs). Simulating the herd is slower because the demographic model
  iterates day by day until the herd structure is stable (up to about 24,000
  days for one of the example herds). Run `python tools/benchmark.py` to
  measure on your machine.
* **Same results.** Every module reproduces the R outputs: same tables, same
  columns, same row order. The tests require numbers to agree within
  1e-9 relative; most agree to the last bit or within a few units in the
  last place. The demographic herd model is the exception: some of its
  formulas are ill-conditioned and amplify 1-ULP differences in R's math
  library, so its outputs, and everything computed from a simulated herd,
  typically agree to 1e-12 to 1e-10 relative, and the herd growth rate of a
  nearly stationary herd only in absolute terms (about 1e-13). Parity
  assumes the R package receives double-typed numeric columns: when `fread`
  reads whole-number columns as integer, R truncates some results, and
  integer products above 2^31 - 1 become NA (for example meat production
  from a given herd structure, after which R stops). Python deliberately
  does neither. Details in [docs/R_PORT_NOTES.md](docs/R_PORT_NOTES.md).

## Installation

The package is installed and imported as `gleampy` (the R package is
`gleam`). It is installed from this repository:

```bash
git clone https://github.com/Sonam525/GLEAM-Python.git
cd GLEAM-Python
pip install -e ".[dev]"   # the package plus pytest and ruff, to run the tests
```

`pip install .` installs the package alone. It requires Python 3.10 or
newer, numpy 1.26 or newer and pandas 2.2 or newer (pandas 2.x or 3.x).

On Windows without long-path support (the default), full paths are limited
to 260 characters. Clone into a directory whose path is shorter than about
120 characters (deeper checkouts exceed the limit when the wheel is built and
when the tests read the golden outputs), and keep the virtual environment's
`site-packages` path under about 160 characters (the longest file of the
installed package adds 98), or enable long paths in Windows.

## Quick start

```python
import gleampy
from gleampy.io import load_example

ex = lambda name: load_example(name, "run_gleam_examples")

results = gleampy.run_gleam(
    has_herd_structure=True,
    cohort_level_data=ex("master_chrt_lvl_structure_data.csv"),
    herd_level_data=ex("master_hrd_lvl_structure_data.csv"),
    feed_rations=ex("feed_rations_share_chrt.csv"),
    feed_params=ex("feed_quality.csv"),
    feed_emissions=ex("feed_emission_factors.csv"),
    manure_management_system_fraction=ex("manure_management_system_fraction.csv"),
    manure_management_system_factors=ex("manure_management_system_factors.csv"),
    simulation_duration=365,
    global_warming_potential_set="AR6",
    show_indicator=False,
)

results["cohort_level_results"]                      # one row per herd x cohort
results["aggregation_results"]["results_emissions"]  # herd totals, allocated, CO2-eq
```

The pipeline can also start without a herd structure: it simulates the
demographic herd and/or the non-demographic production cycles first
(`has_herd_structure=False`, `run_demographic=`, `run_nondemographic=`).
`run_emissions_direct()` computes enteric and manure emissions only.

Each module can be run on its own, and the core equations can be called with
scalars or arrays:

```python
gleampy.run_weights_module(cohort_level_data, herd_level_data)
gleampy.calc_daily_weight_gain(250.0, 41.0, 60.0)         # -> 3.4833...
gleampy.calc_ch4_enteric(
    species_short=["CTL", "SHP"], ch4_conversion_factor_ym=[6.5, 6.7],
    ch4_mitigation_factor=1.0, ration_gross_energy=[18.4, 18.0], ration_intake=[9.8, 1.2],
)                                                          # -> array, kg CH4/head/day
```

Input tables are pandas DataFrames with the same column names as in the R
package. `gleampy.io.read_csv` reads delimited text files the way
`data.table::fread` does with its default arguments: it detects the
separator (comma, tab, pipe or semicolon) and the decimal mark (`.` or `,`),
strips blanks around unquoted fields, treats `NA` as missing in every column
and an empty field as missing in numeric and logical columns but as `""` in
text columns, accepts `NaN`, `Inf` and spreadsheet tokens such as `#N/A` or
`#DIV/0!` in numeric columns, types `TRUE`/`FALSE` columns as logical and
columns without any value as logical NA, and decodes files that are not
valid UTF-8 as Windows-1252. Numbers are parsed with correct rounding. The
`gleampy.io` module documentation gives the exact rules and lists the few
`fread` behaviours that are not reproduced (header auto-detection, row
sampling of large files, quote-rule fallbacks, `integer64`, parser range
limits). The bundled example inputs are available through
`gleampy.io.load_example(name, kind)`.

For repeated runs on inputs that are already validated, pass
`validate_inputs=False` to skip input validation.

`examples/parameter_sweep.py` varies one management lever (daily milk yield)
through the full pipeline and reports emission intensity per kg of milk, the
basic loop for generating surrogate-model training data. Before using the
model as a simulator for optimisation or reinforcement learning, read the
items marked **[RL]** in [docs/R_PORT_NOTES.md](docs/R_PORT_NOTES.md): a few
valid-looking inputs are rejected, crash, or give physically meaningless
results in the R model, and the port reproduces them.

## API

Every function exported by the R reference (83 in all) is available under the
same name with the same argument names; the `run_*` functions also take
`validate_inputs`:

| Pipeline | Modules | Core models |
|---|---|---|
| `run_gleam` | `run_all_herd_module`, `run_demographic_herd_module`, `run_nondemographic_herd_module` | `calc_steady_state_structure`, `calc_nondemo_start_sizes`, ... |
| `run_emissions_direct` | `run_weights_module`, `run_ration_quality_module`, `run_metabolic_energy_req_module` | `calc_cohort_weights`, `calc_metabolic_energy_req_*`, `calc_ration_intake`, ... |
| | `run_emissions_enteric_module`, `run_nitrogen_balance_module`, `run_emissions_manure_module` | `calc_ch4_enteric`, `calc_nitrogen_*`, `calc_ch4_manure`, `calc_n2o_manure_*`, ... |
| | `run_emissions_ration_module`, `run_production_module` | `calc_co2_ration_*`, `calc_milk_production`, `calc_egg_production`, ... |
| | `run_allocation_module`, `run_aggregation_module` | `calc_allocation_shares`, `calc_co2eq`, ... |

The Python package exports a few names that the R package does not:

* `calc_nondemo_phase` and `calc_metabolic_energy_req_eggs`, which are
  internal in R (reachable there as `gleam:::calc_nondemo_phase`);
* `read_csv`, `load_example`, `example_path` and `example_dir` (also in
  `gleampy.io`);
* `GleamValidationError` (raised by every input check, a `ValueError`),
  `GleamWarning`, the context manager `validation_disabled` (turns input
  checks off for direct `calc_*` calls) and the function
  `validation_enabled()` (whether input checks are currently on);
* the `constants` module (species and cohort codes) and `__version__`.

The meaning and units of every input and output column are those of the R
reference's documentation (the `man/` pages of the branches listed in
[tools/r_reference/README.md](tools/r_reference/README.md); the released
0.8.0 reference manual does not cover chickens, non-demographic herds or
`run_emissions_direct()`).

## Package layout

```
src/gleampy/
  constants.py      species and cohort codes, variable metadata (R/gleam_constants.R)
  core/             scientific equations (R/core_model_*.R), vectorised
  modules/          module runners and pipelines (R/run_*.R)
  validation/       input validation (R/validate_*.R)
  io.py             fread-compatible CSV reader, bundled examples
  data/             example inputs and parameter ranges (from the R package)
tests/              ported testthat suites + parity tests against R golden outputs
tests/golden/       R golden outputs
tools/r_reference/  R reference: rebuild script, adaptation patch, golden generator
tools/parity/       randomised R-vs-Python scenario comparison
tools/benchmark.py  run-time benchmark
examples/           usage examples
docs/               contributor guide and R port notes
```

## Verification

```bash
pip install -e ".[dev]"
python -m pytest
```

This runs every ported testthat case and compares all module and pipeline
outputs with golden outputs produced by the R reference (`tests/golden/`).
`tools/parity/` additionally runs randomly perturbed scenarios through both
implementations. CI runs the suite on Linux and Windows, both with the
newest numpy and pandas and with the oldest supported versions, and once
with pyarrow installed (pyarrow-backed strings in pandas 3). See
[docs/R_PORT_NOTES.md](docs/R_PORT_NOTES.md) for the R branches that were
ported, how parity is checked, the intentional differences, and suspected
bugs found in the R package along the way, and
[docs/PORTING_GUIDE.md](docs/PORTING_GUIDE.md) for the conventions to follow
when changing the code.

The golden outputs can be regenerated from the R reference
([tools/r_reference/README.md](tools/r_reference/README.md) explains how to
rebuild it and how to regenerate into a scratch directory without
overwriting `tests/golden/`):

```bash
bash tools/r_reference/build_r_reference.sh ../GLEAM-reference
Rscript tools/r_reference/generate_golden.R ../GLEAM-reference ../golden-new
```

## License and credits

AGPL-3.0-only, like the original R package; see [LICENSE](LICENSE) and
[NOTICE](NOTICE).

GLEAM is developed by the Food and Agriculture Organization of the United
Nations (FAO). The R package `gleam` is Copyright (C) FAO; its authors are
A. Jou, Y. Elaouni, D. Wisser, L. Lanzoni and G. Tempio, with contributors,
and its development was funded by the German Federal Ministry of
Agriculture, Food and Regional Identity (BMLEH).

This Python port is a modified version of that work: the R code was
translated into Python by the GLEAM Python port contributors, and the
bundled example data are copies of the R package's example data (one table
as changed by the adaptation commit). [NOTICE](NOTICE) lists the
modifications; date of the modification notice: 2026-10-08. The port is an
independent translation and is not an official FAO product.
