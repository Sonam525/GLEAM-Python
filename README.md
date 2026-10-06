# gleam (Python)

A Python port of FAO's **Global Livestock Environmental Assessment Model**
(GLEAM-X), translated from the official R package
[`un-fao/GLEAM`](https://github.com/un-fao/GLEAM) and verified against it
numerically.

GLEAM quantifies greenhouse-gas emissions from livestock with an IPCC Tier 2,
life-cycle approach. It simulates herd dynamics, computes energy requirements,
feed intake and nitrogen balance, estimates CH4 and N2O from enteric
fermentation and manure, adds feed-production emissions, and allocates
everything to milk, meat, fibre, eggs and draught work. It covers cattle
(CTL), buffalo (BFL), sheep (SHP), goats (GTS), pigs (PGS), camels (CML) and
chickens (CHK).

## Why a Python port

* **Fast and batchable.** The R package evaluates its formulas row by row;
  here every core function is vectorised over numpy arrays. The full pipeline
  runs in about 0.1 s for the 15 example herds and about 1-2 s for 1,500
  herds, which makes it practical as a simulator for surrogate models and
  reinforcement learning.
* **Same results.** Every module reproduces the R outputs: same tables, same
  columns, same row order, numbers within 1e-11 relative.

## Installation

```bash
pip install -e .
```

Requires Python 3.10 or newer, numpy and pandas (2.x or 3.x).

## Quick start

```python
import gleam
from gleam.io import load_example

ex = lambda name: load_example(name, "run_gleam_examples")

results = gleam.run_gleam(
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
gleam.run_weights_module(cohort_level_data, herd_level_data)
gleam.calc_daily_weight_gain(250.0, 41.0, 60.0)           # -> 3.4833...
gleam.calc_ch4_enteric(
    species_short=["CTL", "SHP"], ch4_conversion_factor_ym=[6.5, 6.7],
    ch4_mitigation_factor=1.0, ration_gross_energy=[18.4, 18.0], ration_intake=[9.8, 1.2],
)                                                          # -> array, kg CH4/head/day
```

Input tables are pandas DataFrames with the same column names as in the R
package. `gleam.io.read_csv` reads CSV files the way `data.table::fread`
does, and the bundled example inputs are available through
`gleam.io.load_example(name, kind)`.

For repeated runs on inputs that are already validated, pass
`validate_inputs=False` to skip input validation.

`examples/parameter_sweep.py` varies one management lever (daily milk yield)
through the full pipeline and reports emission intensity per kg of milk, the
basic loop for generating surrogate-model training data.

## API

The public API mirrors the R package exports one to one (same names and
argument names):

| Pipeline | Modules | Core models |
|---|---|---|
| `run_gleam` | `run_all_herd_module`, `run_demographic_herd_module`, `run_nondemographic_herd_module` | `calc_steady_state_structure`, `calc_nondemo_phase`, ... |
| `run_emissions_direct` | `run_weights_module`, `run_ration_quality_module`, `run_metabolic_energy_req_module` | `calc_cohort_weights`, `calc_metabolic_energy_req_*`, `calc_ration_intake`, ... |
| | `run_emissions_enteric_module`, `run_nitrogen_balance_module`, `run_emissions_manure_module` | `calc_ch4_enteric`, `calc_nitrogen_*`, `calc_ch4_manure`, `calc_n2o_manure_*`, ... |
| | `run_emissions_ration_module`, `run_production_module` | `calc_co2_ration_*`, `calc_milk_production`, `calc_egg_production`, ... |
| | `run_allocation_module`, `run_aggregation_module` | `calc_allocation_shares`, `calc_co2eq`, ... |

See the R package's reference documentation for the meaning and units of
every input and output column; they are unchanged.

## Package layout

```
src/gleam/
  constants.py      species and cohort codes, variable metadata (R/gleam_constants.R)
  core/             scientific equations (R/core_model_*.R), vectorised
  modules/          module runners and pipelines (R/run_*.R)
  validation/       input validation (R/validate_*.R)
  io.py             fread-compatible CSV reader, bundled examples
  data/             example inputs and parameter ranges
tests/              ported testthat suites + parity tests against R golden outputs
tools/r_reference/  R script that regenerates the golden outputs
tools/parity/       randomised R-vs-Python scenario comparison
tools/benchmark.py  run-time benchmark
examples/           usage examples
docs/               porting guide and R port notes
```

## Verification

```bash
python -m pytest
```

This runs every ported testthat case and compares all module and pipeline
outputs with golden outputs produced by the R package
(`tests/golden/`, 30 cases). `tools/parity/` additionally runs randomly
perturbed scenarios through both implementations. See
[docs/R_PORT_NOTES.md](docs/R_PORT_NOTES.md) for the R branches that were
ported, how parity is checked, the intentional differences, and suspected
bugs found in the R package along the way.

To regenerate the golden outputs from an R checkout:

```bash
Rscript tools/r_reference/generate_golden.R ../GLEAM tests/golden
```

## License and credits

AGPL-3.0, like the original R package. GLEAM is developed by the Food and
Agriculture Organization of the United Nations (FAO); the R package `gleam`
is by A. Jou, Y. Elaouni, D. Wisser, L. Lanzoni, G. Tempio and contributors,
with funding from the German Federal Ministry of Agriculture, Food and
Regional Identity (BMLEH). This Python port is an independent translation and
is not an official FAO product.
