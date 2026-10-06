"""gleam: Python port of FAO's Global Livestock Environmental Assessment Model (GLEAM-X).

The public API mirrors the exports of the R package ``gleam``: every
``run_*`` module function and every ``calc_*`` core-model function keeps its R
name and argument names. Tables are pandas DataFrames; core-model functions
are vectorised over numpy arrays (scalars in, scalar out).

>>> import gleam
>>> res = gleam.run_gleam(...)            # full pipeline
>>> gleam.calc_daily_weight_gain(250, 41, 60)
3.4833333333333334
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.8.0"

# Public name -> submodule that defines it (mirrors the R NAMESPACE exports).
_EXPORTS: dict[str, str] = {}


def _register(module: str, *names: str) -> None:
    for n in names:
        _EXPORTS[n] = module


# --- Pipelines and module runners --------------------------------------------
_register("gleam.modules.gleam", "run_gleam")
_register("gleam.modules.emissions_direct", "run_emissions_direct")
_register("gleam.modules.all_herd", "run_all_herd_module")
_register("gleam.modules.demographic_herd", "run_demographic_herd_module")
_register("gleam.modules.nondemographic_herd", "run_nondemographic_herd_module")
_register("gleam.modules.weights", "run_weights_module")
_register("gleam.modules.ration_quality", "run_ration_quality_module")
_register("gleam.modules.metabolic_energy_req", "run_metabolic_energy_req_module")
_register("gleam.modules.emissions_enteric", "run_emissions_enteric_module")
_register("gleam.modules.nitrogen_balance", "run_nitrogen_balance_module")
_register("gleam.modules.emissions_manure", "run_emissions_manure_module")
_register("gleam.modules.emissions_ration", "run_emissions_ration_module")
_register("gleam.modules.production", "run_production_module")
_register("gleam.modules.allocation", "run_allocation_module")
_register("gleam.modules.aggregation", "run_aggregation_module")

# --- Core models --------------------------------------------------------------
_register("gleam.core.weights", "calc_cohort_weights", "calc_avg_weights", "calc_daily_weight_gain")
_register(
    "gleam.core.demographic_herd",
    "calc_fecundity_rates", "calc_transition_probabilities", "calc_steady_state_structure",
    "calc_projected_population_size", "calc_summary_offtake",
)
_register(
    "gleam.core.nondemographic_herd",
    "assign_nondemographic_phase_durations", "calc_nondemographic_total_durations",
    "calc_nondemo_cycle_geometry", "calc_nondemo_start_sizes", "calc_nondemo_phase",
    "calc_nondemo_avg_stock_phase_horizon", "calc_nondemo_offtake_total_horizon",
)
_register("gleam.core.all_herd", "rescale_x_to_y")
_register(
    "gleam.core.ration_quality",
    "calc_feed_digestibility_fraction", "calc_ration_digestibility", "calc_ration_metabolizable_energy",
    "calc_ration_gross_energy", "calc_ration_nitrogen_content", "calc_ration_urinary_energy_fraction",
    "calc_ration_ash",
)
_register(
    "gleam.core.metabolic_energy_req",
    "calc_metabolic_energy_req_maintenance", "calc_metabolic_energy_req_activity",
    "calc_metabolic_energy_req_growth", "calc_metabolic_energy_req_lactation",
    "calc_metabolic_energy_req_eggs", "calc_metabolic_energy_req_work", "calc_metabolic_energy_req_fibre",
    "calc_metabolic_energy_req_pregnancy", "calc_rem_maintenance", "calc_reg_growth",
    "calc_total_metabolic_energy_req", "calc_ration_intake",
)
_register("gleam.core.emissions_enteric", "calc_conversion_factor_ym", "calc_ch4_enteric")
_register("gleam.core.nitrogen_balance", "calc_nitrogen_intake", "calc_nitrogen_retention", "calc_nitrogen_excretion")
_register(
    "gleam.core.emissions_manure",
    "calc_volatile_solids", "calc_ch4_manure", "calc_n2o_manure_direct",
    "calc_n2o_manure_volatilization", "calc_n2o_manure_leaching", "calc_n2o_manure_total",
)
_register(
    "gleam.core.emissions_ration",
    "calc_co2_ration_fertilizer", "calc_co2_ration_pesticides", "calc_co2_ration_crop_activities",
    "calc_co2_ration_luc_nopeat", "calc_co2_ration_luc_peat", "calc_n2o_ration_fertilizer",
    "calc_n2o_ration_manure", "calc_n2o_ration_crop_residues", "calc_ch4_ration_rice",
)
_register(
    "gleam.core.production",
    "calc_milk_production", "calc_fibre_production", "calc_egg_production", "calc_meat_production",
)
_register(
    "gleam.core.allocation",
    "calc_milk_allocation_energy", "calc_meat_allocation_energy", "calc_fibre_allocation_energy",
    "calc_work_allocation_energy", "calc_egg_allocation_energy", "calc_cohort_to_herd_aggregation",
    "calc_allocation_shares", "assign_allocation_shares",
)
_register("gleam.core.aggregation", "calc_cohort_totals", "calc_allocated_emissions", "calc_co2eq")

# --- Utilities ------------------------------------------------------------------
_register("gleam.io", "read_csv", "load_example", "example_path", "example_dir")
_register(
    "gleam.validation._shared",
    "GleamValidationError", "GleamWarning", "validation_disabled", "validation_enabled",
)

__all__ = sorted(_EXPORTS) + ["constants", "__version__"]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'gleam' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


from . import constants  # noqa: E402  (always available)
