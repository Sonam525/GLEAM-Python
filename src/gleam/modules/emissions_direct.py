"""Direct-emissions-only GLEAM pipeline (port of ``R/run_emissions_direct.R``).

From ``feature/run-direct-emissions-only``. In the merged code base the herd
step goes through :func:`gleam.run_all_herd_module` and the ration-quality join
includes ``nondemo_productive_phase_id``, exactly like :func:`gleam.run_gleam`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._utils import Progress, merge_dt
from ..validation._shared import setup_validation
from ..validation.emissions_direct_run import validate_run_emissions_direct_inputs
from .aggregation import run_aggregation_module
from .allocation import run_allocation_module
from .emissions_enteric import run_emissions_enteric_module
from .emissions_manure import run_emissions_manure_module
from .gleam import COHORT_KEYS, build_herd_structure
from .metabolic_energy_req import run_metabolic_energy_req_module
from .nitrogen_balance import run_nitrogen_balance_module
from .production import run_production_module
from .ration_quality import run_ration_quality_module
from .weights import run_weights_module


def run_emissions_direct(
    has_herd_structure: bool = False,
    run_demographic: bool = True,
    run_nondemographic: bool = True,
    cohort_level_data: pd.DataFrame | None = None,
    herd_level_data: pd.DataFrame | None = None,
    feed_rations: pd.DataFrame | None = None,
    feed_params: pd.DataFrame | None = None,
    manure_management_system_fraction: pd.DataFrame | None = None,
    manure_management_system_factors: pd.DataFrame | None = None,
    simulation_duration: float = 365,
    global_warming_potential_set: str = "AR6",
    show_indicator: bool = True,
    emission_factors_only: bool = False,
    validate_inputs: bool = True,
) -> dict[str, Any]:
    """Run GLEAM for direct (enteric and manure) emissions only.

    Feed production emissions are skipped. Ration quality is either derived
    from ``feed_rations`` and ``feed_params`` or supplied directly in
    ``cohort_level_data`` (``ration_gross_energy``,
    ``ration_metabolizable_energy``, ``ration_nitrogen``,
    ``ration_digestibility_fraction``, ``ration_urinary_energy_fraction``,
    ``ration_ash``) when both feed tables are omitted.

    Parameters
    ----------
    emission_factors_only
        If true, stop after the cohort-level enteric and manure emission
        factors; production, allocation and aggregation are skipped and
        ``allocation_long`` / ``aggregation_results`` are ``None``.

    The other parameters are as in :func:`gleam.run_gleam`.

    Returns
    -------
    dict
        ``cohort_level_results``, ``herd_level_results``, ``allocation_long``
        and ``aggregation_results`` (the last two ``None`` when
        ``emission_factors_only``).
    """
    with setup_validation(validate_inputs):
        validate_run_emissions_direct_inputs(
            has_herd_structure=has_herd_structure,
            cohort_level_data=cohort_level_data,
            herd_level_data=herd_level_data,
            feed_rations=feed_rations,
            feed_params=feed_params,
            manure_management_system_fraction=manure_management_system_fraction,
            manure_management_system_factors=manure_management_system_factors,
            simulation_duration=simulation_duration,
            global_warming_potential_set=global_warming_potential_set,
            emission_factors_only=emission_factors_only,
        )
        progress = Progress(show_indicator)
        progress.header("Running GLEAM direct emissions pipeline...")

        chrt, hrd = build_herd_structure(
            has_herd_structure, run_demographic, run_nondemographic,
            cohort_level_data, herd_level_data, simulation_duration, show_indicator,
            validate_inputs,
        )

        chrt = run_weights_module(
            cohort_level_data=chrt, herd_level_data=hrd, show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )["cohort_level_results"]

        if feed_rations is not None:
            rations_summary = run_ration_quality_module(
                rations_share=feed_rations, feed_params=feed_params, show_indicator=show_indicator,
                validate_inputs=validate_inputs,
            )
            chrt = merge_dt(chrt, rations_summary, by=COHORT_KEYS)

        chrt = run_metabolic_energy_req_module(
            cohort_level_data=chrt, herd_level_data=hrd, show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )
        chrt = run_emissions_enteric_module(cohort_level_data=chrt, show_indicator=show_indicator, validate_inputs=validate_inputs)
        chrt = run_nitrogen_balance_module(
            cohort_level_data=chrt, herd_level_data=hrd, show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )
        chrt = run_emissions_manure_module(
            cohort_level_data=chrt,
            manure_management_system_fraction=manure_management_system_fraction,
            manure_management_system_factors=manure_management_system_factors,
            show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )

        allocation_long = None
        aggregation_results = None
        if not emission_factors_only:
            chrt = run_production_module(
                cohort_level_data=chrt,
                herd_level_data=hrd,
                simulation_duration=simulation_duration,
                show_indicator=show_indicator,
            validate_inputs=validate_inputs,
            )
            allocation_results = run_allocation_module(
                cohort_level_data=chrt,
                herd_level_data=hrd,
                simulation_duration=simulation_duration,
                show_indicator=show_indicator,
            validate_inputs=validate_inputs,
            )
            chrt = allocation_results["cohort_allocation_inputs"]
            allocation_long = allocation_results["allocation_long"]
            aggregation_results = run_aggregation_module(
                cohort_level_data=chrt,
                allocation_herd_long=allocation_long,
                simulation_duration=simulation_duration,
                global_warming_potential_set=global_warming_potential_set,
                show_indicator=show_indicator,
            validate_inputs=validate_inputs,
            )
        progress.success("GLEAM direct emissions pipeline complete.")

        return {
            "cohort_level_results": chrt,
            "herd_level_results": hrd,
            "allocation_long": allocation_long,
            "aggregation_results": aggregation_results,
        }
