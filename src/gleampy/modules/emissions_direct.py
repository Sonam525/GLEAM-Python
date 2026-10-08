"""Direct-emissions-only GLEAM pipeline (port of ``R/run_emissions_direct.R``).

From ``feature/run-direct-emissions-only``. In the merged code base the herd
step goes through :func:`gleampy.run_all_herd_module` and the ration-quality join
includes ``nondemo_productive_phase_id``, exactly like :func:`gleampy.run_gleam`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._utils import Progress, is_true_scalar, merge_dt
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

    The logical switches follow R's ``isTRUE()`` and treat a Python ``bool``
    and a ``numpy.bool_`` (e.g. from ``Series.any()``) alike.
    ``has_herd_structure``, ``run_demographic``, ``run_nondemographic`` and
    ``emission_factors_only`` also accept a one-element bool array (R's
    length-1 logical); ``validate_inputs`` accepts only a ``bool`` or
    ``numpy.bool_``. With validation on, any other value (``1``, ``"TRUE"``,
    ``None``, a list, ...) is rejected, as in R. With
    ``validate_inputs=False`` the switches are not checked and such a value
    counts as FALSE (``isTRUE(1)`` is FALSE in R).

    Parameters
    ----------
    has_herd_structure : bool, default False
        If TRUE, ``cohort_level_data`` already contains the herd structure
        (``cohort_stock_size`` and ``offtake_heads_assessment``, in heads)
        and the herd simulation is skipped.
    run_demographic, run_nondemographic : bool, default True
        Which herd modules :func:`gleampy.run_all_herd_module` runs when
        ``has_herd_structure`` is not TRUE (at least one must be TRUE). When
        both are TRUE, the non-demographic start weights (kg) are set to
        ``live_weight_at_weaning`` (kg) for herds that divert juveniles and
        to NA for the other herds, as in :func:`gleampy.run_gleam`.
        ``run_demographic=False`` needs FN / MN rows in
        ``cohort_level_data``.
    cohort_level_data : pandas.DataFrame
        Cohort-level master table (one row per herd x cohort [x non-demographic
        phase]), as for :func:`gleampy.run_gleam`. Without feed tables it must
        also hold the six primary ration quality columns:
        ``ration_gross_energy`` and ``ration_metabolizable_energy`` (MJ / kg
        DM), ``ration_nitrogen`` (kg N / kg DM),
        ``ration_digestibility_fraction`` and
        ``ration_urinary_energy_fraction`` (fractions) and ``ration_ash``
        (fraction of DM).
    herd_level_data : pandas.DataFrame
        Herd-level master table (one row per ``herd_id``) with live weights
        (kg), reproduction, milk, work, fibre, carcass and (CHK) egg
        parameters.
    feed_rations : pandas.DataFrame or None, default None
        Cohort-level feed ration shares (``feed_id``, ``feed_ration_fraction``
        as a fraction of the ration dry matter). Supply together with
        ``feed_params``, or omit both.
    feed_params : pandas.DataFrame or None, default None
        Feed nutritional parameters per ``feed_id``.
    manure_management_system_fraction : pandas.DataFrame
        Share (fraction) of manure handled by each manure management system
        per herd x cohort.
    manure_management_system_factors : pandas.DataFrame
        Emission factors per manure management system.
    simulation_duration : float, default 365
        Assessment period length (days); a single positive number.
    global_warming_potential_set : str, default "AR6"
        ``"AR6"``, ``"AR5_excluding_carbon_feedback"``,
        ``"AR5_including_carbon_feedback"`` or ``"AR4"``.
    show_indicator : bool, default True
        Print progress messages.
    emission_factors_only : bool, default False
        If TRUE, stop after the cohort-level enteric and manure emission
        factors; production, allocation and aggregation are skipped and
        ``allocation_long`` / ``aggregation_results`` are ``None``.
    validate_inputs : bool, default True
        Validate inputs (default); a ``bool`` or ``numpy.bool_``. ``False``
        skips all validation and issues a warning.

    Returns
    -------
    dict
        ``cohort_level_results``, ``herd_level_results``, ``allocation_long``
        and ``aggregation_results`` (the last two ``None`` when
        ``emission_factors_only``).

    Raises
    ------
    GleamValidationError
        If an input is invalid (with ``validate_inputs``). Also, with or
        without validation, if the herd simulation produces no herd-level
        results (``run_demographic=False`` without FN / MN rows; R stops
        with an internal data.table error) or a herd that diverts juveniles
        has no ``live_weight_at_weaning`` (R: "object not found").
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
        if not is_true_scalar(emission_factors_only):
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
