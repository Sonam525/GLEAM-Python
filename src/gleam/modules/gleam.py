"""Full GLEAM pipeline (port of ``R/run_gleam.R``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import Progress, as_float, copy_frame, merge_dt
from ..validation._shared import setup_validation
from ..validation.gleam_run import validate_run_gleam_inputs
from .aggregation import run_aggregation_module
from .all_herd import run_all_herd_module
from .allocation import run_allocation_module
from .emissions_enteric import run_emissions_enteric_module
from .emissions_manure import run_emissions_manure_module
from .emissions_ration import run_emissions_ration_module
from .metabolic_energy_req import run_metabolic_energy_req_module
from .nitrogen_balance import run_nitrogen_balance_module
from .production import run_production_module
from .ration_quality import run_ration_quality_module
from .weights import run_weights_module

#: Herd-level columns only needed for non-demographic cohorts (FN / MN). They
#: are added as NA when absent because downstream joins expect them.
OPTIONAL_NONDEMO_HERD_COLUMNS: tuple[str, ...] = (
    "prop_nondemo_fem_juv",
    "prop_nondemo_mal_juv",
    "rest_between_nondemo_cycles_duration",
    "phase1_nondemo_fem_duration_days",
    "phase2_nondemo_fem_duration_days",
    "phase1_nondemo_mal_duration_days",
    "phase2_nondemo_mal_duration_days",
    "live_weight_female_nondemographic_start",
    "live_weight_male_nondemographic_start",
    "live_weight_female_nondemographic_end",
    "live_weight_male_nondemographic_end",
)

#: Join keys between cohort-level data and per-cohort ration summaries.
COHORT_KEYS: list[str] = ["herd_id", "species_short", "cohort_short", "nondemo_productive_phase_id"]


def build_herd_structure(
    has_herd_structure: bool,
    run_demographic: bool,
    run_nondemographic: bool,
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
    simulation_duration: float,
    show_indicator: bool,
    validate_inputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Steps 2-3 of the pipeline shared by :func:`run_gleam` and :func:`gleam.run_emissions_direct`.

    Runs the herd module(s) when no herd structure is supplied, adds the
    optional non-demographic herd columns, and sets the non-demographic start
    weights from the weaning weight when juveniles are diverted.
    """
    if has_herd_structure:
        chrt = copy_frame(cohort_level_data)
        hrd = copy_frame(herd_level_data)
    else:
        herd_results = run_all_herd_module(
            cohort_level_data=cohort_level_data,
            herd_level_data=herd_level_data,
            simulation_duration=simulation_duration,
            show_indicator=show_indicator,
            validate_inputs=validate_inputs,
            run_demographic=run_demographic,
            run_nondemographic=run_nondemographic,
        )
        chrt = herd_results["cohort_level_results"]
        hrd = copy_frame(herd_results["herd_level_results"])

    for col in OPTIONAL_NONDEMO_HERD_COLUMNS:
        if col not in hrd.columns:
            hrd[col] = np.nan

    if not has_herd_structure and run_demographic is True and run_nondemographic is True:
        weaning = as_float(hrd["live_weight_at_weaning"])
        for prop, col in (
            ("prop_nondemo_mal_juv", "live_weight_male_nondemographic_start"),
            ("prop_nondemo_fem_juv", "live_weight_female_nondemographic_start"),
        ):
            p = as_float(hrd[prop])
            diverted = ~np.isnan(p) & (p > 0)
            # Both data.table assignments together cover every row.
            hrd[col] = np.where(diverted, weaning, np.nan)
    return chrt, hrd


def run_gleam(
    has_herd_structure: bool = False,
    run_demographic: bool = True,
    run_nondemographic: bool = True,
    cohort_level_data: pd.DataFrame | None = None,
    herd_level_data: pd.DataFrame | None = None,
    feed_rations: pd.DataFrame | None = None,
    feed_params: pd.DataFrame | None = None,
    feed_emissions: pd.DataFrame | None = None,
    manure_management_system_fraction: pd.DataFrame | None = None,
    manure_management_system_factors: pd.DataFrame | None = None,
    simulation_duration: float = 365,
    global_warming_potential_set: str = "AR6",
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> dict[str, Any]:
    """Run the full Global Livestock Environmental Assessment Model (GLEAM) pipeline.

    Runs, in order: herd simulation (optional, demographic and/or
    non-demographic), weights, ration quality, metabolic energy requirements
    and dry-matter intake, enteric CH4, nitrogen balance, manure CH4/N2O, feed
    production emissions, production (milk, meat, fibre, eggs), allocation and
    aggregation to herd totals and CO2-eq.

    Parameters
    ----------
    has_herd_structure
        If true, ``cohort_level_data`` already contains the herd structure
        (``cohort_stock_size``, ``offtake_heads_assessment``) and the herd
        simulation is skipped.
    run_demographic, run_nondemographic
        Which herd modules :func:`gleam.run_all_herd_module` runs when
        ``has_herd_structure`` is false.
    cohort_level_data
        Cohort-level master table (one row per herd x cohort [x non-demographic
        phase]): ``herd_id``, ``species_short``, ``cohort_short``,
        ``cohort_duration_days``, ``offtake_rate``, ``low_activity_fraction``,
        ``high_activity_fraction`` and, without herd structure, ``death_rate``
        and ``nondemo_productive_phase_id``; with herd structure,
        ``cohort_stock_size`` and ``offtake_heads_assessment``. Optional:
        ``is_egg_producing`` (CHK), ``ch4_mitigation_factor``.
    herd_level_data
        Herd-level master table (one row per ``herd_id``) with live weights,
        reproduction, milk, work, fibre, carcass and (CHK) egg parameters.
    feed_rations
        Cohort-level feed ration shares (``feed_id``, ``feed_ration_fraction``).
    feed_params
        Feed nutritional parameters per ``feed_id``.
    feed_emissions
        Feed production emission factors per ``feed_id`` (g / kg DM).
    manure_management_system_fraction
        Share of manure handled by each manure management system per herd x cohort.
    manure_management_system_factors
        Emission factors per manure management system.
    simulation_duration
        Assessment period length (days).
    global_warming_potential_set
        ``"AR6"``, ``"AR5_excluding_carbon_feedback"``,
        ``"AR5_including_carbon_feedback"`` or ``"AR4"``.
    show_indicator
        Print progress messages.
    validate_inputs
        Validate inputs (default). ``False`` skips all validation for speed
        (e.g. repeated runs on already-validated inputs) and issues a warning.

    Returns
    -------
    dict
        ``cohort_level_results`` (DataFrame), ``herd_level_results``
        (DataFrame), ``allocation_long`` (DataFrame) and
        ``aggregation_results`` (dict with ``results_emissions``,
        ``results_feed``, ``results_production``, ``results_nitrogen``).
    """
    with setup_validation(validate_inputs):
        validate_run_gleam_inputs(
            has_herd_structure=has_herd_structure,
            cohort_level_data=cohort_level_data,
            herd_level_data=herd_level_data,
            feed_rations=feed_rations,
            feed_params=feed_params,
            feed_emissions=feed_emissions,
            manure_management_system_fraction=manure_management_system_fraction,
            manure_management_system_factors=manure_management_system_factors,
            simulation_duration=simulation_duration,
            global_warming_potential_set=global_warming_potential_set,
        )
        progress = Progress(show_indicator)
        progress.header("Running GLEAM pipeline...")

        # --- Steps 2-3: herd structure and optional non-demographic columns
        chrt, hrd = build_herd_structure(
            has_herd_structure, run_demographic, run_nondemographic,
            cohort_level_data, herd_level_data, simulation_duration, show_indicator,
            validate_inputs,
        )

        # --- Step 3: weights
        chrt = run_weights_module(
            cohort_level_data=chrt, herd_level_data=hrd, show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )["cohort_level_results"]

        # --- Step 4: ration quality
        rations_summary = run_ration_quality_module(
            rations_share=feed_rations, feed_params=feed_params, show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )
        chrt = merge_dt(chrt, rations_summary, by=COHORT_KEYS)

        # --- Step 5: energy requirements and dry-matter intake
        chrt = run_metabolic_energy_req_module(
            cohort_level_data=chrt, herd_level_data=hrd, show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )

        # --- Step 6: enteric methane
        chrt = run_emissions_enteric_module(cohort_level_data=chrt, show_indicator=show_indicator, validate_inputs=validate_inputs)

        # --- Step 7: nitrogen balance
        chrt = run_nitrogen_balance_module(
            cohort_level_data=chrt, herd_level_data=hrd, show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )

        # --- Step 8: manure management emissions
        chrt = run_emissions_manure_module(
            cohort_level_data=chrt,
            manure_management_system_fraction=manure_management_system_fraction,
            manure_management_system_factors=manure_management_system_factors,
            show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )

        # --- Step 9: feed production emissions (diet-level emission factors)
        feed_emissions_summary = run_emissions_ration_module(
            rations_share=feed_rations, feed_emissions=feed_emissions, show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )
        chrt = merge_dt(chrt, feed_emissions_summary, by=COHORT_KEYS)

        # --- Step 10: production
        chrt = run_production_module(
            cohort_level_data=chrt,
            herd_level_data=hrd,
            simulation_duration=simulation_duration,
            show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )

        # --- Step 11: allocation
        allocation_results = run_allocation_module(
            cohort_level_data=chrt,
            herd_level_data=hrd,
            simulation_duration=simulation_duration,
            show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )
        chrt = allocation_results["cohort_allocation_inputs"]

        # --- Step 12: aggregation
        aggregation_results = run_aggregation_module(
            cohort_level_data=chrt,
            allocation_herd_long=allocation_results["allocation_long"],
            simulation_duration=simulation_duration,
            global_warming_potential_set=global_warming_potential_set,
            show_indicator=show_indicator,
            validate_inputs=validate_inputs,
        )
        progress.success("GLEAM pipeline complete.")

        return {
            "cohort_level_results": chrt,
            "herd_level_results": hrd,
            "allocation_long": allocation_results["allocation_long"],
            "aggregation_results": {
                "results_emissions": aggregation_results["results_emissions"],
                "results_feed": aggregation_results["results_feed"],
                "results_production": aggregation_results["results_production"],
                "results_nitrogen": aggregation_results["results_nitrogen"],
            },
        }
