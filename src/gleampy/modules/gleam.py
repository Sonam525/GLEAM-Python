"""Full GLEAM pipeline (port of ``R/run_gleam.R``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import Progress, as_float, copy_frame, is_true_scalar, merge_dt
from ..validation._shared import abort, setup_validation
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


#: Non-demographic start weight set from the weaning weight, per diversion share.
_NONDEMO_START_WEIGHTS: tuple[tuple[str, str], ...] = (
    ("prop_nondemo_mal_juv", "live_weight_male_nondemographic_start"),
    ("prop_nondemo_fem_juv", "live_weight_female_nondemographic_start"),
)


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
    """Steps 2-3 of the pipeline shared by :func:`run_gleam` and :func:`gleampy.run_emissions_direct`.

    Runs the herd module(s) when no herd structure is supplied, adds the
    optional non-demographic herd columns, and sets the non-demographic start
    weights from the weaning weight when juveniles are diverted.

    The switches follow R's ``isTRUE()``: Python ``True``, ``numpy.True_`` and
    a one-element bool array holding ``True`` are TRUE, anything else
    (``1``, ``"TRUE"``, ...) is not (:func:`gleampy._utils.is_true_scalar`).

    Parameters
    ----------
    has_herd_structure : bool
        If TRUE, ``cohort_level_data`` already holds the herd structure and is
        used as is; otherwise :func:`gleampy.run_all_herd_module` simulates it.
    run_demographic, run_nondemographic : bool
        Herd modules to run when ``has_herd_structure`` is not TRUE. When both
        are TRUE, ``live_weight_{male,female}_nondemographic_start`` (kg) are
        set to ``live_weight_at_weaning`` (kg) for herds with a positive
        ``prop_nondemo_{mal,fem}_juv`` and to NA for the other herds.
    cohort_level_data : pandas.DataFrame
        Cohort-level master table (one row per herd x cohort [x
        non-demographic phase]).
    herd_level_data : pandas.DataFrame
        Herd-level master table (one row per ``herd_id``).
    simulation_duration : float
        Assessment period length (days).
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool, default True
        Validate the inputs of the herd modules.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(cohort_level_data, herd_level_data)`` ready for the weights step.

    Raises
    ------
    GleamValidationError
        If the herd simulation produces no herd-level results (no module ran,
        e.g. ``run_demographic = FALSE`` without FN / MN rows; R fails there
        with an internal data.table error), or if a herd diverts juveniles
        and ``live_weight_at_weaning`` is missing (R: "object not found"),
        with or without validation, like the modules' missing-column errors.
    """
    use_structure = is_true_scalar(has_herd_structure)
    both_modules = is_true_scalar(run_demographic) and is_true_scalar(run_nondemographic)
    if use_structure:
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
        if herd_results["herd_level_results"] is None:
            # Neither herd module ran. R goes on with a NULL herd table and
            # fails at its next `:=` with an internal data.table error.
            if not is_true_scalar(run_demographic) and not is_true_scalar(run_nondemographic):
                abort("At least one of `run_demographic` or `run_nondemographic` must be TRUE.")
            abort(
                "`run_demographic = FALSE` requires non-demographic (FN/MN) rows in "
                "`cohort_level_data`: no herd module ran, so there are no herd-level results."
            )
        chrt = herd_results["cohort_level_results"]
        hrd = copy_frame(herd_results["herd_level_results"])

    for col in OPTIONAL_NONDEMO_HERD_COLUMNS:
        if col not in hrd.columns:
            hrd[col] = np.nan

    if not use_structure and both_modules:
        diverted = {}
        for prop, col in _NONDEMO_START_WEIGHTS:
            p = as_float(hrd[prop])
            diverted[col] = ~np.isnan(p) & (p > 0)
        # data.table evaluates `:= live_weight_at_weaning` only when a row is
        # selected, so the weaning weight is needed only for diverted herds.
        weaning = np.nan
        if any(mask.any() for mask in diverted.values()):
            if "live_weight_at_weaning" not in hrd.columns:
                # R: "object 'live_weight_at_weaning' not found"
                abort('Missing required columns in `herd_level_data`: "live_weight_at_weaning"')
            weaning = as_float(hrd["live_weight_at_weaning"])
        for _, col in _NONDEMO_START_WEIGHTS:
            # Both data.table assignments together cover every row.
            hrd[col] = np.where(diverted[col], weaning, np.nan)
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

    The logical switches follow R's ``isTRUE()`` and treat a Python ``bool``
    and a ``numpy.bool_`` (e.g. from ``Series.any()``) alike.
    ``has_herd_structure``, ``run_demographic`` and ``run_nondemographic``
    also accept a one-element bool array (R's length-1 logical);
    ``validate_inputs`` accepts only a ``bool`` or ``numpy.bool_``. With
    validation on, any other value (``1``, ``"TRUE"``, ``None``, a list, ...)
    is rejected, as in R. With ``validate_inputs=False`` the switches are not
    checked and such a value counts as FALSE (``isTRUE(1)`` is FALSE in R).

    Parameters
    ----------
    has_herd_structure : bool, default False
        If TRUE, ``cohort_level_data`` already contains the herd structure
        (``cohort_stock_size`` in heads, ``offtake_heads_assessment`` in
        heads) and the herd simulation is skipped.
    run_demographic, run_nondemographic : bool, default True
        Which herd modules :func:`gleampy.run_all_herd_module` runs when
        ``has_herd_structure`` is not TRUE (at least one must be TRUE). When
        both are TRUE, the non-demographic start weights (kg) are set to
        ``live_weight_at_weaning`` (kg) for herds that divert juveniles
        (positive ``prop_nondemo_*_juv``) and to NA for the other herds, as
        in R. ``run_demographic=False`` needs FN / MN rows in
        ``cohort_level_data``.
    cohort_level_data : pandas.DataFrame
        Cohort-level master table (one row per herd x cohort [x non-demographic
        phase]): ``herd_id``, ``species_short``, ``cohort_short``,
        ``cohort_duration_days`` (days), ``offtake_rate`` (fraction),
        ``low_activity_fraction``, ``high_activity_fraction`` (fractions) and,
        without herd structure, ``death_rate`` (fraction); with herd structure,
        ``cohort_stock_size`` and ``offtake_heads_assessment`` (heads).
        ``nondemo_productive_phase_id`` (1 or 2) is needed only for FN / MN
        rows. ``is_egg_producing`` (logical) is required when ``herd_level_data``
        has CHK herds. Optional: ``ch4_mitigation_factor`` (fraction).
    herd_level_data : pandas.DataFrame
        Herd-level master table (one row per ``herd_id``) with live weights
        (kg), reproduction, milk (kg / day), work (hours / day), fibre
        (kg / year), carcass and (CHK) egg parameters.
    feed_rations : pandas.DataFrame
        Cohort-level feed ration shares (``feed_id``, ``feed_ration_fraction``
        as a fraction of the ration dry matter).
    feed_params : pandas.DataFrame
        Feed nutritional parameters per ``feed_id``.
    feed_emissions : pandas.DataFrame
        Feed production emission factors per ``feed_id`` (g / kg DM).
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
    validate_inputs : bool, default True
        Validate inputs (default); a ``bool`` or ``numpy.bool_``. ``False``
        skips all validation for speed (e.g. repeated runs on
        already-validated inputs) and issues a warning.

    Returns
    -------
    dict
        ``cohort_level_results`` (DataFrame), ``herd_level_results``
        (DataFrame), ``allocation_long`` (DataFrame) and
        ``aggregation_results`` (dict with ``results_emissions``,
        ``results_feed``, ``results_production``, ``results_nitrogen``).

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
