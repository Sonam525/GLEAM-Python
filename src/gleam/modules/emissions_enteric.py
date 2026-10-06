"""Enteric methane emissions module (port of ``R/run_emissions_enteric_module.R``)."""

from __future__ import annotations

import pandas as pd

from .._utils import Progress, copy_frame
from ..core.emissions_enteric import calc_ch4_enteric, calc_conversion_factor_ym
from ..validation._shared import setup_validation
from ..validation.emissions_enteric_run import validate_run_emissions_enteric_module_inputs


def run_emissions_enteric_module(
    cohort_level_data: pd.DataFrame,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> pd.DataFrame:
    """Run the enteric methane (CH4) emissions module.

    Computes daily enteric CH4 emissions by cohort (kg CH4/head/day) with the
    IPCC Tier 2 approach, using species-, cohort- and diet-specific methane
    conversion factors (ym).

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table with ``herd_id``, ``species_short``,
        ``cohort_short``, ``ration_digestibility_fraction`` (fraction),
        ``ration_gross_energy`` (MJ/kg DM), ``ration_intake``
        (kg DM/head/day) and, optionally, ``ch4_mitigation_factor``
        (dimensionless, default 1) and ``nondemo_productive_phase_id``
        (phase rows of ``FN`` / ``MN`` are kept as they are).
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    pandas.DataFrame
        The input columns plus ``ch4_mitigation_factor`` (added as 1 when
        absent), ``ch4_conversion_factor_ym`` (% of gross energy) and
        ``ch4_enteric`` (kg CH4/head/day).

    Notes
    -----
    Steps: default ``ch4_mitigation_factor`` to 1, compute ym with
    :func:`gleam.calc_conversion_factor_ym`, then emissions with
    :func:`gleam.calc_ch4_enteric`.
    """
    with setup_validation(validate_inputs):
        # --- Step 1: validate inputs
        validate_run_emissions_enteric_module_inputs(cohort_level_data)

        progress = Progress(show_indicator)
        progress.status("Calculating enteric methane emissions, please wait...")

        # --- Step 2: working copy
        enteric_results = copy_frame(cohort_level_data)

        # Use the mitigation factor from the data if present; otherwise 1.
        if "ch4_mitigation_factor" not in enteric_results.columns:
            enteric_results["ch4_mitigation_factor"] = 1.0

        # --- Step 3: methane conversion factor (ym)
        enteric_results["ch4_conversion_factor_ym"] = calc_conversion_factor_ym(
            species_short=enteric_results["species_short"],
            cohort_short=enteric_results["cohort_short"],
            ration_digestibility_fraction=enteric_results["ration_digestibility_fraction"],
        )

        # --- Step 4: enteric methane emissions (kg CH4/head/day)
        enteric_results["ch4_enteric"] = calc_ch4_enteric(
            species_short=enteric_results["species_short"],
            ch4_conversion_factor_ym=enteric_results["ch4_conversion_factor_ym"],
            ch4_mitigation_factor=enteric_results["ch4_mitigation_factor"],
            ration_gross_energy=enteric_results["ration_gross_energy"],
            ration_intake=enteric_results["ration_intake"],
        )

        progress.success("Enteric methane emissions calculation complete.")
        return enteric_results
