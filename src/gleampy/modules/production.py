"""Production module (port of ``R/run_production_module.R``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import Progress, copy_frame, lookup
from ..core.production import (
    calc_egg_production,
    calc_fibre_production,
    calc_meat_production,
    calc_milk_production,
)
from ..validation._shared import normalize_optional_is_egg_producing_column, setup_validation
from ..validation.production_run import (
    check_egg_columns,
    validate_production_simulation_duration,
    validate_run_production_module_inputs,
)


def _herd_value(cohort: pd.DataFrame, herd: pd.DataFrame, col: str) -> Any:
    """``herd_level_data[.SD, on = "herd_id", x.<col>]``; ``NA`` when the column is absent.

    R evaluates these look-ups lazily, so a herd-level column that is not
    needed by any row (e.g. the egg columns without egg-producing cohorts)
    may be missing from the table.
    """
    if col not in herd.columns:
        return np.full(len(cohort), np.nan)
    return lookup(cohort, herd, col)


def _cohort_value(cohort: pd.DataFrame, col: str) -> Any:
    """Optional cohort column, or ``NA`` when absent (only forced lazily in R)."""
    if col not in cohort.columns:
        return np.full(len(cohort), np.nan)
    return cohort[col]


def run_production_module(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
    simulation_duration: float = 365,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> pd.DataFrame:
    """Run the production module.

    Computes cohort-level milk, egg, fibre and meat outputs over the
    assessment period from the herd structure and herd-level production
    parameters.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table: ``herd_id``, ``cohort_short``,
        ``cohort_stock_size`` (heads), ``offtake_heads_assessment``
        (heads/assessment period), ``live_weight_cohort_at_slaughter`` (kg)
        and, optionally, ``nondemo_productive_phase_id`` and
        ``is_egg_producing`` (CHK only).
    herd_level_data : pandas.DataFrame
        Herd-level table (one row per ``herd_id``): ``species_short``,
        ``milk_yield_day``, ``lactating_females_fraction``, milk composition
        (``milk_protein_fraction``, ``milk_fat_fraction``,
        ``milk_lactose_fraction`` and their ``*_standard`` values),
        ``fibre_yield_year``, ``carcass_dressing_fraction``,
        ``bone_free_meat_fraction``, ``meat_protein_fraction`` and, for
        egg-producing CHK cohorts, ``egg_output_human_consumption`` and
        ``egg_average_weight``.
    simulation_duration : float
        Length of the assessment period (days); a single positive value (a
        vector is rejected, as in R, even with ``validate_inputs=False``).
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    pandas.DataFrame
        The cohort-level input columns (plus ``is_egg_producing = NA`` when it
        was absent and there are no CHK herds) and the milk
        (``milk_production_mass_cohort``, ``milk_production_protein_cohort``,
        ``milk_production_fpcm_cohort``), egg (``egg_production_number_cohort``,
        ``egg_production_mass_cohort``, ``egg_production_protein_cohort``),
        fibre (``fibre_production_cohort``) and meat
        (``meat_production_live_weight_cohort``,
        ``meat_production_carcass_weight_cohort``,
        ``meat_production_bone_free_meat_cohort``,
        ``meat_production_protein_cohort``) outputs per cohort over the
        assessment period. Outputs that do not apply are 0.
    """
    with setup_validation(validate_inputs):
        # as.data.table() + copy(): never modify the caller's tables.
        cohort = copy_frame(cohort_level_data)
        herd = copy_frame(herd_level_data)

        # --- Step 1: validate inputs (adds the optional is_egg_producing column)
        validate_run_production_module_inputs(cohort, herd)
        normalize_optional_is_egg_producing_column(cohort, herd)
        validate_production_simulation_duration(simulation_duration)

        progress = Progress(show_indicator)
        progress.status("Calculating production (milk, fibre, meat), please wait...")

        species = _herd_value(cohort, herd, "species_short")

        # --- Step 3: milk production outputs
        milk = calc_milk_production(
            species_short=species,
            cohort_short=cohort["cohort_short"],
            milk_yield_day=_herd_value(cohort, herd, "milk_yield_day"),
            simulation_duration=simulation_duration,
            cohort_stock_size=cohort["cohort_stock_size"],
            lactating_females_fraction=_herd_value(cohort, herd, "lactating_females_fraction"),
            milk_protein_fraction=_herd_value(cohort, herd, "milk_protein_fraction"),
            milk_fat_fraction=_herd_value(cohort, herd, "milk_fat_fraction"),
            milk_lactose_fraction=_herd_value(cohort, herd, "milk_lactose_fraction"),
            milk_protein_fraction_standard=_herd_value(cohort, herd, "milk_protein_fraction_standard"),
            milk_fat_fraction_standard=_herd_value(cohort, herd, "milk_fat_fraction_standard"),
            milk_lactose_fraction_standard=_herd_value(cohort, herd, "milk_lactose_fraction_standard"),
        )
        for col, values in milk.items():
            cohort[col] = values

        # --- egg production outputs
        check_egg_columns(cohort, herd, species)
        eggs = calc_egg_production(
            species_short=species,
            cohort_short=cohort["cohort_short"],
            nondemo_productive_phase_id=_cohort_value(cohort, "nondemo_productive_phase_id"),
            is_egg_producing=cohort["is_egg_producing"],
            egg_output_human_consumption=_herd_value(cohort, herd, "egg_output_human_consumption"),
            egg_average_weight=_herd_value(cohort, herd, "egg_average_weight"),
            simulation_duration=simulation_duration,
        )
        for col, values in eggs.items():
            cohort[col] = values

        # --- Step 4: fibre production
        cohort["fibre_production_cohort"] = calc_fibre_production(
            species_short=species,
            cohort_short=cohort["cohort_short"],
            fibre_yield_year=_herd_value(cohort, herd, "fibre_yield_year"),
            simulation_duration=simulation_duration,
            cohort_stock_size=cohort["cohort_stock_size"],
        )

        # --- Step 5: meat production outputs
        meat = calc_meat_production(
            offtake_heads_assessment=cohort["offtake_heads_assessment"],
            live_weight_cohort_at_slaughter=cohort["live_weight_cohort_at_slaughter"],
            carcass_dressing_fraction=_herd_value(cohort, herd, "carcass_dressing_fraction"),
            bone_free_meat_fraction=_herd_value(cohort, herd, "bone_free_meat_fraction"),
            meat_protein_fraction=_herd_value(cohort, herd, "meat_protein_fraction"),
        )
        for col, values in meat.items():
            cohort[col] = values

        progress.success("Production cohort calculations completed.")
        return cohort
