"""Nitrogen balance module (port of ``R/run_nitrogen_balance_module.R``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import Progress, copy_frame, lookup
from ..core.nitrogen_balance import (
    calc_nitrogen_excretion,
    calc_nitrogen_intake,
    calc_nitrogen_retention,
)
from ..validation._shared import normalize_optional_is_egg_producing_column, setup_validation
from ..validation.nitrogen_balance_run import validate_run_nitrogen_balance_module_inputs


def _herd_value(cohort: pd.DataFrame, herd: pd.DataFrame, col: str) -> Any:
    """``herd_level_data[.SD, on = "herd_id", x.<col>]``; ``NA`` when the column is absent.

    R evaluates these look-ups lazily, so a herd-level column that is not
    needed by any row may be missing from the table.
    """
    if col not in herd.columns:
        return np.full(len(cohort), np.nan)
    return lookup(cohort, herd, col)


def _cohort_value(cohort: pd.DataFrame, col: str) -> Any:
    """Optional cohort column, or ``NA`` when absent (only forced lazily in R)."""
    if col not in cohort.columns:
        return np.full(len(cohort), np.nan)
    return cohort[col]


def run_nitrogen_balance_module(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> pd.DataFrame:
    """Run the nitrogen balance module.

    Computes cohort-level daily nitrogen intake, retention and excretion
    (kg N/head/day) following the IPCC Tier 2 structure.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table: ``herd_id``, ``cohort_short``, ``ration_intake``
        (kg DM/head/day), ``ration_nitrogen`` (kg N/kg DM),
        ``daily_weight_gain`` (kg/head/day), ``cohort_duration_days`` (days),
        ``cohort_stock_size`` (heads) and, optionally,
        ``nondemo_productive_phase_id`` and ``is_egg_producing`` (CHK only).
    herd_level_data : pandas.DataFrame
        Herd-level table (one row per ``herd_id``): ``species_short``,
        ``milk_protein_fraction``, ``milk_yield_day``, ``fibre_yield_year``,
        ``litter_size``, ``parturition_rate``, ``live_weight_at_weaning``,
        ``live_weight_at_birth``, ``pregnancy_duration`` and, for CHK,
        ``egg_output_human_consumption`` and ``egg_average_weight``.
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    pandas.DataFrame
        The cohort-level input columns (plus ``is_egg_producing = NA`` when it
        was absent and there are no CHK herds) and ``nitrogen_intake``,
        ``nitrogen_retention`` and ``nitrogen_excretion`` (kg N/head/day).

    Notes
    -----
    Species are taken from ``herd_level_data`` (joined by ``herd_id``).
    Steps: :func:`gleam.calc_nitrogen_intake`,
    :func:`gleam.calc_nitrogen_retention`, :func:`gleam.calc_nitrogen_excretion`.
    """
    with setup_validation(validate_inputs):
        # as.data.table() + copy(): never modify the caller's tables.
        cohort = copy_frame(cohort_level_data)
        herd = copy_frame(herd_level_data)

        # --- Step 1: validate inputs (adds the optional is_egg_producing column)
        validate_run_nitrogen_balance_module_inputs(cohort, herd)
        normalize_optional_is_egg_producing_column(cohort, herd)

        progress = Progress(show_indicator)
        progress.status("Calculating nitrogen balance, please wait...")

        species = _herd_value(cohort, herd, "species_short")

        # --- Step 3: intake - N consumed per head/day
        cohort["nitrogen_intake"] = calc_nitrogen_intake(
            ration_intake=cohort["ration_intake"],
            ration_nitrogen=cohort["ration_nitrogen"],
        )

        # --- Step 4: retention - N in growth, milk, reproduction, fibre, eggs
        cohort["nitrogen_retention"] = calc_nitrogen_retention(
            species_short=species,
            cohort_short=cohort["cohort_short"],
            nondemo_productive_phase_id=_cohort_value(cohort, "nondemo_productive_phase_id"),
            milk_protein_fraction=_herd_value(cohort, herd, "milk_protein_fraction"),
            milk_yield_day=_herd_value(cohort, herd, "milk_yield_day"),
            daily_weight_gain=cohort["daily_weight_gain"],
            fibre_yield_year=_herd_value(cohort, herd, "fibre_yield_year"),
            litter_size=_herd_value(cohort, herd, "litter_size"),
            parturition_rate=_herd_value(cohort, herd, "parturition_rate"),
            live_weight_at_weaning=_herd_value(cohort, herd, "live_weight_at_weaning"),
            live_weight_at_birth=_herd_value(cohort, herd, "live_weight_at_birth"),
            pregnancy_duration=_herd_value(cohort, herd, "pregnancy_duration"),
            cohort_duration_days=cohort["cohort_duration_days"],
            cohort_stock_size=cohort["cohort_stock_size"],
            egg_output_human_consumption=_herd_value(cohort, herd, "egg_output_human_consumption"),
            egg_average_weight=_herd_value(cohort, herd, "egg_average_weight"),
            is_egg_producing=cohort["is_egg_producing"],
        )

        # --- Step 5: excretion - N lost (intake - retention)
        cohort["nitrogen_excretion"] = calc_nitrogen_excretion(
            species_short=species,
            nitrogen_intake=cohort["nitrogen_intake"],
            nitrogen_retention=cohort["nitrogen_retention"],
        )

        progress.success("Nitrogen balance calculation complete.")
        return cohort
