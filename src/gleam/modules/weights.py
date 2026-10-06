"""Weights module (port of ``R/run_weights_module.R``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._utils import Progress, copy_frame, lookup
from ..core.weights import calc_avg_weights, calc_cohort_weights, calc_daily_weight_gain
from ..validation._shared import setup_validation
from ..validation.weights_run import validate_run_weights_module_inputs

# Herd-level inputs of calc_cohort_weights() matched to each cohort row by herd_id.
_HERD_WEIGHT_ARGS = (
    "live_weight_female_adult",
    "live_weight_male_adult",
    "live_weight_at_birth",
    "live_weight_female_at_slaughter",
    "live_weight_male_at_slaughter",
    "live_weight_at_weaning",
    "live_weight_female_nondemographic_start",
    "live_weight_male_nondemographic_start",
    "live_weight_female_nondemographic_end",
    "live_weight_male_nondemographic_end",
    "phase1_nondemo_fem_duration_days",
    "phase2_nondemo_fem_duration_days",
    "phase1_nondemo_mal_duration_days",
    "phase2_nondemo_mal_duration_days",
)


def run_weights_module(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the weights module: cohort live weights, average/final weights and daily gain.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        One row per herd x cohort with ``herd_id``, ``cohort_short``,
        ``cohort_duration_days`` (days), ``offtake_rate`` (fraction) and,
        optionally, ``nondemo_productive_phase_id`` (1 or 2 for ``FN`` /
        ``MN`` rows; NA for demographic cohorts).
    herd_level_data : pandas.DataFrame
        One row per herd with ``herd_id``, ``species_short`` and the herd-level
        live weights (kg): ``live_weight_female_adult``,
        ``live_weight_male_adult``, ``live_weight_at_birth``,
        ``live_weight_at_weaning``, ``live_weight_female_at_slaughter``,
        ``live_weight_male_at_slaughter`` and, for non-demographic cohorts,
        ``live_weight_{female,male}_nondemographic_{start,end}`` and
        ``phase{1,2}_nondemo_{fem,mal}_duration_days`` (days).
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    dict of pandas.DataFrame
        ``cohort_level_results``: the cohort table (with
        ``nondemo_productive_phase_id`` added as NA when absent) plus
        ``live_weight_mature_stage``, ``live_weight_cohort_initial``,
        ``live_weight_cohort_potential_final``,
        ``live_weight_cohort_at_slaughter``, ``live_weight_cohort_average``,
        ``live_weight_cohort_final`` (kg) and ``daily_weight_gain``
        (kg/head/day); ``herd_level_results``: a copy of ``herd_level_data``.

    Notes
    -----
    Steps: herd-level weights are matched to cohorts by ``herd_id`` and passed
    to :func:`gleam.calc_cohort_weights`; average and final weights come from
    :func:`gleam.calc_avg_weights` (offtake ignored for ``FN`` / ``MN``) and the
    daily gain from :func:`gleam.calc_daily_weight_gain`.
    """
    with setup_validation(validate_inputs):
        # --- Step 1: validate inputs
        validate_run_weights_module_inputs(cohort_level_data, herd_level_data)

        progress = Progress(show_indicator)
        progress.status("Calculating cohort weights, please wait...")

        # --- Step 2: working copies
        cohort_level_data = copy_frame(cohort_level_data)
        herd_level_data = copy_frame(herd_level_data)

        if "nondemo_productive_phase_id" not in cohort_level_data.columns:
            cohort_level_data["nondemo_productive_phase_id"] = np.nan

        # --- Step 3: cohort weights (herd-level values joined on herd_id)
        herd = {col: lookup(cohort_level_data, herd_level_data, col) for col in _HERD_WEIGHT_ARGS}
        weights = calc_cohort_weights(
            species_short=lookup(cohort_level_data, herd_level_data, "species_short"),
            cohort_short=cohort_level_data["cohort_short"],
            nondemo_productive_phase_id=cohort_level_data["nondemo_productive_phase_id"],
            **herd,
        )
        for col, values in weights.items():
            cohort_level_data[col] = values

        # --- Step 4: average and final weights
        avg = calc_avg_weights(
            cohort_short=cohort_level_data["cohort_short"],
            live_weight_cohort_initial=cohort_level_data["live_weight_cohort_initial"],
            live_weight_cohort_potential_final=cohort_level_data["live_weight_cohort_potential_final"],
            live_weight_cohort_at_slaughter=cohort_level_data["live_weight_cohort_at_slaughter"],
            offtake_rate=cohort_level_data["offtake_rate"],
        )
        for col, values in avg.items():
            cohort_level_data[col] = values

        # --- Step 5: daily weight gain
        cohort_level_data["daily_weight_gain"] = calc_daily_weight_gain(
            live_weight_cohort_potential_final=cohort_level_data["live_weight_cohort_potential_final"],
            live_weight_cohort_initial=cohort_level_data["live_weight_cohort_initial"],
            cohort_duration_days=cohort_level_data["cohort_duration_days"],
        )

        progress.success("Cohort weights calculation complete.")

        return {
            "cohort_level_results": cohort_level_data,
            "herd_level_results": herd_level_data,
        }
