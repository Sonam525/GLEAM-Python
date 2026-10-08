"""Weights module (port of ``R/run_weights_module.R``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import Lookup, Progress, as_str, copy_frame, isin
from ..core.weights import calc_avg_weights, calc_cohort_weights, calc_daily_weight_gain
from ..validation._shared import abort, setup_validation, validation_enabled
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


def _require_columns(data: pd.DataFrame, arg: str, needs: dict[str, Any]) -> None:
    """Abort when a column that some row needs (``{column: rows}``) is absent.

    R reads these columns lazily (``herd_level_data[.SD, on = "herd_id",
    x.col]`` promises), so a column is an error only when a row evaluates it;
    R then stops with data.table's "column name 'x.col' is not found". The
    port raises :class:`GleamValidationError` instead, naming every absent
    column that is needed.
    """
    missing = [c for c, need in needs.items() if c not in data.columns and np.any(need)]
    if missing:
        abort(f"Missing required columns in `{arg}`: " + ", ".join(f'"{c}"' for c in missing))


def _herd_needs(co: np.ndarray, sp: np.ndarray, validate: bool) -> dict[str, Any]:
    """Rows that evaluate each herd-level input of ``calc_cohort_weights()`` in R.

    With validation on, ``validate_cohort_weight_inputs()`` puts all 14 herd
    columns in a list for every row, so all of them are needed even for
    demographic-only data. Without validation only the calc body reads them:
    each cohort branch reads its own weights, and CHK rows read the birth
    weight (their weaning weight is replaced by it).
    """
    every = np.ones(len(co), dtype=bool)
    if validate:
        return {col: every for col in _HERD_WEIGHT_ARGS}
    chk = sp == "CHK"
    juvenile = isin(co, ("FJ", "MJ"))
    subadult = isin(co, ("FS", "MS"))
    fn, mn = co == "FN", co == "MN"
    return {
        "live_weight_female_adult": isin(co, ("FJ", "FS", "FA")),
        "live_weight_male_adult": isin(co, ("MJ", "MS", "MA")),
        "live_weight_at_birth": juvenile | chk,
        "live_weight_female_at_slaughter": co == "FS",
        "live_weight_male_at_slaughter": co == "MS",
        "live_weight_at_weaning": (juvenile | subadult) & ~chk,
        "live_weight_female_nondemographic_start": fn,
        "live_weight_male_nondemographic_start": mn,
        "live_weight_female_nondemographic_end": fn,
        "live_weight_male_nondemographic_end": mn,
        "phase1_nondemo_fem_duration_days": fn,
        "phase2_nondemo_fem_duration_days": fn,
        "phase1_nondemo_mal_duration_days": mn,
        "phase2_nondemo_mal_duration_days": mn,
    }


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
        One row per herd with ``herd_id``, ``species_short`` and all 14
        herd-level weight inputs: the live weights (kg)
        ``live_weight_female_adult``, ``live_weight_male_adult``,
        ``live_weight_at_birth``, ``live_weight_at_weaning``,
        ``live_weight_female_at_slaughter``, ``live_weight_male_at_slaughter``,
        ``live_weight_{female,male}_nondemographic_{start,end}`` and the
        non-demographic phase durations
        ``phase{1,2}_nondemo_{fem,mal}_duration_days`` (days). As in R, all
        14 columns must be present when validation is on, even for
        demographic-only data without ``FN`` / ``MN`` rows (R's validator
        reads every one of them for every row; see "Effectively required
        columns" in ``docs/R_PORT_NOTES.md``); the non-demographic ones may
        be NA for herds without non-demographic cohorts. With validation off
        only the columns that a row's cohort uses are needed. A missing
        needed column raises :class:`GleamValidationError`.
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
    to :func:`gleampy.calc_cohort_weights`; average and final weights come from
    :func:`gleampy.calc_avg_weights` (offtake ignored for ``FN`` / ``MN``) and the
    daily gain from :func:`gleampy.calc_daily_weight_gain`.

    All weights are computed in float64, whatever the dtype of the inputs
    (int64, nullable ``Int64``, float64 or object). R assigns the outputs
    with ``:=`` and ``by = .I``, which gives each output column the type of
    the first row's value, so when ``fread`` reads a whole-number weight
    column as integer R truncates fractional weights of later rows (and
    turns values beyond the integer range into NA). That R bug is not
    replicated: the port always returns the values R gives for double-typed
    inputs.
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

        # --- Columns that some row reads (R evaluates the joins lazily)
        n = len(cohort_level_data)
        every = np.ones(n, dtype=bool)
        _require_columns(cohort_level_data, "cohort_level_data", {"cohort_short": every})
        _require_columns(herd_level_data, "herd_level_data", {"species_short": every})
        join = Lookup(cohort_level_data, herd_level_data, "herd_id")
        species = join("species_short")
        co = as_str(cohort_level_data["cohort_short"])
        _require_columns(
            herd_level_data, "herd_level_data", _herd_needs(co, as_str(species), validation_enabled())
        )
        _require_columns(
            cohort_level_data,
            "cohort_level_data",
            {"offtake_rate": ~isin(co, ("FN", "MN")), "cohort_duration_days": every},
        )

        # --- Step 3: cohort weights (herd-level values joined on herd_id)
        herd = {
            col: join(col) if col in herd_level_data.columns else np.full(n, np.nan)
            for col in _HERD_WEIGHT_ARGS
        }
        weights = calc_cohort_weights(
            species_short=species,
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
