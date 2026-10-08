"""Input validation for :func:`gleampy.run_metabolic_energy_req_module`.

Port of ``R/validate_run_metabolic_energy_req_inputs.R``: table structure,
required columns, valid codes, herd linkage and row-wise numeric consistency
(activity fractions, cohort live weights, birth vs weaning weight).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import as_float, as_str, in_true, is_na
from ._shared import (
    _vals,
    abort,
    check_cohort_completeness,
    check_data_table,
    check_herd_id_consistency,
    check_herd_id_unique,
    check_required_columns,
    normalize_optional_is_egg_producing_column,
    validate_cohort_short_values,
    validate_species_short_values,
    validator,
)

#: Cohort-level columns always required by the module.
REQUIRED_COHORT_COLS: tuple[str, ...] = (
    "herd_id", "cohort_short",
    "live_weight_cohort_average", "offtake_rate",
    "low_activity_fraction", "high_activity_fraction",
    "live_weight_cohort_initial", "live_weight_cohort_final", "live_weight_mature_stage",
    "daily_weight_gain", "cohort_duration_days",
    "ration_digestibility_fraction", "ration_gross_energy", "ration_metabolizable_energy",
)

#: Herd-level columns required when demographic cohorts are present.
DEMOGRAPHIC_HERD_COLS: tuple[str, ...] = (
    "age_first_parturition", "lactating_females_fraction", "milk_yield_day", "milk_fat_fraction",
    "non_productive_duration", "pregnancy_duration", "litter_size", "death_rate_juvenile",
    "live_weight_at_birth", "live_weight_at_weaning",
    "lactation_duration", "parturition_rate",
    "draught_work_hours_female", "draught_work_hours_male",
    "draught_fraction_female", "draught_fraction_male",
    "fibre_yield_year",
)


def _column(df: pd.DataFrame, col: str) -> np.ndarray:
    """``df$col`` (``NULL`` -> empty array when the column is absent)."""
    if col in df.columns:
        return df[col].to_numpy()
    return np.array([], dtype=object)


def _fmt_id(v: Any) -> str:
    """``paste0()`` rendering of an id value (``1`` not ``1.0``, ``NA`` for missing)."""
    if is_na(v):
        return "NA"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _row_labels(df: pd.DataFrame, mask: np.ndarray) -> list[str]:
    return [
        f"{_fmt_id(h)} / {_fmt_id(c)}"
        for h, c in zip(df["herd_id"].to_numpy()[mask], df["cohort_short"].to_numpy()[mask])
    ]


@validator
def validate_run_metabolic_energy_req_module_inputs(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
) -> None:
    """Validate the inputs of :func:`gleampy.run_metabolic_energy_req_module`.

    As in R, adds ``is_egg_producing = NA`` to ``cohort_level_data`` (in place)
    when the column is absent and no CHK herd is present; the runner passes a
    working copy, never the caller's table. The egg herd columns are required
    when a CHK herd exists and any flag is ``%in% TRUE`` (R also counts ``1``
    and ``"TRUE"`` there).

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level input table of :func:`gleampy.run_metabolic_energy_req_module`.
    herd_level_data : pandas.DataFrame
        Herd-level input table, one row per ``herd_id``.

    Raises
    ------
    GleamValidationError
        On the first check that fails, with R's message.
    """
    # --- Basic type and structure checks ---
    check_data_table(cohort_level_data, "cohort_level_data")
    check_data_table(herd_level_data, "herd_level_data")
    normalize_optional_is_egg_producing_column(cohort_level_data, herd_level_data)

    # --- Required columns ---
    required_herd_cols = ["herd_id", "species_short"]
    cohort_codes_present = set(v for v in _column(cohort_level_data, "cohort_short") if not is_na(v))
    if cohort_codes_present & {"FA", "MA", "FS", "MS", "FJ", "MJ"}:
        required_herd_cols += list(DEMOGRAPHIC_HERD_COLS)

    species = as_str(_column(herd_level_data, "species_short"))
    if np.any(species == "CHK"):
        required_herd_cols.append("average_annual_temperature")
        if "is_egg_producing" in cohort_level_data.columns and np.any(
            in_true(cohort_level_data["is_egg_producing"])  # R: any(x %in% TRUE)
        ):
            required_herd_cols += ["egg_average_weight", "egg_output_human_consumption"]

    check_required_columns(cohort_level_data, REQUIRED_COHORT_COLS, "cohort_level_data")
    check_required_columns(herd_level_data, list(dict.fromkeys(required_herd_cols)), "herd_level_data")

    # --- Cohort: valid cohort_short, the 6 demographic cohorts per herd_id ---
    validate_cohort_short_values(cohort_level_data["cohort_short"], data_arg="cohort_level_data")
    check_cohort_completeness(cohort_level_data, "cohort_level_data")

    # --- Herd: one row per herd_id, valid species_short ---
    check_herd_id_unique(herd_level_data, "herd_level_data")
    validate_species_short_values(herd_level_data["species_short"], data_arg="herd_level_data")

    # --- Cross-table: same herd_id set ---
    check_herd_id_consistency(cohort_level_data, herd_level_data, "cohort_level_data", "herd_level_data")

    # --- Module-specific numeric consistency (cohort level) ---
    activity_sum = as_float(cohort_level_data["low_activity_fraction"]) + as_float(
        cohort_level_data["high_activity_fraction"]
    )
    with np.errstate(invalid="ignore"):
        bad_high = activity_sum > 1  # NA rows are not selected (data.table semantics)
        bad_low = activity_sum < 0
    if bad_high.any():
        abort(
            "For each row, `low_activity_fraction` + `high_activity_fraction` must be <= 1. "
            f"Violation(s): {_vals(_row_labels(cohort_level_data, bad_high))}"
        )
    if bad_low.any():
        abort(
            "For each row, `low_activity_fraction` + `high_activity_fraction` must be >= 0. "
            f"Violation(s): {_vals(_row_labels(cohort_level_data, bad_low))}"
        )

    # live_weight_cohort_initial <= live_weight_cohort_average <= live_weight_cohort_final (skip NA)
    lw_ini = as_float(cohort_level_data["live_weight_cohort_initial"])
    lw_avg = as_float(cohort_level_data["live_weight_cohort_average"])
    lw_fin = as_float(cohort_level_data["live_weight_cohort_final"])
    complete = ~np.isnan(lw_ini) & ~np.isnan(lw_avg) & ~np.isnan(lw_fin)
    with np.errstate(invalid="ignore"):
        inconsistent = complete & ((lw_ini > lw_avg) | (lw_avg > lw_fin))
    if inconsistent.any():
        abort(
            "For each row, `live_weight_cohort_initial` <= `live_weight_cohort_average` <= "
            "`live_weight_cohort_final` must hold. "
            f"Violation(s): {_vals(_row_labels(cohort_level_data, inconsistent))}"
        )

    # --- Numeric consistency (herd level): birth < weaning weight (except CHK) ---
    if {"live_weight_at_birth", "live_weight_at_weaning"} <= set(herd_level_data.columns):
        birth = as_float(herd_level_data["live_weight_at_birth"])
        weaning = as_float(herd_level_data["live_weight_at_weaning"])
        sp = as_str(herd_level_data["species_short"])
        not_chk = np.array([s is not None and s != "CHK" for s in sp], dtype=bool)
        with np.errstate(invalid="ignore"):
            bad = ~np.isnan(birth) & ~np.isnan(weaning) & (birth >= weaning) & not_chk
        if bad.any():
            ids = [
                int(h) if isinstance(h, (float, np.floating)) and float(h).is_integer() else h
                for h in herd_level_data["herd_id"].to_numpy()[bad]
            ]
            abort(
                "For each herd, `live_weight_at_birth` must be strictly less than "
                f"`live_weight_at_weaning`. Violation(s) for herd_id: {_vals(ids)}"
            )
