"""Run-level input validation for the nitrogen balance module.

Port of ``R/validate_run_nitrogen_balance_inputs.R``.
"""

from __future__ import annotations

import pandas as pd

from .._utils import in_true
from ._shared import (
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

#: Columns always required in the cohort-level table.
REQUIRED_COHORT_COLUMNS: tuple[str, ...] = (
    "herd_id", "cohort_short",
    "ration_intake", "ration_nitrogen", "daily_weight_gain", "cohort_duration_days",
    "cohort_stock_size",
)

#: Herd-level columns required when demographic cohorts are present.
DEMOGRAPHIC_HERD_COLUMNS: tuple[str, ...] = (
    "milk_protein_fraction", "milk_yield_day", "fibre_yield_year",
    "litter_size", "parturition_rate",
    "live_weight_at_weaning", "live_weight_at_birth", "pregnancy_duration",
)

#: Herd-level columns required when egg-producing CHK cohorts are present.
EGG_HERD_COLUMNS: tuple[str, ...] = ("egg_output_human_consumption", "egg_average_weight")


@validator
def validate_run_nitrogen_balance_module_inputs(
    cohort_level_data: pd.DataFrame, herd_level_data: pd.DataFrame
) -> None:
    """Validate the inputs of :func:`gleampy.run_nitrogen_balance_module`.

    Checks table structure, required columns (herd-level milk / reproduction
    columns only when demographic cohorts are present, egg columns only for
    egg-producing CHK cohorts), valid codes, the 6 demographic cohorts per
    herd, unique herd ids and the same herd-id set in both tables. As in R
    (``any(is_egg_producing %in% TRUE)``), a flag of ``1`` or ``"TRUE"`` also
    makes the egg columns required.

    Like R, this adds ``is_egg_producing = NA`` to ``cohort_level_data`` (in
    place) when the column is absent and there are no CHK herds; the module
    runner calls it on its own working copy.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level input table of :func:`gleampy.run_nitrogen_balance_module`.
    herd_level_data : pandas.DataFrame
        Herd-level input table, one row per ``herd_id``.

    Raises
    ------
    GleamValidationError
        On the first check that fails, with R's message.
    """
    check_data_table(cohort_level_data, "cohort_level_data")
    check_data_table(herd_level_data, "herd_level_data")
    normalize_optional_is_egg_producing_column(cohort_level_data, herd_level_data)

    required_herd_cols = ["herd_id", "species_short"]
    cohort_codes_present = pd.unique(cohort_level_data["cohort_short"]) if "cohort_short" in cohort_level_data else []
    if any(c in ("FA", "MA", "FS", "MS", "FJ", "MJ") for c in cohort_codes_present):
        required_herd_cols += list(DEMOGRAPHIC_HERD_COLUMNS)

    if (
        "species_short" in herd_level_data.columns
        and (herd_level_data["species_short"] == "CHK").any()
        and "is_egg_producing" in cohort_level_data.columns
        and in_true(cohort_level_data["is_egg_producing"]).any()  # R: any(x %in% TRUE)
    ):
        required_herd_cols += list(EGG_HERD_COLUMNS)

    check_required_columns(cohort_level_data, REQUIRED_COHORT_COLUMNS, "cohort_level_data")
    check_required_columns(herd_level_data, list(dict.fromkeys(required_herd_cols)), "herd_level_data")

    validate_cohort_short_values(cohort_level_data["cohort_short"], data_arg="cohort_level_data")
    check_cohort_completeness(cohort_level_data, "cohort_level_data")

    check_herd_id_unique(herd_level_data, "herd_level_data")
    validate_species_short_values(herd_level_data["species_short"], data_arg="herd_level_data")

    check_herd_id_consistency(cohort_level_data, herd_level_data, "cohort_level_data", "herd_level_data")
