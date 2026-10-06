"""Validation of ``run_demographic_herd_module()`` inputs (port of
``R/validate_run_demographic_herd_inputs.R``)."""

from __future__ import annotations

import pandas as pd

from ._shared import (
    check_cohort_completeness,
    check_data_table,
    check_herd_id_consistency,
    check_herd_id_unique,
    check_required_columns,
    validate_cohort_short_values,
    validator,
)

REQUIRED_COHORT_COLS = ("herd_id", "cohort_short", "cohort_duration_days", "offtake_rate", "death_rate")
REQUIRED_HERD_COLS = (
    "herd_id", "parturition_rate", "litter_size", "birth_fraction_female", "herd_size_total",
    "prop_nondemo_fem_juv", "prop_nondemo_mal_juv",
)


@validator
def validate_run_demographic_herd_module_inputs(
    cohort_level_data: pd.DataFrame, herd_level_data: pd.DataFrame
) -> None:
    """Structure, required columns and herd/cohort relationships of the inputs."""
    check_data_table(cohort_level_data, "cohort_level_data")
    check_data_table(herd_level_data, "herd_level_data")

    check_required_columns(cohort_level_data, REQUIRED_COHORT_COLS, "cohort_level_data")
    check_required_columns(herd_level_data, REQUIRED_HERD_COLS, "herd_level_data")

    validate_cohort_short_values(cohort_level_data["cohort_short"], data_arg="cohort_level_data")
    check_cohort_completeness(cohort_level_data, "cohort_level_data")

    check_herd_id_unique(herd_level_data, "herd_level_data")
    check_herd_id_consistency(cohort_level_data, herd_level_data, "cohort_level_data", "herd_level_data")
