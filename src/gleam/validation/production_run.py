"""Run-level input validation for the production module.

Port of ``R/validate_run_production_inputs.R``.
"""

from __future__ import annotations

import pandas as pd

from ._shared import (
    check_cohort_completeness,
    check_data_table,
    check_herd_id_consistency,
    check_herd_id_unique,
    check_required_columns,
    normalize_optional_is_egg_producing_column,
    validate_cohort_short_values,
    validator,
)

#: Columns required in the cohort-level table.
REQUIRED_COHORT_COLUMNS: tuple[str, ...] = (
    "herd_id", "cohort_short", "cohort_stock_size",
    "offtake_heads_assessment", "live_weight_cohort_at_slaughter",
)

#: Columns required in the herd-level table.
REQUIRED_HERD_COLUMNS: tuple[str, ...] = (
    "herd_id", "species_short",
    "milk_yield_day", "lactating_females_fraction",
    "milk_protein_fraction", "milk_fat_fraction", "milk_lactose_fraction",
    "milk_protein_fraction_standard", "milk_fat_fraction_standard", "milk_lactose_fraction_standard",
    "fibre_yield_year",
    "carcass_dressing_fraction", "bone_free_meat_fraction", "meat_protein_fraction",
)


@validator
def validate_run_production_module_inputs(
    cohort_level_data: pd.DataFrame, herd_level_data: pd.DataFrame
) -> None:
    """Validate the inputs of :func:`gleam.run_production_module`.

    Checks table structure, required columns, valid cohort codes, the 6
    demographic cohorts per herd, unique herd ids and the same herd-id set in
    both tables.

    Like R, this adds ``is_egg_producing = NA`` to ``cohort_level_data`` (in
    place) when the column is absent and there are no CHK herds; the module
    runner calls it on its own working copy.
    """
    check_data_table(cohort_level_data, "cohort_level_data")
    check_data_table(herd_level_data, "herd_level_data")
    normalize_optional_is_egg_producing_column(cohort_level_data, herd_level_data)

    check_required_columns(cohort_level_data, REQUIRED_COHORT_COLUMNS, "cohort_level_data")
    check_required_columns(herd_level_data, REQUIRED_HERD_COLUMNS, "herd_level_data")

    validate_cohort_short_values(cohort_level_data["cohort_short"], data_arg="cohort_level_data")
    check_cohort_completeness(cohort_level_data, "cohort_level_data")

    check_herd_id_unique(herd_level_data, "herd_level_data")

    check_herd_id_consistency(cohort_level_data, herd_level_data, "cohort_level_data", "herd_level_data")
