"""Input validation for :func:`gleampy.run_allocation_module`.

Port of ``R/validate_run_allocation_inputs.R``.
"""

from __future__ import annotations

import pandas as pd

from ._shared import (
    check_data_table,
    check_herd_id_consistency,
    check_herd_id_unique,
    check_required_columns,
    normalize_optional_is_egg_producing_column,
    validator,
)

#: Cohort-level columns required by the allocation module.
REQUIRED_COHORT_COLS: tuple[str, ...] = (
    "herd_id",
    "cohort_short",
    "milk_production_fpcm_cohort",
    "live_weight_cohort_at_slaughter",
    "meat_production_live_weight_cohort",
    "metabolic_energy_req_fibre_production",
    "cohort_stock_size",
    "metabolic_energy_req_work",
    "egg_production_mass_cohort",
    "nondemo_productive_phase_id",
)

#: Herd-level columns required by the allocation module.
REQUIRED_HERD_COLS: tuple[str, ...] = (
    "herd_id",
    "species_short",
    "live_weight_at_birth",
    "milk_protein_fraction_standard",
    "milk_fat_fraction_standard",
    "milk_lactose_fraction_standard",
    "ratio_me_to_ne",
)


@validator
def validate_run_allocation_module_inputs(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
) -> None:
    """Validate the inputs of :func:`gleampy.run_allocation_module`.

    Checks that both tables are non-empty data frames with the required
    columns (``is_egg_producing`` is required only when ``CHK`` herds are
    present, and added as ``NA`` otherwise -- in place, as in R), that both
    tables cover the same ``herd_id`` set and that ``herd_id`` is unique in
    ``herd_level_data``.
    """
    # --- Basic type and structure checks
    check_data_table(cohort_level_data, "cohort_level_data")
    check_data_table(herd_level_data, "herd_level_data")
    normalize_optional_is_egg_producing_column(cohort_level_data, herd_level_data)

    # --- Required columns
    check_required_columns(cohort_level_data, REQUIRED_COHORT_COLS, "cohort_level_data")
    check_required_columns(herd_level_data, REQUIRED_HERD_COLS, "herd_level_data")

    if (herd_level_data["species_short"] == "CHK").any():
        check_required_columns(cohort_level_data, ["is_egg_producing"], "cohort_level_data")

    # --- Cross-table: same herd_id set
    check_herd_id_consistency(cohort_level_data, herd_level_data, "cohort_level_data", "herd_level_data")

    # --- Herd: one row per herd_id
    check_herd_id_unique(herd_level_data, "herd_level_data")
