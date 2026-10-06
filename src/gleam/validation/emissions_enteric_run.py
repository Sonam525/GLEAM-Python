"""Run-level input validation for the enteric methane module.

Port of ``R/validate_run_emissions_enteric_inputs.R``.
"""

from __future__ import annotations

from typing import Any

from ._shared import (
    check_cohort_completeness,
    check_data_table,
    check_required_columns,
    validate_cohort_short_values,
    validate_species_short_values,
    validator,
)

#: Columns required in the cohort-level table.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "herd_id", "species_short", "cohort_short", "ration_digestibility_fraction",
    "ration_gross_energy", "ration_intake",
)


@validator
def validate_run_emissions_enteric_module_inputs(data: Any) -> None:
    """Validate the cohort-level table of :func:`gleam.run_emissions_enteric_module`.

    Checks that ``data`` is a non-empty table with the required columns, valid
    cohort and species codes, and the 6 demographic cohorts once per herd.
    """
    check_data_table(data, "data")
    check_required_columns(data, REQUIRED_COLUMNS, "data")
    validate_cohort_short_values(data["cohort_short"], data_arg="data")
    validate_species_short_values(data["species_short"], data_arg="data")
    check_cohort_completeness(data, "data")
