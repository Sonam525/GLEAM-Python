"""Input validation for the enteric methane core model.

Port of ``R/validate_emissions_enteric_core_model.R``. The R validators are
called row by row on scalars; these versions check whole vectors at once.
"""

from __future__ import annotations

from typing import Any

from ._shared import validate_param_range, validate_scalar_character, validator


@validator
def validate_ym_inputs(
    species_short: Any,
    cohort_short: Any,
    ration_digestibility_fraction: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.calc_conversion_factor_ym`.

    ``species_short`` and ``cohort_short`` must be non-missing character
    values; ``ration_digestibility_fraction`` must lie within its
    ``parameter_ranges`` bounds (fraction of gross energy).
    """
    validate_scalar_character(species_short, "species_short")
    validate_scalar_character(cohort_short, "cohort_short")
    validate_param_range(ration_digestibility_fraction, "ration_digestibility_fraction")


@validator
def validate_enteric_emission_inputs(
    species_short: Any,
    ch4_conversion_factor_ym: Any,
    ch4_mitigation_factor: Any,
    ration_gross_energy: Any,
    ration_intake: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.calc_ch4_enteric`.

    ``species_short`` must be a non-missing character value; the numeric
    arguments are checked against ``parameter_ranges``.
    """
    validate_scalar_character(species_short, "species_short")
    validate_param_range(ch4_conversion_factor_ym, "ch4_conversion_factor_ym")
    validate_param_range(ch4_mitigation_factor, "ch4_mitigation_factor")
    validate_param_range(ration_gross_energy, "ration_gross_energy")
    validate_param_range(ration_intake, "ration_intake")
