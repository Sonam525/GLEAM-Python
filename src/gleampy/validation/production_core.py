"""Input validation for the production core model.

Port of ``R/validate_production_core_model.R``. R validates one row at a time;
the species- and cohort-specific branches become boolean masks here and every
check is applied to all rows of its branch at once, in the R order of the
checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import constants as K
from .._utils import as_float, as_str, is_true, isin
from ._shared import (
    abort,
    validate_animal_species,
    validate_cohort_code,
    validate_is_egg_producing_flag,
    validate_param_range,
    validator,
)
from .nitrogen_balance_core import (
    broadcast_raw,
    masked_param_range,
    masked_positive_numeric,
    masked_scalar_numeric,
)

_FIBRE_SPECIES = ("GTS", "SHP", "CML")
_FIBRE_VALIDATED_COHORTS = ("FA", "FS", "MA", "MS")


@validator
def validate_milk_outputs_inputs(
    species_short: Any,
    cohort_short: Any,
    milk_yield_day: Any,
    simulation_duration: Any,
    cohort_stock_size: Any,
    lactating_females_fraction: Any,
    milk_protein_fraction: Any,
    milk_fat_fraction: Any,
    milk_lactose_fraction: Any,
    milk_protein_fraction_standard: Any,
    milk_fat_fraction_standard: Any,
    milk_lactose_fraction_standard: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.calc_milk_production`.

    Milk parameters are only checked for adult females (``FA``) of
    milk-producing species, the only rows where they are used.
    """
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)

    (sp, co, myd, sd, css, lff, mpf, mff, mlf, mpfs, mffs, mlfs) = broadcast_raw(
        species_short, cohort_short, milk_yield_day, simulation_duration, cohort_stock_size,
        lactating_females_fraction, milk_protein_fraction, milk_fat_fraction,
        milk_lactose_fraction, milk_protein_fraction_standard, milk_fat_fraction_standard,
        milk_lactose_fraction_standard,
    )
    mask = isin(as_str(sp), K.GLEAM_SPECIES_MILK_PRODUCERS) & (as_str(co) == "FA")
    if not mask.any():
        return
    masked_scalar_numeric(sd, "simulation_duration", mask)
    masked_scalar_numeric(css, "cohort_stock_size", mask)
    masked_param_range(myd, "milk_yield_day", mask)
    masked_param_range(lff, "lactating_females_fraction", mask)
    masked_param_range(mpf, "milk_protein_fraction", mask)
    masked_param_range(mff, "milk_fat_fraction", mask)
    masked_param_range(mlf, "milk_lactose_fraction", mask)
    masked_param_range(mpfs, "milk_protein_fraction_standard", mask)
    masked_param_range(mffs, "milk_fat_fraction_standard", mask)
    masked_param_range(mlfs, "milk_lactose_fraction_standard", mask)


@validator
def validate_fibre_output_inputs(
    species_short: Any,
    cohort_short: Any,
    fibre_yield_year: Any,
    simulation_duration: Any,
    cohort_stock_size: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.calc_fibre_production`.

    As in R, fibre parameters are only checked for ``FA``, ``FS``, ``MA`` and
    ``MS`` cohorts of fibre-producing species (``GTS``, ``SHP``, ``CML``);
    ``species_short`` itself is not validated.
    """
    validate_cohort_code(cohort_short)

    sp, co, fy, sd, css = broadcast_raw(species_short, cohort_short, fibre_yield_year, simulation_duration, cohort_stock_size)
    mask = isin(as_str(sp), _FIBRE_SPECIES) & isin(as_str(co), _FIBRE_VALIDATED_COHORTS)
    if not mask.any():
        return
    masked_scalar_numeric(sd, "simulation_duration", mask)
    masked_scalar_numeric(css, "cohort_stock_size", mask)
    masked_param_range(fy, "fibre_yield_year", mask)


@validator
def validate_egg_output_inputs(
    species_short: Any,
    cohort_short: Any,
    egg_output_human_consumption: Any,
    egg_average_weight: Any,
    simulation_duration: Any,
    egg_protein_fraction: Any,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> None:
    """Validate the inputs of :func:`gleampy.calc_egg_production`.

    Checks the egg-producing flag placement (CHK ``FA``, or ``FN`` in phase
    2) and, for cohorts whose flag is a logical TRUE (R's ``isTRUE()``), the
    egg parameters.

    Parameters
    ----------
    species_short, cohort_short : str or array-like
        Species and cohort codes.
    egg_output_human_consumption : float or array-like
        Annual egg output for human consumption (eggs/year).
    egg_average_weight : float or array-like
        Average egg weight (kg/egg).
    simulation_duration : float or array-like
        Length of the assessment period (days).
    egg_protein_fraction : float or array-like
        Protein content of whole egg (kg protein/kg egg).
    nondemo_productive_phase_id : float or array-like
        Productive phase of non-demographic cohorts.
    is_egg_producing : bool or array-like
        Egg-producing ``CHK`` cohort flag (logical).
    """
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    validate_is_egg_producing_flag(
        species_short=species_short,
        cohort_short=cohort_short,
        is_egg_producing=is_egg_producing,
        nondemo_productive_phase_id=nondemo_productive_phase_id,
    )

    # R: if (!isTRUE(is_egg_producing)) return()
    eohc, eaw, sd, epf, laying = broadcast_raw(
        egg_output_human_consumption, egg_average_weight, simulation_duration,
        egg_protein_fraction, is_true(is_egg_producing),
    )
    mask = np.asarray(laying, dtype=bool)
    if not mask.any():
        return
    masked_scalar_numeric(eohc, "egg_output_human_consumption", mask)
    masked_positive_numeric(eaw, "egg_average_weight", mask)
    masked_scalar_numeric(sd, "simulation_duration", mask)
    masked_scalar_numeric(epf, "egg_protein_fraction", mask)
    if (as_float(eohc)[mask] < 0).any():
        abort("`egg_output_human_consumption` must be greater than or equal to 0.")
    epf_f = as_float(epf)[mask]
    if ((epf_f < 0) | (epf_f > 1)).any():
        abort("`egg_protein_fraction` must be between 0 and 1.")


@validator
def validate_meat_outputs_inputs(
    offtake_heads_assessment: Any,
    live_weight_cohort_at_slaughter: Any,
    carcass_dressing_fraction: Any,
    bone_free_meat_fraction: Any,
    meat_protein_fraction: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.calc_meat_production` (``parameter_ranges``)."""
    validate_param_range(offtake_heads_assessment, "offtake_heads_assessment")
    validate_param_range(live_weight_cohort_at_slaughter, "live_weight_cohort_at_slaughter")
    validate_param_range(carcass_dressing_fraction, "carcass_dressing_fraction")
    validate_param_range(bone_free_meat_fraction, "bone_free_meat_fraction")
    validate_param_range(meat_protein_fraction, "meat_protein_fraction")
