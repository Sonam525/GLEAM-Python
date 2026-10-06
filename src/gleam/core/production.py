"""Production outputs: milk, fibre, eggs and meat.

Port of ``R/core_model_production.R``. The functions are vectorised: inputs
may be scalars, lists, numpy arrays or pandas Series and are broadcast against
each other; the scalar species / cohort branches of the R code are reproduced
element by element with boolean masks. Functions returning a named list in R
return a ``dict`` with the same keys.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import constants as K
from .._utils import all_scalar, as_bool, as_float, as_str, broadcast, finalize, finalize_dict, is_true, isin
from ..validation.production_core import (
    validate_egg_output_inputs,
    validate_fibre_output_inputs,
    validate_meat_outputs_inputs,
    validate_milk_outputs_inputs,
)

_FIBRE_SPECIES = ("GTS", "SHP", "CML")
_FIBRE_COHORTS = ("FA", "FS", "MA", "MS", "FN", "MN")


def _vec(*xs: Any) -> list[np.ndarray]:
    """Broadcast the converted inputs against each other as 1-d arrays."""
    return [np.atleast_1d(a) for a in broadcast(*xs)]


def calc_milk_production(
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
) -> dict[str, Any]:
    """Milk production of the adult female cohort over the assessment period.

    Non-zero only for ``FA`` of milk-producing species (``CTL``, ``BFL``,
    ``SHP``, ``GTS``, ``CML``)::

        milk_production_mass_cohort    = milk_yield_day * simulation_duration
                                         * cohort_stock_size * lactating_females_fraction
        milk_production_protein_cohort = milk_production_mass_cohort * milk_protein_fraction
        milk_production_fpcm_cohort    = (E_milk / E_standard) * milk_production_mass_cohort

    with milk energy content (Mcal/kg; IDF, 2022, Eq. 10)
    ``E = 0.0929 * fat + 0.0547 * protein + 0.0395 * lactose`` for the actual
    and the standard milk composition.

    Parameters
    ----------
    species_short, cohort_short : str or array-like
        Species and cohort codes.
    milk_yield_day : float or array-like
        Average milk yield per milk-producing animal (kg/head/day).
    simulation_duration : float or array-like
        Length of the assessment period (days).
    cohort_stock_size : float or array-like
        Average cohort population (heads).
    lactating_females_fraction : float or array-like
        Fraction of adult females lactating during the period.
    milk_protein_fraction, milk_fat_fraction, milk_lactose_fraction : float or array-like
        Milk composition (kg/kg milk).
    milk_protein_fraction_standard, milk_fat_fraction_standard, milk_lactose_fraction_standard : float or array-like
        Standard milk composition for FPCM (suggested 0.033, 0.04, 0.048 kg/kg).

    Returns
    -------
    dict
        ``milk_production_mass_cohort`` (kg), ``milk_production_protein_cohort``
        (kg protein) and ``milk_production_fpcm_cohort`` (kg FPCM) per cohort
        over the assessment period.
    """
    validate_milk_outputs_inputs(
        species_short=species_short,
        cohort_short=cohort_short,
        milk_yield_day=milk_yield_day,
        simulation_duration=simulation_duration,
        cohort_stock_size=cohort_stock_size,
        lactating_females_fraction=lactating_females_fraction,
        milk_protein_fraction=milk_protein_fraction,
        milk_fat_fraction=milk_fat_fraction,
        milk_lactose_fraction=milk_lactose_fraction,
        milk_protein_fraction_standard=milk_protein_fraction_standard,
        milk_fat_fraction_standard=milk_fat_fraction_standard,
        milk_lactose_fraction_standard=milk_lactose_fraction_standard,
    )

    (sp, co, myd, sd, css, lff, mpf, mff, mlf, mpfs, mffs, mlfs) = _vec(
        as_str(species_short), as_str(cohort_short), as_float(milk_yield_day),
        as_float(simulation_duration), as_float(cohort_stock_size),
        as_float(lactating_females_fraction), as_float(milk_protein_fraction),
        as_float(milk_fat_fraction), as_float(milk_lactose_fraction),
        as_float(milk_protein_fraction_standard), as_float(milk_fat_fraction_standard),
        as_float(milk_lactose_fraction_standard),
    )
    producing = isin(sp, K.GLEAM_SPECIES_MILK_PRODUCERS) & (co == "FA")

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        # Energy content of standard and actual milk (Mcal/kg) - IDF 2022
        energy_standard = 0.0929 * mffs + 0.0547 * mpfs + 0.0395 * mlfs
        energy_milk = 0.0929 * mff + 0.0547 * mpf + 0.0395 * mlf
        milk_production = myd * sd * css * lff
        milk_protein_production = milk_production * mpf
        energy_ratio = energy_milk / energy_standard
        fpcm_production = energy_ratio * milk_production

    out = {
        "milk_production_mass_cohort": np.where(producing, milk_production, 0.0),
        "milk_production_protein_cohort": np.where(producing, milk_protein_production, 0.0),
        "milk_production_fpcm_cohort": np.where(producing, fpcm_production, 0.0),
    }
    return finalize_dict(
        out,
        all_scalar(
            species_short, cohort_short, milk_yield_day, simulation_duration, cohort_stock_size,
            lactating_females_fraction, milk_protein_fraction, milk_fat_fraction,
            milk_lactose_fraction, milk_protein_fraction_standard, milk_fat_fraction_standard,
            milk_lactose_fraction_standard,
        ),
    )


def calc_fibre_production(
    species_short: Any,
    cohort_short: Any,
    fibre_yield_year: Any,
    simulation_duration: Any,
    cohort_stock_size: Any,
) -> Any:
    """Fibre production over the assessment period (kg/cohort/assessment period).

    Non-zero only for cohorts ``FA``, ``FS``, ``MA``, ``MS``, ``FN``, ``MN`` of
    fibre-producing species (``CML``, ``SHP``, ``GTS``)::

        fibre_production_cohort = fibre_yield_year / 365 * simulation_duration * cohort_stock_size

    Parameters
    ----------
    species_short, cohort_short : str or array-like
        Species and cohort codes.
    fibre_yield_year : float or array-like
        Annual fibre yield (kg/head/year).
    simulation_duration : float or array-like
        Length of the assessment period (days).
    cohort_stock_size : float or array-like
        Average cohort population (heads).

    Returns
    -------
    float or numpy.ndarray
        Fibre produced by the cohort over the assessment period (kg).
    """
    validate_fibre_output_inputs(
        species_short=species_short,
        cohort_short=cohort_short,
        fibre_yield_year=fibre_yield_year,
        simulation_duration=simulation_duration,
        cohort_stock_size=cohort_stock_size,
    )
    sp, co, fy, sd, css = _vec(
        as_str(species_short), as_str(cohort_short), as_float(fibre_yield_year),
        as_float(simulation_duration), as_float(cohort_stock_size),
    )
    producing = isin(sp, _FIBRE_SPECIES) & isin(co, _FIBRE_COHORTS)
    with np.errstate(invalid="ignore", over="ignore"):
        fibre = fy / 365 * sd * css
    out = np.where(producing, fibre, 0.0)
    return finalize(out, all_scalar(species_short, cohort_short, fibre_yield_year, simulation_duration, cohort_stock_size))


def calc_egg_production(
    species_short: Any,
    cohort_short: Any,
    egg_output_human_consumption: Any,
    egg_average_weight: Any,
    simulation_duration: Any,
    egg_protein_fraction: Any = 0.125,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> dict[str, Any]:
    """Egg outputs of egg-producing chicken cohorts over the assessment period.

    Non-zero only where ``is_egg_producing`` is TRUE (CHK ``FA``, or ``FN``
    in productive phase 2)::

        egg_production_number_cohort  = egg_output_human_consumption / 365 * simulation_duration
        egg_production_mass_cohort    = egg_production_number_cohort * egg_average_weight
        egg_production_protein_cohort = egg_production_mass_cohort * egg_protein_fraction

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
        Protein content of whole egg (kg protein/kg egg; default 0.125,
        Caffa et al., 2025).
    nondemo_productive_phase_id : float or array-like
        Productive phase of non-demographic cohorts (only validated).
    is_egg_producing : bool or array-like
        Egg-producing CHK cohort flag.

    Returns
    -------
    dict
        ``egg_production_number_cohort`` (eggs), ``egg_production_mass_cohort``
        (kg) and ``egg_production_protein_cohort`` (kg protein).
    """
    validate_egg_output_inputs(
        species_short=species_short,
        cohort_short=cohort_short,
        nondemo_productive_phase_id=nondemo_productive_phase_id,
        is_egg_producing=is_egg_producing,
        egg_output_human_consumption=egg_output_human_consumption,
        egg_average_weight=egg_average_weight,
        simulation_duration=simulation_duration,
        egg_protein_fraction=egg_protein_fraction,
    )
    _, _, eohc, eaw, sd, epf, egg = _vec(
        as_str(species_short), as_str(cohort_short), as_float(egg_output_human_consumption),
        as_float(egg_average_weight), as_float(simulation_duration), as_float(egg_protein_fraction),
        as_bool(is_egg_producing),
    )
    producing = is_true(egg)
    with np.errstate(invalid="ignore", over="ignore"):
        number = eohc / 365 * sd
        mass = number * eaw
        protein = mass * epf
    out = {
        "egg_production_number_cohort": np.where(producing, number, 0.0),
        "egg_production_mass_cohort": np.where(producing, mass, 0.0),
        "egg_production_protein_cohort": np.where(producing, protein, 0.0),
    }
    return finalize_dict(
        out,
        all_scalar(
            species_short, cohort_short, egg_output_human_consumption, egg_average_weight,
            simulation_duration, egg_protein_fraction, nondemo_productive_phase_id, is_egg_producing,
        ),
    )


def calc_meat_production(
    offtake_heads_assessment: Any,
    live_weight_cohort_at_slaughter: Any,
    carcass_dressing_fraction: Any,
    bone_free_meat_fraction: Any,
    meat_protein_fraction: Any,
) -> dict[str, Any]:
    """Meat outputs of the cohort offtake over the assessment period.

    ::

        meat_production_live_weight_cohort    = offtake_heads_assessment * live_weight_cohort_at_slaughter
        meat_production_carcass_weight_cohort = live_weight * carcass_dressing_fraction
        meat_production_bone_free_meat_cohort = carcass_weight * bone_free_meat_fraction
        meat_production_protein_cohort        = bone_free_meat * meat_protein_fraction

    Parameters
    ----------
    offtake_heads_assessment : float or array-like
        Animals removed via offtake over the assessment period (heads).
    live_weight_cohort_at_slaughter : float or array-like
        Live weight at slaughter (kg).
    carcass_dressing_fraction : float or array-like
        Carcass weight over live weight (fraction).
    bone_free_meat_fraction : float or array-like
        Bone-free meat over carcass weight (fraction).
    meat_protein_fraction : float or array-like
        Protein content of bone-free meat (kg protein/kg).

    Returns
    -------
    dict
        ``meat_production_live_weight_cohort``,
        ``meat_production_carcass_weight_cohort``,
        ``meat_production_bone_free_meat_cohort`` (kg) and
        ``meat_production_protein_cohort`` (kg protein) per cohort.
    """
    validate_meat_outputs_inputs(
        offtake_heads_assessment=offtake_heads_assessment,
        live_weight_cohort_at_slaughter=live_weight_cohort_at_slaughter,
        carcass_dressing_fraction=carcass_dressing_fraction,
        bone_free_meat_fraction=bone_free_meat_fraction,
        meat_protein_fraction=meat_protein_fraction,
    )
    offtake, lw, cdf, bfm, mpf = _vec(
        as_float(offtake_heads_assessment), as_float(live_weight_cohort_at_slaughter),
        as_float(carcass_dressing_fraction), as_float(bone_free_meat_fraction),
        as_float(meat_protein_fraction),
    )
    with np.errstate(invalid="ignore", over="ignore"):
        live_weight = offtake * lw
        carcass_weight = live_weight * cdf
        bone_free_meat = carcass_weight * bfm
        protein = bone_free_meat * mpf
    out = {
        "meat_production_live_weight_cohort": live_weight,
        "meat_production_carcass_weight_cohort": carcass_weight,
        "meat_production_bone_free_meat_cohort": bone_free_meat,
        "meat_production_protein_cohort": protein,
    }
    return finalize_dict(
        out,
        all_scalar(
            offtake_heads_assessment, live_weight_cohort_at_slaughter, carcass_dressing_fraction,
            bone_free_meat_fraction, meat_protein_fraction,
        ),
    )
