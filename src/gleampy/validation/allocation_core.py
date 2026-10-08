"""Input validation for the allocation core model.

Port of ``R/validate_allocation_core_model.R``. The R validators are called on
scalars inside row-wise ``calc_*`` calls; these versions check whole arrays at
once. Checks that R applies only for some species (e.g. slaughter weights for
non-``PGS`` species, ``ratio_me_to_ne`` for ``CML``) are applied to the
matching elements only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import as_float, as_str, is_na, isin
from ._shared import (
    abort,
    validate_animal_species,
    validate_cohort_code,
    validate_is_egg_producing_flag,
    validate_param_range,
    validate_scalar_numeric,
    validator,
)

_FIBRE_SPECIES = ("SHP", "GTS", "CML")


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


def _values(x: Any) -> np.ndarray:
    """Raw values of ``x`` as a numpy array, keeping the original dtype."""
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy()
    return np.asarray(x) if not isinstance(x, np.ndarray) else x


def _shape(*xs: Any) -> tuple[int, ...]:
    return np.broadcast_shapes(*(np.shape(_values(x)) for x in xs))


def _subset(x: Any, mask: np.ndarray, shape: tuple[int, ...]) -> tuple[Any, list[int] | None]:
    """Elements of ``x`` (broadcast to ``shape``) where ``mask`` holds.

    Returns ``x`` itself when every element is selected (so scalar error
    messages keep R's form), otherwise the selected values with their 1-based
    positions as labels for error messages.
    """
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), shape)
    if mask.all():
        return x, None
    vals = np.broadcast_to(_values(x), shape)[mask]
    return vals, list(np.flatnonzero(mask.ravel()) + 1)


def _is_numeric_values(vals: np.ndarray) -> bool:
    """R ``is.numeric()`` on an array (NA allowed, logicals/strings are not numeric)."""
    if vals.dtype.kind in "iuf":
        return True
    if vals.dtype == object:
        return all(
            is_na(v) or (isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)))
            for v in vals.ravel()
        )
    return False


def check_positive_ratio_me_to_ne(ratio_me_to_ne: Any, mask: np.ndarray, shape: tuple[int, ...]) -> None:
    """``ratio_me_to_ne`` must be a positive number where ``mask`` holds (CML rows)."""
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), shape)
    if not mask.any():
        return
    vals = np.broadcast_to(_values(ratio_me_to_ne), shape)[mask]
    ok = _is_numeric_values(vals)
    if ok:
        f = as_float(vals)
        with np.errstate(invalid="ignore"):
            ok = not (np.isnan(f).any() or (f <= 0).any())
    if not ok:
        abort("`ratio_me_to_ne` must be a positive numeric value (ME/NE).")


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------


@validator
def validate_allocation_milk_inputs(
    milk_production_fpcm_cohort: Any,
    milk_protein_fraction_standard: Any,
    milk_fat_fraction_standard: Any,
    milk_lactose_fraction_standard: Any,
) -> None:
    """Validate inputs of :func:`gleampy.calc_milk_allocation_energy`.

    ``milk_production_fpcm_cohort`` must be within its parameter range; the
    standard milk fractions are only checked where milk output is not 0 (R
    returns early when ``identical(milk_production_fpcm_cohort, 0)``).
    """
    validate_param_range(milk_production_fpcm_cohort, "milk_production_fpcm_cohort")
    shape = _shape(
        milk_production_fpcm_cohort,
        milk_protein_fraction_standard,
        milk_fat_fraction_standard,
        milk_lactose_fraction_standard,
    )
    need = np.broadcast_to(as_float(_values(milk_production_fpcm_cohort)) != 0, shape)
    for x, name in (
        (milk_protein_fraction_standard, "milk_protein_fraction_standard"),
        (milk_fat_fraction_standard, "milk_fat_fraction_standard"),
        (milk_lactose_fraction_standard, "milk_lactose_fraction_standard"),
    ):
        vals, labels = _subset(x, need, shape)
        validate_param_range(vals, name, labels=labels)


@validator
def validate_allocation_meat_inputs(
    species_short: Any,
    cohort_short: Any,
    meat_production_live_weight_cohort: Any,
    live_weight_cohort_at_slaughter: Any = np.nan,
    live_weight_at_birth: Any = np.nan,
    ratio_me_to_ne: Any = np.nan,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> None:
    """Validate inputs of :func:`gleampy.calc_meat_allocation_energy`.

    Slaughter and birth weights are only checked for non-``PGS`` species and
    ``ratio_me_to_ne`` only for ``CML``. ``is_egg_producing`` must be logical
    (``validate_is_egg_producing_flag``; strings and numbers are rejected, as
    in R).

    Parameters
    ----------
    species_short, cohort_short : str or array-like
        Species and cohort codes.
    meat_production_live_weight_cohort : float or array-like
        Meat produced as live weight (kg/cohort/assessment period).
    live_weight_cohort_at_slaughter, live_weight_at_birth : float or array-like
        Live weights (kg).
    ratio_me_to_ne : float or array-like
        Ratio of metabolizable to net energy.
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
    validate_scalar_numeric(meat_production_live_weight_cohort, "meat_production_live_weight_cohort")
    validate_param_range(meat_production_live_weight_cohort, "meat_production_live_weight_cohort")

    shape = _shape(species_short, live_weight_cohort_at_slaughter, live_weight_at_birth, ratio_me_to_ne)
    sp = np.broadcast_to(as_str(_values(species_short)), shape)

    # Slaughter and birth weight only needed for non-PGS species
    non_pgs = sp != "PGS"
    for x, name in (
        (live_weight_cohort_at_slaughter, "live_weight_cohort_at_slaughter"),
        (live_weight_at_birth, "live_weight_at_birth"),
    ):
        vals, labels = _subset(x, non_pgs, shape)
        validate_param_range(vals, name, labels=labels)

    # ratio_me_to_ne only needed for CML
    check_positive_ratio_me_to_ne(ratio_me_to_ne, sp == "CML", shape)


@validator
def validate_allocation_fibre_inputs(
    species_short: Any,
    cohort_stock_size: Any = np.nan,
    metabolic_energy_req_fibre_production: Any = np.nan,
    ratio_me_to_ne: Any = np.nan,
    simulation_duration: Any = np.nan,
) -> None:
    """Validate inputs of :func:`gleampy.calc_fibre_allocation_energy`.

    Numeric inputs are only checked for fibre species (``SHP``, ``GTS``,
    ``CML``); ``ratio_me_to_ne`` only for ``CML``.
    """
    validate_animal_species(species_short)

    shape = _shape(
        species_short, cohort_stock_size, metabolic_energy_req_fibre_production,
        ratio_me_to_ne, simulation_duration,
    )
    sp = np.broadcast_to(as_str(_values(species_short)), shape)
    fibre = isin(sp, _FIBRE_SPECIES)
    if not fibre.any():
        return

    for x, name in (
        (metabolic_energy_req_fibre_production, "metabolic_energy_req_fibre_production"),
        (cohort_stock_size, "cohort_stock_size"),
        (simulation_duration, "simulation_duration"),
    ):
        vals, labels = _subset(x, fibre, shape)
        validate_param_range(vals, name, labels=labels)

    check_positive_ratio_me_to_ne(ratio_me_to_ne, sp == "CML", shape)


@validator
def validate_allocation_work_inputs(
    species_short: Any,
    cohort_stock_size: Any,
    metabolic_energy_req_work: Any,
    simulation_duration: Any,
    ratio_me_to_ne: Any = np.nan,
) -> None:
    """Validate inputs of :func:`gleampy.calc_work_allocation_energy`.

    ``ratio_me_to_ne`` is only checked for ``CML``.
    """
    validate_animal_species(species_short)

    validate_param_range(metabolic_energy_req_work, "metabolic_energy_req_work")
    validate_param_range(cohort_stock_size, "cohort_stock_size")
    validate_param_range(simulation_duration, "simulation_duration")

    shape = _shape(species_short, cohort_stock_size, metabolic_energy_req_work, simulation_duration, ratio_me_to_ne)
    sp = np.broadcast_to(as_str(_values(species_short)), shape)
    check_positive_ratio_me_to_ne(ratio_me_to_ne, sp == "CML", shape)


@validator
def validate_allocation_egg_inputs(
    species_short: Any,
    cohort_short: Any,
    egg_production_mass_cohort: Any,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> None:
    """Validate inputs of :func:`gleampy.calc_egg_allocation_energy`.

    Parameters
    ----------
    species_short, cohort_short : str or array-like
        Species and cohort codes.
    egg_production_mass_cohort : float or array-like
        Egg mass produced (kg/cohort/assessment period), at least 0.
    nondemo_productive_phase_id : float or array-like
        Productive phase of non-demographic cohorts.
    is_egg_producing : bool or array-like
        Egg-producing ``CHK`` cohort flag (logical; strings and numbers are
        rejected, as in R).
    """
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    validate_is_egg_producing_flag(
        species_short=species_short,
        cohort_short=cohort_short,
        is_egg_producing=is_egg_producing,
        nondemo_productive_phase_id=nondemo_productive_phase_id,
    )
    validate_scalar_numeric(egg_production_mass_cohort, "egg_production_mass_cohort")

    if (np.atleast_1d(as_float(_values(egg_production_mass_cohort))) < 0).any():
        abort("`egg_production_mass_cohort` must be greater than or equal to 0.")
