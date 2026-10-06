"""Input validation for the nitrogen balance core model.

Port of ``R/validate_nitrogen_balance_core_model.R``. R validates one row at a
time; the species- and cohort-specific branches become boolean masks here and
every check is applied to all rows of its branch at once, in the R order of
the checks. Range errors for vector inputs report the 1-based position of the
first offending element (``x[i] = ...``); scalar calls give the R message.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import as_bool, as_float, as_str, broadcast, is_true, isin, isna
from ._shared import (
    abort,
    validate_animal_species,
    validate_cohort_code,
    validate_is_egg_producing_flag,
    validate_param_range,
    validate_positive_numeric,
    validate_scalar_numeric,
    validator,
)

_FIBRE_SPECIES = ("SHP", "GTS", "CML")


# --------------------------------------------------------------------------
# Masked (branch-wise) helpers
# --------------------------------------------------------------------------


def _raw(x: Any) -> np.ndarray:
    """Input as a numpy array without type coercion (so type checks still apply)."""
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy()
    return np.asarray(x)


def broadcast_raw(*xs: Any) -> list[np.ndarray]:
    """Broadcast raw inputs against each other and flatten them to 1-d."""
    return [np.ravel(a) for a in broadcast(*(_raw(x) for x in xs))]


def _masked(check: Callable[..., None], x: np.ndarray, name: str, mask: np.ndarray, *, labels: bool = False) -> None:
    """Apply ``check`` to the elements of ``x`` selected by ``mask``."""
    if not mask.any():
        return
    sub = x[mask]
    if labels:
        lab = None if mask.size == 1 else list(np.flatnonzero(mask) + 1)
        check(sub, name, labels=lab)
    else:
        check(sub, name)


def _not_na(x: np.ndarray) -> np.ndarray:
    return ~isna(x)


def masked_param_range(x: np.ndarray, name: str, mask: np.ndarray) -> None:
    """``validate_param_range()`` on the selected elements."""
    _masked(validate_param_range, x, name, mask, labels=True)


def masked_scalar_numeric(x: np.ndarray, name: str, mask: np.ndarray) -> None:
    """``validate_scalar_numeric()`` on the selected elements."""
    _masked(validate_scalar_numeric, x, name, mask)


def masked_positive_numeric(x: np.ndarray, name: str, mask: np.ndarray) -> None:
    """``validate_positive_numeric()`` on the selected elements."""
    _masked(validate_positive_numeric, x, name, mask)


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------


@validator
def validate_nitrogen_intake_inputs(ration_intake: Any, ration_nitrogen: Any) -> None:
    """Validate the inputs of :func:`gleam.calc_nitrogen_intake`."""
    validate_param_range(ration_intake, "ration_intake")
    validate_param_range(ration_nitrogen, "ration_nitrogen")


@validator
def validate_nitrogen_retention_inputs(
    species_short: Any,
    cohort_short: Any,
    milk_protein_fraction: Any = np.nan,
    milk_yield_day: Any = np.nan,
    daily_weight_gain: Any = np.nan,
    fibre_yield_year: Any = np.nan,
    litter_size: Any = np.nan,
    parturition_rate: Any = np.nan,
    live_weight_at_weaning: Any = np.nan,
    live_weight_at_birth: Any = np.nan,
    pregnancy_duration: Any = np.nan,
    cohort_duration_days: Any = np.nan,
    cohort_stock_size: Any = np.nan,
    egg_output_human_consumption: Any = np.nan,
    egg_average_weight: Any = np.nan,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> None:
    """Validate the inputs of :func:`gleam.calc_nitrogen_retention`.

    Only the arguments used by each species / cohort branch are range-checked
    (see the R validator); birth weight must be below weaning weight for all
    species except ``CHK`` when both are given.
    """
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    validate_is_egg_producing_flag(
        species_short=species_short,
        cohort_short=cohort_short,
        is_egg_producing=is_egg_producing,
        nondemo_productive_phase_id=nondemo_productive_phase_id,
    )

    (sp, co, mpf, myd, dwg, fy, ls, pr, lww, lwb, pd_, cdd, css, eohc, eaw, egg) = broadcast_raw(
        species_short, cohort_short, milk_protein_fraction, milk_yield_day, daily_weight_gain,
        fibre_yield_year, litter_size, parturition_rate, live_weight_at_weaning,
        live_weight_at_birth, pregnancy_duration, cohort_duration_days, cohort_stock_size,
        egg_output_human_consumption, egg_average_weight, as_bool(is_egg_producing),
    )
    sp = as_str(sp)
    co = as_str(co)

    pgs = sp == "PGS"
    milk = isin(sp, K.GLEAM_SPECIES_MILK_PRODUCERS)
    chk = sp == "CHK"
    fibre_sp = isin(sp, _FIBRE_SPECIES)
    fa = co == "FA"
    fs = co == "FS"

    # --- PGS
    pgs_fa = pgs & fa
    pgs_fs = pgs & fs
    pgs_other = pgs & ~fa & ~fs
    masked_param_range(dwg, "daily_weight_gain", pgs_fs | pgs_other)
    masked_positive_numeric(pd_, "pregnancy_duration", pgs_fs)
    masked_param_range(cdd, "cohort_duration_days", pgs_fs)
    masked_param_range(ls, "litter_size", pgs_fa | pgs_fs)
    masked_param_range(pr, "parturition_rate", pgs_fa | pgs_fs)
    masked_param_range(lww, "live_weight_at_weaning", pgs_fa | pgs_fs)
    masked_param_range(lwb, "live_weight_at_birth", pgs_fa | pgs_fs)

    # --- milk-producing species (CTL, BFL, SHP, GTS, CML)
    milk_fa = milk & fa
    milk_fsmams = milk & isin(co, ("FS", "MA", "MS"))
    masked_param_range(mpf, "milk_protein_fraction", milk_fa & _not_na(mpf))
    masked_param_range(myd, "milk_yield_day", milk_fa & _not_na(myd))
    masked_param_range(dwg, "daily_weight_gain", milk & _not_na(dwg))
    masked_param_range(fy, "fibre_yield_year", (milk_fa | milk_fsmams) & fibre_sp & _not_na(fy))

    # --- CHK
    masked_scalar_numeric(dwg, "daily_weight_gain", chk)
    chk_egg = chk & is_true(egg)
    if chk_egg.any():
        masked_scalar_numeric(css, "cohort_stock_size", chk_egg)
        masked_positive_numeric(eaw, "egg_average_weight", chk_egg)
        masked_scalar_numeric(eohc, "egg_output_human_consumption", chk_egg)
        masked_scalar_numeric(pr, "parturition_rate", chk_egg)
        if (as_float(css)[chk_egg] < 0).any():
            abort("`cohort_stock_size` must be greater than or equal to 0.")
        if (as_float(eohc)[chk_egg] < 0).any():
            abort("`egg_output_human_consumption` must be greater than or equal to 0.")
        if (as_float(pr)[chk_egg] < 0).any():
            abort("`parturition_rate` must be greater than or equal to 0.")

    # --- Birth weight must be strictly below weaning weight when both are given
    lwb_f = as_float(lwb)
    lww_f = as_float(lww)
    with np.errstate(invalid="ignore"):
        bad_weights = ~chk & ~np.isnan(lwb_f) & ~np.isnan(lww_f) & (lwb_f >= lww_f)
    if bad_weights.any():
        abort("`live_weight_at_birth` must be strictly less than `live_weight_at_weaning`.")


@validator
def validate_nitrogen_excretion_inputs(species_short: Any, nitrogen_intake: Any, nitrogen_retention: Any) -> None:
    """Validate the inputs of :func:`gleam.calc_nitrogen_excretion`.

    Excretion is intake minus retention, so ``nitrogen_intake`` must be at
    least ``nitrogen_retention``.
    """
    validate_animal_species(species_short)
    validate_scalar_numeric(nitrogen_intake, "nitrogen_intake")
    validate_scalar_numeric(nitrogen_retention, "nitrogen_retention")
    ni, nr = broadcast(as_float(nitrogen_intake), as_float(nitrogen_retention))
    if (ni < nr).any():
        abort("`nitrogen_intake` must be greater than or equal to `nitrogen_retention`.")
