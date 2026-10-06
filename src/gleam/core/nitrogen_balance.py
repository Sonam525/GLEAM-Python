"""Nitrogen balance: intake, retention and excretion.

Port of ``R/core_model_nitrogen_balance.R``. The functions are vectorised:
inputs may be scalars, lists, numpy arrays or pandas Series and are broadcast
against each other; the scalar species / cohort branches of the R code are
reproduced element by element with boolean masks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import constants as K
from .._utils import all_scalar, as_bool, as_float, as_str, broadcast, finalize, is_true, isin
from ..validation._shared import abort
from ..validation.nitrogen_balance_core import (
    validate_nitrogen_excretion_inputs,
    validate_nitrogen_intake_inputs,
    validate_nitrogen_retention_inputs,
)

_FIBRE_SPECIES = ("SHP", "GTS", "CML")
_FIBRE_COHORTS = ("FA", "FS", "MA", "MS", "FN", "MN")


def _vec(*xs: Any) -> list[np.ndarray]:
    """Broadcast the converted inputs against each other as 1-d arrays."""
    return [np.atleast_1d(a) for a in broadcast(*xs)]


def calc_nitrogen_intake(ration_intake: Any, ration_nitrogen: Any) -> Any:
    """Daily nitrogen intake (kg N/head/day).

    IPCC (2006/2019) Tier 2, Eq. 10.32::

        nitrogen_intake = ration_intake * ration_nitrogen

    Parameters
    ----------
    ration_intake : float or array-like
        Average daily dry matter intake (kg DM/head/day).
    ration_nitrogen : float or array-like
        Average nitrogen content of the ration (kg N/kg DM).

    Returns
    -------
    float or numpy.ndarray
        Daily nitrogen intake (kg N/head/day).
    """
    validate_nitrogen_intake_inputs(ration_intake, ration_nitrogen)
    dmi, n = _vec(as_float(ration_intake), as_float(ration_nitrogen))
    with np.errstate(invalid="ignore", over="ignore"):
        nitrogen_intake = dmi * n
    return finalize(nitrogen_intake, all_scalar(ration_intake, ration_nitrogen))


def calc_nitrogen_retention(
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
) -> Any:
    """Daily nitrogen retention in tissues and products (kg N/head/day).

    **CTL, BFL, SHP, GTS, CML** (MPI, 2025, chapter 5)::

        retention = milk + growth + fibre
        milk   = milk_yield_day * milk_protein_fraction / 6.25
                 (only FA with milk_yield_day > 0)
        growth = daily_weight_gain * tissue_n   (only daily_weight_gain > 0)
        fibre  = fibre_yield_year / 365 * 0.134
                 (only SHP/GTS/CML, cohorts FA/FS/MA/MS/FN/MN, fibre_yield_year > 0)

    with ``tissue_n = 0.0326`` kg N/kg live weight for CTL/BFL and ``0.026``
    otherwise; components with missing inputs count as 0.

    **PGS** (IPCC 2019, Eq. 10.33A/B; tissue N 0.025 kg N/kg, protein
    digestibility 0.98, piglet correction 0.806)::

        FA:    (0.025 * litter_size * parturition_rate * (lw_weaning - lw_birth) / 0.98
                + 0.025 * litter_size * parturition_rate * lw_birth) / 365
        FS:    0.025 * daily_weight_gain
               + (0.025 * litter_size * (pregnancy_duration / cohort_duration_days)
                  * lw_birth / 0.806) / 365
        other: 0.025 * daily_weight_gain

    **CHK** (Lessire, 2004; Caffa et al., 2025)::

        growth = daily_weight_gain * 0.032          (only daily_weight_gain > 0)
        egg    = (egg_output_human_consumption / 365 / cohort_stock_size
                  + parturition_rate / 365) * egg_average_weight * 0.02
                 (only where is_egg_producing is TRUE)

    Parameters
    ----------
    species_short, cohort_short : str or array-like
        Species and cohort codes.
    milk_protein_fraction : float or array-like
        Milk protein fraction (kg protein/kg milk).
    milk_yield_day : float or array-like
        Average milk yield per milk-producing animal (kg/head/day).
    daily_weight_gain : float or array-like
        Average live weight gain over the cohort stage (kg/head/day).
    fibre_yield_year : float or array-like
        Annual fibre yield (kg/head/year).
    litter_size : float or array-like
        Offspring per parturition.
    parturition_rate : float or array-like
        Parturitions per adult female per year (eggs laid for reproduction
        per hen per year for CHK).
    live_weight_at_weaning, live_weight_at_birth : float or array-like
        Live weights at weaning and at birth (kg).
    pregnancy_duration : float or array-like
        Pregnancy duration (days).
    cohort_duration_days : float or array-like
        Time an animal spends in the cohort (days).
    cohort_stock_size : float or array-like
        Average cohort stock size (heads); CHK egg-producing cohorts.
    egg_output_human_consumption : float or array-like
        Annual egg output for human consumption (eggs/year); CHK.
    egg_average_weight : float or array-like
        Average egg weight (kg/egg); CHK.
    nondemo_productive_phase_id : float or array-like
        Productive phase of non-demographic cohorts (only validated).
    is_egg_producing : bool or array-like
        Egg-producing CHK cohort flag (``FA``, or ``FN`` in phase 2).

    Returns
    -------
    float or numpy.ndarray
        Daily nitrogen retention (kg N/head/day).
    """
    validate_nitrogen_retention_inputs(
        species_short, cohort_short, milk_protein_fraction, milk_yield_day,
        daily_weight_gain, fibre_yield_year, litter_size, parturition_rate,
        live_weight_at_weaning, live_weight_at_birth, pregnancy_duration, cohort_duration_days,
        cohort_stock_size, egg_output_human_consumption, egg_average_weight,
        nondemo_productive_phase_id, is_egg_producing,
    )

    (sp, co, mpf, myd, dwg, fy, ls, pr, lww, lwb, pd_, cdd, css, eohc, eaw, _phase, egg) = _vec(
        as_str(species_short), as_str(cohort_short), as_float(milk_protein_fraction),
        as_float(milk_yield_day), as_float(daily_weight_gain), as_float(fibre_yield_year),
        as_float(litter_size), as_float(parturition_rate), as_float(live_weight_at_weaning),
        as_float(live_weight_at_birth), as_float(pregnancy_duration), as_float(cohort_duration_days),
        as_float(cohort_stock_size), as_float(egg_output_human_consumption),
        as_float(egg_average_weight), as_float(nondemo_productive_phase_id), as_bool(is_egg_producing),
    )

    milk = isin(sp, K.GLEAM_SPECIES_MILK_PRODUCERS)
    pgs = sp == "PGS"
    chk = sp == "CHK"
    if not np.all(milk | pgs | chk):
        # R leaves `nitrogen_retention` undefined (error) for other species.
        bad = sp[~(milk | pgs | chk)][0]
        abort(f"`species_short` must be one of: {', '.join(K.GLEAM_SPECIES)} (got {bad!r}).")

    fa = co == "FA"
    fs = co == "FS"
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        dwg_pos = ~np.isnan(dwg) & (dwg > 0)

        # --- milk-producing species (CTL, BFL, SHP, GTS, CML)
        tissue_n = np.where(isin(sp, ("CTL", "BFL")), 0.0326, 0.026)
        milk_n = mpf / 6.25
        fibre_n = 0.134
        milk_comp = np.where(~np.isnan(myd) & fa & (myd > 0), myd * milk_n, 0.0)
        growth_comp = np.where(dwg_pos, dwg * tissue_n, 0.0)
        fibre_comp = np.where(
            ~np.isnan(fy) & isin(co, _FIBRE_COHORTS) & isin(sp, _FIBRE_SPECIES) & (fy > 0),
            fy / 365 * fibre_n,
            0.0,
        )
        ret_milk = milk_comp + growth_comp + fibre_comp

        # --- PGS
        ret_pgs_fa = (
            0.025 * ls * pr * (lww - lwb) / 0.98 + 0.025 * ls * pr * lwb
        ) / 365
        ret_pgs_fs = 0.025 * dwg + (0.025 * ls * (pd_ / cdd) * lwb / 0.806) / 365
        ret_pgs_other = 0.025 * dwg

        # --- CHK
        growth_n = 0.032
        egg_n = 0.02
        chk_growth = np.where(dwg_pos, dwg * growth_n, 0.0)
        human_consumption_eggs_head_day = eohc / 365 / css
        reproductive_eggs_head_day = pr / 365
        egg_mass_day = (human_consumption_eggs_head_day + reproductive_eggs_head_day) * eaw
        chk_egg = np.where(is_true(egg), egg_mass_day * egg_n, 0.0)
        ret_chk = chk_growth + chk_egg

    nitrogen_retention = np.select(
        [milk, pgs & fa, pgs & fs, pgs, chk],
        [ret_milk, ret_pgs_fa, ret_pgs_fs, ret_pgs_other, ret_chk],
        default=np.nan,
    )
    return finalize(
        nitrogen_retention,
        all_scalar(
            species_short, cohort_short, milk_protein_fraction, milk_yield_day, daily_weight_gain,
            fibre_yield_year, litter_size, parturition_rate, live_weight_at_weaning,
            live_weight_at_birth, pregnancy_duration, cohort_duration_days, cohort_stock_size,
            egg_output_human_consumption, egg_average_weight, nondemo_productive_phase_id,
            is_egg_producing,
        ),
    )


def calc_nitrogen_excretion(species_short: Any, nitrogen_intake: Any, nitrogen_retention: Any) -> Any:
    """Daily nitrogen excretion (kg N/head/day).

    IPCC (2019), Eq. 10.31A: nitrogen consumed and not retained in tissues or
    products is excreted in urine and dung::

        nitrogen_excretion = nitrogen_intake - nitrogen_retention

    Parameters
    ----------
    species_short : str or array-like
        Species code (only validated).
    nitrogen_intake : float or array-like
        Daily nitrogen intake (kg N/head/day).
    nitrogen_retention : float or array-like
        Daily nitrogen retention (kg N/head/day).

    Returns
    -------
    float or numpy.ndarray
        Daily nitrogen excretion (kg N/head/day).
    """
    validate_nitrogen_excretion_inputs(species_short, nitrogen_intake, nitrogen_retention)
    _, ni, nr = _vec(as_str(species_short), as_float(nitrogen_intake), as_float(nitrogen_retention))
    with np.errstate(invalid="ignore", over="ignore"):
        nitrogen_excretion = ni - nr
    return finalize(nitrogen_excretion, all_scalar(species_short, nitrogen_intake, nitrogen_retention))
