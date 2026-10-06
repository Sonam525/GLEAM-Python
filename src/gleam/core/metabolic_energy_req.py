"""Metabolic energy requirements and dry matter intake (IPCC Tier 2).

Port of ``R/core_model_metabolic_energy_req.R``. The R functions are called
row by row with scalars; here every function is vectorised: species / cohort
``if`` branches become element-wise masks (``np.select``) and the coefficients
and arithmetic order of each branch are kept identical to R.

Energy requirements are expressed as net energy (NE) for ruminants (CTL, BFL,
SHP, GTS) and as metabolizable energy (ME) for camels (CML), pigs (PGS) and
chickens (CHK), in MJ/head/day.

Departures from R that only affect invalid inputs:

* where R stops because no branch defines the result (unknown species or
  cohort, ``NA`` code), the vectorised result is ``nan`` for that element;
* where R stops with "missing value where TRUE/FALSE needed" because of a
  missing number in a branching condition (CHK maintenance temperature test,
  SHP/GTS pregnancy litter-size test), a :class:`ValueError` with the same
  message is raised.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .. import constants as K
from .._utils import all_scalar, as_float, as_str, broadcast, finalize, is_true, normalize_rate
from ..validation.metabolic_energy_req_core import (
    validate_activity_inputs,
    validate_dmi_inputs,
    validate_egg_inputs,
    validate_fibre_inputs,
    validate_growth_inputs,
    validate_lactation_inputs,
    validate_maintenance_inputs,
    validate_pregnancy_inputs,
    validate_reg_inputs,
    validate_rem_inputs,
    validate_total_energy_inputs,
    validate_work_inputs,
)

__all__ = [
    "calc_metabolic_energy_req_maintenance",
    "calc_metabolic_energy_req_activity",
    "calc_metabolic_energy_req_growth",
    "calc_metabolic_energy_req_lactation",
    "calc_metabolic_energy_req_eggs",
    "calc_metabolic_energy_req_work",
    "calc_metabolic_energy_req_fibre",
    "calc_metabolic_energy_req_pregnancy",
    "calc_rem_maintenance",
    "calc_reg_growth",
    "calc_total_metabolic_energy_req",
    "calc_ration_intake",
]

_GROWING = ("FS", "FJ", "MS", "MJ", "FN", "MN")
_NAN = np.nan


def _in(x: np.ndarray, values: Iterable[str]) -> np.ndarray:
    """Element-wise ``x %in% values`` for string-code arrays (``None`` never matches)."""
    out = np.zeros(np.shape(x), dtype=bool)
    for v in values:
        out = out | (x == v)
    return out


def _r_if_na(cond_is_na: np.ndarray) -> None:
    """R's ``if (NA)`` error for a branching condition evaluated on a missing number."""
    if np.any(cond_is_na):
        raise ValueError("missing value where TRUE/FALSE needed")


# --------------------------------------------------------------------------
# Maintenance
# --------------------------------------------------------------------------


def calc_metabolic_energy_req_maintenance(
    species_short: Any,
    cohort_short: Any,
    live_weight_cohort_average: Any,
    lactating_females_fraction: Any = np.nan,
    offtake_rate: Any = np.nan,
    age_first_parturition: Any = np.nan,
    average_annual_temperature: Any = np.nan,
    lower_critical_temperature: Any = np.nan,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> Any:
    """Energy requirement for maintenance (MJ/head/day; NE for ruminants, ME otherwise).

    IPCC Tier 2 (Eq. 10.3): ``maintenance = cmain * live_weight_cohort_average ** 0.75``.

    Parameters
    ----------
    species_short, cohort_short : str or array
        Species code (CTL, BFL, SHP, GTS, PGS, CML, CHK) and cohort code
        (FJ, FS, FA, MJ, MS, MA, FN, MN).
    live_weight_cohort_average : float or array
        Average live weight over the cohort stage (kg).
    lactating_females_fraction : float or array
        Fraction of adult females lactating (CTL / BFL ``FA``).
    offtake_rate : float or array
        Annual offtake fraction (clamped to [0, 1]); weights intact vs
        castrated males (CTL / BFL ``MA``/``MS``; SHP males).
    age_first_parturition : float or array
        Age at first parturition (days), SHP ``FS`` / ``MS``.
    average_annual_temperature, lower_critical_temperature : float or array
        Ambient and lower critical temperature (degrees C), CHK only.
    nondemo_productive_phase_id : float or array
        Non-demographic phase id (only validated, CHK ``FN``).
    is_egg_producing : bool or array
        CHK laying flag (``FN`` laying hens use the adult equation).

    Returns
    -------
    float or numpy.ndarray

    Notes
    -----
    ``cmain`` (MJ/day/kg^0.75):

    * CTL/BFL: FA ``0.386 * lff + 0.322 * (1 - lff)``; FS, FJ, MJ, FN, MN
      ``0.322``; MA, MS ``0.322 * otr + 0.37 * (1 - otr)`` (NRC 1996; AFRC 1993).
    * CML ``0.435`` (Wardeh 2004); GTS ``0.315``; PGS ``0.4435`` (NRC 1998).
    * SHP (AFRC 1993): FA ``0.217``; FJ, FN, MN ``0.236``; FS
      ``0.236 * 365/afp + 0.217 * (afp - 365)/afp``; MA
      ``0.217 * otr + 0.217 * 1.15 * (1 - otr)``; MJ same with ``0.236``; MS
      the MA term weighted by ``(afp - 365)/afp`` plus the MJ term weighted by
      ``365/afp``.
    * CHK (Sakomura 2004): MA, FA and laying FN:
      ``max(0, lw ** 0.75 * (0.6935 - 0.0099 * T))``; other cohorts:
      ``0.3866 + 0.0282 * (LCT - T)`` if ``T < LCT`` else
      ``0.3866 + 0.0037 * (LCT - T)``.
    """
    validate_maintenance_inputs(
        species_short, cohort_short, live_weight_cohort_average,
        lactating_females_fraction, offtake_rate, age_first_parturition,
        average_annual_temperature, lower_critical_temperature, nondemo_productive_phase_id,
        is_egg_producing,
    )
    scalar = all_scalar(
        species_short, cohort_short, live_weight_cohort_average, lactating_females_fraction,
        offtake_rate, age_first_parturition, average_annual_temperature,
        lower_critical_temperature, nondemo_productive_phase_id, is_egg_producing,
    )
    sp, co, lw, lff, otr, afp, temp, lct, egg = broadcast(
        as_str(species_short), as_str(cohort_short), as_float(live_weight_cohort_average),
        as_float(lactating_females_fraction), as_float(offtake_rate),
        as_float(age_first_parturition), as_float(average_annual_temperature),
        as_float(lower_critical_temperature), is_true(is_egg_producing),
    )

    with np.errstate(all="ignore"):
        # Normalize offtake_rate if it is available (NA stays NA).
        otr = normalize_rate(otr)
        fa, fs = co == "FA", co == "FS"
        ma, ms, mj = co == "MA", co == "MS", co == "MJ"

        cattle = _in(sp, ("CTL", "BFL"))
        sheep = sp == "SHP"
        cmain = np.select(
            [
                cattle & fa,
                cattle & _in(co, ("FS", "FJ", "MJ", "FN", "MN")),
                cattle & _in(co, ("MA", "MS")),
                sp == "CML",
                sp == "GTS",
                sheep & fa,
                sheep & fs,
                sheep & _in(co, ("FJ", "FN", "MN")),
                sheep & ma,
                sheep & ms,
                sheep & mj,
                sp == "PGS",
            ],
            [
                0.386 * lff + 0.322 * (1 - lff),
                0.322,
                0.322 * otr + 0.37 * (1 - otr),
                0.435,
                0.315,
                0.217,
                (0.236 * (365 / afp)) + (0.217 * ((afp - 365) / afp)),
                0.236,
                0.217 * otr + 0.217 * 1.15 * (1 - otr),
                (0.217 * otr + 0.217 * 1.15 * (1 - otr)) * ((afp - 365) / afp)
                + (0.236 * otr + 0.236 * 1.15 * (1 - otr)) * (365 / afp),
                0.236 * otr + 0.236 * 1.15 * (1 - otr),
                0.4435,
            ],
            default=_NAN,
        )
        out = (lw ** 0.75) * cmain

        # Chickens: adult / laying equation or juvenile temperature equation.
        chk = sp == "CHK"
        chk_adult = chk & (_in(co, ("MA", "FA")) | ((co == "FN") & egg))
        chk_young = chk & ~chk_adult
        _r_if_na(chk_young & (np.isnan(temp) | np.isnan(lct)))
        adult_value = np.maximum(0, (lw ** 0.75) * (0.6935 - 0.0099 * temp))
        young_value = np.where(
            temp < lct,
            0.3866 + 0.0282 * (lct - temp),
            0.3866 + 0.0037 * (lct - temp),
        )
        out = np.select([chk_adult, chk_young], [adult_value, young_value], default=out)
    return finalize(out, scalar)


# --------------------------------------------------------------------------
# Activity
# --------------------------------------------------------------------------


def calc_metabolic_energy_req_activity(
    species_short: Any,
    cohort_short: Any,
    metabolic_energy_req_maintenance: Any,
    live_weight_cohort_average: Any,
    low_activity_fraction: Any,
    high_activity_fraction: Any,
) -> Any:
    """Energy requirement for activity (MJ/head/day).

    IPCC Tier 2 (Eq. 10.4): ``activity = cact * maintenance`` (CTL, BFL, CML,
    PGS) or ``activity = cact * live_weight_cohort_average`` (SHP, GTS), with
    ``cact`` weighted by the low / high activity fractions:

    * CTL, BFL: ``0.17 * low + 0.36 * high``; CML: ``0.1 * (low + high)``;
      PGS: ``0.125 * (low + high)``;
    * SHP: ``0.0107 * low + 0.024 * high``; GTS: ``0.019 * low + 0.024 * high``;
    * CHK: ``maintenance * (low + high) * 0.25`` (Sakomura 2004).

    Parameters
    ----------
    species_short, cohort_short : str or array
    metabolic_energy_req_maintenance : float or array
        Maintenance requirement (MJ/head/day).
    live_weight_cohort_average : float or array
        Average cohort live weight (kg).
    low_activity_fraction, high_activity_fraction : float or array
        Fractions of time with low / high activity (sum in [0, 1]).
    """
    validate_activity_inputs(
        species_short, cohort_short, metabolic_energy_req_maintenance,
        live_weight_cohort_average, low_activity_fraction, high_activity_fraction,
    )
    scalar = all_scalar(
        species_short, cohort_short, metabolic_energy_req_maintenance,
        live_weight_cohort_average, low_activity_fraction, high_activity_fraction,
    )
    sp, _co, maint, lw, low, high = broadcast(
        as_str(species_short), as_str(cohort_short), as_float(metabolic_energy_req_maintenance),
        as_float(live_weight_cohort_average), as_float(low_activity_fraction),
        as_float(high_activity_fraction),
    )
    with np.errstate(all="ignore"):
        out = np.select(
            [_in(sp, ("CTL", "BFL")), sp == "CML", sp == "SHP", sp == "GTS", sp == "PGS", sp == "CHK"],
            [
                ((0.17 * low) + (0.36 * high)) * maint,
                (0.1 * (low + high)) * maint,
                ((0.0107 * low) + (0.024 * high)) * lw,
                ((0.019 * low) + (0.024 * high)) * lw,
                (0.125 * (low + high)) * maint,
                maint * (low + high) * 0.25,
            ],
            default=_NAN,
        )
    return finalize(out, scalar)


# --------------------------------------------------------------------------
# Growth
# --------------------------------------------------------------------------


def calc_metabolic_energy_req_growth(
    species_short: Any,
    cohort_short: Any,
    live_weight_cohort_average: Any = np.nan,
    live_weight_cohort_final: Any = np.nan,
    live_weight_cohort_initial: Any = np.nan,
    live_weight_mature_stage: Any = np.nan,
    daily_weight_gain: Any = np.nan,
    offtake_rate: Any = np.nan,
    cohort_duration_days: Any = np.nan,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> Any:
    """Energy requirement for growth (MJ/head/day); 0 for adult cohorts.

    * CTL, BFL (NRC 1996; IPCC Eq. 10.6):
      ``22.02 * (lw_avg / (cgro * lw_mature)) ** 0.75 * dwg ** 1.097`` with
      ``cgro = 0.8`` (FS, FJ, FN), ``1.2 * (1 - otr) + 1 * otr`` (MS, MJ),
      ``1`` (MN).
    * SHP, GTS (AFRC 1993; IPCC Eq. 10.7):
      ``(lw_final - lw_initial) * (a + 0.5 * b * (lw_initial + lw_final)) / duration``
      with SHP ``a, b = 2.1, 0.45`` (FS, FJ, FN), ``4.4 * otr + 2.5 * (1 - otr),
      0.32 * otr + 0.35 * (1 - otr)`` (MS, MJ), ``4.4, 0.32`` (MN), GTS
      ``5, 0.33`` (growing cohorts) and ``0, 0`` for adults.
    * CML (Al-Jassim 2019): ``41.2 * dwg``.
    * PGS (NRC 1998): ``dwg * (0.65 * 0.23 * 54 + 0.35 * 0.9 * 52.3)`` (ME).
    * CHK (Sakomura 2004): ``dwg * cgro * 1000`` with ``cgro = 0.0279`` MJ/g
      (MA, FA, laying FN) or ``0.0202`` (other cohorts).

    Parameters
    ----------
    species_short, cohort_short : str or array
    live_weight_cohort_average, live_weight_cohort_final, live_weight_cohort_initial,
    live_weight_mature_stage : float or array
        Cohort live weights (kg).
    daily_weight_gain : float or array
        Live weight gain (kg/head/day).
    offtake_rate : float or array
        Offtake fraction (clamped to [0, 1]); offtaken males assumed castrated.
    cohort_duration_days : float or array
        Time spent in the cohort (days).
    nondemo_productive_phase_id : float or array
    is_egg_producing : bool or array
        CHK laying flag.
    """
    validate_growth_inputs(
        species_short, cohort_short, live_weight_cohort_average, live_weight_cohort_final,
        live_weight_cohort_initial, live_weight_mature_stage, daily_weight_gain, offtake_rate,
        cohort_duration_days, nondemo_productive_phase_id, is_egg_producing,
    )
    scalar = all_scalar(
        species_short, cohort_short, live_weight_cohort_average, live_weight_cohort_final,
        live_weight_cohort_initial, live_weight_mature_stage, daily_weight_gain, offtake_rate,
        cohort_duration_days, nondemo_productive_phase_id, is_egg_producing,
    )
    sp, co, lw_avg, lw_fin, lw_ini, lw_mat, dwg, otr, dur, egg = broadcast(
        as_str(species_short), as_str(cohort_short), as_float(live_weight_cohort_average),
        as_float(live_weight_cohort_final), as_float(live_weight_cohort_initial),
        as_float(live_weight_mature_stage), as_float(daily_weight_gain), as_float(offtake_rate),
        as_float(cohort_duration_days), is_true(is_egg_producing),
    )

    with np.errstate(all="ignore"):
        otr = normalize_rate(otr)
        growing = _in(co, _GROWING)
        adult = _in(co, ("FA", "MA"))
        fem_growing = _in(co, ("FS", "FJ", "FN"))
        male_growing = _in(co, ("MS", "MJ"))

        # Cattle and buffalo
        cgro = np.select([fem_growing, male_growing, co == "MN"], [0.8, 1.2 * (1 - otr) + 1 * otr, 1.0], default=_NAN)
        cattle = np.where(
            growing,
            22.02 * ((lw_avg / (cgro * lw_mat)) ** 0.75) * (dwg ** 1.097),
            0.0,
        )

        # Camels
        camel = np.where(growing, 41.2 * dwg, 0.0)

        # Sheep and goats: linear formula with coefficients a, b
        a_shp = np.select(
            [fem_growing, male_growing, co == "MN", adult],
            [2.1, 4.4 * otr + 2.5 * (1 - otr), 4.4, 0.0],
            default=_NAN,
        )
        b_shp = np.select(
            [fem_growing, male_growing, co == "MN", adult],
            [0.45, 0.32 * otr + 0.35 * (1 - otr), 0.32, 0.0],
            default=_NAN,
        )
        a_gts = np.select([growing, adult], [5.0, 0.0], default=_NAN)
        b_gts = np.select([growing, adult], [0.33, 0.0], default=_NAN)
        sheep = ((lw_fin - lw_ini) * (a_shp + 0.5 * b_shp * (lw_ini + lw_fin))) / dur
        goat = ((lw_fin - lw_ini) * (a_gts + 0.5 * b_gts * (lw_ini + lw_fin))) / dur

        # Pigs: protein and fat tissue deposition
        prot_tissue_frac = 0.65
        cgro_pgs = (prot_tissue_frac * 0.23 * 54) + ((1 - prot_tissue_frac) * 0.9 * 52.3)
        pig = np.where(growing, dwg * cgro_pgs, 0.0)

        # Chickens
        fn = co == "FN"
        cgro_chk = np.select(
            [_in(co, ("MA", "FA")) | (fn & egg), _in(co, ("FS", "FJ", "MS", "MJ", "MN")) | (fn & ~egg)],
            [0.0279, 0.0202],
            default=0.0,
        )
        chicken = dwg * cgro_chk * 1000

        out = np.select(
            [_in(sp, ("CTL", "BFL")), sp == "CML", sp == "SHP", sp == "GTS", sp == "PGS", sp == "CHK"],
            [cattle, camel, sheep, goat, pig, chicken],
            default=0.0,
        )
    return finalize(out, scalar)


# --------------------------------------------------------------------------
# Lactation
# --------------------------------------------------------------------------


def calc_metabolic_energy_req_lactation(
    species_short: Any,
    cohort_short: Any,
    lactating_females_fraction: Any = np.nan,
    milk_yield_day: Any = np.nan,
    milk_fat_fraction: Any = np.nan,
    non_productive_duration: Any = np.nan,
    pregnancy_duration: Any = np.nan,
    litter_size: Any = np.nan,
    death_rate_juvenile: Any = np.nan,
    live_weight_at_birth: Any = np.nan,
    live_weight_at_weaning: Any = np.nan,
    lactation_duration: Any = np.nan,
    parturition_rate: Any = np.nan,
) -> Any:
    """Energy requirement for lactation (MJ/head/day); computed for cohort ``FA`` only.

    Milk for offtake plus milk for offspring (5 kg milk per kg gain to weaning):

    * CTL, BFL: ``(myd * lff + pr * 5 * (w - b) / 365) * (mff * 100 * 0.40 + 1.47)``
      (NRC 1989; IPCC Eq. 10.8);
    * CML: same milk term ``* 4.063`` (Wardeh 2004);
    * SHP / GTS: ``(myd * lff + ls * pr * 5 * (w - b) / 365) * 4.6`` / ``* 3``
      (AFRC 1993 / 1998);
    * PGS (NRC 1998): ``ls * (1 - 0.5 * drj) * (0.02059 * (w - b) * 1000 / ld
      - 0.3766 / 0.67) * cadj`` with ``cadj = ld / (npd + pd + ld)``;
    * CHK: 0.

    Parameters
    ----------
    species_short, cohort_short : str or array
    lactating_females_fraction, milk_yield_day, milk_fat_fraction : float or array
        Fraction lactating, milk yield (kg/head/day), milk fat fraction.
    non_productive_duration, pregnancy_duration, lactation_duration : float or array
        Reproductive cycle durations (days), PGS.
    litter_size, death_rate_juvenile : float or array
    live_weight_at_birth, live_weight_at_weaning : float or array
        Weights (kg).
    parturition_rate : float or array
        Parturitions per adult female per year.
    """
    validate_lactation_inputs(
        species_short, cohort_short, lactating_females_fraction, milk_yield_day, milk_fat_fraction,
        non_productive_duration, pregnancy_duration, litter_size, death_rate_juvenile,
        live_weight_at_birth, live_weight_at_weaning, lactation_duration, parturition_rate,
    )
    scalar = all_scalar(
        species_short, cohort_short, lactating_females_fraction, milk_yield_day, milk_fat_fraction,
        non_productive_duration, pregnancy_duration, litter_size, death_rate_juvenile,
        live_weight_at_birth, live_weight_at_weaning, lactation_duration, parturition_rate,
    )
    sp, co, lff, myd, mff, npd, pd_, ls, drj, lwb, lww, ld, pr = broadcast(
        as_str(species_short), as_str(cohort_short), as_float(lactating_females_fraction),
        as_float(milk_yield_day), as_float(milk_fat_fraction), as_float(non_productive_duration),
        as_float(pregnancy_duration), as_float(litter_size), as_float(death_rate_juvenile),
        as_float(live_weight_at_birth), as_float(live_weight_at_weaning),
        as_float(lactation_duration), as_float(parturition_rate),
    )
    with np.errstate(all="ignore"):
        fa = co == "FA"
        cattle = ((myd * lff) + (pr * 5 * (lww - lwb) / 365)) * (mff * 100 * 0.40 + 1.47)
        camel = ((myd * lff) + (pr * 5 * (lww - lwb) / 365)) * 4.063
        sheep = ((myd * lff) + (ls * pr * 5 * (lww - lwb) / 365)) * 4.6
        goat = ((myd * lff) + (ls * pr * 5 * (lww - lwb) / 365)) * 3
        cadj = ld / (npd + pd_ + ld)
        pig = ls * (1 - 0.5 * drj) * ((0.02059 * (lww - lwb) * 1000 / ld) - (0.3766 / 0.67)) * cadj

        is_cattle = _in(sp, ("CTL", "BFL"))
        known = _in(sp, ("CTL", "BFL", "CML", "SHP", "GTS", "PGS", "CHK"))
        out = np.select(
            [
                is_cattle & fa,
                (sp == "CML") & fa,
                (sp == "SHP") & fa,
                (sp == "GTS") & fa,
                (sp == "PGS") & fa,
                known,
            ],
            [cattle, camel, sheep, goat, pig, 0.0],
            default=_NAN,
        )
    return finalize(out, scalar)


# --------------------------------------------------------------------------
# Eggs
# --------------------------------------------------------------------------


def calc_metabolic_energy_req_eggs(
    species_short: Any,
    cohort_short: Any,
    cohort_stock_size: Any = np.nan,
    egg_output_human_consumption: Any = np.nan,
    egg_average_weight: Any = np.nan,
    parturition_rate: Any = np.nan,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> Any:
    """ME required for egg deposition (MJ/head/day); CHK ``FA`` and laying ``FN`` only.

    ``egg_mass = (egg_output_human_consumption / 365 / cohort_stock_size +
    parturition_rate / 365) * egg_average_weight`` (kg/head/day) and
    ``eggs = egg_mass * 10.04`` (10.04 MJ/kg egg, Sakomura 2004). Note that, as
    in R, CHK ``FA`` gets egg energy whatever its ``is_egg_producing`` flag.

    Parameters
    ----------
    species_short, cohort_short : str or array
    cohort_stock_size : float or array
        Average cohort population (heads).
    egg_output_human_consumption : float or array
        Eggs produced for human consumption per year by the flock (eggs/year).
    egg_average_weight : float or array
        Average egg weight (kg/egg).
    parturition_rate : float or array
        Eggs laid for reproduction per hen per year.
    nondemo_productive_phase_id : float or array
    is_egg_producing : bool or array
    """
    validate_egg_inputs(
        species_short, cohort_short, cohort_stock_size, egg_output_human_consumption,
        egg_average_weight, parturition_rate, nondemo_productive_phase_id, is_egg_producing,
    )
    scalar = all_scalar(
        species_short, cohort_short, cohort_stock_size, egg_output_human_consumption,
        egg_average_weight, parturition_rate, nondemo_productive_phase_id, is_egg_producing,
    )
    sp, co, css, eoh, eaw, pr, egg = broadcast(
        as_str(species_short), as_str(cohort_short), as_float(cohort_stock_size),
        as_float(egg_output_human_consumption), as_float(egg_average_weight),
        as_float(parturition_rate), is_true(is_egg_producing),
    )
    with np.errstate(all="ignore"):
        cegg = 10.04
        human_consumption_eggs_head_day = eoh / 365 / css
        reproductive_eggs_head_day = pr / 365
        egg_mass_production = (human_consumption_eggs_head_day + reproductive_eggs_head_day) * eaw
        value = egg_mass_production * cegg
        laying = (sp == "CHK") & ((co == "FA") | ((co == "FN") & egg))
        out = np.where(laying, value, 0.0)
    return finalize(out, scalar)


# --------------------------------------------------------------------------
# Work
# --------------------------------------------------------------------------


def calc_metabolic_energy_req_work(
    species_short: Any,
    cohort_short: Any,
    metabolic_energy_req_maintenance: Any = np.nan,
    draught_work_hours_female: Any = np.nan,
    draught_work_hours_male: Any = np.nan,
    draught_fraction_female: Any = np.nan,
    draught_fraction_male: Any = np.nan,
) -> Any:
    """Energy requirement for draught work (MJ/head/day); adults of CTL, BFL, CML only.

    * CTL, BFL (IPCC Eq. 10.11): ``0.1 * maintenance * hours * fraction``;
    * CML (Wilson 1989): ``4 * hours * fraction`` (ME, 4 MJ/hour);

    using the male (``MA``) or female (``FA``) hours and fraction. 0 otherwise.
    """
    validate_work_inputs(
        species_short, cohort_short, metabolic_energy_req_maintenance, draught_work_hours_female,
        draught_work_hours_male, draught_fraction_female, draught_fraction_male,
    )
    scalar = all_scalar(
        species_short, cohort_short, metabolic_energy_req_maintenance, draught_work_hours_female,
        draught_work_hours_male, draught_fraction_female, draught_fraction_male,
    )
    sp, co, maint, hours_f, hours_m, frac_f, frac_m = broadcast(
        as_str(species_short), as_str(cohort_short), as_float(metabolic_energy_req_maintenance),
        as_float(draught_work_hours_female), as_float(draught_work_hours_male),
        as_float(draught_fraction_female), as_float(draught_fraction_male),
    )
    with np.errstate(all="ignore"):
        cattle = _in(sp, ("CTL", "BFL"))
        camel = sp == "CML"
        ma, fa = co == "MA", co == "FA"
        out = np.select(
            [
                cattle & ma, cattle & fa, cattle,
                camel & ma, camel & fa, camel,
                _in(sp, ("SHP", "GTS", "PGS", "CHK")),
            ],
            [
                0.1 * maint * hours_m * frac_m, 0.1 * maint * hours_f * frac_f, 0.0,
                4 * hours_m * frac_m, 4 * hours_f * frac_f, 0.0,
                0.0,
            ],
            default=_NAN,
        )
    return finalize(out, scalar)


# --------------------------------------------------------------------------
# Fibre
# --------------------------------------------------------------------------


def calc_metabolic_energy_req_fibre(
    species_short: Any,
    cohort_short: Any,
    fibre_yield_year: Any = np.nan,
) -> Any:
    """Energy requirement for fibre production (MJ/head/day).

    Cohorts FA, FS, MA, MS, FN, MN of fibre species (IPCC Eq. 10.12):
    SHP, GTS ``24 * fibre_yield_year / 365``; CML
    ``(24 / 0.43) * (fibre_yield_year / 365)`` (ME, AFRC 1998; Cannas et al. 2007).
    0 for juveniles and other species.

    Parameters
    ----------
    species_short, cohort_short : str or array
    fibre_yield_year : float or array
        Annual fibre yield (kg/head/year).
    """
    validate_fibre_inputs(species_short, cohort_short, fibre_yield_year)
    scalar = all_scalar(species_short, cohort_short, fibre_yield_year)
    sp, co, fyy = broadcast(as_str(species_short), as_str(cohort_short), as_float(fibre_yield_year))
    with np.errstate(all="ignore"):
        producing = _in(co, ("FA", "FS", "MA", "MS", "FN", "MN"))
        small_ruminant = _in(sp, ("GTS", "SHP"))
        camel = sp == "CML"
        out = np.select(
            [
                small_ruminant & producing, small_ruminant,
                camel & producing, camel,
                _in(sp, ("CTL", "BFL", "PGS", "CHK")),
            ],
            [24 * fyy / 365, 0.0, (24 / 0.43) * (fyy / 365), 0.0, 0.0],
            default=_NAN,
        )
    return finalize(out, scalar)


# --------------------------------------------------------------------------
# Pregnancy
# --------------------------------------------------------------------------


def calc_metabolic_energy_req_pregnancy(
    species_short: Any,
    cohort_short: Any,
    metabolic_energy_req_maintenance: Any = np.nan,
    parturition_rate: Any = np.nan,
    litter_size: Any = np.nan,
    pregnancy_duration: Any = np.nan,
    non_productive_duration: Any = np.nan,
    lactation_duration: Any = np.nan,
    cohort_duration_days: Any = np.nan,
    offtake_rate: Any = np.nan,
) -> Any:
    """Energy requirement for pregnancy (MJ/head/day); female cohorts FA and FS only.

    IPCC Eq. 10.13. ``FA`` is scaled by the parturition rate and the gestation
    share of the year (or of the reproductive cycle for PGS); ``FS`` by
    ``pregnancy_duration / cohort_duration_days * (1 - offtake_rate)``:

    * CTL, BFL: FA ``maintenance * 0.1 * pr * pd / 365``;
      FS ``(maintenance * 0.1) * (pd / duration) * (1 - otr)``;
    * CML (Wardeh 2004): FA ``maintenance * 0.12 * pr``;
      FS ``maintenance * 0.12 * (pd / duration) * (1 - otr)``;
    * SHP, GTS: FA ``maintenance * cpreg * pr * pd / 365`` with
      ``cpreg = 0.077 * (2 - ls) + 0.126 * (ls - 1)`` for ``1 <= ls <= 2``,
      ``0.150`` for ``ls > 2`` (0 otherwise); FS
      ``maintenance * 0.077 * (pd / duration) * (1 - otr)``;
    * PGS (NRC 1998, ``cgest = 0.14985`` MJ/piglet): FA
      ``cgest * ls * pd / (npd + pd + ld)``; FS
      ``cgest * ls * (pd / duration) * (1 - otr)``;
    * CHK: 0.

    ``offtake_rate`` is clamped to [0, 1] before validation (as in R).
    """
    # Normalize offtake_rate if it is available (before validation, as in R).
    otr_n = normalize_rate(as_float(offtake_rate))
    validate_pregnancy_inputs(
        species_short, cohort_short, metabolic_energy_req_maintenance, parturition_rate,
        litter_size, pregnancy_duration, non_productive_duration, lactation_duration,
        cohort_duration_days, otr_n,
    )
    scalar = all_scalar(
        species_short, cohort_short, metabolic_energy_req_maintenance, parturition_rate,
        litter_size, pregnancy_duration, non_productive_duration, lactation_duration,
        cohort_duration_days, offtake_rate,
    )
    sp, co, maint, pr, ls, pd_, npd, ld, dur, otr = broadcast(
        as_str(species_short), as_str(cohort_short), as_float(metabolic_energy_req_maintenance),
        as_float(parturition_rate), as_float(litter_size), as_float(pregnancy_duration),
        as_float(non_productive_duration), as_float(lactation_duration),
        as_float(cohort_duration_days), otr_n,
    )
    with np.errstate(all="ignore"):
        fa, fs = co == "FA", co == "FS"
        cattle = _in(sp, ("CTL", "BFL"))
        camel = sp == "CML"
        small_ruminant = _in(sp, ("SHP", "GTS"))
        pig = sp == "PGS"

        # Litter size effect (SHP, GTS adult females)
        _r_if_na(small_ruminant & fa & np.isnan(ls))
        cpreg = np.select(
            [(ls >= 1) & (ls <= 2), ls > 2],
            [0.077 * (2 - ls) + 0.126 * (ls - 1), 0.150],
            default=0.0,
        )
        cgest = 0.14985

        out = np.select(
            [
                cattle & fa, cattle & fs, cattle,
                camel & fa, camel & fs, camel,
                small_ruminant & fa, small_ruminant & fs, small_ruminant,
                pig & fa, pig & fs, pig,
                sp == "CHK",
            ],
            [
                (maint * 0.1 * pr * pd_ / 365),
                (maint * 0.1) * (pd_ / dur) * (1 - otr),
                0.0,
                maint * 0.12 * pr,
                maint * 0.12 * (pd_ / dur) * (1 - otr),
                0.0,
                maint * cpreg * pr * pd_ / 365,
                maint * 0.077 * (pd_ / dur) * (1 - otr),
                0.0,
                cgest * ls * pd_ / (npd + pd_ + ld),
                cgest * ls * (pd_ / dur) * (1 - otr),
                0.0,
                0.0,
            ],
            default=_NAN,
        )
    return finalize(out, scalar)


# --------------------------------------------------------------------------
# REM / REG
# --------------------------------------------------------------------------


def calc_rem_maintenance(species_short: Any, ration_digestibility_fraction: Any = np.nan) -> Any:
    """Ratio of net energy for maintenance to digestible energy (REM, fraction).

    Ruminants only (IPCC Eq. 10.14, Gibbs & Johnson 1993), with ``DE = 100 * digestibility``:
    ``1.123 - 0.004092 * DE + 0.00001126 * DE ** 2 - 25.4 / DE``. ``nan`` for PGS, CML, CHK.
    """
    validate_rem_inputs(species_short, ration_digestibility_fraction)
    scalar = all_scalar(species_short, ration_digestibility_fraction)
    sp, rdf = broadcast(as_str(species_short), as_float(ration_digestibility_fraction))
    with np.errstate(all="ignore"):
        rem = (
            1.123 - (0.004092 * (rdf * 100)) + (0.00001126 * (rdf * 100) ** 2) - (25.4 / (rdf * 100))
        )
        out = np.where(_in(sp, K.GLEAM_SPECIES_RUMINANTS), rem, _NAN)
    return finalize(out, scalar)


def calc_reg_growth(species_short: Any, ration_digestibility_fraction: Any = np.nan) -> Any:
    """Ratio of net energy for growth to digestible energy (REG, fraction).

    Ruminants only (IPCC Eq. 10.15), with ``DE = 100 * digestibility``:
    ``1.164 - 0.005160 * DE + 0.00001308 * DE ** 2 - 37.4 / DE``. ``nan`` for PGS, CML, CHK.
    """
    validate_reg_inputs(species_short, ration_digestibility_fraction)
    scalar = all_scalar(species_short, ration_digestibility_fraction)
    sp, rdf = broadcast(as_str(species_short), as_float(ration_digestibility_fraction))
    with np.errstate(all="ignore"):
        reg = (
            1.164 - (0.005160 * (rdf * 100)) + (0.00001308 * (rdf * 100) ** 2) - (37.4 / (rdf * 100))
        )
        out = np.where(_in(sp, K.GLEAM_SPECIES_RUMINANTS), reg, _NAN)
    return finalize(out, scalar)


# --------------------------------------------------------------------------
# Total requirement and dry matter intake
# --------------------------------------------------------------------------


def calc_total_metabolic_energy_req(
    species_short: Any,
    metabolic_energy_req_maintenance: Any,
    metabolic_energy_req_activity: Any,
    metabolic_energy_req_lactation: Any,
    metabolic_energy_req_work: Any,
    metabolic_energy_req_pregnancy: Any,
    net_energy_maintenance_digestible_energy_ratio: Any,
    metabolic_energy_req_growth: Any,
    metabolic_energy_req_fibre_production: Any,
    metabolic_energy_req_egg_deposition: Any,
    net_energy_growth_digestible_energy_ratio: Any,
    ration_digestibility_fraction: Any,
) -> Any:
    """Total daily energy requirement (MJ/head/day).

    Gross energy for ruminants (IPCC Eq. 10.16):

    * CTL, BFL: ``((m + a + l + w + p) / REM + g / REG) / DE``;
    * SHP, GTS: ``((m + a + l + p) / REM + (g + f) / REG) / DE``;

    summed metabolizable energy otherwise: CML ``m + a + l + w + f + p + g``,
    PGS ``m + a + l + p + g``, CHK ``m + a + g + eggs``.
    """
    validate_total_energy_inputs(
        species_short, metabolic_energy_req_maintenance, metabolic_energy_req_activity,
        metabolic_energy_req_lactation, metabolic_energy_req_work, metabolic_energy_req_pregnancy,
        net_energy_maintenance_digestible_energy_ratio, metabolic_energy_req_growth,
        metabolic_energy_req_fibre_production, metabolic_energy_req_egg_deposition,
        net_energy_growth_digestible_energy_ratio, ration_digestibility_fraction,
    )
    scalar = all_scalar(
        species_short, metabolic_energy_req_maintenance, metabolic_energy_req_activity,
        metabolic_energy_req_lactation, metabolic_energy_req_work, metabolic_energy_req_pregnancy,
        net_energy_maintenance_digestible_energy_ratio, metabolic_energy_req_growth,
        metabolic_energy_req_fibre_production, metabolic_energy_req_egg_deposition,
        net_energy_growth_digestible_energy_ratio, ration_digestibility_fraction,
    )
    sp, m, a, l_, w, p, rem, g, f, e, reg, de = broadcast(
        as_str(species_short), as_float(metabolic_energy_req_maintenance),
        as_float(metabolic_energy_req_activity), as_float(metabolic_energy_req_lactation),
        as_float(metabolic_energy_req_work), as_float(metabolic_energy_req_pregnancy),
        as_float(net_energy_maintenance_digestible_energy_ratio),
        as_float(metabolic_energy_req_growth), as_float(metabolic_energy_req_fibre_production),
        as_float(metabolic_energy_req_egg_deposition),
        as_float(net_energy_growth_digestible_energy_ratio),
        as_float(ration_digestibility_fraction),
    )
    with np.errstate(all="ignore"):
        out = np.select(
            [_in(sp, ("CTL", "BFL")), _in(sp, ("SHP", "GTS")), sp == "CML", sp == "PGS", sp == "CHK"],
            [
                (((m + a + l_ + w + p) / rem) + ((g) / reg)) / de,
                (((m + a + l_ + p) / rem) + ((g + f) / reg)) / de,
                m + a + l_ + w + f + p + g,
                m + a + l_ + p + g,
                m + a + g + e,
            ],
            default=_NAN,
        )
    return finalize(out, scalar)


def calc_ration_intake(
    species_short: Any,
    metabolic_energy_req_total: Any,
    ration_gross_energy: Any,
    ration_metabolizable_energy: Any,
) -> Any:
    """Daily dry matter intake (kg DM/head/day).

    ``total / ration_gross_energy`` for ruminants (GE basis) and
    ``total / ration_metabolizable_energy`` for PGS, CML and CHK (ME basis).
    """
    validate_dmi_inputs(species_short, metabolic_energy_req_total, ration_gross_energy, ration_metabolizable_energy)
    scalar = all_scalar(species_short, metabolic_energy_req_total, ration_gross_energy, ration_metabolizable_energy)
    sp, total, ge, me = broadcast(
        as_str(species_short), as_float(metabolic_energy_req_total), as_float(ration_gross_energy),
        as_float(ration_metabolizable_energy),
    )
    with np.errstate(all="ignore"):
        out = np.select(
            [_in(sp, K.GLEAM_SPECIES_RUMINANTS), _in(sp, ("PGS", "CML", "CHK"))],
            [total / ge, total / me],
            default=_NAN,
        )
    return finalize(out, scalar)
