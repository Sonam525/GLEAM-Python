"""Live weights by cohort (port of ``R/core_model_weights.R``).

All functions are vectorised: they accept scalars, lists, numpy arrays or
pandas Series (broadcast against each other) and reproduce, element by
element, the scalar cohort branches of the R code. All-scalar calls return
Python floats (or a dict of floats).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._utils import (
    all_scalar,
    as_float,
    broadcast,
    finalize,
    finalize_dict,
    normalize_rate,
)
from ..validation._shared import abort
from ..validation.weights_core import (
    _as_codes,
    _codes_in,
    validate_avg_weight_inputs,
    validate_cohort_weight_inputs,
    validate_daily_gain_inputs,
)


def calc_cohort_weights(
    species_short: Any = None,
    cohort_short: Any = None,
    live_weight_female_adult: Any = np.nan,
    live_weight_male_adult: Any = np.nan,
    live_weight_at_birth: Any = np.nan,
    live_weight_female_at_slaughter: Any = np.nan,
    live_weight_male_at_slaughter: Any = np.nan,
    live_weight_at_weaning: Any = np.nan,
    nondemo_productive_phase_id: Any = np.nan,
    live_weight_female_nondemographic_start: Any = np.nan,
    live_weight_female_nondemographic_end: Any = np.nan,
    live_weight_male_nondemographic_start: Any = np.nan,
    live_weight_male_nondemographic_end: Any = np.nan,
    phase1_nondemo_fem_duration_days: Any = np.nan,
    phase2_nondemo_fem_duration_days: Any = np.nan,
    phase1_nondemo_mal_duration_days: Any = np.nan,
    phase2_nondemo_mal_duration_days: Any = np.nan,
) -> dict[str, Any]:
    """Initial, potential final, slaughter and mature live weights by cohort.

    Weights are attributed by cohort:

    * juveniles (``FJ``, ``MJ``): initial = birth weight; potential final =
      slaughter = weaning weight; mature = adult weight of the cohort sex;
    * sub-adults (``FS``, ``MS``): initial = weaning weight; potential final =
      mature = adult weight of the cohort sex; slaughter = sub-adult slaughter
      weight of the cohort sex;
    * adults (``FA``, ``MA``): initial = potential final = slaughter = mature =
      adult weight of the cohort sex;
    * non-demographic cohorts (``FN``, ``MN``): linear gain
      ``(end - start) / (phase1 + phase2 duration)``; phase 1 runs from the
      start weight to ``start + gain * phase1``, phase 2 from there to
      ``+ gain * phase2``; slaughter = potential final; mature = end weight.

    For chickens (``species_short = "CHK"``) the weaning weight is set to the
    birth (hatch) weight, so juvenile cohorts do not gain weight.

    Parameters
    ----------
    species_short : str or array-like, optional
        Species code (``CTL``, ``BFL``, ``SHP``, ``GTS``, ``PGS``, ``CML``,
        ``CHK``); may be NA.
    cohort_short : str or array-like
        Cohort code (``FJ``, ``FS``, ``FA``, ``MJ``, ``MS``, ``MA``, ``FN``, ``MN``).
    live_weight_female_adult, live_weight_male_adult : float or array-like
        Adult live weights (kg).
    live_weight_at_birth, live_weight_at_weaning : float or array-like
        Live weight at birth and at weaning (kg).
    live_weight_female_at_slaughter, live_weight_male_at_slaughter : float or array-like
        Slaughter weights of sub-adult animals (kg).
    nondemo_productive_phase_id : float or array-like
        Productive phase (1 or 2) of non-demographic cohorts; NA otherwise.
    live_weight_female_nondemographic_start, live_weight_female_nondemographic_end : float or array-like
        Female live weight at the start / end of the non-demographic cycle (kg).
    live_weight_male_nondemographic_start, live_weight_male_nondemographic_end : float or array-like
        Male live weight at the start / end of the non-demographic cycle (kg).
    phase1_nondemo_fem_duration_days, phase2_nondemo_fem_duration_days : float or array-like
        Durations of productive phases 1 and 2 for ``FN`` (days).
    phase1_nondemo_mal_duration_days, phase2_nondemo_mal_duration_days : float or array-like
        Durations of productive phases 1 and 2 for ``MN`` (days).

    Returns
    -------
    dict
        ``live_weight_mature_stage``, ``live_weight_cohort_initial``,
        ``live_weight_cohort_potential_final`` and
        ``live_weight_cohort_at_slaughter`` (kg).
    """
    validate_cohort_weight_inputs(
        species_short,
        cohort_short,
        live_weight_female_adult, live_weight_male_adult,
        live_weight_at_birth,
        live_weight_female_at_slaughter, live_weight_male_at_slaughter,
        live_weight_at_weaning,
        nondemo_productive_phase_id,
        live_weight_female_nondemographic_start, live_weight_female_nondemographic_end,
        live_weight_male_nondemographic_start, live_weight_male_nondemographic_end,
        phase1_nondemo_fem_duration_days,
        phase2_nondemo_fem_duration_days,
        phase1_nondemo_mal_duration_days,
        phase2_nondemo_mal_duration_days,
    )
    raw = (
        species_short, cohort_short, live_weight_female_adult, live_weight_male_adult,
        live_weight_at_birth, live_weight_female_at_slaughter, live_weight_male_at_slaughter,
        live_weight_at_weaning, nondemo_productive_phase_id,
        live_weight_female_nondemographic_start, live_weight_female_nondemographic_end,
        live_weight_male_nondemographic_start, live_weight_male_nondemographic_end,
        phase1_nondemo_fem_duration_days, phase2_nondemo_fem_duration_days,
        phase1_nondemo_mal_duration_days, phase2_nondemo_mal_duration_days,
    )
    scalar = all_scalar(*raw)
    (
        sp, co, female_adult, male_adult, birth, female_slaughter, male_slaughter, weaning,
        phase, f_start, f_end, m_start, m_end, p1_f, p2_f, p1_m, p2_m,
    ) = (
        np.atleast_1d(a)
        for a in broadcast(_as_codes(raw[0]), _as_codes(raw[1]), *(as_float(x) for x in raw[2:]))
    )

    # Chickens: the juvenile stage has no growth (weaning weight = hatch weight)
    weaning = np.where(sp == "CHK", birth, weaning)

    juvenile = _codes_in(co, ("FJ", "MJ"))
    subadult = _codes_in(co, ("FS", "MS"))
    fa = co == "FA"
    ma = co == "MA"
    fn = co == "FN"
    mn = co == "MN"
    with np.errstate(invalid="ignore"):
        ph1 = phase == 1
        ph2 = phase == 2

    # R leaves the outputs undefined (and errors) for any other cohort / phase
    defined = juvenile | subadult | fa | ma | ((fn | mn) & (ph1 | ph2))
    if not defined.all():
        i = int(np.flatnonzero(~defined)[0])
        abort(
            f"Cohort weights are undefined for `cohort_short` = {co[i]!r} with "
            f"`nondemo_productive_phase_id` = {phase[i]:g}; R fails with "
            "\"object 'live_weight_mature_stage' not found\"."
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Non-demographic cohorts: linear live-weight gain over both phases
        gain_f = (f_end - f_start) / (p1_f + p2_f)
        gain_m = (m_end - m_start) / (p1_m + p2_m)
        f_end_phase1 = f_start + (gain_f * p1_f)
        m_end_phase1 = m_start + (gain_m * p1_m)
        f_end_phase2 = f_end_phase1 + (gain_f * p2_f)
        m_end_phase2 = m_end_phase1 + (gain_m * p2_m)

    adult_of_sex = np.where(_codes_in(co, ("FJ", "FS")), female_adult, male_adult)
    fn1, fn2, mn1, mn2 = fn & ph1, fn & ph2, mn & ph1, mn & ph2

    mature = np.select(
        [juvenile | subadult, fa, ma, fn, mn],
        [adult_of_sex, female_adult, male_adult, f_end, m_end],
        default=np.nan,
    )
    initial = np.select(
        [juvenile, subadult, fa, ma, fn1, fn2, mn1, mn2],
        [birth, weaning, female_adult, male_adult, f_start, f_end_phase1, m_start, m_end_phase1],
        default=np.nan,
    )
    potential_final = np.select(
        [juvenile, subadult, fa, ma, fn1, fn2, mn1, mn2],
        [weaning, adult_of_sex, female_adult, male_adult, f_end_phase1, f_end_phase2,
         m_end_phase1, m_end_phase2],
        default=np.nan,
    )
    at_slaughter = np.select(
        [juvenile, co == "FS", co == "MS", fa, ma, fn | mn],
        [weaning, female_slaughter, male_slaughter, female_adult, male_adult, potential_final],
        default=np.nan,
    )

    return finalize_dict(
        {
            "live_weight_mature_stage": mature,
            "live_weight_cohort_initial": initial,
            "live_weight_cohort_potential_final": potential_final,
            "live_weight_cohort_at_slaughter": at_slaughter,
        },
        scalar,
    )


def calc_avg_weights(
    cohort_short: Any,
    live_weight_cohort_initial: Any,
    live_weight_cohort_potential_final: Any,
    live_weight_cohort_at_slaughter: Any,
    offtake_rate: Any,
) -> dict[str, Any]:
    """Average and final live weights of a cohort, accounting for offtake.

    Survivors reach the potential final weight while the offtaken fraction
    leaves at the slaughter weight:

    ``final = potential_final * (1 - offtake_rate) + at_slaughter * offtake_rate``

    with ``offtake_rate`` clamped to ``[0, 1]``. For non-demographic cohorts
    (``FN``, ``MN``) offtake is ignored and ``final = potential_final``.
    Then ``average = (initial + final) / 2``.

    Parameters
    ----------
    cohort_short : str or array-like
        Cohort code (``FJ``, ``FS``, ``FA``, ``MJ``, ``MS``, ``MA``, ``FN``, ``MN``).
    live_weight_cohort_initial : float or array-like
        Live weight at the beginning of the cohort stage (kg).
    live_weight_cohort_potential_final : float or array-like
        Potential final live weight in the absence of offtake (kg).
    live_weight_cohort_at_slaughter : float or array-like
        Live weight of animals removed from the cohort (kg).
    offtake_rate : float or array-like
        Annual proportion of animals removed from the cohort (fraction).

    Returns
    -------
    dict
        ``live_weight_cohort_average`` and ``live_weight_cohort_final`` (kg).
    """
    validate_avg_weight_inputs(
        cohort_short,
        live_weight_cohort_initial,
        live_weight_cohort_potential_final,
        live_weight_cohort_at_slaughter,
        offtake_rate,
    )
    scalar = all_scalar(
        cohort_short, live_weight_cohort_initial, live_weight_cohort_potential_final,
        live_weight_cohort_at_slaughter, offtake_rate,
    )
    co, initial, potential_final, at_slaughter, offtake = (
        np.atleast_1d(a)
        for a in broadcast(
            _as_codes(cohort_short),
            as_float(live_weight_cohort_initial),
            as_float(live_weight_cohort_potential_final),
            as_float(live_weight_cohort_at_slaughter),
            as_float(offtake_rate),
        )
    )
    nondemographic = _codes_in(co, ("FN", "MN"))

    with np.errstate(invalid="ignore", over="ignore"):
        offtake = normalize_rate(offtake)
        final = np.where(
            nondemographic,
            potential_final,
            potential_final * (1 - offtake) + at_slaughter * offtake,
        )
        average = (initial + final) / 2

    return finalize_dict(
        {"live_weight_cohort_average": average, "live_weight_cohort_final": final},
        scalar,
    )


def calc_daily_weight_gain(
    live_weight_cohort_potential_final: Any,
    live_weight_cohort_initial: Any,
    cohort_duration_days: Any,
) -> Any:
    """Average daily live weight gain over the cohort stage (kg/head/day).

    ``daily_weight_gain = (potential_final - initial) / cohort_duration_days``

    Parameters
    ----------
    live_weight_cohort_potential_final : float or array-like
        Potential final live weight of the cohort stage (kg).
    live_weight_cohort_initial : float or array-like
        Live weight at the beginning of the cohort stage (kg).
    cohort_duration_days : float or array-like
        Time each animal spends in the cohort (days).

    Returns
    -------
    float or numpy.ndarray
        Daily weight gain (kg/head/day).
    """
    validate_daily_gain_inputs(
        live_weight_cohort_potential_final,
        live_weight_cohort_initial,
        cohort_duration_days,
    )
    scalar = all_scalar(
        live_weight_cohort_potential_final, live_weight_cohort_initial, cohort_duration_days
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = (
            as_float(live_weight_cohort_potential_final) - as_float(live_weight_cohort_initial)
        ) / as_float(cohort_duration_days)
    return finalize(out, scalar)
