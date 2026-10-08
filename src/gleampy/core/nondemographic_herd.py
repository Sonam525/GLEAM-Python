"""Non-demographic herd core model (port of ``R/core_model_nondemographic_herd.R``).

Production-cycle model for the non-demographic cohort blocks ``FN`` / ``MN``
(e.g. fattening or laying operations): annual entrants are spread over the
cycle starts of a fixed 365-day horizon, each cycle has one or two productive
phases (constant daily mortality) and an optional rest period, and average
stock and terminal offtake are derived from the phase trajectories.

The R functions are scalar; these versions are vectorised (scalars in,
scalars out; arrays broadcast element-wise), with every R ``if`` branch
reproduced by a mask.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .._utils import all_scalar, as_float, as_str, broadcast, finalize, finalize_dict, is_scalar
from ..validation.nondemographic_herd_core import (
    validate_nondemo_avg_stock_inputs,
    validate_nondemo_cycle_geometry_inputs,
    validate_nondemo_offtake_inputs,
    validate_nondemo_phase_inputs,
    validate_nondemo_start_size_inputs,
)
from .demographic_herd import _r_pow, _seq_sum

#: Fixed simulation horizon of the non-demographic model (days).
TIME_HORIZON = 365


def assign_nondemographic_phase_durations(
    cohort_short,
    nondemo_productive_phase_id,
    cohort_duration_days=np.nan,
    phase1_nondemo_fem_duration_days=np.nan,
    phase2_nondemo_fem_duration_days=np.nan,
    phase1_nondemo_mal_duration_days=np.nan,
    phase2_nondemo_mal_duration_days=np.nan,
):
    """Fill ``cohort_duration_days`` of ``FN`` / ``MN`` phase rows from herd-level inputs.

    ``FN`` phase 1 / 2 rows take ``phase1_nondemo_fem_duration_days`` /
    ``phase2_nondemo_fem_duration_days`` and ``MN`` rows the ``*_mal_*``
    values, when those are not missing; every other row keeps
    ``cohort_duration_days`` (days). The result is always float64: fractional
    durations are never truncated (R truncates them when they are assigned to
    an integer ``cohort_duration_days`` column, see ``docs/R_PORT_NOTES.md``).

    Parameters
    ----------
    cohort_short : str or array-like of str
        Cohort code of each row (``FN`` / ``MN`` for the non-demographic blocks).
    nondemo_productive_phase_id : float or array-like
        Productive phase of each ``FN`` / ``MN`` row (1, or 2 for an optional
        second phase); NA for demographic cohorts.
    cohort_duration_days : float or array-like, default NA
        Time each animal spends in the cohort (days); kept where no
        herd-level phase duration applies.
    phase1_nondemo_fem_duration_days, phase2_nondemo_fem_duration_days : float or array-like, default NA
        Duration of productive phase 1 / 2 of the female block ``FN`` (days).
    phase1_nondemo_mal_duration_days, phase2_nondemo_mal_duration_days : float or array-like, default NA
        Duration of productive phase 1 / 2 of the male block ``MN`` (days).

    Returns
    -------
    float or numpy.ndarray
        ``cohort_duration_days`` with the phase durations filled in (days).
    """
    cs = as_str(cohort_short)
    ph = as_float(nondemo_productive_phase_id)
    cd = as_float(cohort_duration_days)
    p1f = as_float(phase1_nondemo_fem_duration_days)
    p2f = as_float(phase2_nondemo_fem_duration_days)
    p1m = as_float(phase1_nondemo_mal_duration_days)
    p2m = as_float(phase2_nondemo_mal_duration_days)
    cs, ph, cd, p1f, p2f, p1m, p2m = broadcast(cs, ph, cd, p1f, p2f, p1m, p2m)
    is_fn = np.asarray(cs == "FN", dtype=bool)
    is_mn = np.asarray(cs == "MN", dtype=bool)
    out = np.select(
        [
            is_fn & (ph == 1) & ~np.isnan(p1f),
            is_fn & (ph == 2) & ~np.isnan(p2f),
            is_mn & (ph == 1) & ~np.isnan(p1m),
            is_mn & (ph == 2) & ~np.isnan(p2m),
        ],
        [p1f, p2f, p1m, p2m],
        default=cd,
    )
    return finalize(out, all_scalar(
        cohort_short, nondemo_productive_phase_id, cohort_duration_days,
        phase1_nondemo_fem_duration_days, phase2_nondemo_fem_duration_days,
        phase1_nondemo_mal_duration_days, phase2_nondemo_mal_duration_days,
    ))


def calc_nondemographic_total_durations(cohort_short, cohort_duration_days):
    """Total productive-phase duration of the ``FN`` and ``MN`` blocks of one herd.

    Parameters
    ----------
    cohort_short : array-like of str
        Cohort code of each row of the herd.
    cohort_duration_days : array-like of float
        Time each animal spends in the cohort (days).

    Returns
    -------
    dict
        ``total_nondemo_fem_duration_days`` and ``total_nondemo_mal_duration_days``:
        sums (missing values removed) of ``cohort_duration_days`` over the
        ``FN`` / ``MN`` rows (days).
    """
    cs = np.atleast_1d(as_str(cohort_short))
    cd = np.atleast_1d(as_float(cohort_duration_days))
    cs, cd = broadcast(cs, cd)
    fem = cd[np.asarray(cs == "FN", dtype=bool)]
    mal = cd[np.asarray(cs == "MN", dtype=bool)]
    return {
        "total_nondemo_fem_duration_days": float(_seq_sum(fem, na_rm=True)),
        "total_nondemo_mal_duration_days": float(_seq_sum(mal, na_rm=True)),
    }


def calc_nondemo_cycle_geometry(
    phase1_nondemo_duration,
    phase2_nondemo_duration,
    rest_between_nondemo_cycles_duration,
):
    """Full and partial non-demographic production cycles within 365 days.

    ``cycle_length = phase1 + phase2 (if > 0) + rest``;
    ``number_full_nondemo_cycles = floor(365 / cycle_length)``; the remaining
    time fills a partial phase 1 (``min(remaining, phase1)``) and, if phase 1
    is completed and phase 2 exists, a partial phase 2.
    ``total_nondemo_cycle_starts_to_distribute = number_full + (partial_phase1 > 0)``.
    A phase 1 of zero days gives an all-zero geometry.

    Parameters
    ----------
    phase1_nondemo_duration : float or array-like
        Duration of productive phase 1 of the block (``FN`` or ``MN``) (days).
    phase2_nondemo_duration : float or array-like
        Duration of productive phase 2 (days); 0 when there is no second phase.
    rest_between_nondemo_cycles_duration : float or array-like
        Resting (empty) period between two production cycles (days).

    Returns
    -------
    dict
        ``number_full_nondemo_cycles``, ``partial_phase1_nondemo_duration``,
        ``partial_phase2_nondemo_duration`` (days),
        ``total_nondemo_cycle_starts_to_distribute`` and ``cycle_length`` (days).
    """
    validate_nondemo_cycle_geometry_inputs(
        phase1_nondemo_duration, phase2_nondemo_duration, rest_between_nondemo_cycles_duration
    )
    p1, p2, rest = broadcast(
        as_float(phase1_nondemo_duration), as_float(phase2_nondemo_duration),
        as_float(rest_between_nondemo_cycles_duration),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        no_phase1 = p1 <= 0
        phase2_exists = p2 > 0
        cycle_length = p1 + np.where(phase2_exists, p2, 0.0) + rest
        number_full = np.where(cycle_length > 0, np.floor(TIME_HORIZON / cycle_length), 0.0)
        time_remaining = TIME_HORIZON - number_full * cycle_length
        has_partial1 = (time_remaining > 0) & (p1 > 0)
        partial1 = np.where(has_partial1, np.minimum(time_remaining, p1), 0.0)
        time_remaining2 = time_remaining - partial1
        has_partial2 = has_partial1 & phase2_exists & (partial1 >= p1)
        partial2 = np.where(has_partial2, np.minimum(time_remaining2, p2), 0.0)
        starts = number_full + (partial1 > 0).astype("float64")
    zero = np.zeros_like(p1)
    out = {
        "number_full_nondemo_cycles": np.where(no_phase1, zero, number_full),
        "partial_phase1_nondemo_duration": np.where(no_phase1, zero, partial1),
        "partial_phase2_nondemo_duration": np.where(no_phase1, zero, partial2),
        "total_nondemo_cycle_starts_to_distribute": np.where(no_phase1, zero, starts),
        "cycle_length": np.where(no_phase1, zero, cycle_length),
    }
    return finalize_dict(out, all_scalar(
        phase1_nondemo_duration, phase2_nondemo_duration, rest_between_nondemo_cycles_duration
    ))


def calc_nondemo_start_sizes(
    cohort_stock_nondemo_annual_entrants,
    total_nondemo_cycle_starts_to_distribute,
):
    """Animals entering each production cycle: ``entrants / cycle starts``.

    As in R, when there is no cycle start (``total_nondemo_cycle_starts_to_distribute``
    is ``None`` or ``<= 0``) the bare value ``0`` is returned instead of the
    list (for array inputs those elements are ``0`` in the returned dict).

    Parameters
    ----------
    cohort_stock_nondemo_annual_entrants : float or array-like
        Animals entering the block (``FN`` or ``MN``) over the 365-day horizon
        (heads / year).
    total_nondemo_cycle_starts_to_distribute : float or array-like
        Production cycles started within the 365-day horizon (cycle starts / year).

    Returns
    -------
    dict or float
        ``{"cohort_stock_nondemo_start_cycle": heads per cycle}``, or ``0``.
    """
    validate_nondemo_start_size_inputs(
        cohort_stock_nondemo_annual_entrants, total_nondemo_cycle_starts_to_distribute
    )
    if total_nondemo_cycle_starts_to_distribute is None:
        return 0
    entrants = as_float(cohort_stock_nondemo_annual_entrants)
    starts = as_float(total_nondemo_cycle_starts_to_distribute)
    scalar = all_scalar(cohort_stock_nondemo_annual_entrants, total_nondemo_cycle_starts_to_distribute)
    if scalar and starts <= 0:
        return 0
    with np.errstate(divide="ignore", invalid="ignore"):
        start_cycle = np.where(starts <= 0, 0.0, entrants / starts)
    return finalize_dict({"cohort_stock_nondemo_start_cycle": start_cycle}, scalar)


def calc_nondemo_phase(
    cohort_stock_nondemo_start_by_phase,
    productive_phase_nondemo_duration,
    death_rate_nondemo_phase,
    max_simulation_days_nondemo_phase,
):
    """Stock dynamics of one non-demographic productive phase.

    The phase mortality is spread as a constant daily mortality
    ``d = 1 - (1 - death_rate_nondemo_phase)^(1 / duration)``; over
    ``t = max(0, min(max_simulation_days_nondemo_phase, duration))`` days the
    stock goes from ``start`` to ``end = start * (1 - d)^t``. A phase of zero
    days returns zeros.

    Parameters
    ----------
    cohort_stock_nondemo_start_by_phase : float or array-like
        Animals entering the productive phase (heads / phase): the cycle start
        size for phase 1, the end stock of phase 1 for phase 2.
    productive_phase_nondemo_duration : float or array-like
        Full duration of the productive phase (days).
    death_rate_nondemo_phase : float or array-like
        Fraction of the animals dying over the whole phase (fraction).
    max_simulation_days_nondemo_phase : float or array-like
        Days of the phase to simulate (days): the full duration, or the
        length of a terminal partial phase.

    Returns
    -------
    dict
        ``time_simulated_nondemographic`` (days) and ``cohort_stock_nondemo``
        (``{"start": heads, "end": heads}``).
    """
    validate_nondemo_phase_inputs(
        cohort_stock_nondemo_start_by_phase, productive_phase_nondemo_duration,
        death_rate_nondemo_phase, max_simulation_days_nondemo_phase,
    )
    scalar = all_scalar(
        cohort_stock_nondemo_start_by_phase, productive_phase_nondemo_duration,
        death_rate_nondemo_phase, max_simulation_days_nondemo_phase,
    )
    if productive_phase_nondemo_duration is None:
        return {"time_simulated_nondemographic": 0, "cohort_stock_nondemo": {"start": 0, "end": 0}}
    start = 0.0 if cohort_stock_nondemo_start_by_phase is None else as_float(cohort_stock_nondemo_start_by_phase)
    start, duration, death, max_days = broadcast(
        as_float(start), as_float(productive_phase_nondemo_duration),
        as_float(death_rate_nondemo_phase), as_float(max_simulation_days_nondemo_phase),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        no_phase = duration <= 0
        death_rate_daily = 1 - _r_pow(1 - death, 1 / duration)
        survival_rate_daily = 1 - death_rate_daily
        t = np.maximum(0.0, np.minimum(max_days, duration))
        end = start * _r_pow(survival_rate_daily, t)
    zero = np.zeros_like(duration)
    t = np.where(no_phase, zero, t)
    start_out = np.where(no_phase, zero, start)
    end = np.where(no_phase, zero, end)
    return {
        "time_simulated_nondemographic": finalize(t, scalar),
        "cohort_stock_nondemo": {"start": finalize(start_out, scalar), "end": finalize(end, scalar)},
    }


def calc_nondemo_avg_stock_phase_horizon(
    full_nondemo_phase_duration,
    partial_nondemo_phase,
    number_full_nondemo_cycles,
):
    """Average stock of one productive phase over the 365-day horizon.

    Animal-days of a phase occurrence are ``t * (start + end) / 2`` (0 when
    ``t <= 0``); the average stock is
    ``(number_full_nondemo_cycles * ad_full + ad_partial) / 365`` (heads).

    Parameters
    ----------
    full_nondemo_phase_duration : dict
        :func:`calc_nondemo_phase` result for a full phase
        (``time_simulated_nondemographic`` in days, stocks in heads).
    partial_nondemo_phase : dict
        :func:`calc_nondemo_phase` result for the terminal partial phase.
    number_full_nondemo_cycles : float or array-like
        Complete production cycles within the 365-day horizon (cycles / year).

    Returns
    -------
    dict
        ``{"cohort_stock_size_unscaled": heads}``.
    """
    validate_nondemo_avg_stock_inputs(
        full_nondemo_phase_duration, partial_nondemo_phase, number_full_nondemo_cycles
    )

    def animal_days(phase: Mapping[str, Any]) -> np.ndarray:
        t = phase.get("time_simulated_nondemographic")
        if t is None:
            return np.asarray(0.0)
        t = as_float(t)
        start = as_float(phase["cohort_stock_nondemo"]["start"])
        end = as_float(phase["cohort_stock_nondemo"]["end"])
        with np.errstate(invalid="ignore", over="ignore"):
            return np.where(t <= 0, 0.0, t * (start + end) / 2)

    ad_full = animal_days(full_nondemo_phase_duration)
    ad_part = animal_days(partial_nondemo_phase)
    with np.errstate(invalid="ignore", over="ignore"):
        size = (as_float(number_full_nondemo_cycles) * ad_full + ad_part) / TIME_HORIZON
    scalar = is_scalar(number_full_nondemo_cycles) and np.ndim(ad_full) == 0 and np.ndim(ad_part) == 0
    return finalize_dict({"cohort_stock_size_unscaled": size}, scalar)


def calc_nondemo_offtake_total_horizon(
    cohort_stock_nondemo_end_phase1,
    cohort_stock_nondemo_end_phase2,
    cohort_stock_nondemo_annual_entrants,
    cohort_stock_nondemo_start_cycle,
    number_full_nondemo_cycles,
    partial_phase1_nondemo_duration,
    partial_phase2_nondemo_duration,
    phase1_nondemo_duration,
    phase2_nondemo_duration,
    simulation_duration,
):
    """Offtake of a non-demographic block over the horizon and the assessment period.

    Offtake happens at the end of the last existing productive phase (phase 2
    if ``phase2_nondemo_duration > 0``, else phase 1):
    ``offtake = entrants * end_terminal / start_cycle`` (0 if
    ``start_cycle <= 0``) and ``offtake_assessment = offtake / 365 * simulation_duration``.

    Parameters
    ----------
    cohort_stock_nondemo_end_phase1, cohort_stock_nondemo_end_phase2 : float or array-like
        Animals left at the end of a full productive phase 1 / 2 (heads / cycle).
    cohort_stock_nondemo_annual_entrants : float or array-like
        Animals entering the block over the 365-day horizon (heads / year).
    cohort_stock_nondemo_start_cycle : float or array-like
        Animals starting one production cycle (heads / cycle).
    number_full_nondemo_cycles : float or array-like
        Complete production cycles within the horizon (cycles / year).
    partial_phase1_nondemo_duration, partial_phase2_nondemo_duration : float or array-like
        Durations of the terminal partial phases 1 / 2 (days); as in R, a
        positive value below 1 day fails validation.
    phase1_nondemo_duration, phase2_nondemo_duration : float or array-like
        Full durations of productive phases 1 / 2 (days); phase 2 is 0 when absent.
    simulation_duration : float
        Length of the assessment period (days).

    Returns
    -------
    dict
        ``offtake_heads_nondemo_phase1``, ``offtake_heads_nondemo_phase2``,
        ``offtake_heads_assessment_nondemo_phase1`` and
        ``offtake_heads_assessment_nondemo_phase2`` (heads).
    """
    validate_nondemo_offtake_inputs(
        cohort_stock_nondemo_end_phase1, cohort_stock_nondemo_end_phase2, number_full_nondemo_cycles,
        partial_phase1_nondemo_duration, partial_phase2_nondemo_duration, phase1_nondemo_duration,
        phase2_nondemo_duration, simulation_duration,
    )
    scalar = all_scalar(
        cohort_stock_nondemo_end_phase1, cohort_stock_nondemo_end_phase2,
        cohort_stock_nondemo_annual_entrants, cohort_stock_nondemo_start_cycle,
        phase2_nondemo_duration, simulation_duration,
    )

    def norm(x: Any) -> np.ndarray:  # NULL / length-0 -> 0
        if x is None or np.size(x) == 0:
            return np.asarray(0.0)
        return as_float(x)

    phase2_exists = as_float(phase2_nondemo_duration) > 0
    sim = norm(simulation_duration)
    end1 = norm(cohort_stock_nondemo_end_phase1)
    end2 = norm(cohort_stock_nondemo_end_phase2)
    entrants = norm(cohort_stock_nondemo_annual_entrants)
    start_cycle = norm(cohort_stock_nondemo_start_cycle)
    phase2_exists, sim, end1, end2, entrants, start_cycle = broadcast(
        phase2_exists, sim, end1, end2, entrants, start_cycle
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        survival = np.where(
            start_cycle > 0, np.where(phase2_exists, end2 / start_cycle, end1 / start_cycle), 0.0
        )
        offtake = entrants * survival
        off2 = np.where(phase2_exists, offtake, 0.0)
        off1 = np.where(phase2_exists, 0.0, offtake)
        out = {
            "offtake_heads_nondemo_phase1": off1,
            "offtake_heads_nondemo_phase2": off2,
            "offtake_heads_assessment_nondemo_phase1": off1 / TIME_HORIZON * sim,
            "offtake_heads_assessment_nondemo_phase2": off2 / TIME_HORIZON * sim,
        }
    return finalize_dict(out, scalar)
