"""Demographic herd core model (port of ``R/core_model_demographic_herd.R``).

Steady-state demographic herd projection on a daily time step (Dynmod
*STEADY1*-like): annual death / offtake rates are converted into daily
competing-risk hazards and transition probabilities, the herd is iterated from
an initial structure until the cohort growth rates stabilise, and one year of
dynamics is then projected to derive stocks, offtake and the numbers diverted
to the non-demographic block.

The R functions take one herd at a time (named per-cohort vectors). The Python
public functions keep that per-herd signature (``dict`` / ``pd.Series`` keyed
by cohort code in, ``pd.Series`` out). Internally every algorithm is written
as a *kernel* operating on numpy arrays with one element per herd, so
:func:`gleam.run_demographic_herd_module` simulates all herds at once with the
same floating-point operations, iteration counts and convergence logic as R.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import all_scalar, as_float, finalize_dict
from ..validation.demographic_herd_core import (
    validate_fecundity_inputs,
    validate_offtake_summary_inputs,
    validate_population_size_inputs,
    validate_steady_state_inputs,
    validate_transition_inputs,
)

#: The 6 demographic sex-age cohorts (``gleam_cohorts_demographic``).
SIX_COHORTS: tuple[str, ...] = K.GLEAM_COHORTS_DEMOGRAPHIC
#: The 8 cohorts of the steady-state structure (6 cohorts + 2 birth classes).
EIGHT_COHORTS: tuple[str, ...] = ("FB", "FJ", "FS", "FA", "MB", "MJ", "MS", "MA")
#: The 10 cohorts of the transition model (+ birth and culling classes).
TEN_COHORTS: tuple[str, ...] = ("FB", "FJ", "FS", "FA", "FC", "MB", "MJ", "MS", "MA", "MC")

#: Up to this many herds the steady-state iteration runs with plain Python
#: floats, herd by herd (fastest for few herds); larger batches use the numpy
#: kernel. Both evaluate the same IEEE operations and give identical results.
SCALAR_STEADY_STATE_MAX_HERDS = 48

_NA_CONDITION_MSG = (
    "missing value where TRUE/FALSE needed: a cohort-specific growth rate change "
    "(lambda) is NaN in the steady-state simulation (a cohort reached zero animals)."
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _names_values(x: Any) -> tuple[list[str] | None, np.ndarray]:
    """Names and float values of an R-like named vector (dict / Series)."""
    if isinstance(x, Mapping):
        return [str(k) for k in x.keys()], np.atleast_1d(as_float(list(x.values()))).reshape(-1)
    if isinstance(x, pd.Series):
        return [str(k) for k in x.index], np.atleast_1d(as_float(x)).reshape(-1)
    return None, np.atleast_1d(as_float(x)).reshape(-1)


def _get(x: Any, name: str) -> float:
    """R ``x[[name]]`` on a named vector (first match)."""
    names, vals = _names_values(x)
    if names is None or name not in names:
        raise KeyError(f"subscript out of bounds: no element named {name!r}")
    return float(vals[names.index(name)])


def _pick(x: Any, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """``{name: array([x[[name]]])}`` for the given names (length-1 arrays)."""
    return {n: np.array([_get(x, n)]) for n in names}


def _series(values: Any, names: list[str] | tuple[str, ...]) -> pd.Series:
    vals = np.asarray(values, dtype="float64").reshape(-1)
    return pd.Series(vals, index=list(names), dtype="float64")


def _r_pow(x: Any, y: Any) -> np.ndarray:
    """R's ``x ^ y`` (``R_POW``): ``y == 2`` is ``x * x``; ``1 ^ y`` and ``x ^ 0`` are 1."""
    x_, y_ = np.broadcast_arrays(as_float(x), as_float(y))
    with np.errstate(all="ignore"):
        out = np.power(x_, y_)
        out = np.where((x_ == 1.0) | (y_ == 0.0), 1.0, out)
        out = np.where(y_ == 2.0, x_ * x_, out)
    return out


def _seq_sum(a: Any, axis: int = -1, na_rm: bool = False) -> np.ndarray:
    """R ``sum()`` as evaluated by the reference R build: sequential accumulation."""
    a = np.asarray(a, dtype="float64")
    if na_rm:
        a = np.where(np.isnan(a), 0.0, a)
    if a.shape[axis] == 0:
        return np.zeros(np.delete(a.shape, axis))
    return np.take(np.add.accumulate(a, axis=axis), -1, axis=axis)


# --------------------------------------------------------------------------
# Fecundity
# --------------------------------------------------------------------------


def calc_fecundity_rates(parturition_rate, litter_size, birth_fraction_female):
    """Daily number of female and male offspring per adult female.

    Parameters
    ----------
    parturition_rate : float or array-like
        Annual number of parturitions per adult female (# parturitions/female/year).
    litter_size : float or array-like
        Offspring born per parturition (# offspring/parturition).
    birth_fraction_female : float or array-like
        Probability that a newborn is female (fraction).

    Returns
    -------
    dict
        ``fecundity_female = litter_size * birth_fraction_female * (parturition_rate / 365)``
        and ``fecundity_male = litter_size * (1 - birth_fraction_female) * (parturition_rate / 365)``
        (# offspring/adult female/day).
    """
    validate_fecundity_inputs(parturition_rate, litter_size, birth_fraction_female)
    pr = as_float(parturition_rate)
    ls = as_float(litter_size)
    bf = as_float(birth_fraction_female)
    fecundity_female = ls * bf * (pr / 365)
    fecundity_male = ls * (1 - bf) * (pr / 365)
    return finalize_dict(
        {"fecundity_female": fecundity_female, "fecundity_male": fecundity_male},
        all_scalar(parturition_rate, litter_size, birth_fraction_female),
    )


# --------------------------------------------------------------------------
# Transition probabilities
# --------------------------------------------------------------------------


def _bump_zero_hazard(offtake_rate: np.ndarray, death_rate: np.ndarray) -> np.ndarray:
    """``death_rate[offtake_rate == 0 & death_rate == 0] <- 1e-12`` (avoids 0/0)."""
    death = np.array(death_rate, dtype="float64", copy=True)
    zero = (np.asarray(offtake_rate) == 0) & (death == 0)
    death[zero] = 1e-12
    return death


def _transition_kernel(
    dur: Mapping[str, np.ndarray], off: Mapping[str, np.ndarray], death: Mapping[str, np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    """Hazards and daily transition probabilities for arrays of herds.

    ``dur`` / ``off`` / ``death`` map each of the 6 cohorts to an array (one
    element per herd); ``death`` already carries the ``1e-12`` bump of R.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        hazard_death: dict[str, np.ndarray] = {}
        duration_max365: dict[str, np.ndarray] = {}
        for c in SIX_COHORTS:
            d = np.asarray(dur[c], dtype="float64")
            neg_log = -np.log(1 - np.asarray(death[c], dtype="float64"))
            hazard_death[c] = np.where(d < 365, neg_log / d, neg_log / 365)
            duration_max365[c] = np.where(d < 365, d, 365.0)

        # Offtake hazard by Newton-Raphson (15 iterations, as in R)
        hazard_offtake: dict[str, np.ndarray] = {}
        for c in SIX_COHORTS:
            hda = hazard_death[c] * duration_max365[c]
            rate = np.asarray(off[c], dtype="float64")
            h = f = deriv = rate
            for t in range(1, 16):
                h = rate if t == 1 else h - (f / deriv)
                f = (h / (hda + h)) * (1 - np.exp(-hda - h)) - rate
                deriv = (
                    hda * (1 - np.exp(-hda - h)) + h * (hda + h) * np.exp(-hda - h)
                ) / _r_pow(hda + h, 2)
            hazard_offtake[c] = h / duration_max365[c]

        # Extend to 10 cohorts: FB/MB use juvenile rates, FC/MC adult rates
        src = {"FB": "FJ", "FJ": "FJ", "FS": "FS", "FA": "FA", "FC": "FA",
               "MB": "MJ", "MJ": "MJ", "MS": "MS", "MA": "MA", "MC": "MA"}
        one = np.ones_like(hazard_death["FJ"])
        duration_all = {
            "FB": one, "FJ": dur["FJ"] - 1, "FS": dur["FS"], "FA": dur["FA"], "FC": one,
            "MB": one, "MJ": dur["MJ"] - 1, "MS": dur["MS"], "MA": dur["MA"], "MC": one,
        }
        p_death: dict[str, np.ndarray] = {}
        p_off: dict[str, np.ndarray] = {}
        for c in TEN_COHORTS:
            hd = hazard_death[src[c]]
            ho = hazard_offtake[src[c]]
            p_death[c] = (hd / (hd + ho)) * (1 - np.exp(-(hd + ho)))
            p_off[c] = (ho / (hd + ho)) * (1 - np.exp(-(hd + ho)))
        for c in ("FC", "MC"):  # culling cohorts cannot die again and are all offtaken
            p_death[c] = np.zeros_like(one)
            p_off[c] = np.ones_like(one)
        p_surv: dict[str, np.ndarray] = {}
        p_growth: dict[str, np.ndarray] = {}
        for c in TEN_COHORTS:
            ps = 1 - p_death[c] - p_off[c]
            p_surv[c] = ps
            d = np.asarray(duration_all[c], dtype="float64")
            p_growth[c] = (_r_pow(ps, d - 1) - _r_pow(ps, d)) / (1 - _r_pow(ps, d))

    return {
        "hazard_death": hazard_death,
        "hazard_offtake": hazard_offtake,
        "probability_death": p_death,
        "probability_offtake": p_off,
        "probability_survival": p_surv,
        "probability_growth": p_growth,
    }


def calc_transition_probabilities(cohort_duration_days, offtake_rate, death_rate):
    """Daily hazards and transition probabilities for the sex-age cohorts of one herd.

    Annual death and offtake rates are converted to daily hazards under
    competing risks: ``hazard_death = -log(1 - death_rate) / min(duration, 365)``
    and the offtake hazard ``h`` solves (Newton-Raphson, 15 iterations)
    ``h / (hd + h) * (1 - exp(-hd - h)) = offtake_rate`` with ``hd`` the death
    hazard over the (at most 365-day) cohort duration. Daily probabilities are
    ``p_death = hd / (hd + ho) * (1 - exp(-(hd + ho)))`` (offtake analogous),
    ``p_survival = 1 - p_death - p_offtake`` and
    ``p_growth = (s^(D - 1) - s^D) / (1 - s^D)`` with ``D`` the cohort duration
    (1 day for the birth and culling classes, ``D - 1`` for juveniles).
    Cohorts with zero death and offtake rates get ``death_rate = 1e-12``.

    Parameters
    ----------
    cohort_duration_days : dict or pandas.Series
        Days spent in each of the 6 cohorts (``FJ, FS, FA, MJ, MS, MA``).
    offtake_rate : dict or pandas.Series
        Annual fraction of animals removed per cohort (fraction).
    death_rate : dict or pandas.Series
        Annual fraction of deaths per cohort (fraction).

    Returns
    -------
    dict
        ``hazard_death`` and ``hazard_offtake`` (6 cohorts, day^-1) and
        ``probability_death``, ``probability_offtake``, ``probability_survival``,
        ``probability_growth`` (10 cohorts ``FB, FJ, FS, FA, FC, MB, MJ, MS, MA, MC``),
        each a ``pandas.Series`` indexed by cohort.
    """
    validate_transition_inputs(cohort_duration_days, offtake_rate, death_rate)

    dur_names, dur_vals = _names_values(cohort_duration_days)
    _, off_pos = _names_values(offtake_rate)
    _, death_pos = _names_values(death_rate)
    if dur_names is None:
        raise KeyError("`cohort_duration_days` must be a named vector")
    # Positional, as in R: the bump uses both rate vectors element by element,
    # hazard_death combines death_rate and durations element by element and is
    # named after cohort_duration_days; the Newton step reads offtake_rate by name.
    death_pos = _bump_zero_hazard(off_pos, death_pos)
    dur = {n: np.array([v]) for n, v in reversed(list(zip(dur_names, dur_vals)))}
    death = {n: np.array([v]) for n, v in reversed(list(zip(dur_names, death_pos)))}
    res = _transition_kernel(
        {c: dur[c] for c in SIX_COHORTS},
        _pick(offtake_rate, SIX_COHORTS),
        {c: death[c] for c in SIX_COHORTS},
    )
    out = {
        "hazard_death": _series([res["hazard_death"][n][0] for n in dur_names], dur_names),
        "hazard_offtake": _series([res["hazard_offtake"][n][0] for n in dur_names], dur_names),
    }
    for key in ("probability_death", "probability_offtake", "probability_survival", "probability_growth"):
        out[key] = _series([res[key][c][0] for c in TEN_COHORTS], TEN_COHORTS)
    return out


# --------------------------------------------------------------------------
# Steady-state structure
# --------------------------------------------------------------------------


def _n_steps(max_simulation_years: float) -> int:
    """Length of R's ``1:(max_simulation_years * 365 + 1)`` (at least 1)."""
    last = float(max_simulation_years) * 365 + 1
    return max(int(np.floor(last)), 1)


class _NAConditionError(ValueError):
    """``if (NA)`` in R: a lambda change is NaN and no other one is above threshold."""


def _steady_state_scalar(
    init: Mapping[str, float], n_steps: int, thr: float, fec_f: float, fec_m: float,
    pd_: Mapping[str, float], po: Mapping[str, float], pg: Mapping[str, float], pn: Mapping[str, float],
) -> tuple[int, tuple[float, ...], float]:
    """One herd with Python floats; mirrors the R loop statement by statement.

    Returns ``(days_steady, xend (FB, FJ, FS, FA, MB, MJ, MS, MA), fem_juv_fec[days_steady - 1])``.
    Raises ``ZeroDivisionError`` where R would silently divide by zero (the
    caller then falls back to the IEEE numpy kernel).
    """
    s_fb = 1 - pd_["FB"] - po["FB"]
    s_fj = 1 - pd_["FJ"] - po["FJ"]
    s_fs = 1 - pd_["FS"] - po["FS"]
    s_fa = 1 - pd_["FA"] - po["FA"]
    s_mb = 1 - pd_["MB"] - po["MB"]
    s_mj = 1 - pd_["MJ"] - po["MJ"]
    s_ms = 1 - pd_["MS"] - po["MS"]
    s_ma = 1 - pd_["MA"] - po["MA"]
    g_fj, g_fs, g_fa = pg["FJ"], pg["FS"], pg["FA"]
    g_mj, g_ms, g_ma = pg["MJ"], pg["MS"], pg["MA"]
    k_fj, k_fs, k_fa = 1 - g_fj, 1 - g_fs, 1 - g_fa
    k_mj, k_ms, k_ma = 1 - g_mj, 1 - g_ms, 1 - g_ma
    r_fj, r_fs = 1 - pn["FJ"], 1 - pn["FS"]
    r_mj, r_ms = 1 - pn["MJ"], 1 - pn["MS"]

    p1: tuple[float, ...] | None = None  # fec values at t - 1 (FJ, FS, FA, MJ, MS, MA)
    p2: tuple[float, ...] | None = None  # fec values at t - 2
    fjg = fsg = fag = mjg = msg = mag = 0.0
    all_below = bool(1 < thr)  # lambda_change <- rep(1, 6)
    for t in range(1, n_steps + 1):
        if t == 1:
            fbf = init["FA"] * fec_f
            fjf = init["FJ"]
            fsf = init["FS"]
            faf = init["FA"]
            mbf = init["FA"] * fec_m
            mjf = init["MJ"]
            msf = init["MS"]
            maf = init["MA"]
        else:
            fjf = fjg
            fsf = fsg
            faf = fag
            fbf = faf * fec_f
            mjf = mjg
            msf = msg
            maf = mag
            mbf = faf * fec_m
        cur = (fjf, fsf, faf, mjf, msf, maf)
        if t > 2:
            all_below = True
            na = False
            for a, b, c in zip(cur, p1, p2):
                v = a / b - b / c
                if v < thr:
                    continue
                if v != v:
                    na = True
                else:
                    all_below = False
            if all_below and na:
                raise _NAConditionError(_NA_CONDITION_MSG)
        if all_below:
            prev = p1[0] if p1 is not None else np.nan
            break
        fb = fbf * s_fb
        fj = fjf * s_fj
        fs = fsf * s_fs
        fa = faf * s_fa
        mb = mbf * s_mb
        mj = mjf * s_mj
        ms = msf * s_ms
        ma = maf * s_ma
        fjg = fb + k_fj * fj
        fsg = r_fj * (g_fj * fj) + k_fs * fs
        fag = r_fs * (g_fs * fs) + k_fa * fa
        mjg = mb + k_mj * mj
        msg = r_mj * (g_mj * mj) + k_ms * ms
        mag = r_ms * (g_ms * ms) + k_ma * ma
        p2 = p1
        p1 = cur
    else:
        # loop exhausted: R keeps the values of the last time step
        prev = p2[0] if p2 is not None else np.nan
    return t, (fbf, fjf, fsf, faf, mbf, mjf, msf, maf), prev


def _steady_state_batch(
    init: Mapping[str, Any], n_steps: int, thr: float, fec_f: np.ndarray, fec_m: np.ndarray,
    pd_: Mapping[str, np.ndarray], po: Mapping[str, np.ndarray], pg: Mapping[str, np.ndarray],
    pn: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All herds at once with numpy arrays (IEEE semantics, active-set compaction).

    Returns ``days (n,)``, ``xend (n, 8)`` and ``fem_juv_fec[days - 1] (n,)``.
    """
    ff = np.asarray(fec_f, dtype="float64").reshape(-1)
    fm = np.asarray(fec_m, dtype="float64").reshape(-1)
    n = len(ff)
    days = np.zeros(n, dtype=np.int64)
    xend = np.full((n, 8), np.nan)
    prev_out = np.full(n, np.nan)
    ones = np.ones(n)

    with np.errstate(all="ignore"):
        s = np.stack([1 - pd_[c] - po[c] for c in EIGHT_COHORTS])  # FB FJ FS FA MB MJ MS MA
        gr = np.stack([pg[c] * ones for c in SIX_COHORTS])          # FJ FS FA MJ MS MA
        kk = np.stack([(1 - pg[c]) * ones for c in SIX_COHORTS])
        rr = np.stack([(1 - pn[c]) * ones for c in ("FJ", "FS", "MJ", "MS")])
        idx = np.arange(n)
        p1 = p2 = state = None
        cur = fbf = mbf = None
        below_init = bool(1 < thr)
        for t in range(1, n_steps + 1):
            if t == 1:
                cur = np.stack([np.asarray(init[c], dtype="float64") * ones for c in SIX_COHORTS])
            else:
                cur = state
            fbf = cur[2] * ff
            mbf = cur[2] * fm
            if t > 2:
                lc = cur / p1 - p1 / p2
                isna = np.isnan(lc)
                has_false = (~(lc < thr) & ~isna).any(axis=0)
                if (isna.any(axis=0) & ~has_false).any():
                    raise _NAConditionError(_NA_CONDITION_MSG)
                done = ~has_false
            else:
                done = np.full(len(idx), below_init)
            if done.any():
                sel = idx[done]
                days[sel] = t
                xend[sel] = np.stack(
                    [fbf[done], cur[0, done], cur[1, done], cur[2, done],
                     mbf[done], cur[3, done], cur[4, done], cur[5, done]], axis=1)
                if p1 is not None:
                    prev_out[sel] = p1[0, done]
                keep = ~done
                idx = idx[keep]
                if not len(idx):
                    break
                cur, fbf, mbf = cur[:, keep], fbf[keep], mbf[keep]
                p1 = p1[:, keep] if p1 is not None else None
                p2 = p2[:, keep] if p2 is not None else None
                ff, fm = ff[keep], fm[keep]
                s, gr, kk, rr = s[:, keep], gr[:, keep], kk[:, keep], rr[:, keep]
            fb = fbf * s[0]
            fj = cur[0] * s[1]
            fs = cur[1] * s[2]
            fa = cur[2] * s[3]
            mb = mbf * s[4]
            mj = cur[3] * s[5]
            ms = cur[4] * s[6]
            ma = cur[5] * s[7]
            state = np.stack([
                fb + kk[0] * fj,
                rr[0] * (gr[0] * fj) + kk[1] * fs,
                rr[1] * (gr[1] * fs) + kk[2] * fa,
                mb + kk[3] * mj,
                rr[2] * (gr[3] * mj) + kk[4] * ms,
                rr[3] * (gr[4] * ms) + kk[5] * ma,
            ])
            p2 = p1
            p1 = cur
        else:
            if len(idx):  # not converged within max_simulation_years
                days[idx] = n_steps
                xend[idx] = np.stack([fbf, cur[0], cur[1], cur[2], mbf, cur[3], cur[4], cur[5]], axis=1)
                if p2 is not None:
                    prev_out[idx] = p2[0]
    return days, xend, prev_out


def _steady_state_kernel(
    init: Mapping[str, Any], max_simulation_years: float, min_lambda_change: float,
    fec_f: np.ndarray, fec_m: np.ndarray, pd_: Mapping[str, np.ndarray], po: Mapping[str, np.ndarray],
    pg: Mapping[str, np.ndarray], pn: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Steady-state results for arrays of herds (see :func:`calc_steady_state_structure`)."""
    fec_f = np.atleast_1d(np.asarray(fec_f, dtype="float64"))
    fec_m = np.atleast_1d(np.asarray(fec_m, dtype="float64"))
    n = len(fec_f)
    n_steps = _n_steps(max_simulation_years)
    thr = float(min_lambda_change)
    init_a = {c: np.broadcast_to(np.asarray(init[c], dtype="float64"), (n,)) for c in SIX_COHORTS}

    if n <= SCALAR_STEADY_STATE_MAX_HERDS:
        days = np.zeros(n, dtype=np.int64)
        xend = np.full((n, 8), np.nan)
        prev = np.full(n, np.nan)
        for i in range(n):
            def one(m: Mapping[str, np.ndarray], names) -> dict[str, float]:
                return {c: float(np.broadcast_to(m[c], (n,))[i]) for c in names}
            args = (
                one(init_a, SIX_COHORTS), n_steps, thr, float(fec_f[i]), float(fec_m[i]),
                one(pd_, TEN_COHORTS), one(po, TEN_COHORTS), one(pg, TEN_COHORTS), one(pn, SIX_COHORTS),
            )
            try:
                d, x, p = _steady_state_scalar(*args)
                days[i], xend[i], prev[i] = d, x, p
            except ZeroDivisionError:
                sub = lambda m, names: {c: np.array([v]) for c, v in one(m, names).items()}  # noqa: E731
                d, x, p = _steady_state_batch(
                    sub(init_a, SIX_COHORTS), n_steps, thr, fec_f[i:i + 1], fec_m[i:i + 1],
                    sub(pd_, TEN_COHORTS), sub(po, TEN_COHORTS), sub(pg, TEN_COHORTS), sub(pn, SIX_COHORTS),
                )
                days[i], xend[i], prev[i] = d[0], x[0], p[0]
    else:
        bc = lambda m, names: {c: np.broadcast_to(np.asarray(m[c], dtype="float64"), (n,)) for c in names}  # noqa: E731
        days, xend, prev = _steady_state_batch(
            init_a, n_steps, thr, fec_f, fec_m,
            bc(pd_, TEN_COHORTS), bc(po, TEN_COHORTS), bc(pg, TEN_COHORTS), bc(pn, SIX_COHORTS),
        )

    with np.errstate(all="ignore"):
        structure = xend / _seq_sum(xend, axis=1)[:, None]
        # xend columns: FB FJ FS FA MB MJ MS MA
        grouped = np.stack([xend[:, 0] + xend[:, 1], xend[:, 2], xend[:, 3],
                            xend[:, 4] + xend[:, 5], xend[:, 6], xend[:, 7]], axis=1)
        share = np.stack([structure[:, 0] + structure[:, 1], structure[:, 2], structure[:, 3],
                          structure[:, 4] + structure[:, 5], structure[:, 6], structure[:, 7]], axis=1)
        growth = _r_pow(xend[:, 1] / prev, 365) - 1
    return {
        "days_to_steady_state": days,
        "herd_structure": structure,
        "cohort_share": share,
        "growth_rate_herd": growth,
        "size_unscaled": grouped,
        "herd_size_total_demographic": _seq_sum(grouped, axis=1, na_rm=True),
    }


def calc_steady_state_structure(
    initial_herd_structure,
    max_simulation_years,
    min_lambda_change,
    fecundity_female,
    fecundity_male,
    probability_death,
    probability_offtake,
    probability_growth,
    proportion_nondemographic,
):
    """Iterate the daily herd dynamics until the cohort structure is steady.

    Starting from ``initial_herd_structure``, births (adult females x daily
    fecundity), survival (``1 - p_death - p_offtake``) and cohort growth
    (``p_growth``) are applied day by day. A share
    ``proportion_nondemographic`` of each growth transition is diverted out of
    the demographic herd (``T_nondemo = p * T``, ``T_demo = (1 - p) * T``).
    The loop stops at the first day where all six cohort-specific changes of
    the daily growth ratio, ``x[t]/x[t-1] - x[t-1]/x[t-2]``, are below
    ``min_lambda_change`` (signed comparison, as in R), or after
    ``max_simulation_years * 365 + 1`` days.

    Parameters
    ----------
    initial_herd_structure : dict or pandas.Series
        Starting heads for ``FJ, FS, FA, MJ, MS, MA`` (only affects convergence speed).
    max_simulation_years : float
        Maximum number of simulated years.
    min_lambda_change : float
        Convergence threshold on the change of the daily cohort growth ratios.
    fecundity_female, fecundity_male : float
        Daily female / male offspring per adult female.
    probability_death, probability_offtake, probability_growth : dict or pandas.Series
        Daily probabilities for the 10 cohorts ``FB, FJ, FS, FA, FC, MB, MJ, MS, MA, MC``.
    proportion_nondemographic : dict or pandas.Series
        Fraction of each cohort's growth transition diverted to the non-demographic block.

    Returns
    -------
    dict
        ``days_to_steady_state`` (int), ``herd_structure`` (8 shares), ``cohort_share``
        (6 shares, ``FJ = FB + FJ``, ``MJ = MB + MJ``), ``growth_rate_herd``
        (``(FJ[t]/FJ[t-1])^365 - 1``), ``size_unscaled`` (6 cohort sizes) and
        ``herd_size_total_demographic`` (their sum).
    """
    validate_steady_state_inputs(
        initial_herd_structure, max_simulation_years, min_lambda_change, fecundity_female,
        fecundity_male, probability_death, probability_offtake, probability_growth,
        proportion_nondemographic,
    )
    res = _steady_state_kernel(
        {c: _get(initial_herd_structure, c) for c in SIX_COHORTS},
        float(np.asarray(max_simulation_years).reshape(-1)[0]),
        float(np.asarray(min_lambda_change).reshape(-1)[0]),
        np.array([float(np.asarray(fecundity_female).reshape(-1)[0])]),
        np.array([float(np.asarray(fecundity_male).reshape(-1)[0])]),
        _pick(probability_death, TEN_COHORTS),
        _pick(probability_offtake, TEN_COHORTS),
        _pick(probability_growth, TEN_COHORTS),
        _pick(proportion_nondemographic, SIX_COHORTS),
    )
    return {
        "days_to_steady_state": int(res["days_to_steady_state"][0]),
        "herd_structure": _series(res["herd_structure"][0], EIGHT_COHORTS),
        "cohort_share": _series(res["cohort_share"][0], SIX_COHORTS),
        "growth_rate_herd": float(res["growth_rate_herd"][0]),
        "size_unscaled": _series(res["size_unscaled"][0], SIX_COHORTS),
        "herd_size_total_demographic": float(res["herd_size_total_demographic"][0]),
    }


# --------------------------------------------------------------------------
# One-year projection
# --------------------------------------------------------------------------


def _projection_kernel(
    herd_size_total: np.ndarray, fec_f: np.ndarray, fec_m: np.ndarray,
    pd_: Mapping[str, np.ndarray], po: Mapping[str, np.ndarray], pg: Mapping[str, np.ndarray],
    growth_rate_herd: np.ndarray, herd_structure: Mapping[str, np.ndarray], cohort_share_pos: np.ndarray,
    pn: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """One year (366 daily steps) of dynamics for arrays of herds.

    ``cohort_share_pos`` is ``(n, 6)`` in the positional order of R's
    ``cohort_share`` vector (relabelled ``FJ..MA`` as in R).
    """
    hst = np.asarray(herd_size_total, dtype="float64").reshape(-1)
    n = len(hst)
    ones = np.ones(n)
    ff = np.asarray(fec_f, dtype="float64") * ones
    fm = np.asarray(fec_m, dtype="float64") * ones
    with np.errstate(all="ignore"):
        xini = {c: hst * herd_structure[c] for c in EIGHT_COHORTS}
        size = hst[:, None] * np.asarray(cohort_share_pos, dtype="float64")
        size_end = (1 + np.asarray(growth_rate_herd, dtype="float64"))[:, None] * size
        size_avg = (size + size_end) / 2

        s = {c: (1 - pd_[c] - po[c]) * ones for c in TEN_COHORTS}
        o = {c: po[c] * ones for c in TEN_COHORTS}
        g = {c: pg[c] * ones for c in TEN_COHORTS}
        k = {c: (1 - pg[c]) * ones for c in TEN_COHORTS}
        q = {c: pn[c] * ones for c in SIX_COHORTS}
        r = {c: (1 - pn[c]) * ones for c in SIX_COHORTS}

        zero = np.zeros(n)
        off = {c: zero.copy() for c in ("FB", "FJ", "FS", "FA", "FC", "MB", "MJ", "MS", "MA", "MC")}
        nd = {c: zero.copy() for c in SIX_COHORTS}

        def add_na_rm(acc: np.ndarray, v: np.ndarray) -> np.ndarray:
            return np.where(np.isnan(v), acc, acc + v)

        fjg = fsg = fag = mjg = msg = mag = zero
        for t in range(1, 367):
            if t == 1:
                fjf, fsf, faf = xini["FJ"], xini["FS"], xini["FA"]
                mjf, msf, maf = xini["MJ"], xini["MS"], xini["MA"]
            else:
                fjf, fsf, faf = fjg, fsg, fag
                mjf, msf, maf = mjg, msg, mag
            fbf = faf * ff
            mbf = faf * fm
            if t <= 365:
                fb = fbf * s["FB"]
                fj = fjf * s["FJ"]
                fs = fsf * s["FS"]
                fa = faf * s["FA"]
                mb = mbf * s["MB"]
                mj = mjf * s["MJ"]
                ms = msf * s["MS"]
                ma = maf * s["MA"]

                off["FB"] = off["FB"] + o["FB"] * fbf
                off["FJ"] = off["FJ"] + o["FJ"] * fjf
                off["FS"] = off["FS"] + o["FS"] * fsf
                off["FA"] = off["FA"] + o["FA"] * faf
                off["MB"] = off["MB"] + o["MB"] * mbf
                off["MJ"] = off["MJ"] + o["MJ"] * mjf
                off["MS"] = off["MS"] + o["MS"] * msf
                off["MA"] = off["MA"] + o["MA"] * maf

                fjg = fb + k["FJ"] * fj
                tot = g["FJ"] * fj
                nd["FJ"] = add_na_rm(nd["FJ"], q["FJ"] * tot)
                fsg = r["FJ"] * tot + k["FS"] * fs
                tot = g["FS"] * fs
                nd["FS"] = add_na_rm(nd["FS"], q["FS"] * tot)
                fag = r["FS"] * tot + k["FA"] * fa
                tot = g["FA"] * fa
                nd["FA"] = add_na_rm(nd["FA"], q["FA"] * tot)
                off["FC"] = off["FC"] + r["FA"] * tot

                mjg = mb + k["MJ"] * mj
                tot = g["MJ"] * mj
                nd["MJ"] = add_na_rm(nd["MJ"], q["MJ"] * tot)
                msg = r["MJ"] * tot + k["MS"] * ms
                tot = g["MS"] * ms
                nd["MS"] = add_na_rm(nd["MS"], q["MS"] * tot)
                mag = r["MS"] * tot + k["MA"] * ma
                tot = g["MA"] * ma
                nd["MA"] = add_na_rm(nd["MA"], q["MA"] * tot)
                off["MC"] = off["MC"] + r["MA"] * tot

        size_end_exact = np.stack([fbf, fjf, fsf, faf, zero, mbf, mjf, msf, maf, zero], axis=1)
    return {
        "cohort_stock_start": size,
        "cohort_stock_end_projected": size_end,
        "cohort_stock_end_exact_simulated": size_end_exact,
        "cohort_stock_average": size_avg,
        "cohort_stock_annual_nondemographic": np.stack([nd[c] for c in SIX_COHORTS], axis=1),
        "cohort_offtake_heads": np.stack([off[c] for c in TEN_COHORTS], axis=1),
    }


def calc_projected_population_size(
    herd_size_total,
    fecundity_female,
    fecundity_male,
    probability_death,
    probability_offtake,
    probability_growth,
    growth_rate_herd,
    herd_structure,
    cohort_share,
    proportion_nondemographic,
):
    """Simulate one steady-state year (366 daily steps) of a herd.

    The herd starts from ``herd_size_total * herd_structure``; each day births,
    survival, offtake (``p_offtake * stock``), growth transitions and the
    diversion of ``proportion_nondemographic`` of each transition to the
    non-demographic block are applied. Start / projected end / average cohort
    stocks are ``S = herd_size_total * cohort_share``, ``E = (1 + growth_rate_herd) * S``
    and ``(S + E) / 2``.

    Returns
    -------
    dict
        ``cohort_stock_start``, ``cohort_stock_end_projected``, ``cohort_stock_average``,
        ``cohort_stock_annual_nondemographic`` (6 cohorts, heads),
        ``cohort_stock_end_exact_simulated`` (10 cohorts, heads at day 366) and
        ``cohort_offtake_heads`` (10 cohorts, heads/year; culling classes
        count every animal leaving the adult cohort), each a ``pandas.Series``.
    """
    validate_population_size_inputs(
        herd_size_total, fecundity_female, fecundity_male, probability_death, probability_offtake,
        probability_growth, growth_rate_herd, herd_structure, cohort_share, proportion_nondemographic,
    )
    scalar = lambda v: np.array([float(np.asarray(v, dtype="float64").reshape(-1)[0])])  # noqa: E731
    _, share_pos = _names_values(cohort_share)
    res = _projection_kernel(
        scalar(herd_size_total), scalar(fecundity_female), scalar(fecundity_male),
        _pick(probability_death, TEN_COHORTS), _pick(probability_offtake, TEN_COHORTS),
        _pick(probability_growth, TEN_COHORTS), scalar(growth_rate_herd),
        _pick(herd_structure, EIGHT_COHORTS), share_pos.reshape(1, -1),
        _pick(proportion_nondemographic, SIX_COHORTS),
    )
    return {
        "cohort_stock_start": _series(res["cohort_stock_start"][0], SIX_COHORTS),
        "cohort_stock_end_projected": _series(res["cohort_stock_end_projected"][0], SIX_COHORTS),
        "cohort_stock_end_exact_simulated": _series(res["cohort_stock_end_exact_simulated"][0], TEN_COHORTS),
        "cohort_stock_average": _series(res["cohort_stock_average"][0], SIX_COHORTS),
        "cohort_stock_annual_nondemographic": _series(res["cohort_stock_annual_nondemographic"][0], SIX_COHORTS),
        "cohort_offtake_heads": _series(res["cohort_offtake_heads"][0], TEN_COHORTS),
    }


# --------------------------------------------------------------------------
# Offtake summary
# --------------------------------------------------------------------------


def _summary_offtake_kernel(
    start: np.ndarray, end: np.ndarray, avg: np.ndarray, offtake: Mapping[str, np.ndarray],
    simulation_duration: Any,
) -> dict[str, np.ndarray]:
    """Offtake / stock-variation summary; 6-cohort inputs are ``(n, 6)`` positional."""
    with np.errstate(divide="ignore", invalid="ignore"):
        oh = np.stack([
            offtake["FB"] + offtake["FJ"], offtake["FS"] * 1.0, offtake["FA"] + offtake["FC"],
            offtake["MB"] + offtake["MJ"], offtake["MS"] * 1.0, offtake["MA"] + offtake["MC"],
        ], axis=-1)
        rate_start = oh / start
        rate_avg = oh / avg
        sv = end - start
        osv = sv + oh
        return {
            "stock_variation_heads": sv,
            "offtake_heads": oh,
            "offtake_heads_assessment": oh / 365 * as_float(simulation_duration),
            "offtake_rate_to_stock_start": rate_start,
            "offtake_rate_to_stock_average": rate_avg,
            "offtake_stock_variation_heads": osv,
            "offtake_stock_plus_variation_rate_to_stock_start": osv / start,
            "offtake_stock_plus_variation_rate_to_stock_average": osv / avg,
        }


def calc_summary_offtake(
    cohort_stock_start,
    cohort_stock_end_projected,
    cohort_stock_average,
    cohort_offtake_heads,
    simulation_duration,
):
    """Annual offtake, offtake rates and stock variation of a steady-state year.

    The 10-class offtake is collapsed to the 6 cohorts (``FJ = FB + FJ``,
    ``FA = FA + FC``, ``MJ = MB + MJ``, ``MA = MA + MC``); rates are offtake
    (plus stock variation ``end - start``) over the start or average stock, and
    ``offtake_heads_assessment = offtake_heads / 365 * simulation_duration``.

    Returns
    -------
    dict
        Eight ``pandas.Series`` over ``FJ, FS, FA, MJ, MS, MA``: ``stock_variation_heads``,
        ``offtake_heads``, ``offtake_heads_assessment``, ``offtake_rate_to_stock_start``,
        ``offtake_rate_to_stock_average``, ``offtake_stock_variation_heads``,
        ``offtake_stock_plus_variation_rate_to_stock_start`` and
        ``offtake_stock_plus_variation_rate_to_stock_average``.
    """
    validate_offtake_summary_inputs(
        cohort_stock_start, cohort_stock_end_projected, cohort_stock_average,
        cohort_offtake_heads, simulation_duration,
    )
    # 6-cohort vectors combine positionally (R vector arithmetic); offtake by name
    res = _summary_offtake_kernel(
        _names_values(cohort_stock_start)[1],
        _names_values(cohort_stock_end_projected)[1],
        _names_values(cohort_stock_average)[1],
        {c: np.float64(_get(cohort_offtake_heads, c)) for c in TEN_COHORTS},
        simulation_duration,
    )
    return {k: _series(v, SIX_COHORTS) for k, v in res.items()}
