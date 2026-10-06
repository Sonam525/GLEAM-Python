"""Input validation for the weights core model.

Port of ``R/validate_weights_core_model.R``. The R validators are called row
by row on scalars; these versions check whole vectors at once (element ``i``
is checked exactly as R checks row ``i``) and raise
:class:`~gleam.validation._shared.GleamValidationError` on the first
violation. All are no-ops when validation is disabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._utils import all_scalar, as_float, broadcast, is_na, is_scalar
from ._shared import (
    abort,
    validate_animal_species,
    validate_cohort_code,
    validate_param_range,
    validate_scalar_character,
    validate_scalar_numeric,
    validator,
)

# Numeric arguments of calc_cohort_weights(), in the order of R's `args` list.
_COHORT_WEIGHT_NUMERIC_ARGS: tuple[str, ...] = (
    "live_weight_female_adult",
    "live_weight_male_adult",
    "live_weight_at_birth",
    "live_weight_female_at_slaughter",
    "live_weight_male_at_slaughter",
    "live_weight_at_weaning",
    "nondemo_productive_phase_id",
    "live_weight_female_nondemographic_start",
    "live_weight_female_nondemographic_end",
    "live_weight_male_nondemographic_start",
    "live_weight_male_nondemographic_end",
    "phase1_nondemo_fem_duration_days",
    "phase2_nondemo_fem_duration_days",
    "phase1_nondemo_mal_duration_days",
    "phase2_nondemo_mal_duration_days",
)

# R `switch(cohort_short, ...)`: required (non-NA) inputs by cohort, in R order.
_REQUIRED_BY_COHORT: dict[str, tuple[str, ...]] = {
    "FJ": ("live_weight_at_birth", "live_weight_at_weaning", "live_weight_female_adult"),
    "MJ": ("live_weight_at_birth", "live_weight_at_weaning", "live_weight_male_adult"),
    "FS": ("live_weight_at_weaning", "live_weight_female_adult", "live_weight_female_at_slaughter"),
    "MS": ("live_weight_at_weaning", "live_weight_male_adult", "live_weight_male_at_slaughter"),
    "FA": ("live_weight_female_adult",),
    "MA": ("live_weight_male_adult",),
    "FN": (
        "nondemo_productive_phase_id",
        "live_weight_female_nondemographic_start",
        "live_weight_female_nondemographic_end",
        "phase1_nondemo_fem_duration_days",
        "phase2_nondemo_fem_duration_days",
    ),
    "MN": (
        "nondemo_productive_phase_id",
        "live_weight_male_nondemographic_start",
        "live_weight_male_nondemographic_end",
        "phase1_nondemo_mal_duration_days",
        "phase2_nondemo_mal_duration_days",
    ),
}


# --------------------------------------------------------------------------
# Private helpers (vectorised counterparts of R scalar checks)
# --------------------------------------------------------------------------


def _raw_values(x: Any) -> np.ndarray:
    """Raw elements of ``x`` as a 1-d object array (no type coercion)."""
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy(dtype=object)
    return np.atleast_1d(np.asarray(x, dtype=object)).ravel()


def _as_codes(x: Any) -> np.ndarray:
    """Code vector (species / cohort) as an object array with ``None`` for NA.

    Vectorised equivalent of :func:`gleam._utils.as_str` for validated code
    vectors (``as_str`` loops in Python, which dominates run time on large
    tables); values are not converted to ``str``.
    """
    if x is None:
        return np.array(None, dtype=object)
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy(dtype=object)
    else:
        arr = np.asarray(x, dtype=object)
    if arr.ndim == 0:
        return np.array(None if is_na(arr.item()) else arr.item(), dtype=object)
    na = pd.isna(arr)
    if na.any():
        arr = arr.copy()
        arr[na] = None
    return arr


def _codes_in(codes: np.ndarray, values: Sequence[str]) -> np.ndarray:
    """R ``codes %in% values`` for an :func:`_as_codes` array (``NA`` -> ``FALSE``)."""
    arr = np.asarray(codes, dtype=object)
    if arr.ndim == 0:
        v = arr.item()
        return np.array(v is not None and v in set(values))
    return pd.Series(arr.ravel(), dtype=object).isin(list(values)).to_numpy().reshape(arr.shape)


def _unique_values(x: Any) -> Any:
    """Distinct raw values of a vector (scalars unchanged).

    The shared element-wise checks (``validate_scalar_character``,
    ``validate_animal_species``, ``validate_cohort_code``) loop in Python and
    their messages do not depend on the position of the offending value, so
    running them on the distinct values is equivalent and much faster.
    """
    if is_scalar(x):
        return x
    return pd.unique(_raw_values(x))


def _check_lengths(args: dict[str, Any], char_args: Sequence[str] = ()) -> None:
    """Vectorised replacement of R's ``length(x) != 1`` checks.

    R validates scalars, one row at a time. The Python port accepts vectors,
    so every argument must be a scalar (length 1) or a 1-d vector with the
    common length of the other vector arguments.
    """
    sizes = {}
    for name, v in args.items():
        if is_scalar(v):
            sizes[name] = 1
        elif np.ndim(v) == 1:
            sizes[name] = int(np.size(v))
        else:
            sizes[name] = -1
    n = max(sizes.values()) if sizes else 1
    for name, s in sizes.items():
        if s == -1 or (s != 1 and s != n):
            kind = "character value" if name in char_args else "numeric (scalar)"
            abort(
                f"`{name}` must be a single {kind} or a vector with the same length "
                f"as the other inputs."
            )


def _is_numeric_or_na(x: Any) -> bool:
    """R ``is.na(x) || is.numeric(x)`` for every element of ``x``."""
    if x is None:
        return True
    if isinstance(x, (pd.Series, pd.Index)):
        if x.dtype.kind in "iuf":
            return True
    elif not isinstance(x, (str, bytes)) and np.asarray(x).dtype.kind in "iuf":
        return True
    for v in _raw_values(x):
        if is_na(v):
            continue
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)):
            return False
    return True


def _fmt_vals(xs: Sequence[Any]) -> str:
    """cli ``{.val}`` formatting of a vector: ``"a"``, ``"a" and "b"``, ``"a", "b", and "c"``."""
    items = [f'"{v}"' if isinstance(v, str) else str(v) for v in xs]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _validate_range_where(values: np.ndarray, arg_name: str, mask: np.ndarray, scalar: bool) -> None:
    """``validate_param_range`` on the elements selected by ``mask``.

    For all-scalar calls the R message is reproduced exactly; for vectors the
    offending element is labelled with its 1-based position.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    if scalar:
        validate_param_range(values[idx], arg_name)
    else:
        validate_param_range(values[idx], arg_name, labels=idx + 1)


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------


@validator
def validate_cohort_weight_inputs(
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
) -> None:
    """Validate the inputs of :func:`gleam.calc_cohort_weights`.

    * ``species_short`` may be NA; otherwise it must be a valid species code;
    * ``cohort_short`` must be a valid cohort code;
    * numeric inputs must be numeric or NA;
    * the cohort-specific inputs (e.g. birth and weaning weights and the
      adult female weight for ``FJ``) must be non-missing and within their
      ``parameter_ranges`` bounds;
    * for non-chicken species, ``live_weight_at_birth`` must be strictly less
      than ``live_weight_at_weaning`` when both are given.
    """
    numeric = {
        "live_weight_female_adult": live_weight_female_adult,
        "live_weight_male_adult": live_weight_male_adult,
        "live_weight_at_birth": live_weight_at_birth,
        "live_weight_female_at_slaughter": live_weight_female_at_slaughter,
        "live_weight_male_at_slaughter": live_weight_male_at_slaughter,
        "live_weight_at_weaning": live_weight_at_weaning,
        "nondemo_productive_phase_id": nondemo_productive_phase_id,
        "live_weight_female_nondemographic_start": live_weight_female_nondemographic_start,
        "live_weight_female_nondemographic_end": live_weight_female_nondemographic_end,
        "live_weight_male_nondemographic_start": live_weight_male_nondemographic_start,
        "live_weight_male_nondemographic_end": live_weight_male_nondemographic_end,
        "phase1_nondemo_fem_duration_days": phase1_nondemo_fem_duration_days,
        "phase2_nondemo_fem_duration_days": phase2_nondemo_fem_duration_days,
        "phase1_nondemo_mal_duration_days": phase1_nondemo_mal_duration_days,
        "phase2_nondemo_mal_duration_days": phase2_nondemo_mal_duration_days,
    }
    _check_lengths(
        {"species_short": species_short, "cohort_short": cohort_short, **numeric},
        char_args=("species_short", "cohort_short"),
    )

    # species_short: checked only where it is not NA
    sp_given = [v for v in _raw_values(_unique_values(species_short)) if not is_na(v)]
    if sp_given:
        validate_scalar_character(sp_given, "species_short")
        validate_animal_species(sp_given)

    cohort_values = _unique_values(cohort_short)
    validate_scalar_character(cohort_values, "cohort_short")

    for name, val in numeric.items():
        if not _is_numeric_or_na(val):
            abort(f"`{name}` must be a single numeric (scalar). NA is allowed.")

    validate_cohort_code(cohort_values)

    scalar = all_scalar(species_short, cohort_short, *numeric.values())
    arrays = broadcast(
        _as_codes(species_short), _as_codes(cohort_short), *(as_float(v) for v in numeric.values())
    )
    sp, co = (np.atleast_1d(a) for a in arrays[:2])
    vals = {name: np.atleast_1d(a) for name, a in zip(numeric, arrays[2:])}

    # Cohort-specific required inputs (non-NA)
    is_cohort = {c: co == c for c in _REQUIRED_BY_COHORT}
    required = {}
    for name in _COHORT_WEIGHT_NUMERIC_ARGS:
        mask = np.zeros(co.shape, dtype=bool)
        for c, req in _REQUIRED_BY_COHORT.items():
            if name in req:
                mask |= is_cohort[c]
        required[name] = mask
    missing_any = np.zeros(co.shape, dtype=bool)
    for name in _COHORT_WEIGHT_NUMERIC_ARGS:
        missing_any |= required[name] & np.isnan(vals[name])
    if missing_any.any():
        i = int(np.flatnonzero(missing_any)[0])
        cohort = co[i]
        missing = [n for n in _REQUIRED_BY_COHORT[cohort] if np.isnan(vals[n][i])]
        abort(f'Missing required weight inputs for cohort "{cohort}": {_fmt_vals(missing)}')

    # Configured bounds for the cohort-specific required inputs only
    for name in _COHORT_WEIGHT_NUMERIC_ARGS:
        _validate_range_where(vals[name], name, required[name], scalar)

    # Birth weight strictly below weaning weight when both are provided (not CHK)
    birth = vals["live_weight_at_birth"]
    wean = vals["live_weight_at_weaning"]
    with np.errstate(invalid="ignore"):
        bad = (sp != "CHK") & ~np.isnan(birth) & ~np.isnan(wean) & (birth >= wean)
    if bad.any():
        abort("`live_weight_at_birth` must be strictly less than `live_weight_at_weaning`.")


@validator
def validate_avg_weight_inputs(
    cohort_short: Any,
    live_weight_cohort_initial: Any,
    live_weight_cohort_potential_final: Any,
    live_weight_cohort_at_slaughter: Any,
    offtake_rate: Any,
) -> None:
    """Validate the inputs of :func:`gleam.calc_avg_weights`.

    All arguments must be non-missing; ``live_weight_cohort_at_slaughter`` and,
    except for non-demographic cohorts (``FN``, ``MN``), ``offtake_rate`` must
    lie within their ``parameter_ranges`` bounds.
    """
    _check_lengths(
        {
            "cohort_short": cohort_short,
            "live_weight_cohort_initial": live_weight_cohort_initial,
            "live_weight_cohort_potential_final": live_weight_cohort_potential_final,
            "live_weight_cohort_at_slaughter": live_weight_cohort_at_slaughter,
            "offtake_rate": offtake_rate,
        },
        char_args=("cohort_short",),
    )
    cohort_values = _unique_values(cohort_short)
    validate_scalar_character(cohort_values, "cohort_short")
    validate_cohort_code(cohort_values)
    validate_scalar_numeric(live_weight_cohort_initial, "live_weight_cohort_initial")
    validate_scalar_numeric(live_weight_cohort_potential_final, "live_weight_cohort_potential_final")
    validate_scalar_numeric(live_weight_cohort_at_slaughter, "live_weight_cohort_at_slaughter")
    validate_scalar_numeric(offtake_rate, "offtake_rate")

    validate_param_range(live_weight_cohort_at_slaughter, "live_weight_cohort_at_slaughter")
    scalar = all_scalar(cohort_short, offtake_rate)
    co, off = (np.atleast_1d(a) for a in broadcast(_as_codes(cohort_short), as_float(offtake_rate)))
    _validate_range_where(off, "offtake_rate", ~_codes_in(co, ("FN", "MN")), scalar)


@validator
def validate_daily_gain_inputs(
    live_weight_cohort_potential_final: Any,
    live_weight_cohort_initial: Any,
    cohort_duration_days: Any,
) -> None:
    """Validate the inputs of :func:`gleam.calc_daily_weight_gain`.

    All arguments must be non-missing numbers and ``cohort_duration_days``
    must lie within its ``parameter_ranges`` bounds.
    """
    _check_lengths(
        {
            "live_weight_cohort_potential_final": live_weight_cohort_potential_final,
            "live_weight_cohort_initial": live_weight_cohort_initial,
            "cohort_duration_days": cohort_duration_days,
        }
    )
    validate_scalar_numeric(live_weight_cohort_potential_final, "live_weight_cohort_potential_final")
    validate_scalar_numeric(live_weight_cohort_initial, "live_weight_cohort_initial")
    validate_scalar_numeric(cohort_duration_days, "cohort_duration_days")

    validate_param_range(cohort_duration_days, "cohort_duration_days")
