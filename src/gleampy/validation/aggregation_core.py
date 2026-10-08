"""Input validation for the aggregation core model.

Port of ``R/validate_aggregation_core_model.R``. These R validators already
work on vectors; the Python versions accept scalars, lists, numpy arrays or
pandas Series.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import as_float, is_na, is_scalar
from ._shared import abort, validate_scalar_character, validator

#: Valid ``variable_type`` values of :func:`gleampy.calc_cohort_totals`.
VALID_VARIABLE_TYPES: tuple[str, ...] = ("Production", "Emissions", "Feed", "NitrogenBalance")

#: Valid gases of :func:`gleampy.calc_co2eq`.
VALID_GASES: tuple[str, ...] = ("CH4", "N2O", "CO2")


# --------------------------------------------------------------------------
# Private helpers (R type predicates and cli formatting)
# --------------------------------------------------------------------------


def _values(x: Any) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy()
    return x if isinstance(x, np.ndarray) else np.asarray(x)


def _distinct(x: Any) -> np.ndarray:
    """Distinct values of ``x`` in order of first appearance, as a flat array.

    R checks these predicates once per row (``by = .I``) or on ``unique()``
    values; checking the distinct values is equivalent and avoids a Python
    loop over every row of the long tables. The array has the dtype that
    ``_values(x)`` would have, so dtype-based answers do not change.
    ``pd.unique`` merges only values that compare equal (``1``, ``1.0`` and
    ``True``; ``"a"`` and ``numpy.str_("a")``), which never changes whether a
    value is a string or NA. Unhashable values fall back to all values.

    Parameters
    ----------
    x : scalar, sequence, numpy.ndarray, pandas.Series or pandas.Index
        Values.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of the distinct values.
    """
    if isinstance(x, (pd.Series, pd.Index)):
        try:
            return np.asarray(pd.unique(x)).ravel()
        except TypeError:
            return x.to_numpy().ravel()
    vals = np.atleast_1d(_values(x)).ravel()
    if vals.dtype == object:
        try:
            return pd.unique(vals)
        except TypeError:
            return vals
    return vals


def r_length(x: Any) -> int:
    """R ``length()``: 1 for scalars, number of elements otherwise."""
    if x is None:
        return 0
    return 1 if is_scalar(x) else int(np.size(_values(x)))


#: Types of the missing-value objects recognised by :func:`gleampy._utils.is_na`.
_NA_TYPES: tuple[type, ...] = (type(None), type(pd.NA), type(pd.NaT))


def _is_number_type(t: type) -> bool:
    """Element type counted as numeric by R's ``is.numeric()`` (not logical)."""
    return issubclass(t, (int, float, np.number)) and not issubclass(t, (bool, np.bool_))


def is_numeric(x: Any) -> bool:
    """R ``is.numeric()``: numbers (NA allowed); logicals and strings are not numeric.

    For object arrays the answer depends only on the set of element types
    (``None``, ``pd.NA`` and ``NaT`` count as NA, every ``float`` is numeric
    whether or not it is NaN), so it is computed from that set instead of a
    Python loop over the values. A non-empty array holding only ``None`` /
    ``pd.NA`` is an all-NA logical vector (what ``fread`` makes of an empty
    column) and is not numeric.

    Parameters
    ----------
    x : scalar, sequence, numpy.ndarray, pandas.Series or pandas.Index
        Values to test.

    Returns
    -------
    bool
        Whether ``x`` is numeric.
    """
    if x is None or isinstance(x, (bool, np.bool_, str, bytes)):
        return False
    if isinstance(x, (pd.Series, pd.Index)):
        if x.dtype.kind in "iuf":
            return True
        if x.dtype.kind == "b":
            return False
    vals = _values(x)
    if vals.dtype.kind in "iuf":
        return True
    if vals.dtype == object:
        flat = vals.ravel()
        types = set(map(type, flat))
        if flat.size and types <= {type(None), type(pd.NA)}:
            return False  # an all-NA logical vector
        return all(t in _NA_TYPES or _is_number_type(t) for t in types)
    return False


def is_character(x: Any) -> bool:
    """R ``is.character()``: strings (NA allowed).

    Parameters
    ----------
    x : scalar, sequence, numpy.ndarray, pandas.Series or pandas.Index
        Values to test. Only the distinct values are examined.

    Returns
    -------
    bool
        Whether ``x`` is character.
    """
    if isinstance(x, str):
        return True
    if x is None or is_scalar(x) and not isinstance(x, np.ndarray):
        return False
    if isinstance(x, np.ndarray) and x.dtype.kind in "US":
        return True
    vals = _distinct(x)
    if vals.dtype.kind in "US":
        return True
    if vals.dtype == object:
        items = list(vals)
        if items and not any(isinstance(v, str) for v in items):
            return False
        return all(is_na(v) or isinstance(v, str) for v in items)
    return False


def inline_values(values: Iterable[Any]) -> str:
    """``cli::format_inline("{x}")`` of a vector: ``a, b, and c``."""
    vals = ["NA" if is_na(v) else str(v) for v in values]
    if len(vals) <= 1:
        return "".join(vals)
    if len(vals) == 2:
        return f"{vals[0]} and {vals[1]}"
    return ", ".join(vals[:-1]) + f", and {vals[-1]}"


def _unique(values: Iterable[Any]) -> list[Any]:
    """R ``unique()`` (first-appearance order, all NA values as one).

    Callers pass :func:`_distinct` values, so the loop only sees a few
    elements; the result is the same as on all the values.
    """
    out: list[Any] = []
    seen: set[Any] = set()
    has_na = False
    for v in values:
        if is_na(v):
            if not has_na:
                out.append(None)
                has_na = True
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _abort_if_any(violation: np.ndarray, missing: np.ndarray, message: str, arg_name: str) -> None:
    """R ``if (any(cond)) cli_abort(message)`` where ``cond`` may contain ``NA``.

    ``any()`` is ``TRUE`` if any element is ``TRUE``; otherwise an ``NA``
    makes the ``if`` fail ("missing value where TRUE/FALSE needed").
    """
    if violation.any():
        abort(message)
    if missing.any():
        abort(f"missing value where TRUE/FALSE needed: `{arg_name}` contains NA.")


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------


@validator
def validate_totals_by_cohort_inputs(
    value: Any,
    cohort_stock_size: Any,
    ration_intake: Any,
    simulation_duration: Any,
    variable_name: Any,
    variable_type: Any,
) -> None:
    """Validate inputs of :func:`gleampy.calc_cohort_totals`.

    Numeric inputs must be numeric, names and types character,
    ``cohort_stock_size`` and ``simulation_duration`` positive and
    ``variable_type`` one of ``Production``, ``Emissions``, ``Feed``,
    ``NitrogenBalance``. The character and ``unique()`` checks look at the
    distinct values only, so the cost does not grow with a Python loop over
    every row of the long table.

    Parameters
    ----------
    value : float or array-like
        Variable value ((unit)/head/day, or (unit)/cohort/period for production).
    cohort_stock_size : float or array-like
        Average cohort population (heads).
    ration_intake : float or array-like
        Average daily dry matter intake (kg DM/head/day).
    simulation_duration : float or array-like
        Length of the assessment period (days).
    variable_name : str or array-like
        Variable name.
    variable_type : str or array-like
        Variable group.
    """
    for x, name in (
        (value, "value"),
        (cohort_stock_size, "cohort_stock_size"),
        (ration_intake, "ration_intake"),
        (simulation_duration, "simulation_duration"),
    ):
        if not is_numeric(x):
            abort(f"`{name}` must be numeric.")

    if not is_character(variable_name):
        abort("`variable_name` must be character.")
    if not is_character(variable_type):
        abort("`variable_type` must be character.")

    for x, name in ((cohort_stock_size, "cohort_stock_size"), (simulation_duration, "simulation_duration")):
        v = np.atleast_1d(as_float(_values(x)))
        with np.errstate(invalid="ignore"):
            _abort_if_any(v <= 0, np.isnan(v), f"`{name}` must be positive.", name)

    invalid = [t for t in _unique(_distinct(variable_type)) if t not in VALID_VARIABLE_TYPES]
    if invalid:
        abort(
            f"`variable_type` must be one of: {inline_values(VALID_VARIABLE_TYPES)}. "
            f"Found invalid values: {inline_values(invalid)}"
        )


@validator
def validate_allocated_emissions_inputs(value: Any, allocation_share: Any) -> None:
    """Validate inputs of :func:`gleampy.calc_allocated_emissions`.

    Same length, numeric, and ``allocation_share`` between 0 and 1.
    """
    if r_length(value) != r_length(allocation_share):
        abort("`value` and `allocation_share` must have the same length.")
    if not is_numeric(value):
        abort("`value` must be numeric.")
    if not is_numeric(allocation_share):
        abort("`allocation_share` must be numeric.")

    share = np.atleast_1d(as_float(_values(allocation_share)))
    with np.errstate(invalid="ignore"):
        _abort_if_any(
            (share < 0) | (share > 1), np.isnan(share),
            "`allocation_share` must be between 0 and 1.", "allocation_share",
        )


@validator
def validate_co2eq_inputs(gas: Any, value_allocated: Any, global_warming_potential_set: Any) -> None:
    """Validate inputs of :func:`gleampy.calc_co2eq`.

    ``global_warming_potential_set`` must be one of ``AR6``,
    ``AR5_excluding_carbon_feedback``, ``AR5_including_carbon_feedback``,
    ``AR4``; ``gas`` (``CH4``, ``N2O``, ``CO2``) and ``value_allocated`` must
    have the same length.

    Parameters
    ----------
    gas : str or array-like
        Gas of each value.
    value_allocated : float or array-like
        Allocated emissions (kg gas).
    global_warming_potential_set : str
        GWP-100 set.
    """
    # Validate gwp is scalar character
    if not isinstance(global_warming_potential_set, str):
        abort("`global_warming_potential_set` must be a single character value.")
    validate_scalar_character(global_warming_potential_set, "global_warming_potential_set")

    valid_gwp = K.GLOBAL_WARMING_POTENTIAL_SETS
    if global_warming_potential_set not in valid_gwp:
        abort(f"`global_warming_potential_set` must be one of: {inline_values(valid_gwp)}")

    if r_length(gas) != r_length(value_allocated):
        abort("`gas` and `value_allocated` must have the same length.")

    if not is_character(gas):
        abort("`gas` must be character.")

    invalid = [g for g in _unique(_distinct(gas)) if g not in VALID_GASES]
    if invalid:
        abort(
            f"`gas` must be one of: {inline_values(VALID_GASES)}. "
            f"Found invalid values: {inline_values(invalid)}"
        )

    if not is_numeric(value_allocated):
        abort("`value_allocated` must be numeric.")
