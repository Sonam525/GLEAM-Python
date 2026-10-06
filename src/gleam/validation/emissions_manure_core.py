"""Input validation for the manure management emissions core model.

Port of ``R/validate_emissions_manure_core_model.R``. The R validators are
called row by row on scalars; these versions check whole vectors at once.

Manure management system (MMS) arguments follow the representation documented
in :mod:`gleam.core.emissions_manure`: each MMS is a mapping (``dict`` or
``pandas.Series``, the R named numeric vector) from field name to a scalar or
an array; array elements masked with :mod:`numpy.ma` mark rows for which the
MMS is not supplied (absent from R's ``...``). Element-wise checks (missing
values, ranges, fractions summing to one) only look at the rows where the MMS
is supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._utils import is_na
from ._shared import (
    GleamValidationError,
    abort,
    validate_param_range,
    validate_scalar_numeric,
    validator,
)

#: Order in which R range-checks the MMS fields (``validate_mms_characteristics``).
_MMS_RANGE_ORDER: tuple[str, ...] = (
    "manure_management_system_fraction",
    "nitrogen_fracgas",
    "nitrogen_fracleach",
    "methane_conversion_factor_mcf",
    "ch4_max_producing_capacity_bo",
    "n2o_ef3",
    "n2o_ef4",
    "n2o_ef5",
)


@validator
def validate_calc_volatile_solids(
    ration_intake: Any,
    ration_digestibility_fraction: Any,
    ration_urinary_energy_fraction: Any,
    ration_ash: Any,
) -> None:
    """Validate the inputs of :func:`gleam.calc_volatile_solids` (``parameter_ranges`` bounds)."""
    validate_param_range(ration_intake, "ration_intake")
    validate_param_range(ration_digestibility_fraction, "ration_digestibility_fraction")
    validate_param_range(ration_urinary_energy_fraction, "ration_urinary_energy_fraction")
    validate_param_range(ration_ash, "ration_ash")


@validator
def validate_mms_characteristics(mms_list: Mapping[str, Any], required_names: Sequence[str]) -> None:
    """Validate the MMS arguments of the manure CH4 / N2O functions.

    Mirrors ``validate_mms_characteristics()``: at least one MMS; each MMS is
    numeric with exactly the ``required_names`` fields, has no missing values
    and respects the ``parameter_ranges`` bounds of its fields; the MMS
    fractions sum to 1 (absolute tolerance ``1e-8``). Applied element-wise:
    every row must have at least one supplied MMS and its supplied fractions
    must sum to 1.
    """
    from ..core.emissions_manure import _mms_arrays  # local import: avoids an import cycle

    if len(mms_list) == 0:
        abort("At least one manure management system must be provided.")

    parsed = []
    for mms in mms_list.values():
        if not _is_numeric_mms(mms):
            abort("Each MMS argument must be a numeric vector.")
        names = _mms_names(mms)
        if names is None or set(names) != set(required_names):
            abort(
                "Each MMS must contain exactly these named values: * "
                + _and_collapse(list(required_names))
            )
        vals, absent = _mms_arrays(mms, list(required_names))
        parsed.append((vals, absent))
        # Element-wise checks on the rows where this MMS is supplied.
        own_shape = np.broadcast_shapes(*(a.shape for a in (*vals, absent)))
        present = ~np.broadcast_to(absent, own_shape)
        field_vals = {f: np.broadcast_to(v, own_shape)[present] for f, v in zip(required_names, vals)}
        if any(np.isnan(v).any() for v in field_vals.values()):
            abort("MMS values must not contain missing values.")
        for field in _MMS_RANGE_ORDER:
            if field in field_vals:
                _check_range(field_vals[field], field)

    shape = np.broadcast_shapes(*(a.shape for vals, absent in parsed for a in (*vals, absent)))
    presents = [np.broadcast_to(~absent, shape) for _, absent in parsed]

    # A row without any supplied MMS is R's empty `...` for that cohort.
    if not np.logical_or.reduce(presents).all():
        abort("At least one manure management system must be provided.")

    # The sum of all MMS fractions must equal 1 (R: sum over the cohort's MMS list).
    i_frac = list(required_names).index("manure_management_system_fraction")
    total = np.zeros(shape)
    for (vals, _), present in zip(parsed, presents):
        frac = np.broadcast_to(vals[i_frac], shape)
        total = np.where(present, total + frac, total)
    bad = np.abs(total - 1) > 1e-8
    if bad.any():
        first = total.reshape(-1)[int(np.flatnonzero(bad.reshape(-1))[0])]
        abort(
            "The sum of all MMS fractions must be equal to 1 "
            f"(current sum: {_r_num(first)})."
        )


@validator
def validate_mms_inputs(mms_list: Mapping[str, Any], required_names: Sequence[str], **scalars: Any) -> None:
    """Validate the MMS list and the additional numeric inputs (``validate_mms_inputs()``).

    ``scalars`` are checked, in order, with ``validate_scalar_numeric`` using
    their keyword names as argument names.
    """
    validate_mms_characteristics(mms_list, required_names=required_names)
    for name, value in scalars.items():
        validate_scalar_numeric(value, name)


@validator
def validate_calc_n2o_manure_total(
    n2o_manure_pasture_vol: Any,
    n2o_manure_pasture_leach: Any,
    n2o_manure_burned_vol: Any,
    n2o_manure_burned_leach: Any,
    n2o_manure_other_vol: Any,
    n2o_manure_other_leach: Any,
    n2o_manure_pasture_direct: Any,
    n2o_manure_burned_direct: Any,
    n2o_manure_other_direct: Any,
) -> None:
    """Validate the inputs of :func:`gleam.calc_n2o_manure_total` (numeric, non-missing)."""
    validate_scalar_numeric(n2o_manure_pasture_vol, "n2o_manure_pasture_vol")
    validate_scalar_numeric(n2o_manure_pasture_leach, "n2o_manure_pasture_leach")
    validate_scalar_numeric(n2o_manure_burned_vol, "n2o_manure_burned_vol")
    validate_scalar_numeric(n2o_manure_burned_leach, "n2o_manure_burned_leach")
    validate_scalar_numeric(n2o_manure_other_vol, "n2o_manure_other_vol")
    validate_scalar_numeric(n2o_manure_other_leach, "n2o_manure_other_leach")
    validate_scalar_numeric(n2o_manure_pasture_direct, "n2o_manure_pasture_direct")
    validate_scalar_numeric(n2o_manure_burned_direct, "n2o_manure_burned_direct")
    validate_scalar_numeric(n2o_manure_other_direct, "n2o_manure_other_direct")


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


def _check_range(values: np.ndarray, field: str) -> None:
    """``validate_param_range`` on the supplied values of one MMS field.

    R checks one scalar per cohort, so on failure the first offending value is
    re-checked on its own to give R's scalar message (no ``[i]`` suffix).
    """
    if values.size == 0:
        return
    try:
        validate_param_range(values, field)
    except GleamValidationError:
        for v in values:
            validate_param_range(float(v), field)
        raise


def _is_numeric_value(v: Any) -> bool:
    """Whether one MMS field value is numeric (``NA`` allowed, logicals/strings not)."""
    if v is np.ma.masked or v is None:
        return True
    if isinstance(v, (bool, np.bool_, str, bytes)):
        return False
    if isinstance(v, (int, float, np.number)):
        return True
    if isinstance(v, (pd.Series, pd.Index)):
        v = v.to_numpy()
    if isinstance(v, (list, tuple, np.ndarray)):
        if np.ma.isMaskedArray(v):
            arr = np.ma.getdata(v)
        elif isinstance(v, np.ndarray):
            arr = v
        else:
            arr = np.asarray(v, dtype=object)
        if arr.dtype.kind in "iuf":
            return True
        if arr.dtype == object:
            return all(
                is_na(x) or (isinstance(x, (int, float, np.number)) and not isinstance(x, (bool, np.bool_)))
                for x in arr.ravel()
            )
        return False
    return False


def _is_numeric_mms(mms: Any) -> bool:
    """R ``is.numeric(mms)`` for one MMS argument."""
    if isinstance(mms, pd.Series):
        if mms.dtype.kind in "iuf":
            return True
        return all(_is_numeric_value(v) for v in mms.tolist())
    if isinstance(mms, Mapping):
        return all(_is_numeric_value(v) for v in mms.values())
    if isinstance(mms, (list, tuple, np.ndarray, int, float, np.number)) and not isinstance(mms, (bool, np.bool_)):
        return _is_numeric_value(mms)
    return False


def _mms_names(mms: Any) -> list[str] | None:
    """``names(mms)``: field names of a mapping / Series, ``None`` for unnamed vectors."""
    if isinstance(mms, pd.Series):
        return [str(k) for k in mms.index]
    if isinstance(mms, Mapping):
        return [str(k) for k in mms.keys()]
    return None


def _and_collapse(items: Sequence[str]) -> str:
    """cli-style collapse: ``a``, ``a and b``, ``a, b, and c``."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _r_num(v: float) -> str:
    """``as.character()`` of a double (15 significant digits)."""
    if np.isnan(v):
        return "NA"
    if np.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    return f"{v:.15g}"
