"""Manure management emissions (port of ``R/core_model_emissions_manure.R``).

Volatile solids (VS) excretion, CH4 and direct / indirect N2O emissions from
manure management systems (MMS), reported by MMS group: manure deposited on
pasture (``mms_pasture``), manure burned for fuel (``mms_burned``) and all the
other systems.

All functions are vectorised: numeric arguments accept scalars, lists, numpy
arrays or pandas Series, broadcast against each other, and give the same
result element by element as the scalar R functions.

MMS arguments
-------------
R passes a variable number of MMS through ``...``, one named numeric vector
per system, e.g. ``mms_drylot = c(manure_management_system_fraction = 0.5,
methane_conversion_factor_mcf = 2, ch4_max_producing_capacity_bo = 0.13)``.
In Python each MMS is a keyword argument whose value is a mapping (``dict``
or ``pandas.Series``) from field name to value::

    calc_ch4_manure(volatile_solids=2.0,
                    mms_drylot={"manure_management_system_fraction": 0.5,
                                "methane_conversion_factor_mcf": 2,
                                "ch4_max_producing_capacity_bo": 0.13})

Field values may be scalars or arrays (broadcast against the other inputs),
which evaluates many cohorts at once. Cohorts do not need to share the same
set of systems (R's per-cohort ``mms_list``): an element **masked** with
:mod:`numpy.ma` (``numpy.ma.MaskedArray`` / ``numpy.ma.masked``) in any field
of an MMS means that this MMS is *not supplied* for that row, exactly as if
the argument were absent from R's ``...`` for that cohort (a missing
``mms_pasture`` / ``mms_burned`` gives 0; other systems are left out of the
sum, and a row with no other system gives 0). ``nan`` is R's ``NA``: a
supplied but missing value (rejected by validation, otherwise propagated).

The "other" systems are summed in keyword-argument order, the order of R's
``...``. :func:`gleampy.run_emissions_manure_module` passes them sorted by name,
as R's ``split()`` does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._utils import all_scalar, as_float, finalize, finalize_dict
from ..validation._shared import validate_param_range
from ..validation.emissions_manure_core import (
    validate_calc_n2o_manure_total,
    validate_calc_volatile_solids,
    validate_mms_inputs,
)

#: Reserved MMS name for manure deposited on pasture.
MMS_PASTURE = "mms_pasture"
#: Reserved MMS name for manure burned for fuel.
MMS_BURNED = "mms_burned"

#: Fields of each MMS required by :func:`calc_ch4_manure`.
CH4_MMS_FIELDS: tuple[str, ...] = (
    "manure_management_system_fraction",
    "methane_conversion_factor_mcf",
    "ch4_max_producing_capacity_bo",
)
#: Fields of each MMS required by :func:`calc_n2o_manure_direct`.
N2O_DIRECT_MMS_FIELDS: tuple[str, ...] = ("manure_management_system_fraction", "n2o_ef3")
#: Fields of each MMS required by :func:`calc_n2o_manure_volatilization`.
N2O_VOLATILIZATION_MMS_FIELDS: tuple[str, ...] = (
    "manure_management_system_fraction",
    "n2o_ef4",
    "nitrogen_fracgas",
)
#: Fields of each MMS required by :func:`calc_n2o_manure_leaching`.
N2O_LEACHING_MMS_FIELDS: tuple[str, ...] = (
    "manure_management_system_fraction",
    "n2o_ef5",
    "nitrogen_fracleach",
)

_MISSING: Any = object()


def calc_volatile_solids(
    ration_intake: Any,
    ration_digestibility_fraction: Any,
    ration_urinary_energy_fraction: Any,
    ration_ash: Any,
) -> Any:
    """Daily volatile solids (VS) excretion (kg VS/head/day).

    Algebraically simplified form of IPCC (2006, 2019) Eq. 10.24, taking the
    dry matter intake directly instead of gross energy intake / 18.45::

        volatile_solids = ration_intake
                          * (1 - ration_digestibility_fraction + ration_urinary_energy_fraction)
                          * (1 - ration_ash)

    Parameters
    ----------
    ration_intake : float or array-like
        Daily dry matter intake (kg DM/head/day).
    ration_digestibility_fraction : float or array-like
        Ration digestibility, digestible over gross energy (fraction).
    ration_urinary_energy_fraction : float or array-like
        Fraction of gross energy excreted in urine (fraction).
    ration_ash : float or array-like
        Ash content of the ration (kg ash/kg DM).

    Returns
    -------
    float or numpy.ndarray
        Volatile solids excreted (kg VS/head/day).
    """
    validate_calc_volatile_solids(
        ration_intake, ration_digestibility_fraction, ration_urinary_energy_fraction, ration_ash
    )
    intake = as_float(ration_intake)
    dig = as_float(ration_digestibility_fraction)
    ue = as_float(ration_urinary_energy_fraction)
    ash = as_float(ration_ash)
    with np.errstate(invalid="ignore", over="ignore"):
        volatile_solids = intake * (1 - dig + ue) * (1 - ash)
    return finalize(
        volatile_solids,
        all_scalar(ration_intake, ration_digestibility_fraction, ration_urinary_energy_fraction, ration_ash),
    )


def calc_ch4_manure(
    ratio_m3CH4_to_kgCH4: Any = 0.67,
    volatile_solids: Any = _MISSING,
    **mms: Any,
) -> dict[str, Any]:
    """Daily CH4 emissions from manure management by MMS group (kg CH4/head/day).

    IPCC (2006, 2019) Eq. 10.23 at daily resolution::

        ch4_manure_pasture = volatile_solids * ratio_m3CH4_to_kgCH4 * fraction * (mcf / 100) * bo
        ch4_manure_burned  = (same, for mms_burned)
        ch4_manure_other   = volatile_solids * ratio_m3CH4_to_kgCH4
                             * sum(fraction * mcf * bo over the other MMS) / 100
        ch4_manure_all_noburn = ch4_manure_pasture + ch4_manure_other

    A group without any supplied MMS gives 0.

    Parameters
    ----------
    ratio_m3CH4_to_kgCH4 : float or array-like
        CH4 density used to convert m3 CH4 to kg CH4 (kg/m3), default 0.67.
    volatile_solids : float or array-like
        Volatile solids excreted (kg VS/head/day).
    **mms
        One keyword per MMS (``mms_pasture`` and ``mms_burned`` are reserved
        names); each a mapping with exactly the fields
        ``manure_management_system_fraction`` (fraction of manure handled in
        the system), ``methane_conversion_factor_mcf`` (MCF, %) and
        ``ch4_max_producing_capacity_bo`` (B0, m3 CH4/kg VS). See the module
        docstring for arrays and masked (not supplied) elements. Fractions
        must sum to 1.

    Returns
    -------
    dict
        ``ch4_manure_pasture``, ``ch4_manure_burned``, ``ch4_manure_other``,
        ``ch4_manure_all_noburn`` (kg CH4/head/day).
    """
    if volatile_solids is _MISSING:
        raise TypeError('argument "volatile_solids" is missing, with no default')
    validate_param_range(ratio_m3CH4_to_kgCH4, "ratio_m3CH4_to_kgCH4")
    validate_param_range(volatile_solids, "volatile_solids")
    validate_mms_inputs(
        mms,
        required_names=CH4_MMS_FIELDS,
        ratio_m3CH4_to_kgCH4=ratio_m3CH4_to_kgCH4,
        volatile_solids=volatile_solids,
    )

    ratio = as_float(ratio_m3CH4_to_kgCH4)
    vs = as_float(volatile_solids)
    pasture, burned, other, shape, scalar = _mms_groups(mms, CH4_MMS_FIELDS, ratio, vs)

    with np.errstate(invalid="ignore", over="ignore"):
        ch4_manure_pasture = _single_mms(
            pasture, shape, lambda f, mcf, bo: vs * ratio * f * (mcf / 100) * bo
        )
        ch4_manure_burned = _single_mms(
            burned, shape, lambda f, mcf, bo: vs * ratio * f * (mcf / 100) * bo
        )
        other_sum, any_other = _other_mms_sum(other, shape, lambda f, mcf, bo: f * mcf * bo)
        ch4_manure_other = np.where(any_other, vs * ratio * other_sum / 100, 0.0)
        ch4_manure_all_noburn = ch4_manure_pasture + ch4_manure_other

    return finalize_dict(
        {
            "ch4_manure_pasture": ch4_manure_pasture,
            "ch4_manure_burned": ch4_manure_burned,
            "ch4_manure_other": ch4_manure_other,
            "ch4_manure_all_noburn": ch4_manure_all_noburn,
        },
        scalar,
    )


def calc_n2o_manure_direct(
    ratio_N2ON_to_N2O: Any = 44 / 28,
    nitrogen_excretion: Any = _MISSING,
    **mms: Any,
) -> dict[str, Any]:
    """Daily direct N2O emissions from manure management by MMS group (kg N2O/head/day).

    IPCC (2006, 2019) Tier 2, Eq. 10.25 at daily resolution::

        n2o_manure_pasture_direct = nitrogen_excretion * ratio_N2ON_to_N2O * fraction * n2o_ef3
        n2o_manure_burned_direct  = (same, for mms_burned)
        n2o_manure_other_direct   = nitrogen_excretion * ratio_N2ON_to_N2O
                                    * sum(fraction * n2o_ef3 over the other MMS)
        n2o_manure_all_noburn_direct = pasture + other

    Parameters
    ----------
    ratio_N2ON_to_N2O : float or array-like
        Conversion from kg N2O-N to kg N2O, default 44/28.
    nitrogen_excretion : float or array-like
        Daily nitrogen excretion (kg N/head/day).
    **mms
        One keyword per MMS (see :func:`calc_ch4_manure`), each with exactly
        the fields ``manure_management_system_fraction`` and ``n2o_ef3``
        (kg N2O-N/kg N).

    Returns
    -------
    dict
        ``n2o_manure_pasture_direct``, ``n2o_manure_burned_direct``,
        ``n2o_manure_other_direct``, ``n2o_manure_all_noburn_direct``
        (kg N2O/head/day).
    """
    if nitrogen_excretion is _MISSING:
        raise TypeError('argument "nitrogen_excretion" is missing, with no default')
    validate_mms_inputs(
        mms,
        required_names=N2O_DIRECT_MMS_FIELDS,
        ratio_N2ON_to_N2O=ratio_N2ON_to_N2O,
        nitrogen_excretion=nitrogen_excretion,
    )

    ratio = as_float(ratio_N2ON_to_N2O)
    n_excr = as_float(nitrogen_excretion)
    pasture, burned, other, shape, scalar = _mms_groups(mms, N2O_DIRECT_MMS_FIELDS, ratio, n_excr)

    with np.errstate(invalid="ignore", over="ignore"):
        pasture_direct = _single_mms(pasture, shape, lambda f, ef3: n_excr * ratio * f * ef3)
        burned_direct = _single_mms(burned, shape, lambda f, ef3: n_excr * ratio * f * ef3)
        other_sum, any_other = _other_mms_sum(other, shape, lambda f, ef3: f * ef3)
        other_direct = np.where(any_other, n_excr * ratio * other_sum, 0.0)
        all_noburn_direct = pasture_direct + other_direct

    return finalize_dict(
        {
            "n2o_manure_pasture_direct": pasture_direct,
            "n2o_manure_burned_direct": burned_direct,
            "n2o_manure_other_direct": other_direct,
            "n2o_manure_all_noburn_direct": all_noburn_direct,
        },
        scalar,
    )


def calc_n2o_manure_volatilization(
    ratio_N2ON_to_N2O: Any = 44 / 28,
    nitrogen_excretion: Any = _MISSING,
    **mms: Any,
) -> dict[str, Any]:
    """Daily indirect N2O from volatilised manure N (NH3-N + NOx-N) by MMS group (kg N2O/head/day).

    IPCC Tier 2, Eqs. 10.26 and 10.27 (2006) / 10.28 (2019) at daily resolution::

        n2o_manure_pasture_vol = nitrogen_excretion * ratio_N2ON_to_N2O
                                 * fraction * nitrogen_fracgas * n2o_ef4
        n2o_manure_burned_vol  = (same, for mms_burned)
        n2o_manure_other_vol   = nitrogen_excretion * ratio_N2ON_to_N2O
                                 * sum(fraction * nitrogen_fracgas * n2o_ef4 over the other MMS)
        n2o_manure_all_noburn_vol = pasture + other

    Parameters
    ----------
    ratio_N2ON_to_N2O : float or array-like
        Conversion from kg N2O-N to kg N2O, default 44/28.
    nitrogen_excretion : float or array-like
        Daily nitrogen excretion (kg N/head/day).
    **mms
        One keyword per MMS (see :func:`calc_ch4_manure`), each with exactly
        the fields ``manure_management_system_fraction``, ``n2o_ef4``
        (kg N2O-N/kg volatilised N) and ``nitrogen_fracgas`` (fraction of N
        volatilised).

    Returns
    -------
    dict
        ``n2o_manure_pasture_vol``, ``n2o_manure_burned_vol``,
        ``n2o_manure_other_vol``, ``n2o_manure_all_noburn_vol`` (kg N2O/head/day).
    """
    if nitrogen_excretion is _MISSING:
        raise TypeError('argument "nitrogen_excretion" is missing, with no default')
    validate_mms_inputs(
        mms,
        required_names=N2O_VOLATILIZATION_MMS_FIELDS,
        ratio_N2ON_to_N2O=ratio_N2ON_to_N2O,
        nitrogen_excretion=nitrogen_excretion,
    )

    ratio = as_float(ratio_N2ON_to_N2O)
    n_excr = as_float(nitrogen_excretion)
    pasture, burned, other, shape, scalar = _mms_groups(
        mms, N2O_VOLATILIZATION_MMS_FIELDS, ratio, n_excr
    )

    with np.errstate(invalid="ignore", over="ignore"):
        pasture_vol = _single_mms(
            pasture, shape, lambda f, ef4, fracgas: n_excr * ratio * f * fracgas * ef4
        )
        burned_vol = _single_mms(
            burned, shape, lambda f, ef4, fracgas: n_excr * ratio * f * fracgas * ef4
        )
        other_sum, any_other = _other_mms_sum(other, shape, lambda f, ef4, fracgas: f * fracgas * ef4)
        other_vol = np.where(any_other, n_excr * ratio * other_sum, 0.0)
        all_noburn_vol = pasture_vol + other_vol

    return finalize_dict(
        {
            "n2o_manure_pasture_vol": pasture_vol,
            "n2o_manure_burned_vol": burned_vol,
            "n2o_manure_other_vol": other_vol,
            "n2o_manure_all_noburn_vol": all_noburn_vol,
        },
        scalar,
    )


def calc_n2o_manure_leaching(
    ratio_N2ON_to_N2O: Any = 44 / 28,
    nitrogen_excretion: Any = _MISSING,
    **mms: Any,
) -> dict[str, Any]:
    """Daily indirect N2O from manure N leaching and runoff by MMS group (kg N2O/head/day).

    IPCC Tier 2, Eqs. 10.28 (2006) / 10.27 (2019) and 10.29 at daily resolution::

        n2o_manure_pasture_leach = nitrogen_excretion * ratio_N2ON_to_N2O
                                   * fraction * nitrogen_fracleach * n2o_ef5
        n2o_manure_burned_leach  = (same, for mms_burned)
        n2o_manure_other_leach   = nitrogen_excretion * ratio_N2ON_to_N2O
                                   * sum(fraction * nitrogen_fracleach * n2o_ef5 over the other MMS)
        n2o_manure_all_noburn_leach = pasture + other

    Parameters
    ----------
    ratio_N2ON_to_N2O : float or array-like
        Conversion from kg N2O-N to kg N2O, default 44/28.
    nitrogen_excretion : float or array-like
        Daily nitrogen excretion (kg N/head/day).
    **mms
        One keyword per MMS (see :func:`calc_ch4_manure`), each with exactly
        the fields ``manure_management_system_fraction``, ``n2o_ef5``
        (kg N2O-N/kg N leached) and ``nitrogen_fracleach`` (fraction of N
        leached / lost through runoff).

    Returns
    -------
    dict
        ``n2o_manure_pasture_leach``, ``n2o_manure_burned_leach``,
        ``n2o_manure_other_leach``, ``n2o_manure_all_noburn_leach``
        (kg N2O/head/day).
    """
    if nitrogen_excretion is _MISSING:
        raise TypeError('argument "nitrogen_excretion" is missing, with no default')
    validate_mms_inputs(
        mms,
        required_names=N2O_LEACHING_MMS_FIELDS,
        ratio_N2ON_to_N2O=ratio_N2ON_to_N2O,
        nitrogen_excretion=nitrogen_excretion,
    )

    ratio = as_float(ratio_N2ON_to_N2O)
    n_excr = as_float(nitrogen_excretion)
    pasture, burned, other, shape, scalar = _mms_groups(mms, N2O_LEACHING_MMS_FIELDS, ratio, n_excr)

    with np.errstate(invalid="ignore", over="ignore"):
        pasture_leach = _single_mms(
            pasture, shape, lambda f, ef5, fracleach: n_excr * ratio * f * fracleach * ef5
        )
        burned_leach = _single_mms(
            burned, shape, lambda f, ef5, fracleach: n_excr * ratio * f * fracleach * ef5
        )
        other_sum, any_other = _other_mms_sum(
            other, shape, lambda f, ef5, fracleach: f * fracleach * ef5
        )
        other_leach = np.where(any_other, n_excr * ratio * other_sum, 0.0)
        all_noburn_leach = pasture_leach + other_leach

    return finalize_dict(
        {
            "n2o_manure_pasture_leach": pasture_leach,
            "n2o_manure_burned_leach": burned_leach,
            "n2o_manure_other_leach": other_leach,
            "n2o_manure_all_noburn_leach": all_noburn_leach,
        },
        scalar,
    )


def calc_n2o_manure_total(
    n2o_manure_pasture_vol: Any,
    n2o_manure_pasture_leach: Any,
    n2o_manure_burned_vol: Any,
    n2o_manure_burned_leach: Any,
    n2o_manure_other_vol: Any,
    n2o_manure_other_leach: Any,
    n2o_manure_pasture_direct: Any,
    n2o_manure_burned_direct: Any,
    n2o_manure_other_direct: Any,
) -> dict[str, Any]:
    """Indirect and total N2O emissions from manure by MMS group (kg N2O/head/day).

    For each group ``g`` in pasture, burned, other::

        n2o_manure_<g>_indirect = n2o_manure_<g>_vol + n2o_manure_<g>_leach
        n2o_manure_<g>_total    = n2o_manure_<g>_indirect + n2o_manure_<g>_direct

    Parameters
    ----------
    n2o_manure_pasture_vol, n2o_manure_burned_vol, n2o_manure_other_vol : float or array-like
        Indirect N2O from volatilisation (kg N2O/head/day).
    n2o_manure_pasture_leach, n2o_manure_burned_leach, n2o_manure_other_leach : float or array-like
        Indirect N2O from leaching and runoff (kg N2O/head/day).
    n2o_manure_pasture_direct, n2o_manure_burned_direct, n2o_manure_other_direct : float or array-like
        Direct N2O (kg N2O/head/day).

    Returns
    -------
    dict
        ``n2o_manure_pasture_indirect``, ``n2o_manure_burned_indirect``,
        ``n2o_manure_other_indirect``, ``n2o_manure_pasture_total``,
        ``n2o_manure_burned_total``, ``n2o_manure_other_total`` (kg N2O/head/day).
    """
    validate_calc_n2o_manure_total(
        n2o_manure_pasture_vol=n2o_manure_pasture_vol,
        n2o_manure_pasture_leach=n2o_manure_pasture_leach,
        n2o_manure_burned_vol=n2o_manure_burned_vol,
        n2o_manure_burned_leach=n2o_manure_burned_leach,
        n2o_manure_other_vol=n2o_manure_other_vol,
        n2o_manure_other_leach=n2o_manure_other_leach,
        n2o_manure_pasture_direct=n2o_manure_pasture_direct,
        n2o_manure_burned_direct=n2o_manure_burned_direct,
        n2o_manure_other_direct=n2o_manure_other_direct,
    )
    args = (
        n2o_manure_pasture_vol, n2o_manure_pasture_leach, n2o_manure_burned_vol,
        n2o_manure_burned_leach, n2o_manure_other_vol, n2o_manure_other_leach,
        n2o_manure_pasture_direct, n2o_manure_burned_direct, n2o_manure_other_direct,
    )
    (p_vol, p_leach, b_vol, b_leach, o_vol, o_leach, p_dir, b_dir, o_dir) = (as_float(a) for a in args)

    with np.errstate(invalid="ignore", over="ignore"):
        # indirect components
        pasture_indirect = p_vol + p_leach
        burned_indirect = b_vol + b_leach
        other_indirect = o_vol + o_leach
        # total components
        pasture_total = pasture_indirect + p_dir
        burned_total = burned_indirect + b_dir
        other_total = other_indirect + o_dir

    shape = np.broadcast_shapes(
        *(np.shape(a) for a in (p_vol, p_leach, b_vol, b_leach, o_vol, o_leach, p_dir, b_dir, o_dir))
    )
    out = {
        "n2o_manure_pasture_indirect": pasture_indirect,
        "n2o_manure_burned_indirect": burned_indirect,
        "n2o_manure_other_indirect": other_indirect,
        "n2o_manure_pasture_total": pasture_total,
        "n2o_manure_burned_total": burned_total,
        "n2o_manure_other_total": other_total,
    }
    return finalize_dict({k: _full(v, shape) for k, v in out.items()}, all_scalar(*args))


# --------------------------------------------------------------------------
# Private helpers (MMS argument handling)
# --------------------------------------------------------------------------


def _mms_arrays(mms: Any, fields: Sequence[str]) -> tuple[list[np.ndarray], np.ndarray]:
    """Values of ``fields`` of one MMS argument and its "not supplied" mask.

    Returns the field values as float arrays (``NA`` -> ``nan``; the data
    under masked elements is irrelevant) and a boolean array that is true
    where any field is masked, i.e. where the MMS is not supplied. ``None``
    (R ``NULL``, only reachable with validation off) counts as not supplied.
    """
    if mms is None:
        return [np.array(np.nan) for _ in fields], np.array(True)
    absent = np.array(False)
    vals = []
    for field in fields:
        v = mms[field]  # KeyError for a missing field, like R's `[[` (subscript out of bounds)
        if v is np.ma.masked or np.ma.isMaskedArray(v):
            absent = absent | np.ma.getmaskarray(v)
            v = np.ma.getdata(v)
        vals.append(as_float(v))
    return vals, absent


def _mms_groups(
    mms: Mapping[str, Any], fields: Sequence[str], *scalars: np.ndarray
) -> tuple[Any, Any, list[tuple[list[np.ndarray], np.ndarray]], tuple[int, ...], bool]:
    """Split the MMS arguments into pasture, burned and the other systems (in argument order).

    Also returns the broadcast shape of all inputs and whether every input is a scalar.
    """
    parsed = {name: _mms_arrays(value, fields) for name, value in mms.items()}
    pasture = parsed.get(MMS_PASTURE)
    burned = parsed.get(MMS_BURNED)
    other = [p for name, p in parsed.items() if name not in (MMS_PASTURE, MMS_BURNED)]
    arrays = [*scalars, *(a for vals, absent in parsed.values() for a in (*vals, absent))]
    shape = np.broadcast_shapes(*(np.shape(a) for a in arrays))
    return pasture, burned, other, shape, all_scalar(*arrays)


def _single_mms(group: Any, shape: tuple[int, ...], formula) -> np.ndarray:
    """Emissions of ``mms_pasture`` / ``mms_burned``: ``formula`` where supplied, else 0."""
    if group is None:
        return np.zeros(shape)
    vals, absent = group
    return _full(np.where(absent, 0.0, formula(*vals)), shape)


def _other_mms_sum(
    other: list[tuple[list[np.ndarray], np.ndarray]], shape: tuple[int, ...], term
) -> tuple[np.ndarray, np.ndarray]:
    """R ``sum(other_term)`` over the supplied other systems, in order.

    Returns the sums and a mask of rows with at least one supplied other
    system (R's ``length(mms_other) == 0`` gives 0 otherwise). Terms are
    accumulated left to right in double precision, which reproduces R's
    ``sum()`` bit for bit on the build used for the golden outputs (there R's
    ``long double`` accumulator behaves as a double; on platforms with 80-bit
    accumulation R may differ in the last bit).
    """
    total = np.zeros(shape)
    any_present = np.zeros(shape, dtype=bool)
    for vals, absent in other:
        present = ~np.broadcast_to(absent, shape)
        total = np.where(present, total + term(*vals), total)
        any_present |= present
    return total, any_present


def _full(x: Any, shape: tuple[int, ...]) -> np.ndarray:
    """``x`` broadcast to ``shape`` as a new (writeable) float array."""
    out = np.empty(shape)
    out[...] = x
    return out
