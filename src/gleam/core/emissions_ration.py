"""Emissions from feed production (port of ``R/core_model_emissions_ration.R``).

Each function returns the contribution of one feed component to a diet-level
average emission factor, ``feed_ration_fraction * <feed emission factor>``
(g gas/kg DM); the run module sums the contributions over the feed components
of a cohort's ration. All functions are vectorised: they accept scalars,
lists, numpy arrays or pandas Series (broadcast against each other). A
missing (NA) emission factor gives an NA contribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._utils import all_scalar, as_float, finalize
from ..validation.emissions_ration_core import (
    validate_ch4_ration_rice_inputs,
    validate_co2_ration_crop_activities_inputs,
    validate_co2_ration_fertilizer_inputs,
    validate_co2_ration_luc_nopeat_inputs,
    validate_co2_ration_luc_peat_inputs,
    validate_co2_ration_pesticides_inputs,
    validate_n2o_ration_crop_residues_inputs,
    validate_n2o_ration_fertilizer_inputs,
    validate_n2o_ration_manure_applied_inputs,
)


def _weighted(feed_ration_fraction: Any, emission_factor: Any) -> Any:
    """``feed_ration_fraction * emission_factor`` (scalar in, scalar out)."""
    scalar = all_scalar(feed_ration_fraction, emission_factor)
    with np.errstate(invalid="ignore", over="ignore"):
        out = as_float(feed_ration_fraction) * as_float(emission_factor)
    return finalize(out, scalar)


def calc_co2_ration_fertilizer(feed_ration_fraction: Any, co2_feed_fertilizer: Any) -> Any:
    """Contribution of a feed component to CO2 from fertilizer manufacture (g CO2/kg DM).

    ``co2_ration_fertilizer = feed_ration_fraction * co2_feed_fertilizer``

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    co2_feed_fertilizer : float or array-like
        CO2 emission factor of the feed from fertilizer manufacture
        (g CO2/kg DM); may be NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to the diet-level emission factor (g CO2/kg DM).
    """
    validate_co2_ration_fertilizer_inputs(feed_ration_fraction, co2_feed_fertilizer)
    return _weighted(feed_ration_fraction, co2_feed_fertilizer)


def calc_co2_ration_pesticides(feed_ration_fraction: Any, co2_feed_pesticides: Any) -> Any:
    """Contribution of a feed component to CO2 from pesticide manufacture (g CO2/kg DM).

    ``co2_ration_pesticides = feed_ration_fraction * co2_feed_pesticides``

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    co2_feed_pesticides : float or array-like
        CO2 emission factor of the feed from pesticide manufacture
        (g CO2/kg DM); may be NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to the diet-level emission factor (g CO2/kg DM).
    """
    validate_co2_ration_pesticides_inputs(feed_ration_fraction, co2_feed_pesticides)
    return _weighted(feed_ration_fraction, co2_feed_pesticides)


def calc_co2_ration_crop_activities(feed_ration_fraction: Any, co2_feed_crop_activities: Any) -> Any:
    """Contribution of a feed component to CO2 from on-field crop activities (g CO2/kg DM).

    ``co2_ration_crop_activities = feed_ration_fraction * co2_feed_crop_activities``
    (energy use for tillage, machinery operations, ...).

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    co2_feed_crop_activities : float or array-like
        CO2 emission factor of the feed from on-field activities
        (per kg DM); may be NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to the diet-level emission factor (g CO2/kg DM).
    """
    validate_co2_ration_crop_activities_inputs(feed_ration_fraction, co2_feed_crop_activities)
    return _weighted(feed_ration_fraction, co2_feed_crop_activities)


def calc_co2_ration_luc_nopeat(feed_ration_fraction: Any, co2_feed_luc_nopeat: Any) -> Any:
    """Contribution of a feed component to CO2 from land-use change, excluding peat (g CO2/kg DM).

    ``co2_ration_luc_nopeat = feed_ration_fraction * co2_feed_luc_nopeat``

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    co2_feed_luc_nopeat : float or array-like
        CO2 emission factor of the feed from land-use change, excluding
        peatland drainage (g CO2/kg DM); may be negative or NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to the diet-level emission factor (g CO2/kg DM).
    """
    validate_co2_ration_luc_nopeat_inputs(feed_ration_fraction, co2_feed_luc_nopeat)
    return _weighted(feed_ration_fraction, co2_feed_luc_nopeat)


def calc_co2_ration_luc_peat(feed_ration_fraction: Any, co2_feed_luc_peat: Any) -> Any:
    """Contribution of a feed component to CO2 from peatland drainage (g CO2/kg DM).

    ``co2_ration_luc_peat = feed_ration_fraction * co2_feed_luc_peat``

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    co2_feed_luc_peat : float or array-like
        CO2 emission factor of the feed from peatland drainage
        (g CO2/kg DM); may be negative or NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to the diet-level emission factor (g CO2/kg DM).
    """
    validate_co2_ration_luc_peat_inputs(feed_ration_fraction, co2_feed_luc_peat)
    return _weighted(feed_ration_fraction, co2_feed_luc_peat)


def calc_n2o_ration_fertilizer(feed_ration_fraction: Any, n2o_feed_fertilizer: Any) -> Any:
    """Contribution of a feed component to N2O from synthetic fertilizer use (g N2O/kg DM).

    ``n2o_ration_fertilizer = feed_ration_fraction * n2o_feed_fertilizer``

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    n2o_feed_fertilizer : float or array-like
        N2O emission factor of the feed from fertilizer use (g N2O/kg DM);
        may be NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to the diet-level emission factor (g N2O/kg DM).
    """
    validate_n2o_ration_fertilizer_inputs(feed_ration_fraction, n2o_feed_fertilizer)
    return _weighted(feed_ration_fraction, n2o_feed_fertilizer)


def calc_n2o_ration_manure(feed_ration_fraction: Any, n2o_feed_manure_applied: Any) -> Any:
    """Contribution of a feed component to N2O from manure applied/deposited on soil (g N2O/kg DM).

    ``n2o_ration_manure_applied = feed_ration_fraction * n2o_feed_manure_applied``

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    n2o_feed_manure_applied : float or array-like
        N2O emission factor of the feed from manure applied to or deposited
        on soil (g N2O/kg DM); may be NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to the diet-level emission factor (g N2O/kg DM).
    """
    validate_n2o_ration_manure_applied_inputs(feed_ration_fraction, n2o_feed_manure_applied)
    return _weighted(feed_ration_fraction, n2o_feed_manure_applied)


def calc_n2o_ration_crop_residues(feed_ration_fraction: Any, n2o_feed_crop_residues: Any) -> Any:
    """Contribution of a feed component to N2O from crop residue decomposition (g N2O/kg DM).

    ``n2o_ration_crop_residues = feed_ration_fraction * n2o_feed_crop_residues``

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    n2o_feed_crop_residues : float or array-like
        N2O emission factor of the feed from crop residues (g N2O/kg DM);
        may be NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to the diet-level emission factor (g N2O/kg DM).
    """
    validate_n2o_ration_crop_residues_inputs(feed_ration_fraction, n2o_feed_crop_residues)
    return _weighted(feed_ration_fraction, n2o_feed_crop_residues)


def calc_ch4_ration_rice(feed_ration_fraction: Any, ch4_feed_rice: Any) -> Any:
    """Contribution of a feed component to CH4 from rice cultivation (g CH4/kg DM).

    ``ch4_ration_rice = feed_ration_fraction * ch4_feed_rice``

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    ch4_feed_rice : float or array-like
        CH4 emission factor of the feed from rice cultivation (g CH4/kg DM);
        may be NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to the diet-level emission factor (g CH4/kg DM).
    """
    validate_ch4_ration_rice_inputs(feed_ration_fraction, ch4_feed_rice)
    return _weighted(feed_ration_fraction, ch4_feed_rice)
