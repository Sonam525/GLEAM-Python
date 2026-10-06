"""Herd-level aggregation (port of ``R/core_model_aggregation.R``).

The functions are vectorised: they accept scalars, lists, numpy arrays or
pandas Series (broadcast against each other) and reproduce, element by
element, the scalar branches of the R code (evaluated row by row with
``by = .I`` in :func:`gleam.run_aggregation_module`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._utils import all_scalar, as_float, as_str, broadcast, finalize, finalize_dict, isin
from ..validation.aggregation_core import (
    validate_allocated_emissions_inputs,
    validate_co2eq_inputs,
    validate_totals_by_cohort_inputs,
)

#: GWP-100 factors (kg CO2eq/kg gas) of each IPCC assessment report.
GWP_FACTORS: dict[str, dict[str, float]] = {
    "AR6": {"CH4": 27.0, "N2O": 273.0, "CO2": 1.0},
    "AR5_excluding_carbon_feedback": {"CH4": 28.0, "N2O": 265.0, "CO2": 1.0},
    "AR5_including_carbon_feedback": {"CH4": 34.0, "N2O": 298.0, "CO2": 1.0},
    "AR4": {"CH4": 25.0, "N2O": 298.0, "CO2": 1.0},
}


def calc_cohort_totals(
    value: Any,
    cohort_stock_size: Any,
    ration_intake: Any,
    feed_emissions_list: Sequence[dict[str, str]],
    simulation_duration: Any,
    variable_name: Any,
    variable_type: Any,
) -> Any:
    """Total of a variable per cohort over the assessment period.

    * ``Production`` variables (already per cohort and period): ``value``;
    * ``Emissions`` variables listed in ``feed_emissions_list`` (g/kg DM):
      ``value * ration_intake * cohort_stock_size * simulation_duration / 1000``;
    * other emissions, ``Feed`` and ``NitrogenBalance`` variables
      (per head per day): ``value * cohort_stock_size * simulation_duration``.

    Parameters
    ----------
    value : float or array-like
        Variable value ((unit)/head/day, or (unit)/cohort/period for production).
    cohort_stock_size : float or array-like
        Average cohort population (heads).
    ration_intake : float or array-like
        Average daily dry matter intake (kg DM/head/day).
    feed_emissions_list : sequence of dict
        Feed-related emission sources, each with an ``"emissions_source"``
        key (e.g. :data:`gleam.constants.GLEAM_FEED_EMISSIONS_META`).
    simulation_duration : float or array-like
        Length of the assessment period (days).
    variable_name : str or array-like
        Variable name.
    variable_type : str or array-like
        ``"Production"``, ``"Emissions"``, ``"Feed"`` or ``"NitrogenBalance"``.

    Returns
    -------
    float or numpy.ndarray
        Variable total ((unit)/cohort/assessment period).
    """
    validate_totals_by_cohort_inputs(
        value, cohort_stock_size, ration_intake, simulation_duration, variable_name, variable_type
    )
    scalar = all_scalar(value, cohort_stock_size, ration_intake, simulation_duration, variable_name, variable_type)

    feed_emissions_sources = [m["emissions_source"] for m in feed_emissions_list]

    val, stock, intake, duration, name, vtype = broadcast(
        as_float(value),
        as_float(cohort_stock_size),
        as_float(ration_intake),
        as_float(simulation_duration),
        as_str(variable_name),
        as_str(variable_type),
    )
    production = vtype == "Production"
    feed_emissions = (vtype == "Emissions") & isin(name, feed_emissions_sources)
    with np.errstate(invalid="ignore", over="ignore"):
        total_feed_emissions = val * intake * stock * duration / 1000
        total_other = val * stock * duration
        value_total = np.where(production, val, np.where(feed_emissions, total_feed_emissions, total_other))
    return finalize(value_total, scalar)


def calc_allocated_emissions(value: Any, allocation_share: Any) -> Any:
    """Emissions allocated to a commodity: ``value * allocation_share``.

    Parameters
    ----------
    value : float or array-like
        Total herd-level emissions of a source before allocation (kg gas).
    allocation_share : float or array-like
        Allocation share of the commodity for that source (fraction).

    Returns
    -------
    float or numpy.ndarray
        Allocated emissions (kg gas).
    """
    validate_allocated_emissions_inputs(value, allocation_share)
    scalar = all_scalar(value, allocation_share)
    v, share = broadcast(as_float(value), as_float(allocation_share))
    with np.errstate(invalid="ignore", over="ignore"):
        value_allocated = v * share
    return finalize(value_allocated, scalar)


def calc_co2eq(gas: Any, value_allocated: Any, global_warming_potential_set: str) -> dict[str, Any]:
    """Convert CH4, N2O and CO2 emissions to CO2-equivalents with GWP-100 factors.

    ``value_co2eq = value_allocated * gwp`` with ``gwp`` from the selected
    IPCC report: ``AR6`` (CH4 27, N2O 273), ``AR5_excluding_carbon_feedback``
    (28, 265), ``AR5_including_carbon_feedback`` (34, 298), ``AR4`` (25, 298);
    CO2 is always 1.

    Parameters
    ----------
    gas : str or array-like
        ``"CH4"``, ``"N2O"`` or ``"CO2"``.
    value_allocated : float or array-like
        Allocated emissions (kg gas).
    global_warming_potential_set : str
        GWP-100 set (see above).

    Returns
    -------
    dict
        ``value_co2eq`` (kg CO2eq) and ``gwp`` (kg CO2eq/kg gas).
    """
    validate_co2eq_inputs(gas, value_allocated, global_warming_potential_set)
    scalar = all_scalar(gas, value_allocated)

    gwp_factors = GWP_FACTORS.get(global_warming_potential_set, {})
    g, v = broadcast(as_str(gas), as_float(value_allocated))
    # gwp_factors[gas]: NA for an unknown gas
    gwp = np.array([gwp_factors.get(x, np.nan) if x is not None else np.nan for x in g.ravel()], dtype="float64")
    gwp = gwp.reshape(g.shape)
    with np.errstate(invalid="ignore", over="ignore"):
        value_co2eq = v * gwp
    return finalize_dict({"value_co2eq": value_co2eq, "gwp": gwp}, scalar)
