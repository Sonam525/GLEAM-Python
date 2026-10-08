"""Aggregation module (port of ``R/run_aggregation_module.R``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import Progress, as_float, as_str, copy_frame, isin, merge_dt
from ..core.aggregation import calc_allocated_emissions, calc_co2eq, calc_cohort_totals
from ..core.allocation import _aggregate_groups, _group_codes
from ..validation._shared import abort, setup_validation
from ..validation.aggregation_run import _val, validate_run_aggregation_module_inputs
from ..validation.production_run import check_single_value

_ID_VARS: tuple[str, ...] = ("herd_id", "species_short", "cohort_short", "cohort_stock_size", "ration_intake")


def run_aggregation_module(
    cohort_level_data: pd.DataFrame,
    allocation_herd_long: pd.DataFrame,
    simulation_duration: float = 365,
    global_warming_potential_set: str = "AR6",
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the aggregation module (final step of the GLEAM pipeline).

    Scales cohort-level variables to totals over the assessment period, sums
    them to herd level, allocates herd emissions to commodities and converts
    CH4 and N2O to CO2-equivalents with GWP-100 factors.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table with ``herd_id``, ``species_short``,
        ``cohort_short``, ``cohort_stock_size`` (heads), ``ration_intake``
        (kg DM/head/day) and any of the feed, nitrogen-balance
        (kg N/head/day), production (per cohort and period) and emission
        variables of :mod:`gleampy.constants` (emissions in kg/head/day; feed
        emissions in g/kg DM).
    allocation_herd_long : pandas.DataFrame
        Long-format allocation shares (``herd_id``, ``species_short``,
        ``variable_name``, ``commodity_name``, ``allocation_share``), e.g.
        ``run_allocation_module(...)["allocation_long"]``.
    simulation_duration : float
        Length of the assessment period (days); a single positive value (a
        vector is rejected, as in R, even with ``validate_inputs=False``).
    global_warming_potential_set : str
        ``"AR6"`` (CH4 27, N2O 273), ``"AR5_excluding_carbon_feedback"``
        (28, 265), ``"AR5_including_carbon_feedback"`` (34, 298) or
        ``"AR4"`` (25, 298).
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    dict
        ``results_emissions`` (allocated emissions in kg gas and kg CO2eq,
        with gas, GWP and label), ``results_feed``, ``results_production``
        and ``results_nitrogen`` (herd totals over the assessment period with
        their labels and units).
    """
    with setup_validation(validate_inputs):
        # --- Input validation
        if not isinstance(cohort_level_data, pd.DataFrame):
            cohort_level_data = copy_frame(cohort_level_data)
        validate_run_aggregation_module_inputs(
            cohort_level_data=cohort_level_data,
            allocation_herd_long=allocation_herd_long,
            simulation_duration=simulation_duration,
            global_warming_potential_set=global_warming_potential_set,
        )
        # Also without validation: R's by-row assignments fail on a vector, and
        # a vector here would silently give each long-table row its own value.
        check_single_value(simulation_duration, "simulation_duration")

        progress = Progress(show_indicator)
        progress.status("Aggregating results, please wait...")

        # --- Step 1: variable groups
        feed_vars = [m["feed_source"] for m in K.GLEAM_FEED_META]
        nitrogen_balance_vars = [m["nitrogen_balance_source"] for m in K.GLEAM_NITROGEN_BALANCE_META]
        production_vars = [m["production_source"] for m in K.GLEAM_PRODUCTION_META]
        feed_emissions_list = K.GLEAM_FEED_EMISSIONS_META
        emissions_vars = [m["emissions_source"] for m in K.GLEAM_EMISSIONS_META]

        all_vars = list(dict.fromkeys(feed_vars + nitrogen_balance_vars + production_vars + emissions_vars))
        available_vars = [v for v in all_vars if v in cohort_level_data.columns]
        if not available_vars:
            abort(
                "No recognized variables found in `cohort_level_data`. Expected variables include: "
                + ", ".join(f'"{v}"' for v in all_vars)
            )

        # --- Step 2: reshape to long format
        data_cohort_long = _melt(
            cohort_level_data,
            id_vars=list(_ID_VARS),
            measure_vars=available_vars,
            variable_name="variable_name",
            value_name="value",
        )

        # --- Step 3: classify variables by type. R's fcase runs on every row,
        # but the type depends only on variable_name, which is constant within
        # each melt block of n rows: classify the measure variables once.
        n = len(cohort_level_data)
        name = np.array(available_vars, dtype=object)
        var_type = np.select(
            [
                isin(name, feed_vars),
                isin(name, nitrogen_balance_vars),
                isin(name, production_vars),
                isin(name, emissions_vars),
            ],
            ["Feed", "NitrogenBalance", "Production", "Emissions"],
            default="Other",
        ).astype(object)
        data_cohort_long["variable_type"] = np.repeat(var_type, n)

        # --- Step 4: totals by cohort
        data_cohort_long["value_total"] = calc_cohort_totals(
            value=data_cohort_long["value"],
            cohort_stock_size=data_cohort_long["cohort_stock_size"],
            ration_intake=data_cohort_long["ration_intake"],
            feed_emissions_list=feed_emissions_list,
            simulation_duration=simulation_duration,
            variable_name=data_cohort_long["variable_name"],
            variable_type=data_cohort_long["variable_type"],
        )

        # --- Step 5: aggregate from cohort to herd level
        data_herd_long = _cohort_to_herd_long(cohort_level_data, data_cohort_long, len(available_vars))

        # --- Step 6: subsets by variable type
        vtype = data_herd_long["variable_type"].to_numpy(dtype=object)
        data_herd_long_production = data_herd_long.loc[vtype == "Production"].reset_index(drop=True)
        data_herd_long_nitrogen = data_herd_long.loc[vtype == "NitrogenBalance"].reset_index(drop=True)
        data_herd_long_feed = data_herd_long.loc[vtype == "Feed"].reset_index(drop=True)

        # --- Step 7: merge emissions with allocation shares
        emissions = data_herd_long.loc[
            vtype == "Emissions", ["herd_id", "species_short", "variable_type", "variable_name", "value_total"]
        ].rename(columns={"value_total": "value_total_gas"}).reset_index(drop=True)
        data_herd_long_emissions = merge_dt(
            emissions, allocation_herd_long, by=["herd_id", "species_short", "variable_name"]
        )

        # --- Step 8: allocate emissions to commodities
        data_herd_long_emissions["value_total_allocated_gas"] = calc_allocated_emissions(
            value=data_herd_long_emissions["value_total_gas"],
            allocation_share=data_herd_long_emissions["allocation_share"],
        )

        # --- Step 9: gas type (first match of ch4 / n2o / co2, case-insensitive),
        # worked out once per distinct variable name
        name_codes, names = pd.factorize(data_herd_long_emissions["variable_name"], use_na_sentinel=False)
        lower = [None if v is None else v.lower() for v in as_str(names)]
        gas_of_name = np.array(
            [
                None if v is None else "CH4" if "ch4" in v else "N2O" if "n2o" in v else "CO2" if "co2" in v else None
                for v in lower
            ],
            dtype=object,
        )
        data_herd_long_emissions["gas"] = gas_of_name[name_codes]

        # --- Step 10: CO2-equivalents
        co2eq = calc_co2eq(
            gas=data_herd_long_emissions["gas"],
            value_allocated=data_herd_long_emissions["value_total_allocated_gas"],
            global_warming_potential_set=global_warming_potential_set,
        )
        data_herd_long_emissions["value_total_allocated_co2eq"] = co2eq["value_co2eq"]
        data_herd_long_emissions["gwp"] = co2eq["gwp"]

        # --- Step 11: attach labels / units
        data_herd_long_emissions = merge_dt(
            data_herd_long_emissions, _meta_table(K.GLEAM_EMISSIONS_META, "emissions_source"), by="variable_name"
        )
        data_herd_long_production = merge_dt(
            data_herd_long_production, _meta_table(K.GLEAM_PRODUCTION_META, "production_source"), by="variable_name"
        )
        data_herd_long_feed = merge_dt(
            data_herd_long_feed, _meta_table(K.GLEAM_FEED_META, "feed_source"), by="variable_name"
        )
        data_herd_long_nitrogen = merge_dt(
            data_herd_long_nitrogen,
            _meta_table(K.GLEAM_NITROGEN_BALANCE_META, "nitrogen_balance_source"),
            by="variable_name",
        )

        progress.success("Aggregation complete.")
        return {
            "results_emissions": data_herd_long_emissions,
            "results_feed": data_herd_long_feed,
            "results_production": data_herd_long_production,
            "results_nitrogen": data_herd_long_nitrogen,
        }


def _cohort_to_herd_long(
    cohort_level_data: pd.DataFrame, data_cohort_long: pd.DataFrame, n_vars: int
) -> pd.DataFrame:
    """Herd totals of the long table (step 5 of :func:`run_aggregation_module`).

    Same result as ``calc_cohort_to_herd_aggregation(data_cohort_long,
    id_cols = c("herd_id", "species_short", "variable_type", "variable_name"),
    vars_to_sum = "value_total")``, but without factorising the four key
    columns of the long table. :func:`_melt` lays the long table out block by
    block (one block of ``n`` rows per measure variable, each in cohort order)
    and the variable and its type are constant within a block, so the groups
    in order of first appearance are: block, then (``herd_id``,
    ``species_short``) pair in order of first appearance in the cohort table.
    Group sums are added in the same row order, so the totals are identical.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort table that was melted (``n`` rows).
    data_cohort_long : pandas.DataFrame
        Output of :func:`_melt` plus ``variable_type`` and ``value_total``.
    n_vars : int
        Number of measure variables (blocks).

    Returns
    -------
    pandas.DataFrame
        ``herd_id``, ``species_short``, ``variable_type``, ``variable_name``
        and ``value_total`` per herd, species and variable.
    """
    n = len(cohort_level_data)
    hs_codes, n_hs = _group_codes(cohort_level_data, ["herd_id", "species_short"])
    hs_first = np.full(n_hs, n, dtype=np.int64)
    np.minimum.at(hs_first, hs_codes, np.arange(n, dtype=np.int64))
    block = np.arange(n_vars, dtype=np.int64)
    return _aggregate_groups(
        data_cohort_long,
        ["herd_id", "species_short", "variable_type", "variable_name"],
        ["value_total"],
        codes=np.repeat(block * n_hs, n) + np.tile(hs_codes, n_vars),
        ngroups=n_vars * n_hs,
        first=(block[:, None] * n + hs_first[None, :]).ravel(),
    )


def _meta_table(meta: list[dict[str, str]], source_col: str) -> pd.DataFrame:
    """``rbindlist(meta)`` with the source column renamed to ``variable_name``."""
    df = pd.DataFrame(meta, dtype=object)
    return df.rename(columns={source_col: "variable_name"})


def _melt(
    df: pd.DataFrame,
    id_vars: Sequence[str],
    measure_vars: Sequence[str],
    variable_name: str,
    value_name: str,
) -> pd.DataFrame:
    """``data.table::melt(..., variable.factor = FALSE)`` for numeric measure columns.

    Rows are ordered measure variable by measure variable, each block in the
    original row order; columns are ``id_vars``, ``variable_name``,
    ``value_name``. Numeric measure columns are coerced to double; if any is
    not numeric the values are kept as they are (R would coerce them to
    character and :func:`gleampy.calc_cohort_totals` then rejects them).
    """
    missing = [c for c in id_vars if c not in df.columns]
    if missing:
        # R: melt() stops with "One or more values in id.vars is invalid"
        abort(f"Missing required columns in `cohort_level_data`: {_val(missing)}")
    n = len(df)
    m = len(measure_vars)
    idx = np.tile(np.arange(n), m)
    out = {c: df[c].iloc[idx].reset_index(drop=True) for c in id_vars}
    out[variable_name] = pd.Series(np.repeat(np.array(list(measure_vars), dtype=object), n), dtype=object)
    cols = [df[c] for c in measure_vars]
    if all(c.dtype.kind in "iufb" for c in cols):
        values = np.concatenate([as_float(c) for c in cols]) if m else np.array([], dtype=float)
    else:
        values = np.concatenate([c.to_numpy(dtype=object) for c in cols])
    out[value_name] = pd.Series(values)
    return pd.DataFrame(out)
