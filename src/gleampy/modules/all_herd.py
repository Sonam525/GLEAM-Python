"""Combined herd module runner (port of ``R/run_all_herd_module.R``).

Runs the demographic herd module, the non-demographic herd module, or both.
With both, the annual entrants of the non-demographic blocks are taken from the
juveniles diverted by the demographic module (``FJ`` -> ``FN``, ``MJ`` -> ``MN``),
the two cohort tables are row-bound and the average stocks and offtake are
rescaled so that each herd's stock sums to ``herd_size_total``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import _key_values, as_float, is_true_scalar, isin, isna, lookup, merge_dt, rbind_fill
from ..core.all_herd import rescale_x_to_y
from ..validation._shared import check_required_columns, setup_validation
from ..validation.all_herd_run import validate_run_all_herd_module_inputs
from .demographic_herd import run_demographic_herd_module
from .nondemographic_herd import run_nondemographic_herd_module

_HERD_DURATION_COLS = (
    "rest_between_nondemo_cycles_duration",
    "phase1_nondemo_fem_duration_days",
    "phase2_nondemo_fem_duration_days",
    "phase1_nondemo_mal_duration_days",
    "phase2_nondemo_mal_duration_days",
)
_INTERMEDIATE_COLS = (
    "size_for_rescaling",
    "herd_size_total",
    "cohort_stock_size_unscaled",
    "offtake_heads_unscaled",
    "offtake_heads_assessment_unscaled",
    "cohort_stock_size_scaled",
    "offtake_heads_scaled",
    "offtake_heads_assessment_scaled",
)


def _update_join(x: pd.DataFrame, i: pd.DataFrame, assignments: dict[str, str], on: str = "herd_id") -> None:
    """``x[i, `:=`(col = i.src, ...), on = on]`` (in place).

    Matched rows get the value of the matching ``i`` row; unmatched rows keep
    their value, or ``NA`` for a new column (appended at the end).
    """
    matched = pd.Series(_keys(x[on])).isin(pd.unique(_keys(i[on]))).to_numpy()
    for col, src in assignments.items():
        vals = lookup(x, i, src, on=on)
        if col in x.columns:
            old = x[col]
            if (old.dtype.kind in "iuf" or old.isna().all()) and vals.dtype.kind == "f":
                x[col] = np.where(matched, vals, as_float(old))
            else:
                new = old.astype(object).copy()
                new[matched] = vals[matched]
                x[col] = new
        else:
            if vals.dtype.kind == "f":
                x[col] = np.where(matched, vals, np.nan)
            else:
                new = np.full(len(x), None, dtype=object)
                new[matched] = vals[matched]
                x[col] = new


def _keys(s: pd.Series) -> np.ndarray:
    """Herd ids as join keys: numbers as float64 (nullable / categorical too), NA as NaN / None."""
    return _key_values(s)


def run_all_herd_module(
    cohort_level_data: pd.DataFrame | None = None,
    herd_level_data: pd.DataFrame | None = None,
    simulation_duration: float = 365,
    show_indicator: bool = True,
    run_demographic: bool = True,
    run_nondemographic: bool = True,
    validate_inputs: bool = True,
) -> dict[str, pd.DataFrame | None]:
    """Run the demographic and/or non-demographic herd modules.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort rows: demographic cohorts (``FJ, FS, FA, MJ, MS, MA``) with the
        inputs of :func:`run_demographic_herd_module` and/or non-demographic
        blocks (``FN, MN``) with the inputs of :func:`run_nondemographic_herd_module`.
    herd_level_data : pandas.DataFrame
        One row per herd with the herd-level inputs of the module(s) run. With
        both modules, the annual non-demographic entrants are derived from the
        demographic output and need not be supplied.
    simulation_duration : float
        Length of the reporting period (days); a single value.
    show_indicator : bool
        Print progress messages.
    run_demographic, run_nondemographic : bool
        Which module(s) to run: a single logical value (``bool``,
        ``numpy.bool_`` or a one-element bool array), tested like R's ``isTRUE()``.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    dict
        ``cohort_level_results`` (demographic rows then non-demographic rows,
        with ``cohort_stock_size``, ``offtake_heads`` and
        ``offtake_heads_assessment``; intermediate scaling columns removed) and
        ``herd_level_results`` (``None`` if no module produced one).
    """
    if cohort_level_data is not None and not isinstance(cohort_level_data, pd.DataFrame):
        cohort_level_data = pd.DataFrame(cohort_level_data)
    if herd_level_data is not None and not isinstance(herd_level_data, pd.DataFrame):
        herd_level_data = pd.DataFrame(herd_level_data)

    with setup_validation(validate_inputs):
        validate_run_all_herd_module_inputs(
            cohort_level_data=cohort_level_data,
            herd_level_data=herd_level_data,
            run_demographic=run_demographic,
            run_nondemographic=run_nondemographic,
        )

        cs = cohort_level_data["cohort_short"]
        present = ~isna(cs)
        cohort_demo = cohort_level_data[present & isin(cs, K.GLEAM_COHORTS_DEMOGRAPHIC)].reset_index(drop=True)
        cohort_nondemo = cohort_level_data[present & isin(cs, ("FN", "MN"))].reset_index(drop=True)

        demo_results = None
        nondemo_results = None
        herd_level_data_nondemo = None

        # 1) Demographic module ---------------------------------------------
        if is_true_scalar(run_demographic):
            demo_results = run_demographic_herd_module(
                cohort_level_data=cohort_demo,
                herd_level_data=herd_level_data,
                simulation_duration=simulation_duration,
                show_indicator=show_indicator,
                validate_inputs=validate_inputs,
            )

            # Non-demographic entrants derived from the demographic output
            duration_cols = [c for c in _HERD_DURATION_COLS if c in herd_level_data.columns]
            herd_duration_rest = herd_level_data[["herd_id", *duration_cols]].reset_index(drop=True)
            dcr = demo_results["cohort_level_results"]
            fem_rows = dcr[np.asarray(dcr["cohort_short"] == "FJ", dtype=bool)]
            mal_rows = dcr[np.asarray(dcr["cohort_short"] == "MJ", dtype=bool)]
            fem_entrants = pd.DataFrame({
                "herd_id": fem_rows["herd_id"].to_numpy(),
                "cohort_stock_fem_annual_nondemo": fem_rows["cohort_stock_annual_nondemographic"].to_numpy(),
            })
            mal_entrants = pd.DataFrame({
                "herd_id": mal_rows["herd_id"].to_numpy(),
                "cohort_stock_mal_annual_nondemo": mal_rows["cohort_stock_annual_nondemographic"].to_numpy(),
            })
            herd_level_data_nondemo = merge_dt(fem_entrants, mal_entrants, by="herd_id", all_x=True, all_y=True)
            herd_level_data_nondemo = merge_dt(
                herd_level_data_nondemo, herd_duration_rest, by="herd_id", all_x=True, all_y=True
            )
            if len(cohort_nondemo) > 0:
                keep = isin(herd_level_data_nondemo["herd_id"], pd.unique(cohort_nondemo["herd_id"]))
                herd_level_data_nondemo = herd_level_data_nondemo[keep].reset_index(drop=True)

        # 2) Non-demographic module ------------------------------------------
        if is_true_scalar(run_nondemographic) and len(cohort_nondemo) > 0:
            if herd_level_data_nondemo is None:
                # herd_id is needed to select the herds before the module validates its inputs
                check_required_columns(cohort_nondemo, ["herd_id"], "cohort_level_data")
                check_required_columns(herd_level_data, ["herd_id"], "herd_level_data")
                keep = isin(herd_level_data["herd_id"], pd.unique(cohort_nondemo["herd_id"]))
                herd_level_data_nondemo = herd_level_data[keep].reset_index(drop=True)
            nondemo_results = run_nondemographic_herd_module(
                cohort_level_data=cohort_nondemo,
                herd_level_data=herd_level_data_nondemo,
                simulation_duration=simulation_duration,
                show_indicator=show_indicator,
                validate_inputs=validate_inputs,
            )

        # 3) Combine -----------------------------------------------------------
        cohort_level_results = rbind_fill([
            demo_results["cohort_level_results"] if demo_results is not None else None,
            nondemo_results["cohort_level_results"] if nondemo_results is not None else None,
        ])

        # 4) Rescaling -----------------------------------------------------------
        if is_true_scalar(run_demographic) and is_true_scalar(run_nondemographic):
            unscaled = as_float(cohort_level_results["cohort_stock_size_unscaled"])
            herd_ids = cohort_level_results["herd_id"]
            groups = pd.Index(pd.unique(_keys(herd_ids)))
            g = groups.get_indexer(_keys(herd_ids))
            totals = np.zeros(len(groups))
            ok = ~np.isnan(unscaled)
            np.add.at(totals, g[ok], unscaled[ok])  # sequential, as data.table's gsum
            cohort_level_results["size_for_rescaling"] = totals[g]
            _update_join(cohort_level_results, herd_level_data, {"herd_size_total": "herd_size_total"})

            cohort_level_results["cohort_stock_size"] = rescale_x_to_y(
                x_scaled_variable=cohort_level_results["cohort_stock_size_unscaled"],
                x_reference_from=cohort_level_results["size_for_rescaling"],
                y_scaling_variable=cohort_level_results["herd_size_total"],
            )
            cohort_level_results["offtake_heads"] = rescale_x_to_y(
                x_scaled_variable=cohort_level_results["offtake_heads_unscaled"],
                x_reference_from=cohort_level_results["cohort_stock_size_unscaled"],
                y_scaling_variable=cohort_level_results["cohort_stock_size"],
            )
            cohort_level_results["offtake_heads_assessment"] = rescale_x_to_y(
                x_scaled_variable=cohort_level_results["offtake_heads_assessment_unscaled"],
                x_reference_from=cohort_level_results["cohort_stock_size_unscaled"],
                y_scaling_variable=cohort_level_results["cohort_stock_size"],
            )
            drop = [c for c in ("size_for_rescaling", "herd_size_total") if c in cohort_level_results.columns]
            cohort_level_results = cohort_level_results.drop(columns=drop)
        elif is_true_scalar(run_demographic) and not is_true_scalar(run_nondemographic):
            cohort_level_results = cohort_level_results.rename(columns={
                "cohort_stock_size_scaled": "cohort_stock_size",
                "offtake_heads_scaled": "offtake_heads",
                "offtake_heads_assessment_scaled": "offtake_heads_assessment",
            })
        elif not is_true_scalar(run_demographic) and is_true_scalar(run_nondemographic):
            cohort_level_results = cohort_level_results.rename(columns={
                "cohort_stock_size_unscaled": "cohort_stock_size",
                "offtake_heads_unscaled": "offtake_heads",
                "offtake_heads_assessment_unscaled": "offtake_heads_assessment",
            })

        # Herd-level results ------------------------------------------------------
        herd_level_results = None
        if demo_results is not None and nondemo_results is not None:
            herd_level_results = demo_results["herd_level_results"].copy()
            _update_join(herd_level_results, nondemo_results["herd_level_results"], {
                "cohort_stock_fem_annual_nondemo": "cohort_stock_fem_annual_nondemo",
                "cohort_stock_mal_annual_nondemo": "cohort_stock_mal_annual_nondemo",
                "rest_between_nondemo_cycles_duration": "rest_between_nondemo_cycles_duration",
                "total_nondemo_fem_duration_days": "total_nondemo_fem_duration_days",
                "total_nondemo_mal_duration_days": "total_nondemo_mal_duration_days",
            })
        elif demo_results is not None:
            herd_level_results = demo_results["herd_level_results"]
        elif nondemo_results is not None:
            herd_level_results = nondemo_results["herd_level_results"]

        drop = [c for c in _INTERMEDIATE_COLS if c in cohort_level_results.columns]
        if drop:
            cohort_level_results = cohort_level_results.drop(columns=drop)

    return {
        "cohort_level_results": cohort_level_results,
        "herd_level_results": herd_level_results,
    }
