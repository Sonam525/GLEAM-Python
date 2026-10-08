"""Feed-production emissions module (port of ``R/run_emissions_ration_module.R``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .._utils import Progress, as_float, copy_frame, merge_dt
from ..core.emissions_ration import (
    calc_ch4_ration_rice,
    calc_co2_ration_crop_activities,
    calc_co2_ration_fertilizer,
    calc_co2_ration_luc_nopeat,
    calc_co2_ration_luc_peat,
    calc_co2_ration_pesticides,
    calc_n2o_ration_crop_residues,
    calc_n2o_ration_fertilizer,
    calc_n2o_ration_manure,
)
from ..validation._shared import setup_validation
from ..validation.emissions_ration_run import validate_run_emissions_ration_module_inputs
from ..validation.ration_quality_run import group_ids

# (output column, core function, feed emission factor column), in R order.
_CONTRIBUTIONS = (
    ("co2_ration_fertilizer", calc_co2_ration_fertilizer, "co2_feed_fertilizer"),
    ("co2_ration_pesticides", calc_co2_ration_pesticides, "co2_feed_pesticides"),
    ("co2_ration_crop_activities", calc_co2_ration_crop_activities, "co2_feed_crop_activities"),
    ("co2_ration_luc_nopeat", calc_co2_ration_luc_nopeat, "co2_feed_luc_nopeat"),
    ("co2_ration_luc_peat", calc_co2_ration_luc_peat, "co2_feed_luc_peat"),
    ("n2o_ration_fertilizer", calc_n2o_ration_fertilizer, "n2o_feed_fertilizer"),
    ("n2o_ration_manure_applied", calc_n2o_ration_manure, "n2o_feed_manure_applied"),
    ("n2o_ration_crop_residues", calc_n2o_ration_crop_residues, "n2o_feed_crop_residues"),
    ("ch4_ration_rice", calc_ch4_ration_rice, "ch4_feed_rice"),
)


def _sum_by(df: pd.DataFrame, by: Sequence[str], cols: Sequence[str], na_rm: bool) -> pd.DataFrame:
    """``df[, .(c = sum(c, na.rm = na_rm), ...), by = by]`` with groups in first-appearance order.

    Group sums accumulate in row order; without ``na_rm`` a group containing
    NA sums to NA, with ``na_rm`` an all-NA group sums to 0 (R semantics).
    """
    by = list(by)
    gid = group_ids(df, by)
    first = np.unique(gid, return_index=True)[1]
    out = df[by].iloc[first].reset_index(drop=True)
    for c in cols:
        v = as_float(df[c])
        if na_rm:
            v = np.where(np.isnan(v), 0.0, v)
        with np.errstate(invalid="ignore", over="ignore"):
            out[c] = np.bincount(gid, weights=v, minlength=len(first)).astype("float64", copy=False)
    return out


def run_emissions_ration_module(
    rations_share: pd.DataFrame,
    feed_emissions: pd.DataFrame,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> pd.DataFrame:
    """Run the feed-production emissions module: cohort-level diet emission factors.

    Parameters
    ----------
    rations_share : pandas.DataFrame
        Cohort-level ration composition with ``herd_id``, ``species_short``,
        ``cohort_short``, ``feed_id``, ``feed_name``, ``feed_ration_fraction``
        (fraction of dry matter intake; sums to 1 per cohort) and, optionally,
        ``nondemo_productive_phase_id``.
    feed_emissions : pandas.DataFrame
        Feed emission factors by ``feed_id`` (per kg DM):
        ``co2_feed_fertilizer``, ``co2_feed_pesticides``,
        ``co2_feed_crop_activities``, ``co2_feed_luc_nopeat``,
        ``co2_feed_luc_peat`` (g CO2), ``n2o_feed_fertilizer``,
        ``n2o_feed_manure_applied``, ``n2o_feed_crop_residues`` (g N2O),
        ``ch4_feed_rice`` (g CH4); optionally ``feed_name``.
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    pandas.DataFrame
        One row per ``herd_id`` x ``species_short`` x ``cohort_short`` (x
        ``nondemo_productive_phase_id`` when present), in order of first
        appearance after the ``feed_id`` join, with the diet-level emission
        factors ``co2_ration_fertilizer``, ``co2_ration_pesticides``,
        ``co2_ration_crop_activities``, ``co2_ration_luc_nopeat``,
        ``co2_ration_luc_peat`` (g CO2/kg DM), ``n2o_ration_fertilizer``,
        ``n2o_ration_manure_applied``, ``n2o_ration_crop_residues``
        (g N2O/kg DM) and ``ch4_ration_rice`` (g CH4/kg DM).

    Notes
    -----
    Steps: left join of ``rations_share`` and ``feed_emissions`` on
    ``feed_id`` (rows sorted by ``feed_id``; feeds without emission factors
    get NA); per-feed contributions ``feed_ration_fraction * feed_ef`` (e.g.
    :func:`gleampy.calc_co2_ration_fertilizer`); sums over feeds by cohort with
    NA contributions dropped (``sum(..., na.rm = TRUE)``).
    """
    with setup_validation(validate_inputs):
        # --- Step 1: validate inputs
        validate_run_emissions_ration_module_inputs(rations_share, feed_emissions)

        progress = Progress(show_indicator)
        progress.status("Aggregating feed emissions, please wait...")

        # --- Step 2: working copies
        rations_share = copy_frame(rations_share)
        feed_emissions = copy_frame(feed_emissions)
        group_cols = ["herd_id", "species_short", "cohort_short"]
        if "nondemo_productive_phase_id" in rations_share.columns:
            group_cols.append("nondemo_productive_phase_id")

        # --- Step 3: merge ration shares with feed emission factors (left join)
        detailed = merge_dt(rations_share, feed_emissions, by="feed_id", all_x=True)

        # --- Step 4: per-feed contributions
        frac = detailed["feed_ration_fraction"]
        for out_col, fn, ef_col in _CONTRIBUTIONS:
            detailed[out_col] = fn(frac, detailed[ef_col])

        # --- Step 5: cohort-level sums (NA contributions removed)
        summary = _sum_by(detailed, group_cols, [c for c, _, _ in _CONTRIBUTIONS], na_rm=True)

        progress.success("Feed emissions aggregation complete.")
        return summary
