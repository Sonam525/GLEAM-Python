"""Ration quality module (port of ``R/run_ration_quality_module.R``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .._utils import Progress, as_float, copy_frame, merge_dt
from ..core.ration_quality import (
    calc_feed_digestibility_fraction,
    calc_ration_ash,
    calc_ration_digestibility,
    calc_ration_gross_energy,
    calc_ration_metabolizable_energy,
    calc_ration_nitrogen_content,
    calc_ration_urinary_energy_fraction,
)
from ..validation._shared import setup_validation
from ..validation.ration_quality_run import validate_run_ration_quality_module_inputs

_SUMMARY_COLS = (
    "ration_gross_energy",
    "ration_metabolizable_energy",
    "ration_nitrogen",
    "ration_digestibility_fraction",
    "ration_urinary_energy_fraction",
    "ration_ash",
)


def _group_ids(df: pd.DataFrame, by: Sequence[str]) -> np.ndarray:
    """Group index of each row, numbered in order of first appearance.

    Reproduces data.table's ``by =`` grouping: NA keys form their own group.
    """
    n = len(df)
    codes = [pd.factorize(df[c], use_na_sentinel=False)[0].astype("int64") for c in by]
    if not codes or n == 0:
        return np.zeros(n, dtype="int64")
    radix = 1
    for k in codes:
        radix *= int(k.max()) + 1
    if radix < 2**62:
        combined = np.zeros(n, dtype="int64")
        for k in codes:
            combined = combined * (int(k.max()) + 1) + k
    else:  # pragma: no cover - only for astronomically many groups
        combined = np.empty(n, dtype=object)
        for i, key in enumerate(zip(*codes)):
            combined[i] = key
    return pd.factorize(combined)[0].astype("int64")


def _sum_by(df: pd.DataFrame, by: Sequence[str], cols: Sequence[str], na_rm: bool) -> pd.DataFrame:
    """``df[, .(c = sum(c, na.rm = na_rm), ...), by = by]`` with groups in first-appearance order.

    Group sums accumulate in row order; without ``na_rm`` a group containing
    NA sums to NA, with ``na_rm`` an all-NA group sums to 0 (R semantics).
    """
    by = list(by)
    gid = _group_ids(df, by)
    first = np.unique(gid, return_index=True)[1]
    out = df[by].iloc[first].reset_index(drop=True)
    for c in cols:
        v = as_float(df[c])
        if na_rm:
            v = np.where(np.isnan(v), 0.0, v)
        with np.errstate(invalid="ignore", over="ignore"):
            out[c] = np.bincount(gid, weights=v, minlength=len(first)).astype("float64", copy=False)
    return out


def run_ration_quality_module(
    rations_share: pd.DataFrame,
    feed_params: pd.DataFrame,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> pd.DataFrame:
    """Run the ration quality module: cohort-level diet nutritional metrics.

    Parameters
    ----------
    rations_share : pandas.DataFrame
        Cohort-level ration composition with ``herd_id``, ``species_short``,
        ``cohort_short``, ``feed_id``, ``feed_ration_fraction`` (fraction of
        dry matter intake; sums to 1 per cohort) and, optionally,
        ``nondemo_productive_phase_id`` and ``feed_name``.
    feed_params : pandas.DataFrame
        Feed nutritional parameters by ``feed_id``: ``feed_gross_energy``,
        ``feed_digestible_energy_{ruminant,pigs}``,
        ``feed_metabolizable_energy_{ruminant,pigs}`` (MJ/kg DM),
        ``feed_nitrogen_content`` (kg N/kg DM),
        ``feed_urinary_energy_{ruminant,pigs}`` (fraction) and ``feed_ash``
        (g ash/100 g DM).
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    pandas.DataFrame
        One row per ``herd_id`` x ``species_short`` x ``cohort_short`` (x
        ``nondemo_productive_phase_id`` when present), in order of first
        appearance after the ``feed_id`` join, with ``ration_gross_energy``,
        ``ration_metabolizable_energy`` (MJ/kg DM), ``ration_nitrogen``
        (kg N/kg DM), ``ration_digestibility_fraction``,
        ``ration_urinary_energy_fraction`` (fraction) and ``ration_ash``
        (kg ash/kg DM).

    Notes
    -----
    Steps: feed digestibility ratios (:func:`gleam.calc_feed_digestibility_fraction`);
    inner join of ``rations_share`` and ``feed_params`` on ``feed_id`` (rows
    sorted by ``feed_id``); per-feed contributions
    (:func:`gleam.calc_ration_gross_energy`, :func:`gleam.calc_ration_nitrogen_content`,
    :func:`gleam.calc_ration_digestibility`, :func:`gleam.calc_ration_metabolizable_energy`,
    :func:`gleam.calc_ration_urinary_energy_fraction`, :func:`gleam.calc_ration_ash`);
    sums over feeds by cohort (NA if any contribution is NA).
    """
    with setup_validation(validate_inputs):
        # --- Step 1: validate inputs
        validate_run_ration_quality_module_inputs(rations_share, feed_params)

        progress = Progress(show_indicator)
        progress.status("Aggregating ration quality, please wait...")

        # --- Step 2: working copies
        rations_share = copy_frame(rations_share)
        feed_params = copy_frame(feed_params)
        group_cols = ["herd_id", "species_short", "cohort_short"]
        if "nondemo_productive_phase_id" in rations_share.columns:
            group_cols.append("nondemo_productive_phase_id")

        # --- Step 3: digestibility ratios
        digestibility = calc_feed_digestibility_fraction(
            feed_digestible_energy_ruminant=feed_params["feed_digestible_energy_ruminant"],
            feed_digestible_energy_pigs=feed_params["feed_digestible_energy_pigs"],
            feed_gross_energy=feed_params["feed_gross_energy"],
        )
        for col, values in digestibility.items():
            feed_params[col] = values

        # --- Step 4: merge ration shares with feed parameters (inner join)
        detailed = merge_dt(rations_share, feed_params, by="feed_id")

        # --- Step 5: per-feed contributions
        frac = detailed["feed_ration_fraction"]
        detailed["ration_gross_energy"] = calc_ration_gross_energy(
            feed_ration_fraction=frac, feed_gross_energy=detailed["feed_gross_energy"]
        )
        detailed["ration_nitrogen"] = calc_ration_nitrogen_content(
            feed_ration_fraction=frac, feed_nitrogen_content=detailed["feed_nitrogen_content"]
        )
        detailed["ration_digestibility_fraction"] = calc_ration_digestibility(
            species_short=detailed["species_short"],
            feed_ration_fraction=frac,
            feed_digestibility_fraction_ruminant=detailed["feed_digestibility_fraction_ruminant"],
            feed_digestibility_fraction_pigs=detailed["feed_digestibility_fraction_pigs"],
        )
        detailed["ration_metabolizable_energy"] = calc_ration_metabolizable_energy(
            species_short=detailed["species_short"],
            feed_ration_fraction=frac,
            feed_metabolizable_energy_ruminant=detailed["feed_metabolizable_energy_ruminant"],
            feed_metabolizable_energy_pigs=detailed["feed_metabolizable_energy_pigs"],
        )
        detailed["ration_urinary_energy_fraction"] = calc_ration_urinary_energy_fraction(
            species_short=detailed["species_short"],
            feed_ration_fraction=frac,
            feed_urinary_energy_ruminant=detailed["feed_urinary_energy_ruminant"],
            feed_urinary_energy_pigs=detailed["feed_urinary_energy_pigs"],
        )
        detailed["ration_ash"] = calc_ration_ash(
            feed_ration_fraction=frac, feed_ash=detailed["feed_ash"]
        )

        # --- Step 6: cohort-level sums (R sum(): NA if any contribution is NA)
        summary = _sum_by(detailed, group_cols, _SUMMARY_COLS, na_rm=False)

        progress.success("Ration quality aggregation complete.")
        return summary
