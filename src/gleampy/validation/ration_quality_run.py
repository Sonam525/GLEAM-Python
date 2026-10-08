"""Input validation for :func:`gleampy.run_ration_quality_module`.

Port of ``R/validate_run_ration_quality_inputs.R``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .._utils import as_float, group_sum
from ._shared import abort, check_required_columns, validator

_REQUIRED_RATIONS_COLS = ("herd_id", "species_short", "feed_id", "cohort_short", "feed_ration_fraction")

_REQUIRED_FEED_COLS = (
    "feed_id",
    "feed_gross_energy",
    "feed_digestible_energy_ruminant",
    "feed_digestible_energy_pigs",
    "feed_metabolizable_energy_ruminant",
    "feed_metabolizable_energy_pigs",
    "feed_nitrogen_content",
    "feed_urinary_energy_ruminant",
    "feed_urinary_energy_pigs",
    "feed_ash",
)


def group_ids(df: pd.DataFrame, by: Sequence[str]) -> np.ndarray:
    """Group index of each row, numbered in order of first appearance.

    Reproduces data.table's ``by =`` grouping: NA keys form their own group.
    Categorical and nullable key columns are grouped by value.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with the key columns.
    by : sequence of str
        Key columns.

    Returns
    -------
    numpy.ndarray
        int64 group number (0, 1, ...) of each row.
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


def group_sums_off_one(data: pd.DataFrame, by: Sequence[str], col: str, tol: float = 1e-6) -> bool:
    """Whether any ``by`` group has ``abs(sum(col) - 1) > tol``.

    R: ``data[, .(s = sum(col)), by = by][abs(s - 1) > tol]``. data.table
    computes the grouped ``sum()`` with GForce ``gsum``, a plain double
    accumulation in row order, so the sums here are added the same way
    (:func:`gleampy._utils.group_sum`) rather than with pandas' compensated
    summation: a group whose sum lies within a few ulp of ``1 +/- tol`` is
    accepted or rejected exactly as in R. ``sum()`` is NA when the group has a
    missing value and NA rows are not selected, so such groups never fail.
    NA keys form their own groups.

    Parameters
    ----------
    data : pandas.DataFrame
        Table with the key columns and ``col``.
    by : sequence of str
        Group key columns.
    col : str
        Column whose group sums must be 1.
    tol : float, default 1e-6
        Absolute tolerance.

    Returns
    -------
    bool
        ``True`` if some group sum differs from 1 by more than ``tol``.
    """
    gid = group_ids(data, list(by))
    ngroups = int(gid.max()) + 1 if len(gid) else 0
    with np.errstate(invalid="ignore", over="ignore"):
        sums = group_sum(as_float(data[col]), gid, ngroups)
        return bool((np.abs(sums - 1) > tol).any())


@validator
def validate_run_ration_quality_module_inputs(
    rations_share: pd.DataFrame,
    feed_params: pd.DataFrame,
) -> None:
    """Validate structure, columns and identifiers for the ration quality module.

    * both inputs are non-empty tables with the required columns;
    * feed ration fractions sum to 1 within each ``herd_id``,
      ``species_short``, ``cohort_short`` (and ``nondemo_productive_phase_id``
      when present);
    * ``feed_params$feed_id`` is unique and ``feed_id`` is unique within each
      ration group of ``rations_share``.

    The group sums are added in row order like data.table's ``gsum`` (see
    :func:`group_sums_off_one`), so a sum within a few ulp of ``1 +/- 1e-6``
    is accepted or rejected exactly as in R.

    Parameters
    ----------
    rations_share : pandas.DataFrame
        Ration composition: ``herd_id``, ``species_short``, ``feed_id``,
        ``cohort_short``, ``feed_ration_fraction`` (fraction of dry matter
        intake) and, optionally, ``nondemo_productive_phase_id``.
    feed_params : pandas.DataFrame
        Feed parameters by ``feed_id``: ``feed_gross_energy``,
        ``feed_digestible_energy_{ruminant,pigs}``,
        ``feed_metabolizable_energy_{ruminant,pigs}`` (MJ/kg DM),
        ``feed_nitrogen_content`` (kg N/kg DM),
        ``feed_urinary_energy_{ruminant,pigs}`` (fraction) and ``feed_ash``
        (g ash/100 g DM).

    Raises
    ------
    GleamValidationError
        On the first failed check (no-op when validation is disabled).
    """
    # --- Basic type and structure checks
    if not isinstance(rations_share, pd.DataFrame):
        abort("`rations_share` must be a data.table.")
    if not isinstance(feed_params, pd.DataFrame):
        abort("`feed_params` must be a data.table.")
    if len(rations_share) == 0:
        abort("`rations_share` must contain at least one row.")
    if len(feed_params) == 0:
        abort("`feed_params` must contain at least one row.")

    # --- Required columns validation
    check_required_columns(rations_share, _REQUIRED_RATIONS_COLS, "rations_share")
    check_required_columns(feed_params, _REQUIRED_FEED_COLS, "feed_params")

    # --- Ration share consistency
    group_cols = ["herd_id", "species_short", "cohort_short"]
    if "nondemo_productive_phase_id" in rations_share.columns:
        group_cols.append("nondemo_productive_phase_id")
    if group_sums_off_one(rations_share, group_cols, "feed_ration_fraction"):
        abort(
            "Feed rations must sum to 1 within each herd_id, species_short, cohort_short, "
            "and nondemo_productive_phase_id when provided."
        )

    # --- Feed parameter integrity checks
    if feed_params["feed_id"].duplicated().any():
        abort("`feed_params$feed_id` must be unique.")
    if rations_share[group_cols + ["feed_id"]].duplicated().any():
        abort(
            "`rations_share$feed_id` must be unique within each herd_id, species_short, "
            "cohort_short, and nondemo_productive_phase_id combination when "
            "nondemo_productive_phase_id is provided."
        )
