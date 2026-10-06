"""Input validation for :func:`gleam.run_ration_quality_module`.

Port of ``R/validate_run_ration_quality_inputs.R``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._utils import as_float
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


def _group_sums_off_one(data: pd.DataFrame, by: list[str], col: str, tol: float = 1e-6) -> bool:
    """Whether any ``by`` group has ``abs(sum(col) - 1) > tol``.

    R: ``data[, .(s = sum(col)), by = by][abs(s - 1) > tol]``. ``sum()`` is NA
    when the group has a missing value and NA rows are not selected, so such
    groups never fail. NA keys form their own groups.
    """
    vals = as_float(data[col])
    tmp = pd.DataFrame({"__v": vals, "__na": np.isnan(vals)})
    for i, c in enumerate(by):
        tmp[f"__k{i}"] = data[c].to_numpy()
    g = tmp.groupby([f"__k{i}" for i in range(len(by))], dropna=False, sort=False)
    sums = g["__v"].sum().to_numpy(dtype="float64")
    has_na = g["__na"].any().to_numpy(dtype=bool)
    with np.errstate(invalid="ignore"):
        return bool(((np.abs(sums - 1) > tol) & ~has_na).any())


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
    if _group_sums_off_one(rations_share, group_cols, "feed_ration_fraction"):
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
