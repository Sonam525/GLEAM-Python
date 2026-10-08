"""Input validation for :func:`gleampy.run_emissions_ration_module`.

Port of ``R/validate_run_emissions_ration_inputs.R``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._utils import as_str, is_na, merge_dt
from ._shared import abort, check_required_columns, validator
from .ration_quality_run import group_sums_off_one

_REQUIRED_RATIONS_COLS = (
    "herd_id", "species_short", "feed_name", "feed_id", "cohort_short", "feed_ration_fraction",
)

_REQUIRED_EMISSIONS_COLS = (
    "feed_id",
    "co2_feed_fertilizer",
    "co2_feed_pesticides",
    "co2_feed_crop_activities",
    "co2_feed_luc_nopeat",
    "co2_feed_luc_peat",
    "n2o_feed_fertilizer",
    "n2o_feed_manure_applied",
    "n2o_feed_crop_residues",
    "ch4_feed_rice",
)


def _fmt_vals(xs: Sequence[Any]) -> str:
    """cli ``{.val}`` formatting of a vector: ``"a"``, ``"a" and "b"``, ``"a", "b", and "c"``."""
    items = [f'"{v}"' if isinstance(v, str) else str(v) for v in xs]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _duplicated_both(df: pd.DataFrame) -> np.ndarray:
    """``duplicated(df) | duplicated(df, fromLast = TRUE)``."""
    return (df.duplicated(keep="first") | df.duplicated(keep="last")).to_numpy()


def _is_numeric_column(s: pd.Series) -> bool:
    """R ``is.numeric()`` for a table column (an all-NA object column is logical in R)."""
    if s.dtype.kind in "iuf":
        return True
    if s.dtype.kind == "b":
        return False
    vals = [v for v in s.tolist() if not is_na(v)]
    return bool(vals) and all(
        isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) for v in vals
    )


@validator
def validate_run_emissions_ration_module_inputs(
    rations_share: pd.DataFrame,
    feed_emissions: pd.DataFrame,
) -> None:
    """Validate structure, columns and identifiers for the feed emissions module.

    * both inputs are non-empty tables with the required columns;
    * feed ration fractions sum to 1 within each ``herd_id``,
      ``species_short``, ``cohort_short`` (and ``nondemo_productive_phase_id``
      when present);
    * ``feed_id`` and ``feed_name`` are unique within each ration group;
    * ``feed_emissions$feed_id`` (and ``feed_name`` when present) are unique,
      and when ``feed_emissions`` has ``feed_name`` it matches the
      ``rations_share`` name of every ``feed_id``;
    * emission factor columns are numeric (NA allowed).

    The ration group sums are added in row order like data.table's ``gsum``
    (see :func:`gleampy.validation.ration_quality_run.group_sums_off_one`), so a
    sum within a few ulp of ``1 +/- 1e-6`` is accepted or rejected exactly as
    in R.

    Parameters
    ----------
    rations_share : pandas.DataFrame
        Ration composition: ``herd_id``, ``species_short``, ``feed_name``,
        ``feed_id``, ``cohort_short``, ``feed_ration_fraction`` (fraction of
        dry matter intake) and, optionally, ``nondemo_productive_phase_id``.
    feed_emissions : pandas.DataFrame
        Feed emission factors by ``feed_id`` (and optionally ``feed_name``):
        ``co2_feed_*`` (g CO2/kg DM), ``n2o_feed_*`` (g N2O/kg DM) and
        ``ch4_feed_rice`` (g CH4/kg DM); NA allowed.

    Raises
    ------
    GleamValidationError
        On the first failed check (no-op when validation is disabled).
    """
    # --- Basic type and structure checks
    if not isinstance(rations_share, pd.DataFrame):
        abort("`rations_share` must be a data.table.")
    if not isinstance(feed_emissions, pd.DataFrame):
        abort("`feed_emissions` must be a data.table.")
    if len(rations_share) == 0:
        abort("`rations_share` must contain at least one row.")
    if len(feed_emissions) == 0:
        abort("`feed_emissions` must contain at least one row.")

    # --- Required columns validation
    check_required_columns(rations_share, _REQUIRED_RATIONS_COLS, "rations_share")
    check_required_columns(feed_emissions, _REQUIRED_EMISSIONS_COLS, "feed_emissions")

    # --- Ration share consistency
    scope = ["herd_id", "species_short", "cohort_short"]
    if "nondemo_productive_phase_id" in rations_share.columns:
        scope.append("nondemo_productive_phase_id")
    if group_sums_off_one(rations_share, scope, "feed_ration_fraction"):
        abort(
            "Feed emissions fractions must sum to 1 within each herd_id, species_short, "
            "cohort_short, and nondemo_productive_phase_id when provided."
        )

    # --- rations_share key uniqueness (feed_id, then feed_name, within herd/cohort/species)
    for key in ("feed_id", "feed_name"):
        keys = rations_share[scope + [key]]
        dup = keys[_duplicated_both(keys)]
        if len(dup):
            preview = dup.drop_duplicates().head(10)
            abort(
                f"`rations_share` contains duplicated `{key}` within herd/cohort/species_short. "
                f"Expected unique rows by: {_fmt_vals(scope + [key])}. "
                f"Duplicate keys (first 10): {preview.to_string(index=False)}"
            )

    # --- Feed emissions integrity checks
    fid = feed_emissions["feed_id"]
    dup_ids = list(pd.unique(fid[(fid.duplicated(keep="first") | fid.duplicated(keep="last")).to_numpy()]))
    if dup_ids:
        cols = [c for c in ("feed_id", "feed_name") if c in feed_emissions.columns]
        preview = feed_emissions.loc[fid.isin(dup_ids).to_numpy(), cols].drop_duplicates().head(10)
        abort(
            "`feed_emissions$feed_id` must be unique. "
            f"Duplicated feed_id(s): {_fmt_vals(dup_ids)}. "
            f"Offending rows (first 10): {preview.to_string(index=False)}"
        )

    if "feed_name" in feed_emissions.columns:
        fname = feed_emissions["feed_name"]
        sel = fname.notna() & (fname.duplicated(keep="first") | fname.duplicated(keep="last"))
        dup_names = list(pd.unique(fname[sel.to_numpy()]))
        if dup_names:
            sub = feed_emissions[fname.isin(dup_names).to_numpy()]
            dup_map = (
                sub.groupby("feed_name", sort=False, observed=True)["feed_id"]
                .agg(
                    feed_ids=lambda s: ", ".join(sorted(str(v) for v in pd.unique(s))),
                    n_ids=lambda s: s.nunique(dropna=False),
                )
                .reset_index()
                .sort_values(["n_ids", "feed_name"], ascending=[False, True], kind="mergesort")
            )
            abort(
                "`feed_emissions$feed_name` must be unique. "
                f"Duplicated feed_name(s): {_fmt_vals(dup_names)}. "
                f"feed_name -> feed_id mapping (first 10 names): "
                f"{dup_map.head(10).to_string(index=False)}"
            )

    # Cross-check feed_name consistency between rations_share and feed_emissions by feed_id
    if "feed_name" in feed_emissions.columns:
        check = merge_dt(
            rations_share[["feed_id", "feed_name"]],
            feed_emissions[["feed_id", "feed_name"]].drop_duplicates(),
            by="feed_id",
            all_x=True,
            suffixes=("_input", "_emissions"),
        )
        name_in = as_str(check["feed_name_input"])
        name_em = as_str(check["feed_name_emissions"])
        # `is.na(em) | in != em`: rows where only the input name is NA give NA (not selected)
        mismatch = np.array(
            [e is None or (i is not None and i != e) for i, e in zip(name_in, name_em)], dtype=bool
        )
        bad_ids = list(pd.unique(check["feed_id"][mismatch]))
        if bad_ids:
            abort(
                "feed_id values with missing or mismatched feed_name in `feed_emissions`: "
                f"{_fmt_vals(bad_ids)}"
            )

    # --- Emissions value validation (type): numeric, NA allowed
    for col in _REQUIRED_EMISSIONS_COLS[1:]:
        if not _is_numeric_column(feed_emissions[col]):
            abort(f"`{col}` must be a single numeric (scalar). NA is allowed.")
