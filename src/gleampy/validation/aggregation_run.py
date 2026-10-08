"""Input validation for :func:`gleampy.run_aggregation_module`.

Port of ``R/validate_run_aggregation_inputs.R``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import as_float, is_na
from ._shared import abort, validate_cohort_short_values, validator

REQUIRED_COHORT_COLS: tuple[str, ...] = ("herd_id", "species_short", "cohort_short", "cohort_stock_size")
REQUIRED_ALLOCATION_COLS: tuple[str, ...] = (
    "herd_id", "species_short", "variable_name", "commodity_name", "allocation_share",
)


def _val(values: Iterable[Any]) -> str:
    """``cli`` ``{.val {x}}`` rendering: ``"a", "b", and "c"``."""
    vals = [
        "NA" if is_na(v) else (f'"{v}"' if isinstance(v, str) else _fmt_id(v)) for v in values
    ]
    if len(vals) <= 1:
        return "".join(vals)
    if len(vals) == 2:
        return f"{vals[0]} and {vals[1]}"
    return ", ".join(vals[:-1]) + f", and {vals[-1]}"


def _fmt_id(v: Any) -> str:
    """``as.character()`` of a herd id (``1`` rather than ``1.0``)."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _norm(v: Any) -> Any:
    """Normalise ids so that int / float herd ids compare equal, NA as ``None``."""
    if is_na(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)):
        return float(v)
    return v


def _unique(values: Iterable[Any]) -> list[Any]:
    out, seen = [], set()
    for v in values:
        k = _norm(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _setdiff(x: Iterable[Any], y: Iterable[Any]) -> list[Any]:
    """R ``setdiff()``: unique elements of ``x`` not in ``y`` (order of ``x``)."""
    ys = {_norm(v) for v in y}
    return [v for v in _unique(x) if _norm(v) not in ys]


def _distinct(s: pd.Series) -> list[Any]:
    """``unique(s)`` as a list of Python scalars, in order of first appearance.

    R deduplicates before ``setdiff()`` and ``paste()``; doing the same keeps
    the Python-level work proportional to the number of herds, not rows.
    ``pd.unique`` merges only values that compare equal (``1`` and ``1.0``),
    which :func:`_setdiff` treats as the same id anyway. Unhashable values
    fall back to all values.
    """
    try:
        vals = pd.unique(s)
    except TypeError:
        return s.tolist()
    return np.asarray(vals, dtype=object).tolist()


def _pair_keys(df: pd.DataFrame) -> list[str]:
    """``paste(herd_id, species_short, sep = "|")`` of the distinct (herd, species) pairs."""
    pairs = df[["herd_id", "species_short"]]
    try:
        pairs = pairs.drop_duplicates()
    except TypeError:  # unhashable values: format every row
        pass
    return [
        f"{'NA' if is_na(h) else _fmt_id(h)}|{'NA' if is_na(s) else s}"
        for h, s in zip(pairs["herd_id"].tolist(), pairs["species_short"].tolist())
    ]


@validator
def validate_run_aggregation_module_inputs(
    cohort_level_data: pd.DataFrame,
    allocation_herd_long: pd.DataFrame,
    simulation_duration: Any,
    global_warming_potential_set: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.run_aggregation_module`.

    Checks table types and required columns, ``simulation_duration`` (single
    positive number), ``global_warming_potential_set``, allocation shares in
    ``[0, 1]``, that both tables cover the same ``herd_id`` set and every
    cohort (``herd_id``, ``species_short``) pair, valid ``cohort_short``
    codes and non-negative ``cohort_stock_size``. As in R, the cross-table
    checks work on the distinct herd ids and pairs.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table (see :func:`gleampy.run_aggregation_module`).
    allocation_herd_long : pandas.DataFrame
        Long-format allocation shares (fractions).
    simulation_duration : Any
        Length of the assessment period (days).
    global_warming_potential_set : Any
        GWP-100 set.
    """
    # --- Basic type and structure checks
    if not isinstance(cohort_level_data, pd.DataFrame):
        abort("`cohort_level_data` must be a data.table.")
    if not isinstance(allocation_herd_long, pd.DataFrame):
        abort("`allocation_herd_long` must be a data.table.")
    if len(cohort_level_data) == 0:
        abort("`cohort_level_data` must contain at least one row.")
    if len(allocation_herd_long) == 0:
        abort("`allocation_herd_long` must contain at least one row.")

    # --- Required columns
    missing_cohort_cols = [c for c in REQUIRED_COHORT_COLS if c not in cohort_level_data.columns]
    if missing_cohort_cols:
        abort(f"Missing required columns in `cohort_level_data`: {_val(missing_cohort_cols)}")
    missing_allocation_cols = [c for c in REQUIRED_ALLOCATION_COLS if c not in allocation_herd_long.columns]
    if missing_allocation_cols:
        abort(f"Missing required columns in `allocation_herd_long`: {_val(missing_allocation_cols)}")

    # --- simulation_duration
    sd = simulation_duration
    if (
        isinstance(sd, (bool, np.bool_, str))
        or not isinstance(sd, (int, float, np.integer, np.floating))
        or np.isnan(float(sd))
    ):
        abort("`simulation_duration` must be a single numeric value.")
    if sd <= 0:
        abort("`simulation_duration` must be positive (days).")

    # --- global_warming_potential_set
    valid_gwp = K.GLOBAL_WARMING_POTENTIAL_SETS
    if not isinstance(global_warming_potential_set, str):
        abort("`global_warming_potential_set` must be a single character value.")
    if global_warming_potential_set not in valid_gwp:
        abort(f"`global_warming_potential_set` must be one of: {_val(valid_gwp)}")

    # --- allocation_share bounds (NA ignored)
    share = as_float(allocation_herd_long["allocation_share"])
    with np.errstate(invalid="ignore"):
        if np.any(share < 0) or np.any(share > 1):
            abort("`allocation_share` in `allocation_herd_long` must be between 0 and 1.")

    # --- Cross-table: herd ids
    cohort_herd_ids = _distinct(cohort_level_data["herd_id"])
    allocation_herd_ids = _distinct(allocation_herd_long["herd_id"])
    missing_in_allocation = _setdiff(cohort_herd_ids, allocation_herd_ids)
    if missing_in_allocation:
        abort(
            "Herd IDs in `cohort_level_data` not found in `allocation_herd_long`: "
            f"{_val(missing_in_allocation)}"
        )
    missing_in_cohort = _setdiff(allocation_herd_ids, cohort_herd_ids)
    if missing_in_cohort:
        abort(
            "Herd IDs in `allocation_herd_long` not found in `cohort_level_data`: "
            f"{_val(missing_in_cohort)}"
        )

    # --- Cross-table: (herd_id, species_short) coverage
    missing_keys = _setdiff(_pair_keys(cohort_level_data), _pair_keys(allocation_herd_long))
    if missing_keys:
        # R passes the rest of the sentence ("entries in `allocation_herd_long`:
        # <keys>") as a second argument to cli::cli_abort(), which drops it, so
        # the R message ends at "have no". Reproduced as R emits it.
        abort("Some (herd_id, species_short) combinations in `cohort_level_data` have no")

    # --- Valid cohort_short values
    validate_cohort_short_values(cohort_level_data["cohort_short"], data_arg="cohort_level_data")

    # --- cohort_stock_size non-negative (NA ignored)
    stock = as_float(cohort_level_data["cohort_stock_size"])
    with np.errstate(invalid="ignore"):
        if np.any(stock < 0):
            abort("`cohort_stock_size` in `cohort_level_data` must be non-negative.")
