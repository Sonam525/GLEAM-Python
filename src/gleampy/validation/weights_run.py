"""Input validation for :func:`gleampy.run_weights_module`.

Port of ``R/validate_run_weights_inputs.R``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import as_float, as_str
from ._shared import (
    abort,
    check_data_table,
    check_herd_id_consistency,
    check_herd_id_unique,
    check_required_columns,
    validate_cohort_short_values,
    validator,
)

_REQUIRED_COHORT_COLS = ("herd_id", "cohort_short", "cohort_duration_days", "offtake_rate")

_REQUIRED_DEMOGRAPHIC_HERD_COLS = (
    "live_weight_female_adult",
    "live_weight_male_adult",
    "live_weight_at_birth",
    "live_weight_at_weaning",
    "live_weight_female_at_slaughter",
    "live_weight_male_at_slaughter",
)

_REQUIRED_NONDEMOGRAPHIC_HERD_COLS = (
    "live_weight_female_nondemographic_start",
    "live_weight_male_nondemographic_start",
    "live_weight_female_nondemographic_end",
    "live_weight_male_nondemographic_end",
    "phase1_nondemo_fem_duration_days",
    "phase2_nondemo_fem_duration_days",
    "phase1_nondemo_mal_duration_days",
    "phase2_nondemo_mal_duration_days",
)


def _fmt_ids(xs: Sequence[Any]) -> str:
    """cli ``{.val}`` formatting of herd ids (``1, 2, and 3``)."""
    items = []
    for v in xs:
        if isinstance(v, str):
            items.append(f'"{v}"')
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            items.append(str(int(v)))
        else:
            items.append(str(v))
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _birth_violations(herd_level_data: pd.DataFrame, other: str, extra: np.ndarray | None = None) -> list:
    """herd_id of rows with ``!is.na(birth) & !is.na(other) & birth >= other [& extra]``."""
    birth = as_float(herd_level_data["live_weight_at_birth"])
    oth = as_float(herd_level_data[other])
    with np.errstate(invalid="ignore"):
        sel = ~np.isnan(birth) & ~np.isnan(oth) & (birth >= oth)
    if extra is not None:
        sel &= extra
    return herd_level_data["herd_id"].to_numpy(dtype=object)[sel].tolist()


@validator
def validate_run_weights_module_inputs(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
) -> None:
    """Validate structure, columns and cross-table consistency for the weights module.

    * both inputs are non-empty tables;
    * ``cohort_level_data`` has ``herd_id``, ``cohort_short``,
      ``cohort_duration_days`` and ``offtake_rate``;
    * ``herd_level_data`` has ``herd_id``, the demographic weight columns when
      demographic cohorts are present and the non-demographic start/end
      weights and phase durations when ``FN`` / ``MN`` are present;
    * valid ``cohort_short`` codes, unique ``herd_id`` in ``herd_level_data``
      and the same herd set in both tables;
    * per herd, ``live_weight_at_birth`` is below the female and male
      slaughter weights and (except chickens) below the weaning weight.
    """
    # --- Basic type and structure checks
    check_data_table(cohort_level_data, "cohort_level_data")
    check_data_table(herd_level_data, "herd_level_data")

    # --- Required columns
    cohort_codes = cohort_level_data["cohort_short"] if "cohort_short" in cohort_level_data.columns else None
    required_herd_cols = ["herd_id"]
    if cohort_codes is not None and cohort_codes.isin(K.GLEAM_COHORTS_DEMOGRAPHIC).any():
        required_herd_cols += list(_REQUIRED_DEMOGRAPHIC_HERD_COLS)

    check_required_columns(cohort_level_data, _REQUIRED_COHORT_COLS, "cohort_level_data")
    check_required_columns(herd_level_data, list(dict.fromkeys(required_herd_cols)), "herd_level_data")

    if cohort_level_data["cohort_short"].isin(("FN", "MN")).any():
        check_required_columns(herd_level_data, _REQUIRED_NONDEMOGRAPHIC_HERD_COLS, "herd_level_data")

    # --- Cohort: valid cohort_short
    validate_cohort_short_values(cohort_level_data["cohort_short"], data_arg="cohort_level_data")

    # --- Herd: one row per herd_id
    check_herd_id_unique(herd_level_data, "herd_level_data")

    # --- Cross-table: same herd_id set
    check_herd_id_consistency(cohort_level_data, herd_level_data, "cohort_level_data", "herd_level_data")

    # --- Module-specific: weight ordering (per herd_id)
    cols = set(herd_level_data.columns)
    if {"live_weight_at_birth", "live_weight_female_at_slaughter"} <= cols:
        bad = _birth_violations(herd_level_data, "live_weight_female_at_slaughter")
        if bad:
            abort(
                "For each herd_id, `live_weight_at_birth` must be less than "
                f"`live_weight_female_at_slaughter`. Violation(s) for herd_id: {_fmt_ids(bad)}"
            )

    if {"live_weight_at_birth", "live_weight_male_at_slaughter"} <= cols:
        bad = _birth_violations(herd_level_data, "live_weight_male_at_slaughter")
        if bad:
            abort(
                "For each herd_id, `live_weight_at_birth` must be less than "
                f"`live_weight_male_at_slaughter`. Violation(s) for herd_id: {_fmt_ids(bad)}"
            )

    if {"live_weight_at_birth", "live_weight_at_weaning"} <= cols:
        if "species_short" not in cols:
            # R evaluates `species_short != "CHK"` even when the column is absent
            # (`!"species_short" %in% names(.) | species_short != "CHK"` is not
            # short-circuited), so data.table fails with this error.
            abort(
                "Object 'species_short' not found amongst the columns of `herd_level_data`."
            )
        sp = as_str(herd_level_data["species_short"])
        # NA species: `NA != "CHK"` is NA, and data.table drops NA rows in `i`.
        not_chk = np.array([v is not None and v != "CHK" for v in sp], dtype=bool)
        bad = _birth_violations(herd_level_data, "live_weight_at_weaning", not_chk)
        if bad:
            abort(
                "For each herd_id, `live_weight_at_birth` must be less than "
                f"`live_weight_at_weaning`. Violation(s) for herd_id: {_fmt_ids(bad)}"
            )
