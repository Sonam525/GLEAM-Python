"""Validation of ``run_nondemographic_herd_module()`` inputs (port of
``R/validate_run_nondemographic_herd_inputs.R``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._shared import (
    _vals,
    abort,
    check_data_table,
    check_herd_id_consistency,
    check_herd_id_unique,
    check_nondemographic_phase_completeness,
    check_required_columns,
    validate_cohort_short_values,
    validator,
)

PHASE_DURATION_COLS = (
    "phase1_nondemo_fem_duration_days",
    "phase2_nondemo_fem_duration_days",
    "phase1_nondemo_mal_duration_days",
    "phase2_nondemo_mal_duration_days",
)
REQUIRED_COHORT_COLS = ("herd_id", "cohort_short", "nondemo_productive_phase_id", "death_rate")
REQUIRED_HERD_COLS = (
    "herd_id",
    "cohort_stock_fem_annual_nondemo",
    "cohort_stock_mal_annual_nondemo",
    "rest_between_nondemo_cycles_duration",
)


def is_numeric_column(s: pd.Series) -> bool:
    """R ``is.numeric()`` for a data.table column (logical / character are not numeric)."""
    if s.dtype.kind in "iuf":
        return True
    if s.dtype == object:
        vals = [v for v in s.tolist() if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return bool(vals) and all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            for v in vals
        )
    return False


@validator
def validate_run_nondemographic_herd_module_inputs(
    cohort_level_data: pd.DataFrame, herd_level_data: pd.DataFrame
) -> None:
    """Structure, required columns, cohort codes, phases and herd ids of the inputs."""
    herd_cols = set(herd_level_data.columns) if isinstance(herd_level_data, pd.DataFrame) else set()
    has_any = any(c in herd_cols for c in PHASE_DURATION_COLS)
    has_all = all(c in herd_cols for c in PHASE_DURATION_COLS)

    check_data_table(cohort_level_data, "cohort_level_data")
    check_data_table(herd_level_data, "herd_level_data")

    check_required_columns(cohort_level_data, REQUIRED_COHORT_COLS, "cohort_level_data")
    check_required_columns(herd_level_data, REQUIRED_HERD_COLS, "herd_level_data")

    if has_any and not has_all:
        abort(
            "If herd-level non-demographic phase durations are provided in `herd_level_data`, "
            f"all of the following columns must be present: {_vals(PHASE_DURATION_COLS)}."
        )
    if not has_all:
        check_required_columns(cohort_level_data, ["cohort_duration_days"], "cohort_level_data")

    numeric_cohort_cols = ["nondemo_productive_phase_id", "death_rate"]
    if "cohort_duration_days" in cohort_level_data.columns and not has_all:
        numeric_cohort_cols.append("cohort_duration_days")
    numeric_herd_cols = [
        "cohort_stock_fem_annual_nondemo",
        "cohort_stock_mal_annual_nondemo",
        "rest_between_nondemo_cycles_duration",
    ]
    if has_all:
        numeric_herd_cols.extend(PHASE_DURATION_COLS)

    for col in numeric_cohort_cols:
        if not is_numeric_column(cohort_level_data[col]):
            abort(f"`cohort_level_data${col}` must be numeric.")
    for col in numeric_herd_cols:
        if not is_numeric_column(herd_level_data[col]):
            abort(f"`herd_level_data${col}` must be numeric.")

    validate_cohort_short_values(cohort_level_data["cohort_short"], data_arg="cohort_level_data")
    invalid = [v for v in pd.unique(cohort_level_data["cohort_short"]) if v not in ("FN", "MN")]
    if invalid:
        abort(
            f"Invalid `cohort_short` values in `cohort_level_data`: {_vals(invalid)}. "
            f"Non-demographic inputs must use only {_vals(['FN', 'MN'])}."
        )
    check_nondemographic_phase_completeness(cohort_level_data, data_arg="cohort_level_data")

    check_herd_id_unique(herd_level_data, "herd_level_data")
    check_herd_id_consistency(cohort_level_data, herd_level_data, "cohort_level_data", "herd_level_data")
