"""Validation of ``run_all_herd_module()`` inputs (port of
``R/validate_run_all_herd_inputs.R``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import isin, isna
from ._shared import abort, check_data_table, validator, warn


def _is_flag(x: Any) -> bool:
    """R ``is.logical(x) && length(x) == 1 && !is.na(x)``: ``bool``, ``numpy.bool_`` or a 1-element bool array."""
    if isinstance(x, (bool, np.bool_)):
        return True
    return isinstance(x, np.ndarray) and x.dtype == bool and x.size == 1


@validator
def validate_run_all_herd_module_inputs(
    cohort_level_data: pd.DataFrame | None,
    herd_level_data: pd.DataFrame | None,
    run_demographic: Any,
    run_nondemographic: Any,
) -> None:
    """Module switches, presence of the tables and of the rows each module needs.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame or None
        Cohort table; must contain ``cohort_short`` and the demographic
        (``FJ, FS, FA, MJ, MS, MA``) rows when ``run_demographic`` is TRUE.
    herd_level_data : pandas.DataFrame or None
        Herd table (one row per ``herd_id``).
    run_demographic, run_nondemographic : bool
        Module switches: each must be a single logical value (``bool``,
        ``numpy.bool_`` or a one-element bool array, as R's
        ``is.logical(x) && length(x) == 1``), and at least one must be TRUE.
    """
    if not _is_flag(run_demographic):
        abort("`run_demographic` must be a single logical value (TRUE or FALSE).")
    if not _is_flag(run_nondemographic):
        abort("`run_nondemographic` must be a single logical value (TRUE or FALSE).")
    if not run_demographic and not run_nondemographic:
        abort("At least one of `run_demographic` or `run_nondemographic` must be TRUE.")

    check_data_table(herd_level_data, "herd_level_data")
    if cohort_level_data is None:
        abort("`cohort_level_data` must be provided.")
    check_data_table(cohort_level_data, "cohort_level_data")

    if "cohort_short" not in cohort_level_data.columns:
        abort("`cohort_level_data` must contain `cohort_short`.")

    cs = cohort_level_data["cohort_short"]
    present = ~isna(cs)
    n_demo = int((present & isin(cs, K.GLEAM_COHORTS_DEMOGRAPHIC)).sum())
    n_nondemo = int((present & isin(cs, ("FN", "MN"))).sum())

    if run_demographic and n_demo == 0:
        abort("`run_demographic = TRUE` requires demographic rows in `cohort_level_data`.")
    if run_nondemographic and n_nondemo == 0:
        warn("run_nondemographic=TRUE but no non-demographic rows were found in cohort_level_data.")
