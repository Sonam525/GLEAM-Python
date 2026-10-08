"""Input validation for the feed-production (ration) emissions core model.

Port of ``R/validate_emissions_ration_core_model.R``. The R validators are
called row by row on scalars; these versions check whole vectors at once and
are no-ops when validation is disabled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import is_scalar
from ._shared import abort, validate_param_range, validate_scalar_numeric_or_na, validator


def _check_lengths(feed_ration_fraction: Any, ef_name: str, ef: Any) -> None:
    """Vectorised replacement of R's ``length(x) != 1`` check.

    The emission factor must be a scalar or a 1-d vector of the same length as
    ``feed_ration_fraction`` (when that is a vector).
    """
    def size(v: Any) -> int:
        if is_scalar(v):
            return 1
        return int(np.size(v)) if np.ndim(v) == 1 else -1

    n_frac, n_ef = size(feed_ration_fraction), size(ef)
    if n_frac == -1:
        abort(
            "`feed_ration_fraction` must be a single numeric (scalar) or a vector with the "
            "same length as the other inputs."
        )
    if n_ef == -1 or (n_ef != 1 and n_frac != 1 and n_ef != n_frac):
        abort(
            f"`{ef_name}` must be a single numeric (scalar) or a vector with the same length "
            "as the other inputs. NA is allowed."
        )


def _numeric_or_na(x: Any, arg_name: str, min_val: float) -> None:
    """``validate_scalar_numeric_or_na`` with a vectorised path for numeric arrays.

    The shared helper loops over the elements in Python; for numeric dtypes
    the ``>= min_val`` check is done with numpy and the shared helper is only
    called on the first offending value (to raise its exact message). Other
    inputs (strings, logicals, object arrays) go through the shared helper.
    """
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy()
    elif isinstance(x, (str, bytes)) or x is None:
        arr = None
    else:
        arr = np.asarray(x)
    if arr is None or arr.dtype.kind not in "iuf":
        validate_scalar_numeric_or_na(x, arg_name, min_val=min_val)
        return
    vals = np.ravel(arr).astype("float64")
    with np.errstate(invalid="ignore"):
        bad = np.flatnonzero(vals < min_val)
    if bad.size:
        validate_scalar_numeric_or_na(float(vals[bad[0]]), arg_name, min_val=min_val)


def _validate(feed_ration_fraction: Any, ef_name: str, ef: Any, min_val: float) -> None:
    _check_lengths(feed_ration_fraction, ef_name, ef)
    validate_param_range(feed_ration_fraction, "feed_ration_fraction")
    _numeric_or_na(ef, ef_name, min_val)


@validator
def validate_co2_ration_fertilizer_inputs(feed_ration_fraction: Any, co2_feed_fertilizer: Any) -> None:
    """``feed_ration_fraction`` within bounds; ``co2_feed_fertilizer`` numeric >= 0 or NA."""
    _validate(feed_ration_fraction, "co2_feed_fertilizer", co2_feed_fertilizer, 0)


@validator
def validate_co2_ration_pesticides_inputs(feed_ration_fraction: Any, co2_feed_pesticides: Any) -> None:
    """``feed_ration_fraction`` within bounds; ``co2_feed_pesticides`` numeric >= 0 or NA."""
    _validate(feed_ration_fraction, "co2_feed_pesticides", co2_feed_pesticides, 0)


@validator
def validate_co2_ration_crop_activities_inputs(
    feed_ration_fraction: Any, co2_feed_crop_activities: Any
) -> None:
    """``feed_ration_fraction`` within bounds; ``co2_feed_crop_activities`` numeric >= 0 or NA."""
    _validate(feed_ration_fraction, "co2_feed_crop_activities", co2_feed_crop_activities, 0)


@validator
def validate_co2_ration_luc_nopeat_inputs(feed_ration_fraction: Any, co2_feed_luc_nopeat: Any) -> None:
    """``feed_ration_fraction`` within bounds; ``co2_feed_luc_nopeat`` numeric (any sign) or NA."""
    _validate(feed_ration_fraction, "co2_feed_luc_nopeat", co2_feed_luc_nopeat, -np.inf)


@validator
def validate_co2_ration_luc_peat_inputs(feed_ration_fraction: Any, co2_feed_luc_peat: Any) -> None:
    """``feed_ration_fraction`` within bounds; ``co2_feed_luc_peat`` numeric (any sign) or NA."""
    _validate(feed_ration_fraction, "co2_feed_luc_peat", co2_feed_luc_peat, -np.inf)


@validator
def validate_n2o_ration_fertilizer_inputs(feed_ration_fraction: Any, n2o_feed_fertilizer: Any) -> None:
    """``feed_ration_fraction`` within bounds; ``n2o_feed_fertilizer`` numeric >= 0 or NA."""
    _validate(feed_ration_fraction, "n2o_feed_fertilizer", n2o_feed_fertilizer, 0)


@validator
def validate_n2o_ration_manure_applied_inputs(
    feed_ration_fraction: Any, n2o_feed_manure_applied: Any
) -> None:
    """``feed_ration_fraction`` within bounds; ``n2o_feed_manure_applied`` numeric >= 0 or NA."""
    _validate(feed_ration_fraction, "n2o_feed_manure_applied", n2o_feed_manure_applied, 0)


@validator
def validate_n2o_ration_crop_residues_inputs(
    feed_ration_fraction: Any, n2o_feed_crop_residues: Any
) -> None:
    """``feed_ration_fraction`` within bounds; ``n2o_feed_crop_residues`` numeric >= 0 or NA."""
    _validate(feed_ration_fraction, "n2o_feed_crop_residues", n2o_feed_crop_residues, 0)


@validator
def validate_ch4_ration_rice_inputs(feed_ration_fraction: Any, ch4_feed_rice: Any) -> None:
    """``feed_ration_fraction`` within bounds; ``ch4_feed_rice`` numeric >= 0 or NA."""
    _validate(feed_ration_fraction, "ch4_feed_rice", ch4_feed_rice, 0)
