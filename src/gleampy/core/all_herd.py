"""Combined herd model helpers (port of ``R/core_model_all_herd.R``)."""

from __future__ import annotations

import numpy as np

from .._utils import as_float, finalize, is_scalar


def rescale_x_to_y(x_scaled_variable, x_reference_from, y_scaling_variable):
    """Rescale values expressed relative to one reference total to another one.

    ``ifelse(x == 0, 0, x / x_reference_from * y_scaling_variable)``: values
    are scaled proportionally from ``x_reference_from`` to
    ``y_scaling_variable``; zeros stay zero (even when the references are
    missing or zero) and missing ``x`` stays missing.

    Parameters
    ----------
    x_scaled_variable : float or array-like
        Values to rescale.
    x_reference_from : float or array-like
        Original reference total of ``x_scaled_variable``.
    y_scaling_variable : float or array-like
        Target reference total.

    Returns
    -------
    float or numpy.ndarray
        Rescaled values, with the length of ``x_scaled_variable`` (as R's
        ``ifelse`` returns a result as long as its test).
    """
    x = as_float(x_scaled_variable)
    ref = as_float(x_reference_from)
    y = as_float(y_scaling_variable)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.where(x == 0, 0.0, x / ref * y)
    out = np.asarray(out, dtype="float64").reshape(-1)
    if is_scalar(x_scaled_variable):
        return finalize(out[:1], True)
    return out[: x.size]
