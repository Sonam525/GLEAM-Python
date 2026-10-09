"""Floating-point comparisons that do not depend on the CPU or numpy build.

numpy evaluates ``exp``, ``log`` and ``pow`` on arrays with SIMD code on some
x86 CPUs and numpy versions, and that code can round the last bit differently
from the scalar C library. The vectorised path and element-by-element scalar
calls of the same ``calc_*`` function can therefore differ by a few units in
the last place (ULP) on one machine and be bit-identical on another. Tests that
compare the two paths use :func:`assert_same_float`; comparisons with R
reference values keep their own tolerances.
"""

from __future__ import annotations

import numpy as np

#: Allowed relative difference between two evaluations of the same formula
#: (about 45 ULP); a real vectorisation error is many orders of magnitude larger.
RTOL = 1e-14


def assert_same_float(actual, expected, rtol: float = RTOL, err_msg: str = "") -> None:
    """Assert equal values up to last-bit rounding, with NaN and infinities in the same places."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=rtol,
        atol=0,
        equal_nan=True,
        err_msg=err_msg,
    )
