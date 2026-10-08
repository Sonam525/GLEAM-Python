"""Helpers for comparing Python results with the R golden outputs.

Golden outputs live in ``tests/golden/<case>/<table>.csv`` and are produced by
``tools/r_reference/generate_golden.R`` from the original R package. A case
whose R call failed holds ``ERROR.txt`` (the R condition message) instead.

Numeric tolerance
-----------------
Numbers must agree within ``rtol = 1e-9`` *relative*, with no absolute term.
A value R computes as exactly zero must therefore be exactly zero in Python
(``-0.0`` equals ``0.0``), and the smallest non-zero values are checked as
strictly as the largest. Every golden table and every randomised scenario of
the documented ``tools/parity`` run passes this rule. An exact comparison
(``rtol = atol = 0``) requires equal numbers (bit-identical apart from the
sign of zero; NaN matches NaN).

Opt-in near-zero rule
---------------------
``assert_frame_matches(..., zero_floor=f)`` (``f > 0``, e.g. ``1e-12``) is a
diagnostic for cancellation noise, used by nothing in the suite: a value
counts as zero when its magnitude is at most ``f`` times the largest
magnitude in the same column among the rows that share the same text
identifiers (species, cohort, variable and commodity names), and two values
that both count as zero match (a zero still does not match a larger value
outside the tolerance). A group whose R values are all zero gives no
scale, so its values must still be exactly zero; the floor is never taken
from the rest of the column. It needs a non-zero ``rtol`` or ``atol``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd

from gleampy._utils import as_float, as_str, is_na
from gleampy.io import read_csv

GOLDEN_DIR = Path(__file__).parent / "golden"

RTOL = 1e-9
#: Default of ``assert_frame_matches(zero_floor=...)``: no near-zero rule.
ZERO_FLOOR = 0.0


def golden_path(case: str, table: str) -> Path:
    """Path of a golden table.

    Parameters
    ----------
    case : str
        Golden case (directory under ``tests/golden``).
    table : str
        Table name (flattened result key, e.g. ``aggregation_results__results_feed``).
    """
    return GOLDEN_DIR / case / f"{table}.csv"


def load_golden(case: str, table: str) -> pd.DataFrame:
    """Read a golden table; fails if R errored for the case.

    Parameters
    ----------
    case : str
        Golden case.
    table : str
        Table name.
    """
    path = golden_path(case, table)
    err = GOLDEN_DIR / case / "ERROR.txt"
    if err.exists():
        raise AssertionError(f"R reference failed for case {case!r}: {err.read_text()}")
    return read_csv(path)


def golden_error(case: str) -> str:
    """The R error message recorded for a case that is expected to fail.

    Parameters
    ----------
    case : str
        Golden case whose directory holds ``ERROR.txt``.
    """
    err = GOLDEN_DIR / case / "ERROR.txt"
    assert err.exists(), f"case {case!r} has no ERROR.txt: R did not fail"
    return err.read_text(encoding="utf-8").strip()


def golden_tables(case: str) -> list[str]:
    """Names of the golden tables of a case, sorted.

    Parameters
    ----------
    case : str
        Golden case.
    """
    return sorted(p.stem for p in (GOLDEN_DIR / case).glob("*.csv"))


def normalize_message(msg: str) -> str:
    """Normalise an R or Python validation message for comparison.

    Removes cli markup (``{.arg x}``, ``{.val x}``), ANSI colour codes,
    bullets and backticks, maps ``≥`` / ``≤`` to ``>=`` / ``<=``, writes
    lists of quoted items with ``", "`` whether R joined them with ``and``
    or a comma, drops Python's vector element index (`` `x`[3] `` becomes
    ``x``) and collapses whitespace and line wraps.

    Parameters
    ----------
    msg : str
        R ``conditionMessage()`` or Python exception message.
    """
    s = re.sub(r"\x1b\[[0-9;]*m", "", msg)
    s = re.sub(r"\{\.(?:arg|val|var|field|code|fn|cls|strong|emph)\s+([^{}]*)\}", r"\1", s)
    s = s.replace("≥", ">=").replace("≤", "<=").replace("‘", "'").replace("’", "'")
    # cli bullets at the start of a line ("✖ ...", or "x ..." / "i ..." in ASCII mode)
    s = re.sub(r"^[ \t]*(?:[✖ℹ•×!*]|[xi](?= [A-Z`\"]))[ \t]+", "", s, flags=re.M)
    s = re.sub(r"`([^`]*)`\[\d+\]", r"\1", s)
    s = s.replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r'"\s*(?:,\s*and|,|and)\s*"', '", "', s)
    s = re.sub(r"'\s*(?:,\s*and|,|and)\s*'", "', '", s)
    return s.rstrip(".")


def quoted_items(msg: str) -> list[str]:
    """The double-quoted items of a message (row labels in R's row-wise checks).

    Parameters
    ----------
    msg : str
        R or Python validation message.
    """
    return re.findall(r'"([^"]*)"', normalize_message(msg))


def _kind(s: pd.Series) -> str:
    if s.dtype.kind in "iuf":
        return "num"
    vals = [v for v in s.tolist() if not is_na(v)]
    if not vals:
        return "empty"
    if all(isinstance(v, (bool, np.bool_)) for v in vals):
        return "bool"
    if all(isinstance(v, Number) and not isinstance(v, (bool, np.bool_)) for v in vals):
        return "num"
    return "str"


def _non_numeric_values(s: pd.Series) -> list:
    """Non-missing elements of ``s`` that are not real numbers (bools included)."""
    if s.dtype.kind in "iuf":
        return []
    if s.dtype.kind == "b":
        return s.dropna().tolist()[:3] or [True]
    return [
        v for v in s.tolist()
        if not is_na(v) and (isinstance(v, (bool, np.bool_)) or not isinstance(v, Number))
    ][:3]


def _group_codes(exp: pd.DataFrame, kinds: dict[str, str]) -> np.ndarray | None:
    """Integer code of each row's combination of text identifiers (None if there are none)."""
    keys = [c for c in exp.columns if kinds[c] == "str"]
    if not keys:
        return None
    key_frame = pd.DataFrame({k: as_str(exp[k]) for k in keys}).fillna("<NA>")
    return key_frame.groupby(keys, sort=False).ngroup().to_numpy()


def zero_floor_per_row(values: np.ndarray, groups: np.ndarray | None, zero_floor: float = ZERO_FLOOR) -> np.ndarray:
    """``zero_floor`` times the scale of each value (see the module docstring).

    The scale of a row is the largest finite magnitude among the rows of its
    group. A group whose values are all zero (or missing) has scale 0, so its
    floor is 0: it never borrows the scale of other groups.

    Parameters
    ----------
    values : numpy.ndarray
        Expected (R) values of one column.
    groups : numpy.ndarray or None
        Group code of each row (rows sharing their text identifiers), or None
        to use the whole column as one group.
    zero_floor : float
        Fraction of the scale at or below which a value counts as zero
        (0 disables the rule).

    Returns
    -------
    numpy.ndarray of float
        The floor of each row, in the units of ``values``.
    """
    absv = np.where(np.isfinite(values), np.abs(values), 0.0)
    if zero_floor == 0 or absv.size == 0:
        return np.zeros_like(absv)
    if groups is None:
        return np.full_like(absv, zero_floor * absv.max())
    group_max = np.zeros(int(groups.max()) + 1)
    np.maximum.at(group_max, groups, absv)
    return zero_floor * group_max[groups]


def _check_tolerances(rtol: float, atol: float, zero_floor: float) -> None:
    if not (rtol >= 0 and atol >= 0 and zero_floor >= 0):
        raise ValueError(f"tolerances must be >= 0: rtol={rtol!r}, atol={atol!r}, zero_floor={zero_floor!r}")
    if zero_floor > 0 and rtol == 0 and atol == 0:
        raise ValueError(
            "an exact comparison (rtol = atol = 0) has no near-zero rule: zero_floor must be 0"
        )


def numbers_close(
    actual: np.ndarray, expected: np.ndarray, floor: np.ndarray | float = 0.0, rtol: float = RTOL, atol: float = 0.0
) -> np.ndarray:
    """Element-wise comparison used for golden numbers (see the module docstring).

    Parameters
    ----------
    actual, expected : numpy.ndarray
        Python and R values.
    floor : numpy.ndarray or float
        Magnitude at or below which a value counts as zero (default 0: no
        near-zero rule, only an exact zero is zero). Must be 0 when
        ``rtol`` and ``atol`` are both 0.
    rtol : float
        Relative tolerance.
    atol : float
        Additional absolute tolerance (0 for the golden tests).

    Returns
    -------
    numpy.ndarray of bool
        True where the values match: identical (NaN with NaN, equal
        infinities), within ``rtol * |expected| + atol``, or both at or
        below ``floor`` (both count as zero). A value at or below the floor
        does not match a larger one outside the tolerance.

    Raises
    ------
    ValueError
        A negative tolerance, or a non-zero floor with ``rtol = atol = 0``.
    """
    av = np.asarray(actual, dtype=float)
    ev = np.asarray(expected, dtype=float)
    floor = np.broadcast_to(np.asarray(floor, dtype=float), ev.shape)
    _check_tolerances(rtol, atol, float(floor.max()) if floor.size else 0.0)
    if floor.size and float(floor.min()) < 0:
        raise ValueError("floor must be >= 0")
    with np.errstate(invalid="ignore", over="ignore"):
        exact = (av == ev) | (np.isnan(av) & np.isnan(ev))
        a_zero = np.abs(av) <= floor
        e_zero = np.abs(ev) <= floor
        rel = np.abs(av - ev) <= rtol * np.abs(ev) + atol
    finite = np.isfinite(av) & np.isfinite(ev)
    return exact | (finite & (rel | (a_zero & e_zero)))


def assert_frame_matches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: Sequence[str] | None = None,
    rtol: float = RTOL,
    atol: float = 0.0,
    zero_floor: float = ZERO_FLOOR,
    check_column_order: bool = True,
    check_row_order: bool = True,
    ignore_columns: Sequence[str] = (),
    label: str = "",
) -> None:
    """Assert a Python result table equals an R golden table.

    Parameters
    ----------
    actual, expected : pandas.DataFrame
        Python result and R golden table.
    sort_by : sequence of str, optional
        Sort both tables by these columns before comparing.
    rtol : float
        Relative tolerance for numbers (default 1e-9).
    atol : float
        Extra absolute tolerance for numbers (default 0; only for callers that
        need one, the golden tests do not).
    zero_floor : float
        Opt-in near-zero rule (default 0: off, a value R has as exactly zero
        must be exactly zero unless ``atol`` allows more). When positive,
        values at most ``zero_floor`` times the largest magnitude of their
        group (rows sharing the text identifiers) count as zero, and two
        such values match; an all-zero group keeps floor 0. Not allowed
        with ``rtol = atol = 0``.
    check_column_order, check_row_order : bool
        Require identical column / row order. When ``check_row_order`` is
        false, both tables are sorted by ``sort_by`` (or all identifier-like
        columns) first.
    ignore_columns : sequence of str
        Columns left out of the comparison.
    label : str
        Prefix for failure messages.

    Notes
    -----
    Where R has a numeric column, the Python column must hold numbers (or
    missing values): text and logical values fail instead of being coerced.
    Logical columns must hold ``True`` / ``False`` / missing, text columns
    compare as text (a number matches its R spelling, ``1`` vs ``"1"``).
    Columns R wrote entirely as ``NA`` must be entirely missing in Python.

    Raises
    ------
    AssertionError
        The tables differ.
    ValueError
        A negative tolerance, or ``zero_floor > 0`` with ``rtol = atol = 0``
        (an exact comparison is always exact).
    """
    _check_tolerances(rtol, atol, zero_floor)
    pre = f"[{label}] " if label else ""
    act = actual.drop(columns=[c for c in ignore_columns if c in actual.columns]).reset_index(drop=True)
    exp = expected.drop(columns=[c for c in ignore_columns if c in expected.columns]).reset_index(drop=True)

    missing = [c for c in exp.columns if c not in act.columns]
    extra = [c for c in act.columns if c not in exp.columns]
    assert not missing and not extra, f"{pre}column mismatch: missing={missing} extra={extra}"
    if check_column_order:
        assert list(act.columns) == list(exp.columns), (
            f"{pre}column order differs:\n  actual  ={list(act.columns)}\n  expected={list(exp.columns)}"
        )
    else:
        act = act[list(exp.columns)]
    assert len(act) == len(exp), f"{pre}row count differs: actual={len(act)} expected={len(exp)}"

    if not check_row_order or sort_by is not None:
        keys = list(sort_by) if sort_by is not None else [c for c in exp.columns if _kind(exp[c]) == "str" or c == "herd_id"]
        act = _sorted(act, keys)
        exp = _sorted(exp, keys)

    kinds = {c: _kind(exp[c]) for c in exp.columns}
    groups = _group_codes(exp, kinds) if zero_floor > 0 else None
    problems = []
    for col in exp.columns:
        e, a = exp[col], act[col]
        k = kinds[col]
        if k == "num":
            bad_vals = _non_numeric_values(a)
            if bad_vals:
                problems.append(f"{col}: R has numbers but Python holds non-numeric values, e.g. {bad_vals!r}")
                continue
            ev, av = as_float(e), as_float(a)
            close = numbers_close(av, ev, zero_floor_per_row(ev, groups, zero_floor), rtol=rtol, atol=atol)
            if not close.all():
                i = int(np.flatnonzero(~close)[0])
                problems.append(
                    f"{col}: {int((~close).sum())} mismatches, first at row {i}: actual={av[i]!r} expected={ev[i]!r}"
                )
        elif k == "empty":
            present = [i for i, v in enumerate(a.tolist()) if not is_na(v)]
            if present:
                i = present[0]
                problems.append(
                    f"{col}: R has only NA but Python has {len(present)} values, first at row {i}: {a.iloc[i]!r}"
                )
        elif k == "bool":
            av_list, ev_list = a.tolist(), e.tolist()
            bad_type = [v for v in av_list if not is_na(v) and not isinstance(v, (bool, np.bool_))]
            if bad_type:
                problems.append(f"{col}: R has logicals but Python holds non-logical values, e.g. {bad_type[:3]!r}")
                continue
            norm = lambda v: None if is_na(v) else bool(v)  # noqa: E731
            bad = [i for i in range(len(ev_list)) if norm(ev_list[i]) != norm(av_list[i])]
            if bad:
                problems.append(
                    f"{col}: {len(bad)} mismatches, first at row {bad[0]}: "
                    f"actual={av_list[bad[0]]!r} expected={ev_list[bad[0]]!r}"
                )
        else:
            ev, av = as_str(e), as_str(a)
            # numbers written by R as e.g. "1" vs Python "1.0" in string-typed id columns
            bad = [i for i in range(len(ev)) if not _same_str(av[i], ev[i])]
            if bad:
                problems.append(f"{col}: {len(bad)} mismatches, first at row {bad[0]}: actual={av[bad[0]]!r} expected={ev[bad[0]]!r}")
    assert not problems, pre + "value mismatches:\n  " + "\n  ".join(problems)


def _same_str(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _sorted(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    tmp = df.copy()
    sk = []
    for k in keys:
        col = f"__sort_{k}"
        if df[k].dtype.kind in "iuf":
            tmp[col] = as_float(df[k])
        else:
            tmp[col] = [("" if is_na(v) else str(v)) for v in df[k].tolist()]
        sk.append(col)
    return tmp.sort_values(sk, kind="mergesort").drop(columns=sk).reset_index(drop=True)


def assert_matches_golden(actual: pd.DataFrame, case: str, table: str, **kwargs) -> None:
    """:func:`assert_frame_matches` against a golden table.

    Parameters
    ----------
    actual : pandas.DataFrame
        Python result.
    case, table : str
        Golden case and table.
    **kwargs
        Passed to :func:`assert_frame_matches`.
    """
    assert_frame_matches(actual, load_golden(case, table), label=f"{case}/{table}", **kwargs)
