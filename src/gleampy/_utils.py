"""Shared helpers that reproduce R / data.table semantics with numpy and pandas.

The R package evaluates its ``calc_*`` functions row by row (``by = .I``) on
scalars. The Python port evaluates the same formulas on whole numpy arrays, so
these helpers take care of the R behaviours that matter for numerical parity:

* ``NA`` propagation (``np.nan`` for numbers, ``None`` for strings);
* numbers are always handled as float64, whatever the pandas dtype (numpy,
  nullable ``Int64`` / ``Float64`` / ``boolean``, categorical, object), so no
  value is ever truncated to an integer;
* ``isTRUE()`` (:func:`is_true`, :func:`is_true_scalar`): only a logical TRUE
  counts, not ``1`` or ``"TRUE"``; ``x %in% TRUE`` (:func:`in_true`), which
  also matches ``1`` and ``"TRUE"``;
* ``ifelse`` / ``fifelse`` returning ``NA`` when the condition is ``NA``;
* ``sum()`` returning ``NA`` when any element is ``NA`` (pandas skips NaN), and
  grouped sums added in row order like data.table's ``gsum``
  (:func:`group_sum`);
* ``merge()`` on data.tables (inner join, NA keys match, result sorted by keys,
  ``.x`` / ``.y`` suffixes);
* ``x[i, on = key, x.col]`` look-ups of herd-level values for cohort rows
  (:func:`lookup`; :func:`lookup_columns` / :class:`Lookup` join once for
  several columns).
"""

from __future__ import annotations

import numbers
import sys
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

ArrayLike = Any

# --------------------------------------------------------------------------
# Scalars and arrays
# --------------------------------------------------------------------------


def is_scalar(x: Any) -> bool:
    """True for Python/numpy scalars and ``None`` (R length-1 vectors)."""
    return x is None or np.ndim(x) == 0


def all_scalar(*xs: Any) -> bool:
    return all(is_scalar(x) for x in xs)


def as_float(x: ArrayLike) -> np.ndarray:
    """Convert to a float64 numpy array; ``None`` / ``pd.NA`` / ``NaT`` -> ``nan``.

    Booleans become 0/1 like R's coercion of logicals to numeric. The result
    is always float64, whatever the input dtype (numpy int/float/bool, pandas
    nullable ``Int64`` / ``Float64`` / ``boolean`` with missing values,
    categorical, object), so no value is ever truncated to an integer.

    Parameters
    ----------
    x : scalar, sequence, numpy.ndarray, pandas.Series or pandas.Index
        Values to convert. In a Series or Index, non-numeric strings become
        ``nan``; in other inputs they raise ``ValueError``.

    Returns
    -------
    numpy.ndarray
        float64 array of the same shape (0-d for a scalar).
    """
    if isinstance(x, (pd.Series, pd.Index)):
        if isinstance(x.dtype, pd.CategoricalDtype):
            x = x.astype(object)
        if x.dtype.kind in "iufb":
            if isinstance(x.dtype, np.dtype):
                return x.to_numpy(dtype="float64")
            # nullable extension dtypes: pandas < 2.2 needs an explicit na_value
            return x.to_numpy(dtype="float64", na_value=np.nan)
        return pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    if x is None or x is pd.NA:
        return np.array(np.nan)
    arr = np.asarray(x)
    if arr.dtype == object:
        na = pd.isna(arr)
        out = np.full(arr.shape, np.nan)
        if (~na).any():
            out[~na] = arr[~na].astype("float64")
        return out
    return arr.astype("float64", copy=False)


def _object_values(x: ArrayLike) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy(dtype=object)
    return np.asarray(x, dtype=object)


def as_str(x: ArrayLike) -> np.ndarray:
    """Convert to an object array of ``str`` with missing values as ``None``."""
    if x is None:
        return np.array(None, dtype=object)
    vals = _object_values(x)
    na = pd.isna(vals)
    out = vals.copy() if vals.ndim else np.array(vals.item(), dtype=object)
    if np.ndim(na) == 0:
        return np.array(None if na else str(out.item()), dtype=object)
    present = out[~na]
    if present.size and pd.api.types.infer_dtype(present, skipna=False) != "string":
        out[~na] = np.array([str(v) for v in present], dtype=object)
    out[na] = None
    return out


_LOGICAL_STR = {"TRUE": True, "T": True, "FALSE": False, "F": False}


def as_bool(x: ArrayLike) -> np.ndarray:
    """Convert to an object array of ``True`` / ``False`` / ``None`` (R ``as.logical``).

    Numbers are ``True`` when non-zero and the strings ``"TRUE"``/``"T"`` and
    ``"FALSE"``/``"F"`` (any case, surrounding blanks ignored) are converted;
    anything else is ``None`` (NA). This is a lenient *conversion*: to test a
    flag the way R's ``isTRUE()`` does, use :func:`is_true`, which accepts
    only logical ``TRUE``.

    Parameters
    ----------
    x : scalar, sequence, numpy.ndarray, pandas.Series or pandas.Index
        Values to convert.

    Returns
    -------
    numpy.ndarray
        Object array of the same shape holding ``True``, ``False`` or ``None``.
    """
    if x is None:
        return np.array(None, dtype=object)
    vals = _object_values(x)
    shape = vals.shape
    flat = vals.reshape(-1)
    na = pd.isna(flat)
    out = np.empty(flat.shape, dtype=object)
    out[na] = None
    present = flat[~na]
    if present.size:
        kind = pd.api.types.infer_dtype(present, skipna=False)
        if kind == "boolean":
            conv = present.astype(bool).tolist()
        elif kind in ("integer", "floating", "mixed-integer-float", "decimal"):
            conv = (present.astype("float64") != 0).tolist()
        elif kind == "string":
            conv = [_LOGICAL_STR.get(v.strip().upper()) for v in present]
        else:
            conv = [
                _LOGICAL_STR.get(v.strip().upper()) if isinstance(v, str) else bool(v) for v in present
            ]
        out[~na] = np.array(conv + [None], dtype=object)[:-1]
    return out.reshape(shape)


def _is_logical_true(v: Any) -> bool:
    return v is True or (isinstance(v, np.bool_) and bool(v))


def is_true(x: ArrayLike) -> np.ndarray:
    """Element-wise R ``isTRUE()``: ``True`` only where the value is logical TRUE.

    An element counts as TRUE only if it is a Python ``bool`` or
    ``numpy.bool_`` equal to ``True`` (or a ``True`` of a bool / nullable
    ``boolean`` array). Numbers (``1``, ``1.0``), strings (``"TRUE"``) and
    missing values are ``False``, as ``isTRUE(1)`` and ``isTRUE("TRUE")`` are
    ``FALSE`` in R.

    Parameters
    ----------
    x : scalar, sequence, numpy.ndarray, pandas.Series or pandas.Index
        Values to test.

    Returns
    -------
    numpy.ndarray
        bool array of the same shape (0-d for a scalar).
    """
    if isinstance(x, (bool, np.bool_)):
        return np.array(bool(x))
    if x is None:
        return np.array(False)
    if isinstance(x, (pd.Series, pd.Index)):
        dt = x.dtype
        if dt == bool:
            return x.to_numpy(dtype=bool, copy=True)
        if dt.kind == "b":  # nullable "boolean" (or pyarrow bool): NA is not TRUE
            return x.to_numpy(dtype=bool, na_value=False)
        if isinstance(dt, pd.CategoricalDtype) or dt == object:
            vals = x.to_numpy(dtype=object)
        else:  # numeric, string, datetime, ... columns are never logical TRUE
            return np.zeros(len(x), dtype=bool)
    elif isinstance(x, np.ndarray):
        vals = x
    else:
        # lists keep their element types (np.asarray([True, 1]) would give ints)
        vals = np.asarray(x, dtype=object)
    if vals.dtype == bool:
        return np.array(vals, dtype=bool)
    if vals.dtype != object:
        return np.zeros(vals.shape, dtype=bool)
    flat = vals.reshape(-1)
    out = np.fromiter((_is_logical_true(v) for v in flat), dtype=bool, count=flat.size)
    return out.reshape(vals.shape)


def is_true_scalar(x: Any) -> bool:
    """R ``isTRUE(x)`` for a whole argument (a pipeline switch or flag).

    ``True`` only for a single logical ``TRUE``: Python ``True``,
    ``numpy.True_`` or a one-element bool array holding ``True``.

    Parameters
    ----------
    x : Any
        The value to test.

    Returns
    -------
    bool
        Whether ``x`` is a single logical TRUE.
    """
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.ndarray) and x.dtype == bool and x.size == 1:
        return bool(x.reshape(-1)[0])
    return False


def _in_true_objects(flat: np.ndarray) -> np.ndarray:
    """:func:`in_true` for a 1-d object array (R's coercion to a common type)."""
    out = np.zeros(flat.shape, dtype=bool)
    na = np.asarray(pd.isna(flat), dtype=bool)
    present = flat[~na]
    if not present.size:
        return out
    if any(isinstance(v, str) for v in present):
        # character vector: TRUE is coerced to "TRUE"; numbers never match
        conv = [v == "TRUE" if isinstance(v, str) else _is_logical_true(v) for v in present]
    else:
        # logical / numeric vector: TRUE is coerced to 1
        conv = [
            _is_logical_true(v)
            or (isinstance(v, numbers.Number) and not isinstance(v, (bool, np.bool_)) and v == 1)
            for v in present
        ]
    out[~na] = conv
    return out


def in_true(x: ArrayLike) -> np.ndarray:
    """Element-wise R ``x %in% TRUE`` (``match(x, TRUE, nomatch = 0) > 0``).

    Unlike :func:`is_true` (R ``isTRUE()``), ``%in%`` first coerces ``x``
    and ``TRUE`` to a common type, so besides a logical ``TRUE`` it matches
    the number ``1`` (``TRUE`` becomes ``1`` in a numeric vector) and the
    string ``"TRUE"`` (``TRUE`` becomes ``"TRUE"`` in a character vector).
    ``"T"``, ``"true"``, ``"1"``, ``2`` and missing values do not match. R's
    run validators use it to decide whether egg columns are required
    (``any(is_egg_producing %in% TRUE)``).

    An object array is typed like an R vector: if it holds any string it is
    compared as character (only ``"TRUE"`` and logical ``True`` match),
    otherwise as numbers. Categorical values are compared as their plain
    values.

    Parameters
    ----------
    x : scalar, sequence, numpy.ndarray, pandas.Series or pandas.Index
        Values to test.

    Returns
    -------
    numpy.ndarray
        bool array of the same shape (0-d for a scalar).
    """
    if isinstance(x, (pd.Series, pd.Index)):
        dt = x.dtype
        if isinstance(dt, pd.CategoricalDtype) or dt == object:
            vals = x.to_numpy(dtype=object)
        elif dt.kind == "b":  # bool, nullable "boolean": NA does not match
            return x.to_numpy(dtype=bool, na_value=False)
        elif dt.kind in "iufc":  # numpy or nullable numbers: NA does not match
            vals = x.to_numpy(dtype="complex128" if dt.kind == "c" else "float64", na_value=np.nan)
            return vals == 1
        elif dt.kind in "OSU" or pd.api.types.is_string_dtype(dt):
            return x.to_numpy(dtype=object, na_value=None) == "TRUE"
        else:  # datetimes, intervals, ... never match TRUE
            return np.zeros(len(x), dtype=bool)
    elif isinstance(x, np.ndarray):
        vals = x
    elif isinstance(x, (str, bytes)) or np.ndim(x) == 0:
        vals = np.array(x, dtype=object)
    else:
        # lists keep their element types (np.asarray([True, "a"]) would give strings)
        vals = np.asarray(x, dtype=object)
    if vals.dtype == bool:
        return np.array(vals, dtype=bool)
    if vals.dtype.kind in "iufc":
        return np.asarray(vals == 1, dtype=bool)
    if vals.dtype.kind == "U":
        return np.asarray(vals == "TRUE", dtype=bool)
    if vals.dtype != object:
        return np.zeros(vals.shape, dtype=bool)
    return _in_true_objects(vals.reshape(-1)).reshape(vals.shape)


def is_na(v: Any) -> bool:
    """Scalar missing-value test covering ``None``, ``nan``, ``pd.NA`` and ``NaT``."""
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    return False


def isna(x: ArrayLike) -> np.ndarray:
    """Vectorised ``is.na()`` for numeric, string or object arrays."""
    if isinstance(x, (pd.Series, pd.Index)):
        return x.isna().to_numpy()
    arr = np.asarray(x)
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    if arr.dtype.kind in "iub":
        return np.zeros(arr.shape, dtype=bool)
    return np.asarray(pd.isna(arr), dtype=bool)


def broadcast(*xs: ArrayLike) -> list[np.ndarray]:
    """``np.broadcast_arrays`` that keeps object (string) arrays intact."""
    arrays = [x if isinstance(x, np.ndarray) else np.asarray(x) for x in xs]
    shape = np.broadcast_shapes(*(a.shape for a in arrays))
    return [np.broadcast_to(a, shape) for a in arrays]


def finalize(out: np.ndarray | float, scalar: bool) -> Any:
    """Return a Python float for all-scalar calls, otherwise a 1-d numpy array."""
    if scalar:
        arr = np.asarray(out)
        if arr.size != 1:
            return arr
        v = arr.reshape(-1)[0]
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    return np.atleast_1d(np.asarray(out))


def finalize_dict(d: dict[str, Any], scalar: bool) -> dict[str, Any]:
    return {k: finalize(v, scalar) for k, v in d.items()}


def isin(x: ArrayLike, values: Iterable[Any]) -> np.ndarray:
    """R ``%in%`` (``NA %in% c(...)`` is ``FALSE`` unless NA is in the set)."""
    values = list(values)
    has_na = any(is_na(v) for v in values)
    present = [v for v in values if not is_na(v)]
    arr = x.to_numpy() if isinstance(x, (pd.Series, pd.Index)) else np.asarray(x)
    if arr.dtype.kind in "iuf":
        num = [v for v in present if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)]
        res = np.isin(arr, num)
        if has_na and arr.dtype.kind == "f":
            res |= np.isnan(arr)
        return res
    obj = np.asarray(arr, dtype=object)
    flat = obj.reshape(-1)
    res = pd.Series(flat).isin(present).to_numpy()
    na = pd.isna(flat)
    res = np.where(na, has_na, res)
    return res.reshape(obj.shape)


def ifelse(cond: ArrayLike, yes: ArrayLike, no: ArrayLike) -> np.ndarray:
    """R ``fifelse(cond, yes, no)`` on floats: ``NA`` where ``cond`` is ``NA``.

    ``cond`` may be a bool array or a float array with ``nan`` marking NA
    (e.g. ``np.where(np.isnan(a), np.nan, a > 0)``).
    """
    c = np.asarray(cond)
    yes_a = as_float(yes)
    no_a = as_float(no)
    if c.dtype.kind == "f":
        na = np.isnan(c)
        cb = np.where(na, False, c != 0)
    elif c.dtype == object:
        na = isna(c)
        cb = is_true(c)
    else:
        na = np.zeros(c.shape, dtype=bool)
        cb = c.astype(bool)
    out = np.where(cb, yes_a, no_a)
    return np.where(na, np.nan, out)


def na_cmp(a: ArrayLike, op: str, b: ArrayLike) -> np.ndarray:
    """Comparison that returns ``nan`` where either side is ``NA`` (R semantics).

    The result is a float array (1.0 / 0.0 / nan) suitable for :func:`ifelse`.
    """
    a_ = as_float(a)
    b_ = as_float(b)
    with np.errstate(invalid="ignore"):
        res = {
            "<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
            "==": np.equal, "!=": np.not_equal,
        }[op](a_, b_).astype("float64")
    return np.where(np.isnan(a_) | np.isnan(b_), np.nan, res)


def r_sum(x: ArrayLike, na_rm: bool = False) -> float:
    """R ``sum()``: ``NA`` if any element is ``NA`` unless ``na_rm``."""
    arr = as_float(x)
    if na_rm:
        return float(np.nansum(arr))
    return float(np.sum(arr))


def group_sum(values: ArrayLike, codes: ArrayLike, ngroups: int, na_rm: bool = False) -> np.ndarray:
    """Per-group sums by plain sequential double addition in row order.

    Reproduces data.table's grouped ``sum()`` (``gsum``): each group total
    starts at 0 and adds its values one by one in row order, with no
    compensated or pairwise summation (pandas' ``groupby().sum()`` and
    ``np.sum`` round differently). Use it where R compares a group sum with a
    tolerance, e.g. ration or manure fractions summing to 1.

    Parameters
    ----------
    values : array-like
        Values to add (converted with :func:`as_float`).
    codes : array-like of int
        Group number (0 .. ``ngroups - 1``) of each value.
    ngroups : int
        Number of groups.
    na_rm : bool, default False
        Skip missing values (``sum(x, na.rm = TRUE)``); otherwise a group
        with a missing value sums to NaN, like R.

    Returns
    -------
    numpy.ndarray
        float64 array of length ``ngroups``.
    """
    v = np.atleast_1d(as_float(values))
    c = np.atleast_1d(np.asarray(codes, dtype=np.intp))
    if na_rm:
        keep = ~np.isnan(v)
        v, c = v[keep], c[keep]
    # np.bincount accumulates `out[c[i]] += v[i]` in a plain C loop over i
    # (tests/test_utils.py checks it against an explicit sequential loop).
    return np.bincount(c, weights=v, minlength=int(ngroups)).astype("float64", copy=False)


def pmax(*xs: ArrayLike) -> np.ndarray:
    """R ``pmax`` (NA-propagating)."""
    out = as_float(xs[0])
    for x in xs[1:]:
        out = np.maximum(out, as_float(x))
    return out


def pmin(*xs: ArrayLike) -> np.ndarray:
    """R ``pmin`` (NA-propagating)."""
    out = as_float(xs[0])
    for x in xs[1:]:
        out = np.minimum(out, as_float(x))
    return out


def normalize_rate(x: ArrayLike, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    """Clamp rate-like inputs to ``[lower, upper]`` (``pmax(lower, pmin(upper, x))``)."""
    return pmax(lower, pmin(upper, x))


def safe_divide(num: ArrayLike, den: ArrayLike) -> np.ndarray:
    """Plain IEEE division (R semantics: x/0 -> Inf, 0/0 -> NaN) without warnings."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(as_float(num), as_float(den))


# --------------------------------------------------------------------------
# data.table-like table operations
# --------------------------------------------------------------------------


def copy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """``data.table::copy`` / ``as.data.table`` for an input table."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df.copy(deep=True).reset_index(drop=True)


def _lookup_keys(on: str | Sequence[str]) -> list[str]:
    return [on] if isinstance(on, str) else list(on)


def _missing_lookup_column(col: str, keys: list[str]) -> KeyError:
    return KeyError(
        f"column '{col}' is not found in the herd-level table "
        f"(needed for the join on {keys})"
    )


class Lookup:
    """Join index of ``left`` rows into ``right``, built once for many columns.

    ``Lookup(left, right, on)(col)`` returns exactly what
    ``lookup(left, right, col, on)`` returns, but the key normalisation and
    the hash join are done only once, in the constructor.

    Parameters
    ----------
    left : pandas.DataFrame
        Table whose rows receive the values (e.g. ``cohort_level_data``).
    right : pandas.DataFrame
        Table the values come from (e.g. ``herd_level_data``); the first row
        of each key is used.
    on : str or sequence of str, default "herd_id"
        Join key column(s), present in both tables.
    na_matches : bool, default True
        Whether an NA key matches an NA key. ``True`` reproduces a
        data.table join (``right[left, on = key]``); ``False`` reproduces a
        subset by equality (``right[key == value]``), where ``NA == NA`` is
        NA and selects no row, so ``left`` rows with an NA key get NA.
    """

    def __init__(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        on: str | Sequence[str] = "herd_id",
        na_matches: bool = True,
    ) -> None:
        self.keys = _lookup_keys(on)
        self.right = right.drop_duplicates(subset=self.keys, keep="first")
        self.nrow = len(left)
        if len(self.keys) == 1:
            k = self.keys[0]
            left_keys = [_key_values(left[k])]
            idx = pd.Index(_key_values(self.right[k])).get_indexer(left_keys[0])
        else:
            r_idx = pd.MultiIndex.from_arrays([_key_values(self.right[k]) for k in self.keys])
            left_keys = [_key_values(left[k]) for k in self.keys]
            idx = r_idx.get_indexer(pd.MultiIndex.from_arrays(left_keys))
        if not na_matches:
            left_na = np.zeros(self.nrow, dtype=bool)
            for kv in left_keys:
                left_na |= np.asarray(pd.isna(kv), dtype=bool)
            idx = np.where(left_na, -1, idx)
        self.index = idx
        self.hit = idx >= 0

    def __call__(self, col: str) -> np.ndarray:
        """Values of ``right[col]`` for each row of ``left`` (NA where unmatched)."""
        if col not in self.right.columns:
            raise _missing_lookup_column(col, self.keys)
        vals = _lookup_values(self.right[col])
        out = np.empty(self.nrow, dtype=vals.dtype if vals.dtype.kind == "f" else object)
        hit = self.hit
        out[hit] = vals[self.index[hit]]
        out[~hit] = np.nan if out.dtype.kind == "f" else None
        return out


def _lookup_values(s: pd.Series) -> np.ndarray:
    """Column values for :class:`Lookup`: numbers as float64, logicals as objects."""
    dt = s.dtype
    if not isinstance(dt, np.dtype) and dt.kind in "iuf":  # nullable Int64 / Float64
        return s.to_numpy(dtype="float64", na_value=np.nan)
    if isinstance(dt, pd.BooleanDtype):
        return s.to_numpy(dtype=object, na_value=None)
    vals = s.to_numpy()
    if vals.dtype.kind in "iub":
        vals = vals.astype("float64") if vals.dtype.kind != "b" else vals.astype(object)
    return vals


def lookup(
    left: pd.DataFrame,
    right: pd.DataFrame,
    col: str,
    on: str | Sequence[str] = "herd_id",
    na_matches: bool = True,
) -> np.ndarray:
    """Values of ``right[col]`` matched to each row of ``left`` on ``on``.

    Mirrors ``right[left, on = on, x.col]`` (``herd_level_data[.SD, on =
    "herd_id", x.live_weight_at_birth]``): rows of ``left`` without a match get
    ``NA``. ``right`` is expected to have unique keys (validated upstream);
    if it does not, the first match is used. Integer herd ids match float
    ones and NA keys match NA keys, as in data.table (unless ``na_matches``
    is false).

    Parameters
    ----------
    left : pandas.DataFrame
        Table whose rows receive the values (e.g. ``cohort_level_data``).
    right : pandas.DataFrame
        Table the values come from (e.g. ``herd_level_data``).
    col : str
        Column of ``right`` to look up.
    on : str or sequence of str, default "herd_id"
        Join key column(s).
    na_matches : bool, default True
        ``False`` mirrors ``right[key == value]`` instead of a join: an NA
        key in ``left`` matches nothing (see :class:`Lookup`).

    Returns
    -------
    numpy.ndarray
        One value per row of ``left``: float64 for numeric columns (integer
        and nullable numeric dtypes included), otherwise an object array
        (``None`` where unmatched).

    Raises
    ------
    KeyError
        If ``col`` is missing from ``right``, as data.table does.

    See Also
    --------
    lookup_columns : several columns with a single join.
    """
    keys = _lookup_keys(on)
    if col not in right.columns:
        raise _missing_lookup_column(col, keys)
    return Lookup(left, right, keys, na_matches)(col)


def lookup_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cols: Iterable[str],
    on: str | Sequence[str] = "herd_id",
    na_matches: bool = True,
) -> dict[str, np.ndarray]:
    """:func:`lookup` for several columns, building the join index only once.

    Parameters
    ----------
    left : pandas.DataFrame
        Table whose rows receive the values (e.g. ``cohort_level_data``).
    right : pandas.DataFrame
        Table the values come from (e.g. ``herd_level_data``).
    cols : iterable of str
        Columns of ``right`` to look up.
    on : str or sequence of str, default "herd_id"
        Join key column(s).
    na_matches : bool, default True
        ``False`` mirrors ``right[key == value]``: an NA key in ``left``
        matches nothing (see :class:`Lookup`).

    Returns
    -------
    dict of str to numpy.ndarray
        ``{col: lookup(left, right, col, on)}`` in the order of ``cols``.

    Raises
    ------
    KeyError
        For the first column of ``cols`` missing from ``right``.
    """
    cols = list(cols)
    keys = _lookup_keys(on)
    for c in cols:
        if c not in right.columns:
            raise _missing_lookup_column(c, keys)
    lk = Lookup(left, right, keys, na_matches)
    return {c: lk(c) for c in cols}


def _categorical_values(s: pd.Series) -> np.ndarray:
    """Plain values of a categorical Series (float64 for numeric categories)."""
    cats = s.cat.categories
    codes = s.cat.codes.to_numpy()
    if cats.dtype.kind in "iuf":
        out = cats.to_numpy(dtype="float64")[np.where(codes < 0, 0, codes)] if len(cats) else np.zeros(len(codes))
        out[codes < 0] = np.nan
        return out
    return s.to_numpy(dtype=object)


def _key_values(s: pd.Series) -> np.ndarray:
    """Normalise join keys so int/float herd ids match and NA keys match NA."""
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        arr = _categorical_values(s)
    elif not isinstance(dt, np.dtype) and dt.kind in "iuf":  # nullable Int64 / Float64
        return s.to_numpy(dtype="float64", na_value=np.nan)
    else:
        arr = s.to_numpy()
    if arr.dtype.kind in "iuf":
        return arr.astype("float64")
    return np.array([None if is_na(v) else v for v in arr], dtype=object)


def merge_dt(
    x: pd.DataFrame,
    y: pd.DataFrame,
    by: str | Sequence[str] | None = None,
    all_x: bool = False,
    all_y: bool = False,
    sort: bool = True,
    suffixes: tuple[str, str] = (".x", ".y"),
) -> pd.DataFrame:
    """Reproduce ``data.table::merge(x, y, by = by, all.x, all.y)``.

    * inner join by default; ``all_x`` / ``all_y`` give left / right / outer;
    * ``NA`` keys match each other (data.table semantics);
    * result columns: ``by`` columns, then the other columns of ``x``, then the
      other columns of ``y``; clashing names get ``.x`` / ``.y`` suffixes;
    * when ``sort`` is true, rows are ordered by the ``by`` columns with ``NA``
      first (data.table's ``forder``), otherwise in ``x`` order;
    * an object-dtype key holding only numbers joins a numeric key as float64
      (R holds such a column as numeric).

    Parameters
    ----------
    x, y : pandas.DataFrame
        Left and right tables.
    by : str or sequence of str, optional
        Join columns; default: the columns the two tables share.
    all_x, all_y : bool, default False
        Keep unmatched rows of ``x`` / ``y`` (left, right or outer join).
    sort : bool, default True
        Order the result by the join columns (``NA`` first).
    suffixes : tuple of str, default (".x", ".y")
        Suffixes of non-key columns present in both tables.

    Returns
    -------
    pandas.DataFrame
        The joined table.
    """
    if by is None:
        by = [c for c in x.columns if c in set(y.columns)]
    keys = [by] if isinstance(by, str) else list(by)
    how = "outer" if (all_x and all_y) else "left" if all_x else "right" if all_y else "inner"

    xk = x.reset_index(drop=True).copy()
    yk = y.reset_index(drop=True).copy()
    tmp_keys = [f"__key_{i}__" for i in range(len(keys))]
    for k, t in zip(keys, tmp_keys):
        xk[t] = _merge_key(xk[k])
        yk[t] = _merge_key(yk[k])
        if (xk[t].dtype == object) != (yk[t].dtype == object):
            # an object key of numbers joins a numeric key as numbers (R holds it numeric)
            xk[t], yk[t] = _numeric_object_key(xk[t]), _numeric_object_key(yk[t])
    xk["__xrow__"] = np.arange(len(xk))
    yk["__yrow__"] = np.arange(len(yk))

    x_other = [c for c in x.columns if c not in keys]
    y_other = [c for c in y.columns if c not in keys]
    clash = set(x_other) & set(y_other)
    xr = {c: c + suffixes[0] for c in clash}
    yr = {c: c + suffixes[1] for c in clash}

    left = xk[tmp_keys + keys + x_other + ["__xrow__"]].rename(columns=xr)
    right = yk[tmp_keys + keys + y_other + ["__yrow__"]].rename(columns=yr)
    right = right.rename(columns={k: f"__ykey_{k}" for k in keys})
    m = pd.merge(left, right, on=tmp_keys, how=how, sort=False)

    # Key columns come from x, or from y for right-only rows (outer/right join).
    for k in keys:
        yv = m.pop(f"__ykey_{k}")
        if how in ("outer", "right"):
            m[k] = m[k].where(m["__xrow__"].notna(), yv)

    if sort:
        m = m.sort_values(
            tmp_keys + ["__xrow__", "__yrow__"], na_position="first", kind="mergesort"
        )
    else:
        m = m.sort_values(["__xrow__", "__yrow__"], na_position="last", kind="mergesort")

    out_cols = keys + [xr.get(c, c) for c in x_other] + [yr.get(c, c) for c in y_other]
    out = m[out_cols].reset_index(drop=True)
    for k in keys:
        if k in x.columns and x[k].dtype.kind in "iu" and not out[k].isna().any():
            out[k] = out[k].astype(x[k].dtype)
    return out


def _numeric_object_key(s: pd.Series) -> pd.Series:
    """An object merge key holding only numbers (or only NA) as float64; other keys unchanged."""
    if s.dtype != object:
        return s
    present = s[s.notna()]
    if all(
        isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
        for v in present
    ):
        return pd.Series([np.nan if is_na(v) else float(v) for v in s], index=s.index, dtype="float64")
    return s


def _merge_key(s: pd.Series) -> pd.Series:
    """Merge key with numeric types unified and NA represented consistently."""
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        vals = _categorical_values(s)
        if vals.dtype.kind == "f":
            return pd.Series(vals, index=s.index)
        s = pd.Series(vals, index=s.index, dtype=object)
    elif dt.kind in "iufb":
        if isinstance(dt, np.dtype):
            return s.astype("float64")
        # nullable Int64 / Float64 / boolean with missing values
        return pd.Series(s.to_numpy(dtype="float64", na_value=np.nan), index=s.index)
    return s.astype(object).where(s.notna(), None)


def order_by(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """``setorderv`` / ``setkeyv`` ordering: ascending, NA first, stable."""
    return df.sort_values(list(cols), na_position="first", kind="mergesort").reset_index(drop=True)


def rbind_fill(frames: Sequence[pd.DataFrame | None]) -> pd.DataFrame:
    """``data.table::rbindlist(..., use.names = TRUE, fill = TRUE)``."""
    frames = [f for f in frames if f is not None]
    if not frames:
        return pd.DataFrame()
    cols: list[str] = []
    for f in frames:
        for c in f.columns:
            if c not in cols:
                cols.append(c)
    non_empty = [f for f in frames if len(f.columns)]
    out = pd.concat([f.reindex(columns=cols) for f in non_empty], ignore_index=True, sort=False)
    return out


# --------------------------------------------------------------------------
# Progress indicator (cli::cli_status / cli_alert_success replacement)
# --------------------------------------------------------------------------


class Progress:
    """Minimal replacement for the ``cli`` progress messages of the R package."""

    def __init__(self, show_indicator: bool) -> None:
        self.show = bool(show_indicator)

    def header(self, msg: str) -> None:
        if self.show:
            print(f"== {msg} ==", file=sys.stderr)

    def status(self, msg: str) -> None:
        if self.show:
            print(f"... {msg}", file=sys.stderr)

    def success(self, msg: str) -> None:
        if self.show:
            print(f"v {msg}", file=sys.stderr)
