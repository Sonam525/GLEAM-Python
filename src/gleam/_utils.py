"""Shared helpers that reproduce R / data.table semantics with numpy and pandas.

The R package evaluates its ``calc_*`` functions row by row (``by = .I``) on
scalars. The Python port evaluates the same formulas on whole numpy arrays, so
these helpers take care of the R behaviours that matter for numerical parity:

* ``NA`` propagation (``np.nan`` for numbers, ``None`` for strings);
* ``ifelse`` / ``fifelse`` returning ``NA`` when the condition is ``NA``;
* ``sum()`` returning ``NA`` when any element is ``NA`` (pandas skips NaN);
* ``merge()`` on data.tables (inner join, NA keys match, result sorted by keys,
  ``.x`` / ``.y`` suffixes);
* ``x[i, on = key, x.col]`` look-ups of herd-level values for cohort rows.
"""

from __future__ import annotations

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

    Booleans become 0/1 like R's coercion of logicals to numeric.
    """
    if isinstance(x, (pd.Series, pd.Index)):
        if x.dtype.kind in "iufb":
            return x.to_numpy(dtype="float64")
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
    """Convert a logical vector to an object array of ``True`` / ``False`` / ``None``.

    Use :func:`is_true` for R's ``isTRUE()`` semantics (``NA`` -> ``False``).
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


def is_true(x: ArrayLike) -> np.ndarray:
    """Element-wise ``isTRUE()``: ``True`` only where the value is logical TRUE."""
    b = as_bool(x)
    return np.asarray(b == True, dtype=bool)  # noqa: E712 (None == True is False)


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


def lookup(
    left: pd.DataFrame,
    right: pd.DataFrame,
    col: str,
    on: str | Sequence[str] = "herd_id",
) -> np.ndarray:
    """Values of ``right[col]`` matched to each row of ``left`` on ``on``.

    Mirrors ``right[left, on = on, x.col]`` (``herd_level_data[.SD, on =
    "herd_id", x.live_weight_at_birth]``): rows of ``left`` without a match get
    ``NA``. ``right`` is expected to have unique keys (validated upstream);
    if it does not, the first match is used. Raises ``KeyError`` if ``col`` is
    missing from ``right``, as data.table does.
    """
    keys = [on] if isinstance(on, str) else list(on)
    if col not in right.columns:
        raise KeyError(
            f"column '{col}' is not found in the herd-level table "
            f"(needed for the join on {keys})"
        )
    r = right.drop_duplicates(subset=keys, keep="first")
    if len(keys) == 1:
        idx = pd.Index(_key_values(r[keys[0]])).get_indexer(_key_values(left[keys[0]]))
    else:
        r_idx = pd.MultiIndex.from_arrays([_key_values(r[k]) for k in keys])
        l_idx = pd.MultiIndex.from_arrays([_key_values(left[k]) for k in keys])
        idx = r_idx.get_indexer(l_idx)
    vals = r[col].to_numpy()
    if vals.dtype.kind in "iub":
        vals = vals.astype("float64") if vals.dtype.kind != "b" else vals.astype(object)
    out = np.empty(len(left), dtype=vals.dtype if vals.dtype.kind == "f" else object)
    hit = idx >= 0
    out[hit] = vals[idx[hit]]
    out[~hit] = np.nan if out.dtype.kind == "f" else None
    return out


def _key_values(s: pd.Series) -> np.ndarray:
    """Normalise join keys so int/float herd ids match and NA keys match NA."""
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
      first (data.table's ``forder``), otherwise in ``x`` order.
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


def _merge_key(s: pd.Series) -> pd.Series:
    """Merge key with numeric types unified and NA represented consistently."""
    if s.dtype.kind in "iufb":
        return s.astype("float64")
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
