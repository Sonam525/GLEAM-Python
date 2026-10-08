"""Shared validation helpers (port of ``R/validate_shared_functions.R``).

The R package validates scalar inputs inside every row-wise ``calc_*`` call.
The Python port validates whole arrays at once; each helper accepts a scalar
or an array and raises :class:`GleamValidationError` on the first violation,
with messages that follow the R wording.

Validation can be switched off globally (feature from the
``feature/optional-validation-rule`` branch) with the ``validate_inputs``
argument of every ``run_*`` function, or with :func:`validation_disabled`.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import sys
import warnings
from collections.abc import Iterable, Iterator, Sequence
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import as_bool, as_float, as_str, is_na, is_scalar, isna


class GleamValidationError(ValueError):
    """Raised when GLEAM inputs are invalid (``cli::cli_abort`` in R)."""


class GleamWarning(UserWarning):
    """Warning category for GLEAM (``cli::cli_warn`` in R)."""


def abort(msg: str) -> None:
    """Raise :class:`GleamValidationError` (``cli::cli_abort``).

    Parameters
    ----------
    msg : str
        Message; runs of white space (including line breaks) collapse to one
        space.
    """
    raise GleamValidationError(" ".join(msg.split()))


def _internal_dirs() -> tuple[str, ...]:
    import gleampy

    return (
        os.path.normcase(os.path.join(os.path.dirname(os.path.abspath(gleampy.__file__)), "")),
        os.path.normcase(os.path.abspath(contextlib.__file__)),
    )


def _caller_stacklevel() -> int:
    """``stacklevel`` of the first frame outside the gleampy package and contextlib.

    Counted for a ``warnings.warn`` call made directly in :func:`warn`.
    """
    internal = _internal_dirs()
    level = 1
    frame = sys._getframe(1)  # warn()
    while frame is not None:
        fname = os.path.normcase(os.path.abspath(frame.f_code.co_filename))
        if not fname.startswith(internal):
            return level
        frame = frame.f_back
        level += 1
    return level


def warn(msg: str) -> None:
    """Issue a :class:`GleamWarning` (``cli::cli_warn``) attributed to the user's code.

    The warning is reported at the first stack frame outside the ``gleampy``
    package (and ``contextlib``), i.e. at the user's call of the public
    function that raised it, so location-based warning filters work.

    Parameters
    ----------
    msg : str
        Message; runs of white space (including line breaks) collapse to one
        space.
    """
    warnings.warn(" ".join(msg.split()), GleamWarning, stacklevel=_caller_stacklevel())


# --------------------------------------------------------------------------
# Global validation switch
# --------------------------------------------------------------------------

_VALIDATE: contextvars.ContextVar[bool] = contextvars.ContextVar("gleam_validate", default=True)
_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar("gleam_validation_active", default=False)


def validation_enabled() -> bool:
    """Whether input validation is currently enabled."""
    return _VALIDATE.get()


@contextlib.contextmanager
def setup_validation(validate_inputs: bool) -> Iterator[None]:
    """Context manager used by every ``run_*`` function (``setup_validation()`` in R).

    Warns once when entering an unchecked run, including nested module calls.
    """
    if not isinstance(validate_inputs, (bool, np.bool_)):
        abort("`validate_inputs` must be TRUE or FALSE.")
    if not validate_inputs and (not _ACTIVE.get() or validation_enabled()):
        warn(
            "Input validation has been turned off. Potential inconsistencies or invalid "
            "input values will not be flagged. Use this option with caution and ensure "
            "that input data have been checked before running the pipeline."
        )
    t1 = _VALIDATE.set(bool(validate_inputs))
    t2 = _ACTIVE.set(True)
    try:
        yield
    finally:
        _VALIDATE.reset(t1)
        _ACTIVE.reset(t2)


@contextlib.contextmanager
def validation_disabled() -> Iterator[None]:
    """Disable validation for direct ``calc_*`` calls (no warning)."""
    t = _VALIDATE.set(False)
    try:
        yield
    finally:
        _VALIDATE.reset(t)


def validator(fn):
    """Decorator: skip the wrapped validation function when validation is off."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not validation_enabled():
            return None
        return fn(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------
# Parameter ranges (R/sysdata.rda -> data/parameter_ranges.csv)
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def parameter_ranges() -> pd.DataFrame:
    """The internal ``parameter_ranges`` rule table."""
    with resources.files("gleampy.data").joinpath("parameter_ranges.csv").open("r") as fh:
        df = pd.read_csv(fh, keep_default_na=False, na_values=["NA"])
    df["variable_name"] = df["variable_name"].str.strip()
    for col in ("lower_inclusive", "upper_inclusive"):
        df[col] = df[col].astype(str).str.strip().str.upper().eq("TRUE")
    return df


@functools.lru_cache(maxsize=None)
def _rule(arg_name: str) -> tuple[float, bool, float, bool]:
    pr = parameter_ranges()
    rows = pr[pr["variable_name"] == arg_name]
    if len(rows) != 1:
        abort(f"Internal error: expected exactly one rule for `{arg_name}`, found {len(rows)}.")
    r = rows.iloc[0]
    return float(r["lower_bound"]), bool(r["lower_inclusive"]), float(r["upper_bound"]), bool(r["upper_inclusive"])


def _fmt(v: float) -> str:
    if np.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    return f"{v:g}"


def validate_param_range(x: Any, arg_name: str, labels: Sequence[Any] | None = None) -> None:
    """Validate a numeric scalar or vector against the ``parameter_ranges`` rule.

    Port of ``validate_param_range()``: numeric, no missing values, within the
    (inclusive or strict) lower/upper bounds of ``arg_name``.
    """
    if not validation_enabled():
        return
    if isinstance(x, (str, bytes)) or (not is_scalar(x) and np.asarray(x).dtype.kind in "OUS" and not _all_numeric(x)):
        abort(f"`{arg_name}` must be numeric.")
    if isinstance(x, (bool, np.bool_)):
        abort(f"`{arg_name}` must be numeric.")
    vals = np.atleast_1d(as_float(x))
    if np.isnan(vals).any():
        abort(f"`{arg_name}` must not contain missing values.")
    lo, lo_inc, hi, hi_inc = _rule(arg_name)
    bad_lo = vals < lo if lo_inc else vals <= lo
    bad_hi = vals > hi if hi_inc else vals >= hi
    bad = np.flatnonzero(bad_lo | bad_hi)
    if bad.size:
        i = int(bad[0])
        if vals.size == 1 and labels is None:
            suffix = ""
        else:
            lab = labels[i] if labels is not None else i + 1
            suffix = f"[{lab}]"
        lo_op = ">=" if lo_inc else ">"
        hi_op = "<=" if hi_inc else "<"
        abort(
            f"`{arg_name}`{suffix} = {_fmt(vals[i])} is out of range; expected value should be "
            f"{lo_op} {_fmt(lo)} and {hi_op} {_fmt(hi)}."
        )


def _all_numeric(x: Any) -> bool:
    for v in np.ravel(np.asarray(x, dtype=object)):
        if is_na(v):
            continue
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)):
            return False
    return True


# --------------------------------------------------------------------------
# Scalar / vector type checks (vectorised versions of the R scalar checks)
# --------------------------------------------------------------------------


def _is_numeric_like(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_, str, bytes)) or x is None:
        return False
    if isinstance(x, pd.Series):
        return x.dtype.kind in "iuf"
    arr = np.asarray(x)
    if arr.dtype.kind in "iuf":
        return True
    if arr.dtype == object:
        return _all_numeric(arr) and not all(is_na(v) for v in arr.ravel())
    return False


def validate_scalar_numeric(x: Any, arg_name: str) -> None:
    """Numeric and not NA (element-wise for arrays)."""
    if not validation_enabled():
        return
    if not _is_numeric_like(x) or isna(as_float(x)).any():
        abort(f"`{arg_name}` must be a single numeric value.")


def validate_scalar_character(x: Any, arg_name: str) -> None:
    """Character and not NA (element-wise for arrays)."""
    if not validation_enabled():
        return
    if isinstance(x, str):
        return
    arr = np.atleast_1d(x.to_numpy(dtype=object) if isinstance(x, pd.Series) else np.asarray(x, dtype=object))
    if arr.size == 0 or pd.api.types.infer_dtype(arr.reshape(-1), skipna=False) != "string":
        abort(f"`{arg_name}` must be a single character value.")


def validate_fraction(x: Any, arg_name: str) -> None:
    if not validation_enabled():
        return
    validate_scalar_numeric(x, arg_name)
    v = np.atleast_1d(as_float(x))
    if ((v < 0) | (v > 1)).any():
        abort(f"`{arg_name}` must be between 0 and 1.")


def validate_positive_numeric(x: Any, arg_name: str) -> None:
    if not validation_enabled():
        return
    validate_scalar_numeric(x, arg_name)
    if (np.atleast_1d(as_float(x)) <= 0).any():
        abort(f"`{arg_name}` must be positive.")


def validate_nonnegative_numeric(x: Any, arg_name: str) -> None:
    if not validation_enabled():
        return
    validate_scalar_numeric(x, arg_name)
    if (np.atleast_1d(as_float(x)) < 0).any():
        abort(f"`{arg_name}` must be greater than or equal to 0.")


def validate_scalar_numeric_or_na(x: Any, arg_name: str, min_val: float = 0) -> None:
    """Each value is NA or a numeric ``>= min_val``."""
    if not validation_enabled():
        return
    arr = np.atleast_1d(x.to_numpy() if isinstance(x, pd.Series) else np.asarray(x))
    if arr.dtype.kind not in "iuf":
        vals = arr.astype(object).reshape(-1)
        present = vals[~pd.isna(vals)]
        if present.size and pd.api.types.infer_dtype(present, skipna=False) not in (
            "integer", "floating", "mixed-integer-float", "decimal"
        ):
            abort(f"`{arg_name}` must be a single numeric (scalar). NA is allowed.")
        arr = as_float(arr)
    with np.errstate(invalid="ignore"):
        if (as_float(arr) < min_val).any():
            abort(f"`{arg_name}` must be >= {_fmt(float(min_val))}.")


def validate_named_numeric_vector(
    x: Any, expected_length: int, expected_names: Iterable[str] | None, arg_name: str
) -> None:
    """Port of ``validate_named_numeric_vector()`` for dict / Series inputs."""
    if not validation_enabled():
        return
    if isinstance(x, dict):
        names, vals = list(x.keys()), list(x.values())
    elif isinstance(x, pd.Series):
        names, vals = list(x.index), list(x.values)
    else:
        names, vals = None, None
    if names is None or len(vals) != expected_length or not _all_numeric(vals):
        abort(f"`{arg_name}` must be a numeric vector of length {expected_length} with names.")
    if expected_names is not None and set(names) != set(expected_names):
        abort(f"`{arg_name}` must have names: {', '.join(expected_names)}")


def validate_animal_species(species_short: Any) -> None:
    if not validation_enabled():
        return
    validate_scalar_character(species_short, "species_short")
    bad = [v for v in pd.unique(np.atleast_1d(as_str(species_short)).reshape(-1)) if v not in K.GLEAM_SPECIES]
    if bad:
        abort(f"`species_short` must be one of: {', '.join(K.GLEAM_SPECIES)}")


def validate_cohort_code(cohort_short: Any) -> None:
    if not validation_enabled():
        return
    validate_scalar_character(cohort_short, "cohort_short")
    bad = [v for v in pd.unique(np.atleast_1d(as_str(cohort_short)).reshape(-1)) if v not in K.GLEAM_COHORTS]
    if bad:
        abort(f"`cohort_short` must be one of: {', '.join(K.GLEAM_COHORTS)}")


def validate_is_egg_producing_flag(
    species_short: Any,
    cohort_short: Any,
    is_egg_producing: Any = False,
    nondemo_productive_phase_id: Any = np.nan,
) -> None:
    """Vectorised ``validate_is_egg_producing_flag()``.

    The flag must be logical (``TRUE`` / ``FALSE`` / NA); it may be TRUE only
    for CHK ``FA`` rows and for CHK ``FN`` rows in non-demographic phase 2,
    and must not be NA for CHK. As in R, ``nondemo_productive_phase_id`` is
    only examined on CHK ``FN`` rows whose flag is TRUE, and is compared the
    way R's ``!=`` does (a character ``"2"`` equals 2, ``"2.0"`` or ``"abc"``
    does not), so a malformed phase id elsewhere is ignored like in R.

    Parameters
    ----------
    species_short : str or array-like of str
        Species code(s).
    cohort_short : str or array-like of str
        Cohort code(s).
    is_egg_producing : bool or array-like, default False
        Egg-producing flag(s) (logical; NA allowed except for CHK).
    nondemo_productive_phase_id : scalar or array-like, default NA
        Non-demographic phase (1 or 2) of each row.
    """
    if not validation_enabled():
        return
    raw = np.atleast_1d(
        is_egg_producing.to_numpy(dtype=object) if isinstance(is_egg_producing, pd.Series)
        else np.asarray(is_egg_producing, dtype=object)
    ).reshape(-1)
    present = raw[~pd.isna(raw)]
    if present.size and pd.api.types.infer_dtype(present, skipna=False) != "boolean":
        abort("`is_egg_producing` must be logical (TRUE/FALSE).")
    phase = (
        nondemo_productive_phase_id.to_numpy(dtype=object)
        if isinstance(nondemo_productive_phase_id, (pd.Series, pd.Index))
        else np.asarray(nondemo_productive_phase_id, dtype=object)
    )
    sp, co, egg, ph = np.broadcast_arrays(
        as_str(species_short), as_str(cohort_short), as_bool(is_egg_producing), phase,
    )
    is_chk = sp == "CHK"
    egg_true = np.asarray(egg == True, dtype=bool)  # noqa: E712
    egg_na = np.asarray(pd.isna(egg), dtype=bool)
    if (~is_chk & egg_true).any():
        abort("`is_egg_producing` can be TRUE only for \"CHK\".")
    if (is_chk & egg_na).any():
        abort("`is_egg_producing` must be a single non-missing logical value for \"CHK\".")
    laying = is_chk & egg_true
    if (laying & ~((co == "FA") | (co == "FN"))).any():
        abort("`is_egg_producing` can be TRUE only for CHK cohorts \"FA\" or \"FN\".")
    laying_fn = np.flatnonzero(np.atleast_1d(laying & (co == "FN")))
    if laying_fn.size and any(_phase_is_not_2(v) for v in np.atleast_1d(ph)[laying_fn]):
        abort(
            "`is_egg_producing` can be TRUE for \"FN\" only when "
            "`nondemo_productive_phase_id` is 2."
        )


def _phase_is_not_2(v: Any) -> bool:
    """R's ``is.na(v) || v != 2`` for one phase id of any type."""
    if is_na(v):
        return True
    if isinstance(v, (bool, np.bool_)):
        return True  # TRUE / FALSE coerce to 1 / 0
    if isinstance(v, (str, bytes)):
        return v != "2"  # R compares as character: as.character(2) is "2"
    try:
        return float(v) != 2.0
    except (TypeError, ValueError):
        return True


def normalize_optional_is_egg_producing_column(
    cohort_level_data: pd.DataFrame, herd_level_data: pd.DataFrame
) -> None:
    """Add ``is_egg_producing = NA`` when absent and no CHK herds are present (in place)."""
    if "is_egg_producing" not in cohort_level_data.columns and not (
        "species_short" in herd_level_data.columns
        and (herd_level_data["species_short"] == "CHK").any()
    ):
        cohort_level_data["is_egg_producing"] = pd.Series([None] * len(cohort_level_data), dtype=object)


def validate_species_short_values(x: Any, column_name: str = "species_short", data_arg: str = "data") -> None:
    if not validation_enabled():
        return
    invalid = [v for v in pd.unique(pd.Series(np.asarray(x, dtype=object))) if v not in K.GLEAM_SPECIES]
    if invalid:
        abort(
            f"Invalid `{column_name}` values in `{data_arg}`: {_vals(invalid)}. "
            f"Must be one of: {_vals(K.GLEAM_SPECIES)}"
        )


def validate_cohort_short_values(x: Any, column_name: str = "cohort_short", data_arg: str = "data") -> None:
    if not validation_enabled():
        return
    invalid = [v for v in pd.unique(pd.Series(np.asarray(x, dtype=object))) if v not in K.GLEAM_COHORTS]
    if invalid:
        abort(
            f"Invalid `{column_name}` values in `{data_arg}`: {_vals(invalid)}. "
            f"Must be one of: {_vals(K.GLEAM_COHORTS)}"
        )


def _vals(xs: Iterable[Any]) -> str:
    return ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in xs)


# --------------------------------------------------------------------------
# Run-level (table) checks
# --------------------------------------------------------------------------


def check_data_table(x: Any, arg_name: str) -> None:
    """Non-empty pandas DataFrame (``data.table`` in R)."""
    if not validation_enabled():
        return
    if not isinstance(x, pd.DataFrame):
        abort(f"`{arg_name}` must be a data.table.")
    if len(x) == 0:
        abort(f"`{arg_name}` must contain at least one row.")


def check_required_columns(data: pd.DataFrame, required_cols: Iterable[str], arg_name: str) -> None:
    if not validation_enabled():
        return
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        abort(f"Missing required columns in `{arg_name}`: {_vals(missing)}")


def check_cohort_completeness(cohort_level_data: pd.DataFrame, data_arg: str = "cohort_level_data") -> None:
    """Each herd must include the 6 demographic cohorts exactly once.

    Port of ``check_cohort_completeness()``: the demographic rows (FJ, FS,
    FA, MJ, MS, MA) of every herd must number exactly 6 and cover all six
    cohorts; FN / MN rows are ignored. Herds are reported in order of first
    appearance; an NA ``herd_id`` is a group of its own, as in data.table.
    Runs in a single grouped pass (linear in the number of rows).

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort table with ``herd_id`` and ``cohort_short``.
    data_arg : str, default "cohort_level_data"
        Argument name used in error messages.
    """
    if not validation_enabled():
        return
    demo_cohorts = K.GLEAM_COHORTS_DEMOGRAPHIC
    cohort = cohort_level_data["cohort_short"]
    is_demo = np.asarray(cohort.isin(demo_cohorts), dtype=bool)
    herd = cohort_level_data["herd_id"][is_demo]
    if isinstance(herd.dtype, pd.CategoricalDtype):
        herd = herd.astype(object)
    # herds in order of first appearance, NA as a group of its own (data.table `by`)
    herd_code, herds = pd.factorize(herd, use_na_sentinel=False)
    cohort_code = pd.Index(demo_cohorts).get_indexer(cohort[is_demo].astype(object))
    counts = np.bincount(herd_code, minlength=len(herds))
    wrong = [herds[i] for i in np.flatnonzero(counts != 6)]
    if wrong:
        abort(
            f"Each herd_id must have exactly 6 rows in `{data_arg}` (one per cohort). "
            f"Found incorrect counts for herd_ids: {_vals(wrong)}"
        )
    present = np.zeros((len(herds), len(demo_cohorts)), dtype=bool)
    present[herd_code, cohort_code] = True
    bad = np.flatnonzero(~present.all(axis=1))
    if bad.size:
        incomplete = []
        for i in bad:
            missing = [c for j, c in enumerate(demo_cohorts) if not present[i, j]]
            incomplete.append(f"{herds[i]} (missing: {', '.join(missing)})")
        abort(
            f"Each herd_id must have exactly one row for each of the 6 cohorts in `{data_arg}`. "
            f"Incomplete or duplicate cohorts found for herd_ids: {_vals(incomplete)}"
        )


def check_nondemographic_phase_completeness(
    cohort_level_data: pd.DataFrame, data_arg: str = "cohort_level_data"
) -> None:
    """Exactly one phase-1 row and at most one phase-2 row per herd/cohort block.

    Port of ``check_nondemographic_phase_completeness()``. Blocks are
    ``(herd_id, cohort_short)`` groups in order of first appearance; NA keys
    form groups of their own, as with data.table's ``by``.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Non-demographic cohort rows with ``herd_id``, ``cohort_short`` and
        ``nondemo_productive_phase_id``.
    data_arg : str, default "cohort_level_data"
        Argument name used in error messages.
    """
    if not validation_enabled():
        return
    ph = as_float(cohort_level_data["nondemo_productive_phase_id"])
    invalid = ~np.isin(ph, [1.0, 2.0])
    if invalid.any():
        rows = cohort_level_data[invalid]
        keys = [f"{h}/{c}" for h, c in zip(rows["herd_id"], rows["cohort_short"])]
        abort(
            f"`{data_arg}` must use only 1 or 2 in `nondemo_productive_phase_id`. "
            f"Invalid rows found for herd/cohort combinations: {_vals(keys)}"
        )
    tmp = pd.DataFrame(
        {"herd_id": cohort_level_data["herd_id"].to_numpy(), "cohort_short": cohort_level_data["cohort_short"].to_numpy(),
         "p1": (ph == 1).astype(int), "p2": (ph == 2).astype(int)}
    )
    summ = tmp.groupby(["herd_id", "cohort_short"], sort=False, dropna=False).sum().reset_index()
    bad = summ[(summ["p1"] != 1) | (summ["p2"] > 1)]
    if len(bad):
        keys = [f"{h}/{c} (phase1={a}, phase2={b})" for h, c, a, b in bad[["herd_id", "cohort_short", "p1", "p2"]].itertuples(index=False)]
        abort(
            f"Each herd/cohort block in `{data_arg}` must have exactly one phase 1 row "
            f"and at most one phase 2 row. Invalid blocks: {_vals(keys)}"
        )


def check_herd_id_unique(herd_level_data: pd.DataFrame, arg_name: str = "herd_level_data") -> None:
    if not validation_enabled():
        return
    counts = herd_level_data["herd_id"].value_counts(sort=False, dropna=False)
    dup = list(counts[counts > 1].index)
    if dup:
        abort(f"Each herd_id must appear exactly once in `{arg_name}`. Found duplicates for herd_ids: {_vals(dup)}")


def check_herd_id_consistency(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
    cohort_arg: str = "cohort_level_data",
    herd_arg: str = "herd_level_data",
) -> None:
    """Both tables must contain the same set of herd ids.

    Port of ``check_herd_id_consistency()``, which uses ``setdiff``: an NA
    herd id matches an NA herd id (whatever its representation: ``nan``,
    ``None``, ``pd.NA``) and integer ids match equal float ids.

    Parameters
    ----------
    cohort_level_data, herd_level_data : pandas.DataFrame
        Tables with a ``herd_id`` column.
    cohort_arg, herd_arg : str
        Argument names used in error messages.
    """
    if not validation_enabled():
        return
    c_ids = _unique_ids(cohort_level_data["herd_id"])
    h_ids = _unique_ids(herd_level_data["herd_id"])
    c_set, h_set = set(c_ids), set(h_ids)
    miss_h = [orig for key, orig in c_ids.items() if key not in h_set]
    if miss_h:
        abort(f"Herd IDs in `{cohort_arg}` not found in `{herd_arg}`: {_vals(miss_h)}")
    miss_c = [orig for key, orig in h_ids.items() if key not in c_set]
    if miss_c:
        abort(f"Herd IDs in `{herd_arg}` not found in `{cohort_arg}`: {_vals(miss_c)}")


class _NA:
    """Hashable stand-in for every missing value (R's ``match`` matches NA to NA)."""

    def __repr__(self) -> str:
        return "NA"


_NA_KEY = _NA()


def _id_key(v: Any) -> Any:
    if is_na(v):
        return _NA_KEY
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)):
        return float(v)
    return v


def _unique_ids(s: pd.Series) -> dict[Any, Any]:
    """Unique ids in order of first appearance: normalised key -> first original value."""
    out: dict[Any, Any] = {}
    for v in pd.unique(np.asarray(s.to_numpy(dtype=object), dtype=object)):
        out.setdefault(_id_key(v), v)
    return out
