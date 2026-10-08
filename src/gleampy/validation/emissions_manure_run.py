"""Run-level input validation for the manure management emissions module.

Port of ``R/validate_run_emissions_manure_inputs.R``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._utils import as_float, group_sum, is_na, isna
from ._shared import (
    abort,
    check_data_table,
    check_required_columns,
    validate_cohort_short_values,
    validator,
)

#: Columns required in ``cohort_level_data``.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "herd_id", "cohort_short", "ration_intake",
    "ration_digestibility_fraction", "nitrogen_excretion",
    "ration_urinary_energy_fraction", "ration_ash",
)
#: Columns required in ``manure_management_system_fraction``.
REQUIRED_FRACTION_COLUMNS: tuple[str, ...] = (
    "herd_id", "cohort_short", "manure_management_system", "manure_management_system_fraction",
)
#: Columns required in ``manure_management_system_factors``.
REQUIRED_FACTORS_COLUMNS: tuple[str, ...] = (
    "herd_id", "manure_management_system",
    "methane_conversion_factor_mcf", "ch4_max_producing_capacity_bo",
    "n2o_ef3", "n2o_ef4", "n2o_ef5", "nitrogen_fracgas", "nitrogen_fracleach",
)

_PHASE = "nondemo_productive_phase_id"
_MMS = "manure_management_system"


@validator
def validate_run_emissions_manure_module_inputs(
    cohort_level_data: Any,
    manure_management_system_fraction: Any,
    manure_management_system_factors: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.run_emissions_manure_module`.

    Checks table types and required columns; missing keys; valid cohort codes;
    unique rows per herd / cohort (/ ``nondemo_productive_phase_id`` when that
    column is in both cohort-level tables) and per MMS; MMS fractions summing
    to 1 per group; every MMS of the fractions having factors for its herd;
    consistent MMS sets across the cohorts of a herd; and consistent herd ids
    and herd / cohort (/ phase) combinations across the three tables.

    Groups are formed like data.table's ``by``: only the key combinations
    that occur (also for categorical key columns), in order of first
    appearance, with NA a group of its own. Key values are compared the way
    R compares them, so a float, integer or object-dtype column holding the
    same numbers matches.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table with ``herd_id``, ``cohort_short``,
        ``ration_intake`` (kg DM/head/day), ``ration_digestibility_fraction``
        (fraction), ``nitrogen_excretion`` (kg N/head/day),
        ``ration_urinary_energy_fraction`` (fraction), ``ration_ash``
        (kg ash/kg DM) and optionally ``nondemo_productive_phase_id``.
    manure_management_system_fraction : pandas.DataFrame
        MMS fractions per herd and cohort: ``herd_id``, ``cohort_short``,
        ``manure_management_system``, ``manure_management_system_fraction``
        (fraction; a numeric column summing to 1 per group within an
        absolute tolerance of 1e-8) and optionally
        ``nondemo_productive_phase_id``.
    manure_management_system_factors : pandas.DataFrame
        MMS emission factors per herd and system: ``herd_id``,
        ``manure_management_system``, ``methane_conversion_factor_mcf`` (%),
        ``ch4_max_producing_capacity_bo`` (m3 CH4/kg VS), ``n2o_ef3``
        (kg N2O-N/kg N), ``n2o_ef4`` (kg N2O-N/kg volatilised N), ``n2o_ef5``
        (kg N2O-N/kg N leached), ``nitrogen_fracgas`` (fraction) and
        ``nitrogen_fracleach`` (fraction).

    Raises
    ------
    GleamValidationError
        With R's message, for the first check that fails.
    """
    cohort = cohort_level_data
    fraction = manure_management_system_fraction
    factors = manure_management_system_factors

    # --- Basic type and structure checks
    check_data_table(cohort, "cohort_level_data")
    check_data_table(fraction, "manure_management_system_fraction")
    check_data_table(factors, "manure_management_system_factors")

    # --- Required columns
    check_required_columns(cohort, REQUIRED_INPUT_COLUMNS, "cohort_level_data")
    check_required_columns(fraction, REQUIRED_FRACTION_COLUMNS, "manure_management_system_fraction")
    check_required_columns(factors, REQUIRED_FACTORS_COLUMNS, "manure_management_system_factors")

    # --- Missing key values
    if _any_na(cohort, "herd_id", "cohort_short"):
        abort("`cohort_level_data` must not contain missing herd_id or cohort_short.")
    if _any_na(cohort, "ration_urinary_energy_fraction", "ration_ash"):
        abort(
            "`cohort_level_data` must not contain missing ration_urinary_energy_fraction or ration_ash."
        )
    if _any_na(fraction, "herd_id", "cohort_short", _MMS):
        abort(
            "`manure_management_system_fraction` must not contain missing herd_id, cohort_short, "
            "or manure_management_system."
        )
    if _any_na(factors, "herd_id", _MMS):
        abort(
            "`manure_management_system_factors` must not contain missing herd_id or "
            "manure_management_system."
        )

    # --- Cohort codes (coverage may vary by herd)
    validate_cohort_short_values(cohort["cohort_short"], data_arg="cohort_level_data")
    validate_cohort_short_values(fraction["cohort_short"], data_arg="manure_management_system_fraction")

    group_cols = ["herd_id", "cohort_short"]
    if _PHASE in cohort.columns and _PHASE in fraction.columns:
        group_cols.append(_PHASE)

    # Key columns are coded once and shared by the checks below.
    t_cohort, t_fraction, t_factors = _Table(cohort), _Table(fraction), _Table(factors)

    # --- Uniqueness checks (`DT[, .N, by = cols][N > 1]`)
    if _has_duplicates(t_cohort, group_cols):
        abort(
            "Duplicate herd/cohort rows in `cohort_level_data` for grouping columns "
            f"{_cli_vals(group_cols)}."
        )
    if _has_duplicates(t_fraction, group_cols + [_MMS]):
        abort("Duplicate herd/cohort/manure-management rows in `manure_management_system_fraction`.")

    # --- MMS fraction sum-to-one per herd/cohort[/phase] group
    # R: DT[, .(total_fraction = sum(manure_management_system_fraction)), by = cols].
    # data.table's grouped sum (gsum) rejects a character column and adds the
    # values of each group one by one in row order (no compensated summation).
    frac_col = fraction["manure_management_system_fraction"]
    if _column_kind(frac_col) == "character":
        abort(
            "`manure_management_system_fraction` in `manure_management_system_fraction` must be "
            "numeric (R: Type 'character' is not supported by GForce sum (gsum))."
        )
    codes, n_groups = t_fraction.group(group_cols)
    totals = group_sum(as_float(frac_col), codes, n_groups)
    with np.errstate(invalid="ignore"):
        invalid = np.flatnonzero(np.isnan(totals) | (np.abs(totals - 1) > 1e-8))
    if invalid.size:
        # R: `invalid_sums[, do.call(paste, ...), .SDcols = ...][[1]]` keeps only the
        # first invalid group (j returns a character vector, `[[1]]` its first element).
        row = int(_first_rows(codes, n_groups)[invalid[0]])
        keys = [_paste_key([fraction[c].iloc[row] for c in group_cols])]
        abort(
            "For each herd/cohort group, the sum of MMS fractions in "
            "`manure_management_system_fraction` must equal 1. "
            f"Invalid groups: {_cli_vals(keys)}"
        )

    if _has_duplicates(t_factors, ["herd_id", _MMS]):
        abort(
            "Duplicate herd_id + manure_management_system rows in `manure_management_system_factors`."
        )

    # --- MMS consistency checks: every MMS of the fractions needs factors
    #     (factors may contain additional MMS).
    missing_herds = _setdiff(_unique(fraction["herd_id"]), _unique(factors["herd_id"]))
    if missing_herds:
        abort(
            "Herd IDs in `manure_management_system_fraction` not found in "
            f"`manure_management_system_factors`: {_cli_vals(missing_herds)}"
        )

    # The two MMS checks below first detect a problem with vectorised
    # operations; the per-herd MMS lists that R's message is built from are
    # only computed when one is found.
    if _mms_without_factors(t_fraction, t_factors):
        frac_by_herd = _mms_sets_by(t_fraction, ["herd_id"])
        fact_by_herd = _mms_sets_by(t_factors, ["herd_id"])
        # merge(..., by = "herd_id") of the two per-herd lists: ordered by herd_id.
        fact_lookup = {_norm(k[0]): v for k, v in fact_by_herd.items()}
        merged = sorted(
            (
                (k[0], v, fact_lookup[_norm(k[0])])
                for k, v in frac_by_herd.items()
                if _norm(k[0]) in fact_lookup
            ),
            key=lambda t: _sort_key(t[0]),
        )
        details = [
            f"{_r_str(h)}: {', '.join(m for m in frac_list if m not in set(fact_list))}"
            for h, frac_list, fact_list in merged
            if any(m not in set(fact_list) for m in frac_list)
        ]
        if details:
            abort(
                "Some `manure_management_system` values in `manure_management_system_fraction` "
                "have no matching entry in `manure_management_system_factors`. "
                f"Affected herd_ids and missing systems: {_cli_vals(details)}"
            )

    # MMS list should be the same for all cohorts within a herd in the fraction table.
    if _mms_sets_differ_within_herds(t_fraction):
        sets_by_cohort = _mms_sets_by(t_fraction, ["herd_id", "cohort_short"])
        herd_sets: dict[Any, tuple[Any, set[str]]] = {}
        for (herd, _cohort), mms_list in sets_by_cohort.items():
            entry = herd_sets.setdefault(_norm(herd), (herd, set()))
            entry[1].add("|".join(mms_list))
        inconsistent = [herd for herd, sets in herd_sets.values() if len(sets) > 1]
        if inconsistent:
            abort(
                "Within each herd_id, manure_management_system lists must be consistent across "
                "cohorts in `manure_management_system_fraction`. Inconsistent herds: "
                f"{_cli_vals(inconsistent)}"
            )

    # --- Cross-table herd_id validation
    input_herds = _unique(cohort["herd_id"])
    fraction_herds = _unique(fraction["herd_id"])
    factors_herds = _unique(factors["herd_id"])

    # Each herd/cohort[/phase] of the input must exist in the fraction table
    # (fsetdiff(unique(input_pairs), unique(fraction_pairs)); NA matches NA).
    missing_rows = _missing_key_rows(t_cohort, t_fraction, group_cols)
    if len(missing_rows):
        # As above, R's `[[1]]` reports only the first missing combination.
        first = cohort.iloc[int(missing_rows[0])]
        keys = [_paste_key([first[c] for c in group_cols])]
        abort(
            "Missing herd/cohort combinations in `manure_management_system_fraction`: "
            f"{_cli_vals(keys)}"
        )

    for values, other, msg in (
        (input_herds, fraction_herds,
         "Herd IDs in `cohort_level_data` not found in `manure_management_system_fraction`"),
        (fraction_herds, input_herds,
         "Herd IDs in `manure_management_system_fraction` not found in `cohort_level_data`"),
        (input_herds, factors_herds,
         "Herd IDs in `cohort_level_data` not found in `manure_management_system_factors`"),
        (factors_herds, input_herds,
         "Herd IDs in `manure_management_system_factors` not found in `cohort_level_data`"),
    ):
        missing = _setdiff(values, other)
        if missing:
            abort(f"{msg}: {_cli_vals(missing)}")


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


class _NA:
    """Hashable stand-in for R ``NA`` in key tuples (NA matches NA in data.table)."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NA"


_NA_KEY = _NA()

_NUMBER_KINDS = frozenset({"integer", "floating", "mixed-integer-float", "decimal"})


def _norm(v: Any) -> Any:
    """Normalise a key value: NA -> sentinel, numbers -> float (so 1 == 1.0)."""
    if is_na(v):
        return _NA_KEY
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return v


def _any_na(df: pd.DataFrame, *cols: str) -> bool:
    return any(isna(df[c]).any() for c in cols)


def _unique(s: pd.Series) -> list[Any]:
    """``unique(x)`` in first-appearance order (NA kept once)."""
    out, seen = [], set()
    for v in pd.unique(s):
        k = _norm(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _setdiff(x: Sequence[Any], y: Sequence[Any]) -> list[Any]:
    """R ``setdiff(x, y)``: unique values of ``x`` not in ``y``, in ``x`` order."""
    ys = {_norm(v) for v in y}
    out, seen = [], set()
    for v in x:
        k = _norm(v)
        if k not in ys and k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _value_kind(values: np.ndarray) -> str:
    """R type of an object array's non-missing values: "numeric", "logical" or "character".

    Numbers (and numbers mixed with logicals, as ``rbind`` of a double and a
    logical column gives a double) are numeric; only logicals are logical;
    anything else (strings, mixed values) is character. No values: logical.
    """
    if values.size == 0:
        return "logical"
    kind = pd.api.types.infer_dtype(values, skipna=False)
    if kind in _NUMBER_KINDS:
        return "numeric"
    if kind == "boolean":
        return "logical"
    if kind == "mixed" and all(
        isinstance(v, (bool, np.bool_, int, float, np.number)) for v in values
    ):
        return "numeric"
    return "character"


def _column_kind(s: pd.Series) -> str:
    """R type of a column as data.table would hold it: "numeric", "logical" or "character".

    Integer, float and nullable numeric dtypes (and categoricals with numeric
    categories) are numeric; bool / ``boolean`` columns are logical; an
    object column is typed from its values (all missing is logical, as
    ``fread`` types an empty column); string and other dtypes are character.
    """
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return "numeric" if dt.categories.dtype.kind in "iuf" else "character"
    if dt.kind == "b":
        return "logical"
    if dt.kind in "iuf":
        return "numeric"
    if dt == object:
        vals = s.to_numpy(dtype=object)
        return _value_kind(vals[~pd.isna(vals)])
    return "character"


def _key(s: pd.Series) -> tuple[np.ndarray, np.ndarray, bool]:
    """(values, is-NA mask, numeric?) of a key column.

    Numeric columns (any numeric dtype, a categorical with numeric
    categories, or an object column whose values are all numbers, e.g. after
    ``pd.concat`` with an all-``None`` column) give float64 values, as R holds
    them in a numeric column. All-missing columns count as numeric. Other
    columns give an object array; numbers mixed into it are written as R's
    ``as.character()`` writes them.
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    na = s.isna().to_numpy()
    if s.dtype.kind in "iufb":
        return s.to_numpy(dtype="float64", na_value=np.nan), na, True
    vals = s.to_numpy(dtype=object)
    present = vals[~na]
    if present.size == 0:
        return np.full(len(s), np.nan), na, True
    kind = pd.api.types.infer_dtype(present, skipna=False)
    if kind == "string":
        return vals, na, False
    if _value_kind(present) != "character":
        out = np.full(len(s), np.nan)
        out[~na] = present.astype("float64")
        return out, na, True
    out = vals.copy()
    out[~na] = np.array([_r_str(v) for v in present] + [None], dtype=object)[:-1]
    return out, na, False


def _aligned_keys(*cols: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """(values, is-NA mask) of key columns of several tables, comparable as R's ``==`` compares them.

    Numbers compare as numbers (``1L == 1``); when numeric and character
    columns are mixed, numbers are compared as character strings
    (``as.character()``, 15 significant digits).
    """
    keys = [_key(c) for c in cols]
    if all(k[2] for k in keys) or not any(k[2] for k in keys):
        return [(v, na) for v, na, _ in keys]
    out = []
    for v, na, _numeric in keys:
        out.append((np.array([None if m else _r_str(x) for x, m in zip(v, na)], dtype=object), na))
    return out


def _factorize(values: Any) -> tuple[np.ndarray, int]:
    """Codes of ``values`` in order of first appearance, NA one value of its own, and their number.

    ``pd.factorize`` numbers the non-missing values in order of first
    appearance, but where it puts NA with ``use_na_sentinel=False`` is left
    to the array backend (numpy object / float arrays, the Python and pyarrow
    string arrays) and not documented. :func:`_first_rows` and the checks
    built on it need NA numbered where it first appears too, so NA is coded
    here from the sentinel: it takes the code of its first appearance and the
    values first seen after it move up by one.
    """
    codes, uniques = pd.factorize(values)
    codes = np.asarray(codes, dtype=np.int64)
    n = len(uniques)
    na = codes < 0
    if na.any():
        first_na = int(na.argmax())
        na_code = int(codes[:first_na].max()) + 1 if first_na else 0
        codes = np.where(na, na_code, codes + (codes >= na_code))
        n += 1
    return codes, n


def _codes(s: pd.Series) -> tuple[np.ndarray, int]:
    """Codes of a key column's values in order of first appearance (NA one value) and their number.

    Values are those of :func:`_key`, so numbers compare as numbers
    (``1 == 1.0``) whatever the column's dtype.
    """
    if isinstance(s.dtype, pd.StringDtype):  # fast path: nothing to normalise
        return _factorize(s)
    vals, na, _ = _key(s)
    if vals.dtype.kind == "f":
        vals = np.where(na, np.nan, vals)
    elif na.any():
        vals = vals.copy()
        vals[na] = None
    return _factorize(vals)


def _combine(codes: np.ndarray, n: int, more: np.ndarray, n_more: int) -> tuple[np.ndarray, int]:
    """Codes of the pairs (``codes``, ``more``), again numbered in order of first appearance."""
    combined, uniques = pd.factorize(codes * n_more + more)
    return combined.astype(np.int64, copy=False), len(uniques)


class _Table:
    """A table whose key columns are coded once (see :func:`_codes`) and reused by the checks."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self._columns: dict[str, tuple[np.ndarray, int]] = {}
        self._groups: dict[tuple[str, ...], tuple[np.ndarray, int]] = {}

    def codes(self, col: str) -> tuple[np.ndarray, int]:
        """Value codes of one column and their number."""
        if col not in self._columns:
            self._columns[col] = _codes(self.df[col])
        return self._columns[col]

    def group(self, cols: Sequence[str]) -> tuple[np.ndarray, int]:
        """data.table ``by = cols`` groups: group number of each row and number of groups."""
        key = tuple(cols)
        if key not in self._groups:
            codes, n = np.zeros(len(self.df), dtype=np.int64), 1
            for col in cols:
                codes, n = _combine(codes, n, *self.codes(col))
            self._groups[key] = (codes, n if len(self.df) else 0)
        return self._groups[key]


def _as_table(df: pd.DataFrame | _Table) -> _Table:
    return df if isinstance(df, _Table) else _Table(df)


def _group_codes(df: pd.DataFrame | _Table, cols: Sequence[str]) -> tuple[np.ndarray, int]:
    """data.table ``by = cols`` groups: group number of each row and number of groups.

    Only combinations that occur are groups (also for categorical columns),
    numbered in order of first appearance; NA is a value of its own and
    numbers compare as numbers (``1 == 1.0``).
    """
    return _as_table(df).group(cols)


def _first_rows(codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Position of the first row of each group.

    ``codes`` must number the groups in order of first appearance (as
    :func:`_group_codes` and :func:`_codes` do), so a group starts where its
    code exceeds every earlier code.
    """
    if len(codes) == 0:
        return np.zeros(0, dtype=np.int64)
    is_first = np.empty(len(codes), dtype=bool)
    is_first[0] = True
    np.greater(codes[1:], np.maximum.accumulate(codes)[:-1], out=is_first[1:])
    return np.flatnonzero(is_first)


def _has_duplicates(df: pd.DataFrame | _Table, cols: Sequence[str]) -> bool:
    """Whether any ``by = cols`` group has more than one row."""
    codes, n = _group_codes(df, cols)
    return bool(n) and int(np.bincount(codes, minlength=n).max()) > 1


def _distinct_values(t: _Table, col: str) -> pd.Series:
    """The first row of each distinct value of ``col``, in order of first appearance."""
    codes, n = t.codes(col)
    return t.df[col].iloc[_first_rows(codes, n)].reset_index(drop=True)


def _shared_codes(x: _Table, y: _Table, col: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Codes of ``col`` in two tables on one scale: equal codes for values R's ``==`` finds equal.

    Numbers compare as numbers; when one column is numeric and the other
    character, as character strings (see :func:`_aligned_keys`); NA equals NA.
    """
    (vx, nx), (vy, ny) = _aligned_keys(_distinct_values(x, col), _distinct_values(y, col))
    both = np.concatenate([vx, vy])
    na = np.concatenate([nx, ny])
    if both.dtype.kind == "f":
        both = np.where(na, np.nan, both)
    else:
        both = both.astype(object)
        both[na] = None
    shared, n_shared = _factorize(both)
    return shared[: len(vx)][x.codes(col)[0]], shared[len(vx):][y.codes(col)[0]], n_shared


def _mms_strings(t: _Table) -> tuple[np.ndarray, np.ndarray]:
    """``as.character()`` of the MMS column (an object array) and its value codes.

    Codes identify the distinct values (numbers compare as numbers, like
    :func:`_group_codes`); missing values get the string ``"NA"``.
    """
    codes, n = t.codes(_MMS)
    first = _first_rows(codes, n)
    vals = t.df[_MMS].to_numpy(dtype=object)
    strings = np.array([_r_str(vals[i]) for i in first] + [None], dtype=object)[:-1]
    return strings, codes


def _mms_without_factors(fraction: pd.DataFrame | _Table, factors: pd.DataFrame | _Table) -> bool:
    """Whether some (herd, MMS) of the fraction table has no row in the factors table.

    Vectorised detection for R's per-herd ``setdiff(mms_list_fraction,
    mms_list_factors)`` check. It never misses a herd that R reports (it can
    only flag more, when two distinct numeric MMS ids print alike).
    """
    tf, ta = _as_table(fraction), _as_table(factors)
    h_f, h_a, _ = _shared_codes(tf, ta, "herd_id")
    m_f, m_a, n_m = _shared_codes(tf, ta, _MMS)
    return bool((~np.isin(h_f * n_m + m_f, h_a * n_m + m_a)).any())


def _mms_sets_differ_within_herds(fraction: pd.DataFrame | _Table) -> bool:
    """Whether the cohorts of some herd have different sets of MMS in the fraction table.

    Every cohort's set is part of its herd's union, so all cohort sets of a
    herd are equal exactly when each holds as many distinct systems as the
    union. Distinct raw values are counted, so this never misses a herd that
    R (which compares the pasted, sorted system names) reports.
    """
    t = _as_table(fraction)
    m_codes, n_m = t.codes(_MMS)
    hc_codes, n_hc = t.group(["herd_id", "cohort_short"])
    h_codes, n_h = t.group(["herd_id"])
    per_cohort = np.bincount(np.unique(hc_codes * n_m + m_codes) // n_m, minlength=n_hc)
    per_herd = np.bincount(np.unique(h_codes * n_m + m_codes) // n_m, minlength=n_h)
    herd_of_cohort = h_codes[_first_rows(hc_codes, n_hc)]
    return bool((per_cohort != per_herd[herd_of_cohort]).any())


def _mms_sets_by(df: pd.DataFrame | _Table, by: list[str]) -> dict[tuple[Any, ...], tuple[str, ...]]:
    """``df[, .(mms_list = list(sort(unique(manure_management_system)))), by = by]``.

    Groups in first-appearance order (only those that occur), keyed by their
    original key values; each list holds the distinct systems as R's
    ``as.character()`` writes them, sorted.
    """
    t = _as_table(df)
    codes, n_groups = t.group(by)
    strings, m_codes = _mms_strings(t)
    n_m = max(len(strings), 1)
    present = ~isna(t.df[_MMS])
    pairs = np.unique(codes[present] * n_m + m_codes[present])
    pair_group = pairs // n_m
    pair_strings = strings[pairs % n_m]
    bounds = np.searchsorted(pair_group, np.arange(n_groups + 1))
    first = _first_rows(codes, n_groups)
    key_values = [t.df[c].to_numpy(dtype=object)[first] for c in by]
    out: dict[tuple[Any, ...], tuple[str, ...]] = {}
    for g in range(n_groups):
        key = tuple(kv[g] for kv in key_values)
        out[key] = tuple(sorted(pair_strings[bounds[g]:bounds[g + 1]].tolist()))
    return out


def _missing_key_rows(
    x: pd.DataFrame | _Table, y: pd.DataFrame | _Table, cols: Sequence[str]
) -> np.ndarray:
    """Positions in ``x`` of the first rows of the key combinations absent from ``y``.

    ``data.table::fsetdiff(unique(x[, cols]), unique(y[, cols]))`` with NA
    matching NA, in ``x`` order.
    """
    tx, ty = _as_table(x), _as_table(y)
    kx = np.zeros(len(tx.df), dtype=np.int64)
    ky = np.zeros(len(ty.df), dtype=np.int64)
    n = 1
    for col in cols:
        cx, cy, n_col = _shared_codes(tx, ty, col)
        both, n = _combine(np.concatenate([kx, ky]), n, np.concatenate([cx, cy]), n_col)
        kx, ky = both[: len(kx)], both[len(kx):]
    first = _first_rows(*tx.group(cols))
    return first[~np.isin(kx[first], ky)]


def _as_character(v: Any) -> str:
    """R ``as.character()`` of a number (15 significant digits)."""
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.15g}"
    return str(v)


def _sort_key(v: Any) -> tuple[int, Any]:
    """data.table ordering of a key column: NA first, then numbers or strings ascending."""
    k = _norm(v)
    if k is _NA_KEY:
        return (0, 0)
    if isinstance(k, float):
        return (1, k)
    return (2, str(k))


def _r_str(v: Any) -> str:
    """``as.character()`` of one value (15 significant digits for doubles)."""
    if is_na(v):
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return f"{float(v):.15g}"
    return str(v)


def _paste_key(values: Sequence[Any]) -> str:
    """``paste(..., sep = " / ")`` of one group key."""
    return " / ".join(_r_str(v) for v in values)


def _cli_vals(values: Sequence[Any]) -> str:
    """cli ``{.val {x}}``: strings quoted, numbers plain, ``a, b, and c`` collapse.

    Like cli, more than 20 values are truncated to the first 18, ``…`` and the last 2.
    """
    items = [f'"{v}"' if isinstance(v, str) else _r_str(v) for v in values]
    if len(items) > 20:
        items = items[:18] + ["…"] + items[-2:]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]
