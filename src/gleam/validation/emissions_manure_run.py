"""Run-level input validation for the manure management emissions module.

Port of ``R/validate_run_emissions_manure_inputs.R``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._utils import as_float, is_na, isna
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
    """Validate the inputs of :func:`gleam.run_emissions_manure_module`.

    Checks table types and required columns; missing keys; valid cohort codes;
    unique rows per herd / cohort (/ ``nondemo_productive_phase_id`` when that
    column is in both cohort-level tables) and per MMS; MMS fractions summing
    to 1 per group; every MMS of the fractions having factors for its herd;
    consistent MMS sets across the cohorts of a herd; and consistent herd ids
    and herd / cohort (/ phase) combinations across the three tables.
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

    # --- Uniqueness checks
    if cohort.duplicated(subset=group_cols, keep=False).any():
        abort(
            "Duplicate herd/cohort rows in `cohort_level_data` for grouping columns "
            f"{_cli_vals(group_cols)}."
        )
    if fraction.duplicated(subset=group_cols + [_MMS], keep=False).any():
        abort("Duplicate herd/cohort/manure-management rows in `manure_management_system_fraction`.")

    # --- MMS fraction sum-to-one per herd/cohort[/phase] group
    frac = as_float(fraction["manure_management_system_fraction"])
    tmp = fraction[group_cols].copy()
    tmp["__frac__"] = np.where(np.isnan(frac), 0.0, frac)
    tmp["__na__"] = np.isnan(frac)
    grouped = tmp.groupby(group_cols, sort=False, dropna=False)
    sums = grouped.agg(total=("__frac__", "sum"), has_na=("__na__", "any"))
    invalid = sums[sums["has_na"].to_numpy() | (np.abs(sums["total"].to_numpy() - 1) > 1e-8)]
    if len(invalid):
        # R: `invalid_sums[, do.call(paste, ...), .SDcols = ...][[1]]` keeps only the
        # first invalid group (j returns a character vector, `[[1]]` its first element).
        first = invalid.index[0]
        keys = [_paste_key(first if isinstance(first, tuple) else (first,))]
        abort(
            "For each herd/cohort group, the sum of MMS fractions in "
            "`manure_management_system_fraction` must equal 1. "
            f"Invalid groups: {_cli_vals(keys)}"
        )

    if factors.duplicated(subset=["herd_id", _MMS], keep=False).any():
        abort(
            "Duplicate herd_id + manure_management_system rows in `manure_management_system_factors`."
        )

    # --- MMS consistency checks: every MMS of the fractions needs factors
    #     (factors may contain additional MMS).
    frac_by_herd = _mms_sets_by(fraction, ["herd_id"])
    fact_by_herd = _mms_sets_by(factors, ["herd_id"])

    missing_herds = _setdiff([k[0] for k in frac_by_herd], [k[0] for k in fact_by_herd])
    if missing_herds:
        abort(
            "Herd IDs in `manure_management_system_fraction` not found in "
            f"`manure_management_system_factors`: {_cli_vals(missing_herds)}"
        )

    # merge(..., by = "herd_id") of the two per-herd lists: ordered by herd_id.
    fact_lookup = {_norm(k[0]): v for k, v in fact_by_herd.items()}
    merged = sorted(
        ((k[0], v, fact_lookup[_norm(k[0])]) for k, v in frac_by_herd.items() if _norm(k[0]) in fact_lookup),
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
    sets_by_cohort = _mms_sets_by(fraction, ["herd_id", "cohort_short"])
    herd_sets: dict[Any, tuple[Any, set[str]]] = {}
    for (herd, _cohort), mms_list in sets_by_cohort.items():
        entry = herd_sets.setdefault(_norm(herd), (herd, set()))
        entry[1].add("|".join(mms_list))
    inconsistent = [herd for herd, sets in herd_sets.values() if len(sets) > 1]
    if inconsistent:
        abort(
            "Within each herd_id, manure_management_system lists must be consistent across cohorts "
            f"in `manure_management_system_fraction`. Inconsistent herds: {_cli_vals(inconsistent)}"
        )

    # --- Cross-table herd_id validation
    input_herds = _unique(cohort["herd_id"])
    fraction_herds = _unique(fraction["herd_id"])
    factors_herds = _unique(factors["herd_id"])

    # Each herd/cohort[/phase] of the input must exist in the fraction table
    # (fsetdiff(unique(input_pairs), unique(fraction_pairs)); NA matches NA).
    missing_rows = _missing_key_rows(cohort, fraction, group_cols)
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


def _mms_sets_by(df: pd.DataFrame, by: list[str]) -> dict[tuple[Any, ...], tuple[str, ...]]:
    """``df[, .(mms_list = list(sort(unique(manure_management_system)))), by = by]``.

    Groups in first-appearance order, keyed by their original key values.
    """
    t = df[by + [_MMS]].drop_duplicates()
    t = t[t[_MMS].notna()]
    sets = t.groupby(by, sort=False, dropna=False)[_MMS].agg(lambda x: tuple(sorted(str(v) for v in x)))
    return {(k if isinstance(k, tuple) else (k,)): v for k, v in sets.items()}


def _missing_key_rows(x: pd.DataFrame, y: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Positions in ``x`` of the first rows of the key combinations absent from ``y``.

    ``data.table::fsetdiff(unique(x[, cols]), unique(y[, cols]))`` with NA
    matching NA, in ``x`` order.
    """
    fx: dict[str, np.ndarray] = {}
    fy: dict[str, np.ndarray] = {}
    for i, col in enumerate(cols):
        (vx, nx), (vy, ny) = _aligned_keys(x[col], y[col])
        filler = 0.0 if vx.dtype.kind == "f" else ""
        fx[f"v{i}"], fy[f"v{i}"] = np.where(nx, filler, vx), np.where(ny, filler, vy)
        fx[f"n{i}"], fy[f"n{i}"] = nx, ny
    on = list(fx)
    kx = pd.DataFrame(fx)
    kx["__row__"] = np.arange(len(x))
    kx = kx.drop_duplicates(subset=on)
    ky = pd.DataFrame(fy).drop_duplicates()
    m = kx.merge(ky, on=on, how="left", indicator=True, sort=False)
    return m.loc[m["_merge"] == "left_only", "__row__"].to_numpy()


def _key(s: pd.Series) -> tuple[np.ndarray, np.ndarray, bool]:
    """(values, is-NA mask, numeric?) of a key column; all-missing columns count as numeric."""
    na = s.isna().to_numpy()
    if s.dtype.kind in "iufb":
        return s.to_numpy(dtype="float64", na_value=np.nan), na, True
    if na.all():
        return np.full(len(s), np.nan), na, True
    return s.to_numpy(dtype=object), na, False


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
    for v, na, numeric in keys:
        conv = _as_character if numeric else str
        out.append((np.array([None if m else conv(x) for x, m in zip(v, na)], dtype=object), na))
    return out


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
