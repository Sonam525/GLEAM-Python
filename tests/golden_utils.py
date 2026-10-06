"""Helpers for comparing Python results with the R golden outputs.

Golden outputs live in ``tests/golden/<case>/<table>.csv`` and are produced by
``tools/r_reference/generate_golden.R`` from the original R package.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from gleam._utils import as_bool, as_float, as_str, is_na
from gleam.io import read_csv

GOLDEN_DIR = Path(__file__).parent / "golden"


def golden_path(case: str, table: str) -> Path:
    return GOLDEN_DIR / case / f"{table}.csv"


def load_golden(case: str, table: str) -> pd.DataFrame:
    path = golden_path(case, table)
    err = GOLDEN_DIR / case / "ERROR.txt"
    if err.exists():
        raise AssertionError(f"R reference failed for case {case!r}: {err.read_text()}")
    return read_csv(path)


def golden_tables(case: str) -> list[str]:
    return sorted(p.stem for p in (GOLDEN_DIR / case).glob("*.csv"))


def _kind(s: pd.Series) -> str:
    if s.dtype.kind in "iuf":
        return "num"
    vals = [v for v in s.tolist() if not is_na(v)]
    if vals and all(isinstance(v, (bool, np.bool_)) for v in vals):
        return "bool"
    if not vals:
        return "empty"
    return "str"


def assert_frame_matches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: Sequence[str] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-10,
    check_column_order: bool = True,
    check_row_order: bool = True,
    ignore_columns: Sequence[str] = (),
    label: str = "",
) -> None:
    """Assert a Python result table equals an R golden table.

    Numbers are compared with ``rtol``/``atol`` and matching NA positions;
    strings and logicals exactly. When ``check_row_order`` is false, both
    tables are sorted by ``sort_by`` (or all identifier-like columns) first.
    """
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

    problems = []
    for col in exp.columns:
        e, a = exp[col], act[col]
        k = _kind(e)
        if k == "num" or (k == "empty" and _kind(a) in ("num", "empty")):
            ev, av = as_float(e), as_float(a)
            same_na = np.isnan(ev) == np.isnan(av)
            close = np.isclose(av, ev, rtol=rtol, atol=atol, equal_nan=True) & same_na
            if not close.all():
                i = int(np.flatnonzero(~close)[0])
                problems.append(
                    f"{col}: {int((~close).sum())} mismatches, first at row {i}: actual={av[i]!r} expected={ev[i]!r}"
                )
        elif k == "bool":
            ev, av = as_bool(e), as_bool(a)
            bad = [i for i in range(len(ev)) if ev[i] != av[i]]
            if bad:
                problems.append(f"{col}: {len(bad)} mismatches, first at row {bad[0]}: actual={av[bad[0]]!r} expected={ev[bad[0]]!r}")
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
    assert_frame_matches(actual, load_golden(case, table), label=f"{case}/{table}", **kwargs)
