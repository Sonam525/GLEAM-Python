"""Run perturbed scenarios through the Python port and compare with R outputs.

Usage:
    python tools/parity/compare_scenarios.py SCENARIOS_DIR [--zero-floor F] [--stats]

Expects ``run_scenarios.R`` to have written ``<scenario>/r_out``. A scenario
passes when

* both succeed and every output table matches (same tables, same column and
  row order, numbers within ``rtol = 1e-9`` relative with no absolute term,
  so an R zero must be an exact zero: the rule of ``tests/golden_utils.py``),
  or
* both reject it with the same validation error: Python raises
  ``GleamValidationError`` and the two messages are identical once
  formatting is normalised (cli markup, backticks, Python's element index,
  ``>=``/``<=`` symbols, line wraps, and ``" and "`` vs ``", "`` between quoted
  items, see ``golden_utils.normalize_message``). This also requires the same
  check and the same list of violating rows.

Anything else fails: one side only rejecting the scenario, different
messages or violation lists (an R crash such as ``object 'x' not found``
never matches a Python validation message), or any other Python exception,
which is reported as a FAIL without stopping the run.

``--zero-floor F`` (e.g. ``1e-12``) turns on the opt-in near-zero rule of
``golden_utils.assert_frame_matches`` to tell cancellation noise around zero
from real differences when a scenario fails; the documented run uses no
floor. ``--stats`` also prints how closely the scenarios that ran agree:
the number of non-zero numbers in R's output tables (all numeric columns,
and the floating-point columns alone, i.e. without whole-number columns such
as ids, durations and counts), the share Python reproduces bit for bit, and
the largest relative difference, per base and overall.

The documented run (30 scenarios, 10 per base, seed 20261006)::

    python tools/parity/make_scenarios.py OUT 10 20261006
    Rscript tools/parity/run_scenarios.R ../GLEAM-reference OUT
    python tools/parity/compare_scenarios.py OUT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import gleampy  # noqa: E402
from gleampy._utils import as_float  # noqa: E402
from gleampy.io import read_csv  # noqa: E402
from golden_utils import assert_frame_matches, normalize_message, quoted_items  # noqa: E402


def _flatten(result: dict, prefix: str = "") -> dict[str, pd.DataFrame]:
    out = {}
    for k, v in result.items():
        name = f"{prefix}__{k}" if prefix else k
        if isinstance(v, pd.DataFrame):
            out[name] = v
        elif isinstance(v, dict):
            out.update(_flatten(v, name))
    return out


def _read_input(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    for col in df.columns:
        if df[col].dtype.kind not in "iufb":
            # fwrite(na = "") writes NA as an empty field, which read_csv (like
            # fread) keeps as "" in a character column such as these hex strings.
            vals = [v for v in df[col].tolist() if isinstance(v, str) and v != ""]
            if vals and all(v.lstrip("-").startswith("0x") or v in ("Inf", "-Inf", "NaN") for v in vals):
                df[col] = [
                    float.fromhex(v) if isinstance(v, str) and "0x" in v
                    else float(v.replace("Inf", "inf")) if isinstance(v, str) and v != "" else float("nan")
                    for v in df[col].tolist()
                ]
    return df


def errors_match(
    r_error: str | None, py_error: str | None, r_error_class: list[str] | None = None
) -> tuple[bool, str]:
    """Whether R and Python rejected a scenario with the same validation error.

    Parameters
    ----------
    r_error : str or None
        ``conditionMessage()`` of the R error (``r_out/ERROR.txt``), or None.
    py_error : str or None
        Message of the Python ``GleamValidationError``, or None.
    r_error_class : list of str, optional
        Classes of the R condition (``r_out/ERROR_CLASS.txt``). When given,
        an R error that is not an ``rlang_error`` (raised by
        ``cli::cli_abort()`` in GLEAM's validators) is an R crash and fails.

    Returns
    -------
    tuple of (bool, str)
        Whether they match, and the reason.
    """
    if not r_error and not py_error:
        return True, "neither failed"
    if not r_error:
        return False, "only Python rejected the scenario"
    if not py_error:
        return False, "only R failed"
    if r_error_class is not None and "rlang_error" not in r_error_class:
        return False, f"R crashed ({'/'.join(r_error_class)}), not a validation error"
    r_norm, py_norm = normalize_message(r_error), normalize_message(py_error)
    if quoted_items(r_error) != quoted_items(py_error):
        return False, (
            f"different violation lists: R {quoted_items(r_error)} vs Python {quoted_items(py_error)}"
        )
    if r_norm != py_norm:
        return False, f"different messages: R {r_norm!r} vs Python {py_norm!r}"
    return True, "same validation error"


def agreement(py: pd.DataFrame, r: pd.DataFrame, label: str = "") -> dict:
    """Bit-level agreement of the numbers of one output table that matched.

    Parameters
    ----------
    py, r : pandas.DataFrame
        Python and R versions of the table (same columns and row order).
    label : str
        Table name, recorded with the largest difference.

    Returns
    -------
    dict
        ``values`` / ``identical``: the non-zero finite numbers of R's
        numeric columns and how many of them Python reproduces bit for bit;
        ``float_values`` / ``float_identical``: the same for floating-point
        columns only; ``max_rel`` and ``where`` (table, column, R value,
        Python value): the largest relative difference.
    """
    out = dict(values=0, identical=0, float_values=0, float_identical=0, max_rel=0.0, where=None)
    for col in r.columns:
        kind = r[col].dtype.kind
        if kind not in "iuf":
            continue
        rv, pv = as_float(r[col]), as_float(py[col])
        m = np.isfinite(rv) & (rv != 0)
        if not m.any():
            continue
        same = int((pv[m] == rv[m]).sum())
        out["values"] += int(m.sum())
        out["identical"] += same
        if kind == "f":
            out["float_values"] += int(m.sum())
            out["float_identical"] += same
        rel = np.abs(pv[m] - rv[m]) / np.abs(rv[m])
        i = int(np.argmax(rel))
        if rel[i] > out["max_rel"]:
            out["max_rel"] = float(rel[i])
            out["where"] = (label, col, float(rv[m][i]), float(pv[m][i]))
    return out


def _add_agreement(total: dict, part: dict) -> None:
    for k in ("values", "identical", "float_values", "float_identical"):
        total[k] = total.get(k, 0) + part[k]
    if part["max_rel"] >= total.get("max_rel", 0.0):
        total["max_rel"], total["where"] = part["max_rel"], part["where"] or total.get("where")


def compare(sd: Path, zero_floor: float = 0.0, stats: dict | None = None) -> tuple[bool, str, float]:
    """Run one scenario through Python and compare it with R's outcome.

    Parameters
    ----------
    sd : pathlib.Path
        Scenario directory (``args.json``, input tables, ``r_in/`` and
        ``r_out/`` from ``run_scenarios.R``).
    zero_floor : float
        Opt-in near-zero rule passed to ``assert_frame_matches`` (default 0:
        an R zero must be an exact zero).
    stats : dict, optional
        When given and every table matches, the :func:`agreement` of the
        scenario's tables is added to ``stats[base]`` and ``stats["all"]``.

    Returns
    -------
    tuple of (bool, str, float)
        Whether R and Python agree, a description, and the Python run time (s).
    """
    args = json.loads((sd / "args.json").read_text())
    base = args.pop("base", "?")
    # Prefer the inputs exactly as R parsed them (hex floats) when available.
    in_dir = sd / "r_in" if (sd / "r_in").exists() else sd
    tables = {
        n: _read_input(in_dir / f"{n}.csv")
        for n in (
            "cohort_level_data", "herd_level_data", "feed_rations", "feed_params", "feed_emissions",
            "manure_management_system_fraction", "manure_management_system_factors",
        )
    }
    r_out = sd / "r_out"
    r_error = (r_out / "ERROR.txt").read_text(encoding="utf-8").strip() if (r_out / "ERROR.txt").exists() else None
    r_class_file = r_out / "ERROR_CLASS.txt"
    r_error_class = r_class_file.read_text(encoding="utf-8").split() if r_class_file.exists() else None
    t0 = time.perf_counter()
    try:
        res = gleampy.run_gleam(**tables, **args, show_indicator=False)
        py_error = None
    except gleampy.GleamValidationError as e:
        res, py_error = None, str(e)
    except Exception as e:  # a crash is never a match, but must not stop the run
        dt = time.perf_counter() - t0
        return False, f"Python crashed with {type(e).__name__}: {e} | R error: {r_error!r}", dt
    dt = time.perf_counter() - t0
    if r_error or py_error:
        ok, why = errors_match(r_error, py_error, r_error_class)
        return ok, f"{why} | R error: {r_error!r} | Python error: {py_error!r}", dt
    py_tables = _flatten(res)
    r_tables = sorted(p.stem for p in r_out.glob("*.csv"))
    if sorted(py_tables) != r_tables:
        return False, f"table sets differ: {sorted(py_tables)} vs {r_tables}", dt
    r_frames = {t: read_csv(r_out / f"{t}.csv") for t in r_tables}
    try:
        for t in r_tables:
            assert_frame_matches(py_tables[t], r_frames[t], label=t, zero_floor=zero_floor)
    except AssertionError as e:
        return False, str(e)[:2000], dt
    if stats is not None:
        for t in r_tables:
            part = agreement(py_tables[t], r_frames[t], f"{sd.name}/{t}")
            for key in (base, "all"):
                entry = stats.setdefault(key, {"scenarios": set()})
                entry["scenarios"].add(sd.name)
                _add_agreement(entry, part)
    return True, "all tables match", dt


def main() -> None:
    """Command-line entry point (arguments in the module docstring); exits 1 if any scenario fails."""
    # R messages contain symbols such as "≥" that a legacy console code page
    # cannot encode; escape them instead of crashing.
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Compare R and Python outputs of perturbed scenarios.")
    parser.add_argument("scenarios_dir", type=Path, help="directory written by make_scenarios.py")
    parser.add_argument(
        "--zero-floor", type=float, default=0.0,
        help="opt-in near-zero rule (e.g. 1e-12) for diagnosing cancellation noise; default 0 (off)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="also print the bit-level agreement of the scenarios that ran"
    )
    opts = parser.parse_args()
    n_ok = 0
    stats: dict | None = {} if opts.stats else None
    dirs = sorted(d for d in opts.scenarios_dir.iterdir() if (d / "args.json").exists())
    for sd in dirs:
        ok, msg, dt = compare(sd, zero_floor=opts.zero_floor, stats=stats)
        n_ok += ok
        print(f"{'PASS' if ok else 'FAIL'} {sd.name} ({dt:.2f}s python): {msg}")
    print(f"\n{n_ok}/{len(dirs)} scenarios match")
    for key, st in sorted((stats or {}).items(), key=lambda kv: (kv[0] == "all", kv[0])):
        print(
            f"{key}: {len(st['scenarios'])} scenarios ran; {st['values']} non-zero numbers, "
            f"{st['identical'] / max(st['values'], 1):.1%} bit-identical; "
            f"{st['float_values']} in floating-point columns, "
            f"{st['float_identical'] / max(st['float_values'], 1):.1%} bit-identical; "
            f"largest relative difference {st['max_rel']:.3g} at {st['where']}"
        )
    sys.exit(0 if n_ok == len(dirs) else 1)


if __name__ == "__main__":
    main()
