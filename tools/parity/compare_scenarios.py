"""Run perturbed scenarios through the Python port and compare with R outputs.

Usage:
    python tools/parity/compare_scenarios.py SCENARIOS_DIR

Expects ``run_scenarios.R`` to have written ``<scenario>/r_out``. A scenario
passes when R and Python both fail validation, or both succeed and every
output table matches (rtol 1e-9, exact column and row order).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import gleam  # noqa: E402
from gleam.io import read_csv  # noqa: E402
from golden_utils import assert_frame_matches  # noqa: E402


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
            vals = [v for v in df[col].tolist() if isinstance(v, str)]
            if vals and all(v.lstrip("-").startswith("0x") or v in ("Inf", "-Inf", "NaN") for v in vals):
                df[col] = [
                    float.fromhex(v) if isinstance(v, str) and "0x" in v
                    else float(v.replace("Inf", "inf")) if isinstance(v, str) else float("nan")
                    for v in df[col].tolist()
                ]
    return df


def compare(sd: Path) -> tuple[bool, str, float]:
    args = json.loads((sd / "args.json").read_text())
    args.pop("base", None)
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
    r_error = (r_out / "ERROR.txt").read_text().strip() if (r_out / "ERROR.txt").exists() else None
    t0 = time.perf_counter()
    try:
        res = gleam.run_gleam(**tables, **args, show_indicator=False)
        py_error = None
    except gleam.GleamValidationError as e:
        res, py_error = None, str(e)
    dt = time.perf_counter() - t0
    if r_error or py_error:
        ok = bool(r_error) and bool(py_error)
        return ok, f"R error: {r_error!r} | Python error: {py_error!r}", dt
    py_tables = _flatten(res)
    r_tables = sorted(p.stem for p in r_out.glob("*.csv"))
    if sorted(py_tables) != r_tables:
        return False, f"table sets differ: {sorted(py_tables)} vs {r_tables}", dt
    try:
        for t in r_tables:
            assert_frame_matches(py_tables[t], read_csv(r_out / f"{t}.csv"), label=t)
    except AssertionError as e:
        return False, str(e)[:2000], dt
    return True, "all tables match", dt


def main() -> None:
    root = Path(sys.argv[1])
    n_ok = 0
    dirs = sorted(d for d in root.iterdir() if (d / "args.json").exists())
    for sd in dirs:
        ok, msg, dt = compare(sd)
        n_ok += ok
        print(f"{'PASS' if ok else 'FAIL'} {sd.name} ({dt:.2f}s python): {msg}")
    print(f"\n{n_ok}/{len(dirs)} scenarios match")
    sys.exit(0 if n_ok == len(dirs) else 1)


if __name__ == "__main__":
    main()
