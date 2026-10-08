"""Benchmark run_gleam() on the bundled examples and on replicated herds.

Usage::

    python tools/benchmark.py [N_REPLICATES] [--save FILE | --compare FILE]

``--save FILE`` also stores every output table of the replicated runs
(both pipeline modes, validated) in ``FILE`` (a pickle). ``--compare FILE``
runs the same inputs and checks that every table is bit-identical to the
saved one (same columns, dtypes, values and NaN positions), which is the
before/after check required for performance changes::

    python tools/benchmark.py 100 --save before.pkl   # on the old code
    python tools/benchmark.py 100 --compare before.pkl  # on the new code
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import gleampy
from gleampy.io import example_path, read_csv

MODES = (("with herd structure", True), ("herd simulation (demo + non-demo)", False))


def ex(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_gleam_examples"))


def inputs(structure: bool) -> dict:
    """run_gleam() arguments for the structure or the mixed no-structure example.

    Parameters
    ----------
    structure : bool
        True for the herd-structure example (15 herds), False for the mixed
        no-structure example without the CHK non-demographic herds 14 and 15.
    """
    keep = lambda df: df[~df["herd_id"].isin([14, 15])].reset_index(drop=True)  # noqa: E731
    if structure:
        return dict(
            has_herd_structure=True, run_demographic=False, run_nondemographic=False,
            cohort_level_data=ex("master_chrt_lvl_structure_data.csv"),
            herd_level_data=ex("master_hrd_lvl_structure_data.csv"),
            feed_rations=ex("feed_rations_share_chrt.csv"), feed_params=ex("feed_quality.csv"),
            feed_emissions=ex("feed_emission_factors.csv"),
            manure_management_system_fraction=ex("manure_management_system_fraction.csv"),
            manure_management_system_factors=ex("manure_management_system_factors.csv"),
        )
    return dict(
        has_herd_structure=False, run_demographic=True, run_nondemographic=True,
        cohort_level_data=keep(ex("master_chrt_lvl_no_structure_mixed_data.csv")),
        herd_level_data=keep(ex("master_hrd_lvl_mixed_data.csv")),
        feed_rations=keep(ex("feed_rations_share_chrt.csv")), feed_params=ex("feed_quality.csv"),
        feed_emissions=ex("feed_emission_factors.csv"),
        manure_management_system_fraction=keep(ex("manure_management_system_fraction.csv")),
        manure_management_system_factors=keep(ex("manure_management_system_factors.csv")),
    )


def replicate(args: dict, n: int) -> dict:
    """Copy every herd ``n`` times under new herd ids (``herd_id + 1000 * i``).

    Parameters
    ----------
    args : dict
        run_gleam() arguments.
    n : int
        Number of copies.
    """
    out = dict(args)
    for k, v in args.items():
        if isinstance(v, pd.DataFrame) and "herd_id" in v.columns:
            parts = []
            for i in range(n):
                p = v.copy()
                p["herd_id"] = p["herd_id"] + 1000 * i
                parts.append(p)
            out[k] = pd.concat(parts, ignore_index=True)
    return out


def run(args: dict, validate: bool) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gleampy.run_gleam(**args, show_indicator=False, validate_inputs=validate)


def timeit(args: dict, validate: bool, reps: int = 3) -> float:
    """Best wall time of ``reps`` run_gleam() calls (seconds)."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        run(args, validate)
        best = min(best, time.perf_counter() - t0)
    return best


def flatten(result: dict, prefix: str = "") -> dict[str, pd.DataFrame]:
    out = {}
    for k, v in result.items():
        name = f"{prefix}__{k}" if prefix else k
        if isinstance(v, pd.DataFrame):
            out[name] = v
        elif isinstance(v, dict):
            out.update(flatten(v, name))
    return out


def identical_tables(new: dict[str, pd.DataFrame], old: dict[str, pd.DataFrame]) -> list[str]:
    """Differences between two sets of output tables (empty when bit-identical).

    Parameters
    ----------
    new, old : dict of str to pandas.DataFrame
        Output tables by name.
    """
    problems = []
    if sorted(new) != sorted(old):
        return [f"table sets differ: {sorted(new)} vs {sorted(old)}"]
    for name, b in old.items():
        a = new[name]
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            problems.append(f"{name}: columns or row count differ")
            continue
        for col in b.columns:
            x, y = a[col], b[col]
            if x.dtype != y.dtype:
                problems.append(f"{name}.{col}: dtype {x.dtype} vs {y.dtype}")
            elif x.dtype.kind == "f":
                xv, yv = x.to_numpy(), y.to_numpy()
                same = (xv.view(np.int64) == yv.view(np.int64)) | (np.isnan(xv) & np.isnan(yv))
                if not same.all():
                    i = int(np.flatnonzero(~same)[0])
                    problems.append(f"{name}.{col}: {int((~same).sum())} values differ, first row {i}: {xv[i]!r} vs {yv[i]!r}")
            elif not x.equals(y):
                problems.append(f"{name}.{col}: values differ")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("n", nargs="?", type=int, default=100, help="number of herd replicates (default 100)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", type=Path, help="store the replicated runs' output tables here")
    group.add_argument("--compare", type=Path, help="check the outputs are bit-identical to a saved file")
    opts = parser.parse_args()

    outputs = {}
    for label, structure in MODES:
        base = inputs(structure)
        big = replicate(base, opts.n)
        n_herds = base["herd_level_data"]["herd_id"].nunique()
        print(f"{label}:")
        for name, args, reps in ((f"{n_herds} herds", base, 5), (f"{n_herds * opts.n} herds", big, 1)):
            rows = len(args["cohort_level_data"])
            t_v = timeit(args, True, reps)
            t_nv = timeit(args, False, reps)
            print(f"  {name:>11} ({rows:>6} cohort rows): {t_v:7.3f} s validated, {t_nv:7.3f} s unvalidated")
        if opts.save or opts.compare:
            outputs[label] = flatten(run(big, True))

    if opts.save:
        with open(opts.save, "wb") as fh:
            pickle.dump({"n": opts.n, "outputs": outputs}, fh)
        print(f"saved outputs of {len(outputs)} runs to {opts.save}")
    elif opts.compare:
        with open(opts.compare, "rb") as fh:
            saved = pickle.load(fh)
        if saved["n"] != opts.n:
            sys.exit(f"saved file is for {saved['n']} replicates, not {opts.n}")
        problems = [f"[{label}] {p}" for label in outputs for p in identical_tables(outputs[label], saved["outputs"][label])]
        if problems:
            print("NOT bit-identical:\n  " + "\n  ".join(problems[:50]))
            sys.exit(1)
        print("bit-identical: every output table matches the saved run")


if __name__ == "__main__":
    main()
