"""Benchmark run_gleam() on the bundled examples and on replicated herds.

Usage: python tools/benchmark.py [N_REPLICATES]
"""

from __future__ import annotations

import sys
import time
import warnings

import pandas as pd

import gleam
from gleam.io import example_path, read_csv


def ex(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_gleam_examples"))


def inputs(structure: bool) -> dict:
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
    """Copy every herd n times under new herd ids."""
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


def timeit(args: dict, validate: bool, reps: int = 3) -> float:
    best = float("inf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(reps):
            t0 = time.perf_counter()
            gleam.run_gleam(**args, show_indicator=False, validate_inputs=validate)
            best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    for label, structure in (("with herd structure", True), ("herd simulation (demo + non-demo)", False)):
        base = inputs(structure)
        big = replicate(base, n)
        n_herds = base["herd_level_data"]["herd_id"].nunique()
        print(f"{label}:")
        for name, args, reps in ((f"{n_herds} herds", base, 5), (f"{n_herds * n} herds", big, 1)):
            rows = len(args["cohort_level_data"])
            t_v = timeit(args, True, reps)
            t_nv = timeit(args, False, reps)
            print(f"  {name:>11} ({rows:>6} cohort rows): {t_v:7.3f} s validated, {t_nv:7.3f} s unvalidated")


if __name__ == "__main__":
    main()
