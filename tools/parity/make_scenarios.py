"""Generate randomly perturbed GLEAM scenarios for R-vs-Python parity checks.

Each scenario is a directory with the run_gleam() input tables (tab-separated,
full precision) and ``args.json`` (pipeline switches, simulation duration and
GWP set). ``run_scenarios.R`` runs them through the R package and
``compare_scenarios.py`` runs them through the Python port and compares.

Usage:
    python tools/parity/make_scenarios.py OUT_DIR [N_PER_BASE] [SEED]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gleam.io import example_path, read_csv

BASES = {
    # name: (cohort file, herd file, herd filter, has_herd_structure, run_demographic, run_nondemographic)
    "structure": ("master_chrt_lvl_structure_data.csv", "master_hrd_lvl_structure_data.csv", None, True, False, False),
    "mixed": ("master_chrt_lvl_no_structure_mixed_data.csv", "master_hrd_lvl_mixed_data.csv", "demo", False, True, True),
    "nondemo": ("master_chrt_lvl_no_structure_nondemo_data.csv", "master_hrd_lvl_nondemo_data.csv", "nondemo", False, False, True),
}
NONDEMO_HERDS = (14, 15)
GWP_SETS = ["AR6", "AR5_excluding_carbon_feedback", "AR5_including_carbon_feedback", "AR4"]

# Herd-level parameters scaled by one common factor per herd (keeps orderings).
LIVE_WEIGHTS = [
    "live_weight_female_adult", "live_weight_male_adult", "live_weight_at_birth", "live_weight_at_weaning",
    "live_weight_female_at_slaughter", "live_weight_male_at_slaughter",
    "live_weight_female_nondemographic_start", "live_weight_male_nondemographic_start",
    "live_weight_female_nondemographic_end", "live_weight_male_nondemographic_end",
]
# Herd-level parameters scaled independently, with (low, high) clip bounds.
HERD_SCALED = {
    "milk_yield_day": (0, 100), "milk_fat_fraction": (0, 1), "milk_protein_fraction": (0, 1),
    "fibre_yield_year": (0, None), "egg_average_weight": (0, None), "egg_output_human_consumption": (0, None),
    "draught_work_hours_female": (0, 24), "draught_work_hours_male": (0, 24),
    "carcass_dressing_fraction": (0, 1), "bone_free_meat_fraction": (0, 1), "meat_protein_fraction": (0, 1),
    "parturition_rate": (0.01, None), "litter_size": (0.01, 25), "lactating_females_fraction": (0, 1),
    "pregnancy_duration": (1, None), "age_first_parturition": (110, 3300), "herd_size_total": (1, None),
}
COHORT_SCALED = {
    "offtake_rate": (0, 0.99), "death_rate": (0, 0.99), "high_activity_fraction": (0, 1),
    "low_activity_fraction": (0, 1), "cohort_stock_size": (0, None), "offtake_heads_assessment": (0, None),
}
FEED_PARAM_SCALED = [
    "feed_gross_energy", "feed_digestible_energy_ruminant", "feed_digestible_energy_pigs",
    "feed_metabolizable_energy_ruminant", "feed_metabolizable_energy_pigs",
    "feed_metabolizable_energy_chicken", "feed_nitrogen_content",
]
FEED_EMISSION_SCALED = [
    "co2_feed_fertilizer", "co2_feed_pesticides", "co2_feed_crop_activities", "co2_feed_luc_nopeat",
    "co2_feed_luc_peat", "n2o_feed_fertilizer", "n2o_feed_manure_applied", "n2o_feed_crop_residues", "ch4_feed_rice",
]
MMS_FACTOR_SCALED = {
    "methane_conversion_factor_mcf": (0, 100), "ch4_max_producing_capacity_bo": (1e-6, None),
    "n2o_ef3": (0, 1), "n2o_ef4": (0, 1), "n2o_ef5": (0, 1), "nitrogen_fracgas": (0, 1), "nitrogen_fracleach": (0, 1),
}


def _scale(df: pd.DataFrame, col: str, factors: np.ndarray, lo=None, hi=None) -> None:
    if col not in df.columns or df[col].dtype.kind not in "iuf":
        return
    v = df[col].to_numpy(dtype=float) * factors
    if lo is not None:
        v = np.maximum(v, lo)
    if hi is not None:
        v = np.minimum(v, hi)
    df[col] = v


def _renormalize(df: pd.DataFrame, value_col: str, keys: list[str], rng: np.random.Generator, spread: float) -> None:
    v = df[value_col].to_numpy(dtype=float) * rng.uniform(1 - spread, 1 + spread, len(df))
    tmp = df[keys].copy()
    for k in keys:
        tmp[k] = tmp[k].astype(str)
    tmp["v"] = v
    tot = tmp.groupby(keys, sort=False)["v"].transform("sum").to_numpy()
    df[value_col] = np.where(tot > 0, v / tot, df[value_col].to_numpy(dtype=float))


def _filter(df: pd.DataFrame, which: str | None) -> pd.DataFrame:
    if which is None or "herd_id" not in df.columns:
        return df.reset_index(drop=True)
    m = df["herd_id"].isin(NONDEMO_HERDS)
    return df[m if which == "nondemo" else ~m].reset_index(drop=True)


def make_scenario(base: str, rng: np.random.Generator, spread: float = 0.15) -> tuple[dict, dict]:
    chrt_f, hrd_f, which, has_structure, run_demo, run_nondemo = BASES[base]
    ex = lambda name: read_csv(example_path(name, "run_gleam_examples"))  # noqa: E731
    chrt = _filter(ex(chrt_f), which)
    hrd = _filter(ex(hrd_f), which)
    tables = {
        "cohort_level_data": chrt,
        "herd_level_data": hrd,
        "feed_rations": _filter(ex("feed_rations_share_chrt.csv"), which),
        "feed_params": ex("feed_quality.csv"),
        "feed_emissions": ex("feed_emission_factors.csv"),
        "manure_management_system_fraction": _filter(ex("manure_management_system_fraction.csv"), which),
        "manure_management_system_factors": _filter(ex("manure_management_system_factors.csv"), which),
    }
    u = lambda n: rng.uniform(1 - spread, 1 + spread, n)  # noqa: E731

    weight_factor = u(len(hrd))
    for col in LIVE_WEIGHTS:
        _scale(hrd, col, weight_factor, 0, None)
    for col, (lo, hi) in HERD_SCALED.items():
        _scale(hrd, col, u(len(hrd)), lo, hi)
    if "average_annual_temperature" in hrd.columns:
        hrd["average_annual_temperature"] = hrd["average_annual_temperature"] + rng.uniform(-3, 3, len(hrd))
    for col, (lo, hi) in COHORT_SCALED.items():
        _scale(chrt, col, u(len(chrt)), lo, hi)
    # One factor per feed for all energy columns keeps digestibility (DE/GE,
    # ME/GE) ratios valid; nitrogen content is scaled independently.
    energy_factor = rng.uniform(0.95, 1.05, len(tables["feed_params"]))
    for col in FEED_PARAM_SCALED[:-1]:
        _scale(tables["feed_params"], col, energy_factor, 0, 45)
    _scale(tables["feed_params"], "feed_nitrogen_content", u(len(tables["feed_params"])), 0, 0.15)
    for col in FEED_EMISSION_SCALED:
        _scale(tables["feed_emissions"], col, u(len(tables["feed_emissions"])), 0, None)
    for col, (lo, hi) in MMS_FACTOR_SCALED.items():
        _scale(tables["manure_management_system_factors"], col, u(len(tables["manure_management_system_factors"])), lo, hi)
    _renormalize(tables["feed_rations"], "feed_ration_fraction",
                 ["herd_id", "cohort_short", "nondemo_productive_phase_id"], rng, spread)
    _renormalize(tables["manure_management_system_fraction"], "manure_management_system_fraction",
                 ["herd_id", "cohort_short", "nondemo_productive_phase_id"], rng, spread)

    args = {
        "has_herd_structure": has_structure,
        "run_demographic": run_demo,
        "run_nondemographic": run_nondemo,
        "simulation_duration": float(rng.choice([365.0, 180.0, 730.0])),
        "global_warming_potential_set": str(rng.choice(GWP_SETS)),
        "base": base,
    }
    return tables, args


def write_scenario(out: Path, tables: dict, args: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", sep="\t", index=False, na_rep="", float_format="%.17g")
    (out / "args.json").write_text(json.dumps(args, indent=2))


def main() -> None:
    out_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261006
    rng = np.random.default_rng(seed)
    for base in BASES:
        for i in range(n):
            tables, args = make_scenario(base, rng)
            write_scenario(out_dir / f"{base}_{i:03d}", tables, args)
    print(f"wrote {n * len(BASES)} scenarios to {out_dir}")


if __name__ == "__main__":
    main()
