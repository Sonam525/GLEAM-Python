"""Build the extra golden-case inputs in tests/data/ from the bundled examples.

The bundled ``run_gleam`` examples (and therefore the original golden cases)
contain non-demographic cohorts only for CTL males, PGS and CHK, and every
CHK ``FA`` row lays eggs. The inputs written here close those gaps:

``ruminant_nondemo/``
    Herds 1 (CTL), 3 (SHP), 5 (GTS), 7 (BFL) and 11 (CML) of the mixed
    no-structure example, each with non-demographic female (``FN``) and male
    (``MN``) fattening cohorts in phases 1 and 2. SHP gets ``FN`` only: R
    rejects every SHP ``MN`` row (see ``shp_mn_rejected/``). Rations and
    manure shares of ``FN`` / ``MN`` copy the ``FS`` / ``MS`` rows.
    ``cohort_level_data_structure.csv`` is the herd structure the
    no-structure run produces (cohort stocks, offtake and durations), used as
    input to the ``has_herd_structure = TRUE`` path.
``chk_nonlaying/``
    CHK herd 13 with a non-laying adult female cohort
    (``is_egg_producing = FALSE`` on ``FA``), in the no-structure and the
    structure layout.
``shp_mn_rejected/``
    SHP herd 3 with ``MN`` cohorts. R's maintenance validator checks
    ``offtake_rate < 1`` for every SHP male cohort while the
    non-demographic herd module sets ``offtake_rate = 1``, so R (and the port)
    reject it.
``direct_rq/``
    The run_gleam structure example (CHK, ``FN``, ``MN``) with the primary
    ration quality columns merged in by herd, cohort and phase, for
    ``run_emissions_direct()`` without feed tables.
``mer_chk_growth/``
    The ``run_metabolic_energy_req_module()`` example plus two CHK herds
    whose adult females gain weight (in the pipelines adult cohorts have a
    daily weight gain of 0, so their growth coefficient never matters): herd
    16 does not lay (``is_egg_producing = FALSE`` on ``FA``), herd 17 does.

Run from the repository root (the files are committed; rerun only to change
them, then regenerate the matching golden cases with
``tools/r_reference/generate_golden.R``)::

    python tests/data/build_golden_inputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import gleampy
from gleampy.io import example_path, read_csv

DATA = Path(__file__).resolve().parent
KEYS = ["herd_id", "species_short", "cohort_short", "nondemo_productive_phase_id"]
# Written as plain integers (like the bundled examples), so R reads them as
# integer ids; every other numeric column keeps its float spelling.
INT_COLS = ("herd_id", "nondemo_productive_phase_id")


def _ex(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_gleam_examples"))


def _herds(df: pd.DataFrame, herds) -> pd.DataFrame:
    return df[df["herd_id"].isin(herds)].reset_index(drop=True)


def _write(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in INT_COLS:
        if col in out.columns:
            out[col] = pd.array(
                [None if pd.isna(v) else int(v) for v in out[col].tolist()], dtype="Int64"
            )
    if "is_egg_producing" in out.columns:
        out["is_egg_producing"] = [
            "NA" if v is None or (isinstance(v, float) and np.isnan(v)) else ("TRUE" if v else "FALSE")
            for v in out["is_egg_producing"].tolist()
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False, na_rep="NA", lineterminator="\n")


def _copy_cohort_rows(df: pd.DataFrame, herd: int, src: str, dst: str, phases=(1, 2)) -> pd.DataFrame:
    """Rows of cohort ``src`` of ``herd``, relabelled as cohort ``dst`` for each phase."""
    rows = df[(df["herd_id"] == herd) & (df["cohort_short"] == src)]
    out = []
    for ph in phases:
        r = rows.copy()
        r["cohort_short"] = dst
        r["nondemo_productive_phase_id"] = float(ph)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in ("herd_id", "cohort_short", "nondemo_productive_phase_id") if k in df.columns]
    return df.sort_values(keys, kind="mergesort", na_position="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ruminant / camel non-demographic cohorts
# ---------------------------------------------------------------------------

RUMINANT_HERDS = [1, 3, 5, 7, 11]  # CTL, SHP, GTS, BFL, CML
NO_MN = {3}  # SHP: MN is rejected by R (see shp_mn_rejected)


def _ruminant_tables() -> dict[str, pd.DataFrame]:
    chrt = _herds(_ex("master_chrt_lvl_no_structure_mixed_data.csv"), RUMINANT_HERDS)
    hrd = _herds(_ex("master_hrd_lvl_mixed_data.csv"), RUMINANT_HERDS)
    rat = _herds(_ex("feed_rations_share_chrt.csv"), RUMINANT_HERDS)
    frac = _herds(_ex("manure_management_system_fraction.csv"), RUMINANT_HERDS)
    fac = _herds(_ex("manure_management_system_factors.csv"), RUMINANT_HERDS)

    template = chrt[(chrt["herd_id"] == 1) & (chrt["cohort_short"] == "MN")]
    new_rows, new_rat, new_frac = [], [], []
    for herd in RUMINANT_HERDS:
        sp = hrd.loc[hrd["herd_id"] == herd, "species_short"].iloc[0]
        act = chrt.loc[chrt["herd_id"] == herd, ["high_activity_fraction", "low_activity_fraction"]].iloc[0]
        for cohort in ("FN", "MN"):
            if cohort == "MN" and (herd in NO_MN or herd == 1):
                continue  # herd 1 already has MN rows
            r = template.copy()
            r["herd_id"] = herd
            r["species_short"] = sp
            r["cohort_short"] = cohort
            r["high_activity_fraction"] = act["high_activity_fraction"]
            r["low_activity_fraction"] = act["low_activity_fraction"]
            r["death_rate"] = [0.08, 0.05]
            new_rows.append(r)
            src = "FS" if cohort == "FN" else "MS"
            new_rat.append(_copy_cohort_rows(rat, herd, src, cohort))
            new_frac.append(_copy_cohort_rows(frac, herd, src, cohort))
    chrt = _sort(pd.concat([chrt, *new_rows], ignore_index=True))
    rat = _sort(pd.concat([rat, *new_rat], ignore_index=True))
    frac = _sort(pd.concat([frac, *new_frac], ignore_index=True))

    big = {1: True, 7: True, 11: True}
    for i, herd in enumerate(hrd["herd_id"].tolist()):
        large = big.get(herd, False)
        hrd.loc[i, "prop_nondemo_fem_juv"] = 0.3
        hrd.loc[i, "prop_nondemo_mal_juv"] = 0.0 if herd in NO_MN else (0.95 if herd == 1 else 0.6)
        hrd.loc[i, "rest_between_nondemo_cycles_duration"] = 20.0 if large else 15.0
        hrd.loc[i, "phase1_nondemo_fem_duration_days"] = 60.0 if large else 30.0
        hrd.loc[i, "phase2_nondemo_fem_duration_days"] = 120.0 if large else 90.0
        if herd in NO_MN:
            hrd.loc[i, "phase1_nondemo_mal_duration_days"] = np.nan
            hrd.loc[i, "phase2_nondemo_mal_duration_days"] = np.nan
        else:
            hrd.loc[i, "phase1_nondemo_mal_duration_days"] = 60.0 if large else 30.0
            hrd.loc[i, "phase2_nondemo_mal_duration_days"] = 120.0 if large else 90.0
        if herd == 1:
            hrd.loc[i, "live_weight_female_nondemographic_start"] = 250.0
            hrd.loc[i, "live_weight_female_nondemographic_end"] = 600.0
    return {
        "cohort_level_data": chrt,
        "herd_level_data": hrd,
        "feed_rations": rat,
        "manure_management_system_fraction": frac,
        "manure_management_system_factors": fac,
    }


STRUCTURE_COLS = [
    "herd_id", "species_short", "cohort_short", "nondemo_productive_phase_id", "cohort_duration_days",
    "offtake_rate", "death_rate", "high_activity_fraction", "low_activity_fraction", "is_egg_producing",
    "offtake_heads", "offtake_heads_assessment", "cohort_stock_size",
]


def _structure_from_herd_model(tables: dict) -> pd.DataFrame:
    """Herd structure produced by the herd model of the no-structure run."""
    res = gleampy.run_gleam(
        has_herd_structure=False, run_demographic=True, run_nondemographic=True,
        feed_params=_ex("feed_quality.csv"), feed_emissions=_ex("feed_emission_factors.csv"),
        simulation_duration=365, show_indicator=False, **tables,
    )
    out = res["cohort_level_results"][STRUCTURE_COLS].copy()
    # Durations are whole days here; keep them exact.
    return _sort(out)


def build_ruminant() -> None:
    d = DATA / "ruminant_nondemo"
    tables = _ruminant_tables()
    for name, df in tables.items():
        _write(df, d / f"{name}.csv")
    _write(_structure_from_herd_model(tables), d / "cohort_level_data_structure.csv")


# ---------------------------------------------------------------------------
# CHK with a non-laying adult female cohort
# ---------------------------------------------------------------------------


def build_chk_nonlaying() -> None:
    d = DATA / "chk_nonlaying"
    herds = [13]
    for name, src in [
        ("cohort_level_data", "master_chrt_lvl_no_structure_mixed_data.csv"),
        ("cohort_level_data_structure", "master_chrt_lvl_structure_data.csv"),
    ]:
        df = _herds(_ex(src), herds)
        df["is_egg_producing"] = [False] * len(df)
        _write(df, d / f"{name}.csv")
    _write(_herds(_ex("master_hrd_lvl_mixed_data.csv"), herds), d / "herd_level_data.csv")
    for name in ("feed_rations_share_chrt", "manure_management_system_fraction", "manure_management_system_factors"):
        _write(_herds(_ex(f"{name}.csv"), herds), d / f"{name}.csv")


# ---------------------------------------------------------------------------
# SHP with MN cohorts (rejected by R)
# ---------------------------------------------------------------------------


def build_shp_mn() -> None:
    d = DATA / "shp_mn_rejected"
    tables = _ruminant_tables()
    herd = 3
    chrt = _herds(tables["cohort_level_data"], [herd])
    mn = chrt[chrt["cohort_short"] == "FN"].copy()
    mn["cohort_short"] = "MN"
    chrt = _sort(pd.concat([chrt, mn], ignore_index=True))
    hrd = _herds(tables["herd_level_data"], [herd])
    hrd["prop_nondemo_mal_juv"] = 0.5
    hrd["phase1_nondemo_mal_duration_days"] = 30.0
    hrd["phase2_nondemo_mal_duration_days"] = 90.0
    rat = _herds(tables["feed_rations"], [herd])
    frac = _herds(tables["manure_management_system_fraction"], [herd])
    rat = _sort(pd.concat([rat, _copy_cohort_rows(rat, herd, "MS", "MN")], ignore_index=True))
    frac = _sort(pd.concat([frac, _copy_cohort_rows(frac, herd, "MS", "MN")], ignore_index=True))
    _write(chrt, d / "cohort_level_data.csv")
    _write(hrd, d / "herd_level_data.csv")
    _write(rat, d / "feed_rations.csv")
    _write(frac, d / "manure_management_system_fraction.csv")
    _write(_herds(tables["manure_management_system_factors"], [herd]), d / "manure_management_system_factors.csv")


# ---------------------------------------------------------------------------
# run_emissions_direct with primary ration quality and phase rows
# ---------------------------------------------------------------------------

QUALITY_COLS = [
    "ration_gross_energy", "ration_metabolizable_energy", "ration_nitrogen",
    "ration_digestibility_fraction", "ration_urinary_energy_fraction", "ration_ash",
]


def build_direct_rq() -> None:
    chrt = _ex("master_chrt_lvl_structure_data.csv")
    quality = gleampy.run_ration_quality_module(
        _ex("feed_rations_share_chrt.csv"), _ex("feed_quality.csv"), show_indicator=False
    )
    merged = chrt.merge(quality[KEYS + QUALITY_COLS], on=KEYS, how="left", sort=False)
    assert len(merged) == len(chrt) and not merged[QUALITY_COLS].isna().any().any()
    _write(merged, DATA / "direct_rq" / "cohort_level_data.csv")


def build_mer_chk_growth() -> None:
    d = DATA / "mer_chk_growth"
    mod = lambda name: read_csv(example_path(name, "run_modules_examples"))  # noqa: E731
    chrt = mod("metabolic_energy_req_input_chrt_data.csv")
    hrd = mod("metabolic_energy_req_input_hrd_data.csv")
    extra_c, extra_h = [], []
    for herd, laying in ((16, False), (17, True)):
        c = chrt[chrt["herd_id"] == 13].copy()
        c["herd_id"] = herd
        fa = c["cohort_short"] == "FA"
        c.loc[fa, "is_egg_producing"] = laying
        c.loc[fa, "live_weight_cohort_initial"] = 1.6
        c.loc[fa, "live_weight_cohort_final"] = 1.8
        c.loc[fa, "live_weight_cohort_average"] = 1.7
        c.loc[fa, "daily_weight_gain"] = 0.2 / 365
        extra_c.append(c)
        h = hrd[hrd["herd_id"] == 13].copy()
        h["herd_id"] = herd
        extra_h.append(h)
    _write(pd.concat([chrt, *extra_c], ignore_index=True), d / "cohort_level_data.csv")
    _write(pd.concat([hrd, *extra_h], ignore_index=True), d / "herd_level_data.csv")


def main() -> None:
    build_ruminant()
    build_chk_nonlaying()
    build_shp_mn()
    build_direct_rq()
    build_mer_chk_growth()
    print(f"wrote inputs under {DATA}")


if __name__ == "__main__":
    main()
