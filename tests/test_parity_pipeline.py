"""End-to-end parity of run_gleam() / run_emissions_direct() with the R package.

Each case reproduces one call in tools/r_reference/generate_golden.R and
compares every output table with the R golden output (rtol 1e-9, exact column
and row order).
"""

from __future__ import annotations

import pandas as pd
import pytest

import gleam
from gleam.io import example_path, read_csv
from golden_utils import assert_matches_golden, golden_tables

NONDEMO_HERDS = (14, 15)


def _run(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_gleam_examples"))


def _mod(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_modules_examples"))


def _filter(df: pd.DataFrame, keep_nondemo: bool | None) -> pd.DataFrame:
    if keep_nondemo is None:
        return df
    mask = df["herd_id"].isin(NONDEMO_HERDS)
    return df[mask if keep_nondemo else ~mask].reset_index(drop=True)


def _gleam_inputs(keep_nondemo: bool | None) -> dict:
    return dict(
        feed_rations=_filter(_run("feed_rations_share_chrt.csv"), keep_nondemo),
        feed_params=_run("feed_quality.csv"),
        feed_emissions=_run("feed_emission_factors.csv"),
        manure_management_system_fraction=_filter(_run("manure_management_system_fraction.csv"), keep_nondemo),
        manure_management_system_factors=_filter(_run("manure_management_system_factors.csv"), keep_nondemo),
    )


def _run_gleam_case(case: str) -> dict:
    if case.startswith("run_gleam_mixed_no_structure"):
        return gleam.run_gleam(
            has_herd_structure=False, run_demographic=True, run_nondemographic=True,
            cohort_level_data=_filter(_run("master_chrt_lvl_no_structure_mixed_data.csv"), False),
            herd_level_data=_filter(_run("master_hrd_lvl_mixed_data.csv"), False),
            simulation_duration=180 if case.endswith("_d180") else 365,
            show_indicator=False, **_gleam_inputs(False),
        )
    if case == "run_gleam_nondemo_only":
        return gleam.run_gleam(
            has_herd_structure=False, run_demographic=False, run_nondemographic=True,
            cohort_level_data=_run("master_chrt_lvl_no_structure_nondemo_data.csv"),
            herd_level_data=_run("master_hrd_lvl_nondemo_data.csv"),
            simulation_duration=365, show_indicator=False, **_gleam_inputs(True),
        )
    gwp = case.removeprefix("run_gleam_structure_")
    return gleam.run_gleam(
        has_herd_structure=True, run_demographic=False, run_nondemographic=False,
        cohort_level_data=_run("master_chrt_lvl_structure_data.csv"),
        herd_level_data=_run("master_hrd_lvl_structure_data.csv"),
        simulation_duration=365, global_warming_potential_set=gwp,
        show_indicator=False, **_gleam_inputs(None),
    )


def _run_direct_case(case: str) -> dict:
    herd = _mod("emissions_direct_input_hrd_data.csv")

    def in_direct(df):
        return df[df["herd_id"].isin(herd["herd_id"])].reset_index(drop=True)

    common = dict(
        herd_level_data=herd,
        manure_management_system_fraction=in_direct(_mod("manure_management_system_fraction.csv")),
        manure_management_system_factors=in_direct(_mod("manure_management_system_factors.csv")),
        simulation_duration=365, global_warming_potential_set="AR6", show_indicator=False,
    )
    feed = dict(feed_rations=in_direct(_mod("feed_rations_share_chrt.csv")), feed_params=_mod("feed_quality.csv"))
    cohort_file = {
        "emissions_direct_1a_no_structure": "emissions_direct_input_chrt_no_structure_data.csv",
        "emissions_direct_1b_structure": "emissions_direct_input_chrt_structure_data.csv",
        "emissions_direct_2a_no_structure_rq": "emissions_direct_input_chrt_no_structure_ration_quality_data.csv",
        "emissions_direct_2b_structure_rq": "emissions_direct_input_chrt_structure_ration_quality_data.csv",
        "emissions_direct_1b_structure_ef_only": "emissions_direct_input_chrt_structure_data.csv",
    }[case]
    args = dict(common, cohort_level_data=_mod(cohort_file), has_herd_structure="_structure" in case and "no_structure" not in case)
    if not case.endswith("_rq"):
        args.update(feed)
    if case.endswith("_ef_only"):
        args["emission_factors_only"] = True
    return gleam.run_emissions_direct(**args)


def _flatten(result: dict, prefix: str = "") -> dict[str, pd.DataFrame]:
    out = {}
    for k, v in result.items():
        name = f"{prefix}__{k}" if prefix else k
        if isinstance(v, pd.DataFrame):
            out[name] = v
        elif isinstance(v, dict):
            out.update(_flatten(v, name))
    return out


GLEAM_CASES = [
    "run_gleam_mixed_no_structure",
    "run_gleam_mixed_no_structure_d180",
    "run_gleam_nondemo_only",
    "run_gleam_structure_AR6",
    "run_gleam_structure_AR5_excluding_carbon_feedback",
    "run_gleam_structure_AR5_including_carbon_feedback",
    "run_gleam_structure_AR4",
]
DIRECT_CASES = [
    "emissions_direct_1a_no_structure",
    "emissions_direct_1b_structure",
    "emissions_direct_2a_no_structure_rq",
    "emissions_direct_2b_structure_rq",
    "emissions_direct_1b_structure_ef_only",
]


@pytest.fixture(scope="module")
def results() -> dict:
    return {}


@pytest.mark.parametrize("case", GLEAM_CASES + DIRECT_CASES)
def test_pipeline_matches_r(case, results):
    res = _run_gleam_case(case) if case.startswith("run_gleam") else _run_direct_case(case)
    tables = _flatten(res)
    expected = golden_tables(case)
    assert sorted(tables) == expected, f"output tables differ: {sorted(tables)} vs {expected}"
    for table in expected:
        assert_matches_golden(tables[table], case, table)
