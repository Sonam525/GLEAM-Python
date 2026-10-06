"""Golden parity tests for the allocation and aggregation modules.

The modules are run on the bundled example inputs exactly as
``tools/r_reference/generate_golden.R`` does, and every output table is
compared with the R output (rtol 1e-9, exact column and row order).
"""

from __future__ import annotations

import pytest
from golden_utils import assert_frame_matches, assert_matches_golden, load_golden

from gleam import run_aggregation_module, run_allocation_module
from gleam.io import load_example

ALLOCATION_TABLES = ("cohort_allocation_inputs", "allocation_long")
AGGREGATION_TABLES = ("results_emissions", "results_feed", "results_production", "results_nitrogen")
AGGREGATION_CASES = {
    "aggregation_module": "AR6",
    "aggregation_module_AR4": "AR4",
    "aggregation_module_AR5_excluding_carbon_feedback": "AR5_excluding_carbon_feedback",
    "aggregation_module_AR5_including_carbon_feedback": "AR5_including_carbon_feedback",
}


@pytest.fixture(scope="module")
def allocation_result():
    return run_allocation_module(
        cohort_level_data=load_example("allocation_input_chrt_data.csv"),
        herd_level_data=load_example("allocation_input_hrd_data.csv"),
        show_indicator=False,
    )


@pytest.fixture(scope="module")
def aggregation_results():
    return {
        case: run_aggregation_module(
            cohort_level_data=load_example("aggregation_input_chrt_data.csv"),
            allocation_herd_long=load_example("aggregation_allocation_input_data.csv"),
            simulation_duration=365,
            global_warming_potential_set=gwp,
            show_indicator=False,
        )
        for case, gwp in AGGREGATION_CASES.items()
    }


def test_allocation_module_returns_r_tables(allocation_result):
    assert list(allocation_result) == list(ALLOCATION_TABLES)


@pytest.mark.parametrize("table", ALLOCATION_TABLES)
def test_allocation_module_matches_golden(allocation_result, table):
    assert_matches_golden(allocation_result[table], "allocation_module", table)


def test_aggregation_module_returns_r_tables(aggregation_results):
    for result in aggregation_results.values():
        assert list(result) == list(AGGREGATION_TABLES)


@pytest.mark.parametrize("case", list(AGGREGATION_CASES))
@pytest.mark.parametrize("table", AGGREGATION_TABLES)
def test_aggregation_module_matches_golden(aggregation_results, case, table):
    assert_matches_golden(aggregation_results[case][table], case, table)


#: Full-pipeline golden cases whose final cohort / herd tables are the inputs
#: R passed to run_allocation_module (allocation columns are overwritten in
#: place by `:=`, so re-running allocation on them reproduces the same table).
PIPELINE_CASES = {
    "run_gleam_structure_AR6": ("AR6", 365),
    "run_gleam_structure_AR4": ("AR4", 365),
    "run_gleam_structure_AR5_excluding_carbon_feedback": ("AR5_excluding_carbon_feedback", 365),
    "run_gleam_structure_AR5_including_carbon_feedback": ("AR5_including_carbon_feedback", 365),
    "run_gleam_mixed_no_structure": ("AR6", 365),
    "run_gleam_mixed_no_structure_d180": ("AR6", 180),
    "run_gleam_nondemo_only": ("AR6", 365),
    "emissions_direct_1a_no_structure": ("AR6", 365),
    "emissions_direct_1b_structure": ("AR6", 365),
    "emissions_direct_2a_no_structure_rq": ("AR6", 365),
    "emissions_direct_2b_structure_rq": ("AR6", 365),
}


@pytest.mark.parametrize("case", list(PIPELINE_CASES))
def test_pipeline_allocation_and_aggregation_stages_match_golden(case):
    """Allocation + aggregation on the pipeline's final cohort / herd tables."""
    gwp, duration = PIPELINE_CASES[case]
    cohort = load_golden(case, "cohort_level_results")
    herd = load_golden(case, "herd_level_results")
    alloc = run_allocation_module(cohort, herd, simulation_duration=duration, show_indicator=False)
    assert_frame_matches(alloc["cohort_allocation_inputs"], cohort, label=f"{case}/cohort_level_results")
    assert_matches_golden(alloc["allocation_long"], case, "allocation_long")
    agg = run_aggregation_module(
        alloc["cohort_allocation_inputs"], alloc["allocation_long"],
        simulation_duration=duration, global_warming_potential_set=gwp, show_indicator=False,
    )
    for table in AGGREGATION_TABLES:
        assert_matches_golden(agg[table], case, f"aggregation_results__{table}")


def test_allocation_long_feeds_aggregation(allocation_result):
    """The allocation output is a valid ``allocation_herd_long`` input (pipeline use)."""
    chrt = load_example("aggregation_input_chrt_data.csv")
    res = run_aggregation_module(chrt, allocation_result["allocation_long"], show_indicator=False)
    em = res["results_emissions"]
    assert "commodity_type" in em.columns
    assert list(em.columns[:7]) == [
        "variable_name", "herd_id", "species_short", "variable_type", "value_total_gas",
        "commodity_name", "commodity_type",
    ]
