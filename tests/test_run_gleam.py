"""Tests for run_gleam() (port of tests/testthat/test-run_gleam.R)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gleam
from gleam import GleamValidationError
from gleam._utils import as_float, merge_dt, rbind_fill
from gleam.io import example_path, read_csv

DEMOGRAPHIC_COHORTS = ["FA", "FJ", "FS", "MA", "MJ", "MS"]


def _run(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_gleam_examples"))


def _mod(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_modules_examples"))


@pytest.fixture(scope="module")
def d_gleam() -> dict:
    cohort_no_structure = _run("master_chrt_lvl_no_structure_mixed_data.csv")
    cohort_structure = _run("master_chrt_lvl_structure_data.csv")
    herd = _run("master_hrd_lvl_mixed_data.csv")
    fn_herds = set(cohort_no_structure.loc[cohort_no_structure.cohort_short == "FN", "herd_id"])
    mn_herds = set(cohort_no_structure.loc[cohort_no_structure.cohort_short == "MN", "herd_id"])
    herd["prop_nondemo_fem_juv"] = herd["prop_nondemo_fem_juv"].fillna(0)
    herd["prop_nondemo_mal_juv"] = herd["prop_nondemo_mal_juv"].fillna(0)
    herd.loc[~herd.herd_id.isin(fn_herds), "prop_nondemo_fem_juv"] = 0
    herd.loc[~herd.herd_id.isin(mn_herds), "prop_nondemo_mal_juv"] = 0
    herd["live_weight_female_nondemographic_end"] = herd["live_weight_female_nondemographic_end"].fillna(
        herd["live_weight_female_at_slaughter"]
    )
    herd["live_weight_male_nondemographic_end"] = herd["live_weight_male_nondemographic_end"].fillna(
        herd["live_weight_male_at_slaughter"]
    )
    return {
        "cohort_no_structure": cohort_no_structure,
        "cohort_structure": cohort_structure,
        "herd": herd,
        "feed_rations": _run("feed_rations_share_chrt.csv"),
        "feed_params": _run("feed_quality.csv"),
        "feed_emissions": _run("feed_emission_factors.csv"),
        "mms_fraction": _run("manure_management_system_fraction.csv"),
        "mms_factors": _run("manure_management_system_factors.csv"),
    }


NONDEMO_HERD_COLS = [
    "prop_nondemo_fem_juv",
    "prop_nondemo_mal_juv",
    "rest_between_nondemo_cycles_duration",
    "phase1_nondemo_fem_duration_days",
    "phase2_nondemo_fem_duration_days",
    "phase1_nondemo_mal_duration_days",
    "phase2_nondemo_mal_duration_days",
]


@pytest.fixture(scope="module")
def d_gleam_mixed(d_gleam) -> dict:
    nondemo_cohorts = _mod("herd_all_input_chrt_data.csv")
    nondemo_cohorts = nondemo_cohorts[nondemo_cohorts.cohort_short.isin(["FN", "MN"])]
    herd_species_activity = d_gleam["cohort_no_structure"][
        ["herd_id", "species_short", "high_activity_fraction", "low_activity_fraction"]
    ].drop_duplicates()
    cohort_no_structure = rbind_fill(
        [
            d_gleam["cohort_no_structure"],
            merge_dt(nondemo_cohorts, herd_species_activity, by="herd_id", all_x=True),
        ]
    )
    herd_nondemo_full = _mod("herd_all_input_hrd_data.csv")[["herd_id"] + NONDEMO_HERD_COLS]
    herd = merge_dt(
        d_gleam["herd"].drop(columns=NONDEMO_HERD_COLS), herd_nondemo_full, by="herd_id", all_x=True
    )
    d = dict(d_gleam)
    d["cohort_no_structure"] = cohort_no_structure
    d["herd"] = herd
    return d


def run_gleam_default(data: dict, has_herd_structure=False, **overrides):
    use_structure = has_herd_structure is True
    cohort = data["cohort_structure"] if use_structure else data["cohort_no_structure"]
    args = dict(
        has_herd_structure=has_herd_structure,
        cohort_level_data=cohort,
        herd_level_data=data["herd"],
        feed_rations=data["feed_rations"],
        feed_params=data["feed_params"],
        feed_emissions=data["feed_emissions"],
        manure_management_system_fraction=data["mms_fraction"],
        manure_management_system_factors=data["mms_factors"],
        run_demographic=bool(cohort["cohort_short"].isin(DEMOGRAPHIC_COHORTS).any()),
        run_nondemographic=bool(cohort["cohort_short"].isin(["FN", "MN"]).any()),
        show_indicator=False,
    )
    args.update(overrides)
    return gleam.run_gleam(**args)


@pytest.fixture(scope="module")
def res_with_structure(d_gleam):
    return run_gleam_default(d_gleam, has_herd_structure=True)


# ---- validate_run_gleam_inputs: has_herd_structure ---------------------------


def test_rejects_non_logical_has_herd_structure(d_gleam):
    with pytest.raises(GleamValidationError, match="single logical value"):
        run_gleam_default(d_gleam, has_herd_structure="yes")


def test_rejects_na_has_herd_structure(d_gleam):
    with pytest.raises(GleamValidationError, match="not NA"):
        run_gleam_default(d_gleam, has_herd_structure=np.nan)


def test_existing_herd_structure_ignores_herd_simulation_defaults(d_gleam):
    gleam.run_gleam(
        has_herd_structure=True,
        cohort_level_data=d_gleam["cohort_structure"],
        herd_level_data=_run("master_hrd_lvl_mixed_data.csv"),
        feed_rations=d_gleam["feed_rations"],
        feed_params=d_gleam["feed_params"],
        feed_emissions=d_gleam["feed_emissions"],
        manure_management_system_fraction=d_gleam["mms_fraction"],
        manure_management_system_factors=d_gleam["mms_factors"],
        show_indicator=False,
    )


# ---- validate_run_gleam_inputs: simulation_duration --------------------------


def test_rejects_non_numeric_simulation_duration(d_gleam):
    with pytest.raises(GleamValidationError, match="simulation_duration.*numeric"):
        run_gleam_default(d_gleam, simulation_duration="365")


def test_rejects_non_positive_simulation_duration(d_gleam):
    with pytest.raises(GleamValidationError, match="simulation_duration.*positive"):
        run_gleam_default(d_gleam, simulation_duration=0)


def test_rejects_invalid_global_warming_potential_set(d_gleam):
    with pytest.raises(GleamValidationError, match="global_warming_potential_set"):
        run_gleam_default(d_gleam, global_warming_potential_set="AR3")


# ---- validate_run_gleam_inputs: data frame checks ----------------------------


@pytest.mark.parametrize(
    "arg",
    [
        "cohort_level_data",
        "herd_level_data",
        "feed_rations",
        "feed_params",
        "feed_emissions",
        "manure_management_system_fraction",
        "manure_management_system_factors",
    ],
)
def test_rejects_none_input_tables(d_gleam, arg):
    with pytest.raises(GleamValidationError, match=f"{arg}.*must be a data frame"):
        run_gleam_default(d_gleam, **{arg: None})


# ---- validate_run_gleam_inputs: calculated columns blocked -------------------


def test_rejects_cohort_data_containing_calculated_columns(d_gleam):
    bad = d_gleam["cohort_no_structure"].copy()
    bad["daily_weight_gain"] = 0.5
    with pytest.raises(GleamValidationError, match="daily_weight_gain"):
        run_gleam_default(d_gleam, cohort_level_data=bad)


def test_blocks_cohort_stock_size_in_no_structure_mode(d_gleam):
    bad = d_gleam["cohort_no_structure"].copy()
    bad["cohort_stock_size"] = 100
    with pytest.raises(GleamValidationError, match="cohort_stock_size"):
        run_gleam_default(d_gleam, cohort_level_data=bad)


def test_allows_cohort_stock_size_in_structure_mode(d_gleam):
    assert "cohort_stock_size" in d_gleam["cohort_structure"].columns


# ---- validate_run_gleam_inputs: herd_id consistency -------------------------


def test_rejects_mismatched_herd_id_across_inputs(d_gleam):
    bad = d_gleam["herd"].copy()
    bad["prop_nondemo_fem_juv"] = 0
    bad["prop_nondemo_mal_juv"] = 0
    bad["herd_id"] = bad["herd_id"].astype(str) + "_bad"
    with pytest.raises(GleamValidationError, match="same.*herd_id"):
        run_gleam_default(d_gleam, herd_level_data=bad)


def test_rejects_missing_fn_rows_when_prop_nondemo_fem_juv_is_positive(d_gleam_mixed):
    c = d_gleam_mixed["cohort_no_structure"]
    bad = c[~((c.herd_id == 9) & (c.cohort_short == "FN"))]
    with pytest.raises(GleamValidationError, match="Missing .*FN.*herd_id.*9|proportion_nondemographic"):
        run_gleam_default(d_gleam_mixed, cohort_level_data=bad)


def test_rejects_missing_mn_rows_when_prop_nondemo_mal_juv_is_positive(d_gleam_mixed):
    c = d_gleam_mixed["cohort_no_structure"]
    bad = c[~((c.herd_id == 1) & (c.cohort_short == "MN"))]
    with pytest.raises(GleamValidationError, match="Missing .*MN.*herd_id.*1|proportion_nondemographic"):
        run_gleam_default(d_gleam_mixed, cohort_level_data=bad)


# ---- run_gleam: return structure ---------------------------------------------


def test_run_gleam_returns_expected_elements(res_with_structure):
    assert list(res_with_structure) == [
        "cohort_level_results", "herd_level_results", "allocation_long", "aggregation_results",
    ]


def test_aggregation_results_has_expected_sub_elements(res_with_structure):
    assert list(res_with_structure["aggregation_results"]) == [
        "results_emissions", "results_feed", "results_production", "results_nitrogen",
    ]


def test_run_gleam_succeeds_with_herd_structure(res_with_structure):
    cohort = res_with_structure["cohort_level_results"]
    assert isinstance(cohort, pd.DataFrame)
    assert len(cohort) > 0
    assert {"herd_id", "cohort_short"} <= set(cohort.columns)


@pytest.mark.parametrize(
    "cols",
    [
        ["live_weight_mature_stage", "daily_weight_gain", "live_weight_cohort_average"],
        ["metabolic_energy_req_maintenance", "metabolic_energy_req_total", "ration_intake"],
        ["ration_gross_energy", "ration_digestibility_fraction", "ration_nitrogen"],
        ["ch4_conversion_factor_ym", "ch4_enteric"],
        ["nitrogen_intake", "nitrogen_retention", "nitrogen_excretion"],
        [
            "volatile_solids", "ch4_manure_pasture", "ch4_manure_burned", "ch4_manure_other",
            "n2o_manure_pasture_total", "n2o_manure_burned_total", "n2o_manure_other_total",
        ],
        ["co2_ration_fertilizer", "co2_ration_pesticides", "n2o_ration_fertilizer", "ch4_ration_rice"],
        ["milk_production_fpcm_cohort", "meat_production_live_weight_cohort", "meat_production_protein_cohort"],
        ["milk_allocation_energy", "meat_allocation_energy", "fibre_allocation_energy", "work_allocation_energy"],
    ],
    ids=["weights", "energy", "ration", "enteric", "nitrogen", "manure", "feed", "production", "allocation"],
)
def test_structure_path_produces_module_columns(res_with_structure, cols):
    missing = [c for c in cols if c not in res_with_structure["cohort_level_results"].columns]
    assert not missing


def _rows(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    return out.sort_values(cols, na_position="first", kind="mergesort").reset_index(drop=True)


def test_structure_path_preserves_cohort_set_per_herd(res_with_structure, d_gleam):
    cohort = res_with_structure["cohort_level_results"]
    inp = d_gleam["cohort_structure"]
    for hid in pd.unique(cohort.herd_id):
        a = _rows(cohort[cohort.herd_id == hid], ["cohort_short", "nondemo_productive_phase_id"])
        b = _rows(inp[inp.herd_id == hid], ["cohort_short", "nondemo_productive_phase_id"])
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_structure_path_preserves_cohort_stock_size(res_with_structure, d_gleam):
    cols = ["herd_id", "cohort_short", "nondemo_productive_phase_id", "cohort_stock_size"]
    a = _rows(res_with_structure["cohort_level_results"], cols)
    b = _rows(d_gleam["cohort_structure"], cols)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_key_numeric_outputs_have_no_na(res_with_structure):
    cohort = res_with_structure["cohort_level_results"]
    for col in ["metabolic_energy_req_total", "ration_intake", "ration_gross_energy"]:
        assert not cohort[col].isna().any(), col


def test_allocation_long_has_expected_columns(res_with_structure):
    alloc = res_with_structure["allocation_long"]
    assert isinstance(alloc, pd.DataFrame)
    expected = ["herd_id", "species_short", "variable_name", "commodity_name", "commodity_type", "allocation_share"]
    assert set(expected) <= set(alloc.columns)


def test_allocation_share_between_0_and_1(res_with_structure):
    share = as_float(res_with_structure["allocation_long"]["allocation_share"])
    assert ((share >= 0) & (share <= 1)).all()


@pytest.mark.parametrize("table", ["results_emissions", "results_production", "results_feed", "results_nitrogen"])
def test_aggregation_tables_non_empty(res_with_structure, table):
    df = res_with_structure["aggregation_results"][table]
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


# ---- Python-specific: inputs are not mutated, validation switch -------------


def test_run_gleam_does_not_mutate_inputs(d_gleam):
    before = {k: v.copy() for k, v in d_gleam.items()}
    run_gleam_default(d_gleam, has_herd_structure=True)
    for k, v in d_gleam.items():
        pd.testing.assert_frame_equal(v, before[k])


def test_validate_inputs_false_warns_and_matches(d_gleam, res_with_structure):
    with pytest.warns(gleam.GleamWarning, match="validation has been turned off"):
        res = run_gleam_default(d_gleam, has_herd_structure=True, validate_inputs=False)
    pd.testing.assert_frame_equal(res["cohort_level_results"], res_with_structure["cohort_level_results"])
