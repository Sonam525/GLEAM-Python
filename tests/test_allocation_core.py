"""Port of ``tests/testthat/test-allocation_core.R`` plus vectorisation checks."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from gleampy import (
    GleamValidationError,
    assign_allocation_shares,
    calc_allocation_shares,
    calc_cohort_to_herd_aggregation,
    calc_egg_allocation_energy,
    calc_fibre_allocation_energy,
    calc_meat_allocation_energy,
    calc_milk_allocation_energy,
    calc_work_allocation_energy,
    run_allocation_module,
)
from gleampy.io import load_example

REL = 1.5e-8  # testthat tolerance


def approx(x):
    return pytest.approx(x, rel=REL)


# ---- calc_milk_allocation_energy -------------------------------------------


def test_calc_milk_allocation_energy_returns_correct_value_for_valid_inputs():
    result = calc_milk_allocation_energy(
        milk_production_fpcm_cohort=100,
        milk_protein_fraction_standard=0.033,
        milk_fat_fraction_standard=0.04,
        milk_lactose_fraction_standard=0.048,
    )
    assert isinstance(result, float)
    energy_standard = (0.0929 * 0.04 + 0.0547 * 0.033 + 0.0395 * 0.048) * 4.184 * 100
    assert result == approx(energy_standard * 100)


def test_calc_milk_allocation_energy_handles_zero_milk_output():
    result = calc_milk_allocation_energy(
        milk_production_fpcm_cohort=0,
        milk_protein_fraction_standard=0.033,
        milk_fat_fraction_standard=0.04,
        milk_lactose_fraction_standard=0.048,
    )
    assert result == approx(0)


def test_calc_milk_allocation_energy_validates_bounds():
    with pytest.raises(GleamValidationError, match="is out of range"):
        calc_milk_allocation_energy(
            milk_production_fpcm_cohort=-10,
            milk_protein_fraction_standard=0.033,
            milk_fat_fraction_standard=0.04,
            milk_lactose_fraction_standard=0.048,
        )
    with pytest.raises(GleamValidationError, match="is out of range"):
        calc_milk_allocation_energy(
            milk_production_fpcm_cohort=100,
            milk_protein_fraction_standard=1.5,
            milk_fat_fraction_standard=0.04,
            milk_lactose_fraction_standard=0.048,
        )


# ---- calc_meat_allocation_energy -------------------------------------------


def test_calc_meat_allocation_energy_returns_correct_value_for_cattle_female():
    result = calc_meat_allocation_energy(
        species_short="CTL",
        cohort_short="FA",
        meat_production_live_weight_cohort=100,
        live_weight_cohort_at_slaughter=500,
        live_weight_at_birth=40,
    )
    assert isinstance(result, float)
    # For CTL/FA: growth_efficiency_factor = 0.8
    expected_specific = (22.02 * (((500 - 40) / 2) / (0.8 * 500)) ** 0.75 * (500 - 40) ** 1.097) / 500
    assert result == approx(expected_specific * 100)


def test_calc_meat_allocation_energy_returns_correct_value_for_cattle_male():
    result = calc_meat_allocation_energy(
        species_short="CTL",
        cohort_short="MA",
        meat_production_live_weight_cohort=150,
        live_weight_cohort_at_slaughter=600,
        live_weight_at_birth=45,
    )
    assert isinstance(result, float)
    assert not np.isnan(result)
    assert result >= 0


def test_calc_meat_allocation_energy_returns_correct_value_for_camelids():
    result = calc_meat_allocation_energy(
        species_short="CML",
        cohort_short="FA",
        meat_production_live_weight_cohort=80,
        live_weight_cohort_at_slaughter=450,
        live_weight_at_birth=35,
        ratio_me_to_ne=2.33,
    )
    assert isinstance(result, float)
    assert not np.isnan(result)
    expected_specific = (41.8 * (450 - 35) / 450) / 2.33
    assert result == approx(expected_specific * 80)


def test_calc_meat_allocation_energy_returns_correct_value_for_sheep_female():
    result = calc_meat_allocation_energy(
        species_short="SHP",
        cohort_short="FA",
        meat_production_live_weight_cohort=20,
        live_weight_cohort_at_slaughter=60,
        live_weight_at_birth=4,
    )
    assert isinstance(result, float)
    assert not np.isnan(result)
    expected_specific = ((60 - 4) * (2.1 + 0.5 * 0.45 * (4 + 60))) / 60
    assert result == approx(expected_specific * 20)


def test_calc_meat_allocation_energy_returns_correct_value_for_sheep_male():
    result = calc_meat_allocation_energy(
        species_short="SHP",
        cohort_short="MA",
        meat_production_live_weight_cohort=25,
        live_weight_cohort_at_slaughter=70,
        live_weight_at_birth=4.5,
    )
    assert isinstance(result, float)
    assert not np.isnan(result)
    expected_specific = ((70 - 4.5) * (4.4 + 0.5 * 0.32 * (4.5 + 70))) / 70
    assert result == approx(expected_specific * 25)


def test_calc_meat_allocation_energy_returns_correct_value_for_goats():
    result = calc_meat_allocation_energy(
        species_short="GTS",
        cohort_short="FA",
        meat_production_live_weight_cohort=15,
        live_weight_cohort_at_slaughter=50,
        live_weight_at_birth=3.5,
    )
    assert isinstance(result, float)
    assert not np.isnan(result)
    expected_specific = ((50 - 3.5) * (5 + 0.5 * 0.33 * (3.5 + 50))) / 50
    assert result == approx(expected_specific * 15)


def test_calc_meat_allocation_energy_returns_correct_value_for_chickens():
    result = calc_meat_allocation_energy(
        species_short="CHK",
        cohort_short="FJ",
        meat_production_live_weight_cohort=10,
        live_weight_cohort_at_slaughter=2,
        live_weight_at_birth=0.04,
    )
    expected_specific = (((0.0279 + 0.0202) / 2) * 1000 * (2 - 0.04)) / 2
    assert result == approx(expected_specific * 10)


def test_calc_meat_allocation_energy_uses_explicit_egg_producing_fn_flag():
    result = calc_meat_allocation_energy(
        species_short="CHK",
        cohort_short="FN",
        nondemo_productive_phase_id=2,
        meat_production_live_weight_cohort=10,
        live_weight_cohort_at_slaughter=2,
        live_weight_at_birth=0.04,
        is_egg_producing=True,
    )
    expected_specific = (((0.0279 + 0.0202) / 2) * 1000 * (2 - 0.04)) / 2
    assert result == approx(expected_specific * 10)


def test_calc_meat_allocation_energy_validates_animal_species():
    with pytest.raises(GleamValidationError, match="must be one of"):
        calc_meat_allocation_energy(
            species_short="INVALID", cohort_short="FA", meat_production_live_weight_cohort=100
        )


def test_calc_meat_allocation_energy_validates_cohort_codes():
    with pytest.raises(GleamValidationError, match="must be one of"):
        calc_meat_allocation_energy(
            species_short="CTL",
            cohort_short="INVALID",
            meat_production_live_weight_cohort=100,
            live_weight_cohort_at_slaughter=500,
            live_weight_at_birth=40,
        )


def test_calc_meat_allocation_energy_validates_weight_bounds_for_non_pgs():
    with pytest.raises(GleamValidationError, match="is out of range"):
        calc_meat_allocation_energy(
            species_short="CTL",
            cohort_short="FA",
            meat_production_live_weight_cohort=100,
            live_weight_cohort_at_slaughter=-10,
            live_weight_at_birth=40,
        )
    with pytest.raises(GleamValidationError, match="is out of range"):
        calc_meat_allocation_energy(
            species_short="CTL",
            cohort_short="FA",
            meat_production_live_weight_cohort=100,
            live_weight_cohort_at_slaughter=500,
            live_weight_at_birth=1500,
        )


def test_calc_meat_allocation_energy_validates_ratio_me_to_ne_for_cml_only():
    with pytest.raises(GleamValidationError, match="must be a positive numeric value"):
        calc_meat_allocation_energy(
            species_short="CML",
            cohort_short="FA",
            meat_production_live_weight_cohort=80,
            live_weight_cohort_at_slaughter=450,
            live_weight_at_birth=35,
            ratio_me_to_ne=-1,
        )
    # Non-CML: ratio_me_to_ne is not validated even if negative
    calc_meat_allocation_energy(
        species_short="CTL",
        cohort_short="FA",
        meat_production_live_weight_cohort=100,
        live_weight_cohort_at_slaughter=500,
        live_weight_at_birth=40,
        ratio_me_to_ne=-1,
    )


# ---- calc_fibre_allocation_energy ------------------------------------------


def test_calc_fibre_allocation_energy_returns_correct_value_for_sheep():
    result = calc_fibre_allocation_energy(
        species_short="SHP", cohort_stock_size=100, metabolic_energy_req_fibre_production=5,
        simulation_duration=365,
    )
    assert isinstance(result, float)
    assert result == approx(5 * 365 * 100)


def test_calc_fibre_allocation_energy_returns_correct_value_for_goats():
    result = calc_fibre_allocation_energy(
        species_short="GTS", cohort_stock_size=100, metabolic_energy_req_fibre_production=4,
        simulation_duration=365,
    )
    assert result == approx(4 * 365 * 100)


def test_calc_fibre_allocation_energy_returns_correct_value_for_camelids():
    result = calc_fibre_allocation_energy(
        species_short="CML", cohort_stock_size=100, metabolic_energy_req_fibre_production=6,
        ratio_me_to_ne=2.33, simulation_duration=365,
    )
    assert isinstance(result, float)
    assert result == approx((6 / 2.33) * 365 * 100)


def test_calc_fibre_allocation_energy_returns_zero_for_non_fibre_species():
    assert calc_fibre_allocation_energy(species_short="CTL") == approx(0)
    assert calc_fibre_allocation_energy(species_short="BFL") == approx(0)
    assert calc_fibre_allocation_energy(species_short="PGS") == approx(0)


def test_calc_fibre_allocation_energy_validates_inputs():
    with pytest.raises(GleamValidationError, match="must be one of"):
        calc_fibre_allocation_energy(species_short="INVALID")
    with pytest.raises(GleamValidationError, match="is out of range"):
        calc_fibre_allocation_energy(
            species_short="SHP", cohort_stock_size=100, metabolic_energy_req_fibre_production=-5,
            simulation_duration=365,
        )
    # ratio_me_to_ne only validated for CML, not SHP
    calc_fibre_allocation_energy(
        species_short="SHP", cohort_stock_size=100, metabolic_energy_req_fibre_production=5,
        ratio_me_to_ne=-0.1, simulation_duration=365,
    )
    with pytest.raises(GleamValidationError, match="must be a positive numeric value"):
        calc_fibre_allocation_energy(
            species_short="CML", cohort_stock_size=100, metabolic_energy_req_fibre_production=5,
            ratio_me_to_ne=-0.1, simulation_duration=365,
        )


# ---- calc_work_allocation_energy -------------------------------------------


def test_calc_work_allocation_energy_returns_correct_value_for_camelids():
    result = calc_work_allocation_energy(
        species_short="CML", cohort_stock_size=100, metabolic_energy_req_work=10,
        simulation_duration=365, ratio_me_to_ne=2.33,
    )
    assert isinstance(result, float)
    assert result == approx((10 * 365 * 100) / 2.33)


def test_calc_work_allocation_energy_returns_correct_value_for_non_camelids():
    result = calc_work_allocation_energy(
        species_short="CTL", cohort_stock_size=100, metabolic_energy_req_work=8, simulation_duration=365,
    )
    assert isinstance(result, float)
    assert result == approx(8 * 365 * 100)


def test_calc_work_allocation_energy_handles_zero_energy_requirement():
    result = calc_work_allocation_energy(
        species_short="CTL", cohort_stock_size=100, metabolic_energy_req_work=0, simulation_duration=365,
    )
    assert result == approx(0)


def test_calc_work_allocation_energy_validates_inputs():
    with pytest.raises(GleamValidationError, match="must be one of"):
        calc_work_allocation_energy(
            species_short="INVALID", cohort_stock_size=100, metabolic_energy_req_work=10,
            simulation_duration=365,
        )
    with pytest.raises(GleamValidationError, match="is out of range"):
        calc_work_allocation_energy(
            species_short="CTL", cohort_stock_size=100, metabolic_energy_req_work=-5,
            simulation_duration=365,
        )
    with pytest.raises(GleamValidationError, match="is out of range"):
        calc_work_allocation_energy(
            species_short="CTL", cohort_stock_size=100, metabolic_energy_req_work=10,
            simulation_duration=5000,
        )
    # ratio_me_to_ne only validated for CML
    with pytest.raises(GleamValidationError, match="must be a positive numeric value"):
        calc_work_allocation_energy(
            species_short="CML", cohort_stock_size=100, metabolic_energy_req_work=10,
            simulation_duration=365, ratio_me_to_ne=-0.1,
        )
    calc_work_allocation_energy(
        species_short="CTL", cohort_stock_size=100, metabolic_energy_req_work=10,
        simulation_duration=365, ratio_me_to_ne=-0.1,
    )


# ---- calc_egg_allocation_energy --------------------------------------------


def test_calc_egg_allocation_energy_returns_expected_value_for_chickens():
    result = calc_egg_allocation_energy(
        species_short="CHK", cohort_short="FA", egg_production_mass_cohort=100, is_egg_producing=True
    )
    assert result == approx(100 * 10.04)
    assert calc_egg_allocation_energy(
        species_short="CHK", cohort_short="FS", egg_production_mass_cohort=100
    ) == approx(0)


def test_calc_egg_allocation_energy_returns_expected_value_for_egg_producing_chicken_fn():
    result = calc_egg_allocation_energy(
        species_short="CHK", cohort_short="FN", nondemo_productive_phase_id=2,
        egg_production_mass_cohort=100, is_egg_producing=True,
    )
    assert result == approx(100 * 10.04)


def test_calc_egg_allocation_energy_validates_egg_producing_flag_placement():
    with pytest.raises(GleamValidationError, match="can be TRUE only for CHK cohorts.*FA.*FN"):
        calc_egg_allocation_energy(
            species_short="CHK", cohort_short="FS", egg_production_mass_cohort=100, is_egg_producing=True
        )
    with pytest.raises(GleamValidationError, match="can be TRUE for.*FN.*only when.*2"):
        calc_egg_allocation_energy(
            species_short="CHK", cohort_short="FN", nondemo_productive_phase_id=1,
            egg_production_mass_cohort=100, is_egg_producing=True,
        )


# ---- run_allocation_module -------------------------------------------------


def _allocation_inputs():
    return load_example("allocation_input_chrt_data.csv"), load_example("allocation_input_hrd_data.csv")


def test_run_allocation_module_example_inputs_work_with_explicit_egg_columns():
    cohort_level_data, herd_level_data = _allocation_inputs()
    result = run_allocation_module(
        cohort_level_data=cohort_level_data,
        herd_level_data=herd_level_data,
        simulation_duration=365,
        show_indicator=False,
    )
    assert "egg_allocation_energy" in result["cohort_allocation_inputs"].columns
    assert "allocation_long" in result


def test_run_allocation_module_requires_explicit_egg_allocation_columns():
    cohort_level_data, herd_level_data = _allocation_inputs()

    with pytest.raises(GleamValidationError, match="egg_production_mass_cohort"):
        run_allocation_module(
            cohort_level_data=cohort_level_data.drop(columns="egg_production_mass_cohort"),
            herd_level_data=herd_level_data,
            simulation_duration=365,
            show_indicator=False,
        )

    non_chk_cohort = cohort_level_data[cohort_level_data["species_short"] != "CHK"].drop(columns="is_egg_producing")
    non_chk_herd = herd_level_data[herd_level_data["species_short"] != "CHK"]
    run_allocation_module(
        cohort_level_data=non_chk_cohort,
        herd_level_data=non_chk_herd,
        simulation_duration=365,
        show_indicator=False,
    )

    with pytest.raises(GleamValidationError, match="is_egg_producing"):
        run_allocation_module(
            cohort_level_data=cohort_level_data.drop(columns="is_egg_producing"),
            herd_level_data=herd_level_data,
            simulation_duration=365,
            show_indicator=False,
        )


def test_run_allocation_module_does_not_mutate_inputs():
    cohort_level_data, herd_level_data = _allocation_inputs()
    non_chk_cohort = cohort_level_data[cohort_level_data["species_short"] != "CHK"].drop(columns="is_egg_producing")
    non_chk_herd = herd_level_data[herd_level_data["species_short"] != "CHK"]
    before = non_chk_cohort.copy()
    res = run_allocation_module(non_chk_cohort, non_chk_herd, show_indicator=False)
    pd.testing.assert_frame_equal(non_chk_cohort, before)
    # R adds the optional flag as NA (after the input columns) when absent and no CHK herd
    cols = list(res["cohort_allocation_inputs"].columns)
    assert cols[: len(before.columns) + 1] == list(before.columns) + ["is_egg_producing"]
    assert cols[-5:] == [
        "milk_allocation_energy", "meat_allocation_energy", "fibre_allocation_energy",
        "work_allocation_energy", "egg_allocation_energy",
    ]


def test_run_allocation_module_without_validation():
    cohort_level_data, herd_level_data = _allocation_inputs()
    with pytest.warns(UserWarning, match="validation has been turned off"):
        res = run_allocation_module(cohort_level_data, herd_level_data, show_indicator=False, validate_inputs=False)
    ref = run_allocation_module(cohort_level_data, herd_level_data, show_indicator=False)
    pd.testing.assert_frame_equal(res["allocation_long"], ref["allocation_long"])


# ---- calc_allocation_shares ------------------------------------------------


def test_calc_allocation_shares_returns_meat_1_and_others_0_for_pigs():
    result = calc_allocation_shares(
        species_short="PGS",
        meat_allocation_energy=np.nan,
        milk_allocation_energy=0,
        fibre_allocation_energy=0,
        work_allocation_energy=0,
        egg_allocation_energy=0,
    )
    assert result["meat_share_allocation"] == approx(1)
    assert result["milk_share_allocation"] == approx(0)
    assert result["fibre_share_allocation"] == approx(0)
    assert result["work_share_allocation"] == approx(0)
    assert result["eggs_share_allocation"] == approx(0)


def test_calc_allocation_shares_returns_correct_proportions_for_milk_and_meat():
    result = calc_allocation_shares(
        species_short="CTL",
        meat_allocation_energy=300,
        milk_allocation_energy=700,
        fibre_allocation_energy=0,
        work_allocation_energy=0,
        egg_allocation_energy=0,
    )
    assert result["meat_share_allocation"] == approx(0.3)
    assert result["milk_share_allocation"] == approx(0.7)
    assert result["fibre_share_allocation"] == approx(0)
    assert result["work_share_allocation"] == approx(0)
    assert result["eggs_share_allocation"] == approx(0)


def test_calc_allocation_shares_shares_sum_to_1():
    result = calc_allocation_shares(
        species_short="SHP",
        meat_allocation_energy=200,
        milk_allocation_energy=0,
        fibre_allocation_energy=300,
        work_allocation_energy=0,
        egg_allocation_energy=0,
    )
    total = sum(result.values())
    assert total == approx(1)
    assert result["meat_share_allocation"] == approx(0.4)
    assert result["fibre_share_allocation"] == approx(0.6)


def test_calc_allocation_shares_returns_a_named_list_with_5_elements():
    result = calc_allocation_shares(
        species_short="CTL",
        meat_allocation_energy=500,
        milk_allocation_energy=500,
        fibre_allocation_energy=0,
        work_allocation_energy=0,
        egg_allocation_energy=0,
    )
    assert isinstance(result, dict)
    assert list(result) == [
        "meat_share_allocation", "milk_share_allocation",
        "fibre_share_allocation", "work_share_allocation", "eggs_share_allocation",
    ]


def test_calc_allocation_shares_includes_eggs_for_chickens():
    result = calc_allocation_shares(
        species_short="CHK",
        meat_allocation_energy=20,
        milk_allocation_energy=0,
        fibre_allocation_energy=0,
        work_allocation_energy=0,
        egg_allocation_energy=80,
    )
    assert result["meat_share_allocation"] == approx(0.2)
    assert result["eggs_share_allocation"] == approx(0.8)


# ---- calc_cohort_to_herd_aggregation / assign_allocation_shares -------------


def test_calc_cohort_to_herd_aggregation_keeps_first_appearance_order_and_na_sums():
    data = pd.DataFrame(
        {
            "herd_id": [2, 1, 2, 1, 3],
            "cohort_short": ["FA", "FA", "MA", "MA", "FA"],
            "a": [1.0, 2.0, 3.0, np.nan, 5.0],
            "b": [1, 2, 3, 4, 5],
        }
    )
    out = calc_cohort_to_herd_aggregation(data, "herd_id", ["a", "b"], "cohort_short")
    assert list(out.columns) == ["herd_id", "a", "b"]
    assert out["herd_id"].tolist() == [2, 1, 3]
    np.testing.assert_array_equal(out["a"].to_numpy(), [4.0, np.nan, 5.0])
    assert out["b"].tolist() == [4, 6, 5]


def test_assign_allocation_shares_applies_rules_and_sorts_by_commodity():
    long = pd.DataFrame(
        {
            "herd_id": [1, 1, 1],
            "commodity_name": ["Meat", "Milk", "Other"],
            "allocation_share": [0.4, 0.6, 0.0],
        }
    )
    out = assign_allocation_shares(
        long,
        emissions_vars=["ch4_manure_pasture", "ch4_enteric"],
        commodities=["Other", "Milk", "Meat"],
        non_allocated_emission_sources=["ch4_manure_pasture"],
        commodity_col="commodity_name",
        allocation_col="allocation_share",
    )
    assert list(out.columns) == ["commodity_name", "herd_id", "allocation_share", "variable_name"]
    assert out["commodity_name"].tolist() == ["Meat", "Meat", "Milk", "Milk", "Other", "Other"]
    assert out["variable_name"].tolist() == ["ch4_enteric", "ch4_manure_pasture"] * 3
    assert out["allocation_share"].tolist() == [0.4, 0.0, 0.6, 0.0, 0.0, 1.0]


# ---- vectorisation: arrays give the same result as element-wise scalar calls --


SPECIES = ["CTL", "BFL", "SHP", "GTS", "PGS", "CML", "CHK", "SHP", "CTL", "CHK"]
COHORTS = ["FA", "MS", "FJ", "MA", "FN", "FS", "FA", "MN", "FN", "FN"]
PHASE = [np.nan, np.nan, np.nan, np.nan, 1, np.nan, np.nan, 2, 2, 2]
EGG = [False, False, False, False, False, False, True, False, False, True]
SLAUGHTER = [500, 620, 45, 38, 110, 400, 1.9, 52, 480, 1.8]
BIRTH = [40, 38, 4.2, 3.3, 1.2, 30, 0.04, 4.0, 41, 0.04]
MEAT = [100, 50.5, 20, 15, 1000, 80, 10, 0, 33, 7]
RATIO = [0.43, 0.43, 0.43, 0.43, 0.43, 2.33, 0.43, 0.43, 0.43, 0.43]
STOCK = [100, 5, 250.5, 30, 1000, 12, 5000, 40, 7, 900]
FIBRE = [0, 0, 0.8, 0.5, 0, 1.2, 0, 0.3, 0, 0]
WORK = [2.5, 0, 0, 0, 0, 4, 0, 0, 1.5, 0]
MILK = [4600, 0, 30, 25, 0, 900, 0, 0, 0, 0]
EGG_MASS = [0, 0, 0, 0, 0, 0, 1080, 0, 0, 55]


def _check_vectorised(fn, columns: dict, scalar_kwargs: dict | None = None):
    n = len(next(iter(columns.values())))
    scalar_kwargs = scalar_kwargs or {}
    vec = fn(**{k: np.asarray(v, dtype=object if isinstance(v[0], (str, bool)) else float) for k, v in columns.items()}, **scalar_kwargs)
    ser = fn(**{k: pd.Series(v) for k, v in columns.items()}, **scalar_kwargs)
    rows = [fn(**{k: v[i] for k, v in columns.items()}, **scalar_kwargs) for i in range(n)]
    if isinstance(vec, dict):
        for key in vec:
            expected = np.array([r[key] for r in rows], dtype=float)
            np.testing.assert_array_equal(vec[key], expected)
            np.testing.assert_array_equal(ser[key], expected)
    else:
        expected = np.array(rows, dtype=float)
        assert isinstance(vec, np.ndarray) and vec.shape == (n,)
        np.testing.assert_array_equal(vec, expected)
        np.testing.assert_array_equal(ser, expected)
    return rows


def test_vectorised_milk_allocation_energy_matches_scalar_calls():
    _check_vectorised(
        calc_milk_allocation_energy,
        {
            "milk_production_fpcm_cohort": MILK,
            "milk_protein_fraction_standard": [0.033] * 9 + [0.035],
            "milk_fat_fraction_standard": [0.04] * 10,
            "milk_lactose_fraction_standard": [0.048] * 10,
        },
    )


def test_vectorised_meat_allocation_energy_matches_scalar_calls():
    rows = _check_vectorised(
        calc_meat_allocation_energy,
        {
            "species_short": SPECIES,
            "cohort_short": COHORTS,
            "meat_production_live_weight_cohort": MEAT,
            "live_weight_cohort_at_slaughter": SLAUGHTER,
            "live_weight_at_birth": BIRTH,
            "ratio_me_to_ne": RATIO,
            "nondemo_productive_phase_id": PHASE,
            "is_egg_producing": EGG,
        },
    )
    assert rows[4] == 0  # PGS
    assert all(np.isfinite(rows))


def test_vectorised_meat_allocation_energy_pgs_without_weights():
    out = calc_meat_allocation_energy(
        species_short=np.array(["PGS", "CTL"], dtype=object),
        cohort_short=np.array(["FA", "MA"], dtype=object),
        meat_production_live_weight_cohort=[10.0, 20.0],
        live_weight_cohort_at_slaughter=[np.nan, 500],
        live_weight_at_birth=[np.nan, 40],
    )
    assert out[0] == 0
    assert out[1] == calc_meat_allocation_energy("CTL", "MA", 20.0, 500, 40)
    # a missing weight is still rejected for non-PGS rows
    with pytest.raises(GleamValidationError, match="live_weight_cohort_at_slaughter"):
        calc_meat_allocation_energy(
            species_short=["PGS", "CTL"], cohort_short=["FA", "MA"],
            meat_production_live_weight_cohort=[10.0, 20.0],
            live_weight_cohort_at_slaughter=[500, np.nan], live_weight_at_birth=[40, 40],
        )


def test_vectorised_fibre_allocation_energy_matches_scalar_calls():
    rows = _check_vectorised(
        calc_fibre_allocation_energy,
        {
            "species_short": SPECIES,
            "cohort_stock_size": STOCK,
            "metabolic_energy_req_fibre_production": FIBRE,
            "ratio_me_to_ne": RATIO,
        },
        {"simulation_duration": 365},
    )
    assert rows[0] == 0 and rows[2] > 0 and rows[5] > 0


def test_vectorised_work_allocation_energy_matches_scalar_calls():
    _check_vectorised(
        calc_work_allocation_energy,
        {
            "species_short": SPECIES,
            "cohort_stock_size": STOCK,
            "metabolic_energy_req_work": WORK,
            "ratio_me_to_ne": RATIO,
            "simulation_duration": [365] * 9 + [180],
        },
    )


def test_vectorised_egg_allocation_energy_matches_scalar_calls():
    rows = _check_vectorised(
        calc_egg_allocation_energy,
        {
            "species_short": SPECIES,
            "cohort_short": COHORTS,
            "egg_production_mass_cohort": EGG_MASS,
            "nondemo_productive_phase_id": PHASE,
            "is_egg_producing": EGG,
        },
    )
    assert rows[6] == pytest.approx(1080 * 10.04) and rows[9] == pytest.approx(55 * 10.04)


def test_vectorised_allocation_shares_matches_scalar_calls():
    _check_vectorised(
        calc_allocation_shares,
        {
            "species_short": SPECIES,
            "meat_allocation_energy": [300, np.nan, 200, 0, 5, 50, 20, 0, 1, 2],
            "milk_allocation_energy": [700, 10, 0, 0, 5, 50, 0, 0, 1, 0],
            "fibre_allocation_energy": [0, 0, 300, 0, 0, 10, 0, 0, np.nan, 0],
            "work_allocation_energy": [0, 0, 0, 0, 0, 5, 0, 0, 1, 0],
            "egg_allocation_energy": [0, 0, 0, 0, 0, 0, 80, 0, 0, 8],
        },
    )


# ---- assign_allocation_shares: rules evaluated per grid row (088) --------------


def _assign_reference(allocation_herd_long, emissions_vars, commodities, non_allocated, commodity_col, allocation_col):
    """Row-by-row reference: R's three ``:=`` rules applied to every expanded row."""
    from gleampy._utils import as_float, as_str, isin, merge_dt

    vars_ = sorted({v for v in as_str(list(emissions_vars)).tolist()}, key=lambda v: (v is not None, v or ""))
    comms = sorted({v for v in as_str(list(commodities)).tolist()}, key=lambda v: (v is not None, v or ""))
    grid = pd.DataFrame(
        {
            "variable_name": np.repeat(np.array(vars_, dtype=object), len(comms)),
            "commodity_name": np.tile(np.array(comms, dtype=object), len(vars_)),
        }
    )
    out = merge_dt(allocation_herd_long, grid, by=commodity_col)
    non_alloc = isin(as_str(out["variable_name"]), list(non_allocated))
    comm = as_str(out[commodity_col])
    is_other = np.array([v is not None and v == "Other" for v in comm], dtype=bool)
    not_other = np.array([v is not None and v != "Other" for v in comm], dtype=bool)
    share = as_float(out[allocation_col]).copy()
    share[non_alloc & is_other] = 1.0
    share[non_alloc & not_other] = 0.0
    share[~non_alloc & is_other] = 0.0
    out[allocation_col] = share
    return out


@pytest.mark.parametrize("commodity_dtype", [object, "string", "category"])
def test_assign_allocation_shares_matches_rowwise_rules(commodity_dtype):
    long = pd.DataFrame(
        {
            "herd_id": [2, 2, 1, 1, 1, 3, 3],
            "species_short": ["SHP", "SHP", "CTL", "CTL", "CTL", "PGS", None],
            "commodity_name": pd.Series(["Fibre", "Other", "Meat", None, "Other", "Meat", "Eggs"], dtype=commodity_dtype),
            "allocation_share": [0.2, 0.0, 0.7, 0.3, np.nan, 1.0, 0.5],
        }
    )
    args = dict(
        emissions_vars=["ch4_manure_pasture", "ch4_enteric", "n2o_manure_burned_total", "ch4_enteric", None],
        commodities=["Other", "Milk", "Meat", "Fibre", None, "Eggs"],
        non_allocated_emission_sources=["ch4_manure_pasture", "n2o_manure_burned_total"],
        commodity_col="commodity_name",
        allocation_col="allocation_share",
    )
    got = assign_allocation_shares(long, **args)
    ref = _assign_reference(
        long, args["emissions_vars"], args["commodities"], args["non_allocated_emission_sources"],
        args["commodity_col"], args["allocation_col"],
    )
    pd.testing.assert_frame_equal(got, ref, check_exact=True)
    assert "__gleam_grid_row__" not in got.columns
    # a missing commodity matches the NA grid commodity and is left unassigned
    na_rows = got[got["commodity_name"].isna()]
    assert na_rows["allocation_share"].tolist() == [0.3] * len(na_rows) and len(na_rows) == 4


def test_assign_allocation_shares_with_variable_name_column_fails_like_r():
    long = pd.DataFrame({"commodity_name": ["Meat"], "allocation_share": [1.0], "variable_name": ["x"]})
    with pytest.raises(KeyError, match="variable_name"):
        assign_allocation_shares(long, ["ch4_enteric"], ["Meat"], [], "commodity_name", "allocation_share")


def test_calc_cohort_to_herd_aggregation_nullable_and_categorical_keys():
    data = pd.DataFrame(
        {
            "herd_id": pd.Categorical(["b", "a", "b", None]),
            "x": pd.array([1, 2, None, 4], dtype="Int64"),
            "y": [1.5, 2.5, 3.5, 4.5],
            # Explicit int64: a bare np.array([1, 2, 3, 4]) is int32 on Windows under numpy 1.x.
            "z": np.array([1, 2, 3, 4], dtype=np.int64),
        }
    )
    out = calc_cohort_to_herd_aggregation(data, "herd_id", ["x", "y", "z"], "cohort_short")
    assert out["herd_id"].astype(object).tolist()[:2] == ["b", "a"] and pd.isna(out["herd_id"].iloc[2])
    assert str(out["x"].dtype) == "Int64"
    assert out["x"].isna().tolist() == [True, False, False] and out["x"].iloc[1] == 2
    assert out["y"].tolist() == [5.0, 2.5, 4.5]
    assert out["z"].dtype == np.int64 and out["z"].tolist() == [4, 2, 4]


# ---- egg flag: logical only (047, 094) ----------------------------------------


@pytest.mark.parametrize("flag", ["TRUE", "FALSE", "T", 1, 0, 1.0])
def test_non_logical_egg_flag_is_rejected_like_r(flag):
    """R's validate_is_egg_producing_flag rejects character and numeric flags."""
    with pytest.raises(GleamValidationError, match=r"`is_egg_producing` must be logical \(TRUE/FALSE\)"):
        calc_egg_allocation_energy("CHK", "FA", 10, is_egg_producing=flag)
    with pytest.raises(GleamValidationError, match=r"`is_egg_producing` must be logical \(TRUE/FALSE\)"):
        calc_meat_allocation_energy(
            "CHK", "FA", 10, live_weight_cohort_at_slaughter=1.9, live_weight_at_birth=0.04, is_egg_producing=flag
        )


def test_check_logical_flag_is_gone():
    import gleampy.validation.allocation_core as ac

    assert not hasattr(ac, "check_logical_flag")


@pytest.mark.parametrize(
    "flag,expected",
    [(True, 100.4), (np.True_, 100.4), (1, 0.0), (1.0, 0.0), ("TRUE", 0.0), (np.int64(1), 0.0), (None, 0.0)],
)
def test_egg_allocation_energy_uses_istrue_without_validation(flag, expected):
    """094: with validation off only a logical TRUE produces egg energy (R's isTRUE)."""
    from gleampy.validation._shared import validation_disabled

    with validation_disabled():
        assert calc_egg_allocation_energy("CHK", "FA", 10, is_egg_producing=flag) == approx(expected)
        vec = calc_egg_allocation_energy(["CHK", "CHK"], ["FA", "FA"], [10, 10],
                                         is_egg_producing=np.array([flag, False], dtype=object))
    np.testing.assert_allclose(vec, [expected, 0.0], rtol=1e-15)


# ---- run_allocation_module: simulation_duration and columns -------------------


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("bad", ["two", "per-row", "series", "none"])
def test_run_allocation_module_rejects_non_single_simulation_duration(bad, validate):
    """R cannot run with a vector simulation_duration (its by-row assignments fail)."""
    chrt, hrd = _allocation_inputs()
    value = {
        "two": [365, 365],
        "per-row": np.full(len(chrt), 365.0),
        "series": pd.Series(np.full(len(chrt), 365.0)),
        "none": None,
    }[bad]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(GleamValidationError, match=r"^`simulation_duration` must be a single numeric value\.$"):
            run_allocation_module(chrt, hrd, simulation_duration=value, show_indicator=False, validate_inputs=validate)


def test_run_allocation_module_accepts_length_one_simulation_duration():
    chrt, hrd = _allocation_inputs()
    ref = run_allocation_module(chrt, hrd, simulation_duration=180, show_indicator=False)
    for value in ([180], np.array([180.0]), np.int64(180)):
        res = run_allocation_module(chrt, hrd, simulation_duration=value, show_indicator=False)
        pd.testing.assert_frame_equal(res["allocation_long"], ref["allocation_long"])
        pd.testing.assert_frame_equal(res["cohort_allocation_inputs"], ref["cohort_allocation_inputs"])


def test_run_allocation_module_nondemo_phase_id_is_always_required():
    """046: only is_egg_producing is conditional on CHK herds."""
    chrt, hrd = _allocation_inputs()
    non_chk = chrt[chrt["species_short"] != "CHK"].drop(columns="is_egg_producing")
    non_chk_herd = hrd[hrd["species_short"] != "CHK"]
    with pytest.raises(GleamValidationError, match=r'Missing required columns in `cohort_level_data`: "nondemo_productive_phase_id"'):
        run_allocation_module(non_chk.drop(columns="nondemo_productive_phase_id"), non_chk_herd, show_indicator=False)
    doc = " ".join(run_allocation_module.__doc__.split())
    assert "``nondemo_productive_phase_id`` (required; NA for demographic cohorts)" in doc
    assert "``is_egg_producing`` (required when ``CHK`` herds are present; added as NA otherwise)" in doc


def test_run_allocation_module_unvalidated_reads_missing_phase_id_as_na():
    """R only validates nondemo_productive_phase_id, so an unvalidated run works without it."""
    chrt, hrd = _allocation_inputs()
    no_phase = chrt.drop(columns="nondemo_productive_phase_id")
    with pytest.warns(UserWarning, match="validation has been turned off"):
        res = run_allocation_module(no_phase, hrd, show_indicator=False, validate_inputs=False)
    ref = run_allocation_module(chrt, hrd, show_indicator=False)
    pd.testing.assert_frame_equal(res["allocation_long"], ref["allocation_long"])


def test_run_allocation_module_unvalidated_without_egg_flag_for_chk_is_a_validation_error():
    """R's egg energy reads is_egg_producing on every row ("object not found")."""
    chrt, hrd = _allocation_inputs()
    with pytest.warns(UserWarning, match="validation has been turned off"):
        with pytest.raises(GleamValidationError, match=r'Missing required columns in `cohort_level_data`: "is_egg_producing"'):
            run_allocation_module(chrt.drop(columns="is_egg_producing"), hrd, show_indicator=False, validate_inputs=False)
