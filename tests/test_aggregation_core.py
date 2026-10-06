"""Port of ``tests/testthat/test-aggregation_core.R`` plus vectorisation checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gleam import (
    GleamValidationError,
    calc_allocated_emissions,
    calc_co2eq,
    calc_cohort_totals,
    run_aggregation_module,
)
from gleam.constants import GLEAM_FEED_EMISSIONS_META
from gleam.io import load_example

REL = 1.5e-8  # testthat tolerance

# Feed emissions use ration_intake in scaling; pass empty list when not testing feed emissions
feed_emissions_empty: list = []


def approx(x):
    return pytest.approx(x, rel=REL)


# ---- calc_cohort_totals ----------------------------------------------------


def test_calc_cohort_totals_returns_correct_value_for_production_variables():
    result = calc_cohort_totals(
        value=1000,
        cohort_stock_size=50,
        ration_intake=10,
        feed_emissions_list=feed_emissions_empty,
        simulation_duration=365,
        variable_name="milk_production_mass_cohort",
        variable_type="Production",
    )
    assert isinstance(result, float)
    # Production variables are returned as-is (not scaled)
    assert result == approx(1000)


def test_calc_cohort_totals_returns_correct_value_for_emissions_variables():
    result = calc_cohort_totals(
        value=0.5,
        cohort_stock_size=100,
        ration_intake=5,
        feed_emissions_list=feed_emissions_empty,
        simulation_duration=365,
        variable_name="ch4_enteric",
        variable_type="Emissions",
    )
    assert isinstance(result, float)
    assert result == approx(0.5 * 100 * 365)


def test_calc_cohort_totals_returns_correct_value_for_feed_emissions():
    feed_emissions = [{"emissions_source": "co2_ration_fertilizer", "label": "Feed-Fertilizer_CO2"}]
    result = calc_cohort_totals(
        value=0.1,
        cohort_stock_size=100,
        ration_intake=10,
        feed_emissions_list=feed_emissions,
        simulation_duration=365,
        variable_name="co2_ration_fertilizer",
        variable_type="Emissions",
    )
    assert result == approx(0.1 * 10 * 100 * 365 / 1000)


def test_calc_cohort_totals_returns_correct_value_for_feed_variables():
    result = calc_cohort_totals(
        value=10,
        cohort_stock_size=30,
        ration_intake=10,
        feed_emissions_list=feed_emissions_empty,
        simulation_duration=365,
        variable_name="ration_intake",
        variable_type="Feed",
    )
    assert result == approx(10 * 30 * 365)


def test_calc_cohort_totals_returns_correct_value_for_nitrogen_balance_variables():
    result = calc_cohort_totals(
        value=0.2,
        cohort_stock_size=25,
        ration_intake=8,
        feed_emissions_list=feed_emissions_empty,
        simulation_duration=365,
        variable_name="nitrogen_intake",
        variable_type="NitrogenBalance",
    )
    assert result == approx(0.2 * 25 * 365)


def test_calc_cohort_totals_validates_variable_type():
    with pytest.raises(GleamValidationError, match="must be one of"):
        calc_cohort_totals(
            value=100,
            cohort_stock_size=50,
            ration_intake=10,
            feed_emissions_list=feed_emissions_empty,
            simulation_duration=365,
            variable_name="x",
            variable_type="Invalid",
        )


def test_calc_cohort_totals_validates_bounds():
    with pytest.raises(GleamValidationError, match="must be positive"):
        calc_cohort_totals(
            value=100,
            cohort_stock_size=0,
            ration_intake=10,
            feed_emissions_list=feed_emissions_empty,
            simulation_duration=365,
            variable_name="ch4_enteric",
            variable_type="Emissions",
        )
    with pytest.raises(GleamValidationError, match="must be positive"):
        calc_cohort_totals(
            value=100,
            cohort_stock_size=50,
            ration_intake=10,
            feed_emissions_list=feed_emissions_empty,
            simulation_duration=-10,
            variable_name="ch4_enteric",
            variable_type="Emissions",
        )


def test_calc_cohort_totals_invalid_type_message_lists_values_like_cli():
    with pytest.raises(
        GleamValidationError,
        match="must be one of: Production, Emissions, Feed, and NitrogenBalance. Found invalid values: Bad and Worse",
    ):
        calc_cohort_totals(1, 1, 1, [], 365, "x", ["Bad", "Worse"])


# ---- calc_allocated_emissions ----------------------------------------------


def test_calc_allocated_emissions_returns_correct_value_for_valid_inputs():
    result = calc_allocated_emissions(value=1000, allocation_share=0.6)
    assert isinstance(result, float)
    assert result == approx(1000 * 0.6)


def test_calc_allocated_emissions_handles_zero_allocation():
    assert calc_allocated_emissions(value=1000, allocation_share=0) == approx(0)


def test_calc_allocated_emissions_handles_full_allocation():
    assert calc_allocated_emissions(value=1000, allocation_share=1) == approx(1000)


def test_calc_allocated_emissions_handles_vectorized_inputs():
    result = calc_allocated_emissions(value=[1000, 500, 200], allocation_share=[0.6, 0.4, 0.8])
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    assert result == approx([1000 * 0.6, 500 * 0.4, 200 * 0.8])


def test_calc_allocated_emissions_validates_input_lengths():
    with pytest.raises(GleamValidationError, match="must have the same length"):
        calc_allocated_emissions(value=[100, 200], allocation_share=[0.5, 0.6, 0.7])


def test_calc_allocated_emissions_validates_bounds():
    with pytest.raises(GleamValidationError, match="must be between 0 and 1"):
        calc_allocated_emissions(value=100, allocation_share=-0.1)
    with pytest.raises(GleamValidationError, match="must be between 0 and 1"):
        calc_allocated_emissions(value=100, allocation_share=1.5)


def test_calc_allocated_emissions_rejects_missing_share_like_r():
    # R: if (any(NA < 0 | NA > 1)) -> "missing value where TRUE/FALSE needed"
    with pytest.raises(GleamValidationError, match="missing value where TRUE/FALSE needed"):
        calc_allocated_emissions(value=100, allocation_share=np.nan)


# ---- calc_co2eq ------------------------------------------------------------


def test_calc_co2eq_returns_correct_value_for_ch4_with_ar6():
    result = calc_co2eq(gas="CH4", value_allocated=100, global_warming_potential_set="AR6")
    assert isinstance(result, dict)
    assert list(result) == ["value_co2eq", "gwp"]
    assert result["value_co2eq"] == approx(100 * 27)
    assert result["gwp"] == approx(27)


def test_calc_co2eq_returns_correct_value_for_n2o_with_ar6():
    result = calc_co2eq(gas="N2O", value_allocated=10, global_warming_potential_set="AR6")
    assert result["value_co2eq"] == approx(10 * 273)
    assert result["gwp"] == approx(273)


def test_calc_co2eq_returns_correct_value_for_co2():
    result = calc_co2eq(gas="CO2", value_allocated=1000, global_warming_potential_set="AR6")
    assert result["value_co2eq"] == approx(1000 * 1)
    assert result["gwp"] == approx(1)


def test_calc_co2eq_handles_ar5_excluding_carbon_feedback():
    result_ch4 = calc_co2eq(gas="CH4", value_allocated=100, global_warming_potential_set="AR5_excluding_carbon_feedback")
    assert result_ch4["value_co2eq"] == approx(100 * 28)
    assert result_ch4["gwp"] == approx(28)
    result_n2o = calc_co2eq(gas="N2O", value_allocated=10, global_warming_potential_set="AR5_excluding_carbon_feedback")
    assert result_n2o["value_co2eq"] == approx(10 * 265)
    assert result_n2o["gwp"] == approx(265)


def test_calc_co2eq_handles_ar5_including_carbon_feedback():
    result_ch4 = calc_co2eq(gas="CH4", value_allocated=100, global_warming_potential_set="AR5_including_carbon_feedback")
    assert result_ch4["value_co2eq"] == approx(100 * 34)
    assert result_ch4["gwp"] == approx(34)
    result_n2o = calc_co2eq(gas="N2O", value_allocated=10, global_warming_potential_set="AR5_including_carbon_feedback")
    assert result_n2o["value_co2eq"] == approx(10 * 298)
    assert result_n2o["gwp"] == approx(298)


def test_calc_co2eq_handles_ar4():
    result_ch4 = calc_co2eq(gas="CH4", value_allocated=100, global_warming_potential_set="AR4")
    assert result_ch4["value_co2eq"] == approx(100 * 25)
    assert result_ch4["gwp"] == approx(25)
    result_n2o = calc_co2eq(gas="N2O", value_allocated=10, global_warming_potential_set="AR4")
    assert result_n2o["value_co2eq"] == approx(10 * 298)
    assert result_n2o["gwp"] == approx(298)


def test_calc_co2eq_handles_vectorized_inputs():
    result = calc_co2eq(gas=["CH4", "N2O", "CO2"], value_allocated=[100, 10, 1000], global_warming_potential_set="AR6")
    assert isinstance(result, dict)
    assert len(result["value_co2eq"]) == 3
    assert len(result["gwp"]) == 3
    assert result["value_co2eq"] == approx([100 * 27, 10 * 273, 1000 * 1])
    assert result["gwp"] == approx([27, 273, 1])


def test_calc_co2eq_handles_zero_emissions():
    result = calc_co2eq(gas="CH4", value_allocated=0, global_warming_potential_set="AR6")
    assert result["value_co2eq"] == approx(0)
    assert result["gwp"] == approx(27)


def test_calc_co2eq_validates_gwp_version():
    with pytest.raises(GleamValidationError, match="must be one of"):
        calc_co2eq(gas="CH4", value_allocated=100, global_warming_potential_set="INVALID")


def test_calc_co2eq_validates_input_lengths():
    with pytest.raises(GleamValidationError, match="must have the same length"):
        calc_co2eq(gas=["CH4", "N2O"], value_allocated=[100, 10, 50], global_warming_potential_set="AR6")


def test_calc_co2eq_validates_gas_types():
    with pytest.raises(GleamValidationError, match="must be one of"):
        calc_co2eq(gas="INVALID", value_allocated=100, global_warming_potential_set="AR6")


# ---- run_aggregation_module validation ---------------------------------------


def _aggregation_inputs():
    return load_example("aggregation_input_chrt_data.csv"), load_example("aggregation_allocation_input_data.csv")


def test_run_aggregation_module_validates_inputs():
    chrt, alloc = _aggregation_inputs()
    with pytest.raises(GleamValidationError, match="global_warming_potential_set"):
        run_aggregation_module(chrt, alloc, global_warming_potential_set="AR9", show_indicator=False)
    with pytest.raises(GleamValidationError, match="must be positive"):
        run_aggregation_module(chrt, alloc, simulation_duration=0, show_indicator=False)
    with pytest.raises(GleamValidationError, match="must be a data.table"):
        run_aggregation_module(chrt, alloc.to_dict(), show_indicator=False)
    bad = chrt.copy()
    bad.loc[bad["herd_id"] == 1, "species_short"] = "BFL"
    with pytest.raises(GleamValidationError, match=r"combinations in `cohort_level_data` have no"):
        run_aggregation_module(bad, alloc, show_indicator=False)
    with pytest.raises(GleamValidationError, match="not found in `allocation_herd_long`"):
        run_aggregation_module(chrt, alloc[alloc["herd_id"] != 3], show_indicator=False)


def test_run_aggregation_module_does_not_mutate_inputs():
    chrt, alloc = _aggregation_inputs()
    c0, a0 = chrt.copy(), alloc.copy()
    run_aggregation_module(chrt, alloc, show_indicator=False)
    pd.testing.assert_frame_equal(chrt, c0)
    pd.testing.assert_frame_equal(alloc, a0)


# ---- vectorisation: arrays give the same result as element-wise scalar calls --


def test_vectorised_cohort_totals_matches_scalar_calls():
    names = ["milk_production_mass_cohort", "ch4_enteric", "co2_ration_fertilizer", "ration_intake",
             "nitrogen_intake", "co2_ration_fertilizer", "ch4_ration_rice"]
    types = ["Production", "Emissions", "Emissions", "Feed", "NitrogenBalance", "Feed", "Emissions"]
    value = [1000.0, 0.5, 33.4, 10.0, 0.2, 7.0, 0.3]
    stock = [50.0, 100.0, 100.0, 30.0, 25.0, 4.0, 12.5]
    intake = [10.0, 5.0, 12.5, 10.0, 8.0, 3.0, 6.5]
    duration = [365.0, 365.0, 365.0, 180.0, 365.0, 365.0, 90.0]
    feed = GLEAM_FEED_EMISSIONS_META
    vec = calc_cohort_totals(np.array(value), np.array(stock), np.array(intake), feed, np.array(duration),
                             np.array(names, dtype=object), np.array(types, dtype=object))
    ser = calc_cohort_totals(pd.Series(value), pd.Series(stock), pd.Series(intake), feed, pd.Series(duration),
                             pd.Series(names), pd.Series(types))
    rows = np.array([calc_cohort_totals(value[i], stock[i], intake[i], feed, duration[i], names[i], types[i])
                     for i in range(len(value))])
    np.testing.assert_array_equal(vec, rows)
    np.testing.assert_array_equal(ser, rows)
    # Feed variable named like a feed emission is not scaled by ration_intake
    assert rows[5] == 7.0 * 4.0 * 365.0
    assert rows[2] == 33.4 * 12.5 * 100.0 * 365.0 / 1000


def test_vectorised_allocated_emissions_and_co2eq_match_scalar_calls():
    value = [1000.0, 500.0, 200.0, 0.0, 12.5]
    share = [0.6, 0.4, 0.8, 1.0, 0.0]
    gas = ["CH4", "N2O", "CO2", "N2O", "CH4"]
    vec = calc_allocated_emissions(np.array(value), pd.Series(share))
    rows = np.array([calc_allocated_emissions(v, s) for v, s in zip(value, share)])
    np.testing.assert_array_equal(vec, rows)
    for gwp in ("AR6", "AR5_excluding_carbon_feedback", "AR5_including_carbon_feedback", "AR4"):
        out = calc_co2eq(pd.Series(gas), vec, gwp)
        rows_co2 = [calc_co2eq(g, v, gwp) for g, v in zip(gas, rows)]
        np.testing.assert_array_equal(out["value_co2eq"], [r["value_co2eq"] for r in rows_co2])
        np.testing.assert_array_equal(out["gwp"], [r["gwp"] for r in rows_co2])
