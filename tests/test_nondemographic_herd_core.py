"""Port of tests/testthat/test-nondemographic_herd_core.R (plus a few checks of the port)."""

import numpy as np
import pandas as pd
import pytest

import gleam
from gleam.validation import GleamValidationError
from gleam.validation.nondemographic_herd_run import validate_run_nondemographic_herd_module_inputs

TOL = 1.5e-8  # testthat's default tolerance


def _cohorts(**extra):
    data = {
        "herd_id": [1, 1, 1, 1],
        "cohort_short": ["FN", "FN", "MN", "MN"],
        "nondemo_productive_phase_id": [1.0, 2.0, 1.0, 2.0],
    }
    data.update(extra)
    data["death_rate"] = [0.1, 0.1, 0.1, 0.1]
    return pd.DataFrame(data)


def _herd(**extra):
    data = {
        "herd_id": [1],
        "cohort_stock_fem_annual_nondemo": [100.0],
        "cohort_stock_mal_annual_nondemo": [120.0],
        "rest_between_nondemo_cycles_duration": [20.0],
    }
    data.update({k: [v] for k, v in extra.items()})
    return pd.DataFrame(data)


PHASES_30_50 = {
    "phase1_nondemo_fem_duration_days": 30.0,
    "phase2_nondemo_fem_duration_days": 50.0,
    "phase1_nondemo_mal_duration_days": 30.0,
    "phase2_nondemo_mal_duration_days": 50.0,
}


def test_validate_run_nondemographic_herd_module_inputs_accepts_valid_example_data():
    cohort_level_data = _cohorts(cohort_duration_days=[30.0, 50.0, 30.0, 50.0])
    validate_run_nondemographic_herd_module_inputs(cohort_level_data, _herd())


def test_validate_run_nondemographic_herd_module_inputs_accepts_herd_level_phase_durations():
    validate_run_nondemographic_herd_module_inputs(_cohorts(), _herd(**PHASES_30_50))


def test_validate_run_nondemographic_herd_module_inputs_rejects_invalid_non_demo_cohorts():
    cohort_level_data = pd.DataFrame({
        "herd_id": [1], "cohort_short": ["FA"], "nondemo_productive_phase_id": [1.0],
        "cohort_duration_days": [30.0], "death_rate": [0.1],
    })
    with pytest.raises(GleamValidationError, match="cohort_short"):
        validate_run_nondemographic_herd_module_inputs(cohort_level_data, _herd())


def test_validate_run_nondemographic_herd_module_inputs_rejects_missing_phase_1():
    cohort_level_data = pd.DataFrame({
        "herd_id": [1, 1], "cohort_short": ["FN", "MN"], "nondemo_productive_phase_id": [2.0, 1.0],
        "cohort_duration_days": [50.0, 30.0], "death_rate": [0.1, 0.1],
    })
    with pytest.raises(GleamValidationError, match="phase 1 row"):
        validate_run_nondemographic_herd_module_inputs(cohort_level_data, _herd())


def test_calc_nondemo_cycle_geometry_returns_expected_cycle_structure():
    res = gleam.calc_nondemo_cycle_geometry(
        phase1_nondemo_duration=30, phase2_nondemo_duration=50, rest_between_nondemo_cycles_duration=20
    )
    assert res["cycle_length"] == pytest.approx(100, rel=TOL)
    assert res["number_full_nondemo_cycles"] == pytest.approx(3, rel=TOL)
    assert res["partial_phase1_nondemo_duration"] == pytest.approx(30, rel=TOL)
    assert res["partial_phase2_nondemo_duration"] == pytest.approx(35, rel=TOL)
    assert res["total_nondemo_cycle_starts_to_distribute"] == pytest.approx(4, rel=TOL)


def test_calc_nondemo_start_sizes_returns_expected_entrants_per_cycle():
    res = gleam.calc_nondemo_start_sizes(
        cohort_stock_nondemo_annual_entrants=120, total_nondemo_cycle_starts_to_distribute=4
    )
    assert isinstance(res, dict)
    assert res["cohort_stock_nondemo_start_cycle"] == pytest.approx(30, rel=TOL)


def test_calc_nondemo_offtake_total_horizon_follows_assessment_year_cycle_starts():
    res = gleam.calc_nondemo_offtake_total_horizon(
        cohort_stock_nondemo_end_phase1=90,
        cohort_stock_nondemo_end_phase2=80,
        cohort_stock_nondemo_annual_entrants=120,
        cohort_stock_nondemo_start_cycle=30,
        number_full_nondemo_cycles=3,
        partial_phase1_nondemo_duration=30,
        partial_phase2_nondemo_duration=50,
        phase1_nondemo_duration=30,
        phase2_nondemo_duration=50,
        simulation_duration=365,
    )
    assert res["offtake_heads_nondemo_phase1"] == pytest.approx(0, abs=TOL)
    assert res["offtake_heads_nondemo_phase2"] == pytest.approx(120 * (80 / 30), rel=TOL)
    assert res["offtake_heads_assessment_nondemo_phase2"] == pytest.approx(120 * (80 / 30), rel=TOL)


def test_calc_nondemo_offtake_total_horizon_keeps_annual_offtake_when_cycle_exceeds_365_days():
    res = gleam.calc_nondemo_offtake_total_horizon(
        cohort_stock_nondemo_end_phase1=90,
        cohort_stock_nondemo_end_phase2=75,
        cohort_stock_nondemo_annual_entrants=120,
        cohort_stock_nondemo_start_cycle=120,
        number_full_nondemo_cycles=0,
        partial_phase1_nondemo_duration=200,
        partial_phase2_nondemo_duration=0,
        phase1_nondemo_duration=200,
        phase2_nondemo_duration=150,
        simulation_duration=365,
    )
    assert res["offtake_heads_nondemo_phase1"] == pytest.approx(0, abs=TOL)
    assert res["offtake_heads_nondemo_phase2"] == pytest.approx(75, rel=TOL)
    assert res["offtake_heads_assessment_nondemo_phase2"] == pytest.approx(75, rel=TOL)


def test_calc_nondemo_phase_returns_zero_stock_for_zero_duration_phases():
    res = gleam.calc_nondemo_phase(
        cohort_stock_nondemo_start_by_phase=25,
        productive_phase_nondemo_duration=0,
        death_rate_nondemo_phase=0,
        max_simulation_days_nondemo_phase=0,
    )
    assert res["time_simulated_nondemographic"] == 0
    assert res["cohort_stock_nondemo"]["start"] == 0
    assert res["cohort_stock_nondemo"]["end"] == 0


def test_run_nondemographic_herd_module_returns_expected_output_columns():
    res = gleam.run_nondemographic_herd_module(
        cohort_level_data=_cohorts(cohort_duration_days=[30.0, 50.0, 30.0, 50.0]),
        herd_level_data=_herd(),
        simulation_duration=365,
    )
    assert isinstance(res, dict)
    assert list(res) == ["cohort_level_results", "herd_level_results"]
    clr, hlr = res["cohort_level_results"], res["herd_level_results"]
    assert isinstance(clr, pd.DataFrame) and isinstance(hlr, pd.DataFrame)
    assert len(clr) == 4
    for col in ("cohort_stock_size_unscaled", "partial_nondemo_phase_duration", "offtake_heads_unscaled",
                "offtake_heads_assessment_unscaled", "offtake_rate"):
        assert col in clr.columns
    assert (clr["offtake_rate"] == 1).all()
    assert "total_nondemo_fem_duration_days" in hlr.columns
    assert "total_nondemo_mal_duration_days" in hlr.columns
    np.testing.assert_allclose(hlr["total_nondemo_fem_duration_days"], [80], rtol=TOL)
    np.testing.assert_allclose(hlr["total_nondemo_mal_duration_days"], [80], rtol=TOL)


def test_run_nondemographic_herd_module_assigns_cohort_durations_from_herd_level_inputs():
    res = gleam.run_nondemographic_herd_module(
        cohort_level_data=_cohorts(), herd_level_data=_herd(**PHASES_30_50), simulation_duration=365
    )
    clr = res["cohort_level_results"]

    def dur(cohort, phase):
        sel = (clr["cohort_short"] == cohort) & (clr["nondemo_productive_phase_id"] == phase)
        return clr.loc[sel, "cohort_duration_days"].tolist()

    assert dur("FN", 1) == pytest.approx([30], rel=TOL)
    assert dur("FN", 2) == pytest.approx([50], rel=TOL)
    assert dur("MN", 1) == pytest.approx([30], rel=TOL)
    assert dur("MN", 2) == pytest.approx([50], rel=TOL)
    np.testing.assert_allclose(res["herd_level_results"]["total_nondemo_fem_duration_days"], [80], rtol=TOL)
    np.testing.assert_allclose(res["herd_level_results"]["total_nondemo_mal_duration_days"], [80], rtol=TOL)


def test_run_nondemographic_herd_module_works_without_cohort_duration_days_column():
    cohort_level_data = _cohorts()
    res = gleam.run_nondemographic_herd_module(
        cohort_level_data=cohort_level_data, herd_level_data=_herd(**PHASES_30_50), simulation_duration=365
    )
    assert "cohort_duration_days" not in cohort_level_data.columns
    assert res["cohort_level_results"]["cohort_duration_days"].tolist() == pytest.approx([30, 50, 30, 50], rel=TOL)


def test_run_nondemographic_herd_module_drops_zero_stock_non_demo_rows():
    cohort_level_data = pd.DataFrame({
        "herd_id": [1, 1], "cohort_short": ["FN", "MN"], "nondemo_productive_phase_id": [1.0, 1.0],
        "death_rate": [0.1, 0.1],
    })
    herd_level_data = pd.DataFrame({
        "herd_id": [1],
        "cohort_stock_fem_annual_nondemo": [0.0],
        "cohort_stock_mal_annual_nondemo": [100.0],
        "rest_between_nondemo_cycles_duration": [20.0],
        "phase1_nondemo_fem_duration_days": [20.0],
        "phase2_nondemo_fem_duration_days": [0.0],
        "phase1_nondemo_mal_duration_days": [20.0],
        "phase2_nondemo_mal_duration_days": [0.0],
    })
    res = gleam.run_nondemographic_herd_module(
        cohort_level_data=cohort_level_data, herd_level_data=herd_level_data, simulation_duration=365
    )
    clr = res["cohort_level_results"]
    assert not (clr["cohort_short"] == "FN").any()
    assert (clr["cohort_short"] == "MN").any()
    assert (clr["cohort_stock_size_unscaled"] > 0).all()


# ---- additional checks of the Python port ----
def test_nondemo_core_functions_are_vectorised():
    geom = gleam.calc_nondemo_cycle_geometry(np.array([30.0, 60, 0]), np.array([50.0, 365, 0]), 20.0)
    np.testing.assert_array_equal(geom["number_full_nondemo_cycles"], [3, 0, 0])
    np.testing.assert_array_equal(geom["partial_phase2_nondemo_duration"], [35, 305, 0])
    np.testing.assert_array_equal(geom["total_nondemo_cycle_starts_to_distribute"], [4, 1, 0])
    phase = gleam.calc_nondemo_phase(np.array([10.0, 10.0]), np.array([10.0, 0.0]), 0.1, np.array([5.0, 0.0]))
    assert phase["time_simulated_nondemographic"].tolist() == [5.0, 0.0]
    assert phase["cohort_stock_nondemo"]["end"][1] == 0.0
    out = gleam.assign_nondemographic_phase_durations(
        ["FN", "FN", "MN", "FA"], [1, 2, 2, np.nan], [5.0, 6.0, 7.0, 8.0], 30, np.nan, 31, 32
    )
    np.testing.assert_array_equal(out, [30, 6, 32, 8])
    assert gleam.calc_nondemographic_total_durations(["FN", "MN", "FN"], [30, 40, np.nan]) == {
        "total_nondemo_fem_duration_days": 30.0, "total_nondemo_mal_duration_days": 40.0,
    }


def test_calc_nondemo_start_sizes_returns_zero_without_cycle_start():
    # R returns the bare value 0 (not a list) when there is no cycle start
    assert gleam.calc_nondemo_start_sizes(120, 0) == 0


def test_run_nondemographic_herd_module_fails_like_r_without_cycle_start():
    herd = _herd(**{**PHASES_30_50, "phase1_nondemo_mal_duration_days": 0.0})
    with pytest.raises(ValueError, match="operator is invalid for atomic vectors"):
        gleam.run_nondemographic_herd_module(_cohorts(), herd, show_indicator=False)


def test_run_nondemographic_herd_module_truncates_into_integer_duration_column():
    # data.table coerces the assigned doubles to an integer cohort_duration_days column
    cohort = pd.DataFrame({
        "herd_id": [1, 1], "cohort_short": ["FN", "FN"], "nondemo_productive_phase_id": [1.0, 2.0],
        "death_rate": [0.1, 0.1], "cohort_duration_days": np.array([5, 10], dtype="int64"),
    })
    herd = _herd(**{**PHASES_30_50, "phase1_nondemo_fem_duration_days": 30.7,
                    "phase2_nondemo_fem_duration_days": 50.2,
                    "phase1_nondemo_mal_duration_days": np.nan, "phase2_nondemo_mal_duration_days": np.nan})
    with pytest.warns(gleam.GleamWarning, match="truncated"):
        res = gleam.run_nondemographic_herd_module(cohort, herd, show_indicator=False)
    clr = res["cohort_level_results"]
    assert clr["cohort_duration_days"].tolist() == [30, 50]
    # values printed by R for the same case
    assert clr["cohort_stock_size_unscaled"].tolist() == pytest.approx([7.808219, 10.865082], rel=1e-6)


def test_run_nondemographic_herd_module_reports_r_scalar_validation_message():
    herd = _herd(**{**PHASES_30_50, "phase2_nondemo_mal_duration_days": 9000.0})
    with pytest.raises(GleamValidationError, match=r"^`cohort_duration_days` = 9000 is out of range"):
        gleam.run_nondemographic_herd_module(_cohorts(), herd, show_indicator=False)
