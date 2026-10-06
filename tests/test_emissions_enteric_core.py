"""Port of ``tests/testthat/test-emissions_enteric_core.R`` plus vectorisation checks."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from gleam import (
    GleamValidationError,
    calc_ch4_enteric,
    calc_conversion_factor_ym,
    load_example,
    run_emissions_enteric_module,
    validation_disabled,
)

COHORTS_SMALL = ["FJ", "FS", "FA", "MJ", "MS", "MA"]
ALL_SPECIES = ["CTL", "BFL", "SHP", "GTS", "PGS", "CML", "CHK"]
ALL_COHORTS = ["FJ", "FS", "FA", "MJ", "MS", "MA", "FN", "MN"]


def approx(x):
    return pytest.approx(x, rel=1.5e-8)


# ---- calc_conversion_factor_ym ----------------------------------------------


def test_calc_conversion_factor_ym_validates_inputs_and_computes_ym():
    # Valid cattle case
    ch4_ym = calc_conversion_factor_ym("CTL", "FA", 0.6)
    assert isinstance(ch4_ym, float)
    assert ch4_ym > 0

    # Digestibility bounds
    with pytest.raises(GleamValidationError):
        calc_conversion_factor_ym("CTL", "FA", -0.1)
    with pytest.raises(GleamValidationError):
        calc_conversion_factor_ym("CTL", "FA", 1.1)

    # Boundary values for digestibility
    assert calc_conversion_factor_ym("CTL", "FA", 0) == approx(9.75)
    assert calc_conversion_factor_ym("CTL", "FA", 1) == approx(4.75)

    # Pigs: adult vs juvenile
    assert calc_conversion_factor_ym("PGS", "FA", 0.65) == approx(1.01)
    assert calc_conversion_factor_ym("PGS", "FJ", 0.65) == approx(0)

    # Small ruminants/camels: juvenile/subadult vs adult
    ym_juv = calc_conversion_factor_ym("SHP", "FJ", 0)
    ym_adult = calc_conversion_factor_ym("SHP", "FA", 0.65)  # 9.75 rule
    assert ym_juv < ym_adult

    # Buffalo: same formula as cattle
    assert calc_conversion_factor_ym("BFL", "FA", 0.6) == approx(calc_conversion_factor_ym("CTL", "FA", 0.6))

    # Invalid species should error
    with pytest.raises(GleamValidationError):
        calc_conversion_factor_ym("XXX", "FA", 0.65)


def test_calc_conversion_factor_ym_branch_values():
    d = 0.7
    assert calc_conversion_factor_ym("SHP", "FS", d) == approx(7.75 - 0.05 * d * 100)
    assert calc_conversion_factor_ym("CML", "MN", d) == approx(7.75 - 0.05 * d * 100)
    assert calc_conversion_factor_ym("GTS", "MA", d) == approx(9.75 - 0.05 * d * 100)
    assert calc_conversion_factor_ym("CTL", "FN", d) == approx(9.75 - 0.05 * d * 100)
    assert calc_conversion_factor_ym("PGS", "MN", d) == approx(0.39)
    assert calc_conversion_factor_ym("CHK", "FA", d) == 0
    assert calc_conversion_factor_ym("BFL", "MJ", d) == 0


def test_calc_conversion_factor_ym_unknown_species_errors_without_validation():
    # R fails ("object not found") for species outside every branch, even
    # though validate_ym_inputs() only checks the type of species_short.
    with validation_disabled(), pytest.raises(GleamValidationError):
        calc_conversion_factor_ym(["CTL", "XXX"], "FA", 0.65)


# ---- calc_ch4_enteric -----------------------------------------------------


def test_calc_ch4_enteric_validates_inputs_and_returns_expected_numeric():
    ch4_ym = calc_conversion_factor_ym("CTL", "FA", 0.6)
    ch4 = calc_ch4_enteric(
        species_short="CTL",
        ch4_conversion_factor_ym=ch4_ym,
        ch4_mitigation_factor=1,
        ration_gross_energy=18.4,
        ration_intake=10,
    )
    assert isinstance(ch4, float)
    assert ch4 > 0

    # Zero inputs give zero emissions
    assert calc_ch4_enteric("CTL", 0, 1, 18, 10) == approx(0)
    assert calc_ch4_enteric("CTL", 5, 1, 18, 0) == approx(0)

    # Invalid arguments
    with pytest.raises(GleamValidationError):
        calc_ch4_enteric("CTL", -1, 1, 18, 10)
    with pytest.raises(GleamValidationError):
        calc_ch4_enteric("CTL", 5, 1, -1, 10)
    with pytest.raises(GleamValidationError):
        calc_ch4_enteric("CTL", 5, 1, 18, -1)

    # Emissions must be non-negative
    assert ch4 >= 0


def test_calc_ch4_enteric_formula():
    assert calc_ch4_enteric("CTL", 6.5, 0.9, 18.4, 10) == approx(18.4 * 10 * (6.5 / 100) * 0.9 / 55.65)


def test_calc_ch4_enteric_error_messages_follow_r():
    with pytest.raises(GleamValidationError, match=r"`ration_intake` = -1 is out of range"):
        calc_ch4_enteric("CTL", 5, 1, 18, -1)
    with pytest.raises(GleamValidationError, match=r"`species_short` must be a single character value"):
        calc_ch4_enteric(None, 5, 1, 18, 1)


# ---- vectorisation ------------------------------------------------------------


def _mixed_inputs():
    combos = list(itertools.product(ALL_SPECIES, ALL_COHORTS))
    rng = np.random.default_rng(42)
    n = len(combos)
    return (
        np.array([c[0] for c in combos], dtype=object),
        np.array([c[1] for c in combos], dtype=object),
        rng.uniform(0.3, 0.95, n),
        rng.uniform(0.5, 1.0, n),
        rng.uniform(15, 22, n),
        rng.uniform(0.05, 20, n),
    )


def test_vectorised_ym_matches_elementwise_scalar_calls():
    sp, co, dig, *_ = _mixed_inputs()
    vec = calc_conversion_factor_ym(sp, co, dig)
    scal = np.array([calc_conversion_factor_ym(s, c, d) for s, c, d in zip(sp, co, dig)])
    assert isinstance(vec, np.ndarray) and vec.shape == sp.shape
    np.testing.assert_array_equal(vec, scal)
    # pandas Series input gives the same values
    np.testing.assert_array_equal(calc_conversion_factor_ym(pd.Series(sp), pd.Series(co), pd.Series(dig)), scal)


def test_vectorised_ch4_enteric_matches_elementwise_scalar_calls():
    sp, co, dig, mit, ge, dmi = _mixed_inputs()
    ym = calc_conversion_factor_ym(sp, co, dig)
    vec = calc_ch4_enteric(sp, ym, mit, ge, dmi)
    scal = np.array([calc_ch4_enteric(*args) for args in zip(sp, ym, mit, ge, dmi)])
    np.testing.assert_array_equal(vec, scal)
    # scalars broadcast against vectors
    np.testing.assert_array_equal(
        calc_ch4_enteric("CTL", ym, 1, 18.0, dmi),
        np.array([calc_ch4_enteric("CTL", y, 1, 18.0, d) for y, d in zip(ym, dmi)]),
    )


def test_vectorised_validation_reports_offending_element():
    with pytest.raises(GleamValidationError, match=r"`ration_digestibility_fraction`\[2\] = 1.5"):
        calc_conversion_factor_ym(["CTL", "CTL"], ["FA", "FS"], [0.5, 1.5])


# ---- run module ----------------------------------------------------------------


def test_run_emissions_enteric_module_keeps_input_and_adds_columns():
    chrt = load_example("emissions_enteric_input_chrt_data.csv")
    before = chrt.copy()
    res = run_emissions_enteric_module(chrt, show_indicator=False)
    pd.testing.assert_frame_equal(chrt, before)  # caller's table untouched
    assert list(res.columns) == list(chrt.columns) + [
        "ch4_mitigation_factor", "ch4_conversion_factor_ym", "ch4_enteric",
    ]
    assert (res["ch4_mitigation_factor"] == 1).all()


def test_run_emissions_enteric_module_uses_supplied_mitigation_factor():
    chrt = load_example("emissions_enteric_input_chrt_data.csv")
    base = run_emissions_enteric_module(chrt, show_indicator=False)
    chrt["ch4_mitigation_factor"] = 0.9
    mitigated = run_emissions_enteric_module(chrt, show_indicator=False)
    assert list(mitigated.columns) == list(base.columns)
    np.testing.assert_allclose(mitigated["ch4_enteric"], base["ch4_enteric"] * 0.9, rtol=1e-12)


def test_run_emissions_enteric_module_validates_inputs():
    chrt = load_example("emissions_enteric_input_chrt_data.csv")
    with pytest.raises(GleamValidationError, match="Missing required columns in `data`"):
        run_emissions_enteric_module(chrt.drop(columns="ration_intake"), show_indicator=False)
    with pytest.raises(GleamValidationError, match="must contain at least one row"):
        run_emissions_enteric_module(chrt.iloc[:0], show_indicator=False)
    bad = chrt.copy()
    bad.loc[0, "species_short"] = "XXX"
    with pytest.raises(GleamValidationError, match="Invalid `species_short` values"):
        run_emissions_enteric_module(bad, show_indicator=False)
    with pytest.raises(GleamValidationError, match="exactly 6 rows"):
        run_emissions_enteric_module(chrt.iloc[1:], show_indicator=False)


def test_run_emissions_enteric_module_without_validation_warns():
    chrt = load_example("emissions_enteric_input_chrt_data.csv")
    with pytest.warns(UserWarning, match="Input validation has been turned off"):
        res = run_emissions_enteric_module(chrt, show_indicator=False, validate_inputs=False)
    ref = run_emissions_enteric_module(chrt, show_indicator=False)
    pd.testing.assert_frame_equal(res, ref)
