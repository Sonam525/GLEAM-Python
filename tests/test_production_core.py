"""Port of ``tests/testthat/test-production_core.R`` plus vectorisation checks."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from gleampy import (
    GleamValidationError,
    calc_egg_production,
    calc_fibre_production,
    calc_meat_production,
    calc_milk_production,
    load_example,
    run_production_module,
)
from float_compare import assert_same_float

ALL_SPECIES = ["CTL", "BFL", "SHP", "GTS", "PGS", "CML", "CHK"]
ALL_COHORTS = ["FJ", "FS", "FA", "MJ", "MS", "MA", "FN", "MN"]
MILK_KEYS = ["milk_production_mass_cohort", "milk_production_protein_cohort", "milk_production_fpcm_cohort"]
EGG_KEYS = ["egg_production_number_cohort", "egg_production_mass_cohort", "egg_production_protein_cohort"]
MEAT_KEYS = [
    "meat_production_live_weight_cohort",
    "meat_production_carcass_weight_cohort",
    "meat_production_bone_free_meat_cohort",
    "meat_production_protein_cohort",
]

STANDARD = dict(
    milk_protein_fraction_standard=0.033,
    milk_fat_fraction_standard=0.04,
    milk_lactose_fraction_standard=0.048,
)


def approx(x, rel=1.5e-8):
    return pytest.approx(x, rel=rel)


def milk(species_short="CTL", cohort_short="FA", milk_yield_day=10, simulation_duration=365,
         cohort_stock_size=100, lactating_females_fraction=0.8, milk_protein_fraction=0.033,
         milk_fat_fraction=0.04, milk_lactose_fraction=0.048, **kw):
    args = dict(
        species_short=species_short, cohort_short=cohort_short, milk_yield_day=milk_yield_day,
        simulation_duration=simulation_duration, cohort_stock_size=cohort_stock_size,
        lactating_females_fraction=lactating_females_fraction,
        milk_protein_fraction=milk_protein_fraction, milk_fat_fraction=milk_fat_fraction,
        milk_lactose_fraction=milk_lactose_fraction, **STANDARD,
    )
    args.update(kw)
    return calc_milk_production(**args)


# ---- calc_milk_production ----------------------------------------------------------


def test_calc_milk_production_returns_expected_output_structure():
    result = milk("CTL", "FA", 10, 365, 100, 0.8, 0.033, 0.04, 0.048)
    assert isinstance(result, dict)
    assert list(result) == MILK_KEYS


def test_calc_milk_production_calculates_milk_mass_production_correctly():
    result = milk("BFL", "FA", 20, 365, 50, 0.75, 0.032, 0.038, 0.047)
    assert result["milk_production_mass_cohort"] == approx(20 * 365 * 50 * 0.75)


def test_calc_milk_production_calculates_milk_protein_production_correctly():
    result = milk("SHP", "FA", 15, 365, 80, 0.9, 0.035, 0.042, 0.049)
    expected_mass = 15 * 365 * 80 * 0.9
    assert result["milk_production_protein_cohort"] == approx(expected_mass * 0.035)


def test_calc_milk_production_calculates_fpcm_using_energy_ratio():
    result = milk("SHP", "FA", 12, 365, 60, 0.85, 0.033, 0.04, 0.048)
    # When milk composition equals standard, the energy ratio is 1 and FPCM equals milk production
    assert result["milk_production_fpcm_cohort"] == approx(12 * 365 * 60 * 0.85, rel=1e-10)


def test_calc_milk_production_calculates_fpcm_correctly_with_different_composition():
    result = milk("GTS", "FA", 12, 365, 60, 0.85, 0.034, 0.041, 0.048)
    expected_mass = 12 * 365 * 60 * 0.85
    energy_standard = 0.0929 * 0.04 + 0.0547 * 0.033 + 0.0395 * 0.048
    energy_milk = 0.0929 * 0.041 + 0.0547 * 0.034 + 0.0395 * 0.048
    expected_fpcm = energy_milk / energy_standard * expected_mass
    assert result["milk_production_fpcm_cohort"] == approx(expected_fpcm, rel=1e-6)


def test_calc_milk_production_handles_higher_fat_content_correctly():
    standard_result = milk("GTS", "FA", 10, 365, 100, 0.8, 0.033, 0.04, 0.048)
    high_fat_result = milk("GTS", "FA", 10, 365, 100, 0.8, 0.033, 0.05, 0.048)
    assert high_fat_result["milk_production_fpcm_cohort"] > standard_result["milk_production_fpcm_cohort"]


def test_calc_milk_production_handles_zero_size():
    result = milk("CTL", "FA", 10, 365, 0, 0.8, 0.033, 0.04, 0.048)
    assert all(result[k] == 0 for k in MILK_KEYS)


def test_calc_milk_production_handles_zero_milking_fraction():
    result = milk("CTL", "FA", 10, 365, 100, 0, 0.033, 0.04, 0.048)
    assert all(result[k] == 0 for k in MILK_KEYS)


def test_calc_milk_production_handles_validation_errors():
    with pytest.raises(GleamValidationError, match="milk_yield_day"):
        milk("CTL", "FA", milk_yield_day=-10)
    with pytest.raises(GleamValidationError, match="lactating_females_fraction"):
        milk("CTL", "FA", lactating_females_fraction=1.5)
    with pytest.raises(GleamValidationError, match="milk_fat_fraction"):
        milk("CTL", "FA", milk_fat_fraction=1.5)


def test_calc_milk_production_returns_zeros_for_pgs():
    result = milk("PGS", "FA", 10, 365, 100, 0.8, 0.033, 0.04, 0.048)
    assert list(result) == MILK_KEYS
    assert all(result[k] == 0 for k in MILK_KEYS)


def test_calc_milk_production_skips_milk_checks_outside_fa():
    # Milk parameters are only validated (and used) for FA of milk producers.
    result = milk("CTL", "FS", milk_yield_day=np.nan, lactating_females_fraction=np.nan)
    assert all(result[k] == 0 for k in MILK_KEYS)
    with pytest.raises(GleamValidationError, match=r"`species_short` must be one of"):
        milk("XXX", "FA")


# ---- calc_egg_production ----------------------------------------------------------


def test_calc_egg_production_returns_expected_chicken_outputs():
    result = calc_egg_production(
        species_short="CHK", cohort_short="FA", egg_output_human_consumption=36500,
        egg_average_weight=0.06, simulation_duration=365, is_egg_producing=True,
    )
    assert result["egg_production_number_cohort"] == approx(36500)
    assert result["egg_production_mass_cohort"] == approx(36500 * 0.06)
    assert result["egg_production_protein_cohort"] == approx(36500 * 0.06 * 0.125)


def test_calc_egg_production_returns_expected_outputs_for_egg_producing_chicken_fn():
    result = calc_egg_production(
        species_short="CHK", cohort_short="FN", nondemo_productive_phase_id=2,
        egg_output_human_consumption=36500, egg_average_weight=0.06, simulation_duration=365,
        is_egg_producing=True,
    )
    assert result["egg_production_number_cohort"] == approx(36500)
    assert result["egg_production_mass_cohort"] == approx(36500 * 0.06)
    assert result["egg_production_protein_cohort"] == approx(36500 * 0.06 * 0.125)


def test_calc_egg_production_validates_egg_producing_flag_placement():
    with pytest.raises(GleamValidationError, match=r"can be TRUE only for CHK cohorts.*FA.*FN"):
        calc_egg_production(
            species_short="CHK", cohort_short="FS", egg_output_human_consumption=36500,
            egg_average_weight=0.06, simulation_duration=365, is_egg_producing=True,
        )
    with pytest.raises(GleamValidationError, match=r"can be TRUE for.*FN.*only when.*2"):
        calc_egg_production(
            species_short="CHK", cohort_short="FN", nondemo_productive_phase_id=1,
            egg_output_human_consumption=36500, egg_average_weight=0.06, simulation_duration=365,
            is_egg_producing=True,
        )


def test_calc_egg_production_zero_for_non_egg_producing_and_validates_parameters():
    result = calc_egg_production("CHK", "FS", np.nan, np.nan, 365, is_egg_producing=False)
    assert result == {k: 0.0 for k in EGG_KEYS}
    with pytest.raises(GleamValidationError, match=r"`egg_protein_fraction` must be between 0 and 1"):
        calc_egg_production("CHK", "FA", 36500, 0.06, 365, egg_protein_fraction=1.5, is_egg_producing=True)
    with pytest.raises(GleamValidationError, match=r"`egg_output_human_consumption` must be greater than or equal to 0"):
        calc_egg_production("CHK", "FA", -1, 0.06, 365, is_egg_producing=True)
    with pytest.raises(GleamValidationError, match=r"`egg_average_weight` must be positive"):
        calc_egg_production("CHK", "FA", 36500, 0, 365, is_egg_producing=True)


# ---- calc_fibre_production ----------------------------------------------------------


def test_calc_fibre_production_returns_expected_value():
    result = calc_fibre_production("SHP", "FS", 0.1, 365, 100)
    assert result == approx(0.1 / 365 * 365 * 100)


def test_calc_fibre_production_handles_zero_fibre_yield():
    assert calc_fibre_production("SHP", "FS", 0, 365, 100) == approx(0)


def test_calc_fibre_production_handles_zero_size():
    assert calc_fibre_production("SHP", "FA", 0.1, 365, 0) == approx(0)


def test_calc_fibre_production_handles_different_assessment_durations():
    result_365 = calc_fibre_production("GTS", "MA", 0.1, 365, 100)
    result_180 = calc_fibre_production("CML", "MA", 0.1, 180, 100)
    assert result_365 / result_180 == approx(365 / 180)


def test_calc_fibre_production_handles_large_values():
    result = calc_fibre_production("CML", "MS", 5.0, 365, 1000)
    assert result == approx(5.0 / 365 * 365 * 1000)


def test_calc_fibre_production_returns_zero_for_non_fibre_animals_pgs():
    assert calc_fibre_production("PGS", "MS", 1, 365, 1000) == approx(0)


def test_calc_fibre_production_returns_zero_for_non_fibre_animals_ctl():
    assert calc_fibre_production("CTL", "MS", 1, 365, 1000) == approx(0)


def test_calc_fibre_production_handles_validation_errors():
    with pytest.raises(GleamValidationError, match="fibre_yield_year"):
        calc_fibre_production("SHP", "MA", -0.1, 365, 100)


def test_calc_fibre_production_nondemographic_cohorts_produce_but_are_not_validated():
    # R computes fibre for FN/MN but only validates FA/FS/MA/MS.
    assert calc_fibre_production("SHP", "FN", 0.5, 365, 10) == approx(0.5 / 365 * 365 * 10)
    assert calc_fibre_production("SHP", "MN", -0.5, 365, 10) == approx(-0.5 / 365 * 365 * 10)  # R: -5
    # R reference values (sprintf("%a") of calc_fibre_production(sp, co, 2.5, 180, 1000))
    r_value = float.fromhex("0x1.34381c0e07038p+10")
    for sp, co in [("SHP", "FN"), ("GTS", "MN"), ("CML", "FN")]:
        assert calc_fibre_production(sp, co, 2.5, 180, 1000) == approx(r_value, rel=1e-12)
    assert calc_fibre_production("CTL", "FN", 2.5, 180, 1000) == 0


# ---- calc_meat_production ----------------------------------------------------------


def test_calc_meat_production_returns_expected_output_structure():
    result = calc_meat_production(10, 400, 0.55, 0.75, 0.20)
    assert isinstance(result, dict)
    assert list(result) == MEAT_KEYS


def test_calc_meat_production_calculates_liveweight_correctly():
    result = calc_meat_production(50, 300, 0.60, 0.80, 0.22)
    assert result["meat_production_live_weight_cohort"] == approx(50 * 300)


def test_calc_meat_production_calculates_carcass_weight_correctly():
    result = calc_meat_production(25, 450, 0.58, 0.78, 0.21)
    assert result["meat_production_carcass_weight_cohort"] == approx(25 * 450 * 0.58)


def test_calc_meat_production_calculates_boneless_meat_correctly():
    result = calc_meat_production(30, 350, 0.55, 0.70, 0.20)
    assert result["meat_production_bone_free_meat_cohort"] == approx(30 * 350 * 0.55 * 0.70)


def test_calc_meat_production_calculates_meat_protein_correctly():
    result = calc_meat_production(20, 400, 0.56, 0.75, 0.23)
    assert result["meat_production_protein_cohort"] == approx(20 * 400 * 0.56 * 0.75 * 0.23)


def test_calc_meat_production_handles_zero_offtake():
    result = calc_meat_production(0, 400, 0.55, 0.75, 0.20)
    assert all(result[k] == 0 for k in MEAT_KEYS)


def test_calc_meat_production_handles_zero_slaughter_weight():
    result = calc_meat_production(10, 0, 0.55, 0.75, 0.20)
    assert all(result[k] == 0 for k in MEAT_KEYS)


def test_calc_meat_production_verifies_sequential_calculation_chain():
    result = calc_meat_production(100, 300, 0.50, 0.80, 0.25)
    liveweight, carcass, meat, protein = (result[k] for k in MEAT_KEYS)
    assert carcass == approx(liveweight * 0.50)
    assert meat == approx(carcass * 0.80)
    assert protein == approx(meat * 0.25)


def test_calc_meat_production_handles_validation_errors():
    with pytest.raises(GleamValidationError, match="offtake_heads_assessment"):
        calc_meat_production(-10, 400, 0.55, 0.75, 0.20)
    with pytest.raises(GleamValidationError, match="live_weight_cohort_at_slaughter"):
        calc_meat_production(10, -400, 0.55, 0.75, 0.20)
    with pytest.raises(GleamValidationError, match="carcass_dressing_fraction"):
        calc_meat_production(10, 400, 1.5, 0.75, 0.20)
    with pytest.raises(GleamValidationError, match="bone_free_meat_fraction"):
        calc_meat_production(10, 400, 0.55, -0.1, 0.20)
    with pytest.raises(GleamValidationError, match="meat_protein_fraction"):
        calc_meat_production(10, 400, 0.55, 0.75, 1.5)


# ---- vectorisation ----------------------------------------------------------------


def _mixed_inputs() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for sp, co in itertools.product(ALL_SPECIES, ALL_COHORTS):
        for ph in ([1, 2] if co in ("FN", "MN") else [np.nan]):
            egg = None
            if sp == "CHK":
                egg = (co == "FA") or (co == "FN" and ph == 2)
            rows.append({
                "species_short": sp,
                "cohort_short": co,
                "milk_yield_day": rng.uniform(0, 30),
                "simulation_duration": rng.choice([180.0, 365.0]),
                "cohort_stock_size": rng.uniform(0, 1e6),
                "lactating_females_fraction": rng.uniform(0, 1),
                "milk_protein_fraction": rng.uniform(0.02, 0.06),
                "milk_fat_fraction": rng.uniform(0.03, 0.08),
                "milk_lactose_fraction": rng.uniform(0.04, 0.05),
                "milk_protein_fraction_standard": 0.033,
                "milk_fat_fraction_standard": 0.04,
                "milk_lactose_fraction_standard": 0.048,
                "fibre_yield_year": rng.uniform(0, 3),
                "egg_output_human_consumption": rng.uniform(0, 1e7),
                "egg_average_weight": rng.uniform(0.04, 0.07),
                "egg_protein_fraction": rng.uniform(0.1, 0.15),
                "nondemo_productive_phase_id": ph,
                "is_egg_producing": egg,
                "offtake_heads_assessment": rng.uniform(0, 1e5),
                "live_weight_cohort_at_slaughter": rng.uniform(1, 700),
                "carcass_dressing_fraction": rng.uniform(0.4, 0.8),
                "bone_free_meat_fraction": rng.uniform(0.5, 0.9),
                "meat_protein_fraction": rng.uniform(0.15, 0.25),
            })
    df = pd.DataFrame(rows)
    df["is_egg_producing"] = df["is_egg_producing"].astype(object)
    return df


MILK_ARGS = [
    "species_short", "cohort_short", "milk_yield_day", "simulation_duration", "cohort_stock_size",
    "lactating_females_fraction", "milk_protein_fraction", "milk_fat_fraction", "milk_lactose_fraction",
    "milk_protein_fraction_standard", "milk_fat_fraction_standard", "milk_lactose_fraction_standard",
]
FIBRE_ARGS = ["species_short", "cohort_short", "fibre_yield_year", "simulation_duration", "cohort_stock_size"]
EGG_ARGS = [
    "species_short", "cohort_short", "egg_output_human_consumption", "egg_average_weight",
    "simulation_duration", "egg_protein_fraction", "nondemo_productive_phase_id", "is_egg_producing",
]
MEAT_ARGS = [
    "offtake_heads_assessment", "live_weight_cohort_at_slaughter", "carcass_dressing_fraction",
    "bone_free_meat_fraction", "meat_protein_fraction",
]


def _check_vectorised(fn, df, args):
    vec = fn(**{a: df[a] for a in args})
    rows = [fn(**{a: r[a] for a in args}) for r in df.to_dict("records")]
    if isinstance(vec, dict):
        assert list(vec) == list(rows[0])
        for k, v in vec.items():
            assert isinstance(v, np.ndarray) and v.shape == (len(df),)
            assert_same_float(v, [r[k] for r in rows], err_msg=k)
    else:
        assert isinstance(vec, np.ndarray) and vec.shape == (len(df),)
        assert_same_float(vec, rows)


def test_vectorised_milk_matches_elementwise_scalar_calls():
    _check_vectorised(calc_milk_production, _mixed_inputs(), MILK_ARGS)


def test_vectorised_fibre_matches_elementwise_scalar_calls():
    _check_vectorised(calc_fibre_production, _mixed_inputs(), FIBRE_ARGS)


def test_vectorised_egg_matches_elementwise_scalar_calls():
    _check_vectorised(calc_egg_production, _mixed_inputs(), EGG_ARGS)


def test_vectorised_meat_matches_elementwise_scalar_calls():
    _check_vectorised(calc_meat_production, _mixed_inputs(), MEAT_ARGS)


def test_vectorised_scalar_simulation_duration_broadcasts():
    df = _mixed_inputs()
    vec = calc_fibre_production(df["species_short"], df["cohort_short"], df["fibre_yield_year"], 365, df["cohort_stock_size"])
    assert_same_float(
        vec,
        [calc_fibre_production(s, c, f, 365, n)
         for s, c, f, n in zip(df["species_short"], df["cohort_short"], df["fibre_yield_year"], df["cohort_stock_size"])],
    )


# ---- run module -----------------------------------------------------------------


def _example():
    return load_example("production_input_chrt_data.csv"), load_example("production_input_hrd_data.csv")


OUTPUT_COLUMNS = MILK_KEYS + EGG_KEYS + ["fibre_production_cohort"] + MEAT_KEYS


def test_run_production_module_keeps_inputs_and_adds_columns():
    chrt, hrd = _example()
    c0, h0 = chrt.copy(), hrd.copy()
    res = run_production_module(chrt, hrd, simulation_duration=365, show_indicator=False)
    pd.testing.assert_frame_equal(chrt, c0)
    pd.testing.assert_frame_equal(hrd, h0)
    assert list(res.columns) == list(chrt.columns) + OUTPUT_COLUMNS


def test_run_production_module_scales_with_simulation_duration():
    chrt, hrd = _example()
    r365 = run_production_module(chrt, hrd, simulation_duration=365, show_indicator=False)
    r180 = run_production_module(chrt, hrd, simulation_duration=180, show_indicator=False)
    for col in MILK_KEYS + EGG_KEYS + ["fibre_production_cohort"]:
        np.testing.assert_allclose(r180[col], r365[col] * 180 / 365, rtol=1e-12, err_msg=col)
    for col in MEAT_KEYS:
        np.testing.assert_array_equal(r180[col], r365[col])


def test_run_production_module_adds_is_egg_producing_without_chk():
    chrt, hrd = _example()
    hrd = hrd[hrd["species_short"] != "CHK"].drop(columns=["egg_output_human_consumption", "egg_average_weight"])
    chrt = chrt[chrt["herd_id"].isin(hrd["herd_id"])].drop(columns=["is_egg_producing", "nondemo_productive_phase_id"])
    chrt = chrt[~chrt["cohort_short"].isin(["FN", "MN"])]
    res = run_production_module(chrt, hrd, show_indicator=False)
    assert "is_egg_producing" not in chrt.columns
    assert list(res.columns) == list(chrt.columns) + ["is_egg_producing"] + OUTPUT_COLUMNS
    assert (res[EGG_KEYS] == 0).all().all()


def test_run_production_module_validates_inputs():
    chrt, hrd = _example()
    with pytest.raises(GleamValidationError, match="Missing required columns in `herd_level_data`"):
        run_production_module(chrt, hrd.drop(columns="meat_protein_fraction"), show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `cohort_level_data`"):
        run_production_module(chrt.drop(columns="offtake_heads_assessment"), hrd, show_indicator=False)
    with pytest.raises(GleamValidationError, match=r"`simulation_duration` must be positive"):
        run_production_module(chrt, hrd, simulation_duration=0, show_indicator=False)
    with pytest.raises(GleamValidationError, match=r"`simulation_duration` must be a single numeric value"):
        run_production_module(chrt, hrd, simulation_duration=np.nan, show_indicator=False)


def test_run_production_module_without_validation_warns_and_matches():
    chrt, hrd = _example()
    ref = run_production_module(chrt, hrd, show_indicator=False)
    with pytest.warns(UserWarning, match="Input validation has been turned off"):
        res = run_production_module(chrt, hrd, show_indicator=False, validate_inputs=False)
    pd.testing.assert_frame_equal(res, ref)


# ---- simulation_duration must be a single value (032, 067) ----------------------


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("bad", ["two", "two-equal", "per-row", "per-row-varying", "series", "none"])
def test_run_production_module_rejects_non_single_simulation_duration(bad, validate):
    """R: validate_scalar_numeric() requires length 1 ("must be a single numeric value")."""
    chrt, hrd = _example()
    per_row = np.full(len(chrt), 365.0)
    varying = per_row.copy()
    varying[0] = 1.0
    value = {
        "two": [365, 180],
        "two-equal": [365, 365],
        "per-row": per_row,
        "per-row-varying": varying,
        "series": pd.Series(per_row),
        "none": None,
    }[bad]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(GleamValidationError, match=r"^`simulation_duration` must be a single numeric value\.$"):
            run_production_module(chrt, hrd, simulation_duration=value, show_indicator=False, validate_inputs=validate)


def test_run_production_module_accepts_length_one_simulation_duration():
    chrt, hrd = _example()
    ref = run_production_module(chrt, hrd, simulation_duration=180, show_indicator=False)
    for value in ([180], np.array([180.0]), pd.Series([180.0]), np.int64(180), 180.0):
        res = run_production_module(chrt, hrd, simulation_duration=value, show_indicator=False)
        pd.testing.assert_frame_equal(res, ref)


def test_run_production_module_rejects_non_numeric_simulation_duration():
    chrt, hrd = _example()
    for value in ("365", True, np.nan, [np.nan]):
        with pytest.raises(GleamValidationError, match=r"`simulation_duration` must be a single numeric value"):
            run_production_module(chrt, hrd, simulation_duration=value, show_indicator=False)
    with pytest.raises(GleamValidationError, match=r"`simulation_duration` must be positive"):
        run_production_module(chrt, hrd, simulation_duration=[-1], show_indicator=False)


# ---- is_egg_producing: only a logical TRUE produces eggs (094) ---------------------


@pytest.mark.parametrize(
    "flag,producing",
    [(True, True), (np.True_, True), (1, False), (1.0, False), ("TRUE", False), ("T", False),
     (np.int64(1), False), (2.0, False), (None, False), (np.nan, False)],
)
def test_calc_egg_production_gates_on_istrue_without_validation(flag, producing):
    """R's calc_egg_production returns zeros unless isTRUE(is_egg_producing)."""
    from gleampy.validation._shared import validation_disabled

    with validation_disabled():
        out = calc_egg_production("CHK", "FA", 36500, 0.06, 365, is_egg_producing=flag)
        vec = calc_egg_production(
            ["CHK", "CHK"], ["FA", "FA"], 36500, 0.06, 365, is_egg_producing=np.array([flag, True], dtype=object)
        )
    assert out["egg_production_number_cohort"] == (36500.0 if producing else 0.0)
    assert vec["egg_production_number_cohort"].tolist() == [36500.0 if producing else 0.0, 36500.0]


def test_run_production_module_unvalidated_numeric_flags_give_no_eggs():
    """A 1/0 flag column is rejected with validation and gives zero eggs without (R: isTRUE(1) is FALSE)."""
    chrt, hrd = _example()
    numeric = chrt.copy()
    numeric["is_egg_producing"] = [1.0 if v is True else 0.0 for v in chrt["is_egg_producing"]]
    with pytest.raises(GleamValidationError, match=r"must be logical \(TRUE/FALSE\)"):
        run_production_module(numeric, hrd, show_indicator=False)
    with pytest.warns(UserWarning, match="validation has been turned off"):
        res = run_production_module(numeric, hrd, show_indicator=False, validate_inputs=False)
    assert (res[EGG_KEYS] == 0).all().all()
    ref = run_production_module(chrt, hrd, show_indicator=False)
    assert (ref[EGG_KEYS] > 0).any().any()


# ---- columns R reads only for the rows that need them ------------------------------


def _with_laying_fn(chrt: pd.DataFrame) -> pd.DataFrame:
    fa = chrt[(chrt["species_short"] == "CHK") & (chrt["cohort_short"] == "FA")].iloc[[0]].copy()
    fa["cohort_short"] = "FN"
    fa["nondemo_productive_phase_id"] = 2.0
    return pd.concat([chrt, fa], ignore_index=True)


@pytest.mark.parametrize("validate", [True, False])
def test_run_production_module_missing_egg_flag_with_chk_is_a_validation_error(validate):
    chrt, hrd = _example()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(GleamValidationError, match=r'^Missing required columns in `cohort_level_data`: "is_egg_producing"$'):
            run_production_module(chrt.drop(columns="is_egg_producing"), hrd, show_indicator=False,
                                  validate_inputs=validate)


@pytest.mark.parametrize("col", ["egg_output_human_consumption", "egg_average_weight"])
def test_run_production_module_missing_egg_herd_columns_for_laying_cohorts(col):
    chrt, hrd = _example()
    with pytest.raises(GleamValidationError, match=rf'^Missing required columns in `herd_level_data`: "{col}"$'):
        run_production_module(chrt, hrd.drop(columns=col), show_indicator=False)
    # not needed when no cohort lays eggs (R reads them lazily)
    no_laying = chrt.assign(is_egg_producing=False)
    res = run_production_module(no_laying, hrd.drop(columns=col), show_indicator=False)
    assert (res[EGG_KEYS] == 0).all().all()


def test_run_production_module_missing_phase_id_only_needed_for_laying_fn():
    chrt, hrd = _example()
    with_fn = _with_laying_fn(chrt)
    ref = run_production_module(with_fn, hrd, show_indicator=False)
    assert ref.loc[len(chrt), "egg_production_number_cohort"] > 0
    with pytest.raises(
        GleamValidationError, match=r'^Missing required columns in `cohort_level_data`: "nondemo_productive_phase_id"$'
    ):
        run_production_module(with_fn.drop(columns="nondemo_productive_phase_id"), hrd, show_indicator=False)
    # only laying FA cohorts: the phase id is never read, as in R
    res = run_production_module(chrt.drop(columns="nondemo_productive_phase_id"), hrd, show_indicator=False)
    full = run_production_module(chrt, hrd, show_indicator=False)
    np.testing.assert_array_equal(res[OUTPUT_COLUMNS].to_numpy(), full[OUTPUT_COLUMNS].to_numpy())


# R validates the flag placement before it forces the herd egg columns, so a
# misplaced TRUE flag is reported as such even when those columns are absent
# (messages checked with Rscript on the same inputs).
_EGG_FLAG_PLACEMENT_CASES = {
    "non_chk_true": ("CTL", "FA", None, True, r'^`is_egg_producing` can be TRUE only for "CHK"\.$'),
    "chk_ma_true": ("CHK", "MA", None, True, r'^`is_egg_producing` can be TRUE only for CHK cohorts "FA" or "FN"\.$'),
    "chk_fn_phase_1": ("CHK", "FN", 1.0, True,
                       r'^`is_egg_producing` can be TRUE for "FN" only when `nondemo_productive_phase_id` is'),
    "chk_fn_no_phase_column": ("CHK", "FN", "drop", True,
                               r'^Missing required columns in `cohort_level_data`: "nondemo_productive_phase_id"$'),
    # the cohort check comes before R's phase check, which forces the absent column
    "chk_ma_and_fn_no_phase_column": ("CHK", "MA+FN", "drop", True,
                                      r'^`is_egg_producing` can be TRUE only for CHK cohorts "FA" or "FN"\.$'),
    "chk_na": ("CHK", "ALL", None, None, r'^`is_egg_producing` must be a single non-missing logical value for "CHK"\.$'),
}


@pytest.mark.parametrize("case", list(_EGG_FLAG_PLACEMENT_CASES))
def test_run_production_module_flag_placement_errors_come_before_missing_egg_columns(case):
    species, cohorts, phase, value, message = _EGG_FLAG_PLACEMENT_CASES[case]
    chrt, hrd = _example()
    chrt = chrt.assign(is_egg_producing=pd.Series([False] * len(chrt), dtype=object))
    fa = chrt[(chrt["species_short"] == "CHK") & (chrt["cohort_short"] == "FA")].iloc[[0]].copy()
    fa["cohort_short"] = "FN"
    chrt = pd.concat([chrt, fa], ignore_index=True)  # a CHK FN cohort, not laying
    sel = chrt["species_short"].eq(species)
    if cohorts != "ALL":
        sel &= chrt["cohort_short"].isin(cohorts.split("+"))
    chrt.loc[sel, "is_egg_producing"] = value
    if phase == "drop":
        chrt = chrt.drop(columns="nondemo_productive_phase_id")
    elif phase is not None:
        chrt.loc[sel, "nondemo_productive_phase_id"] = phase
    no_egg_cols = hrd.drop(columns=["egg_output_human_consumption", "egg_average_weight"])
    with pytest.raises(GleamValidationError, match=message):
        run_production_module(chrt, no_egg_cols, show_indicator=False)
    # with the egg columns present the error is the same
    with pytest.raises(GleamValidationError, match=message):
        run_production_module(chrt, hrd, show_indicator=False)
