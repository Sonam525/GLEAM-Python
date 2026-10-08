"""Port of ``tests/testthat/test-nitrogen_balance_core.R`` plus vectorisation checks."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from gleampy import (
    GleamValidationError,
    calc_nitrogen_excretion,
    calc_nitrogen_intake,
    calc_nitrogen_retention,
    load_example,
    run_nitrogen_balance_module,
)

ALL_SPECIES = ["CTL", "BFL", "SHP", "GTS", "PGS", "CML", "CHK"]
ALL_COHORTS = ["FJ", "FS", "FA", "MJ", "MS", "MA", "FN", "MN"]


def approx(x, rel=1.5e-8, abs=None):
    return pytest.approx(x, rel=rel, abs=abs)


# ---- calc_nitrogen_intake -----------------------------------------------------


def test_calc_nitrogen_intake_produces_expected_results():
    assert calc_nitrogen_intake(10, 0.03) == approx(0.3)
    assert calc_nitrogen_intake(0, 0.03) == approx(0)
    assert calc_nitrogen_intake(5, 0) == approx(0)
    assert calc_nitrogen_intake(2.5, 0.04) == approx(0.1)
    assert calc_nitrogen_intake(8, 0.1) == approx(0.8)  # upper bound


def test_calc_nitrogen_intake_validates_ranges():
    with pytest.raises(GleamValidationError, match=r"`ration_nitrogen` = 0.2 is out of range"):
        calc_nitrogen_intake(8, 0.2)
    with pytest.raises(GleamValidationError, match=r"`ration_intake` must not contain missing values"):
        calc_nitrogen_intake(np.nan, 0.02)


# ---- calc_nitrogen_retention ----------------------------------------------------


def test_retention_for_cattle_milk_plus_growth_add_up_correctly():
    # milk_protein_fraction is kg protein/kg milk (0-1); 0.032 = 3.2%
    base = calc_nitrogen_retention(
        "CTL", "FA", milk_protein_fraction=0.032, milk_yield_day=20,
        daily_weight_gain=0, fibre_yield_year=0, litter_size=1, parturition_rate=1,
    )
    with_growth = calc_nitrogen_retention(
        "CTL", "FA", milk_protein_fraction=0.032, milk_yield_day=20,
        daily_weight_gain=0.5, fibre_yield_year=0, litter_size=1, parturition_rate=1,
    )
    assert with_growth - base == approx(0.5 * 0.0326, rel=1e-12)

    none = calc_nitrogen_retention(
        "CTL", "FA", milk_protein_fraction=0.032, milk_yield_day=0,
        daily_weight_gain=0, fibre_yield_year=0, litter_size=1, parturition_rate=1,
    )
    assert base - none == approx(20 * (0.032 / 6.25), rel=1e-12)
    assert base > 0


def test_retention_for_goats_includes_fibre_component():
    base = calc_nitrogen_retention(
        "GTS", "FA", milk_protein_fraction=np.nan, milk_yield_day=np.nan,
        daily_weight_gain=0, fibre_yield_year=0, litter_size=1, parturition_rate=1,
    )
    with_fibre = calc_nitrogen_retention(
        "GTS", "FA", milk_protein_fraction=np.nan, milk_yield_day=np.nan,
        daily_weight_gain=0, fibre_yield_year=10, litter_size=1, parturition_rate=1,
    )
    assert with_fibre - base == approx((10 / 365) * 0.134, rel=1e-12)


def test_retention_for_sheep_with_only_fibre_is_positive():
    val = calc_nitrogen_retention(
        "SHP", "FA", milk_protein_fraction=np.nan, milk_yield_day=np.nan,
        daily_weight_gain=np.nan, fibre_yield_year=20, litter_size=1, parturition_rate=1,
    )
    assert val > 0


def test_retention_for_pigs_fa_cohort_matches_reproductive_formula():
    val = calc_nitrogen_retention(
        "PGS", "FA", litter_size=10, parturition_rate=2,
        live_weight_at_weaning=30, live_weight_at_birth=1,
    )
    expected = ((0.025 * 10 * 2 * (30 - 1) / 0.98) + (0.025 * 10 * 2 * 1)) / 365
    assert val == approx(expected, rel=1e-12)


def test_retention_for_pigs_fs_cohort_matches_reproductive_formula():
    val = calc_nitrogen_retention(
        "PGS", "FS", daily_weight_gain=0.5, litter_size=12, parturition_rate=2.2,
        live_weight_at_weaning=20, live_weight_at_birth=1,
        pregnancy_duration=115, cohort_duration_days=200,
    )
    expected = 0.025 * 0.5 + (0.025 * 12 * (115 / 200) * 1 / 0.806) / 365
    assert val == approx(expected, rel=1e-12)


def test_retention_for_pigs_growers_matches_0_025_times_daily_weight_gain():
    val = calc_nitrogen_retention("PGS", "MS", daily_weight_gain=0.8)
    assert val == approx(0.025 * 0.8, rel=1e-12)


def test_retention_for_chickens_includes_growth_and_egg_deposition():
    val = calc_nitrogen_retention(
        "CHK", "FA", daily_weight_gain=0.01, parturition_rate=120, cohort_stock_size=100,
        egg_output_human_consumption=36500, egg_average_weight=0.06, is_egg_producing=True,
    )
    egg_mass = ((36500 / 365 / 100) + (120 / 365)) * 0.06
    expected = 0.01 * 0.032 + egg_mass * 0.02
    assert val == approx(expected, rel=1e-12)


def test_retention_for_egg_producing_chicken_fn_includes_egg_deposition():
    val = calc_nitrogen_retention(
        "CHK", "FN", nondemo_productive_phase_id=2, daily_weight_gain=0.01,
        parturition_rate=120, cohort_stock_size=100, egg_output_human_consumption=36500,
        egg_average_weight=0.06, is_egg_producing=True,
    )
    egg_mass = ((36500 / 365 / 100) + (120 / 365)) * 0.06
    expected = 0.01 * 0.032 + egg_mass * 0.02
    assert val == approx(expected, rel=1e-12)


def test_retention_for_non_laying_chicken_is_growth_only():
    val = calc_nitrogen_retention(
        "CHK", "FS", daily_weight_gain=0.01, parturition_rate=120, cohort_stock_size=100,
        egg_output_human_consumption=36500, egg_average_weight=0.06, is_egg_producing=False,
    )
    assert val == approx(0.01 * 0.032)


def test_retention_validation_follows_r_rules():
    with pytest.raises(GleamValidationError, match="must be strictly less than"):
        calc_nitrogen_retention("CTL", "FA", daily_weight_gain=0.1,
                                live_weight_at_weaning=30, live_weight_at_birth=40)
    with pytest.raises(GleamValidationError, match=r"`pregnancy_duration` must be positive"):
        calc_nitrogen_retention("PGS", "FS", daily_weight_gain=0.5, litter_size=12, parturition_rate=2,
                                live_weight_at_weaning=20, live_weight_at_birth=1,
                                pregnancy_duration=0, cohort_duration_days=200)
    with pytest.raises(GleamValidationError, match=r"`litter_size` must not contain missing values"):
        calc_nitrogen_retention("PGS", "FA", parturition_rate=2,
                                live_weight_at_weaning=20, live_weight_at_birth=1)
    with pytest.raises(GleamValidationError, match=r"`daily_weight_gain` must be a single numeric value"):
        calc_nitrogen_retention("CHK", "FS", is_egg_producing=False)
    with pytest.raises(GleamValidationError, match=r"can be TRUE only for \"CHK\""):
        calc_nitrogen_retention("CTL", "FA", daily_weight_gain=0.1, is_egg_producing=True)
    with pytest.raises(GleamValidationError, match=r"`egg_average_weight` must be positive"):
        calc_nitrogen_retention("CHK", "FA", daily_weight_gain=0.01, parturition_rate=120,
                                cohort_stock_size=100, egg_output_human_consumption=36500,
                                egg_average_weight=0, is_egg_producing=True)
    with pytest.raises(GleamValidationError, match=r"`species_short` must be one of"):
        calc_nitrogen_retention("XXX", "FA")
    # NA inputs are skipped for milk producers; birth/weaning check ignored for CHK
    assert calc_nitrogen_retention("CTL", "MJ") == 0
    calc_nitrogen_retention("CHK", "FS", daily_weight_gain=0.0, live_weight_at_weaning=0.04,
                            live_weight_at_birth=0.04, is_egg_producing=False)


# Reference values computed with the R package (calc_nitrogen_retention, written
# with sprintf("%a")) for branches the golden example data does not cover:
# non-demographic cohorts of milk-producing species and laying CHK FN.
# Common inputs: litter_size=1.5, parturition_rate=0.75 (45 for CHK),
# live weights 100/25 (NA for CHK), pregnancy_duration=150,
# cohort_duration_days=180, cohort_stock_size=1000,
# egg_output_human_consumption=180000, egg_average_weight=0.0625.
R_RETENTION_REFERENCE = [
    # species, cohort, milk_protein_fraction, milk_yield_day, daily_weight_gain, fibre_yield_year, phase, egg, R value
    ("CTL", "FN", 0.0325, 20, 0.75, 0, 1, False, "0x1.9096bb98c7e28p-6"),
    ("BFL", "MN", 0.04, 5, 0.5, 0, 2, False, "0x1.0b0f27bb2fec5p-6"),
    ("SHP", "FN", 0.055, 0.5, 0.125, 2.5, 1, False, "0x1.112437ff53a1ap-8"),
    ("GTS", "MN", 0.035, 1.25, 0.0625, 0.75, 2, False, "0x1.f229d2f768dcfp-10"),
    ("CML", "FN", 0.035, 4, 0.25, 1, 2, False, "0x1.c20b35f02a76bp-8"),
    ("SHP", "MJ", 0.055, 0.5, 0.125, 2.5, np.nan, False, "0x1.a9fbe76c8b439p-9"),
    ("CML", "FA", 0.035, 4, 0.25, 1, np.nan, False, "0x1.df833657964a3p-6"),
    ("PGS", "FN", np.nan, np.nan, 0.5, np.nan, 1, False, "0x1.999999999999ap-7"),
    ("CHK", "FN", np.nan, np.nan, 0.0125, np.nan, 2, True, "0x1.32da24927dacap-10"),
    ("CHK", "FN", np.nan, np.nan, 0.0125, np.nan, 1, False, "0x1.a36e2eb1c432dp-12"),
    ("CHK", "MN", np.nan, np.nan, 0.025, np.nan, 1, False, "0x1.a36e2eb1c432dp-11"),
]


def _reference_kwargs(sp, co, mpf, myd, dwg, fy, ph, egg):
    chk = sp == "CHK"
    return dict(
        species_short=sp, cohort_short=co, milk_protein_fraction=mpf, milk_yield_day=myd,
        daily_weight_gain=dwg, fibre_yield_year=fy, litter_size=1.5,
        parturition_rate=45 if chk else 0.75,
        live_weight_at_weaning=np.nan if chk else 100, live_weight_at_birth=np.nan if chk else 25,
        pregnancy_duration=150, cohort_duration_days=180, cohort_stock_size=1000,
        egg_output_human_consumption=180000, egg_average_weight=0.0625,
        nondemo_productive_phase_id=ph, is_egg_producing=egg,
    )


@pytest.mark.parametrize("case", R_RETENTION_REFERENCE, ids=lambda c: f"{c[0]}-{c[1]}-{c[6]}")
def test_retention_matches_r_reference_for_nondemographic_branches(case):
    *inputs, expected = case
    # R parses decimal literals with platform-dependent last-bit rounding, hence rel=1e-12.
    assert calc_nitrogen_retention(**_reference_kwargs(*inputs)) == approx(float.fromhex(expected), rel=1e-12)


def test_retention_matches_r_reference_vectorised():
    kwargs = [_reference_kwargs(*c[:-1]) for c in R_RETENTION_REFERENCE]
    vec = calc_nitrogen_retention(**{k: [kw[k] for kw in kwargs] for k in kwargs[0]})
    np.testing.assert_allclose(vec, [float.fromhex(c[-1]) for c in R_RETENTION_REFERENCE], rtol=1e-12, atol=0)


# ---- calc_nitrogen_excretion ----------------------------------------------------


def test_excretion_subtracts_intake_and_retention():
    assert calc_nitrogen_excretion("CTL", 0.5, 0.2) == approx(0.3)
    assert calc_nitrogen_excretion("PGS", 0.4, 0.1) == approx(0.3)


def test_excretion_handles_zero_retention_and_errors_when_intake_lt_retention():
    assert calc_nitrogen_excretion("CTL", 0.5, 0) == approx(0.5)
    with pytest.raises(
        GleamValidationError,
        match="nitrogen_intake.*must be greater than or equal to.*nitrogen_retention",
    ):
        calc_nitrogen_excretion("CTL", 0, 0.2)


# ---- vectorisation ----------------------------------------------------------------


def _mixed_retention_inputs() -> pd.DataFrame:
    """One row per species x cohort (plus a laying CHK FN phase-2 row) with valid inputs."""
    rng = np.random.default_rng(7)
    rows = []
    for sp, co in itertools.product(ALL_SPECIES, ALL_COHORTS):
        phases = [1, 2] if co in ("FN", "MN") else [np.nan]
        for ph in phases:
            egg = None
            if sp == "CHK":
                egg = (co == "FA") or (co == "FN" and ph == 2)
            elif rng.random() < 0.5:
                egg = False
            lwb = rng.uniform(0.5, 40)
            rows.append({
                "species_short": sp,
                "cohort_short": co,
                "milk_protein_fraction": rng.uniform(0.02, 0.06),
                "milk_yield_day": rng.choice([np.nan, 0.0, rng.uniform(0.2, 30)]),
                "daily_weight_gain": rng.choice([0.0, rng.uniform(0.005, 1.2)]),
                "fibre_yield_year": rng.choice([np.nan, 0.0, rng.uniform(0.1, 3)]),
                "litter_size": rng.uniform(1, 13),
                "parturition_rate": rng.uniform(0.4, 2.5) if sp != "CHK" else rng.uniform(10, 150),
                "live_weight_at_weaning": lwb + rng.uniform(1, 200),
                "live_weight_at_birth": lwb,
                "pregnancy_duration": rng.uniform(110, 390),
                "cohort_duration_days": rng.uniform(3, 3000),
                "cohort_stock_size": rng.uniform(10, 1e6),
                "egg_output_human_consumption": rng.uniform(0, 1e7),
                "egg_average_weight": rng.uniform(0.04, 0.07),
                "nondemo_productive_phase_id": ph,
                "is_egg_producing": egg,
            })
    df = pd.DataFrame(rows)
    df["is_egg_producing"] = df["is_egg_producing"].astype(object)
    return df


def test_vectorised_retention_matches_elementwise_scalar_calls():
    df = _mixed_retention_inputs()
    args = list(df.columns)
    vec = calc_nitrogen_retention(**{a: df[a] for a in args})
    scal = np.array([
        calc_nitrogen_retention(**row)
        for row in df.to_dict("records")
    ])
    assert isinstance(vec, np.ndarray) and vec.shape == (len(df),)
    np.testing.assert_array_equal(vec, scal)
    # numpy array inputs give the same result as Series
    vec_np = calc_nitrogen_retention(**{a: df[a].to_numpy() for a in args})
    np.testing.assert_array_equal(vec_np, scal)


def test_vectorised_intake_and_excretion_match_elementwise_scalar_calls():
    df = _mixed_retention_inputs()
    rng = np.random.default_rng(3)
    dmi = rng.uniform(0.05, 20, len(df))
    n = rng.uniform(0.005, 0.05, len(df))
    intake = calc_nitrogen_intake(dmi, n)
    np.testing.assert_array_equal(intake, [calc_nitrogen_intake(a, b) for a, b in zip(dmi, n)])

    retention = calc_nitrogen_retention(**{a: df[a] for a in df.columns})
    intake_big = intake + retention  # guarantees intake >= retention
    excr = calc_nitrogen_excretion(df["species_short"], intake_big, retention)
    np.testing.assert_array_equal(
        excr,
        [calc_nitrogen_excretion(s, i, r) for s, i, r in zip(df["species_short"], intake_big, retention)],
    )


def test_vectorised_excretion_validation_reports_failure():
    with pytest.raises(GleamValidationError, match="must be greater than or equal to"):
        calc_nitrogen_excretion(["CTL", "PGS"], [0.5, 0.1], [0.2, 0.3])


# ---- run module -----------------------------------------------------------------


def _example():
    return (
        load_example("nitrogen_balance_input_chrt_data.csv"),
        load_example("nitrogen_balance_input_hrd_data.csv"),
    )


def test_run_nitrogen_balance_module_keeps_inputs_and_adds_columns():
    chrt, hrd = _example()
    c0, h0 = chrt.copy(), hrd.copy()
    res = run_nitrogen_balance_module(chrt, hrd, show_indicator=False)
    pd.testing.assert_frame_equal(chrt, c0)
    pd.testing.assert_frame_equal(hrd, h0)
    assert list(res.columns) == list(chrt.columns) + [
        "nitrogen_intake", "nitrogen_retention", "nitrogen_excretion",
    ]
    np.testing.assert_allclose(
        res["nitrogen_excretion"], res["nitrogen_intake"] - res["nitrogen_retention"], rtol=0, atol=0
    )


def test_run_nitrogen_balance_module_adds_is_egg_producing_without_chk():
    chrt, hrd = _example()
    keep = hrd["species_short"] != "CHK"
    hrd = hrd[keep].drop(columns=["egg_output_human_consumption", "egg_average_weight"])
    chrt = chrt[chrt["herd_id"].isin(hrd["herd_id"])].drop(
        columns=["is_egg_producing", "nondemo_productive_phase_id"]
    )
    chrt = chrt[~chrt["cohort_short"].isin(["FN", "MN"])]
    res = run_nitrogen_balance_module(chrt, hrd, show_indicator=False)
    assert "is_egg_producing" not in chrt.columns  # caller's table untouched
    assert list(res.columns) == list(chrt.columns) + [
        "is_egg_producing", "nitrogen_intake", "nitrogen_retention", "nitrogen_excretion",
    ]
    assert res["is_egg_producing"].isna().all()


def test_run_nitrogen_balance_module_validates_inputs():
    chrt, hrd = _example()
    with pytest.raises(GleamValidationError, match="Missing required columns in `herd_level_data`"):
        run_nitrogen_balance_module(chrt, hrd.drop(columns="egg_average_weight"), show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `herd_level_data`"):
        run_nitrogen_balance_module(chrt, hrd.drop(columns="litter_size"), show_indicator=False)
    with pytest.raises(GleamValidationError, match="not found in `herd_level_data`"):
        run_nitrogen_balance_module(chrt, hrd[hrd["herd_id"] != 1], show_indicator=False)
    with pytest.raises(GleamValidationError, match="Each herd_id must appear exactly once"):
        run_nitrogen_balance_module(chrt, pd.concat([hrd, hrd.iloc[:1]]), show_indicator=False)


def test_run_nitrogen_balance_module_without_validation_warns_and_matches():
    chrt, hrd = _example()
    ref = run_nitrogen_balance_module(chrt, hrd, show_indicator=False)
    with pytest.warns(UserWarning, match="Input validation has been turned off"):
        res = run_nitrogen_balance_module(chrt, hrd, show_indicator=False, validate_inputs=False)
    pd.testing.assert_frame_equal(res, ref)


def test_retention_na_handling_follows_r():
    # A component is 0 when its driver is missing or not positive,
    # but a missing milk_protein_fraction with positive milk yield gives NA in
    # R: calc_nitrogen_retention("CTL", "FA", milk_protein_fraction = NA_real_,
    # milk_yield_day = 20, daily_weight_gain = 0) is NA.
    assert np.isnan(calc_nitrogen_retention("CTL", "FA", milk_protein_fraction=np.nan, milk_yield_day=20,
                                            daily_weight_gain=0))
    assert calc_nitrogen_retention("CTL", "FA", milk_protein_fraction=np.nan, milk_yield_day=0,
                                   daily_weight_gain=0.5) == approx(0.5 * 0.0326)
    assert calc_nitrogen_retention("CTL", "MA", milk_protein_fraction=np.nan, milk_yield_day=20,
                                   daily_weight_gain=0.5) == approx(0.5 * 0.0326)
    assert calc_nitrogen_retention("CTL", "FA", milk_yield_day=np.nan, daily_weight_gain=np.nan) == 0.0
    assert calc_nitrogen_retention("SHP", "FN", 0.05, 0.0, 0.1, fibre_yield_year=np.nan) == approx(0.1 * 0.026)
