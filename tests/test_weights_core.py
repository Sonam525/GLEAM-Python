"""Port of ``tests/testthat/test-weights_core.R`` plus vectorisation checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gleampy
from gleampy import (
    GleamValidationError,
    calc_avg_weights,
    calc_cohort_weights,
    calc_daily_weight_gain,
    validation_disabled,
)

REL = 1.5e-8  # testthat tolerance


# ---- test calc_cohort_weights ----


def test_calc_cohort_weights_returns_valid_weights_for_juvenile_non_pig():
    result = calc_cohort_weights(
        cohort_short="FJ",
        live_weight_female_adult=500, live_weight_male_adult=600,
        live_weight_at_birth=35, live_weight_female_at_slaughter=480,
        live_weight_male_at_slaughter=550, live_weight_at_weaning=90,
    )
    assert list(result) == [
        "live_weight_mature_stage",
        "live_weight_cohort_initial",
        "live_weight_cohort_potential_final",
        "live_weight_cohort_at_slaughter",
    ]
    assert isinstance(result["live_weight_cohort_initial"], float)
    assert result["live_weight_cohort_initial"] == 35
    assert result["live_weight_cohort_potential_final"] > result["live_weight_cohort_initial"]
    assert result["live_weight_cohort_potential_final"] == pytest.approx(
        result["live_weight_cohort_at_slaughter"], rel=REL
    )


def test_calc_cohort_weights_returns_correct_weights_for_adult_female():
    result = calc_cohort_weights(
        cohort_short="FA",
        live_weight_female_adult=70, live_weight_male_adult=90,
        live_weight_at_birth=4, live_weight_female_at_slaughter=65,
        live_weight_male_at_slaughter=85, live_weight_at_weaning=18,
    )
    assert result["live_weight_cohort_initial"] == pytest.approx(70, rel=REL)
    assert result["live_weight_cohort_potential_final"] == pytest.approx(70, rel=REL)
    assert result["live_weight_cohort_at_slaughter"] == pytest.approx(70, rel=REL)


def test_calc_cohort_weights_handles_pig_juvenile_with_weaning_weight():
    result = calc_cohort_weights(
        cohort_short="FJ",
        live_weight_female_adult=180, live_weight_male_adult=220,
        live_weight_at_birth=1.5, live_weight_female_at_slaughter=160,
        live_weight_male_at_slaughter=200, live_weight_at_weaning=10,
    )
    assert result["live_weight_cohort_initial"] == pytest.approx(1.5, rel=REL)
    assert result["live_weight_cohort_potential_final"] == pytest.approx(10, rel=REL)
    assert result["live_weight_cohort_at_slaughter"] == pytest.approx(10, rel=REL)


def test_calc_cohort_weights_uses_hatch_weight_as_weaning_weight_for_chickens():
    result = calc_cohort_weights(
        species_short="CHK",
        cohort_short="FJ",
        live_weight_female_adult=2.2, live_weight_male_adult=2.8,
        live_weight_at_birth=0.04, live_weight_female_at_slaughter=1.8,
        live_weight_male_at_slaughter=2.1, live_weight_at_weaning=0.2,
    )
    assert result["live_weight_cohort_initial"] == pytest.approx(0.04, rel=REL)
    assert result["live_weight_cohort_potential_final"] == pytest.approx(0.04, rel=REL)
    assert result["live_weight_cohort_at_slaughter"] == pytest.approx(0.04, rel=REL)


def test_calc_cohort_weights_interpolates_non_demo_female_weights_across_two_phases():
    common = dict(
        cohort_short="FN",
        live_weight_female_nondemographic_start=20,
        live_weight_female_nondemographic_end=100,
        phase1_nondemo_fem_duration_days=30,
        phase2_nondemo_fem_duration_days=50,
    )
    phase1 = calc_cohort_weights(nondemo_productive_phase_id=1, **common)
    phase2 = calc_cohort_weights(nondemo_productive_phase_id=2, **common)

    assert phase1["live_weight_cohort_initial"] == pytest.approx(20, rel=REL)
    assert phase1["live_weight_cohort_potential_final"] == pytest.approx(50, rel=REL)
    assert phase2["live_weight_cohort_initial"] == pytest.approx(50, rel=REL)
    assert phase2["live_weight_cohort_potential_final"] == pytest.approx(100, rel=REL)


def test_calc_cohort_weights_uses_full_non_demo_range_when_only_phase_1_is_present():
    phase1 = calc_cohort_weights(
        cohort_short="MN",
        nondemo_productive_phase_id=1,
        live_weight_male_nondemographic_start=30,
        live_weight_male_nondemographic_end=90,
        phase1_nondemo_mal_duration_days=20,
        phase2_nondemo_mal_duration_days=0,
    )
    assert phase1["live_weight_cohort_initial"] == pytest.approx(30, rel=REL)
    assert phase1["live_weight_cohort_potential_final"] == pytest.approx(90, rel=REL)
    assert phase1["live_weight_cohort_at_slaughter"] == pytest.approx(90, rel=REL)


def test_calc_cohort_weights_allows_na_inputs_for_the_unused_non_demo_sex():
    mn = calc_cohort_weights(
        cohort_short="MN",
        nondemo_productive_phase_id=1,
        live_weight_female_nondemographic_start=np.nan,
        live_weight_female_nondemographic_end=np.nan,
        live_weight_male_nondemographic_start=30,
        live_weight_male_nondemographic_end=90,
        phase1_nondemo_fem_duration_days=np.nan,
        phase2_nondemo_fem_duration_days=np.nan,
        phase1_nondemo_mal_duration_days=20,
        phase2_nondemo_mal_duration_days=0,
    )
    fn = calc_cohort_weights(
        cohort_short="FN",
        nondemo_productive_phase_id=1,
        live_weight_female_nondemographic_start=20,
        live_weight_female_nondemographic_end=100,
        live_weight_male_nondemographic_start=np.nan,
        live_weight_male_nondemographic_end=np.nan,
        phase1_nondemo_fem_duration_days=30,
        phase2_nondemo_fem_duration_days=50,
        phase1_nondemo_mal_duration_days=np.nan,
        phase2_nondemo_mal_duration_days=np.nan,
    )
    assert mn["live_weight_cohort_initial"] == pytest.approx(30, rel=REL)
    assert fn["live_weight_cohort_initial"] == pytest.approx(20, rel=REL)


# ---- test calc_avg_weights ----


def test_calc_avg_weights_returns_correct_average_and_final_weights():
    result = calc_avg_weights(
        cohort_short="FS",
        live_weight_cohort_initial=100,
        live_weight_cohort_potential_final=300,
        live_weight_cohort_at_slaughter=200,
        offtake_rate=0.4,
    )
    assert result["live_weight_cohort_final"] == pytest.approx(260, rel=REL)
    assert result["live_weight_cohort_average"] == pytest.approx(180, rel=REL)


def test_calc_avg_weights_handles_zero_offtake():
    result = calc_avg_weights("FS", 100, 300, 200, 0)
    assert result["live_weight_cohort_final"] == pytest.approx(300, rel=REL)
    assert result["live_weight_cohort_average"] == pytest.approx(200, rel=REL)


def test_calc_avg_weights_ignores_offtake_for_fn_and_mn():
    result_fn = calc_avg_weights("FN", 30, 80, 60, 0.9)
    result_mn = calc_avg_weights("MN", 35, 90, 55, 0.8)
    assert result_fn["live_weight_cohort_final"] == pytest.approx(80, rel=REL)
    assert result_fn["live_weight_cohort_average"] == pytest.approx(55, rel=REL)
    assert result_mn["live_weight_cohort_final"] == pytest.approx(90, rel=REL)
    assert result_mn["live_weight_cohort_average"] == pytest.approx(62.5, rel=REL)


def test_calc_avg_weights_allows_offtake_rate_equal_to_1_for_fn_and_mn():
    calc_avg_weights("FN", 30, 80, 60, 1)
    calc_avg_weights("MN", 35, 90, 55, 1)


# ---- test calc_daily_weight_gain ----


def test_calc_daily_weight_gain_computes_correct_value():
    assert calc_daily_weight_gain(300, 100, 100) == pytest.approx(2, rel=REL)


def test_calc_daily_weight_gain_returns_0_when_weights_are_equal():
    assert calc_daily_weight_gain(200, 200, 100) == pytest.approx(0, rel=REL)


def test_calc_daily_weight_gain_handles_negative_gain():
    assert calc_daily_weight_gain(100, 200, 100) == pytest.approx(-1, rel=REL)


# ---- test cohort-specific validation ----


def test_calc_cohort_weights_rejects_missing_live_weight_female_adult_for_fj():
    with pytest.raises(GleamValidationError, match="Missing required weight inputs"):
        calc_cohort_weights(
            cohort_short="FJ",
            live_weight_female_adult=np.nan,
            live_weight_male_adult=600,
            live_weight_at_birth=35,
            live_weight_female_at_slaughter=480,
            live_weight_male_at_slaughter=550,
            live_weight_at_weaning=90,
        )


def test_calc_cohort_weights_rejects_missing_live_weight_female_at_slaughter_for_fs():
    with pytest.raises(GleamValidationError, match="Missing required weight inputs"):
        calc_cohort_weights(
            cohort_short="FS",
            live_weight_female_adult=500,
            live_weight_male_adult=600,
            live_weight_at_birth=35,
            live_weight_female_at_slaughter=np.nan,
            live_weight_male_at_slaughter=550,
            live_weight_at_weaning=90,
        )


def test_calc_cohort_weights_rejects_missing_live_weight_male_adult_for_ma():
    with pytest.raises(GleamValidationError, match="Missing required weight inputs"):
        calc_cohort_weights(
            cohort_short="MA",
            live_weight_female_adult=500,
            live_weight_male_adult=np.nan,
            live_weight_at_birth=35,
            live_weight_female_at_slaughter=480,
            live_weight_male_at_slaughter=550,
            live_weight_at_weaning=90,
        )


# ---- additional checks (Python port) ----


def test_public_names_are_registered():
    assert gleampy.calc_cohort_weights is calc_cohort_weights
    assert gleampy.calc_avg_weights is calc_avg_weights
    assert gleampy.calc_daily_weight_gain is calc_daily_weight_gain


def test_missing_required_message_lists_inputs_like_r():
    with pytest.raises(GleamValidationError) as err:
        calc_cohort_weights(
            cohort_short="FJ", live_weight_female_adult=np.nan, live_weight_at_birth=np.nan,
            live_weight_at_weaning=3,
        )
    assert str(err.value) == (
        'Missing required weight inputs for cohort "FJ": "live_weight_at_birth" and '
        '"live_weight_female_adult"'
    )


def test_calc_cohort_weights_validation_messages():
    with pytest.raises(GleamValidationError, match="`cohort_short` must be one of"):
        calc_cohort_weights(cohort_short="XX", live_weight_female_adult=1)
    with pytest.raises(GleamValidationError, match="`species_short` must be one of"):
        calc_cohort_weights(species_short="DOG", cohort_short="FA", live_weight_female_adult=1)
    with pytest.raises(GleamValidationError, match="must be a single numeric \\(scalar\\). NA is allowed"):
        calc_cohort_weights(cohort_short="FA", live_weight_female_adult="500")
    with pytest.raises(GleamValidationError, match="strictly less than `live_weight_at_weaning`"):
        calc_cohort_weights(
            cohort_short="FJ", live_weight_female_adult=500, live_weight_at_birth=90,
            live_weight_at_weaning=90,
        )
    # chickens are exempt from the birth < weaning rule
    calc_cohort_weights(
        species_short="CHK", cohort_short="FJ", live_weight_female_adult=2,
        live_weight_at_birth=0.04, live_weight_at_weaning=0.04,
    )
    with pytest.raises(GleamValidationError, match="`live_weight_at_birth` = 1001 is out of range"):
        calc_cohort_weights(
            cohort_short="FJ", live_weight_female_adult=500, live_weight_at_birth=1001,
            live_weight_at_weaning=1002,
        )


def test_calc_avg_and_daily_gain_validation_messages():
    with pytest.raises(GleamValidationError, match="`offtake_rate` = 1 is out of range"):
        calc_avg_weights("FS", 1, 2, 3, 1)
    with pytest.raises(GleamValidationError, match="`offtake_rate` must be a single numeric value"):
        calc_avg_weights("FN", 1, 2, 3, np.nan)
    with pytest.raises(GleamValidationError, match="`cohort_duration_days` = 0 is out of range"):
        calc_daily_weight_gain(1, 2, 0)


def test_vector_validation_reports_offending_element():
    with pytest.raises(GleamValidationError, match=r"`offtake_rate`\[2\] = 1 is out of range"):
        calc_avg_weights(["FS", "MS", "FN"], 1, 2, 3, [0.5, 1, 1])
    with pytest.raises(GleamValidationError, match='Missing required weight inputs for cohort "MA"'):
        calc_cohort_weights(
            cohort_short=["FA", "MA"], live_weight_female_adult=[500, 500],
            live_weight_male_adult=[600, np.nan],
        )
    with pytest.raises(GleamValidationError, match="same length"):
        calc_daily_weight_gain([1, 2, 3], [1, 2], 10)


def test_undefined_cohort_or_phase_errors_like_r_even_without_validation():
    # R fails with "object 'live_weight_mature_stage' not found" for a phase
    # that is neither 1 nor 2 (1.5 is inside the parameter range [1, 2]).
    with pytest.raises(GleamValidationError, match="undefined"):
        calc_cohort_weights(
            cohort_short="FN", nondemo_productive_phase_id=1.5,
            live_weight_female_nondemographic_start=1, live_weight_female_nondemographic_end=2,
            phase1_nondemo_fem_duration_days=1, phase2_nondemo_fem_duration_days=1,
        )
    with validation_disabled(), pytest.raises(GleamValidationError, match="undefined"):
        calc_cohort_weights(cohort_short="XX", live_weight_female_adult=1)


def test_validation_can_be_disabled():
    with validation_disabled():
        res = calc_cohort_weights(cohort_short="FJ", live_weight_at_birth=35, live_weight_at_weaning=90)
        assert np.isnan(res["live_weight_mature_stage"])
        assert res["live_weight_cohort_initial"] == 35
        avg = calc_avg_weights("FS", 100, 300, 200, 1.5)  # offtake clamped to 1
        assert avg["live_weight_cohort_final"] == pytest.approx(200, rel=REL)


# ---- vectorised == element-by-element ----

_MIXED = pd.DataFrame(
    [
        # species, cohort, phase, fad, mad, birth, fsl, msl, wean, fst, fen, mst, men, p1f, p2f, p1m, p2m
        ("CTL", "FJ", np.nan, 680, 916, 41, 557, 605, 250, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("CTL", "MJ", np.nan, 680, 916, 41, 557, 605, 250, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("BFL", "FS", np.nan, 600, 800, 38, 420, 420, 130, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("SHP", "MS", np.nan, 51, 59, 4.19, 35, 35, 30, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("GTS", "FA", np.nan, 70, 110, 3.5, 25, 25, 15, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("CML", "MA", np.nan, 352, 382, 28.7, 352, 382, 120, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("PGS", "FN", 1, 225, 265, 1.2, 122, 122, 7, 7, 120, 7, 120, 60, 110, 60, 100),
        ("PGS", "FN", 2, 225, 265, 1.2, 122, 122, 7, 7, 120, 7, 120, 60, 110, 60, 100),
        ("PGS", "MN", 1, 64, 71, 1, 60, 60, 6, 6, 60, 6, 60, 90, 180, 90, 180),
        ("CTL", "MN", 2, 680, 916, 41, 557, 605, 250, np.nan, np.nan, 250, 700, np.nan, np.nan, 60, 120),
        ("CHK", "FJ", np.nan, 1.8, 2.4, 0.04, 1.6, 2.1, 0.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        ("CHK", "MS", np.nan, 1.8, 2.4, 0.04, 1.6, 2.1, 0.04, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        (None, "MJ", np.nan, 350, 450, 14, 250, 250, 220, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
    ],
    columns=[
        "species_short", "cohort_short", "nondemo_productive_phase_id",
        "live_weight_female_adult", "live_weight_male_adult", "live_weight_at_birth",
        "live_weight_female_at_slaughter", "live_weight_male_at_slaughter", "live_weight_at_weaning",
        "live_weight_female_nondemographic_start", "live_weight_female_nondemographic_end",
        "live_weight_male_nondemographic_start", "live_weight_male_nondemographic_end",
        "phase1_nondemo_fem_duration_days", "phase2_nondemo_fem_duration_days",
        "phase1_nondemo_mal_duration_days", "phase2_nondemo_mal_duration_days",
    ],
)


def _assert_same(vec: dict, rows: list[dict]) -> None:
    for key, values in vec.items():
        assert isinstance(values, np.ndarray) and values.shape == (len(rows),)
        scal = np.array([r[key] for r in rows], dtype=float)
        np.testing.assert_array_equal(values, scal, err_msg=key)


def test_vectorised_matches_elementwise_calls():
    cols = {c: _MIXED[c] for c in _MIXED.columns}
    vec = calc_cohort_weights(**cols)
    rows = [
        calc_cohort_weights(**{c: (None if v is None else v) for c, v in row.items()})
        for row in _MIXED.to_dict("records")
    ]
    assert all(isinstance(v, float) for r in rows for v in r.values())
    _assert_same(vec, rows)

    offtake = np.linspace(0, 0.9, len(_MIXED))
    avg_vec = calc_avg_weights(
        _MIXED["cohort_short"], vec["live_weight_cohort_initial"],
        vec["live_weight_cohort_potential_final"], vec["live_weight_cohort_at_slaughter"], offtake,
    )
    avg_rows = [
        calc_avg_weights(
            _MIXED["cohort_short"][i], float(vec["live_weight_cohort_initial"][i]),
            float(vec["live_weight_cohort_potential_final"][i]),
            float(vec["live_weight_cohort_at_slaughter"][i]), float(offtake[i]),
        )
        for i in range(len(_MIXED))
    ]
    _assert_same(avg_vec, avg_rows)

    duration = np.arange(1, len(_MIXED) + 1) * 30.0
    gain_vec = calc_daily_weight_gain(
        vec["live_weight_cohort_potential_final"], vec["live_weight_cohort_initial"], duration
    )
    gain_rows = [
        calc_daily_weight_gain(
            float(vec["live_weight_cohort_potential_final"][i]),
            float(vec["live_weight_cohort_initial"][i]), float(duration[i]),
        )
        for i in range(len(_MIXED))
    ]
    _assert_same({"gain": gain_vec}, [{"gain": g} for g in gain_rows])


def test_scalar_and_list_inputs_broadcast():
    res = calc_cohort_weights(
        species_short="CTL", cohort_short=["FA", "MA"],
        live_weight_female_adult=500, live_weight_male_adult=600,
    )
    np.testing.assert_array_equal(res["live_weight_cohort_initial"], [500.0, 600.0])
