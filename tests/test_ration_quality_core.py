"""Port of ``tests/testthat/test-ration_quality_core.R`` plus vectorisation checks."""

from __future__ import annotations

import numpy as np
import pytest

import gleampy
from gleampy import (
    GleamValidationError,
    calc_feed_digestibility_fraction,
    calc_ration_ash,
    calc_ration_digestibility,
    calc_ration_gross_energy,
    calc_ration_metabolizable_energy,
    calc_ration_nitrogen_content,
    calc_ration_urinary_energy_fraction,
    validation_disabled,
)

REL = 1.5e-8  # testthat tolerance


# ---- calc_ration_digestibility ----


def test_calc_ration_digestibility_selects_ruminant_digestibility():
    value = calc_ration_digestibility(
        species_short="CTL",
        feed_ration_fraction=0.6,
        feed_digestibility_fraction_ruminant=0.7,
        feed_digestibility_fraction_pigs=0.5,
    )
    assert value == pytest.approx(0.42, rel=REL)


def test_calc_ration_digestibility_selects_pig_digestibility():
    value = calc_ration_digestibility(
        species_short="PGS",
        feed_ration_fraction=0.6,
        feed_digestibility_fraction_ruminant=0.7,
        feed_digestibility_fraction_pigs=0.5,
    )
    assert value == pytest.approx(0.3, rel=REL)


# ---- calc_ration_metabolizable_energy ----


def test_calc_ration_metabolizable_energy_selects_ruminant_me():
    value = calc_ration_metabolizable_energy(
        species_short="CTL",
        feed_ration_fraction=0.6,
        feed_metabolizable_energy_ruminant=10,
        feed_metabolizable_energy_pigs=12,
    )
    assert value == pytest.approx(6, rel=REL)


def test_calc_ration_metabolizable_energy_selects_pig_me():
    value = calc_ration_metabolizable_energy(
        species_short="PGS",
        feed_ration_fraction=0.6,
        feed_metabolizable_energy_ruminant=10,
        feed_metabolizable_energy_pigs=12,
    )
    assert value == pytest.approx(7.2, rel=REL)


def test_calc_ration_metabolizable_energy_rejects_invalid_species_short():
    with pytest.raises(GleamValidationError, match="`species_short` must be one of"):
        calc_ration_metabolizable_energy(
            species_short="DOG",
            feed_ration_fraction=0.6,
            feed_metabolizable_energy_ruminant=10,
            feed_metabolizable_energy_pigs=12,
        )


def test_calc_ration_digestibility_rejects_na_inputs():
    with pytest.raises(GleamValidationError, match="must be a single numeric value"):
        calc_ration_digestibility(
            species_short="CTL",
            feed_ration_fraction=np.nan,
            feed_digestibility_fraction_ruminant=0.7,
            feed_digestibility_fraction_pigs=0.5,
        )


def test_calc_ration_metabolizable_energy_rejects_na_inputs():
    with pytest.raises(GleamValidationError, match="Missing required metabolizable energy inputs"):
        calc_ration_metabolizable_energy(
            species_short="CTL",
            feed_ration_fraction=0.6,
            feed_metabolizable_energy_ruminant=np.nan,
            feed_metabolizable_energy_pigs=12,
        )


def test_calc_ration_digestibility_allows_na_for_unused_animals():
    value = calc_ration_digestibility(
        species_short="PGS",
        feed_ration_fraction=0.6,
        feed_digestibility_fraction_ruminant=np.nan,
        feed_digestibility_fraction_pigs=0.5,
    )
    assert value == pytest.approx(0.3, rel=REL)


def test_calc_ration_digestibility_rejects_missing_required_inputs_by_species_short():
    with pytest.raises(GleamValidationError, match="Missing required digestibility inputs"):
        calc_ration_digestibility(
            species_short="PGS",
            feed_ration_fraction=0.6,
            feed_digestibility_fraction_ruminant=0.7,
            feed_digestibility_fraction_pigs=np.nan,
        )


# ---- calc_feed_digestibility_fraction ----


def test_calc_feed_digestibility_fraction_computes_ratios():
    results = calc_feed_digestibility_fraction(
        feed_digestible_energy_ruminant=8,
        feed_digestible_energy_pigs=7,
        feed_gross_energy=16,
    )
    assert list(results) == ["feed_digestibility_fraction_ruminant", "feed_digestibility_fraction_pigs"]
    assert results == pytest.approx(
        {"feed_digestibility_fraction_ruminant": 0.5, "feed_digestibility_fraction_pigs": 0.4375}, rel=REL
    )


def test_calc_feed_digestibility_fraction_treats_na_numerator_as_zero():
    results = calc_feed_digestibility_fraction(
        feed_digestible_energy_ruminant=np.nan,
        feed_digestible_energy_pigs=7,
        feed_gross_energy=16,
    )
    assert results == pytest.approx(
        {"feed_digestibility_fraction_ruminant": 0, "feed_digestibility_fraction_pigs": 0.4375}, rel=REL
    )


# ---- calc_ration_gross_energy ----


def test_calc_ration_gross_energy_computes_contribution():
    assert calc_ration_gross_energy(0.6, 18) == pytest.approx(10.8, rel=REL)


def test_calc_ration_gross_energy_rejects_na_inputs():
    with pytest.raises(GleamValidationError, match="must be a single numeric value"):
        calc_ration_gross_energy(np.nan, 18)


# ---- calc_ration_nitrogen_content ----


def test_calc_ration_nitrogen_content_computes_contribution():
    assert calc_ration_nitrogen_content(0.6, 0.02) == pytest.approx(0.012, rel=REL)


def test_calc_ration_nitrogen_content_rejects_na_inputs():
    with pytest.raises(GleamValidationError, match="must be a single numeric value"):
        calc_ration_nitrogen_content(0.6, np.nan)


# ---- calc_ration_urinary_energy_fraction ----


def test_calc_ration_urinary_energy_fraction_selects_ruminant_urinary_energy():
    value = calc_ration_urinary_energy_fraction(
        species_short="CTL",
        feed_ration_fraction=0.6,
        feed_urinary_energy_ruminant=0.12,
        feed_urinary_energy_pigs=0.02,
    )
    assert value == pytest.approx(0.072, rel=REL)


def test_calc_ration_urinary_energy_fraction_selects_pig_urinary_energy():
    value = calc_ration_urinary_energy_fraction(
        species_short="PGS",
        feed_ration_fraction=0.6,
        feed_urinary_energy_ruminant=0.12,
        feed_urinary_energy_pigs=0.02,
    )
    assert value == pytest.approx(0.012, rel=REL)


def test_calc_ration_urinary_energy_fraction_rejects_invalid_species_short():
    with pytest.raises(GleamValidationError, match="`species_short` must be one of"):
        calc_ration_urinary_energy_fraction(
            species_short="DOG",
            feed_ration_fraction=0.6,
            feed_urinary_energy_ruminant=0.12,
            feed_urinary_energy_pigs=0.02,
        )


def test_calc_ration_urinary_energy_fraction_allows_na_for_unused_animals():
    value = calc_ration_urinary_energy_fraction(
        species_short="PGS",
        feed_ration_fraction=0.6,
        feed_urinary_energy_ruminant=np.nan,
        feed_urinary_energy_pigs=0.02,
    )
    assert value == pytest.approx(0.012, rel=REL)


def test_calc_ration_urinary_energy_fraction_rejects_missing_required_inputs():
    with pytest.raises(GleamValidationError, match="Missing required urinary energy inputs"):
        calc_ration_urinary_energy_fraction(
            species_short="PGS",
            feed_ration_fraction=0.6,
            feed_urinary_energy_ruminant=0.12,
            feed_urinary_energy_pigs=np.nan,
        )


# ---- calc_ration_ash ----


def test_calc_ration_ash_computes_contribution():
    assert calc_ration_ash(0.6, 10) == pytest.approx(0.06, rel=REL)


def test_calc_ration_ash_rejects_na_inputs():
    with pytest.raises(GleamValidationError, match="must be a single numeric value"):
        calc_ration_ash(np.nan, 10)


# ---- additional checks (Python port) ----


def test_public_names_are_registered():
    for name in (
        "calc_feed_digestibility_fraction", "calc_ration_digestibility",
        "calc_ration_metabolizable_energy", "calc_ration_gross_energy",
        "calc_ration_nitrogen_content", "calc_ration_urinary_energy_fraction", "calc_ration_ash",
    ):
        assert callable(getattr(gleampy, name))


def test_chickens_use_the_pig_parameters_like_r():
    # R: `species_short %in% gleam_species_milk_producers` else -> pig parameter
    assert calc_ration_digestibility("CHK", 0.5, 0.8, 0.6) == pytest.approx(0.3, rel=REL)
    assert calc_ration_metabolizable_energy("CHK", 0.5, 10, 12) == pytest.approx(6, rel=REL)
    assert calc_ration_urinary_energy_fraction("CHK", 0.5, 0.1, 0.02) == pytest.approx(0.01, rel=REL)


def test_validation_messages_follow_r():
    with pytest.raises(GleamValidationError) as err:
        calc_ration_digestibility("PGS", 0.6, 0.7, np.nan)
    assert str(err.value) == (
        'Missing required digestibility inputs for species_short "PGS": "feed_digestibility_fraction_pigs"'
    )
    with pytest.raises(GleamValidationError, match="`feed_gross_energy` must not contain missing values"):
        calc_feed_digestibility_fraction(8, 7, np.nan)
    with pytest.raises(GleamValidationError, match="`feed_gross_energy` = 50 is out of range"):
        calc_feed_digestibility_fraction(8, 7, 50)
    with pytest.raises(GleamValidationError, match="`feed_digestible_energy_pigs` must be numeric"):
        calc_feed_digestibility_fraction(8, "7", 16)
    with pytest.raises(GleamValidationError, match="`feed_ration_fraction` = 1.5 is out of range"):
        calc_ration_ash(1.5, 10)
    with pytest.raises(GleamValidationError, match="`feed_ash` = 30 is out of range"):
        calc_ration_ash(0.5, 30)
    with pytest.raises(GleamValidationError, match="NA is allowed"):
        calc_ration_digestibility("CTL", 0.5, "0.7", 0.5)
    with pytest.raises(GleamValidationError, match="`feed_digestibility_fraction_pigs` = 2 is out of range"):
        calc_ration_digestibility("CTL", 0.5, 0.7, 2)
    with pytest.raises(GleamValidationError, match="`species_short` must be a single character value"):
        calc_ration_digestibility(np.nan, 0.5, 0.7, 0.5)


def test_vector_inputs_are_validated_element_wise():
    with pytest.raises(GleamValidationError, match='inputs for species_short "PGS"'):
        calc_ration_urinary_energy_fraction(["CTL", "PGS"], [0.5, 0.5], [0.1, 0.1], [0.02, np.nan])
    with pytest.raises(GleamValidationError, match=r"`feed_ash`\[2\] = 30 is out of range"):
        calc_ration_ash([0.5, 0.5], [10, 30])
    with pytest.raises(GleamValidationError, match="same length"):
        calc_ration_gross_energy([0.5, 0.5, 0.2], [10, 30])


def test_validation_can_be_disabled():
    with validation_disabled():
        assert np.isnan(calc_ration_digestibility("PGS", 0.6, 0.7, np.nan))
        assert calc_feed_digestibility_fraction(8, 7, 0)["feed_digestibility_fraction_ruminant"] == np.inf


# ---- vectorised == element-by-element ----


def test_vectorised_matches_elementwise_calls():
    species = np.array(["CTL", "BFL", "SHP", "GTS", "CML", "PGS", "CHK", "PGS", "CTL"], dtype=object)
    frac = np.array([0.3, 0.2, 0.1, 0.05, 0.15, 0.6, 0.25, 0.4, 0.35])
    rum = np.array([0.7, 0.6, 0.55, 0.65, 0.5, np.nan, 0.6, 0.4, 0.45])
    pig = np.array([np.nan, 0.5, 0.45, np.nan, 0.4, 0.8, 0.75, 0.55, 0.65])
    me_rum = np.array([10, 9, 8.5, 11, 7, np.nan, 9.5, 8, 10.5])
    me_pig = np.array([np.nan, 12, 11, np.nan, 10, 13, 12.5, 11.5, 9])
    ue_rum = rum / 5
    ue_pig = pig / 10
    ge = np.array([17.8, 18.2, 17.9, 18.4, 18.0, 16.5, 19.1, 17.0, 18.3])
    de_rum = np.array([10.4, np.nan, 9.2, 14.8, 12.0, 11.0, np.nan, 9.9, 10.1])
    de_pig = np.array([7.39, 2.5, np.nan, 14.8, 8.0, 13.0, 15.0, np.nan, 9.0])
    nit = np.array([0.0169, 0.00608, 0.0114, 0.0189, 0.02, 0.03, 0.025, 0.018, 0.016])
    ash = np.array([4, 4, 4, 5, 6, 7, 8, 3, 2.5])
    n = len(species)

    def check(fn, *args):
        vec = fn(*args)
        rows = [fn(*(a[i] if isinstance(a, np.ndarray) else a for a in args)) for i in range(n)]
        if isinstance(vec, dict):
            for k in vec:
                assert all(isinstance(r[k], float) for r in rows)
                np.testing.assert_array_equal(vec[k], [r[k] for r in rows], err_msg=k)
        else:
            assert isinstance(vec, np.ndarray) and vec.shape == (n,)
            assert all(isinstance(r, float) for r in rows)
            np.testing.assert_array_equal(vec, rows)

    check(calc_feed_digestibility_fraction, de_rum, de_pig, ge)
    check(calc_ration_digestibility, species, frac, rum, pig)
    check(calc_ration_metabolizable_energy, species, frac, me_rum, me_pig)
    check(calc_ration_urinary_energy_fraction, species, frac, ue_rum, ue_pig)
    check(calc_ration_gross_energy, frac, ge)
    check(calc_ration_nitrogen_content, frac, nit)
    check(calc_ration_ash, frac, ash)
