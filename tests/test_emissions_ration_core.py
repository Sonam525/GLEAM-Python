"""Port of ``tests/testthat/test-emissions_ration_core.R`` plus vectorisation checks.

R's "invalid type/length" tests pass ``c(1, 2)`` with a scalar
``feed_ration_fraction``. The Python functions are vectorised, so a length-2
vector with a scalar fraction is valid input (it broadcasts); the length rule
becomes "scalar or the same length as the other inputs", tested here with a
length-3 ``feed_ration_fraction``. Non-numeric values (``"10"``) are rejected
exactly as in R.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import gleampy
from gleampy import (
    GleamValidationError,
    calc_ch4_ration_rice,
    calc_co2_ration_crop_activities,
    calc_co2_ration_fertilizer,
    calc_co2_ration_luc_nopeat,
    calc_co2_ration_luc_peat,
    calc_co2_ration_pesticides,
    calc_n2o_ration_crop_residues,
    calc_n2o_ration_fertilizer,
    calc_n2o_ration_manure,
    validation_disabled,
)
from float_compare import assert_same_float

REL = 1.5e-8  # testthat tolerance

# R's `c(1, 2)` can only be an invalid length when it cannot be matched to the
# other inputs (see module docstring).
BAD_LENGTH = dict(feed_ration_fraction=[0.6, 0.3, 0.1], ef=[1, 2])


def _bad_length(fn):
    return fn(BAD_LENGTH["feed_ration_fraction"], BAD_LENGTH["ef"])


# CO2: fertilizer
def test_calc_co2_ration_fertilizer_computes_contribution():
    assert calc_co2_ration_fertilizer(0.6, 10) == pytest.approx(6, rel=REL)


def test_calc_co2_ration_fertilizer_allows_na_for_co2_feed_fertilizer():
    assert math.isnan(calc_co2_ration_fertilizer(0.6, np.nan))


def test_calc_co2_ration_fertilizer_rejects_negative_co2_feed_fertilizer():
    with pytest.raises(GleamValidationError, match="must be >= 0"):
        calc_co2_ration_fertilizer(0.6, -1)


def test_calc_co2_ration_fertilizer_rejects_invalid_co2_feed_fertilizer_type_length():
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        _bad_length(calc_co2_ration_fertilizer)


def test_calc_co2_ration_fertilizer_rejects_na_feed_ration_fraction():
    with pytest.raises(GleamValidationError, match="must not contain missing values"):
        calc_co2_ration_fertilizer(np.nan, 10)


# CO2: pesticides
def test_calc_co2_ration_pesticides_computes_contribution():
    assert calc_co2_ration_pesticides(0.6, 10) == pytest.approx(6, rel=REL)


def test_calc_co2_ration_pesticides_allows_na_for_co2_feed_pesticides():
    assert math.isnan(calc_co2_ration_pesticides(0.6, np.nan))


def test_calc_co2_ration_pesticides_rejects_negative_co2_feed_pesticides():
    with pytest.raises(GleamValidationError, match="must be >= 0"):
        calc_co2_ration_pesticides(0.6, -1)


def test_calc_co2_ration_pesticides_rejects_invalid_co2_feed_pesticides_type_length():
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        calc_co2_ration_pesticides(0.6, "10")


# CO2: crop operations
def test_calc_co2_ration_crop_activities_computes_contribution():
    assert calc_co2_ration_crop_activities(0.6, 10) == pytest.approx(6, rel=REL)


def test_calc_co2_ration_crop_activities_allows_na_for_co2_feed_crop_activities():
    assert math.isnan(calc_co2_ration_crop_activities(0.6, np.nan))


def test_calc_co2_ration_crop_activities_rejects_negative_co2_feed_crop_activities():
    with pytest.raises(GleamValidationError, match="must be >= 0"):
        calc_co2_ration_crop_activities(0.6, -1)


def test_calc_co2_ration_crop_activities_rejects_invalid_co2_feed_crop_activities_type_length():
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        _bad_length(calc_co2_ration_crop_activities)


# CO2: LUC no peat
def test_calc_co2_ration_luc_nopeat_computes_contribution():
    assert calc_co2_ration_luc_nopeat(0.6, 10) == pytest.approx(6, rel=REL)


def test_calc_co2_ration_luc_nopeat_allows_na_for_co2_feed_luc_nopeat():
    assert math.isnan(calc_co2_ration_luc_nopeat(0.6, np.nan))


def test_calc_co2_ration_luc_nopeat_rejects_invalid_co2_feed_luc_nopeat_type_length():
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        _bad_length(calc_co2_ration_luc_nopeat)


# CO2: LUC peat
def test_calc_co2_ration_luc_peat_computes_contribution():
    assert calc_co2_ration_luc_peat(0.6, 10) == pytest.approx(6, rel=REL)


def test_calc_co2_ration_luc_peat_allows_na_for_co2_feed_luc_peat():
    assert math.isnan(calc_co2_ration_luc_peat(0.6, np.nan))


def test_calc_co2_ration_luc_peat_rejects_invalid_co2_feed_luc_peat_type_length():
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        _bad_length(calc_co2_ration_luc_peat)


# N2O: fertilizer
def test_calc_n2o_ration_fertilizer_computes_contribution():
    assert calc_n2o_ration_fertilizer(0.6, 10) == pytest.approx(6, rel=REL)


def test_calc_n2o_ration_fertilizer_allows_na_for_n2o_feed_fertilizer():
    assert math.isnan(calc_n2o_ration_fertilizer(0.6, np.nan))


def test_calc_n2o_ration_fertilizer_rejects_negative_n2o_feed_fertilizer():
    with pytest.raises(GleamValidationError, match="must be >= 0"):
        calc_n2o_ration_fertilizer(0.6, -1)


def test_calc_n2o_ration_fertilizer_rejects_invalid_n2o_feed_fertilizer_type_length():
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        calc_n2o_ration_fertilizer(0.6, "10")


# N2O: manure applied
def test_calc_n2o_ration_manure_computes_contribution():
    assert calc_n2o_ration_manure(0.6, 10) == pytest.approx(6, rel=REL)


def test_calc_n2o_ration_manure_allows_na_for_n2o_feed_manure_applied():
    assert math.isnan(calc_n2o_ration_manure(0.6, np.nan))


def test_calc_n2o_ration_manure_rejects_negative_n2o_feed_manure_applied():
    with pytest.raises(GleamValidationError, match="must be >= 0"):
        calc_n2o_ration_manure(0.6, -1)


def test_calc_n2o_ration_manure_rejects_invalid_n2o_feed_manure_applied_type_length():
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        _bad_length(calc_n2o_ration_manure)


# N2O: crop residues
def test_calc_n2o_ration_crop_residues_computes_contribution():
    assert calc_n2o_ration_crop_residues(0.6, 10) == pytest.approx(6, rel=REL)


def test_calc_n2o_ration_crop_residues_allows_na_for_n2o_feed_crop_residues():
    assert math.isnan(calc_n2o_ration_crop_residues(0.6, np.nan))


def test_calc_n2o_ration_crop_residues_rejects_negative_n2o_feed_crop_residues():
    with pytest.raises(GleamValidationError, match="must be >= 0"):
        calc_n2o_ration_crop_residues(0.6, -1)


def test_calc_n2o_ration_crop_residues_rejects_invalid_n2o_feed_crop_residues_type_length():
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        calc_n2o_ration_crop_residues(0.6, "10")


# CH4: rice
def test_calc_ch4_ration_rice_computes_contribution():
    assert calc_ch4_ration_rice(0.6, 10) == pytest.approx(6, rel=REL)


def test_calc_ch4_ration_rice_allows_na_for_ch4_feed_rice():
    assert math.isnan(calc_ch4_ration_rice(0.6, np.nan))


def test_calc_ch4_ration_rice_rejects_negative_ch4_feed_rice():
    with pytest.raises(GleamValidationError, match="must be >= 0"):
        calc_ch4_ration_rice(0.6, -1)


def test_calc_ch4_ration_rice_rejects_invalid_ch4_feed_rice_type_length():
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        _bad_length(calc_ch4_ration_rice)


# ---- additional checks (Python port) ----

ALL = (
    calc_co2_ration_fertilizer, calc_co2_ration_pesticides, calc_co2_ration_crop_activities,
    calc_co2_ration_luc_nopeat, calc_co2_ration_luc_peat, calc_n2o_ration_fertilizer,
    calc_n2o_ration_manure, calc_n2o_ration_crop_residues, calc_ch4_ration_rice,
)


def test_public_names_are_registered():
    for fn in ALL:
        assert getattr(gleampy, fn.__name__) is fn


@pytest.mark.parametrize("fn", ALL, ids=lambda f: f.__name__)
def test_r_type_length_cases(fn):
    # R rejects c(1, 2) and "10"; Python rejects the non-numeric string and a
    # vector that cannot be matched to the other inputs.
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        fn(0.6, "10")
    with pytest.raises(GleamValidationError, match="must be a single numeric"):
        _bad_length(fn)
    np.testing.assert_allclose(fn(0.6, [1, 2]), [0.6, 1.2], rtol=1e-15)


def test_land_use_change_factors_may_be_negative():
    assert calc_co2_ration_luc_nopeat(0.5, -10) == pytest.approx(-5, rel=REL)
    assert calc_co2_ration_luc_peat(0.5, -10) == pytest.approx(-5, rel=REL)


def test_feed_ration_fraction_range_and_messages():
    with pytest.raises(GleamValidationError) as err:
        calc_ch4_ration_rice(0.6, -1)
    assert str(err.value) == "`ch4_feed_rice` must be >= 0."
    with pytest.raises(GleamValidationError, match="`feed_ration_fraction` = 1.2 is out of range"):
        calc_co2_ration_fertilizer(1.2, 10)
    with pytest.raises(GleamValidationError, match=r"`feed_ration_fraction`\[2\] = -0.1 is out of range"):
        calc_co2_ration_fertilizer([0.5, -0.1], 10)
    with validation_disabled():
        assert calc_co2_ration_fertilizer(1.2, -1) == pytest.approx(-1.2, rel=REL)


def test_vectorised_matches_elementwise_calls():
    frac = np.array([0.3, 0.2, 0.1, 0.05, 0.15, 0.2, 1.0, 0.0])
    ef = np.array([57.3, 0.0, np.nan, 114.0, 9.98, 0.246, 0.454, 3.3])
    signed = np.array([57.3, -12.5, np.nan, 114.0, -9.98, 0.246, 0.454, -3.3])
    for fn in ALL:
        values = signed if fn in (calc_co2_ration_luc_nopeat, calc_co2_ration_luc_peat) else ef
        vec = fn(frac, values)
        rows = [fn(float(f), float(v)) for f, v in zip(frac, values)]
        assert isinstance(vec, np.ndarray) and vec.shape == frac.shape
        assert all(isinstance(r, float) for r in rows)
        assert_same_float(vec, rows, err_msg=fn.__name__)
