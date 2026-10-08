"""Tests for the metabolic energy requirement core model.

Port of ``tests/testthat/test-metabolic_energy_req_core.R`` plus Python-only
checks of the vectorised implementation (vector call == element-wise scalar
calls on a mixed set of species / cohorts) and of the vectorised validation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import gleampy
from gleampy import (
    calc_metabolic_energy_req_activity,
    calc_metabolic_energy_req_eggs,
    calc_metabolic_energy_req_fibre,
    calc_metabolic_energy_req_growth,
    calc_metabolic_energy_req_lactation,
    calc_metabolic_energy_req_maintenance,
    calc_metabolic_energy_req_pregnancy,
    calc_metabolic_energy_req_work,
    calc_ration_intake,
    calc_reg_growth,
    calc_rem_maintenance,
    calc_total_metabolic_energy_req,
)
from gleampy._utils import lookup
from gleampy.io import load_example
from gleampy.validation import GleamValidationError, validation_disabled


def approx(x):
    """testthat ``expect_equal`` tolerance."""
    return pytest.approx(x, rel=1.5e-8)


# ---- calc_metabolic_energy_req_maintenance ----------------------------------


def test_maintenance_returns_correct_values_for_cattle():
    # Adult female with lactating fraction
    result = calc_metabolic_energy_req_maintenance(
        species_short="CTL", cohort_short="FA", live_weight_cohort_average=500,
        lactating_females_fraction=0.7,
    )
    assert result == approx((500 ** 0.75) * (0.386 * 0.7 + 0.322 * 0.3))

    # Juvenile female
    result = calc_metabolic_energy_req_maintenance(
        species_short="CTL", cohort_short="FJ", live_weight_cohort_average=200
    )
    assert result == approx((200 ** 0.75) * 0.322)

    # Adult male with offtake
    result = calc_metabolic_energy_req_maintenance(
        species_short="CTL", cohort_short="MA", live_weight_cohort_average=600, offtake_rate=0.3
    )
    assert result == approx((600 ** 0.75) * (0.322 * 0.3 + 0.37 * 0.7))


def test_maintenance_handles_sheep_with_age_at_first_parturition():
    result = calc_metabolic_energy_req_maintenance(
        species_short="SHP", cohort_short="FS", live_weight_cohort_average=40,
        age_first_parturition=400,
    )
    expected = (40 ** 0.75) * ((0.236 * (365 / 400)) + (0.217 * ((400 - 365) / 400)))
    assert result == approx(expected)

    result = calc_metabolic_energy_req_maintenance(
        species_short="SHP", cohort_short="FA", live_weight_cohort_average=60
    )
    assert result == approx((60 ** 0.75) * 0.217)


def test_maintenance_handles_pigs():
    result = calc_metabolic_energy_req_maintenance(
        species_short="PGS", cohort_short="FA", live_weight_cohort_average=150
    )
    assert result == approx((150 ** 0.75) * 0.4435)


def test_maintenance_handles_fixed_coefficients():
    result = calc_metabolic_energy_req_maintenance(
        species_short="CML", cohort_short="FA", live_weight_cohort_average=400
    )
    assert result == approx((400 ** 0.75) * 0.435)

    result = calc_metabolic_energy_req_maintenance(
        species_short="GTS", cohort_short="FA", live_weight_cohort_average=50
    )
    assert result == approx((50 ** 0.75) * 0.315)


def test_maintenance_zero_lactating_females_fraction():
    result = calc_metabolic_energy_req_maintenance(
        species_short="CTL", cohort_short="FA", live_weight_cohort_average=500,
        lactating_females_fraction=0,
    )
    assert result == pytest.approx((500 ** 0.75) * 0.322, rel=1e-8)


def test_maintenance_handles_chickens():
    adult = calc_metabolic_energy_req_maintenance(
        species_short="CHK", cohort_short="FA", live_weight_cohort_average=2,
        average_annual_temperature=5, is_egg_producing=True,
    )
    assert adult == approx((2 ** 0.75) * (0.6935 - 0.0099 * 5))

    juvenile = calc_metabolic_energy_req_maintenance(
        species_short="CHK", cohort_short="FJ", live_weight_cohort_average=0.04,
        average_annual_temperature=15, lower_critical_temperature=18,
    )
    assert juvenile == approx(0.3866 + 0.0282 * (18 - 15))


def test_maintenance_uses_explicit_egg_producing_fn_flag():
    result = calc_metabolic_energy_req_maintenance(
        species_short="CHK", cohort_short="FN", nondemo_productive_phase_id=2,
        live_weight_cohort_average=1.8, average_annual_temperature=18, is_egg_producing=True,
    )
    assert result == approx(max(0, (1.8 ** 0.75) * (0.6935 - 0.0099 * 18)))


# ---- calc_metabolic_energy_req_activity -------------------------------------


def test_activity_returns_correct_values_for_cattle():
    result = calc_metabolic_energy_req_activity(
        species_short="CTL", cohort_short="FA", metabolic_energy_req_maintenance=15.0,
        live_weight_cohort_average=500, low_activity_fraction=0.6, high_activity_fraction=0.2,
    )
    assert result == approx(((0.17 * 0.6) + (0.36 * 0.2)) * 15.0)


def test_activity_returns_correct_values_for_buffalo():
    result = calc_metabolic_energy_req_activity(
        species_short="BFL", cohort_short="FA", metabolic_energy_req_maintenance=18.0,
        live_weight_cohort_average=600, low_activity_fraction=0.5, high_activity_fraction=0.2,
    )
    assert result == approx(((0.17 * 0.5) + (0.36 * 0.2)) * 18.0)


def test_activity_handles_sheep_complexity():
    result = calc_metabolic_energy_req_activity(
        species_short="SHP", cohort_short="MS", metabolic_energy_req_maintenance=8.0,
        live_weight_cohort_average=45, low_activity_fraction=0.5, high_activity_fraction=0.2,
    )
    assert result == approx(((0.0107 * 0.5) + (0.024 * 0.2)) * 45)

    result = calc_metabolic_energy_req_activity(
        species_short="SHP", cohort_short="FA", metabolic_energy_req_maintenance=8.0,
        live_weight_cohort_average=60, low_activity_fraction=0.5, high_activity_fraction=0.2,
    )
    assert result == approx(((0.0107 * 0.5) + (0.024 * 0.2)) * 60)


def test_activity_handles_different_species():
    result = calc_metabolic_energy_req_activity(
        species_short="CML", cohort_short="FA", metabolic_energy_req_maintenance=12.0,
        live_weight_cohort_average=400, low_activity_fraction=0.5, high_activity_fraction=0,
    )
    assert result == approx((0.1 * 0.5) * 12.0)

    result = calc_metabolic_energy_req_activity(
        species_short="PGS", cohort_short="FA", metabolic_energy_req_maintenance=10.0,
        live_weight_cohort_average=150, low_activity_fraction=0.5, high_activity_fraction=0.3,
    )
    assert result == approx(0.125 * (0.5 + 0.3) * 10.0)

    result = calc_metabolic_energy_req_activity(
        species_short="GTS", cohort_short="FA", metabolic_energy_req_maintenance=8.0,
        live_weight_cohort_average=50, low_activity_fraction=0.4, high_activity_fraction=0.2,
    )
    assert result == approx(((0.019 * 0.4) + (0.024 * 0.2)) * 50)


def test_activity_zero_high_activity_fraction_for_cattle():
    result = calc_metabolic_energy_req_activity(
        species_short="CTL", cohort_short="FA", metabolic_energy_req_maintenance=12.0,
        live_weight_cohort_average=500, low_activity_fraction=0.6, high_activity_fraction=0,
    )
    assert result == approx(((0.17 * 0.6) + (0.36 * 0)) * 12.0)


def test_activity_handles_chickens():
    result = calc_metabolic_energy_req_activity(
        species_short="CHK", cohort_short="FA", metabolic_energy_req_maintenance=1.2,
        live_weight_cohort_average=2, low_activity_fraction=0.4, high_activity_fraction=0.3,
    )
    assert result == approx(1.2 * 0.7 * 0.25)


# ---- calc_metabolic_energy_req_growth ---------------------------------------


def test_growth_returns_correct_values_for_cattle():
    result = calc_metabolic_energy_req_growth(
        species_short="CTL", cohort_short="FJ",
        live_weight_cohort_average=200, live_weight_cohort_final=300,
        live_weight_cohort_initial=150, live_weight_mature_stage=500,
        daily_weight_gain=0.5, offtake_rate=0.1, cohort_duration_days=100,
    )
    assert result == approx(22.02 * ((200 / (0.8 * 500)) ** 0.75) * (0.5 ** 1.097))

    result = calc_metabolic_energy_req_growth(
        species_short="CTL", cohort_short="FA",
        live_weight_cohort_average=500, live_weight_cohort_final=500,
        live_weight_cohort_initial=500, live_weight_mature_stage=500,
        daily_weight_gain=0, offtake_rate=0.1, cohort_duration_days=365,
    )
    assert result == 0


def test_growth_handles_sheep_linear_formula():
    result = calc_metabolic_energy_req_growth(
        species_short="SHP", cohort_short="FJ",
        live_weight_cohort_average=30, live_weight_cohort_final=50,
        live_weight_cohort_initial=25, live_weight_mature_stage=60,
        daily_weight_gain=0.1, offtake_rate=0.1, cohort_duration_days=250,
    )
    assert result == approx(((50 - 25) * (2.1 + 0.5 * 0.45 * (25 + 50))) / 250)

    result = calc_metabolic_energy_req_growth(
        species_short="SHP", cohort_short="FA",
        live_weight_cohort_average=60, live_weight_cohort_final=60,
        live_weight_cohort_initial=60, live_weight_mature_stage=60,
        daily_weight_gain=0, offtake_rate=0.1, cohort_duration_days=365,
    )
    assert result == 0


def test_growth_handles_pigs():
    result = calc_metabolic_energy_req_growth(
        species_short="PGS", cohort_short="FJ",
        live_weight_cohort_average=50, live_weight_cohort_final=80,
        live_weight_cohort_initial=40, live_weight_mature_stage=300,
        daily_weight_gain=0.3, offtake_rate=0.1, cohort_duration_days=133,
    )
    prot_tissue_frac = 0.65
    cgro = (prot_tissue_frac * 0.23 * 54) + ((1 - prot_tissue_frac) * 0.9 * 52.3)
    assert result == approx(0.3 * cgro)


def test_growth_handles_chickens():
    juvenile = calc_metabolic_energy_req_growth(
        species_short="CHK", cohort_short="FJ", daily_weight_gain=0.02
    )
    assert juvenile == approx(0.02 * 0.0202 * 1000)

    adult = calc_metabolic_energy_req_growth(
        species_short="CHK", cohort_short="FA", daily_weight_gain=0.01, is_egg_producing=True
    )
    assert adult == approx(0.01 * 0.0279 * 1000)


def test_growth_uses_explicit_egg_producing_fn_flag():
    result = calc_metabolic_energy_req_growth(
        species_short="CHK", cohort_short="FN", nondemo_productive_phase_id=2,
        daily_weight_gain=0.01, is_egg_producing=True,
    )
    assert result == approx(0.01 * 0.0279 * 1000)


def test_growth_uses_broiler_coefficient_for_non_laying_fn():
    result = calc_metabolic_energy_req_growth(
        species_short="CHK", cohort_short="FN", nondemo_productive_phase_id=1,
        daily_weight_gain=0.01, is_egg_producing=False,
    )
    assert result == approx(0.01 * 0.0202 * 1000)


# ---- calc_metabolic_energy_req_eggs -----------------------------------------


def test_eggs_handles_chickens():
    result = calc_metabolic_energy_req_eggs(
        species_short="CHK", cohort_short="FA", cohort_stock_size=100,
        egg_output_human_consumption=36500, egg_average_weight=0.06, parturition_rate=120,
        is_egg_producing=True,
    )
    egg_mass = ((36500 / 365 / 100) + (120 / 365)) * 0.06
    assert result == approx(egg_mass * 10.04)
    assert calc_metabolic_energy_req_eggs(
        species_short="CHK", cohort_short="FS", cohort_stock_size=100,
        egg_output_human_consumption=36500, egg_average_weight=0.06, parturition_rate=120,
    ) == approx(0)


def test_eggs_uses_explicit_egg_producing_fn_flag():
    result = calc_metabolic_energy_req_eggs(
        species_short="CHK", cohort_short="FN", nondemo_productive_phase_id=2,
        cohort_stock_size=100, egg_output_human_consumption=36500, egg_average_weight=0.06,
        parturition_rate=120, is_egg_producing=True,
    )
    egg_mass = ((36500 / 365 / 100) + (120 / 365)) * 0.06
    assert result == approx(egg_mass * 10.04)
    assert calc_metabolic_energy_req_eggs(
        species_short="CHK", cohort_short="FN", nondemo_productive_phase_id=1,
        cohort_stock_size=100, egg_output_human_consumption=36500, egg_average_weight=0.06,
        parturition_rate=120,
    ) == approx(0)


def test_eggs_validates_egg_producing_flag_placement():
    with pytest.raises(GleamValidationError, match="can be TRUE only for CHK cohorts.*FA.*FN"):
        calc_metabolic_energy_req_eggs(
            species_short="CHK", cohort_short="FS", cohort_stock_size=100,
            egg_output_human_consumption=36500, egg_average_weight=0.06, parturition_rate=120,
            is_egg_producing=True,
        )
    with pytest.raises(GleamValidationError, match="can be TRUE for.*FN.*only when.*2"):
        calc_metabolic_energy_req_eggs(
            species_short="CHK", cohort_short="FN", nondemo_productive_phase_id=1,
            cohort_stock_size=100, egg_output_human_consumption=36500, egg_average_weight=0.06,
            parturition_rate=120, is_egg_producing=True,
        )


# ---- calc_metabolic_energy_req_lactation ------------------------------------


def test_lactation_returns_correct_values_for_cattle():
    result = calc_metabolic_energy_req_lactation(
        species_short="CTL", cohort_short="FA", lactating_females_fraction=0.8,
        milk_yield_day=20, milk_fat_fraction=0.04, non_productive_duration=0,
        pregnancy_duration=0, litter_size=1, death_rate_juvenile=0,
        live_weight_at_birth=35, live_weight_at_weaning=90, lactation_duration=0,
        parturition_rate=0.8,
    )
    expected = ((20 * 0.8) + (0.8 * 5 * (90 - 35) / 365)) * (0.04 * 100 * 0.40 + 1.47)
    assert result == approx(expected)


def test_lactation_handles_sheep_with_litter_size():
    result = calc_metabolic_energy_req_lactation(
        species_short="SHP", cohort_short="FA", lactating_females_fraction=0.9,
        milk_yield_day=1.5, milk_fat_fraction=0.06, non_productive_duration=0,
        pregnancy_duration=0, litter_size=1.5, death_rate_juvenile=0,
        live_weight_at_birth=4, live_weight_at_weaning=18, lactation_duration=0,
        parturition_rate=1.2,
    )
    expected = ((1.5 * 0.9) + (1.5 * 1.2 * 5 * (18 - 4) / 365)) * 4.6
    assert result == approx(expected)


def test_lactation_handles_pigs():
    result = calc_metabolic_energy_req_lactation(
        species_short="PGS", cohort_short="FA", lactating_females_fraction=0,
        milk_yield_day=0, milk_fat_fraction=0, non_productive_duration=0.2,
        pregnancy_duration=0.3, litter_size=10, death_rate_juvenile=0.1,
        live_weight_at_birth=1.5, live_weight_at_weaning=8, lactation_duration=0.5,
        parturition_rate=2.2,
    )
    cadj = 0.5 / (0.2 + 0.3 + 0.5)
    expected = 10 * (1 - 0.5 * 0.1) * ((0.02059 * (8 - 1.5) * 1000 / 0.5) - (0.3766 / 0.67)) * cadj
    assert result == approx(expected)


# ---- calc_metabolic_energy_req_work -----------------------------------------


def test_work_returns_correct_values_for_working_animals():
    result = calc_metabolic_energy_req_work(
        species_short="CTL", cohort_short="MA", metabolic_energy_req_maintenance=20.0,
        draught_work_hours_female=2, draught_work_hours_male=4,
        draught_fraction_female=0.5, draught_fraction_male=0.3,
    )
    assert result == approx(0.1 * 20.0 * 4 * 0.3)

    result = calc_metabolic_energy_req_work(
        species_short="CTL", cohort_short="FA", metabolic_energy_req_maintenance=15.0,
        draught_work_hours_female=2, draught_work_hours_male=4,
        draught_fraction_female=0.5, draught_fraction_male=0.3,
    )
    assert result == approx(0.1 * 15.0 * 2 * 0.5)


def test_work_handles_different_species():
    result = calc_metabolic_energy_req_work(
        species_short="CML", cohort_short="MA", metabolic_energy_req_maintenance=18.0,
        draught_work_hours_female=2, draught_work_hours_male=6,
        draught_fraction_female=0.5, draught_fraction_male=0.4,
    )
    assert result == approx(4 * 6 * 0.4)

    result = calc_metabolic_energy_req_work(
        species_short="SHP", cohort_short="MA", metabolic_energy_req_maintenance=10.0,
        draught_work_hours_female=8, draught_work_hours_male=4,
        draught_fraction_female=0.3, draught_fraction_male=0.3,
    )
    assert result == 0


# ---- calc_metabolic_energy_req_fibre ----------------------------------------


def test_fibre_returns_correct_values_for_fibre_producing_animals():
    result = calc_metabolic_energy_req_fibre(species_short="SHP", cohort_short="FA", fibre_yield_year=2.5)
    assert result == approx(24 * 2.5 / 365)

    result = calc_metabolic_energy_req_fibre(species_short="SHP", cohort_short="FJ", fibre_yield_year=1.0)
    assert result == 0


def test_fibre_handles_camelids():
    result = calc_metabolic_energy_req_fibre(species_short="CML", cohort_short="FA", fibre_yield_year=3.0)
    assert result == approx((24 / 0.43) * (3.0 / 365))


def test_fibre_returns_zero_for_non_fibre_animals():
    result = calc_metabolic_energy_req_fibre(species_short="CTL", cohort_short="FA", fibre_yield_year=1.0)
    assert result == 0


# ---- calc_metabolic_energy_req_pregnancy ------------------------------------


def test_pregnancy_returns_correct_values_for_cattle():
    result = calc_metabolic_energy_req_pregnancy(
        species_short="CTL", cohort_short="FA", metabolic_energy_req_maintenance=15.0,
        parturition_rate=0.8, litter_size=1, pregnancy_duration=283,
        non_productive_duration=10, lactation_duration=30, cohort_duration_days=730,
        offtake_rate=0.2,
    )
    assert result == approx(15.0 * 0.1 * 0.8 * 283 / 365)

    result = calc_metabolic_energy_req_pregnancy(
        species_short="CTL", cohort_short="FS", metabolic_energy_req_maintenance=12.0,
        parturition_rate=0.8, litter_size=1, pregnancy_duration=283,
        non_productive_duration=10, lactation_duration=30, cohort_duration_days=730,
        offtake_rate=0.2,
    )
    assert result == approx((12.0 * 0.1) * (283 / 730) * (1 - 0.2))


def test_pregnancy_handles_sheep_with_litter_size_effects():
    result = calc_metabolic_energy_req_pregnancy(
        species_short="SHP", cohort_short="FA", metabolic_energy_req_maintenance=8.0,
        parturition_rate=1.2, litter_size=1.5, pregnancy_duration=152,
        non_productive_duration=10, lactation_duration=30, cohort_duration_days=700,
        offtake_rate=0.1,
    )
    cpreg = 0.077 * 0.5 + 0.126 * 0.5
    assert result == approx(8.0 * cpreg * 1.2 * 152 / 365)

    result = calc_metabolic_energy_req_pregnancy(
        species_short="SHP", cohort_short="FA", metabolic_energy_req_maintenance=8.0,
        parturition_rate=1.2, litter_size=2.5, pregnancy_duration=152,
        non_productive_duration=10, lactation_duration=30, cohort_duration_days=365,
        offtake_rate=0.1,
    )
    assert result == approx(8.0 * 0.150 * 1.2 * 152 / 365)


def test_pregnancy_handles_pigs():
    result = calc_metabolic_energy_req_pregnancy(
        species_short="PGS", cohort_short="FA", metabolic_energy_req_maintenance=12.0,
        parturition_rate=2.2, litter_size=10, pregnancy_duration=115,
        non_productive_duration=10, lactation_duration=30, cohort_duration_days=365,
        offtake_rate=0.1,
    )
    assert result == approx(0.14985 * 10 * 115 / (10 + 115 + 30))


# ---- calc_rem_maintenance / calc_reg_growth ---------------------------------


def test_rem_returns_correct_values_for_ruminants():
    result = calc_rem_maintenance(species_short="CTL", ration_digestibility_fraction=0.65)
    assert result == approx(1.123 - (0.004092 * 65) + (0.00001126 * 65 ** 2) - (25.4 / 65))

    result = calc_rem_maintenance(species_short="SHP", ration_digestibility_fraction=0.55)
    assert result == approx(1.123 - (0.004092 * 55) + (0.00001126 * 55 ** 2) - (25.4 / 55))


def test_rem_returns_na_for_non_ruminants():
    assert math.isnan(calc_rem_maintenance(species_short="PGS", ration_digestibility_fraction=0.75))
    assert math.isnan(calc_rem_maintenance(species_short="CML", ration_digestibility_fraction=0.60))


def test_reg_returns_correct_values_for_ruminants():
    result = calc_reg_growth(species_short="CTL", ration_digestibility_fraction=0.65)
    assert result == approx(1.164 - (0.005160 * 65) + (0.00001308 * 65 ** 2) - (37.4 / 65))

    result = calc_reg_growth(species_short="GTS", ration_digestibility_fraction=0.55)
    assert result == approx(1.164 - (0.005160 * 55) + (0.00001308 * 55 ** 2) - (37.4 / 55))


def test_reg_returns_na_for_non_ruminants():
    assert math.isnan(calc_reg_growth(species_short="PGS", ration_digestibility_fraction=0.75))
    assert math.isnan(calc_reg_growth(species_short="CML", ration_digestibility_fraction=0.60))


# ---- calc_total_metabolic_energy_req ----------------------------------------


def _total(species, m, a, l_, w, p, rem, g, f, e, reg, de):
    return calc_total_metabolic_energy_req(
        species_short=species,
        metabolic_energy_req_maintenance=m,
        metabolic_energy_req_activity=a,
        metabolic_energy_req_lactation=l_,
        metabolic_energy_req_work=w,
        metabolic_energy_req_pregnancy=p,
        net_energy_maintenance_digestible_energy_ratio=rem,
        metabolic_energy_req_growth=g,
        metabolic_energy_req_fibre_production=f,
        metabolic_energy_req_egg_deposition=e,
        net_energy_growth_digestible_energy_ratio=reg,
        ration_digestibility_fraction=de,
    )


def test_total_returns_correct_values_for_cattle():
    result = _total("CTL", 15.0, 3.0, 8.0, 0, 1.5, 0.6, 0, 0, 0, 0.5, 0.65)
    assert result == approx((((15.0 + 3.0 + 8.0 + 0 + 1.5) / 0.6) + (0 / 0.5)) / 0.65)


def test_total_handles_sheep_with_fibre():
    result = _total("SHP", 8.0, 1.5, 4.0, 0, 1.0, 0.55, 0, 0.2, 0, 0.45, 0.60)
    assert result == approx((((8.0 + 1.5 + 4.0 + 1.0) / 0.55) + ((0 + 0.2) / 0.45)) / 0.60)


def test_total_handles_different_species():
    result = _total("CML", 12.0, 2.0, 6.0, 1.0, 1.5, np.nan, 0, 0.3, 0, np.nan, 0.70)
    assert result == approx(12.0 + 2.0 + 6.0 + 1.0 + 0.3 + 1.5 + 0)

    result = _total("PGS", 10.0, 1.0, 5.0, 0, 2.0, np.nan, 0, 0, 0, np.nan, 0.75)
    assert result == approx(10.0 + 1.0 + 5.0 + 2.0 + 0)


def test_total_handles_chickens():
    result = _total("CHK", 1.5, 0.3, 0, 0, 0, np.nan, 0.4, 0, 0.7, np.nan, 0.75)
    assert result == approx(1.5 + 0.3 + 0.4 + 0.7)


# ---- calc_ration_intake -----------------------------------------------------


def test_ration_intake_uses_gross_energy_for_ruminants():
    result = calc_ration_intake(
        species_short="CTL", metabolic_energy_req_total=25.0,
        ration_gross_energy=18.5, ration_metabolizable_energy=12.0,
    )
    assert result == approx(25.0 / 18.5)

    result = calc_ration_intake(
        species_short="SHP", metabolic_energy_req_total=12.0,
        ration_gross_energy=16.0, ration_metabolizable_energy=10.5,
    )
    assert result == approx(12.0 / 16.0)


def test_ration_intake_uses_metabolizable_energy_for_chickens():
    result = calc_ration_intake(
        species_short="CHK", metabolic_energy_req_total=5,
        ration_gross_energy=18, ration_metabolizable_energy=12.5,
    )
    assert result == approx(5 / 12.5)


def test_ration_intake_uses_metabolizable_energy_for_monogastrics():
    result = calc_ration_intake(
        species_short="PGS", metabolic_energy_req_total=15.0,
        ration_gross_energy=18.0, ration_metabolizable_energy=13.5,
    )
    assert result == approx(15.0 / 13.5)

    result = calc_ration_intake(
        species_short="CML", metabolic_energy_req_total=20.0,
        ration_gross_energy=17.0, ration_metabolizable_energy=12.5,
    )
    assert result == approx(20.0 / 12.5)


# ============================================================================
# Python-only checks
# ============================================================================


def test_public_api_exports_core_functions():
    assert gleampy.calc_metabolic_energy_req_maintenance is calc_metabolic_energy_req_maintenance
    assert gleampy.calc_ration_intake is calc_ration_intake
    assert gleampy.calc_metabolic_energy_req_eggs is calc_metabolic_energy_req_eggs


def test_scalar_inputs_return_python_float_and_vectors_return_arrays():
    out = calc_metabolic_energy_req_fibre("SHP", "FA", 2.5)
    assert isinstance(out, float)
    out = calc_metabolic_energy_req_fibre(["SHP", "CTL"], "FA", [2.5, 1.0])
    assert isinstance(out, np.ndarray) and out.shape == (2,)
    assert out[0] == approx(24 * 2.5 / 365) and out[1] == 0


# ---- Vectorised == element-wise (scalar) evaluation --------------------------


def _mixed_inputs() -> dict[str, np.ndarray]:
    """Every row of the bundled example (CTL, BFL, SHP, GTS, PGS, CML, CHK; FN/MN
    phases; laying and non-laying CHK) plus synthetic rows covering the CHK
    laying FN and below-critical-temperature branches, SHP/CTL non-demographic
    cohorts and litter sizes on each side of the SHP/GTS pregnancy breakpoints."""
    c = load_example("metabolic_energy_req_input_chrt_data.csv")
    h = load_example("metabolic_energy_req_input_hrd_data.csv")
    d = {col: c[col].to_numpy() for col in c.columns}
    for col in h.columns:
        if col not in ("herd_id", "species_short"):
            d[col] = lookup(c, h, col)
    d["species_short"] = lookup(c, h, "species_short")
    d["is_egg_producing"] = np.array([v is True for v in c["is_egg_producing"]], dtype=object)
    df = pd.DataFrame(d)

    extra = pd.DataFrame(
        [
            # CHK laying FN (phase 2), cold climate (temperature below the LCT)
            dict(df.iloc[-6], cohort_short="FN", nondemo_productive_phase_id=2, is_egg_producing=True,
                 average_annual_temperature=10.0),
            # CHK non-laying FN (phase 1) and FS in a cold climate
            dict(df.iloc[-4], cohort_short="FN", nondemo_productive_phase_id=1, is_egg_producing=False,
                 average_annual_temperature=12.0),
            dict(df.iloc[-4], average_annual_temperature=5.0),
            # SHP non-demographic cohorts and litter sizes > 2 / < 1
            dict(df.iloc[16], cohort_short="FN", nondemo_productive_phase_id=1),
            dict(df.iloc[19], cohort_short="MN", nondemo_productive_phase_id=1),
            dict(df.iloc[14], litter_size=2.4),
            dict(df.iloc[14], litter_size=0.6),
            # GTS twin litter, CTL FN
            dict(df.iloc[26], litter_size=1.7),
            dict(df.iloc[1], cohort_short="FN", nondemo_productive_phase_id=1),
            # Draught cattle and camels
            dict(df.iloc[8], draught_work_hours_female=3.0, draught_fraction_female=0.4),
            dict(df.iloc[70], draught_work_hours_female=5.0, draught_fraction_female=0.2),
        ]
    )
    df = pd.concat([df, extra], ignore_index=True)
    return {col: df[col].to_numpy() for col in df.columns}


def _call_args(d):
    """Argument builders (lazy: later functions use earlier outputs stored in ``d``)."""
    sp, co = d["species_short"], d["cohort_short"]
    lct = np.where(sp == "CHK", 18.89, np.nan)
    return {
        calc_metabolic_energy_req_maintenance: lambda: dict(
            species_short=sp, cohort_short=co, live_weight_cohort_average=d["live_weight_cohort_average"],
            lactating_females_fraction=d["lactating_females_fraction"], offtake_rate=d["offtake_rate"],
            age_first_parturition=d["age_first_parturition"],
            average_annual_temperature=d["average_annual_temperature"],
            lower_critical_temperature=lct, nondemo_productive_phase_id=d["nondemo_productive_phase_id"],
            is_egg_producing=d["is_egg_producing"],
        ),
        calc_metabolic_energy_req_activity: lambda: dict(
            species_short=sp, cohort_short=co, metabolic_energy_req_maintenance=d["maintenance"],
            live_weight_cohort_average=d["live_weight_cohort_average"],
            low_activity_fraction=d["low_activity_fraction"], high_activity_fraction=d["high_activity_fraction"],
        ),
        calc_metabolic_energy_req_growth: lambda: dict(
            species_short=sp, cohort_short=co, live_weight_cohort_average=d["live_weight_cohort_average"],
            live_weight_cohort_final=d["live_weight_cohort_final"],
            live_weight_cohort_initial=d["live_weight_cohort_initial"],
            live_weight_mature_stage=d["live_weight_mature_stage"], daily_weight_gain=d["daily_weight_gain"],
            offtake_rate=d["offtake_rate"], cohort_duration_days=d["cohort_duration_days"],
            nondemo_productive_phase_id=d["nondemo_productive_phase_id"], is_egg_producing=d["is_egg_producing"],
        ),
        calc_metabolic_energy_req_lactation: lambda: dict(
            species_short=sp, cohort_short=co, lactating_females_fraction=d["lactating_females_fraction"],
            milk_yield_day=d["milk_yield_day"], milk_fat_fraction=d["milk_fat_fraction"],
            non_productive_duration=d["non_productive_duration"], pregnancy_duration=d["pregnancy_duration"],
            litter_size=d["litter_size"], death_rate_juvenile=d["death_rate_juvenile"],
            live_weight_at_birth=d["live_weight_at_birth"], live_weight_at_weaning=d["live_weight_at_weaning"],
            lactation_duration=d["lactation_duration"], parturition_rate=d["parturition_rate"],
        ),
        calc_metabolic_energy_req_eggs: lambda: dict(
            species_short=sp, cohort_short=co, cohort_stock_size=d["cohort_stock_size"],
            egg_output_human_consumption=d["egg_output_human_consumption"],
            egg_average_weight=d["egg_average_weight"], parturition_rate=d["parturition_rate"],
            nondemo_productive_phase_id=d["nondemo_productive_phase_id"], is_egg_producing=d["is_egg_producing"],
        ),
        calc_metabolic_energy_req_work: lambda: dict(
            species_short=sp, cohort_short=co, metabolic_energy_req_maintenance=d["maintenance"],
            draught_work_hours_female=d["draught_work_hours_female"],
            draught_work_hours_male=d["draught_work_hours_male"],
            draught_fraction_female=d["draught_fraction_female"], draught_fraction_male=d["draught_fraction_male"],
        ),
        calc_metabolic_energy_req_fibre: lambda: dict(
            species_short=sp, cohort_short=co, fibre_yield_year=d["fibre_yield_year"],
        ),
        calc_metabolic_energy_req_pregnancy: lambda: dict(
            species_short=sp, cohort_short=co, metabolic_energy_req_maintenance=d["maintenance"],
            parturition_rate=d["parturition_rate"], litter_size=d["litter_size"],
            pregnancy_duration=d["pregnancy_duration"], non_productive_duration=d["non_productive_duration"],
            lactation_duration=d["lactation_duration"], cohort_duration_days=d["cohort_duration_days"],
            offtake_rate=d["offtake_rate"],
        ),
        calc_rem_maintenance: lambda: dict(species_short=sp, ration_digestibility_fraction=d["ration_digestibility_fraction"]),
        calc_reg_growth: lambda: dict(species_short=sp, ration_digestibility_fraction=d["ration_digestibility_fraction"]),
        calc_total_metabolic_energy_req: lambda: dict(
            species_short=sp, metabolic_energy_req_maintenance=d["maintenance"],
            metabolic_energy_req_activity=d["activity"], metabolic_energy_req_lactation=d["lactation"],
            metabolic_energy_req_work=d["work"], metabolic_energy_req_pregnancy=d["pregnancy"],
            net_energy_maintenance_digestible_energy_ratio=d["rem"], metabolic_energy_req_growth=d["growth"],
            metabolic_energy_req_fibre_production=d["fibre"], metabolic_energy_req_egg_deposition=d["eggs"],
            net_energy_growth_digestible_energy_ratio=d["reg"],
            ration_digestibility_fraction=d["ration_digestibility_fraction"],
        ),
        calc_ration_intake: lambda: dict(
            species_short=sp, metabolic_energy_req_total=d["total"],
            ration_gross_energy=d["ration_gross_energy"],
            ration_metabolizable_energy=d["ration_metabolizable_energy"],
        ),
    }


_OUTPUT_KEY = {
    calc_metabolic_energy_req_maintenance: "maintenance",
    calc_metabolic_energy_req_activity: "activity",
    calc_metabolic_energy_req_growth: "growth",
    calc_metabolic_energy_req_lactation: "lactation",
    calc_metabolic_energy_req_eggs: "eggs",
    calc_metabolic_energy_req_work: "work",
    calc_metabolic_energy_req_fibre: "fibre",
    calc_metabolic_energy_req_pregnancy: "pregnancy",
    calc_rem_maintenance: "rem",
    calc_reg_growth: "reg",
    calc_total_metabolic_energy_req: "total",
    calc_ration_intake: "ration_intake",
}


def _scalar(v):
    """Element of an input column as a plain Python scalar (str, float, bool or None)."""
    if v is None or isinstance(v, (str, bool)):
        return v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    v = float(v)
    return v


def test_vectorised_core_functions_match_elementwise_scalar_calls():
    d = _mixed_inputs()
    species = set(d["species_short"])
    assert {"CHK", "PGS", "CML", "CTL", "BFL", "SHP", "GTS"} <= species
    n = len(d["species_short"])

    # Evaluate the pipeline in dependency order, vectorised and element by element.
    for fn, key in _OUTPUT_KEY.items():
        args = _call_args(d)[fn]()
        vec = fn(**args)
        assert isinstance(vec, np.ndarray) and vec.shape == (n,), fn.__name__
        elem = np.array(
            [fn(**{k: _scalar(np.asarray(v)[i]) for k, v in args.items()}) for i in range(n)], dtype=float
        )
        assert all(isinstance(fn(**{k: _scalar(np.asarray(v)[i]) for k, v in args.items()}), float) for i in (0, n - 1))
        np.testing.assert_allclose(vec, elem, rtol=1e-14, atol=0, equal_nan=True, err_msg=fn.__name__)
        d[key] = vec

    # The run produces finite values everywhere except REM/REG of non-ruminants.
    assert np.isfinite(d["total"]).all() and np.isfinite(d["ration_intake"]).all()
    ruminant = np.isin(d["species_short"], ["CTL", "BFL", "SHP", "GTS"])
    assert np.isnan(d["rem"][~ruminant]).all() and np.isfinite(d["rem"][ruminant]).all()


def test_vectorised_maintenance_branches_per_element():
    sp = ["CTL", "BFL", "SHP", "SHP", "SHP", "GTS", "PGS", "CML", "CHK", "CHK", "CHK"]
    co = ["MA", "FA", "MS", "MJ", "FS", "MN", "FN", "MS", "FN", "FJ", "MA"]
    lw = np.array([600, 500, 40, 15, 35, 30, 80, 300, 1.8, 0.04, 2.4])
    egg = [None, None, None, None, None, None, None, None, True, False, False]
    out = calc_metabolic_energy_req_maintenance(
        sp, co, lw, lactating_females_fraction=0.5, offtake_rate=0.3, age_first_parturition=500,
        average_annual_temperature=[np.nan] * 8 + [15.0, 15.0, 25.0],
        lower_critical_temperature=[np.nan] * 8 + [18.89] * 3,
        nondemo_productive_phase_id=[np.nan] * 8 + [2, np.nan, np.nan], is_egg_producing=egg,
    )
    exp = [
        600 ** 0.75 * (0.322 * 0.3 + 0.37 * 0.7),
        500 ** 0.75 * (0.386 * 0.5 + 0.322 * 0.5),
        40 ** 0.75 * ((0.217 * 0.3 + 0.217 * 1.15 * 0.7) * (135 / 500) + (0.236 * 0.3 + 0.236 * 1.15 * 0.7) * (365 / 500)),
        15 ** 0.75 * (0.236 * 0.3 + 0.236 * 1.15 * 0.7),
        35 ** 0.75 * (0.236 * 365 / 500 + 0.217 * 135 / 500),
        30 ** 0.75 * 0.315,
        80 ** 0.75 * 0.4435,
        300 ** 0.75 * 0.435,
        max(0, 1.8 ** 0.75 * (0.6935 - 0.0099 * 15)),
        0.3866 + 0.0282 * (18.89 - 15),
        max(0, 2.4 ** 0.75 * (0.6935 - 0.0099 * 25)),
    ]
    np.testing.assert_allclose(out, exp, rtol=1e-12)


def test_chicken_adult_maintenance_is_floored_at_zero():
    out = calc_metabolic_energy_req_maintenance(
        "CHK", "MA", 2.0, average_annual_temperature=80, lower_critical_temperature=18.89,
        is_egg_producing=False,
    )
    assert out == 0


def test_chicken_young_maintenance_without_critical_temperature_errors_like_r():
    # R: `if (average_annual_temperature < lower_critical_temperature)` with NA
    with pytest.raises(ValueError, match="missing value where TRUE/FALSE needed"):
        calc_metabolic_energy_req_maintenance("CHK", "FJ", 0.04, average_annual_temperature=15)


def test_chicken_fa_gets_egg_energy_regardless_of_flag():
    # R computes egg deposition for every CHK FA row (flag only matters for FN).
    out = calc_metabolic_energy_req_eggs(
        "CHK", "FA", cohort_stock_size=100, egg_output_human_consumption=36500,
        egg_average_weight=0.06, parturition_rate=120, is_egg_producing=False,
    )
    assert out == approx(((36500 / 365 / 100) + (120 / 365)) * 0.06 * 10.04)


def test_pregnancy_small_ruminant_litter_size_below_one_gives_zero():
    out = calc_metabolic_energy_req_pregnancy(
        "SHP", "FA", 8.0, parturition_rate=1, litter_size=0.5, pregnancy_duration=150
    )
    assert out == 0


# ---- Validation ---------------------------------------------------------------


def test_validation_rejects_invalid_species_and_cohort():
    with pytest.raises(GleamValidationError, match="`species_short` must be one of"):
        calc_metabolic_energy_req_fibre("XXX", "FA", 1.0)
    with pytest.raises(GleamValidationError, match="`cohort_short` must be one of"):
        calc_metabolic_energy_req_fibre("SHP", "ZZ", 1.0)
    with pytest.raises(GleamValidationError, match="`species_short` must be one of"):
        calc_rem_maintenance(["CTL", "XXX"], 0.6)


def test_validation_checks_only_used_arguments():
    # Missing lactating fraction is fine for cohorts / species that do not use it ...
    assert calc_metabolic_energy_req_maintenance("CTL", "FJ", 200) == approx(200 ** 0.75 * 0.322)
    # ... but not for CTL FA.
    with pytest.raises(GleamValidationError, match="`lactating_females_fraction` must not contain missing values"):
        calc_metabolic_energy_req_maintenance("CTL", "FA", 500)
    with pytest.raises(GleamValidationError, match=r"`lactating_females_fraction`\[2\] = 1.5 is out of range"):
        calc_metabolic_energy_req_maintenance(["CTL", "CTL"], ["FA", "FA"], 500, lactating_females_fraction=[0.5, 1.5])
    with pytest.raises(GleamValidationError, match="`live_weight_cohort_average` must be positive"):
        calc_metabolic_energy_req_maintenance(["PGS", "CML"], "FA", [100, 0])
    with pytest.raises(GleamValidationError, match="`offtake_rate` = 1.2 is out of range"):
        calc_metabolic_energy_req_maintenance("SHP", "MJ", 15, offtake_rate=1.2)


def test_validation_of_activity_fractions():
    with pytest.raises(GleamValidationError, match="Sum of `low_activity_fraction` \\+ `high_activity_fraction`"):
        calc_metabolic_energy_req_activity("CTL", "FA", 10, 500, 0.7, 0.5)
    with pytest.raises(GleamValidationError, match="`metabolic_energy_req_maintenance` must be positive"):
        calc_metabolic_energy_req_activity("CTL", "FA", 0, 500, 0.5, 0.2)


def test_validation_of_growth_weights():
    with pytest.raises(GleamValidationError, match="cannot be lower than live_weight_cohort_initial"):
        calc_metabolic_energy_req_growth(
            "CTL", "FS", live_weight_cohort_average=100, live_weight_cohort_final=300,
            live_weight_cohort_initial=150, live_weight_mature_stage=500,
            daily_weight_gain=0.5, offtake_rate=0.1, cohort_duration_days=100,
        )
    with pytest.raises(GleamValidationError, match="live_weight_cohort_final cannot be lower"):
        calc_metabolic_energy_req_growth(
            "GTS", "FJ", live_weight_cohort_final=10, live_weight_cohort_initial=15,
            cohort_duration_days=100,
        )
    with pytest.raises(GleamValidationError, match="`daily_weight_gain` must be a single numeric value"):
        calc_metabolic_energy_req_growth("CHK", "FJ")


def test_validation_of_lactation_and_pregnancy_inputs():
    with pytest.raises(GleamValidationError, match="strictly less than `live_weight_at_weaning`"):
        calc_metabolic_energy_req_lactation(
            "CTL", "FA", lactating_females_fraction=0.8, milk_yield_day=20, milk_fat_fraction=0.04,
            live_weight_at_birth=90, live_weight_at_weaning=35, parturition_rate=0.8,
        )
    with pytest.raises(GleamValidationError, match="`death_rate_juvenile` must be between 0 and 1"):
        calc_metabolic_energy_req_lactation(
            "PGS", "FA", non_productive_duration=10, pregnancy_duration=115, litter_size=10,
            death_rate_juvenile=1.5, live_weight_at_birth=1.5, live_weight_at_weaning=8,
            lactation_duration=30, parturition_rate=2,
        )
    # Offtake is clamped to [0, 1] before validation, and 1 is outside [-2, 1).
    with pytest.raises(GleamValidationError, match="`offtake_rate` = 1 is out of range"):
        calc_metabolic_energy_req_pregnancy(
            "CTL", "FS", 12.0, parturition_rate=0.8, pregnancy_duration=283,
            cohort_duration_days=730, offtake_rate=1.4,
        )


def test_validation_of_total_and_intake():
    with pytest.raises(GleamValidationError, match="`net_energy_maintenance_digestible_energy_ratio` must be a single numeric"):
        _total("CTL", 15.0, 3.0, 8.0, 0, 1.5, np.nan, 0, 0, 0, 0.5, 0.65)
    with pytest.raises(GleamValidationError, match="`metabolic_energy_req_total` must be positive"):
        calc_ration_intake("CTL", 0, 18, 12)
    with pytest.raises(GleamValidationError, match="`ration_gross_energy` = 50 is out of range"):
        calc_ration_intake("CTL", 10, 50, 12)


def test_validation_can_be_disabled():
    with validation_disabled():
        out = calc_ration_intake("CTL", -10, 50, 12)
    assert out == approx(-10 / 50)


# --------------------------------------------------------------------------
# R's ^ (R_pow) in the growth and maintenance equations
# --------------------------------------------------------------------------

# x, y, R's x ^ y (sprintf("%a") in R 4.6) for the cases where C pow, and so
# numpy's **, differs from R: negative infinite base, negative base with an
# infinite exponent, signed zero.
_R_POW_EDGES = (
    ("-Inf", "-Inf", "NaN"),
    ("-Inf", "-0x1.8p+1", "0x0p+0"),
    ("-Inf", "-0x1.8p-1", "NaN"),
    ("-Inf", "-0x1p-1", "NaN"),
    ("-Inf", "0x1.8p-1", "NaN"),
    ("-Inf", "0x1.18d4fdf3b645ap+0", "NaN"),
    ("-Inf", "Inf", "NaN"),
    ("-Inf", "0x1p+1", "Inf"),
    ("-Inf", "0x1.8p+1", "-Inf"),
    ("-0x1p+1", "-Inf", "NaN"),
    ("-0x1p+1", "Inf", "NaN"),
    ("-0x1p+0", "-Inf", "NaN"),
    ("-0x1p+0", "Inf", "NaN"),
    ("-0x1p-1", "-Inf", "NaN"),
    ("-0x1p-1", "Inf", "NaN"),
    ("-0x1p+1", "0x1.8p-1", "NaN"),
    ("-0x0p+0", "-0x1.8p+1", "Inf"),
    ("-0x0p+0", "0x1p-1", "0x0p+0"),
    ("-0x0p+0", "0x1.8p-1", "0x0p+0"),
    ("-0x0p+0", "0x1.8p+1", "0x0p+0"),
    ("0x0p+0", "-0x1.8p-1", "Inf"),
    ("0x1p+0", "NaN", "0x1p+0"),
    ("NaN", "0x0p+0", "0x1p+0"),
    ("0x1p-1", "Inf", "0x0p+0"),
    ("0x1p+1", "-Inf", "0x0p+0"),
    ("Inf", "-0x1.8p-1", "0x0p+0"),
)


def _r_num(s):
    return {"Inf": math.inf, "-Inf": -math.inf, "NaN": math.nan}.get(s) if s in ("Inf", "-Inf", "NaN") \
        else float.fromhex(s)


def test_power_follows_r_pow_edge_cases():
    from gleampy.core.metabolic_energy_req import _r_pow

    x = np.array([_r_num(c[0]) for c in _R_POW_EDGES])
    y = np.array([_r_num(c[1]) for c in _R_POW_EDGES])
    expected = np.array([_r_num(c[2]) for c in _R_POW_EDGES])
    got = _r_pow(x, y)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(np.signbit(got), np.signbit(expected))
    # Finite inputs: exactly numpy's ** (C pow), so valid results are unchanged.
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 800, 1000)
    for e in (0.75, 1.097):
        np.testing.assert_array_equal(_r_pow(base, e), base**e)


def test_growth_with_negative_zero_mature_weight_is_nan_like_r():
    # R: (lw_avg / (0.8 * -0))^0.75 = (-Inf)^0.75 = NaN; numpy's ** gives +Inf.
    args = ("CTL", "FS", 449.8095, 649.619, 250)
    kw = {"daily_weight_gain": 0.6056, "offtake_rate": 0.247, "cohort_duration_days": 710}
    assert math.isnan(calc_metabolic_energy_req_growth(*args, live_weight_mature_stage=-0.0, **kw))
    assert calc_metabolic_energy_req_growth(*args, live_weight_mature_stage=0.0, **kw) == math.inf
    # MN mature weights are not range-checked: a tiny negative one overflows to -Inf.
    with validation_disabled():
        assert math.isnan(
            calc_metabolic_energy_req_growth("CTL", "MN", *args[2:], live_weight_mature_stage=-5e-324, **kw)
        )


def test_module_rejects_nan_growth_from_negative_zero_mature_weight():
    cohort = load_example("metabolic_energy_req_input_chrt_data.csv")
    herd = load_example("metabolic_energy_req_input_hrd_data.csv")
    c = cohort[cohort["herd_id"] == 1].reset_index(drop=True)
    h = herd[herd["herd_id"] == 1].reset_index(drop=True)
    c["live_weight_mature_stage"] = c["live_weight_mature_stage"].astype(float)
    c.loc[c["cohort_short"] == "FS", "live_weight_mature_stage"] = -0.0
    # R stops in validate_total_energy_inputs.
    with pytest.raises(GleamValidationError, match="`metabolic_energy_req_growth` must be a single numeric value"):
        gleampy.run_metabolic_energy_req_module(c, h, show_indicator=False)
    with pytest.warns(gleampy.GleamWarning):
        out = gleampy.run_metabolic_energy_req_module(c, h, show_indicator=False, validate_inputs=False)
    fs = out[out["cohort_short"] == "FS"].iloc[0]
    assert math.isnan(fs["metabolic_energy_req_growth"]) and math.isnan(fs["ration_intake"])


# --------------------------------------------------------------------------
# Behaviours replicated from R
# --------------------------------------------------------------------------


def test_laying_cohort_with_zero_stock_gets_infinite_egg_energy_like_r():
    # R: Inf (and NaN for 0 / 0); the core validators accept a stock size of 0.
    assert calc_metabolic_energy_req_eggs("CHK", "FA", 0, 180000, 0.06, 45, is_egg_producing=True) == math.inf
    assert math.isnan(calc_metabolic_energy_req_eggs("CHK", "FA", 0, 0, 0.06, 45, is_egg_producing=True))


def test_growth_of_chicken_adults_is_not_zero():
    # R: calc_metabolic_energy_req_growth("CHK", "MA", daily_weight_gain = 0.01)
    assert calc_metabolic_energy_req_growth("CHK", "MA", daily_weight_gain=0.01) == float.fromhex(
        "0x1.1db22d0e56042p-2"
    )
    assert calc_metabolic_energy_req_growth("CTL", "FA") == 0.0
    with validation_disabled():
        assert math.isnan(calc_metabolic_energy_req_growth("SHP", "FA", 20, np.nan, 10, 50, 0.1, 0.1, 100))


# --------------------------------------------------------------------------
# Docstrings of the public functions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "core.metabolic_energy_req", "modules.metabolic_energy_req",
        "core.nitrogen_balance", "modules.nitrogen_balance",
        "core.emissions_enteric", "modules.emissions_enteric",
        "core.ration_quality", "modules.ration_quality",
        "core.emissions_ration", "modules.emissions_ration",
        "core.weights", "modules.weights",
    ],
)
def test_public_functions_document_every_parameter(module):
    import importlib
    import inspect
    import re

    mod = importlib.import_module(f"gleampy.{module}")
    public = [
        fn for name, fn in inspect.getmembers(mod, inspect.isfunction)
        if fn.__module__ == mod.__name__ and name.startswith(("calc_", "run_"))
    ]
    assert public
    for fn in public:
        doc = inspect.getdoc(fn) or ""
        section = re.search(r"Parameters\n-+\n(.*?)(?:\n\n[A-Z][a-z]+\n-+\n|\Z)", doc, re.S)
        assert section, f"{fn.__name__}: no Parameters section"
        assert "Returns\n-------" in doc, f"{fn.__name__}: no Returns section"
        documented = set()
        lines = section.group(1).splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^(\w[\w, ]*) : ", line)
            if m:
                described = i + 1 < len(lines) and lines[i + 1].startswith("    ")
                assert described, f"{fn.__name__}: {m.group(1)} has no description"
                documented.update(p.strip() for p in m.group(1).split(","))
        missing = [p for p in inspect.signature(fn).parameters if p not in documented]
        assert not missing, f"{fn.__name__}: undocumented {missing}"
