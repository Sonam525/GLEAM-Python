"""Golden parity: ``run_nitrogen_balance_module`` vs the R package.

Mirrors ``tools/r_reference/generate_golden.R`` (case ``nitrogen_balance_module``).
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest
from golden_utils import assert_matches_golden

from gleampy import GleamValidationError, GleamWarning, load_example, run_nitrogen_balance_module


def test_nitrogen_balance_module_matches_r():
    result = run_nitrogen_balance_module(
        cohort_level_data=load_example("nitrogen_balance_input_chrt_data.csv"),
        herd_level_data=load_example("nitrogen_balance_input_hrd_data.csv"),
        show_indicator=False,
    )
    assert_matches_golden(result, "nitrogen_balance_module", "result")


# --------------------------------------------------------------------------
# Herd columns that some row needs
# --------------------------------------------------------------------------
#
# R passes every herd column as a lazy `herd_level_data[.SD, on = "herd_id",
# x.col]` promise. The milk and fibre columns are read for every row of a
# milk-producing species (FN / MN included) and the birth / weaning check of
# the validator reads the birth weight of every non-CHK row, so R stops when
# such a column is absent even if the run validator did not require it. The
# port raises GleamValidationError instead of silently using NA (which turned
# the fibre or milk N into 0).


def _nondemo_cohort(herd_id="h1"):
    return pd.DataFrame(
        {
            "herd_id": [herd_id, herd_id],
            "cohort_short": ["FN", "MN"],
            "ration_intake": [1.0, 1.2],
            "ration_nitrogen": [0.02, 0.02],
            "daily_weight_gain": [0.15, 0.2],
            "cohort_duration_days": [100, 100],
            "cohort_stock_size": [10, 10],
        }
    )


_SHP_HERD = pd.DataFrame(
    {
        "herd_id": ["h1"],
        "species_short": ["SHP"],
        "milk_protein_fraction": [0.05],
        "milk_yield_day": [0.0],
        "fibre_yield_year": [3.0],
        "live_weight_at_weaning": [20.0],
        "live_weight_at_birth": [4.0],
    }
)


@pytest.mark.parametrize("validate", [True, False])
def test_nondemographic_sheep_needs_fibre_yield_like_r(validate):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        out = run_nitrogen_balance_module(
            _nondemo_cohort(), _SHP_HERD, show_indicator=False, validate_inputs=validate
        )
        # R (sprintf("%a")): 0.0050013698630137, 0.0063013698630137
        assert out["nitrogen_retention"].tolist() == [
            float.fromhex("0x1.47c50fff31285p-8"), float.fromhex("0x1.9cf7717b4d02ap-8"),
        ]
        # R: "column name 'x.fibre_yield_year' is not found"; the port used to
        # return [0.0039, 0.0052] (fibre N dropped).
        with pytest.raises(
            GleamValidationError, match='Missing required columns in `herd_level_data`: "fibre_yield_year"'
        ):
            run_nitrogen_balance_module(
                _nondemo_cohort(), _SHP_HERD.drop(columns="fibre_yield_year"),
                show_indicator=False, validate_inputs=validate,
            )
        for col in ("milk_protein_fraction", "milk_yield_day"):
            with pytest.raises(GleamValidationError, match=f'"{col}"'):
                run_nitrogen_balance_module(
                    _nondemo_cohort(), _SHP_HERD.drop(columns=col),
                    show_indicator=False, validate_inputs=validate,
                )


def test_birth_weight_is_read_by_the_validator_for_every_non_chicken_row():
    minimal = pd.DataFrame({"herd_id": ["h1"], "species_short": ["PGS"]})
    # R's validator stops on 'x.live_weight_at_birth'.
    with pytest.raises(
        GleamValidationError, match='Missing required columns in `herd_level_data`: "live_weight_at_birth"'
    ):
        run_nitrogen_balance_module(_nondemo_cohort(), minimal, show_indicator=False)
    with pytest.raises(GleamValidationError, match='"live_weight_at_weaning"'):
        run_nitrogen_balance_module(
            _nondemo_cohort(), minimal.assign(live_weight_at_birth=1.5), show_indicator=False
        )
    # Without validation PGS FN / MN rows only read daily_weight_gain.
    with pytest.warns(GleamWarning):
        out = run_nitrogen_balance_module(_nondemo_cohort(), minimal, show_indicator=False, validate_inputs=False)
    assert out["nitrogen_retention"].tolist() == pytest.approx([0.025 * 0.15, 0.025 * 0.2], rel=1e-15)


@pytest.mark.parametrize("col", ["milk_yield_day", "fibre_yield_year"])
@pytest.mark.parametrize("validate", [True, False])
def test_missing_milk_or_fibre_column_is_an_error_without_validation_too(col, validate):
    # Dropping milk_yield_day used to give herd 1 CTL FA a retention of 0 and a
    # doubled excretion when validation was off; R's calc body stops.
    cohort = load_example("nitrogen_balance_input_chrt_data.csv")
    herd = load_example("nitrogen_balance_input_hrd_data.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        with pytest.raises(GleamValidationError, match=f'Missing required columns in `herd_level_data`: "{col}"'):
            run_nitrogen_balance_module(cohort, herd.drop(columns=col), show_indicator=False, validate_inputs=validate)


def test_chicken_herd_without_egg_flag_or_egg_columns():
    chk = pd.DataFrame(
        {"herd_id": ["h1"], "species_short": ["CHK"], "egg_output_human_consumption": [250000.0],
         "egg_average_weight": [0.06]}
    )
    # R: "object 'is_egg_producing' not found"
    with pytest.raises(
        GleamValidationError, match='Missing required columns in `cohort_level_data`: "is_egg_producing"'
    ):
        run_nitrogen_balance_module(_nondemo_cohort(), chk, show_indicator=False)
    laying = _nondemo_cohort().assign(
        nondemo_productive_phase_id=[2, 1], is_egg_producing=[True, False], cohort_stock_size=[1000, 1000]
    )
    # A laying hen reads parturition_rate (R: 'x.parturition_rate' is not found).
    with pytest.raises(
        GleamValidationError, match='Missing required columns in `herd_level_data`: "parturition_rate"'
    ):
        run_nitrogen_balance_module(laying, chk, show_indicator=False)
    out = run_nitrogen_balance_module(laying, chk.assign(parturition_rate=0.0), show_indicator=False)
    egg = (250000.0 / 365 / 1000 + 0.0 / 365) * 0.06 * 0.02
    assert out["nitrogen_retention"].tolist() == pytest.approx([0.15 * 0.032 + egg, 0.2 * 0.032], rel=1e-15)
    # Non-laying chickens need neither the egg columns nor parturition_rate.
    not_laying = laying.assign(is_egg_producing=[False, False])
    out = run_nitrogen_balance_module(not_laying, chk[["herd_id", "species_short"]], show_indicator=False)
    assert out["nitrogen_retention"].tolist() == pytest.approx([0.15 * 0.032, 0.2 * 0.032], rel=1e-15)
