"""Golden parity of :func:`gleampy.run_weights_module` with the R package.

Runs the bundled example exactly as ``tools/r_reference/generate_golden.R``
(case ``weights_module``) and compares every output table.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from golden_utils import assert_matches_golden, golden_tables

import gleampy
from gleampy import GleamValidationError, GleamWarning
from gleampy.io import load_example

CASE = "weights_module"


@pytest.fixture(scope="module")
def inputs():
    return load_example("weights_input_chrt_data.csv"), load_example("weights_input_hrd_data.csv")


@pytest.fixture(scope="module")
def result(inputs):
    cohort, herd = inputs
    return gleampy.run_weights_module(cohort, herd, show_indicator=False)


def test_output_tables(result):
    assert list(result) == ["cohort_level_results", "herd_level_results"]
    assert sorted(result) == golden_tables(CASE)


@pytest.mark.parametrize("table", ["cohort_level_results", "herd_level_results"])
def test_matches_golden(result, table):
    assert_matches_golden(result[table], CASE, table)


def test_inputs_not_mutated(inputs):
    cohort, herd = inputs
    c0, h0 = cohort.copy(), herd.copy()
    gleampy.run_weights_module(cohort, herd, show_indicator=False)
    pd.testing.assert_frame_equal(cohort, c0)
    pd.testing.assert_frame_equal(herd, h0)


def test_unvalidated_run_matches_and_warns(inputs, result):
    cohort, herd = inputs
    with pytest.warns(GleamWarning, match="Input validation has been turned off"):
        res = gleampy.run_weights_module(cohort, herd, show_indicator=False, validate_inputs=False)
    for table in result:
        pd.testing.assert_frame_equal(res[table], result[table])


def test_adds_phase_column_when_absent(inputs):
    cohort, herd = inputs
    demo = cohort[~cohort["cohort_short"].isin(["FN", "MN"])].drop(columns="nondemo_productive_phase_id")
    res = gleampy.run_weights_module(demo, herd, show_indicator=False)["cohort_level_results"]
    assert list(res.columns[: demo.shape[1] + 1]) == list(demo.columns) + ["nondemo_productive_phase_id"]
    assert res["nondemo_productive_phase_id"].isna().all()


def test_show_indicator_prints_progress(inputs, capsys):
    cohort, herd = inputs
    gleampy.run_weights_module(cohort, herd, show_indicator=True)
    assert "Cohort weights calculation complete." in capsys.readouterr().err


def test_run_validation_errors(inputs):
    cohort, herd = inputs
    with pytest.raises(GleamValidationError, match="`cohort_level_data` must be a data.table"):
        gleampy.run_weights_module(cohort.to_dict(), herd, show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `cohort_level_data`"):
        gleampy.run_weights_module(cohort.drop(columns="offtake_rate"), herd, show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `herd_level_data`"):
        gleampy.run_weights_module(cohort, herd.drop(columns="live_weight_at_birth"), show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `herd_level_data`"):
        gleampy.run_weights_module(
            cohort, herd.drop(columns="phase1_nondemo_fem_duration_days"), show_indicator=False
        )
    with pytest.raises(GleamValidationError, match="Each herd_id must appear exactly once"):
        gleampy.run_weights_module(cohort, pd.concat([herd, herd.iloc[[0]]]), show_indicator=False)
    with pytest.raises(GleamValidationError, match="not found in `herd_level_data`"):
        gleampy.run_weights_module(cohort, herd.iloc[1:], show_indicator=False)

    bad = herd.copy()
    bad.loc[1, "live_weight_at_birth"] = 300.0  # herd 2: above female slaughter weight
    with pytest.raises(
        GleamValidationError,
        match="`live_weight_at_birth` must be less than `live_weight_female_at_slaughter`. "
        "Violation\\(s\\) for herd_id: 2",
    ):
        gleampy.run_weights_module(cohort, bad, show_indicator=False)

    bad = herd.copy()
    bad.loc[1, "live_weight_at_weaning"] = 14.0  # herd 2: weaning == birth
    with pytest.raises(GleamValidationError, match="must be less than `live_weight_at_weaning`"):
        gleampy.run_weights_module(cohort, bad, show_indicator=False)

    # chickens are exempt from the birth < weaning rule (herd 13 has 0.04 / 0.04)
    assert herd.loc[herd["herd_id"] == 13, "live_weight_at_weaning"].item() == 0.04


def test_row_level_validation_inside_run(inputs):
    cohort, herd = inputs
    bad = cohort.copy()
    bad.loc[0, "cohort_duration_days"] = 0
    with pytest.raises(GleamValidationError, match="`cohort_duration_days`\\[1\\] = 0 is out of range"):
        gleampy.run_weights_module(bad, herd, show_indicator=False)


def test_validation_off_skips_checks(inputs):
    cohort, herd = inputs
    bad = cohort.copy()
    bad.loc[0, "cohort_duration_days"] = 0  # herd 1 FA: gain 0 / 0 -> NaN, as in R
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        res = gleampy.run_weights_module(bad, herd, show_indicator=False, validate_inputs=False)
    assert np.isnan(res["cohort_level_results"]["daily_weight_gain"].iloc[0])


# --------------------------------------------------------------------------
# Integer-typed weight columns
# --------------------------------------------------------------------------
#
# R assigns the cohort weights with `:=` and `by = .I`, which gives each output
# column the type of the first row's value: when fread reads a whole-number
# weight column as integer, R truncates the fractional weights of later rows
# (FJ birth weight 1.2 -> 1, FS slaughter weight 110.5 -> 110, ...) and turns
# values beyond the integer range into NA. That is an R bug the port does not
# replicate: whatever the input dtypes, the results are the ones R gives for
# double-typed inputs. The reference values below were computed with R
# (run_weights_module on the same tables with every number stored as double)
# and are written as hex literals so the comparison is exact.

_WEIGHT_OUTPUTS = (
    "live_weight_mature_stage",
    "live_weight_cohort_initial",
    "live_weight_cohort_potential_final",
    "live_weight_cohort_at_slaughter",
    "live_weight_cohort_average",
    "live_weight_cohort_final",
    "daily_weight_gain",
)

_NONDEMO_NA = {
    "live_weight_female_nondemographic_start": np.nan,
    "live_weight_male_nondemographic_start": np.nan,
    "live_weight_female_nondemographic_end": np.nan,
    "live_weight_male_nondemographic_end": np.nan,
    "phase1_nondemo_fem_duration_days": np.nan,
    "phase2_nondemo_fem_duration_days": np.nan,
    "phase1_nondemo_mal_duration_days": np.nan,
    "phase2_nondemo_mal_duration_days": np.nan,
}

# PGS herd with whole-number adult weights (225 / 265 kg) and fractional
# birth, weaning and slaughter weights; R (double inputs), cohorts in this order.
_R_PGS_DEMOGRAPHIC = (
    ("FA", "0x1.c2p+7", "0x1.c2p+7", "0x1.c2p+7", "0x1.c2p+7", "0x1.c2p+7", "0x1.c2p+7", "0x0p+0"),
    ("FJ", "0x1.c2p+7", "0x1.3333333333333p+0", "0x1.ep+2", "0x1.ep+2", "0x1.1666666666666p+2",
     "0x1.ep+2", "0x1.ddddddddddddep-3"),
    ("FS", "0x1.c2p+7", "0x1.ep+2", "0x1.c2p+7", "0x1.bap+6", "0x1.efd0e56041894p+5",
     "0x1.d1d0e56041894p+6", "0x1.3631ea9b743bep-1"),
    ("MA", "0x1.09p+8", "0x1.09p+8", "0x1.09p+8", "0x1.09p+8", "0x1.09p+8", "0x1.09p+8", "0x0p+0"),
    ("MJ", "0x1.09p+8", "0x1.3333333333333p+0", "0x1.ep+2", "0x1.ep+2", "0x1.1666666666666p+2",
     "0x1.ep+2", "0x1.ddddddddddddep-3"),
    ("MS", "0x1.09p+8", "0x1.ep+2", "0x1.09p+8", "0x1.cep+6", "0x1.fc25604189375p+5",
     "0x1.de25604189375p+6", "0x1.6f3e09fbb8b0ep-1"),
)

# The same herd with an FS slaughter weight of 1e300 (beyond the integer range;
# R's parser reads the literal 1e300 as 0x1.7e43c880075ap+996). Only FS changes.
_R_FS_HUGE = (
    "FS", "0x1.c2p+7", "0x1.ep+2", "0x1.c2p+7", "0x1.7e43c880075ap+996", "0x1.6a631310ec586p+995",
    "0x1.6a631310ec586p+996", "0x1.3631ea9b743bep-1",
)

# All-integer PGS and CHK herds with FN / MN rows (every number column read by
# fread as integer); R with double inputs: initial, potential final, final and
# daily gain, in cohort-table order.
_HERD_INT_CSV = (
    "herd_id,species_short,live_weight_female_adult,live_weight_male_adult,live_weight_at_birth,"
    "live_weight_female_at_slaughter,live_weight_male_at_slaughter,live_weight_at_weaning,"
    "live_weight_female_nondemographic_start,live_weight_male_nondemographic_start,"
    "live_weight_female_nondemographic_end,live_weight_male_nondemographic_end,"
    "phase1_nondemo_fem_duration_days,phase2_nondemo_fem_duration_days,"
    "phase1_nondemo_mal_duration_days,phase2_nondemo_mal_duration_days\n"
    "1,PGS,225,265,1,122,122,7,7,7,120,121,60,110,60,100\n"
    "2,CHK,2,3,1,2,2,1,1,1,3,4,30,40,25,45\n"
)
_COHORT_INT_CSV = (
    "herd_id,cohort_short,cohort_duration_days,offtake_rate,nondemo_productive_phase_id\n"
    "1,FA,890,0,\n1,FJ,27,0,\n1,FN,60,1,1\n1,FN,110,1,2\n1,FS,359,0.948,\n"
    "1,MA,890,0,\n1,MJ,27,0,\n1,MN,60,1,1\n1,MN,100,1,2\n1,MS,359,0.973,\n"
    "2,FA,365,0.18,\n2,FJ,3,0.08,\n2,FN,30,1,1\n2,FN,40,1,2\n2,FS,167,0.12,\n"
    "2,MA,365,0.75,\n2,MJ,3,0.12,\n2,MN,25,1,1\n2,MN,45,1,2\n2,MS,167,0.7,\n"
)
_R_ALL_INTEGER = (
    ("0x1.c2p+7", "0x1.c2p+7", "0x1.c2p+7", "0x0p+0"),
    ("0x1p+0", "0x1.cp+2", "0x1.cp+2", "0x1.c71c71c71c71cp-3"),
    ("0x1.cp+2", "0x1.770f0f0f0f0f1p+5", "0x1.770f0f0f0f0f1p+5", "0x1.5454545454545p-1"),
    ("0x1.770f0f0f0f0f1p+5", "0x1.ep+6", "0x1.ep+6", "0x1.5454545454546p-1"),
    ("0x1.cp+2", "0x1.c2p+7", "0x1.fd6c8b439581p+6", "0x1.36e877cca84a3p-1"),
    ("0x1.09p+8", "0x1.09p+8", "0x1.09p+8", "0x0p+0"),
    ("0x1p+0", "0x1.cp+2", "0x1.cp+2", "0x1.c71c71c71c71cp-3"),
    ("0x1.cp+2", "0x1.8ep+5", "0x1.8ep+5", "0x1.6cccccccccccdp-1"),
    ("0x1.8ep+5", "0x1.e4p+6", "0x1.e4p+6", "0x1.6cccccccccccdp-1"),
    ("0x1.cp+2", "0x1.09p+8", "0x1.f771a9fbe76c9p+6", "0x1.6ff4972cecbf2p-1"),
    ("0x1p+1", "0x1p+1", "0x1p+1", "0x0p+0"),
    ("0x1p+0", "0x1p+0", "0x1p+0", "0x0p+0"),
    ("0x1p+0", "0x1.db6db6db6db6ep+0", "0x1.db6db6db6db6ep+0", "0x1.d41d41d41d41ep-6"),
    ("0x1.db6db6db6db6ep+0", "0x1.8p+1", "0x1.8p+1", "0x1.d41d41d41d41dp-6"),
    ("0x1p+0", "0x1p+1", "0x1p+1", "0x1.886e5f0abb04ap-8"),
    ("0x1.8p+1", "0x1.8p+1", "0x1.8p+1", "0x0p+0"),
    ("0x1p+0", "0x1p+0", "0x1p+0", "0x0p+0"),
    ("0x1p+0", "0x1.0924924924924p+1", "0x1.0924924924924p+1", "0x1.5f15f15f15f14p-5"),
    ("0x1.0924924924924p+1", "0x1p+2", "0x1p+2", "0x1.5f15f15f15f17p-5"),
    ("0x1p+0", "0x1.8p+1", "0x1.2666666666666p+1", "0x1.886e5f0abb04ap-7"),
)


def _hex_table(rows):
    return np.array([[float.fromhex(v) for v in row] for row in rows])


def _pgs_herd(female_adult, male_adult, female_at_slaughter=110.5):
    return pd.DataFrame(
        {
            "herd_id": [1],
            "species_short": ["PGS"],
            "live_weight_female_adult": female_adult,
            "live_weight_male_adult": male_adult,
            "live_weight_at_birth": [1.2],
            "live_weight_female_at_slaughter": [female_at_slaughter],
            "live_weight_male_at_slaughter": [115.5],
            "live_weight_at_weaning": [7.5],
            **{k: [v] for k, v in _NONDEMO_NA.items()},
        }
    )


_PGS_COHORTS = pd.DataFrame(
    {
        "herd_id": 1,
        "cohort_short": ["FA", "FJ", "FS", "MA", "MJ", "MS"],
        "cohort_duration_days": [890, 27, 359, 890, 27, 359],
        "offtake_rate": [0, 0.1, 0.948, 0, 0.1, 0.973],
    }
)


@pytest.mark.parametrize("dtype", ["int64", "Int64", "object", "float64"])
@pytest.mark.parametrize("validate", [True, False])
def test_integer_typed_herd_weights_are_never_truncated(dtype, validate):
    # In R the FA first row copies the integer adult weight, so the output
    # columns become integer and later rows are truncated (FJ 1.2 -> 1,
    # 7.5 -> 7, FS / MS slaughter 110.5 -> 110, 115.5 -> 115).
    herd = _pgs_herd(pd.Series([225]).astype(dtype), pd.Series([265]).astype(dtype))
    assert herd["live_weight_female_adult"].dtype == dtype
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        res = gleampy.run_weights_module(_PGS_COHORTS, herd, show_indicator=False, validate_inputs=validate)
    out = res["cohort_level_results"]
    assert out["cohort_short"].tolist() == [r[0] for r in _R_PGS_DEMOGRAPHIC]
    for col in _WEIGHT_OUTPUTS:
        assert out[col].dtype == np.float64
    np.testing.assert_array_equal(
        out[list(_WEIGHT_OUTPUTS)].to_numpy(), _hex_table(r[1:] for r in _R_PGS_DEMOGRAPHIC)
    )
    fj = out[out["cohort_short"] == "FJ"].iloc[0]
    assert (fj["live_weight_cohort_initial"], fj["live_weight_cohort_potential_final"]) == (1.2, 7.5)


def test_integer_typed_herd_keeps_weights_beyond_the_integer_range():
    # R stores NA for 1e300 in an integer output column, then fails in
    # validate_avg_weight_inputs; double inputs give these values.
    huge = float.fromhex(_R_FS_HUGE[4])
    herd = _pgs_herd(pd.Series([225]), pd.Series([265]), huge)
    assert herd["live_weight_female_adult"].dtype == np.int64
    out = gleampy.run_weights_module(_PGS_COHORTS, herd, show_indicator=False)["cohort_level_results"]
    expected = [r[1:] for r in _R_PGS_DEMOGRAPHIC]
    expected[2] = _R_FS_HUGE[1:]
    np.testing.assert_array_equal(out[list(_WEIGHT_OUTPUTS)].to_numpy(), _hex_table(expected))


@pytest.mark.parametrize("dtype", ["int64", "Int64", "float64"])
def test_all_integer_csv_with_nondemographic_rows_matches_r_double(tmp_path, dtype):
    from gleampy.io import read_csv

    (tmp_path / "herd.csv").write_text(_HERD_INT_CSV)
    (tmp_path / "cohort.csv").write_text(_COHORT_INT_CSV)
    herd = read_csv(tmp_path / "herd.csv")
    cohort = read_csv(tmp_path / "cohort.csv")
    # read_csv types every whole-number column as int64, like fread's integer.
    assert herd["live_weight_female_adult"].dtype == np.int64
    assert cohort["cohort_duration_days"].dtype == np.int64
    numeric = [c for c in herd.columns if c not in ("herd_id", "species_short")]
    herd[numeric] = herd[numeric].astype(dtype)
    cohort["cohort_duration_days"] = cohort["cohort_duration_days"].astype(dtype)
    out = gleampy.run_weights_module(cohort, herd, show_indicator=False)["cohort_level_results"]
    cols = [
        "live_weight_cohort_initial", "live_weight_cohort_potential_final",
        "live_weight_cohort_final", "daily_weight_gain",
    ]
    np.testing.assert_array_equal(out[cols].to_numpy(dtype="float64"), _hex_table(_R_ALL_INTEGER))


# --------------------------------------------------------------------------
# Herd columns needed by every row
# --------------------------------------------------------------------------

_NONDEMO_HERD_COLS = list(_NONDEMO_NA)


def test_demographic_only_data_needs_all_herd_weight_columns_with_validation(inputs):
    # R's validate_cohort_weight_inputs() reads all 14 herd columns for every
    # row, so R stops ("column name 'x.live_weight_female_nondemographic_start'
    # is not found") even without FN / MN rows.
    cohort, herd = inputs
    demo = cohort[~cohort["cohort_short"].isin(["FN", "MN"])]
    expected = "Missing required columns in `herd_level_data`: " + ", ".join(
        f'"{c}"' for c in _HERD_WEIGHT_ORDER if c in _NONDEMO_HERD_COLS
    )
    with pytest.raises(GleamValidationError) as err:
        gleampy.run_weights_module(demo, herd.drop(columns=_NONDEMO_HERD_COLS), show_indicator=False)
    assert str(err.value) == expected
    # All-NA columns are fine (as in R).
    na_herd = herd.assign(**{c: np.nan for c in _NONDEMO_HERD_COLS})
    res = gleampy.run_weights_module(demo, na_herd, show_indicator=False)
    assert len(res["cohort_level_results"]) == len(demo)


_HERD_WEIGHT_ORDER = (
    "live_weight_female_adult", "live_weight_male_adult", "live_weight_at_birth",
    "live_weight_female_at_slaughter", "live_weight_male_at_slaughter", "live_weight_at_weaning",
    "live_weight_female_nondemographic_start", "live_weight_male_nondemographic_start",
    "live_weight_female_nondemographic_end", "live_weight_male_nondemographic_end",
    "phase1_nondemo_fem_duration_days", "phase2_nondemo_fem_duration_days",
    "phase1_nondemo_mal_duration_days", "phase2_nondemo_mal_duration_days",
)


def test_without_validation_only_the_columns_a_row_uses_are_needed(inputs):
    cohort, herd = inputs
    demo = cohort[~cohort["cohort_short"].isin(["FN", "MN"])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        full = gleampy.run_weights_module(demo, herd, show_indicator=False, validate_inputs=False)
        res = gleampy.run_weights_module(
            demo, herd.drop(columns=_NONDEMO_HERD_COLS), show_indicator=False, validate_inputs=False
        )
        pd.testing.assert_frame_equal(res["cohort_level_results"], full["cohort_level_results"])
        # ... but a column that some row uses must be there (R stops too).
        with pytest.raises(
            GleamValidationError,
            match='Missing required columns in `herd_level_data`: "live_weight_female_at_slaughter"',
        ):
            gleampy.run_weights_module(
                demo, herd.drop(columns="live_weight_female_at_slaughter"),
                show_indicator=False, validate_inputs=False,
            )
        with pytest.raises(
            GleamValidationError, match='Missing required columns in `herd_level_data`: "species_short"'
        ):
            gleampy.run_weights_module(
                demo, herd.drop(columns="species_short"), show_indicator=False, validate_inputs=False
            )
        with pytest.raises(
            GleamValidationError, match='Missing required columns in `cohort_level_data`: "offtake_rate"'
        ):
            gleampy.run_weights_module(
                demo.drop(columns="offtake_rate"), herd, show_indicator=False, validate_inputs=False
            )
