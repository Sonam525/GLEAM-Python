"""Golden parity of :func:`gleam.run_weights_module` with the R package.

Runs the bundled example exactly as ``tools/r_reference/generate_golden.R``
(case ``weights_module``) and compares every output table.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from golden_utils import assert_matches_golden, golden_tables

import gleam
from gleam import GleamValidationError, GleamWarning
from gleam.io import load_example

CASE = "weights_module"


@pytest.fixture(scope="module")
def inputs():
    return load_example("weights_input_chrt_data.csv"), load_example("weights_input_hrd_data.csv")


@pytest.fixture(scope="module")
def result(inputs):
    cohort, herd = inputs
    return gleam.run_weights_module(cohort, herd, show_indicator=False)


def test_output_tables(result):
    assert list(result) == ["cohort_level_results", "herd_level_results"]
    assert sorted(result) == golden_tables(CASE)


@pytest.mark.parametrize("table", ["cohort_level_results", "herd_level_results"])
def test_matches_golden(result, table):
    assert_matches_golden(result[table], CASE, table)


def test_inputs_not_mutated(inputs):
    cohort, herd = inputs
    c0, h0 = cohort.copy(), herd.copy()
    gleam.run_weights_module(cohort, herd, show_indicator=False)
    pd.testing.assert_frame_equal(cohort, c0)
    pd.testing.assert_frame_equal(herd, h0)


def test_unvalidated_run_matches_and_warns(inputs, result):
    cohort, herd = inputs
    with pytest.warns(GleamWarning, match="Input validation has been turned off"):
        res = gleam.run_weights_module(cohort, herd, show_indicator=False, validate_inputs=False)
    for table in result:
        pd.testing.assert_frame_equal(res[table], result[table])


def test_adds_phase_column_when_absent(inputs):
    cohort, herd = inputs
    demo = cohort[~cohort["cohort_short"].isin(["FN", "MN"])].drop(columns="nondemo_productive_phase_id")
    res = gleam.run_weights_module(demo, herd, show_indicator=False)["cohort_level_results"]
    assert list(res.columns[: demo.shape[1] + 1]) == list(demo.columns) + ["nondemo_productive_phase_id"]
    assert res["nondemo_productive_phase_id"].isna().all()


def test_show_indicator_prints_progress(inputs, capsys):
    cohort, herd = inputs
    gleam.run_weights_module(cohort, herd, show_indicator=True)
    assert "Cohort weights calculation complete." in capsys.readouterr().err


def test_run_validation_errors(inputs):
    cohort, herd = inputs
    with pytest.raises(GleamValidationError, match="`cohort_level_data` must be a data.table"):
        gleam.run_weights_module(cohort.to_dict(), herd, show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `cohort_level_data`"):
        gleam.run_weights_module(cohort.drop(columns="offtake_rate"), herd, show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `herd_level_data`"):
        gleam.run_weights_module(cohort, herd.drop(columns="live_weight_at_birth"), show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `herd_level_data`"):
        gleam.run_weights_module(
            cohort, herd.drop(columns="phase1_nondemo_fem_duration_days"), show_indicator=False
        )
    with pytest.raises(GleamValidationError, match="Each herd_id must appear exactly once"):
        gleam.run_weights_module(cohort, pd.concat([herd, herd.iloc[[0]]]), show_indicator=False)
    with pytest.raises(GleamValidationError, match="not found in `herd_level_data`"):
        gleam.run_weights_module(cohort, herd.iloc[1:], show_indicator=False)

    bad = herd.copy()
    bad.loc[1, "live_weight_at_birth"] = 300.0  # herd 2: above female slaughter weight
    with pytest.raises(
        GleamValidationError,
        match="`live_weight_at_birth` must be less than `live_weight_female_at_slaughter`. "
        "Violation\\(s\\) for herd_id: 2",
    ):
        gleam.run_weights_module(cohort, bad, show_indicator=False)

    bad = herd.copy()
    bad.loc[1, "live_weight_at_weaning"] = 14.0  # herd 2: weaning == birth
    with pytest.raises(GleamValidationError, match="must be less than `live_weight_at_weaning`"):
        gleam.run_weights_module(cohort, bad, show_indicator=False)

    # chickens are exempt from the birth < weaning rule (herd 13 has 0.04 / 0.04)
    assert herd.loc[herd["herd_id"] == 13, "live_weight_at_weaning"].item() == 0.04


def test_row_level_validation_inside_run(inputs):
    cohort, herd = inputs
    bad = cohort.copy()
    bad.loc[0, "cohort_duration_days"] = 0
    with pytest.raises(GleamValidationError, match="`cohort_duration_days`\\[1\\] = 0 is out of range"):
        gleam.run_weights_module(bad, herd, show_indicator=False)


def test_validation_off_skips_checks(inputs):
    cohort, herd = inputs
    bad = cohort.copy()
    bad.loc[0, "cohort_duration_days"] = 0  # herd 1 FA: gain 0 / 0 -> NaN, as in R
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        res = gleam.run_weights_module(bad, herd, show_indicator=False, validate_inputs=False)
    assert np.isnan(res["cohort_level_results"]["daily_weight_gain"].iloc[0])
