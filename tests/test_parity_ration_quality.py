"""Golden parity of :func:`gleam.run_ration_quality_module` with the R package.

Runs the bundled example exactly as ``tools/r_reference/generate_golden.R``
(case ``ration_quality_module``, result wrapped as ``list(result = ...)``).
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

CASE = "ration_quality_module"


@pytest.fixture(scope="module")
def inputs():
    return load_example("feed_rations_share_chrt.csv"), load_example("feed_quality.csv")


@pytest.fixture(scope="module")
def result(inputs):
    rations, feed = inputs
    return gleam.run_ration_quality_module(rations_share=rations, feed_params=feed, show_indicator=False)


def test_matches_golden(result):
    assert golden_tables(CASE) == ["result"]
    assert isinstance(result, pd.DataFrame)
    assert_matches_golden(result, CASE, "result")


def test_inputs_not_mutated(inputs):
    rations, feed = inputs
    r0, f0 = rations.copy(), feed.copy()
    gleam.run_ration_quality_module(rations, feed, show_indicator=False)
    pd.testing.assert_frame_equal(rations, r0)
    pd.testing.assert_frame_equal(feed, f0)


def test_unvalidated_run_matches_and_warns(inputs, result):
    rations, feed = inputs
    with pytest.warns(GleamWarning, match="Input validation has been turned off"):
        res = gleam.run_ration_quality_module(rations, feed, show_indicator=False, validate_inputs=False)
    pd.testing.assert_frame_equal(res, result)


def test_groups_without_phase_column(inputs):
    rations, feed = inputs
    demo = rations[rations["nondemo_productive_phase_id"].isna()].drop(columns="nondemo_productive_phase_id")
    res = gleam.run_ration_quality_module(demo, feed, show_indicator=False)
    assert list(res.columns[:3]) == ["herd_id", "species_short", "cohort_short"]
    assert "nondemo_productive_phase_id" not in res.columns


def test_na_contribution_gives_na_sum(inputs, result):
    # R uses sum() without na.rm: one NA contribution makes the cohort value NA.
    rations, feed = inputs
    feed = feed.copy()
    feed_id = rations["feed_id"].iloc[0]
    feed.loc[feed["feed_id"] == feed_id, "feed_nitrogen_content"] = np.nan
    with pytest.raises(GleamValidationError, match="`feed_nitrogen_content`.* must be a single numeric value"):
        gleam.run_ration_quality_module(rations, feed, show_indicator=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        res = gleam.run_ration_quality_module(rations, feed, show_indicator=False, validate_inputs=False)
    keys = ["herd_id", "species_short", "cohort_short", "nondemo_productive_phase_id"]
    uses = rations.loc[rations["feed_id"] == feed_id, keys].drop_duplicates()
    flag = res[keys].merge(uses.assign(__uses=True), on=keys, how="left")["__uses"].notna().to_numpy()
    assert flag.any() and not flag.all()
    assert res["ration_nitrogen"].isna().to_numpy().tolist() == flag.tolist()
    pd.testing.assert_series_equal(res["ration_ash"], result["ration_ash"])


def test_run_validation_errors(inputs):
    rations, feed = inputs
    with pytest.raises(GleamValidationError, match="`rations_share` must be a data.table"):
        gleam.run_ration_quality_module(rations.to_dict(), feed, show_indicator=False)
    with pytest.raises(GleamValidationError, match="`feed_params` must contain at least one row"):
        gleam.run_ration_quality_module(rations, feed.iloc[:0], show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `rations_share`"):
        gleam.run_ration_quality_module(rations.drop(columns="feed_id"), feed, show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `feed_params`"):
        gleam.run_ration_quality_module(rations, feed.drop(columns="feed_ash"), show_indicator=False)

    bad = rations.copy()
    bad.loc[0, "feed_ration_fraction"] += 0.1
    with pytest.raises(GleamValidationError, match="Feed rations must sum to 1"):
        gleam.run_ration_quality_module(bad, feed, show_indicator=False)

    with pytest.raises(GleamValidationError, match="`feed_params\\$feed_id` must be unique"):
        gleam.run_ration_quality_module(rations, pd.concat([feed, feed.iloc[[0]]]), show_indicator=False)

    dup = pd.concat([rations, rations.iloc[[0]]], ignore_index=True)
    dup.loc[len(dup) - 1, "feed_ration_fraction"] = 0.0  # keeps the group sum at 1
    with pytest.raises(GleamValidationError, match="`rations_share\\$feed_id` must be unique"):
        gleam.run_ration_quality_module(dup, feed, show_indicator=False)
