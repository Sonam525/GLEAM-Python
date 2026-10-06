"""Golden parity of :func:`gleam.run_emissions_ration_module` with the R package.

Runs the bundled example exactly as ``tools/r_reference/generate_golden.R``
(case ``emissions_ration_module``, result wrapped as ``list(result = ...)``).
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

CASE = "emissions_ration_module"
EF_COLS = [
    "co2_feed_fertilizer", "co2_feed_pesticides", "co2_feed_crop_activities",
    "co2_feed_luc_nopeat", "co2_feed_luc_peat", "n2o_feed_fertilizer",
    "n2o_feed_manure_applied", "n2o_feed_crop_residues", "ch4_feed_rice",
]
OUT_COLS = [
    "co2_ration_fertilizer", "co2_ration_pesticides", "co2_ration_crop_activities",
    "co2_ration_luc_nopeat", "co2_ration_luc_peat", "n2o_ration_fertilizer",
    "n2o_ration_manure_applied", "n2o_ration_crop_residues", "ch4_ration_rice",
]


@pytest.fixture(scope="module")
def inputs():
    return load_example("feed_rations_share_chrt.csv"), load_example("feed_emission_factors.csv")


@pytest.fixture(scope="module")
def result(inputs):
    rations, ef = inputs
    return gleam.run_emissions_ration_module(rations_share=rations, feed_emissions=ef, show_indicator=False)


def test_matches_golden(result):
    assert golden_tables(CASE) == ["result"]
    assert isinstance(result, pd.DataFrame)
    assert_matches_golden(result, CASE, "result")


def test_inputs_not_mutated(inputs):
    rations, ef = inputs
    r0, e0 = rations.copy(), ef.copy()
    gleam.run_emissions_ration_module(rations, ef, show_indicator=False)
    pd.testing.assert_frame_equal(rations, r0)
    pd.testing.assert_frame_equal(ef, e0)


def test_unvalidated_run_matches_and_warns(inputs, result):
    rations, ef = inputs
    with pytest.warns(GleamWarning, match="Input validation has been turned off"):
        res = gleam.run_emissions_ration_module(rations, ef, show_indicator=False, validate_inputs=False)
    pd.testing.assert_frame_equal(res, result)


def test_missing_feed_factors_are_dropped_from_sums(inputs, result):
    # Left join + sum(na.rm = TRUE): feeds without emission factors count as 0.
    rations, ef = inputs
    feed_id = rations["feed_id"].iloc[0]
    res = gleam.run_emissions_ration_module(rations, ef[ef["feed_id"] != feed_id], show_indicator=False)
    assert len(res) == len(result) and res[OUT_COLS].notna().all().all()
    keys = ["herd_id", "species_short", "cohort_short", "nondemo_productive_phase_id"]
    rows = rations[rations["feed_id"] == feed_id]
    contrib = rows[keys].copy()
    for out, col in zip(OUT_COLS, EF_COLS):
        contrib[out] = rows["feed_ration_fraction"].to_numpy() * ef.loc[ef["feed_id"] == feed_id, col].item()
    # the left join keeps every ration row, so the group order is unchanged
    pd.testing.assert_frame_equal(res[keys], result[keys])
    merged = res.merge(contrib, on=keys, how="left", suffixes=("", "_feed"))
    for out in OUT_COLS:
        np.testing.assert_allclose(
            merged[out] + merged[out + "_feed"].fillna(0), result[out], rtol=1e-12, atol=1e-12,
            err_msg=out,
        )


def test_run_validation_errors(inputs):
    rations, ef = inputs
    with pytest.raises(GleamValidationError, match="`feed_emissions` must be a data.table"):
        gleam.run_emissions_ration_module(rations, ef.to_dict(), show_indicator=False)
    with pytest.raises(GleamValidationError, match="`rations_share` must contain at least one row"):
        gleam.run_emissions_ration_module(rations.iloc[:0], ef, show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `rations_share`: \"feed_name\""):
        gleam.run_emissions_ration_module(rations.drop(columns="feed_name"), ef, show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `feed_emissions`"):
        gleam.run_emissions_ration_module(rations, ef.drop(columns="ch4_feed_rice"), show_indicator=False)

    bad = rations.copy()
    bad.loc[0, "feed_ration_fraction"] += 0.1
    with pytest.raises(GleamValidationError, match="Feed emissions fractions must sum to 1"):
        gleam.run_emissions_ration_module(bad, ef, show_indicator=False)

    dup = pd.concat([rations, rations.iloc[[0]]], ignore_index=True)
    dup.loc[len(dup) - 1, "feed_ration_fraction"] = 0.0
    with pytest.raises(GleamValidationError, match="contains duplicated `feed_id`"):
        gleam.run_emissions_ration_module(dup, ef, show_indicator=False)
    dup.loc[len(dup) - 1, "feed_id"] = "F_other"
    with pytest.raises(GleamValidationError, match="contains duplicated `feed_name`"):
        gleam.run_emissions_ration_module(dup, ef, show_indicator=False)

    with pytest.raises(GleamValidationError, match="`feed_emissions\\$feed_id` must be unique"):
        gleam.run_emissions_ration_module(rations, pd.concat([ef, ef.iloc[[0]]]), show_indicator=False)

    bad_ef = ef.copy()
    bad_ef["co2_feed_pesticides"] = bad_ef["co2_feed_pesticides"].astype(str)
    with pytest.raises(GleamValidationError, match="`co2_feed_pesticides` must be a single numeric"):
        gleam.run_emissions_ration_module(rations, bad_ef, show_indicator=False)


def test_feed_name_cross_checks(inputs, result):
    rations, ef = inputs
    names = rations[["feed_id", "feed_name"]].drop_duplicates("feed_id")
    named = ef.merge(names, on="feed_id", how="left")
    used = set(rations["feed_id"])
    # every feed used in the rations must have a matching feed_name
    res = gleam.run_emissions_ration_module(rations, named[named["feed_id"].isin(used)], show_indicator=False)
    pd.testing.assert_frame_equal(res, result)

    wrong = named[named["feed_id"].isin(used)].copy()
    wrong.loc[wrong.index[0], "feed_name"] = "Something else"
    with pytest.raises(GleamValidationError, match="missing or mismatched feed_name in `feed_emissions`"):
        gleam.run_emissions_ration_module(rations, wrong, show_indicator=False)

    clash = named[named["feed_id"].isin(used)].copy()
    clash.loc[clash.index[1], "feed_name"] = clash["feed_name"].iloc[0]
    with pytest.raises(GleamValidationError, match="`feed_emissions\\$feed_name` must be unique"):
        gleam.run_emissions_ration_module(rations, clash, show_indicator=False)


def test_row_level_validation_inside_run(inputs):
    rations, ef = inputs
    bad = ef.copy()
    bad.loc[0, "co2_feed_fertilizer"] = -1.0
    with pytest.raises(GleamValidationError, match="`co2_feed_fertilizer` must be >= 0"):
        gleam.run_emissions_ration_module(rations, bad, show_indicator=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        gleam.run_emissions_ration_module(rations, bad, show_indicator=False, validate_inputs=False)
