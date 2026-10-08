"""Golden parity of :func:`gleampy.run_ration_quality_module` with the R package.

Runs the bundled example exactly as ``tools/r_reference/generate_golden.R``
(case ``ration_quality_module``, result wrapped as ``list(result = ...)``).
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

CASE = "ration_quality_module"


@pytest.fixture(scope="module")
def inputs():
    return load_example("feed_rations_share_chrt.csv"), load_example("feed_quality.csv")


@pytest.fixture(scope="module")
def result(inputs):
    rations, feed = inputs
    return gleampy.run_ration_quality_module(rations_share=rations, feed_params=feed, show_indicator=False)


def test_matches_golden(result):
    assert golden_tables(CASE) == ["result"]
    assert isinstance(result, pd.DataFrame)
    assert_matches_golden(result, CASE, "result")


def test_inputs_not_mutated(inputs):
    rations, feed = inputs
    r0, f0 = rations.copy(), feed.copy()
    gleampy.run_ration_quality_module(rations, feed, show_indicator=False)
    pd.testing.assert_frame_equal(rations, r0)
    pd.testing.assert_frame_equal(feed, f0)


def test_unvalidated_run_matches_and_warns(inputs, result):
    rations, feed = inputs
    with pytest.warns(GleamWarning, match="Input validation has been turned off"):
        res = gleampy.run_ration_quality_module(rations, feed, show_indicator=False, validate_inputs=False)
    pd.testing.assert_frame_equal(res, result)


def test_groups_without_phase_column(inputs):
    rations, feed = inputs
    demo = rations[rations["nondemo_productive_phase_id"].isna()].drop(columns="nondemo_productive_phase_id")
    res = gleampy.run_ration_quality_module(demo, feed, show_indicator=False)
    assert list(res.columns[:3]) == ["herd_id", "species_short", "cohort_short"]
    assert "nondemo_productive_phase_id" not in res.columns


def test_na_contribution_gives_na_sum(inputs, result):
    # R uses sum() without na.rm: one NA contribution makes the cohort value NA.
    rations, feed = inputs
    feed = feed.copy()
    feed_id = rations["feed_id"].iloc[0]
    feed.loc[feed["feed_id"] == feed_id, "feed_nitrogen_content"] = np.nan
    with pytest.raises(GleamValidationError, match="`feed_nitrogen_content`.* must be a single numeric value"):
        gleampy.run_ration_quality_module(rations, feed, show_indicator=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        res = gleampy.run_ration_quality_module(rations, feed, show_indicator=False, validate_inputs=False)
    keys = ["herd_id", "species_short", "cohort_short", "nondemo_productive_phase_id"]
    uses = rations.loc[rations["feed_id"] == feed_id, keys].drop_duplicates()
    flag = res[keys].merge(uses.assign(__uses=True), on=keys, how="left")["__uses"].notna().to_numpy()
    assert flag.any() and not flag.all()
    assert res["ration_nitrogen"].isna().to_numpy().tolist() == flag.tolist()
    pd.testing.assert_series_equal(res["ration_ash"], result["ration_ash"])


def test_run_validation_errors(inputs):
    rations, feed = inputs
    with pytest.raises(GleamValidationError, match="`rations_share` must be a data.table"):
        gleampy.run_ration_quality_module(rations.to_dict(), feed, show_indicator=False)
    with pytest.raises(GleamValidationError, match="`feed_params` must contain at least one row"):
        gleampy.run_ration_quality_module(rations, feed.iloc[:0], show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `rations_share`"):
        gleampy.run_ration_quality_module(rations.drop(columns="feed_id"), feed, show_indicator=False)
    with pytest.raises(GleamValidationError, match="Missing required columns in `feed_params`"):
        gleampy.run_ration_quality_module(rations, feed.drop(columns="feed_ash"), show_indicator=False)

    bad = rations.copy()
    bad.loc[0, "feed_ration_fraction"] += 0.1
    with pytest.raises(GleamValidationError, match="Feed rations must sum to 1"):
        gleampy.run_ration_quality_module(bad, feed, show_indicator=False)

    with pytest.raises(GleamValidationError, match="`feed_params\\$feed_id` must be unique"):
        gleampy.run_ration_quality_module(rations, pd.concat([feed, feed.iloc[[0]]]), show_indicator=False)

    dup = pd.concat([rations, rations.iloc[[0]]], ignore_index=True)
    dup.loc[len(dup) - 1, "feed_ration_fraction"] = 0.0  # keeps the group sum at 1
    with pytest.raises(GleamValidationError, match="`rations_share\\$feed_id` must be unique"):
        gleampy.run_ration_quality_module(dup, feed, show_indicator=False)


# --------------------------------------------------------------------------
# Sum-to-one check at the tolerance edge
# --------------------------------------------------------------------------
#
# R sums the fractions of each ration group with data.table's GForce gsum: a
# plain double accumulation in row order (pandas' groupby().sum() is
# compensated and rounds differently). These herd 1 CTL FA groups have sums
# within a few ulp of 1 +/- 1e-6; R's decisions (run_ration_quality_module and
# run_emissions_ration_module agree) were checked with R 4.6 / data.table
# 1.18.6.1. The pandas sum flips every one of them.

_E54 = 2.0**-54
_EDGE_GROUPS = {
    "constructed_R_accepts": (
        True, [float.fromhex("0x1.000010c6f7a0bp+0") - 0.5, 0.5, _E54, _E54, _E54, 0.0, 0.0, 0.0, 0.0],
    ),
    "low_R_rejects": (False, [
        "0x1.2f5b2e1fb4b13p-7", "0x1.fa3fbabd8b886p-4", "0x1.b7ec285a8e00dp-4",
        "0x1.ddb7290d6c2eap-7", "0x1.331c8ccd5b0d0p-3", "0x1.984c6ea2f89d9p-3",
        "0x1.1bf0c2664b522p-3", "0x1.f235984f95913p-5", "0x1.92313ededfc6dp-3",
    ]),
    "low_R_accepts": (True, [
        "0x1.6f82eb82c9746p-5", "0x1.394e785501778p-6", "0x1.9ab2c5c5394fep-3",
        "0x1.9d9b575eb039dp-3", "0x1.a4ee918275798p-3", "0x1.c53e096acbf14p-4",
        "0x1.073533f9c650fp-4", "0x1.b3ec475816583p-10", "0x1.3616c9f5981d2p-3",
    ]),
    "high_R_rejects": (False, [
        "0x1.1d8d292573ebep-3", "0x1.016e0f45193cdp-3", "0x1.538d3e2f09203p-4",
        "0x1.5577c7ce89748p-3", "0x1.f1bfadb86074cp-4", "0x1.3f0231bb6a94ep-3",
        "0x1.0d21d5c532c6dp-4", "0x1.ac1708926b0cfp-4", "0x1.3521bc8ee07bfp-5",
    ]),
    "high_R_accepts": (True, [
        "0x1.619db95cb9712p-4", "0x1.1753a0952cecep-3", "0x1.7f06a615185cep-4",
        "0x1.ecbbeeaa9e9c9p-4", "0x1.60b5a5f6d52aap-3", "0x1.3127c069b2c01p-3",
        "0x1.32a4842f873f1p-4", "0x1.78b2a244b7fe2p-4", "0x1.34e789f360b4bp-4",
    ]),
}


def _edge_rations(rations, values):
    values = [float.fromhex(v) if isinstance(v, str) else v for v in values]
    out = rations.copy()
    sel = (out["herd_id"] == 1) & (out["species_short"] == "CTL") & (out["cohort_short"] == "FA")
    assert sel.sum() == len(values)
    out.loc[sel, "feed_ration_fraction"] = values
    return out


@pytest.mark.parametrize("case", list(_EDGE_GROUPS))
def test_ration_sum_check_adds_in_row_order_like_r(inputs, case):
    rations, feed = inputs
    r_accepts, values = _EDGE_GROUPS[case]
    bad = _edge_rations(rations, values)
    if r_accepts:
        gleampy.run_ration_quality_module(bad, feed, show_indicator=False)
    else:
        with pytest.raises(GleamValidationError, match="Feed rations must sum to 1"):
            gleampy.run_ration_quality_module(bad, feed, show_indicator=False)


def test_ration_sum_check_groups_like_data_table():
    from gleampy.validation.ration_quality_run import group_ids, group_sums_off_one

    # NA keys form their own group; categorical / nullable keys group by value.
    df = pd.DataFrame(
        {
            "k": pd.Categorical(["a", "b", None, "a", None, "b"]),
            "p": pd.array([1, 1, 1, 1, 1, 2], dtype="Int64"),
            "v": [0.5, 1.0, 0.25, 0.5, 0.75, 1.0],
        }
    )
    assert group_ids(df, ["k"]).tolist() == [0, 1, 2, 0, 2, 1]
    assert group_ids(df, ["k", "p"]).tolist() == [0, 1, 2, 0, 2, 3]
    assert not group_sums_off_one(df, ["k"], "v", tol=1.5)
    assert not group_sums_off_one(df.iloc[[0, 3, 2, 4]], ["k"], "v")
    assert group_sums_off_one(df, ["k"], "v")
    # A group with a missing value has an NA sum and never fails (as in R).
    na = df.iloc[[0, 3, 2, 4]].assign(v=[0.5, np.nan, 3.0, 0.75])
    assert not group_sums_off_one(na.iloc[:2], ["k"], "v")
    assert group_sums_off_one(na, ["k"], "v")


# --------------------------------------------------------------------------
# All-empty feed parameter columns (fread: logical NA columns)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("col", "message"),
    [
        # R 4.6 messages for the same feed_quality.csv with the column emptied.
        ("feed_digestible_energy_pigs", "`feed_digestible_energy_pigs` must be numeric."),
        ("feed_metabolizable_energy_pigs",
         "`feed_metabolizable_energy_pigs` must be a single numeric (scalar). NA is allowed."),
        ("feed_urinary_energy_ruminant",
         "`feed_urinary_energy_ruminant` must be a single numeric (scalar). NA is allowed."),
    ],
)
def test_all_empty_feed_parameter_column_is_rejected_like_r(inputs, tmp_path, col, message):
    from gleampy.io import read_csv

    rations, feed = inputs
    path = tmp_path / "feed.csv"
    feed.assign(**{col: ""}).to_csv(path, index=False)
    empty = read_csv(path)
    assert empty[col].isna().all() and empty[col].dtype == object  # a logical NA column in R
    with pytest.raises(GleamValidationError) as err:
        gleampy.run_ration_quality_module(rations, empty, show_indicator=False)
    assert str(err.value) == message



def test_float_column_of_missing_values_is_numeric(inputs):
    # A float column of NaN is a double NA column in R: it is numeric, so the
    # digestible-energy check of calc_feed_digestibility_fraction accepts it.
    rations, feed = inputs
    gleampy.run_ration_quality_module(rations, feed.assign(feed_digestible_energy_pigs=np.nan), show_indicator=False)
