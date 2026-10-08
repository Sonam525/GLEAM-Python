"""Tests for run_emissions_direct() (port of tests/testthat/test-run_emissions_direct.R)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gleampy
from gleampy import GleamValidationError
from gleampy.constants import GLEAM_COHORTS_DEMOGRAPHIC
from gleampy.io import example_path, read_csv
from gleampy.modules import emissions_direct as direct_mod
from gleampy.validation.emissions_direct_run import validate_run_emissions_direct_inputs

KEYS = ["herd_id", "species_short", "cohort_short"]

# testthat (edition 3) expect_equal() tolerance: sqrt(.Machine$double.eps), relative.
TESTTHAT_TOLERANCE = 1.5e-8


def assert_equal_r(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Port of testthat ``expect_equal()`` on two tables: relative tolerance 1.5e-8.

    pandas' own default (``rtol=1e-5``, ``atol=1e-8``) is about 650 times
    looser and would accept differences the R tests reject.
    """
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=TESTTHAT_TOLERANCE, atol=0.0)


def _mod(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_modules_examples"))


@pytest.fixture(scope="module")
def d_direct() -> dict:
    herd = _mod("emissions_direct_input_hrd_data.csv")

    def in_herds(df):
        return df[df["herd_id"].isin(herd["herd_id"])].reset_index(drop=True) if "herd_id" in df else df

    feed_rations = in_herds(_mod("feed_rations_share_chrt.csv"))
    feed_rations = feed_rations[feed_rations.cohort_short.isin(GLEAM_COHORTS_DEMOGRAPHIC)].reset_index(drop=True)
    return {
        "cohort_no_structure": _mod("emissions_direct_input_chrt_no_structure_data.csv"),
        "cohort_structure": _mod("emissions_direct_input_chrt_structure_data.csv"),
        "herd": herd,
        "feed_rations": feed_rations,
        "feed_params": _mod("feed_quality.csv"),
        "feed_emissions": in_herds(_mod("feed_emission_factors.csv")),
        "mms_fraction": in_herds(_mod("manure_management_system_fraction.csv")),
        "mms_factors": in_herds(_mod("manure_management_system_factors.csv")),
    }


@pytest.fixture(scope="module")
def direct_quality(d_direct) -> pd.DataFrame:
    q = gleampy.run_ration_quality_module(d_direct["feed_rations"], d_direct["feed_params"], show_indicator=False)
    return q.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


@pytest.fixture(scope="module")
def direct_quality_cols(direct_quality) -> list[str]:
    return [c for c in direct_quality.columns if c not in KEYS + ["nondemo_productive_phase_id"]]


def direct_inputs(d, quality, has_herd_structure=True, primary=False) -> dict:
    cohort = d["cohort_structure"] if has_herd_structure else d["cohort_no_structure"]
    if primary:
        cohort = gleampy._utils.merge_dt(cohort, quality, by=KEYS)
    args = dict(
        has_herd_structure=has_herd_structure,
        cohort_level_data=cohort.copy(),
        herd_level_data=d["herd"].copy(),
        manure_management_system_fraction=d["mms_fraction"],
        manure_management_system_factors=d["mms_factors"],
        show_indicator=False,
    )
    if not primary:
        args["feed_rations"] = d["feed_rations"]
        args["feed_params"] = d["feed_params"]
    return args


def _validator_args(args: dict) -> dict:
    a = dict(args)
    a.pop("show_indicator", None)
    return a


def test_validator_accepts_both_nutritional_input_modes(d_direct, direct_quality):
    for has_structure in (True, False):
        for primary in (True, False):
            for factors_only in (True, False):
                args = _validator_args(direct_inputs(d_direct, direct_quality, has_structure, primary))
                args["emission_factors_only"] = factors_only
                assert validate_run_emissions_direct_inputs(**args) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("has_herd_structure", np.nan),
        ("simulation_duration", 0),
        ("global_warming_potential_set", "unknown"),
        ("cohort_level_data", None),
        ("herd_level_data", None),
        ("manure_management_system_fraction", None),
        ("manure_management_system_factors", None),
    ],
)
@pytest.mark.parametrize("primary", [True, False])
def test_validator_checks_pipeline_inputs(d_direct, direct_quality, field, value, primary):
    args = _validator_args(direct_inputs(d_direct, direct_quality, primary=primary))
    args[field] = value
    with pytest.raises(GleamValidationError, match=field):
        validate_run_emissions_direct_inputs(**args)


def _key_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


@pytest.mark.parametrize("has_structure", [True, False])
def test_primary_nutrition_matches_feed_derived_results(d_direct, direct_quality, direct_quality_cols, has_structure, monkeypatch):
    feed_result = gleampy.run_emissions_direct(**direct_inputs(d_direct, direct_quality, has_structure))
    primary_result = gleampy.run_emissions_direct(**direct_inputs(d_direct, direct_quality, has_structure, primary=True))

    feed_chrt = _key_sorted(feed_result["cohort_level_results"])
    prim_chrt = _key_sorted(primary_result["cohort_level_results"])[list(feed_chrt.columns)]
    assert_equal_r(prim_chrt, feed_chrt)
    for key in ("herd_level_results", "allocation_long"):
        assert_equal_r(primary_result[key], feed_result[key])
    for key, df in feed_result["aggregation_results"].items():
        assert_equal_r(primary_result["aggregation_results"][key], df)

    assert_equal_r(
        prim_chrt[direct_quality_cols].reset_index(drop=True),
        direct_quality[direct_quality_cols].reset_index(drop=True),
    )
    assert not any("_ration_" in c for c in feed_result["cohort_level_results"].columns)
    emissions = feed_result["aggregation_results"]["results_emissions"]
    assert len(emissions) > 0
    assert emissions["variable_name"].str.match(r"^(ch4|n2o)_(enteric|manure)").all()
    assert np.isfinite(emissions["value_total_allocated_co2eq"].to_numpy(dtype=float)).all()

    for primary in (False, True):
        def boom(name):
            def _f(*a, **k):
                raise RuntimeError(f"{name} must be skipped")
            return _f

        with monkeypatch.context() as m:
            m.setattr(direct_mod, "run_production_module", boom("Production"))
            m.setattr(direct_mod, "run_allocation_module", boom("Allocation"))
            m.setattr(direct_mod, "run_aggregation_module", boom("Aggregation"))
            args = direct_inputs(d_direct, direct_quality, has_structure, primary=primary)
            args["emission_factors_only"] = True
            args["herd_level_data"] = args["herd_level_data"].drop(
                columns=[
                    "milk_protein_fraction_standard", "milk_fat_fraction_standard",
                    "milk_lactose_fraction_standard", "carcass_dressing_fraction",
                    "bone_free_meat_fraction", "meat_protein_fraction",
                ]
            )
            factors = gleampy.run_emissions_direct(**args)
        assert list(factors) == list(feed_result)
        assert factors["allocation_long"] is None
        assert factors["aggregation_results"] is None
        factor_cols = list(factors["cohort_level_results"].columns)
        assert not any(("_production_" in c) or ("_allocation_" in c) for c in factor_cols)
        assert_equal_r(_key_sorted(factors["cohort_level_results"]), feed_chrt[factor_cols])


@pytest.mark.parametrize("bad_value", [None, np.nan, 1, "TRUE", [], [True, False]])
def test_emission_factors_only_requires_single_logical(d_direct, direct_quality, bad_value):
    args = direct_inputs(d_direct, direct_quality)
    args["emission_factors_only"] = bad_value
    with pytest.raises(GleamValidationError, match="emission_factors_only.*single logical value"):
        gleampy.run_emissions_direct(**args)
    with pytest.raises(GleamValidationError, match="emission_factors_only.*single logical value"):
        validate_run_emissions_direct_inputs(**_validator_args(args))


def test_direct_emissions_match_full_pipeline_sources(d_direct, direct_quality):
    args = direct_inputs(d_direct, direct_quality)
    direct = gleampy.run_emissions_direct(**args)
    args["feed_emissions"] = d_direct["feed_emissions"]
    full = gleampy.run_gleam(**args)
    full_em = full["aggregation_results"]["results_emissions"]
    full_em = full_em[full_em["variable_name"].str.match(r"^(ch4|n2o)_(enteric|manure)")].reset_index(drop=True)
    assert_equal_r(direct["aggregation_results"]["results_emissions"], full_em)


def test_assert_equal_r_uses_testthat_tolerance(d_direct, direct_quality):
    em = gleampy.run_emissions_direct(**direct_inputs(d_direct, direct_quality))["aggregation_results"]["results_emissions"]
    for eps, fails in ((1e-7, True), (5e-6, True), (1e-9, False)):
        bad = em.copy()
        bad.loc[0, "value_total_allocated_co2eq"] = em["value_total_allocated_co2eq"].iloc[0] * (1 + eps)
        if fails:
            with pytest.raises(AssertionError):
                assert_equal_r(bad, em)
        else:
            assert_equal_r(bad, em)


def test_user_supplied_nitrogen_changes_intake_and_manure(d_direct, direct_quality):
    args = direct_inputs(d_direct, direct_quality, primary=True)
    baseline = gleampy.run_emissions_direct(**args)
    args["cohort_level_data"]["ration_nitrogen"] = args["cohort_level_data"]["ration_nitrogen"] * 1.1
    changed = gleampy.run_emissions_direct(**args)
    before, after = baseline["cohort_level_results"], changed["cohort_level_results"]
    np.testing.assert_allclose(after["ration_nitrogen"], before["ration_nitrogen"] * 1.1, rtol=1.5e-8)
    np.testing.assert_allclose(after["nitrogen_intake"], before["nitrogen_intake"] * 1.1, rtol=1.5e-8)
    np.testing.assert_allclose(after["ch4_enteric"], before["ch4_enteric"], rtol=1.5e-8)
    assert (after["n2o_manure_other_direct"] > before["n2o_manure_other_direct"]).any()


def test_omitted_feed_tables_require_every_primary_quality_column(d_direct, direct_quality, direct_quality_cols):
    args = direct_inputs(d_direct, direct_quality)
    args.pop("feed_rations")
    args.pop("feed_params")
    with pytest.raises(GleamValidationError, match="primary nutritional quality"):
        gleampy.run_emissions_direct(**args)
    for col in direct_quality_cols:
        args = direct_inputs(d_direct, direct_quality, primary=True)
        args["cohort_level_data"] = args["cohort_level_data"].drop(columns=[col])
        with pytest.raises(GleamValidationError, match=col):
            gleampy.run_emissions_direct(**args)


@pytest.mark.parametrize("primary", [True, False])
@pytest.mark.parametrize("field", ["feed_rations", "feed_params"])
def test_feed_tables_must_be_supplied_together(d_direct, direct_quality, primary, field):
    args = direct_inputs(d_direct, direct_quality, primary=primary)
    args["feed_rations"] = d_direct["feed_rations"]
    args["feed_params"] = d_direct["feed_params"]
    args[field] = None
    with pytest.raises(GleamValidationError, match="together, or omit both"):
        gleampy.run_emissions_direct(**args)


def test_primary_quality_cannot_be_combined_with_feed_tables(d_direct, direct_quality):
    args = direct_inputs(d_direct, direct_quality, primary=True)
    args["feed_rations"] = d_direct["feed_rations"]
    args["feed_params"] = d_direct["feed_params"]
    with pytest.raises(GleamValidationError, match="Do not provide.*ration_gross_energy"):
        gleampy.run_emissions_direct(**args)


@pytest.mark.parametrize("bad_value", [np.nan, "unknown", -1, np.inf])
def test_primary_quality_uses_range_validation(d_direct, direct_quality, direct_quality_cols, bad_value):
    for col in direct_quality_cols:
        args = direct_inputs(d_direct, direct_quality, primary=True)
        n = len(args["cohort_level_data"])
        args["cohort_level_data"][col] = pd.Series([bad_value] * n, dtype=object if isinstance(bad_value, str) else float)
        with pytest.raises(GleamValidationError, match=col):
            gleampy.run_emissions_direct(**args)


def test_primary_mode_retains_other_input_checks(d_direct, direct_quality):
    args = direct_inputs(d_direct, direct_quality, primary=True)
    args["cohort_level_data"]["daily_weight_gain"] = 0.5
    with pytest.raises(GleamValidationError, match="Do not provide.*daily_weight_gain"):
        gleampy.run_emissions_direct(**args)

    args = direct_inputs(d_direct, direct_quality, primary=True)
    args["herd_level_data"]["herd_id"] = args["herd_level_data"]["herd_id"].astype(str) + "_bad"
    with pytest.raises(GleamValidationError, match="same.*herd_id"):
        gleampy.run_emissions_direct(**args)

    args = direct_inputs(d_direct, direct_quality, primary=True)
    args["manure_management_system_factors"] = None
    with pytest.raises(GleamValidationError, match="manure_management_system_factors"):
        gleampy.run_emissions_direct(**args)


@pytest.mark.parametrize("has_structure", [False, True])
@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("factors_only", [False, True])
def test_direct_emissions_require_all_six_cohorts(d_direct, direct_quality, has_structure, primary, factors_only):
    args = direct_inputs(d_direct, direct_quality, has_herd_structure=has_structure, primary=primary)
    c = args["cohort_level_data"]
    args["cohort_level_data"] = c[c.cohort_short == "FA"].reset_index(drop=True)
    args["emission_factors_only"] = factors_only
    with pytest.raises(GleamValidationError, match="exactly 6 rows"):
        gleampy.run_emissions_direct(**args)


# ---- numpy booleans behave like Python booleans (R's isTRUE) ---------------


def _run_example(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_gleam_examples"))


def _herd9_args() -> dict:
    """Herd 9 (PGS with FN / MN) of the run_gleam examples, start weights != weaning weight."""
    def h9(df):
        return df[df["herd_id"] == 9].reset_index(drop=True)

    herd = h9(_run_example("master_hrd_lvl_mixed_data.csv"))
    herd["live_weight_female_nondemographic_start"] = 10.0
    herd["live_weight_male_nondemographic_start"] = 10.0
    return dict(
        cohort_level_data=h9(_run_example("master_chrt_lvl_no_structure_mixed_data.csv")),
        herd_level_data=herd,
        feed_rations=h9(_run_example("feed_rations_share_chrt.csv")),
        feed_params=_run_example("feed_quality.csv"),
        manure_management_system_fraction=h9(_run_example("manure_management_system_fraction.csv")),
        manure_management_system_factors=h9(_run_example("manure_management_system_factors.csv")),
        show_indicator=False,
    )


def _assert_results_identical(actual: dict, expected: dict) -> None:
    assert list(actual) == list(expected)
    for key, exp in expected.items():
        if isinstance(exp, dict):
            _assert_results_identical(actual[key], exp)
        elif exp is None:
            assert actual[key] is None, key
        else:
            pd.testing.assert_frame_equal(actual[key], exp, check_exact=True, obj=key)


@pytest.mark.parametrize("factors_only", [False, True])
def test_numpy_bool_switches_match_python_bools(factors_only):
    args = _herd9_args()
    expected = gleampy.run_emissions_direct(
        has_herd_structure=False, run_demographic=True, run_nondemographic=True,
        emission_factors_only=factors_only, **args,
    )
    cohort = args["cohort_level_data"]
    got = gleampy.run_emissions_direct(
        has_herd_structure=np.False_,
        run_demographic=cohort["cohort_short"].isin(GLEAM_COHORTS_DEMOGRAPHIC).any(),
        run_nondemographic=cohort["cohort_short"].isin(["FN", "MN"]).any(),
        emission_factors_only=np.bool_(factors_only),
        validate_inputs=np.True_,
        **args,
    )
    _assert_results_identical(got, expected)
    # R replaces the start weights by the weaning weight when both herd modules run
    herd = expected["herd_level_results"]
    assert (herd["live_weight_female_nondemographic_start"] == herd["live_weight_at_weaning"]).all()
