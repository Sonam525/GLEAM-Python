"""Tests for run_emissions_direct() (port of tests/testthat/test-run_emissions_direct.R)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gleam
from gleam import GleamValidationError
from gleam.constants import GLEAM_COHORTS_DEMOGRAPHIC
from gleam.io import example_path, read_csv
from gleam.modules import emissions_direct as direct_mod
from gleam.validation.emissions_direct_run import validate_run_emissions_direct_inputs

KEYS = ["herd_id", "species_short", "cohort_short"]


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
    q = gleam.run_ration_quality_module(d_direct["feed_rations"], d_direct["feed_params"], show_indicator=False)
    return q.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


@pytest.fixture(scope="module")
def direct_quality_cols(direct_quality) -> list[str]:
    return [c for c in direct_quality.columns if c not in KEYS + ["nondemo_productive_phase_id"]]


def direct_inputs(d, quality, has_herd_structure=True, primary=False) -> dict:
    cohort = d["cohort_structure"] if has_herd_structure else d["cohort_no_structure"]
    if primary:
        cohort = gleam._utils.merge_dt(cohort, quality, by=KEYS)
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
    feed_result = gleam.run_emissions_direct(**direct_inputs(d_direct, direct_quality, has_structure))
    primary_result = gleam.run_emissions_direct(**direct_inputs(d_direct, direct_quality, has_structure, primary=True))

    feed_chrt = _key_sorted(feed_result["cohort_level_results"])
    prim_chrt = _key_sorted(primary_result["cohort_level_results"])[list(feed_chrt.columns)]
    pd.testing.assert_frame_equal(prim_chrt, feed_chrt, check_dtype=False)
    for key in ("herd_level_results", "allocation_long"):
        pd.testing.assert_frame_equal(primary_result[key], feed_result[key], check_dtype=False)
    for key, df in feed_result["aggregation_results"].items():
        pd.testing.assert_frame_equal(primary_result["aggregation_results"][key], df, check_dtype=False)

    pd.testing.assert_frame_equal(
        prim_chrt[direct_quality_cols].reset_index(drop=True),
        direct_quality[direct_quality_cols].reset_index(drop=True),
        check_dtype=False,
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
            factors = gleam.run_emissions_direct(**args)
        assert list(factors) == list(feed_result)
        assert factors["allocation_long"] is None
        assert factors["aggregation_results"] is None
        factor_cols = list(factors["cohort_level_results"].columns)
        assert not any(("_production_" in c) or ("_allocation_" in c) for c in factor_cols)
        pd.testing.assert_frame_equal(
            _key_sorted(factors["cohort_level_results"]),
            feed_chrt[factor_cols],
            check_dtype=False,
        )


@pytest.mark.parametrize("bad_value", [None, np.nan, 1, "TRUE", [], [True, False]])
def test_emission_factors_only_requires_single_logical(d_direct, direct_quality, bad_value):
    args = direct_inputs(d_direct, direct_quality)
    args["emission_factors_only"] = bad_value
    with pytest.raises(GleamValidationError, match="emission_factors_only.*single logical value"):
        gleam.run_emissions_direct(**args)
    with pytest.raises(GleamValidationError, match="emission_factors_only.*single logical value"):
        validate_run_emissions_direct_inputs(**_validator_args(args))


def test_direct_emissions_match_full_pipeline_sources(d_direct, direct_quality):
    args = direct_inputs(d_direct, direct_quality)
    direct = gleam.run_emissions_direct(**args)
    args["feed_emissions"] = d_direct["feed_emissions"]
    full = gleam.run_gleam(**args)
    full_em = full["aggregation_results"]["results_emissions"]
    full_em = full_em[full_em["variable_name"].str.match(r"^(ch4|n2o)_(enteric|manure)")].reset_index(drop=True)
    pd.testing.assert_frame_equal(direct["aggregation_results"]["results_emissions"], full_em, check_dtype=False)


def test_user_supplied_nitrogen_changes_intake_and_manure(d_direct, direct_quality):
    args = direct_inputs(d_direct, direct_quality, primary=True)
    baseline = gleam.run_emissions_direct(**args)
    args["cohort_level_data"]["ration_nitrogen"] = args["cohort_level_data"]["ration_nitrogen"] * 1.1
    changed = gleam.run_emissions_direct(**args)
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
        gleam.run_emissions_direct(**args)
    for col in direct_quality_cols:
        args = direct_inputs(d_direct, direct_quality, primary=True)
        args["cohort_level_data"] = args["cohort_level_data"].drop(columns=[col])
        with pytest.raises(GleamValidationError, match=col):
            gleam.run_emissions_direct(**args)


@pytest.mark.parametrize("primary", [True, False])
@pytest.mark.parametrize("field", ["feed_rations", "feed_params"])
def test_feed_tables_must_be_supplied_together(d_direct, direct_quality, primary, field):
    args = direct_inputs(d_direct, direct_quality, primary=primary)
    args["feed_rations"] = d_direct["feed_rations"]
    args["feed_params"] = d_direct["feed_params"]
    args[field] = None
    with pytest.raises(GleamValidationError, match="together, or omit both"):
        gleam.run_emissions_direct(**args)


def test_primary_quality_cannot_be_combined_with_feed_tables(d_direct, direct_quality):
    args = direct_inputs(d_direct, direct_quality, primary=True)
    args["feed_rations"] = d_direct["feed_rations"]
    args["feed_params"] = d_direct["feed_params"]
    with pytest.raises(GleamValidationError, match="Do not provide.*ration_gross_energy"):
        gleam.run_emissions_direct(**args)


@pytest.mark.parametrize("bad_value", [np.nan, "unknown", -1, np.inf])
def test_primary_quality_uses_range_validation(d_direct, direct_quality, direct_quality_cols, bad_value):
    for col in direct_quality_cols:
        args = direct_inputs(d_direct, direct_quality, primary=True)
        n = len(args["cohort_level_data"])
        args["cohort_level_data"][col] = pd.Series([bad_value] * n, dtype=object if isinstance(bad_value, str) else float)
        with pytest.raises(GleamValidationError, match=col):
            gleam.run_emissions_direct(**args)


def test_primary_mode_retains_other_input_checks(d_direct, direct_quality):
    args = direct_inputs(d_direct, direct_quality, primary=True)
    args["cohort_level_data"]["daily_weight_gain"] = 0.5
    with pytest.raises(GleamValidationError, match="Do not provide.*daily_weight_gain"):
        gleam.run_emissions_direct(**args)

    args = direct_inputs(d_direct, direct_quality, primary=True)
    args["herd_level_data"]["herd_id"] = args["herd_level_data"]["herd_id"].astype(str) + "_bad"
    with pytest.raises(GleamValidationError, match="same.*herd_id"):
        gleam.run_emissions_direct(**args)

    args = direct_inputs(d_direct, direct_quality, primary=True)
    args["manure_management_system_factors"] = None
    with pytest.raises(GleamValidationError, match="manure_management_system_factors"):
        gleam.run_emissions_direct(**args)


@pytest.mark.parametrize("has_structure", [False, True])
@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("factors_only", [False, True])
def test_direct_emissions_require_all_six_cohorts(d_direct, direct_quality, has_structure, primary, factors_only):
    args = direct_inputs(d_direct, direct_quality, has_herd_structure=has_structure, primary=primary)
    c = args["cohort_level_data"]
    args["cohort_level_data"] = c[c.cohort_short == "FA"].reset_index(drop=True)
    args["emission_factors_only"] = factors_only
    with pytest.raises(GleamValidationError, match="exactly 6 rows"):
        gleam.run_emissions_direct(**args)
