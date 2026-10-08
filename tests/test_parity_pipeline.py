"""End-to-end parity of run_gleam() / run_emissions_direct() with the R package.

Each case reproduces one call in tools/r_reference/generate_golden.R and
compares every output table with the R golden output (exact column and row
order; numbers within ``rtol = 1e-9`` relative, see ``golden_utils``).

Besides the bundled examples, the cases built from ``tests/data`` (see
``tests/data/build_golden_inputs.py``) cover combinations the examples lack:
non-demographic ``FN`` / ``MN`` cohorts of BFL, SHP (``FN``), GTS, CML and
CTL, a CHK herd whose adult females do not lay, run_emissions_direct() with
CHK and non-demographic cohorts (also with primary ration quality per phase),
and the SHP ``MN`` rejection R makes.

The file also tests the comparison helpers in ``golden_utils`` and the
randomised parity tooling in ``tools/parity``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gleampy
from gleampy import GleamValidationError
from gleampy.io import example_path, read_csv
from golden_utils import (
    assert_frame_matches,
    assert_matches_golden,
    golden_error,
    golden_tables,
    load_golden,
    normalize_message,
    quoted_items,
)

NONDEMO_HERDS = (14, 15)
DATA_DIR = Path(__file__).parent / "data"
TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools" / "parity"


def _run(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_gleam_examples"))


def _mod(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_modules_examples"))


def _data(case: str, name: str) -> pd.DataFrame:
    return read_csv(DATA_DIR / case / f"{name}.csv")


def _filter(df: pd.DataFrame, keep_nondemo: bool | None) -> pd.DataFrame:
    if keep_nondemo is None:
        return df
    mask = df["herd_id"].isin(NONDEMO_HERDS)
    return df[mask if keep_nondemo else ~mask].reset_index(drop=True)


def _gleam_inputs(keep_nondemo: bool | None) -> dict:
    return dict(
        feed_rations=_filter(_run("feed_rations_share_chrt.csv"), keep_nondemo),
        feed_params=_run("feed_quality.csv"),
        feed_emissions=_run("feed_emission_factors.csv"),
        manure_management_system_fraction=_filter(_run("manure_management_system_fraction.csv"), keep_nondemo),
        manure_management_system_factors=_filter(_run("manure_management_system_factors.csv"), keep_nondemo),
    )


def _data_inputs(case: str, rations: str = "feed_rations") -> dict:
    """Herd, ration and manure tables of a tests/data case (run_gleam_examples feed tables)."""
    return dict(
        herd_level_data=_data(case, "herd_level_data"),
        feed_rations=_data(case, rations),
        feed_params=_run("feed_quality.csv"),
        feed_emissions=_run("feed_emission_factors.csv"),
        manure_management_system_fraction=_data(case, "manure_management_system_fraction"),
        manure_management_system_factors=_data(case, "manure_management_system_factors"),
    )


# Cases built from tests/data: (data directory, rations file, cohort file, switches)
DATA_GLEAM_CASES = {
    "run_gleam_ruminant_nondemo_no_structure": (
        "ruminant_nondemo", "feed_rations", "cohort_level_data",
        dict(has_herd_structure=False, run_demographic=True, run_nondemographic=True),
    ),
    "run_gleam_ruminant_nondemo_structure": (
        "ruminant_nondemo", "feed_rations", "cohort_level_data_structure",
        dict(has_herd_structure=True, run_demographic=False, run_nondemographic=False),
    ),
    "run_gleam_chk_nonlaying_no_structure": (
        "chk_nonlaying", "feed_rations_share_chrt", "cohort_level_data",
        dict(has_herd_structure=False, run_demographic=True, run_nondemographic=False),
    ),
    "run_gleam_chk_nonlaying_structure": (
        "chk_nonlaying", "feed_rations_share_chrt", "cohort_level_data_structure",
        dict(has_herd_structure=True, run_demographic=False, run_nondemographic=False),
    ),
    "run_gleam_shp_mn_rejected": (
        "shp_mn_rejected", "feed_rations", "cohort_level_data",
        dict(has_herd_structure=False, run_demographic=True, run_nondemographic=True),
    ),
}


def _run_gleam_case(case: str) -> dict:
    if case in DATA_GLEAM_CASES:
        data, rations, cohort, switches = DATA_GLEAM_CASES[case]
        return gleampy.run_gleam(
            cohort_level_data=_data(data, cohort), simulation_duration=365,
            global_warming_potential_set="AR6", show_indicator=False,
            **switches, **_data_inputs(data, rations),
        )
    if case.startswith("run_gleam_mixed_no_structure"):
        return gleampy.run_gleam(
            has_herd_structure=False, run_demographic=True, run_nondemographic=True,
            cohort_level_data=_filter(_run("master_chrt_lvl_no_structure_mixed_data.csv"), False),
            herd_level_data=_filter(_run("master_hrd_lvl_mixed_data.csv"), False),
            simulation_duration=180 if case.endswith("_d180") else 365,
            show_indicator=False, **_gleam_inputs(False),
        )
    if case == "run_gleam_nondemo_only":
        return gleampy.run_gleam(
            has_herd_structure=False, run_demographic=False, run_nondemographic=True,
            cohort_level_data=_run("master_chrt_lvl_no_structure_nondemo_data.csv"),
            herd_level_data=_run("master_hrd_lvl_nondemo_data.csv"),
            simulation_duration=365, show_indicator=False, **_gleam_inputs(True),
        )
    gwp = case.removeprefix("run_gleam_structure_")
    return gleampy.run_gleam(
        has_herd_structure=True, run_demographic=False, run_nondemographic=False,
        cohort_level_data=_run("master_chrt_lvl_structure_data.csv"),
        herd_level_data=_run("master_hrd_lvl_structure_data.csv"),
        simulation_duration=365, global_warming_potential_set=gwp,
        show_indicator=False, **_gleam_inputs(None),
    )


def _run_direct_structure_chk_case(case: str) -> dict:
    """run_emissions_direct() on the run_gleam structure example (CHK, FN, MN)."""
    args = dict(
        has_herd_structure=True,
        herd_level_data=_run("master_hrd_lvl_structure_data.csv"),
        manure_management_system_fraction=_run("manure_management_system_fraction.csv"),
        manure_management_system_factors=_run("manure_management_system_factors.csv"),
        simulation_duration=365, global_warming_potential_set="AR6", show_indicator=False,
    )
    if case.endswith("_rq"):
        return gleampy.run_emissions_direct(cohort_level_data=_data("direct_rq", "cohort_level_data"), **args)
    return gleampy.run_emissions_direct(
        cohort_level_data=_run("master_chrt_lvl_structure_data.csv"),
        feed_rations=_run("feed_rations_share_chrt.csv"), feed_params=_run("feed_quality.csv"), **args,
    )


def _run_direct_case(case: str) -> dict:
    if case.startswith("emissions_direct_3b_"):
        return _run_direct_structure_chk_case(case)
    if case == "emissions_direct_3a_no_structure_mixed":
        inputs = _gleam_inputs(False)
        inputs.pop("feed_emissions")
        return gleampy.run_emissions_direct(
            has_herd_structure=False, run_demographic=True, run_nondemographic=True,
            cohort_level_data=_filter(_run("master_chrt_lvl_no_structure_mixed_data.csv"), False),
            herd_level_data=_filter(_run("master_hrd_lvl_mixed_data.csv"), False),
            simulation_duration=365, global_warming_potential_set="AR6", show_indicator=False, **inputs,
        )
    if case == "emissions_direct_3c_no_structure_ruminant_nondemo":
        inputs = _data_inputs("ruminant_nondemo")
        inputs.pop("feed_emissions")
        return gleampy.run_emissions_direct(
            has_herd_structure=False, run_demographic=True, run_nondemographic=True,
            cohort_level_data=_data("ruminant_nondemo", "cohort_level_data"),
            simulation_duration=365, global_warming_potential_set="AR6", show_indicator=False, **inputs,
        )

    herd = _mod("emissions_direct_input_hrd_data.csv")

    def in_direct(df):
        return df[df["herd_id"].isin(herd["herd_id"])].reset_index(drop=True)

    common = dict(
        herd_level_data=herd,
        manure_management_system_fraction=in_direct(_mod("manure_management_system_fraction.csv")),
        manure_management_system_factors=in_direct(_mod("manure_management_system_factors.csv")),
        simulation_duration=365, global_warming_potential_set="AR6", show_indicator=False,
    )
    feed = dict(feed_rations=in_direct(_mod("feed_rations_share_chrt.csv")), feed_params=_mod("feed_quality.csv"))
    cohort_file = {
        "emissions_direct_1a_no_structure": "emissions_direct_input_chrt_no_structure_data.csv",
        "emissions_direct_1b_structure": "emissions_direct_input_chrt_structure_data.csv",
        "emissions_direct_2a_no_structure_rq": "emissions_direct_input_chrt_no_structure_ration_quality_data.csv",
        "emissions_direct_2b_structure_rq": "emissions_direct_input_chrt_structure_ration_quality_data.csv",
        "emissions_direct_1b_structure_ef_only": "emissions_direct_input_chrt_structure_data.csv",
    }[case]
    args = dict(common, cohort_level_data=_mod(cohort_file), has_herd_structure="_structure" in case and "no_structure" not in case)
    if not case.endswith("_rq"):
        args.update(feed)
    if case.endswith("_ef_only"):
        args["emission_factors_only"] = True
    return gleampy.run_emissions_direct(**args)


def _flatten(result: dict, prefix: str = "") -> dict[str, pd.DataFrame]:
    out = {}
    for k, v in result.items():
        name = f"{prefix}__{k}" if prefix else k
        if isinstance(v, pd.DataFrame):
            out[name] = v
        elif isinstance(v, dict):
            out.update(_flatten(v, name))
    return out


GLEAM_CASES = [
    "run_gleam_mixed_no_structure",
    "run_gleam_mixed_no_structure_d180",
    "run_gleam_nondemo_only",
    "run_gleam_structure_AR6",
    "run_gleam_structure_AR5_excluding_carbon_feedback",
    "run_gleam_structure_AR5_including_carbon_feedback",
    "run_gleam_structure_AR4",
    "run_gleam_ruminant_nondemo_no_structure",
    "run_gleam_ruminant_nondemo_structure",
    "run_gleam_chk_nonlaying_no_structure",
    "run_gleam_chk_nonlaying_structure",
]
DIRECT_CASES = [
    "emissions_direct_1a_no_structure",
    "emissions_direct_1b_structure",
    "emissions_direct_2a_no_structure_rq",
    "emissions_direct_2b_structure_rq",
    "emissions_direct_1b_structure_ef_only",
    "emissions_direct_3a_no_structure_mixed",
    "emissions_direct_3b_structure_chk_nondemo",
    "emissions_direct_3b_structure_chk_nondemo_rq",
    "emissions_direct_3c_no_structure_ruminant_nondemo",
]
# Cases where R stops with a validation error (golden ERROR.txt).
ERROR_CASES = ["run_gleam_shp_mn_rejected"]


def _run_case(case: str) -> dict:
    return _run_gleam_case(case) if case.startswith("run_gleam") else _run_direct_case(case)


@pytest.mark.parametrize("case", GLEAM_CASES + DIRECT_CASES)
def test_pipeline_matches_r(case):
    tables = _flatten(_run_case(case))
    expected = golden_tables(case)
    assert sorted(tables) == expected, f"output tables differ: {sorted(tables)} vs {expected}"
    for table in expected:
        assert_matches_golden(tables[table], case, table)


@pytest.mark.parametrize("case", ERROR_CASES)
def test_pipeline_rejects_like_r(case):
    with pytest.raises(GleamValidationError) as err:
        _run_case(case)
    assert normalize_message(str(err.value)) == normalize_message(golden_error(case))


def test_every_golden_case_is_tested():
    cases = {p.name for p in (Path(__file__).parent / "golden").iterdir() if p.is_dir()}
    pipeline = {c for c in cases if c.startswith(("run_gleam", "emissions_direct"))}
    assert pipeline == set(GLEAM_CASES + DIRECT_CASES + ERROR_CASES)


def test_metabolic_energy_of_growing_chk_hens_matches_r():
    """CHK FA with weight gain, laying and not (R ignores the flag there)."""
    res = gleampy.run_metabolic_energy_req_module(
        cohort_level_data=_data("mer_chk_growth", "cohort_level_data"),
        herd_level_data=_data("mer_chk_growth", "herd_level_data"),
        show_indicator=False,
    )
    case = "metabolic_energy_req_module_chk_fa_growth"
    assert golden_tables(case) == ["result"]
    assert_matches_golden(res, case, "result")
    exp = load_golden(case, "result")
    fa = exp[(exp["species_short"] == "CHK") & (exp["cohort_short"] == "FA") & (exp["daily_weight_gain"] > 0)]
    assert sorted(fa["is_egg_producing"].tolist()) == [False, True]


# Which combinations the extra cases are there for: if an input file or the
# pipeline drops them, the parity check above would silently cover less.
@pytest.mark.parametrize(
    "case,combos",
    [
        ("run_gleam_ruminant_nondemo_no_structure",
         {"CTL-FN", "CTL-MN", "BFL-FN", "BFL-MN", "SHP-FN", "GTS-FN", "GTS-MN", "CML-FN", "CML-MN"}),
        ("run_gleam_ruminant_nondemo_structure",
         {"CTL-FN", "CTL-MN", "BFL-FN", "BFL-MN", "SHP-FN", "GTS-FN", "GTS-MN", "CML-FN", "CML-MN"}),
        ("emissions_direct_3c_no_structure_ruminant_nondemo",
         {"CTL-FN", "BFL-MN", "SHP-FN", "GTS-MN", "CML-FN"}),
        ("emissions_direct_3b_structure_chk_nondemo", {"CHK-FA", "CHK-FN", "CHK-MN", "CTL-MN", "PGS-FN"}),
        ("emissions_direct_3b_structure_chk_nondemo_rq", {"CHK-FA", "CHK-FN", "CHK-MN", "CTL-MN", "PGS-FN"}),
    ],
)
def test_extra_cases_cover_their_combinations(case, combos):
    chrt = load_golden(case, "cohort_level_results")
    present = set(chrt["species_short"] + "-" + chrt["cohort_short"])
    assert combos <= present, combos - present


@pytest.mark.parametrize("case", ["run_gleam_chk_nonlaying_no_structure", "run_gleam_chk_nonlaying_structure"])
def test_nonlaying_chk_case_has_no_eggs(case):
    chrt = load_golden(case, "cohort_level_results")
    fa = chrt[chrt["cohort_short"] == "FA"]
    assert len(fa) == 1 and fa["is_egg_producing"].tolist() == [False]
    assert (chrt["egg_production_mass_cohort"] == 0).all()
    assert (fa["daily_weight_gain"] != 0).all() or (fa["metabolic_energy_req_maintenance"] > 0).all()


# ---------------------------------------------------------------------------
# golden_utils: the comparison itself
# ---------------------------------------------------------------------------


def _perturbed(case: str, table: str, col: str, rows, factor: float) -> pd.DataFrame:
    exp = load_golden(case, table)
    act = exp.copy()
    v = act[col].to_numpy(dtype=float).copy()
    v[rows] = v[rows] * factor
    act[col] = v
    return act


def test_small_values_are_compared_relatively():
    """A 1e-5 relative error on values below 1e-5 must fail (atol=1e-10 used to hide it)."""
    case, table = "emissions_direct_1a_no_structure", "cohort_level_results"
    exp = load_golden(case, table)
    for col in ("n2o_manure_other_leach", "ch4_manure_pasture", "n2o_manure_pasture_vol"):
        v = exp[col].to_numpy(dtype=float)
        rows = np.flatnonzero((np.abs(v) < 1e-5) & (v != 0))
        assert len(rows) > 0
        with pytest.raises(AssertionError, match=col):
            assert_matches_golden(_perturbed(case, table, col, rows, 1 + 1e-6), case, table)
        assert_matches_golden(_perturbed(case, table, col, rows, 1 + 1e-11), case, table)


def test_tiny_values_in_long_tables_are_compared_relatively():
    """In long tables one column mixes quantities 1e12 apart; each is checked at rtol."""
    case, table = "run_gleam_structure_AR6", "aggregation_results__results_emissions"
    exp = load_golden(case, table)
    v = exp["value_total_gas"].to_numpy(dtype=float)
    smallest = int(np.argmin(np.where(v > 0, v, np.inf)))
    assert v[smallest] < 1e-9 * np.nanmax(v)
    with pytest.raises(AssertionError, match="value_total_gas"):
        assert_matches_golden(_perturbed(case, table, "value_total_gas", [smallest], 1 + 1e-7), case, table)


def _with_values(df: pd.DataFrame, col: str, rows, value) -> pd.DataFrame:
    out = df.copy()
    v = out[col].to_numpy(dtype=float).copy()
    v[rows] = value
    out[col] = v
    return out


def test_r_zeros_must_be_exact_zeros():
    """No absolute term by default: an R zero matches only a zero, however large the column.

    Regression: a floor of 1e-12 x the column maximum let 0.1 pass for 59 of
    the 61 R zeros of this column (maximum 1.45e11).
    """
    case, table, col = "run_gleam_structure_AR6", "cohort_level_results", "milk_allocation_energy"
    v = load_golden(case, table)[col].to_numpy(dtype=float)
    zeros = np.flatnonzero(v == 0)
    assert len(zeros) == 61 and np.nanmax(v) > 1e11
    exp = load_golden(case, table)
    for leak in (0.1, 1e-300):
        with pytest.raises(AssertionError, match=f"{col}: 61 mismatches"):
            assert_matches_golden(_with_values(exp, col, zeros, leak), case, table)
    assert_matches_golden(_with_values(exp, col, zeros, -0.0), case, table)


def test_exact_comparison_is_bit_exact():
    """rtol = atol = 0 (the bit-identity checks) has no near-zero rule."""
    exp = pd.DataFrame({"species_short": ["CTL"] * 3, "x": [1000.0, 0.0, -3e-10]})
    assert_frame_matches(exp.copy(), exp, rtol=0, atol=0)
    for bad in ([1000.0, 5e-10, -3e-10], [1000.0, 0.0, 5e-10], [np.nextafter(1000.0, 2000.0), 0.0, -3e-10]):
        act = exp.copy()
        act["x"] = bad
        with pytest.raises(AssertionError, match="x: 1 mismatches"):
            assert_frame_matches(act, exp, rtol=0, atol=0)
    with pytest.raises(ValueError, match="exact comparison"):
        assert_frame_matches(exp.copy(), exp, rtol=0, atol=0, zero_floor=1e-12)
    with pytest.raises(ValueError, match=">= 0"):
        assert_frame_matches(exp.copy(), exp, zero_floor=-1e-12)


def test_atol_applies_to_zeros_too():
    exp = pd.DataFrame({"x": [0.0, 1.0]})
    act = pd.DataFrame({"x": [1e-12, 1.0]})
    assert_frame_matches(act, exp, atol=1e-10)
    assert_frame_matches(act, exp, rtol=0, atol=1e-10)
    with pytest.raises(AssertionError, match="x: 1 mismatches"):
        assert_frame_matches(act, exp)


def test_opt_in_zero_floor_only_absorbs_cancellation_noise():
    exp = pd.DataFrame({"species_short": ["CTL", "CTL", "CHK", "CHK"], "x": [1000.0, 0.0, 2.0, 0.0]})
    noise = exp.copy()
    noise["x"] = [1000.0, 1e-13, 2.0, -1e-15]  # <= 1e-12 x the CTL / CHK scale
    assert_frame_matches(noise, exp, zero_floor=1e-12)
    with pytest.raises(AssertionError, match="x: 2 mismatches"):
        assert_frame_matches(noise, exp)  # off by default
    for bad in ([1000.0, 1e-8, 2.0, 0.0], [1000.0, 0.0, 2.0, 1e-11]):
        act = exp.copy()
        act["x"] = bad
        with pytest.raises(AssertionError, match="x: 1 mismatches"):
            assert_frame_matches(act, exp, zero_floor=1e-12)
    # an R value is never "zero" because it is small next to another species
    exp2 = pd.DataFrame({"species_short": ["CTL", "CHK"], "x": [1e6, 1e-8]})
    act2 = exp2.copy()
    act2["x"] = [1e6, 1.0001e-8]
    with pytest.raises(AssertionError, match="x: 1 mismatches"):
        assert_frame_matches(act2, exp2, zero_floor=1e-12)
    # a group whose R values are all zero never borrows the column's scale
    exp3 = pd.DataFrame({"species_short": ["CTL", "PGS", "PGS"], "x": [1.5e11, 0.0, 0.0]})
    act3 = exp3.copy()
    act3["x"] = [1.5e11, 0.1, 1e-300]
    with pytest.raises(AssertionError, match="x: 2 mismatches"):
        assert_frame_matches(act3, exp3, zero_floor=1e-12)
    # the golden column of test_r_zeros_must_be_exact_zeros: only the R zero of
    # herd 2 CTL FA shares a group (CTL, FA) with a large non-zero value
    case, table, col = "run_gleam_structure_AR6", "cohort_level_results", "milk_allocation_energy"
    exp4 = load_golden(case, table)
    zeros = np.flatnonzero(exp4[col].to_numpy(dtype=float) == 0)
    with pytest.raises(AssertionError, match=f"{col}: 60 mismatches"):
        assert_matches_golden(_with_values(exp4, col, zeros, 0.1), case, table, zero_floor=1e-12)


def test_numbers_close_rejects_a_floor_in_exact_mode():
    from golden_utils import numbers_close, zero_floor_per_row

    a, e = np.array([5e-10, 0.0]), np.array([-3e-10, 0.0])
    assert numbers_close(a, e).tolist() == [False, True]
    with pytest.raises(ValueError, match="exact comparison"):
        numbers_close(a, e, 1e-9, rtol=0, atol=0)
    floors = zero_floor_per_row(np.array([1000.0, 0.0, 0.0, 0.0]), np.array([0, 0, 1, 1]), 1e-12)
    assert floors.tolist() == [1e-9, 1e-9, 0.0, 0.0]


@pytest.mark.parametrize(
    "case,table,col,bad",
    [
        ("emissions_direct_1a_no_structure", "cohort_level_results", "is_egg_producing", "oops"),
        ("emissions_direct_1a_no_structure", "cohort_level_results", "nondemo_productive_phase_id", "FN"),
        ("weights_module", "cohort_level_results", "daily_weight_gain", "as_text"),
        ("weights_module", "cohort_level_results", "daily_weight_gain", "as_bool"),
        ("run_gleam_nondemo_only", "herd_level_results", "milk_yield_day", "not-a-number"),
    ],
)
def test_non_numeric_values_never_pass_as_numbers(case, table, col, bad):
    exp = load_golden(case, table)
    act = exp.copy()
    if bad == "as_text":
        act[col] = [repr(v) for v in exp[col].tolist()]
    elif bad == "as_bool":
        act[col] = exp[col].to_numpy(dtype=float) != 0
    else:
        vals = act[col].astype(object).tolist()
        act[col] = pd.Series([bad if pd.isna(v) else v for v in vals], dtype=object)
    with pytest.raises(AssertionError, match=col):
        assert_matches_golden(act, case, table)


def test_logical_columns_require_logical_values():
    exp = pd.DataFrame({"flag": [True, False, None]}, dtype=object)
    assert_frame_matches(pd.DataFrame({"flag": pd.array([True, False, None], dtype="boolean")}), exp)
    for bad in ([1, 0, None], ["TRUE", "FALSE", None]):
        with pytest.raises(AssertionError, match="flag"):
            assert_frame_matches(pd.DataFrame({"flag": pd.Series(bad, dtype=object)}), exp)


def test_nullable_numeric_actuals_compare_as_numbers():
    exp = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    assert_frame_matches(pd.DataFrame({"x": pd.array([1, None, 3], dtype="Int64")}), exp)
    assert_frame_matches(pd.DataFrame({"x": pd.array([1.0, None, 3.0], dtype="Float64")}), exp)
    with pytest.raises(AssertionError, match="x"):
        assert_frame_matches(pd.DataFrame({"x": pd.array([1.0, 2.0, 3.0], dtype="Float64")}), exp)


# ---------------------------------------------------------------------------
# Message normalisation (R vs Python validation errors)
# ---------------------------------------------------------------------------

R_WEIGHT_MSG = (
    "For each row, live_weight_cohort_initial <= live_weight_cohort_average\n"
    '<= live_weight_cohort_final must hold. Violation(s): "11 / MA" and "13 / FA"'
)
PY_WEIGHT_MSG = (
    "For each row, `live_weight_cohort_initial` <= `live_weight_cohort_average` <= "
    '`live_weight_cohort_final` must hold. Violation(s): "11 / MA", "13 / FA"'
)


def test_normalize_message_removes_formatting_only():
    assert normalize_message(R_WEIGHT_MSG) == normalize_message(PY_WEIGHT_MSG)
    assert quoted_items(R_WEIGHT_MSG) == quoted_items(PY_WEIGHT_MSG) == ["11 / MA", "13 / FA"]
    r = "`offtake_rate` = 1 is out of range; expected value should be ≥ -2 and <\n1."
    py = "`offtake_rate`[6] = 1 is out of range; expected value should be >= -2 and < 1."
    assert normalize_message(r) == normalize_message(py)
    assert normalize_message("Missing {.arg x} in {.val y}") == "Missing x in y"
    assert normalize_message(r) != normalize_message(r.replace("= 1", "= 2"))


# ---------------------------------------------------------------------------
# tools/parity
# ---------------------------------------------------------------------------


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_parity_{name}", TOOLS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compare_scenarios():
    return _tool("compare_scenarios")


@pytest.fixture(scope="module")
def make_scenarios():
    return _tool("make_scenarios")


def test_errors_match_requires_same_validation_error(compare_scenarios):
    match = compare_scenarios.errors_match
    assert match(R_WEIGHT_MSG, PY_WEIGHT_MSG)[0]
    assert match(R_WEIGHT_MSG, PY_WEIGHT_MSG, ["rlang_error", "error", "condition"])[0]
    # different rows, an R crash, one-sided failures
    ok, why = match(R_WEIGHT_MSG, PY_WEIGHT_MSG.replace('"13 / FA"', '"13 / FJ"'))
    assert not ok and "violation lists" in why
    assert not match("missing value where TRUE/FALSE needed", PY_WEIGHT_MSG)[0]
    assert not match("object 'xyz' not found", "`milk_yield_day`[1] = -5 is out of range")[0]
    assert not match(R_WEIGHT_MSG, PY_WEIGHT_MSG, ["simpleError", "error", "condition"])[0]
    assert not match(R_WEIGHT_MSG, None)[0]
    assert not match(None, PY_WEIGHT_MSG)[0]
    # same row, different check
    r = "`milk_fat_fraction` = 2 is out of range; expected value should be ≥ 0\nand ≤ 1."
    py = "`milk_yield_day`[9] = -5 is out of range; expected value should be >= 0 and <= 100."
    assert not match(r, py)[0]


def _scenario_dir(tmp_path: Path, r_error: str, py_args_extra: dict | None = None) -> Path:
    """A one-scenario directory whose Python run is rejected (negative milk yield)."""
    import json

    sd = tmp_path / "structure_000"
    (sd / "r_out").mkdir(parents=True)
    tables = dict(
        cohort_level_data=_run("master_chrt_lvl_structure_data.csv"),
        herd_level_data=_run("master_hrd_lvl_structure_data.csv"),
        **_gleam_inputs(None),
    )
    tables["herd_level_data"].loc[0, "milk_yield_day"] = -5.0
    for name, df in tables.items():
        df.to_csv(sd / f"{name}.csv", sep="\t", index=False, na_rep="")
    args = dict(has_herd_structure=True, run_demographic=False, run_nondemographic=False,
                simulation_duration=365.0, global_warming_potential_set="AR6", base="structure")
    args.update(py_args_extra or {})
    (sd / "args.json").write_text(json.dumps(args))
    (sd / "r_out" / "ERROR.txt").write_text(r_error, encoding="utf-8")
    return sd


def test_compare_fails_on_unrelated_r_error(compare_scenarios, tmp_path):
    sd = _scenario_dir(tmp_path, "object 'xyz' not found")
    ok, msg, _ = compare_scenarios.compare(sd)
    assert not ok, msg


def test_compare_passes_on_same_r_error(compare_scenarios, tmp_path):
    r = "`milk_yield_day` = -5 is out of range; expected value should be ≥ 0\nand ≤ 100."
    ok, msg, _ = compare_scenarios.compare(_scenario_dir(tmp_path, r))
    assert ok, msg


def test_compare_fails_when_r_crashed_with_the_same_text(compare_scenarios, tmp_path):
    r = "`milk_yield_day` = -5 is out of range; expected value should be ≥ 0\nand ≤ 100."
    sd = _scenario_dir(tmp_path, r)
    (sd / "r_out" / "ERROR_CLASS.txt").write_text("simpleError\nerror\ncondition\n")
    ok, msg, _ = compare_scenarios.compare(sd)
    assert not ok and "R crashed" in msg
    (sd / "r_out" / "ERROR_CLASS.txt").write_text("rlang_error\nerror\ncondition\n")
    ok, msg, _ = compare_scenarios.compare(sd)
    assert ok, msg


def test_compare_scenarios_cli_zero_floor_is_opt_in(compare_scenarios, tmp_path, monkeypatch, capsys):
    import inspect

    assert inspect.signature(compare_scenarios.compare).parameters["zero_floor"].default == 0.0
    monkeypatch.setattr(sys, "argv", ["compare_scenarios.py", str(tmp_path), "--zero-floor", "1e-12", "--stats"])
    with pytest.raises(SystemExit) as exc:
        compare_scenarios.main()
    assert exc.value.code == 0 and "0/0 scenarios match" in capsys.readouterr().out


def test_agreement_counts_bit_identical_numbers(compare_scenarios):
    """The statistics behind the documented randomised-scenario agreement."""
    r = pd.DataFrame({"herd_id": [1, 2, 3], "x": [1.0, 0.0, 3.0], "s": ["a", "b", "c"], "na": [None] * 3})
    py = r.copy()
    py["x"] = [1.0, 0.0, np.nextafter(3.0, 4.0)]
    a = compare_scenarios.agreement(py, r, "t")
    # non-zero numbers: herd_id 1, 2, 3 (whole numbers) and x 1.0, 3.0; the zero is not counted
    assert (a["values"], a["identical"], a["float_values"], a["float_identical"]) == (5, 4, 2, 1)
    assert a["where"][:2] == ("t", "x") and a["max_rel"] == np.spacing(3.0) / 3.0
    assert compare_scenarios.agreement(r.copy(), r)["max_rel"] == 0.0


def test_compare_reports_python_crash_as_failure(compare_scenarios, tmp_path):
    sd = _scenario_dir(tmp_path, "anything", {"not_an_argument": 1})
    ok, msg, _ = compare_scenarios.compare(sd)
    assert not ok and "Python crashed" in msg


def test_renormalize_includes_missing_phase_ids(make_scenarios):
    df = pd.DataFrame({
        "herd_id": [1, 1, 14, 14],
        "cohort_short": ["FA", "FA", "FN", "FN"],
        "nondemo_productive_phase_id": [np.nan, np.nan, 1.0, 1.0],
        "x": [0.5, 0.5, 0.5, 0.5],
    })
    make_scenarios._renormalize(df, "x", ["herd_id", "cohort_short", "nondemo_productive_phase_id"],
                                np.random.default_rng(0), 0.15)
    # pandas 2 and 3 alike: the demographic (NA phase) group is perturbed too
    assert df["x"].iloc[0] != 0.5 and df["x"].iloc[2] != 0.5
    np.testing.assert_allclose(df.groupby("herd_id")["x"].sum(), [1.0, 1.0], rtol=1e-15)
    # the values pandas 2.2 gave before the fix (up to platform rounding of the draws)
    expected = [0.5279287835754416, 0.47207121642455835, 0.5021353224739558, 0.49786467752604424]
    np.testing.assert_allclose(df["x"], expected, rtol=1e-14)


@pytest.mark.parametrize("base", ["structure", "mixed"])
def test_scenarios_perturb_demographic_shares(make_scenarios, base):
    tables, _ = make_scenarios.make_scenario(base, np.random.default_rng(20261006))
    which = make_scenarios.BASES[base][2]
    for table, col in (("feed_rations", "feed_ration_fraction"),
                       ("manure_management_system_fraction", "manure_management_system_fraction")):
        src = "feed_rations_share_chrt.csv" if table == "feed_rations" else f"{table}.csv"
        orig = make_scenarios._filter(_run(src), which)
        new = tables[table]
        keys = ["herd_id", "cohort_short", "nondemo_productive_phase_id"]
        group_size = orig.groupby(keys, dropna=False)[col].transform("size").to_numpy()
        # shares of 0 and single-feed shares of 1 cannot change
        demo = orig["nondemo_productive_phase_id"].isna().to_numpy() & (group_size > 1) & (orig[col].to_numpy() > 0)
        assert demo.sum() > 100
        changed = new[col].to_numpy(dtype=float) != orig[col].to_numpy(dtype=float)
        assert changed[demo].all(), f"{table}: {int((~changed[demo]).sum())} demographic shares unchanged"
        sums = new.groupby(keys, dropna=False)[col].sum().to_numpy()
        np.testing.assert_allclose(sums, 1.0, rtol=1e-12)
