"""Port of ``tests/testthat/test-aggregation_core.R`` plus vectorisation checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gleampy import (
    GleamValidationError,
    calc_allocated_emissions,
    calc_co2eq,
    calc_cohort_totals,
    run_aggregation_module,
)
from gleampy.constants import GLEAM_FEED_EMISSIONS_META
from gleampy.io import load_example
from float_compare import assert_same_float

REL = 1.5e-8  # testthat tolerance

# Feed emissions use ration_intake in scaling; pass empty list when not testing feed emissions
feed_emissions_empty: list = []


def approx(x):
    return pytest.approx(x, rel=REL)


# ---- calc_cohort_totals ----------------------------------------------------


def test_calc_cohort_totals_returns_correct_value_for_production_variables():
    result = calc_cohort_totals(
        value=1000,
        cohort_stock_size=50,
        ration_intake=10,
        feed_emissions_list=feed_emissions_empty,
        simulation_duration=365,
        variable_name="milk_production_mass_cohort",
        variable_type="Production",
    )
    assert isinstance(result, float)
    # Production variables are returned as-is (not scaled)
    assert result == approx(1000)


def test_calc_cohort_totals_returns_correct_value_for_emissions_variables():
    result = calc_cohort_totals(
        value=0.5,
        cohort_stock_size=100,
        ration_intake=5,
        feed_emissions_list=feed_emissions_empty,
        simulation_duration=365,
        variable_name="ch4_enteric",
        variable_type="Emissions",
    )
    assert isinstance(result, float)
    assert result == approx(0.5 * 100 * 365)


def test_calc_cohort_totals_returns_correct_value_for_feed_emissions():
    feed_emissions = [{"emissions_source": "co2_ration_fertilizer", "label": "Feed-Fertilizer_CO2"}]
    result = calc_cohort_totals(
        value=0.1,
        cohort_stock_size=100,
        ration_intake=10,
        feed_emissions_list=feed_emissions,
        simulation_duration=365,
        variable_name="co2_ration_fertilizer",
        variable_type="Emissions",
    )
    assert result == approx(0.1 * 10 * 100 * 365 / 1000)


def test_calc_cohort_totals_returns_correct_value_for_feed_variables():
    result = calc_cohort_totals(
        value=10,
        cohort_stock_size=30,
        ration_intake=10,
        feed_emissions_list=feed_emissions_empty,
        simulation_duration=365,
        variable_name="ration_intake",
        variable_type="Feed",
    )
    assert result == approx(10 * 30 * 365)


def test_calc_cohort_totals_returns_correct_value_for_nitrogen_balance_variables():
    result = calc_cohort_totals(
        value=0.2,
        cohort_stock_size=25,
        ration_intake=8,
        feed_emissions_list=feed_emissions_empty,
        simulation_duration=365,
        variable_name="nitrogen_intake",
        variable_type="NitrogenBalance",
    )
    assert result == approx(0.2 * 25 * 365)


def test_calc_cohort_totals_validates_variable_type():
    with pytest.raises(GleamValidationError, match="must be one of"):
        calc_cohort_totals(
            value=100,
            cohort_stock_size=50,
            ration_intake=10,
            feed_emissions_list=feed_emissions_empty,
            simulation_duration=365,
            variable_name="x",
            variable_type="Invalid",
        )


def test_calc_cohort_totals_validates_bounds():
    with pytest.raises(GleamValidationError, match="must be positive"):
        calc_cohort_totals(
            value=100,
            cohort_stock_size=0,
            ration_intake=10,
            feed_emissions_list=feed_emissions_empty,
            simulation_duration=365,
            variable_name="ch4_enteric",
            variable_type="Emissions",
        )
    with pytest.raises(GleamValidationError, match="must be positive"):
        calc_cohort_totals(
            value=100,
            cohort_stock_size=50,
            ration_intake=10,
            feed_emissions_list=feed_emissions_empty,
            simulation_duration=-10,
            variable_name="ch4_enteric",
            variable_type="Emissions",
        )


def test_calc_cohort_totals_invalid_type_message_lists_values_like_cli():
    with pytest.raises(
        GleamValidationError,
        match="must be one of: Production, Emissions, Feed, and NitrogenBalance. Found invalid values: Bad and Worse",
    ):
        calc_cohort_totals(1, 1, 1, [], 365, "x", ["Bad", "Worse"])


# ---- calc_allocated_emissions ----------------------------------------------


def test_calc_allocated_emissions_returns_correct_value_for_valid_inputs():
    result = calc_allocated_emissions(value=1000, allocation_share=0.6)
    assert isinstance(result, float)
    assert result == approx(1000 * 0.6)


def test_calc_allocated_emissions_handles_zero_allocation():
    assert calc_allocated_emissions(value=1000, allocation_share=0) == approx(0)


def test_calc_allocated_emissions_handles_full_allocation():
    assert calc_allocated_emissions(value=1000, allocation_share=1) == approx(1000)


def test_calc_allocated_emissions_handles_vectorized_inputs():
    result = calc_allocated_emissions(value=[1000, 500, 200], allocation_share=[0.6, 0.4, 0.8])
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    assert result == approx([1000 * 0.6, 500 * 0.4, 200 * 0.8])


def test_calc_allocated_emissions_validates_input_lengths():
    with pytest.raises(GleamValidationError, match="must have the same length"):
        calc_allocated_emissions(value=[100, 200], allocation_share=[0.5, 0.6, 0.7])


def test_calc_allocated_emissions_validates_bounds():
    with pytest.raises(GleamValidationError, match="must be between 0 and 1"):
        calc_allocated_emissions(value=100, allocation_share=-0.1)
    with pytest.raises(GleamValidationError, match="must be between 0 and 1"):
        calc_allocated_emissions(value=100, allocation_share=1.5)


def test_calc_allocated_emissions_rejects_missing_share_like_r():
    # R: if (any(NA < 0 | NA > 1)) -> "missing value where TRUE/FALSE needed"
    with pytest.raises(GleamValidationError, match="missing value where TRUE/FALSE needed"):
        calc_allocated_emissions(value=100, allocation_share=np.nan)


# ---- calc_co2eq ------------------------------------------------------------


def test_calc_co2eq_returns_correct_value_for_ch4_with_ar6():
    result = calc_co2eq(gas="CH4", value_allocated=100, global_warming_potential_set="AR6")
    assert isinstance(result, dict)
    assert list(result) == ["value_co2eq", "gwp"]
    assert result["value_co2eq"] == approx(100 * 27)
    assert result["gwp"] == approx(27)


def test_calc_co2eq_returns_correct_value_for_n2o_with_ar6():
    result = calc_co2eq(gas="N2O", value_allocated=10, global_warming_potential_set="AR6")
    assert result["value_co2eq"] == approx(10 * 273)
    assert result["gwp"] == approx(273)


def test_calc_co2eq_returns_correct_value_for_co2():
    result = calc_co2eq(gas="CO2", value_allocated=1000, global_warming_potential_set="AR6")
    assert result["value_co2eq"] == approx(1000 * 1)
    assert result["gwp"] == approx(1)


def test_calc_co2eq_handles_ar5_excluding_carbon_feedback():
    result_ch4 = calc_co2eq(gas="CH4", value_allocated=100, global_warming_potential_set="AR5_excluding_carbon_feedback")
    assert result_ch4["value_co2eq"] == approx(100 * 28)
    assert result_ch4["gwp"] == approx(28)
    result_n2o = calc_co2eq(gas="N2O", value_allocated=10, global_warming_potential_set="AR5_excluding_carbon_feedback")
    assert result_n2o["value_co2eq"] == approx(10 * 265)
    assert result_n2o["gwp"] == approx(265)


def test_calc_co2eq_handles_ar5_including_carbon_feedback():
    result_ch4 = calc_co2eq(gas="CH4", value_allocated=100, global_warming_potential_set="AR5_including_carbon_feedback")
    assert result_ch4["value_co2eq"] == approx(100 * 34)
    assert result_ch4["gwp"] == approx(34)
    result_n2o = calc_co2eq(gas="N2O", value_allocated=10, global_warming_potential_set="AR5_including_carbon_feedback")
    assert result_n2o["value_co2eq"] == approx(10 * 298)
    assert result_n2o["gwp"] == approx(298)


def test_calc_co2eq_handles_ar4():
    result_ch4 = calc_co2eq(gas="CH4", value_allocated=100, global_warming_potential_set="AR4")
    assert result_ch4["value_co2eq"] == approx(100 * 25)
    assert result_ch4["gwp"] == approx(25)
    result_n2o = calc_co2eq(gas="N2O", value_allocated=10, global_warming_potential_set="AR4")
    assert result_n2o["value_co2eq"] == approx(10 * 298)
    assert result_n2o["gwp"] == approx(298)


def test_calc_co2eq_handles_vectorized_inputs():
    result = calc_co2eq(gas=["CH4", "N2O", "CO2"], value_allocated=[100, 10, 1000], global_warming_potential_set="AR6")
    assert isinstance(result, dict)
    assert len(result["value_co2eq"]) == 3
    assert len(result["gwp"]) == 3
    assert result["value_co2eq"] == approx([100 * 27, 10 * 273, 1000 * 1])
    assert result["gwp"] == approx([27, 273, 1])


def test_calc_co2eq_handles_zero_emissions():
    result = calc_co2eq(gas="CH4", value_allocated=0, global_warming_potential_set="AR6")
    assert result["value_co2eq"] == approx(0)
    assert result["gwp"] == approx(27)


def test_calc_co2eq_validates_gwp_version():
    with pytest.raises(GleamValidationError, match="must be one of"):
        calc_co2eq(gas="CH4", value_allocated=100, global_warming_potential_set="INVALID")


def test_calc_co2eq_validates_input_lengths():
    with pytest.raises(GleamValidationError, match="must have the same length"):
        calc_co2eq(gas=["CH4", "N2O"], value_allocated=[100, 10, 50], global_warming_potential_set="AR6")


def test_calc_co2eq_validates_gas_types():
    with pytest.raises(GleamValidationError, match="must be one of"):
        calc_co2eq(gas="INVALID", value_allocated=100, global_warming_potential_set="AR6")


# ---- run_aggregation_module validation ---------------------------------------


def _aggregation_inputs():
    return load_example("aggregation_input_chrt_data.csv"), load_example("aggregation_allocation_input_data.csv")


def test_run_aggregation_module_validates_inputs():
    chrt, alloc = _aggregation_inputs()
    with pytest.raises(GleamValidationError, match="global_warming_potential_set"):
        run_aggregation_module(chrt, alloc, global_warming_potential_set="AR9", show_indicator=False)
    with pytest.raises(GleamValidationError, match="must be positive"):
        run_aggregation_module(chrt, alloc, simulation_duration=0, show_indicator=False)
    with pytest.raises(GleamValidationError, match="must be a data.table"):
        run_aggregation_module(chrt, alloc.to_dict(), show_indicator=False)
    bad = chrt.copy()
    bad.loc[bad["herd_id"] == 1, "species_short"] = "BFL"
    with pytest.raises(GleamValidationError, match=r"combinations in `cohort_level_data` have no"):
        run_aggregation_module(bad, alloc, show_indicator=False)
    with pytest.raises(GleamValidationError, match="not found in `allocation_herd_long`"):
        run_aggregation_module(chrt, alloc[alloc["herd_id"] != 3], show_indicator=False)


def test_run_aggregation_module_does_not_mutate_inputs():
    chrt, alloc = _aggregation_inputs()
    c0, a0 = chrt.copy(), alloc.copy()
    run_aggregation_module(chrt, alloc, show_indicator=False)
    pd.testing.assert_frame_equal(chrt, c0)
    pd.testing.assert_frame_equal(alloc, a0)


# ---- vectorisation: arrays give the same result as element-wise scalar calls --


def test_vectorised_cohort_totals_matches_scalar_calls():
    names = ["milk_production_mass_cohort", "ch4_enteric", "co2_ration_fertilizer", "ration_intake",
             "nitrogen_intake", "co2_ration_fertilizer", "ch4_ration_rice"]
    types = ["Production", "Emissions", "Emissions", "Feed", "NitrogenBalance", "Feed", "Emissions"]
    value = [1000.0, 0.5, 33.4, 10.0, 0.2, 7.0, 0.3]
    stock = [50.0, 100.0, 100.0, 30.0, 25.0, 4.0, 12.5]
    intake = [10.0, 5.0, 12.5, 10.0, 8.0, 3.0, 6.5]
    duration = [365.0, 365.0, 365.0, 180.0, 365.0, 365.0, 90.0]
    feed = GLEAM_FEED_EMISSIONS_META
    vec = calc_cohort_totals(np.array(value), np.array(stock), np.array(intake), feed, np.array(duration),
                             np.array(names, dtype=object), np.array(types, dtype=object))
    ser = calc_cohort_totals(pd.Series(value), pd.Series(stock), pd.Series(intake), feed, pd.Series(duration),
                             pd.Series(names), pd.Series(types))
    rows = np.array([calc_cohort_totals(value[i], stock[i], intake[i], feed, duration[i], names[i], types[i])
                     for i in range(len(value))])
    assert_same_float(vec, rows)
    assert_same_float(ser, rows)
    # Feed variable named like a feed emission is not scaled by ration_intake
    assert rows[5] == 7.0 * 4.0 * 365.0
    assert rows[2] == 33.4 * 12.5 * 100.0 * 365.0 / 1000


def test_vectorised_allocated_emissions_and_co2eq_match_scalar_calls():
    value = [1000.0, 500.0, 200.0, 0.0, 12.5]
    share = [0.6, 0.4, 0.8, 1.0, 0.0]
    gas = ["CH4", "N2O", "CO2", "N2O", "CH4"]
    vec = calc_allocated_emissions(np.array(value), pd.Series(share))
    rows = np.array([calc_allocated_emissions(v, s) for v, s in zip(value, share)])
    assert_same_float(vec, rows)
    for gwp in ("AR6", "AR5_excluding_carbon_feedback", "AR5_including_carbon_feedback", "AR4"):
        out = calc_co2eq(pd.Series(gas), vec, gwp)
        rows_co2 = [calc_co2eq(g, v, gwp) for g, v in zip(gas, rows)]
        assert_same_float(out["value_co2eq"], [r["value_co2eq"] for r in rows_co2])
        assert_same_float(out["gwp"], [r["gwp"] for r in rows_co2])


# ---- validators look at distinct values only (R deduplicates / runs per value) --


def _ref_is_character(x):
    """Per-element reference implementation (the original row-by-row loop)."""
    from gleampy._utils import is_na

    if isinstance(x, str):
        return True
    if x is None or np.ndim(x) == 0 and not isinstance(x, np.ndarray):
        return False
    vals = x.to_numpy() if isinstance(x, (pd.Series, pd.Index)) else np.asarray(x)
    if vals.dtype.kind in "US":
        return True
    if vals.dtype == object:
        items = list(vals.ravel())
        if items and not any(isinstance(v, str) for v in items):
            return False
        return all(is_na(v) or isinstance(v, str) for v in items)
    return False


def _ref_is_numeric(x):
    from gleampy._utils import is_na

    if x is None or isinstance(x, (bool, np.bool_, str, bytes)):
        return False
    if isinstance(x, (pd.Series, pd.Index)):
        if x.dtype.kind in "iuf":
            return True
        if x.dtype.kind == "b":
            return False
    vals = x.to_numpy() if isinstance(x, (pd.Series, pd.Index)) else np.asarray(x)
    if vals.dtype.kind in "iuf":
        return True
    if vals.dtype == object:
        items = list(vals.ravel())
        if items and all(v is None or v is pd.NA for v in items):
            return False
        return all(
            is_na(v) or (isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)))
            for v in items
        )
    return False


def _ref_unique(values):
    from gleampy._utils import is_na

    out, seen, has_na = [], set(), False
    for v in values:
        if is_na(v):
            if not has_na:
                out.append(None)
                has_na = True
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _unhashable():
    arr = np.empty(3, dtype=object)
    arr[0], arr[1], arr[2] = [1], [2], None
    return arr


EDGE_VALUES = [
    np.array(["a", None, np.nan, pd.NA, "b", "a"], dtype=object),
    np.array([1, "a"], dtype=object),
    np.array([True, "a", 1], dtype=object),
    np.array([1, True, 1.0, None], dtype=object),
    np.array([True, False], dtype=object),
    np.array([None, None], dtype=object),
    np.array([None, pd.NA], dtype=object),
    np.array([np.nan, np.nan], dtype=object),
    np.array([pd.NaT, pd.NaT], dtype=object),
    np.array([], dtype=object),
    np.array(["a", np.str_("a"), "b"], dtype=object),
    np.array([1.5, np.float64(2), np.int64(3), None], dtype=object),
    _unhashable(),
    ["Production", "Feed", "Production"],
    "Production",
    np.array(["Feed", "Emissions"]),
    pd.Series(["Feed", None, "Feed"]),
    pd.Series(["Feed", None], dtype="string"),
    pd.Series(["Feed", "Other", "Feed"], dtype="category"),
    pd.Series([1, 2, 1], dtype="category"),
    pd.Series([1, None], dtype="Int64"),
    pd.Series([1.0, np.nan]),
    pd.Series([], dtype=object),
    pd.Series([None, None], dtype=object),
]


@pytest.mark.parametrize("x", EDGE_VALUES, ids=[str(i) for i in range(len(EDGE_VALUES))])
def test_type_predicates_and_unique_match_per_element_reference(x):
    from gleampy.validation.aggregation_core import _distinct, _unique, is_character, is_numeric

    assert is_character(x) == _ref_is_character(x)
    assert is_numeric(x) == _ref_is_numeric(x)
    if isinstance(x, np.ndarray) and x is EDGE_VALUES[12]:
        return  # unhashable values: R unique() is not defined for them
    all_values = np.atleast_1d(x.to_numpy() if isinstance(x, (pd.Series, pd.Index)) else np.asarray(x)).ravel()
    got, ref = _unique(_distinct(x)), _ref_unique(all_values)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g == r or (g is None and r is None)
        assert type(g) is type(r) or (isinstance(g, str) and isinstance(r, str))


def test_cohort_totals_validation_does_not_loop_over_rows(monkeypatch):
    """012/044: is.character / unique() checks look at distinct values only."""
    import gleampy.validation.aggregation_core as vc

    calls = {"n": 0}
    real_is_na = vc.is_na

    def counting_is_na(v):
        calls["n"] += 1
        return real_is_na(v)

    monkeypatch.setattr(vc, "is_na", counting_is_na)
    n = 20_000
    names = pd.Series(np.tile(np.array(["ch4_enteric", "milk_production_mass_cohort"], dtype=object), n), dtype=object)
    types = pd.Series(np.tile(np.array(["Emissions", "Production"], dtype=object), n), dtype=object)
    ones = pd.Series(np.ones(2 * n))
    calc_cohort_totals(ones, ones, ones, GLEAM_FEED_EMISSIONS_META, 365, names, types)
    calc_co2eq(pd.Series(np.tile(np.array(["CH4", "N2O"], dtype=object), n)), ones, "AR6")
    assert calls["n"] < 50


def test_run_aggregation_validation_works_on_distinct_herds(monkeypatch):
    """012: herd-id and (herd_id, species_short) checks deduplicate first, as R does."""
    import gleampy.validation.aggregation_run as vr

    calls = {"n": 0}
    real_norm = vr._norm

    def counting_norm(v):
        calls["n"] += 1
        return real_norm(v)

    monkeypatch.setattr(vr, "_norm", counting_norm)
    chrt, alloc = _aggregation_inputs()
    big_c = pd.concat([chrt] * 50, ignore_index=True)
    big_a = pd.concat([alloc] * 50, ignore_index=True)
    vr.validate_run_aggregation_module_inputs(big_c, big_a, 365, "AR6")
    n_herds = chrt["herd_id"].nunique()
    n_pairs = len(chrt[["herd_id", "species_short"]].drop_duplicates())
    # three _norm calls per distinct id in each setdiff; the old code made them per row
    assert calls["n"] <= 6 * (n_herds + n_pairs) < len(big_c)


def test_run_aggregation_validation_keeps_r_messages_with_mixed_ids():
    from gleampy.validation.aggregation_run import _distinct, _setdiff

    x = pd.Series([1, 2.0, None, np.nan, 3, 1.0, "h"], dtype=object)
    y = pd.Series([2, 3.0, "h", None], dtype=object)
    assert _setdiff(_distinct(x), _distinct(y)) == _setdiff(list(x), list(y)) == [1]
    assert _setdiff(_distinct(y), _distinct(x)) == _setdiff(list(y), list(x)) == []
    chrt, alloc = _aggregation_inputs()
    with pytest.raises(GleamValidationError, match=r"not found in `allocation_herd_long`: 3$"):
        run_aggregation_module(chrt, alloc[alloc["herd_id"] != 3], show_indicator=False)
    extra = pd.concat([alloc, alloc.iloc[[0]].assign(herd_id=99)], ignore_index=True)
    with pytest.raises(GleamValidationError, match=r"not found in `cohort_level_data`: 99$"):
        run_aggregation_module(chrt, extra, show_indicator=False)


# ---- herd totals from cohort-level group codes (087) ---------------------------


def _herd_long_inputs(herd_id, species):
    from gleampy.modules.aggregation import _melt

    n = len(herd_id)
    rng = np.random.default_rng(7)
    cohort = pd.DataFrame(
        {
            "herd_id": herd_id,
            "species_short": species,
            "cohort_short": ["FA"] * n,
            "cohort_stock_size": rng.uniform(1, 100, n),
            "ration_intake": rng.uniform(1, 10, n),
            "ch4_enteric": rng.uniform(0, 1, n),
            "milk_production_mass_cohort": rng.uniform(0, 1e4, n),
            "nitrogen_intake": rng.uniform(0, 1, n),
        }
    )
    measure = ["ch4_enteric", "milk_production_mass_cohort", "nitrogen_intake"]
    long = _melt(cohort, ["herd_id", "species_short", "cohort_short", "cohort_stock_size", "ration_intake"],
                 measure, "variable_name", "value")
    long["variable_type"] = np.repeat(np.array(["Emissions", "Production", "NitrogenBalance"], dtype=object), n)
    vals = rng.uniform(0, 1e6, len(long))
    vals[3] = np.nan
    long["value_total"] = vals
    return cohort, long, len(measure)


@pytest.mark.parametrize(
    "herd_id,species",
    [
        ([3, 1, 3, 2, 1, 2, 3], ["SHP", "CTL", "SHP", "PGS", "CTL", "PGS", "SHP"]),
        ([1, 1, 2, 2, 1], ["CTL", "BFL", "CTL", "CTL", "CTL"]),  # one herd, two species
        ([1.0, np.nan, 2.0, np.nan, 1.0], ["CTL", "CTL", None, "CTL", "CTL"]),  # NA keys
        (pd.Series([5, 4, 5], dtype="Int64"), pd.Series(["CTL", "GTS", "CTL"], dtype="category")),
        (["h2", "h1", "h2", "h1"], pd.Series(["CTL", "CML", "CTL", "CML"], dtype="string")),
    ],
    ids=["interleaved", "two-species", "na-keys", "nullable-categorical", "string-ids"],
)
def test_block_group_codes_match_generic_herd_aggregation(herd_id, species):
    from gleampy.core.allocation import calc_cohort_to_herd_aggregation
    from gleampy.modules.aggregation import _cohort_to_herd_long

    cohort, long, m = _herd_long_inputs(herd_id, species)
    expected = calc_cohort_to_herd_aggregation(
        long, ["herd_id", "species_short", "variable_type", "variable_name"], "value_total", "cohort_short"
    )
    got = _cohort_to_herd_long(cohort, long, m)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)


def test_run_aggregation_module_on_interleaved_herds_matches_contiguous_order():
    """Row order of the cohort table changes only the group order, not the totals."""
    chrt, alloc = _aggregation_inputs()
    shuffled = chrt.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a = run_aggregation_module(chrt, alloc, show_indicator=False)
    b = run_aggregation_module(shuffled, alloc, show_indicator=False)
    for key in ("results_feed", "results_production", "results_nitrogen", "results_emissions"):
        cols = [c for c in a[key].columns if a[key][c].dtype.kind != "f"]
        x = a[key].sort_values(cols, kind="mergesort").reset_index(drop=True)
        y = b[key].sort_values(cols, kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(x, y, check_exact=False, rtol=1e-12)


# ---- all-empty and missing columns ------------------------------------------------


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


_ALLOC_TSV = (
    "herd_id\tspecies_short\tvariable_name\tcommodity_name\tallocation_share\n"
    "1\tCTL\tch4_enteric\tMeat\t0.3\n1\tCTL\tch4_enteric\tMilk\t0.7\n"
    "1\tCTL\tco2_ration_fertilizer\tMeat\t0.3\n1\tCTL\tco2_ration_fertilizer\tMilk\t0.7\n"
    "2\tPGS\tch4_enteric\tMeat\t1\n2\tPGS\tco2_ration_fertilizer\tMeat\t1\n"
)
_COHORT_HEADER = (
    "herd_id\tspecies_short\tcohort_short\tcohort_stock_size\tration_intake\tch4_enteric\t"
    "co2_ration_fertilizer\tmilk_production_mass_cohort\n"
)


@pytest.mark.parametrize(
    "rows,message",
    [
        # R (fread + run_aggregation_module): "`ration_intake` must be numeric."
        ("1\tCTL\tFA\t100.5\t\t0.2\t30.1\t1234.5\n1\tCTL\tMJ\t20.25\t\t0.05\t22.2\t0\n"
         "2\tPGS\tFA\t30\t\t0.01\t11\t0\n2\tPGS\tMA\t7\t\t0.012\t12\t0\n", "`ration_intake` must be numeric."),
        ("1\tCTL\tFA\t100.5\tNA\t0.2\t30.1\t1234.5\n1\tCTL\tMJ\t20.25\tNA\t0.05\t22.2\t0\n"
         "2\tPGS\tFA\t30\tNA\t0.01\t11\t0\n2\tPGS\tMA\t7\tNA\t0.012\t12\t0\n", "`ration_intake` must be numeric."),
        # R: "`cohort_stock_size` must be numeric."
        ("1\tCTL\tFA\t\t10.5\t0.2\t30.1\t1234.5\n1\tCTL\tMJ\t\t5.25\t0.05\t22.2\t0\n"
         "2\tPGS\tFA\t\t2.5\t0.01\t11.1\t0\n2\tPGS\tMJ\t\t1.5\t0.005\t10\t0\n", "`cohort_stock_size` must be numeric."),
    ],
    ids=["ration-empty", "ration-NA", "stock-empty"],
)
def test_all_empty_numeric_id_column_is_rejected_like_r(tmp_path, rows, message):
    """043: fread types an all-empty column as logical, which calc_cohort_totals rejects."""
    from gleampy.io import read_csv

    chrt = read_csv(_write(tmp_path / "c.tsv", _COHORT_HEADER + rows))
    alloc = read_csv(_write(tmp_path / "a.tsv", _ALLOC_TSV))
    with pytest.raises(GleamValidationError, match=f"^{message}$"):
        run_aggregation_module(chrt, alloc, show_indicator=False)


def test_mostly_empty_ration_intake_gives_na_feed_totals_like_r(tmp_path):
    """With one non-missing value fread reads a double column: R returns NA, no error."""
    from gleampy.io import read_csv

    rows = ("1\tCTL\tFA\t100.5\t\t0.2\t30.1\t1234.5\n1\tCTL\tMJ\t20.25\t\t0.05\t22.2\t0\n"
            "2\tPGS\tFA\t30\t3.5\t0.01\t11\t0\n2\tPGS\tMA\t7\t\t0.012\t12\t0\n")
    chrt = read_csv(_write(tmp_path / "c.tsv", _COHORT_HEADER + rows))
    alloc = read_csv(_write(tmp_path / "a.tsv", _ALLOC_TSV))
    res = run_aggregation_module(chrt, alloc, show_indicator=False)
    em = res["results_emissions"]
    fert = em[em["variable_name"] == "co2_ration_fertilizer"]
    assert fert["value_total_gas"].isna().all()
    ch4 = em[(em["variable_name"] == "ch4_enteric") & (em["herd_id"] == 1)]
    assert ch4["value_total_gas"].tolist() == approx([7706.0625, 7706.0625])


def test_missing_ration_intake_column_is_a_validation_error():
    """R's melt() stops on the missing id column; Python raises GleamValidationError."""
    chrt, alloc = _aggregation_inputs()
    chrt = chrt.drop(columns="ration_intake")
    msg = r'^Missing required columns in `cohort_level_data`: "ration_intake"$'
    with pytest.raises(GleamValidationError, match=msg):
        run_aggregation_module(chrt, alloc, show_indicator=False)
    with pytest.warns(UserWarning, match="validation has been turned off"):
        with pytest.raises(GleamValidationError, match=msg):
            run_aggregation_module(chrt, alloc, show_indicator=False, validate_inputs=False)


def test_calc_co2eq_unknown_or_missing_gas_gives_na_gwp():
    from gleampy.validation._shared import validation_disabled

    with validation_disabled():
        out = calc_co2eq(["CH4", None, "XYZ", "N2O", "CH4"], [1.0, 2.0, 3.0, 4.0, 5.0], "AR4")
    np.testing.assert_array_equal(out["gwp"], [25.0, np.nan, np.nan, 298.0, 25.0])
    np.testing.assert_array_equal(out["value_co2eq"], [25.0, np.nan, np.nan, 1192.0, 125.0])
    assert calc_co2eq("N2O", 2.0, "AR6") == {"value_co2eq": 546.0, "gwp": 273.0}


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("value", [[365, 365], np.full(2992, 365.0), None], ids=["two", "per-long-row", "none"])
def test_run_aggregation_module_rejects_non_single_simulation_duration(value, validate):
    """R's validator (and, without it, its by-row assignments) reject a vector."""
    import warnings

    chrt, alloc = _aggregation_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(GleamValidationError, match=r"^`simulation_duration` must be a single numeric value\.$"):
            run_aggregation_module(chrt, alloc, simulation_duration=value, show_indicator=False,
                                   validate_inputs=validate)
