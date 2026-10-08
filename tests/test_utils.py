"""Tests of the shared helpers in gleampy._utils and gleampy.validation._shared.

Covers strict ``isTRUE`` (and ``%in% TRUE`` where R uses it), nullable
dtypes, NA herd ids, the linear cohort-completeness check, the egg-flag phase
check and warning attribution, plus the lookup / group-sum helpers.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

import gleampy
from gleampy import constants as K
from gleampy._utils import (
    Lookup,
    as_float,
    group_sum,
    in_true,
    is_true,
    is_true_scalar,
    lookup,
    lookup_columns,
    merge_dt,
)
from gleampy.core.nitrogen_balance import calc_nitrogen_retention
from gleampy.core.production import calc_egg_production
from gleampy.io import load_example
from gleampy.validation import GleamValidationError
from gleampy.validation._shared import (
    GleamWarning,
    check_cohort_completeness,
    check_herd_id_consistency,
    setup_validation,
    validate_is_egg_producing_flag,
    validation_disabled,
)

# --------------------------------------------------------------------------
# is_true: R isTRUE() semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 1.0, 2.5, np.int64(1), np.float64(1.0), "TRUE", "T", "true", None,
                                   np.nan, pd.NA, [True], "1"])
def test_is_true_rejects_non_logical_values(value):
    # R: isTRUE(1L), isTRUE(1), isTRUE("TRUE"), isTRUE(NA) are all FALSE
    if isinstance(value, list):
        assert is_true(value).tolist() == [True]  # element-wise on a logical vector
        return
    assert not bool(is_true(value))


@pytest.mark.parametrize("value", [True, np.True_, np.bool_(True)])
def test_is_true_accepts_logical_true(value):
    assert bool(is_true(value)) and is_true(value).shape == ()


def test_is_true_element_wise():
    obj = np.array([True, 1, "TRUE", None, np.True_, False, 1.0, pd.NA], dtype=object)
    assert is_true(obj).tolist() == [True, False, False, False, True, False, False, False]
    assert is_true([True, 1, False]).tolist() == [True, False, False]  # list keeps element types
    assert is_true(np.array([True, False])).tolist() == [True, False]
    assert is_true(np.array([1, 0, 2])).tolist() == [False, False, False]
    assert is_true(pd.Series([True, None, False], dtype=object)).tolist() == [True, False, False]
    assert is_true(pd.Series([True, None], dtype="boolean")).tolist() == [True, False]
    assert is_true(pd.Series([1.0, 0.0])).tolist() == [False, False]
    assert is_true(pd.Series(["TRUE", "FALSE"])).tolist() == [False, False]
    assert is_true(pd.Series([True, False]).astype("category")).tolist() == [True, False]
    assert is_true(np.array([[True, 1]], dtype=object)).shape == (1, 2)


def test_is_true_scalar():
    assert is_true_scalar(True) and is_true_scalar(np.True_) and is_true_scalar(np.array([True]))
    for v in (False, np.False_, 1, 1.0, "TRUE", None, np.nan, np.array([True, True]), [True]):
        assert not is_true_scalar(v)


def test_egg_outputs_need_logical_true_with_validation_off():
    """With validation off, a 0/1 or "TRUE" flag gives no eggs, as R's isTRUE gates do."""
    with validation_disabled():
        for flag in (1, 1.0, "TRUE"):
            out = calc_egg_production("CHK", "FA", 365, 0.06, 365, is_egg_producing=flag)
            assert out == {"egg_production_number_cohort": 0.0, "egg_production_mass_cohort": 0.0,
                           "egg_production_protein_cohort": 0.0}
            ret = calc_nitrogen_retention("CHK", "FA", daily_weight_gain=0.0, parturition_rate=0.0,
                                          cohort_stock_size=100, egg_output_human_consumption=30000,
                                          egg_average_weight=0.06, is_egg_producing=flag)
            assert ret == 0.0
        out = calc_egg_production("CHK", "FA", 365, 0.06, 365, is_egg_producing=np.True_)
        assert out["egg_production_number_cohort"] == pytest.approx(365.0)
        ret = calc_nitrogen_retention("CHK", "FA", daily_weight_gain=0.0, parturition_rate=0.0,
                                      cohort_stock_size=100, egg_output_human_consumption=30000,
                                      egg_average_weight=0.06, is_egg_producing=True)
        assert ret > 0


def test_numeric_egg_flag_rejected_with_validation_and_ignored_without():
    chrt = load_example("production_input_chrt_data.csv")
    hrd = load_example("production_input_hrd_data.csv")
    coded = chrt.copy()
    coded["is_egg_producing"] = [np.nan if v is None else float(v) for v in chrt["is_egg_producing"]]
    with pytest.raises(GleamValidationError, match="must be logical"):
        gleampy.run_production_module(coded, hrd, show_indicator=False)
    with pytest.warns(GleamWarning):
        out = gleampy.run_production_module(coded, hrd, show_indicator=False, validate_inputs=False)
    ref = gleampy.run_production_module(chrt, hrd, show_indicator=False)
    eggs = ref["egg_production_mass_cohort"].to_numpy() > 0
    assert eggs.any()
    assert (out["egg_production_mass_cohort"].to_numpy()[eggs] == 0).all()


# --------------------------------------------------------------------------
# in_true: R `x %in% TRUE`, used by the nitrogen / energy run validators
# --------------------------------------------------------------------------


def test_in_true_matches_r():
    # R 4.6: c(1L, 0L, NA, 2L) %in% TRUE -> TRUE FALSE FALSE FALSE
    assert in_true(pd.Series([1, 0, None, 2], dtype="Int64")).tolist() == [True, False, False, False]
    assert in_true(np.array([1, 0, 2])).tolist() == [True, False, False]
    # c(1, 1.0000001, NaN, 2, NA, 0) %in% TRUE -> TRUE FALSE FALSE FALSE FALSE FALSE
    floats = [1.0, 1.0000001, np.nan, 2.0, np.nan, 0.0]
    assert in_true(pd.Series(floats)).tolist() == [True] + [False] * 5
    assert in_true(np.array(floats)).tolist() == [True] + [False] * 5
    # c("TRUE", "T", "true", " TRUE", "1", NA, "FALSE") %in% TRUE -> only the first
    strings = ["TRUE", "T", "true", " TRUE", "1", None, "FALSE"]
    expected = [True] + [False] * 6
    assert in_true(pd.Series(strings, dtype=object)).tolist() == expected
    assert in_true(pd.Series(strings, dtype="string")).tolist() == expected
    assert in_true(strings).tolist() == expected
    # c(TRUE, NA, FALSE) %in% TRUE -> TRUE FALSE FALSE
    assert in_true(pd.Series([True, None, False], dtype=object)).tolist() == [True, False, False]
    assert in_true(pd.Series([True, None, False], dtype="boolean")).tolist() == [True, False, False]
    assert in_true(pd.Series([True, False])).tolist() == [True, False]
    # an object vector is typed like an R vector: c(TRUE, "TRUE", 1) is character
    assert in_true([True, "TRUE", 1]).tolist() == [True, True, False]
    assert in_true([True, 1.0, np.int64(1), 2]).tolist() == [True, True, True, False]
    assert in_true(pd.Series([1.0, 0.0]).astype("category")).tolist() == [True, False]
    assert in_true(pd.Series(pd.to_datetime(["2020-01-01"]))).tolist() == [False]
    for v in (True, np.True_, 1, 1.0, np.float64(1.0), "TRUE"):
        assert bool(in_true(v)) and in_true(v).shape == ()
    for v in (False, 0, 2.5, "T", "true", None, np.nan, pd.NA):
        assert not bool(in_true(v))


@pytest.mark.parametrize("module", ["nitrogen_balance", "metabolic_energy_req"])
def test_numeric_egg_flag_requires_the_egg_columns_like_r(module):
    """R's run validators test ``any(is_egg_producing %in% TRUE)``, which counts a 1.

    With the flag coded 1/0 and the egg herd columns dropped, R stops on the
    missing columns before it checks the flag type.
    """
    run = {"nitrogen_balance": gleampy.run_nitrogen_balance_module,
           "metabolic_energy_req": gleampy.run_metabolic_energy_req_module}[module]
    chrt = load_example(f"{module}_input_chrt_data.csv")
    hrd = load_example(f"{module}_input_hrd_data.csv")
    chrt["is_egg_producing"] = [np.nan if v is None else float(v) for v in chrt["is_egg_producing"]]
    hrd = hrd.drop(columns=["egg_average_weight", "egg_output_human_consumption"])
    with pytest.raises(GleamValidationError) as exc:
        run(chrt, hrd, show_indicator=False)
    msg = str(exc.value)
    assert msg.startswith("Missing required columns in `herd_level_data`:")
    assert '"egg_average_weight"' in msg and '"egg_output_human_consumption"' in msg


# --------------------------------------------------------------------------
# as_float and nullable / categorical dtypes
# --------------------------------------------------------------------------


NULLABLE = {
    "Int64": pd.Series([1, None, 3], dtype="Int64"),
    "Float64": pd.Series([1.5, None, 3.0], dtype="Float64"),
    "boolean": pd.Series([True, None, False], dtype="boolean"),
    "Index Int64": pd.Index([1, None, 3], dtype="Int64"),
    "categorical": pd.Series([1.0, np.nan, 3.0]).astype("category"),
    "object": pd.Series([1, None, 3.0], dtype=object),
}


@pytest.mark.parametrize("name", list(NULLABLE))
def test_as_float_nullable(name):
    out = as_float(NULLABLE[name])
    assert out.dtype == np.float64 and np.isnan(out[1])
    assert out[[0, 2]].tolist() == ([1.0, 0.0] if name == "boolean" else [1.0 if name != "Float64" else 1.5, 3.0])


def test_as_float_nullable_without_pandas_22_inference(monkeypatch):
    """pandas 2.0/2.1 had no float na_value inference in masked to_numpy (emulated here)."""
    masked = pytest.importorskip("pandas.core.arrays.masked")
    if not hasattr(masked, "to_numpy_dtype_inference"):
        pytest.skip("this pandas has no to_numpy_dtype_inference (pre-2.2 behaviour natively)")
    from pandas._libs import lib
    from pandas._libs import missing as libmissing

    def pre22(arr, dtype, na_value, hasna):
        if na_value is lib.no_default:
            na_value = libmissing.NA
        return (np.dtype(dtype) if dtype is not None else None), na_value

    monkeypatch.setattr(masked, "to_numpy_dtype_inference", pre22)
    for name in ("Int64", "Float64", "boolean", "Index Int64"):
        assert np.isnan(as_float(NULLABLE[name])[1])


def test_production_module_on_nullable_dtypes():
    chrt = load_example("production_input_chrt_data.csv")
    hrd = load_example("production_input_hrd_data.csv")
    ref = gleampy.run_production_module(chrt, hrd, show_indicator=False)
    out = gleampy.run_production_module(chrt.convert_dtypes(), hrd.convert_dtypes(), show_indicator=False)
    for col in ref.columns:
        if ref[col].dtype.kind == "f":
            np.testing.assert_array_equal(as_float(out[col]), ref[col].to_numpy())


# --------------------------------------------------------------------------
# lookup / lookup_columns / merge_dt keys
# --------------------------------------------------------------------------


def _frames():
    left = pd.DataFrame({"herd_id": [2, 1, 3, np.nan, 2], "x": range(5)})
    right = pd.DataFrame({
        "herd_id": [1.0, 2.0, np.nan, 2.0],
        "w": [10, 20, 30, 99],                       # int -> float64
        "flag": [True, False, True, True],           # bool -> object
        "name": ["a", "b", None, "z"],
        "nullable": pd.array([1, None, 3, 4], dtype="Int64"),
    })
    return left, right


def test_lookup_semantics():
    left, right = _frames()
    w = lookup(left, right, "w")
    assert w.dtype == np.float64
    np.testing.assert_array_equal(w, [20.0, 10.0, np.nan, 30.0, 20.0])  # NA matches NA, first dup wins
    assert lookup(left, right, "flag").tolist() == [False, True, None, True, False]
    assert lookup(left, right, "name").tolist()[:3] == ["b", "a", None]
    nb = lookup(left, right, "nullable")
    assert nb.dtype == np.float64 and np.isnan(nb[0]) and nb[1] == 1.0
    with pytest.raises(KeyError, match="'missing'"):
        lookup(left, right, "missing")


def test_lookup_columns_matches_lookup():
    left, right = _frames()
    cols = ["w", "flag", "name", "nullable"]
    many = lookup_columns(left, right, cols)
    assert list(many) == cols
    lk = Lookup(left, right)
    for c in cols:
        expected = lookup(left, right, c)
        for got in (many[c], lk(c)):
            assert got.dtype == expected.dtype
            assert pd.Series(got).equals(pd.Series(expected))
    with pytest.raises(KeyError, match="'nope'"):
        lookup_columns(left, right, ["w", "nope"])


def test_lookup_multi_key_and_nullable_or_categorical_keys():
    left = pd.DataFrame({"h": [1, 1, 2], "c": ["FA", "MA", "FA"]})
    right = pd.DataFrame({"h": [1.0, 2.0, 1.0], "c": ["MA", "FA", "FA"], "v": [5.0, 6.0, 7.0]})
    np.testing.assert_array_equal(lookup(left, right, "v", on=["h", "c"]), [7.0, 5.0, 6.0])
    for conv in (lambda s: s.astype("Int64"), lambda s: s.astype("category")):
        l2 = left.assign(h=conv(left["h"]))
        r2 = right.assign(h=conv(right["h"].astype("int64")))
        np.testing.assert_array_equal(lookup(l2, r2, "v", on=["h", "c"]), [7.0, 5.0, 6.0])


def test_merge_dt_nullable_and_categorical_keys():
    x = pd.DataFrame({"k": [2, 1, None], "a": [1, 2, 3]})
    y = pd.DataFrame({"k": [1.0, 2.0, np.nan], "b": [10, 20, 30]})
    ref = merge_dt(x, y, by="k")
    for xk in (x["k"].astype("Int64"), x["k"].astype("category")):
        out = merge_dt(x.assign(k=xk), y, by="k")
        assert out["a"].tolist() == ref["a"].tolist() and out["b"].tolist() == ref["b"].tolist()


# --------------------------------------------------------------------------
# group_sum: data.table gsum
# --------------------------------------------------------------------------


def test_group_sum_is_sequential_row_order():
    rng = np.random.default_rng(1)
    v = rng.random(5000) * 10.0 ** rng.integers(-8, 8, 5000)
    codes = rng.integers(0, 7, 5000)
    expected = np.zeros(7)
    for val, c in zip(v.tolist(), codes.tolist()):
        expected[c] += val
    out = group_sum(v, codes, 7)
    assert out.tobytes() == expected.tobytes()
    # differs from pairwise / compensated summation for at least one group on such data
    assert out[0] == expected[0]


def test_group_sum_missing_values():
    out = group_sum([1.0, np.nan, 2.0, 3.0], [0, 0, 1, 2], 4)
    assert np.isnan(out[0]) and out[1:].tolist() == [2.0, 3.0, 0.0]
    assert group_sum([1.0, np.nan, 2.0], [0, 0, 1], 2, na_rm=True).tolist() == [1.0, 2.0]
    assert group_sum([0.1, 0.2, 0.3], [0, 0, 0], 1)[0] == (0.1 + 0.2) + 0.3


# --------------------------------------------------------------------------
# check_cohort_completeness
# --------------------------------------------------------------------------


def _cohorts(herds: list, drop: str | None = None, dup: tuple[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for h in herds:
        cs = [c for c in K.GLEAM_COHORTS_DEMOGRAPHIC if c != drop]
        if dup is not None:
            cs = [dup[1] if c == dup[0] else c for c in cs]
        rows += [(h, c) for c in cs] + [(h, "FN")]
    return pd.DataFrame(rows, columns=["herd_id", "cohort_short"])


def test_cohort_completeness_messages():
    check_cohort_completeness(_cohorts([1, 2]))
    with pytest.raises(GleamValidationError,
                       match=r"exactly 6 rows in `cohort_level_data` .* herd_ids: 2, 3$"):
        check_cohort_completeness(pd.concat([_cohorts([1]), _cohorts([2, 3], drop="MA")]))
    # 6 rows but MS twice (MA missing): the second check, with the missing cohorts
    with pytest.raises(GleamValidationError,
                       match=r'Incomplete or duplicate cohorts found for herd_ids: "5 \(missing: MA\)", "4 \(missing: FJ\)"$'):
        check_cohort_completeness(pd.concat([_cohorts([1]), _cohorts([5], dup=("MA", "MS")),
                                             _cohorts([4], dup=("FJ", "FS"))]))


def test_cohort_completeness_na_and_categorical_herds():
    df = _cohorts([1.0, np.nan], drop="FA")
    with pytest.raises(GleamValidationError, match=r"herd_ids: 1.0, nan$"):
        check_cohort_completeness(df)
    cat = _cohorts([1, 2])
    cat["herd_id"] = pd.Categorical(cat["herd_id"], categories=[1, 2, 3])  # unobserved category 3
    check_cohort_completeness(cat)


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_cohort_completeness_is_linear(case):
    n = 32000
    if case == "missing":
        df = _cohorts(list(range(n)), drop="MA")
    else:
        df = _cohorts(list(range(n)), dup=("MA", "MS"))
    t0 = time.perf_counter()
    with pytest.raises(GleamValidationError):
        check_cohort_completeness(df)
    assert time.perf_counter() - t0 < 5.0  # was ~19 s (quadratic)


# --------------------------------------------------------------------------
# check_herd_id_consistency: NA matches NA like R's setdiff
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ids",
    [[1.0, np.nan], ["a", None], ["a", np.nan], pd.array([1, None], dtype="Int64")],
)
def test_herd_id_consistency_na_matches_na(ids):
    c = pd.DataFrame({"herd_id": pd.Series(ids).repeat(2).reset_index(drop=True)})
    h = pd.DataFrame({"herd_id": pd.Series(ids)})
    check_herd_id_consistency(c, h)


def test_herd_id_consistency_messages():
    c = pd.DataFrame({"herd_id": [1, 2, np.nan]})
    with pytest.raises(GleamValidationError, match=r"not found in `herd_level_data`: nan$"):
        check_herd_id_consistency(c, pd.DataFrame({"herd_id": [1.0, 2.0]}))
    with pytest.raises(GleamValidationError, match=r"Herd IDs in `herd_level_data` not found in `cohort_level_data`: 3$"):
        check_herd_id_consistency(pd.DataFrame({"herd_id": [1, 2]}), pd.DataFrame({"herd_id": [1, 2, 3]}))
    check_herd_id_consistency(pd.DataFrame({"herd_id": [1, 2]}), pd.DataFrame({"herd_id": [2.0, 1.0]}))


def test_allocation_with_na_herd_id_like_r():
    """R accepts an NA herd_id in run_allocation_module (allocation_share 0.984002 for CTL meat)."""
    ch = load_example("allocation_input_chrt_data.csv")
    hd = load_example("allocation_input_hrd_data.csv")
    ch = ch[ch.herd_id.isin([1, 2])].reset_index(drop=True)
    hd = hd[hd.herd_id.isin([1, 2])].reset_index(drop=True)
    for df in (ch, hd):
        df["herd_id"] = df["herd_id"].astype(float).where(df["herd_id"] != 2)
    out = gleampy.run_allocation_module(ch, hd, show_indicator=False)["allocation_long"]
    na_rows = out[out["herd_id"].isna() & (out["commodity_name"] == "Meat")]
    assert len(na_rows) and na_rows["allocation_share"].to_numpy() == pytest.approx(0.984002, rel=1e-6)


def _with_na_herd(chrt: pd.DataFrame, hrd: pd.DataFrame, rep: str, herd=2):
    """Copies of the tables with herd ``herd`` turned into an NA herd id of type ``rep``."""
    out = []
    for df in (chrt, hrd):
        df = df.copy()
        is_herd = (df["herd_id"] == herd).to_numpy()
        if rep == "float":
            df["herd_id"] = df["herd_id"].astype(float).where(~is_herd)
        elif rep == "Int64":
            df["herd_id"] = df["herd_id"].astype("Int64").where(~is_herd)
        else:
            df["herd_id"] = df["herd_id"].astype(str).astype(object).where(~is_herd, None)
        out.append(df)
    return out


@pytest.mark.parametrize("rep", ["float", "Int64", "str"])
@pytest.mark.parametrize("module", ["allocation", "metabolic_energy_req", "nitrogen_balance", "production",
                                    "weights"])
def test_na_herd_id_rows_like_r(module, rep):
    """R joins herd values with ``on = "herd_id"``, so an NA herd id that is in
    both tables gets exactly the results of the same herd with a real id
    (checked in R 4.6 for these five modules)."""
    run = getattr(gleampy, f"run_{module}_module")
    chrt = load_example(f"{module}_input_chrt_data.csv")
    hrd = load_example(f"{module}_input_hrd_data.csv")
    chrt = chrt[chrt["herd_id"].isin([1, 2])].reset_index(drop=True)
    hrd = hrd[hrd["herd_id"].isin([1, 2])].reset_index(drop=True)

    def result(c, h):
        r = run(c, h, show_indicator=False)
        return r if isinstance(r, pd.DataFrame) else r["allocation_long" if module == "allocation" else
                                                       "cohort_level_results"]

    ref = result(chrt, hrd)
    out = result(*_with_na_herd(chrt, hrd, rep))
    got = out[out["herd_id"].isna()].drop(columns="herd_id").reset_index(drop=True)
    exp = ref[ref["herd_id"] == 2].drop(columns="herd_id").reset_index(drop=True)
    assert len(got) == len(exp) > 0
    pd.testing.assert_frame_equal(got, exp, check_exact=True, check_dtype=False)


def test_lookup_na_matches_option():
    right = pd.DataFrame({"herd_id": [1.0, np.nan], "x": [10.0, 20.0]})
    left = pd.DataFrame({"herd_id": [np.nan, 1.0, 3.0]})
    # right[left, on = "herd_id"]: NA matches NA
    assert lookup(left, right, "x").tolist()[:2] == [20.0, 10.0]
    # right[herd_id == h]: NA == NA is NA and selects nothing
    got = lookup(left, right, "x", na_matches=False)
    assert np.isnan(got[0]) and got[1] == 10.0 and np.isnan(got[2])
    lk = Lookup(left, right, na_matches=False)
    assert lk.hit.tolist() == [False, True, False]
    assert np.isnan(lookup_columns(left, right, ["x"], na_matches=False)["x"][0])
    obj = pd.DataFrame({"herd_id": ["a", None], "c": ["FA", "FA"]})
    two = pd.DataFrame({"herd_id": ["a", None], "c": ["FA", "FA"], "x": [1.0, 2.0]})
    assert lookup(obj, two, "x", on=["herd_id", "c"]).tolist() == [1.0, 2.0]
    got = lookup(obj, two, "x", on=["herd_id", "c"], na_matches=False)
    assert got[0] == 1.0 and np.isnan(got[1])


@pytest.mark.parametrize("rep", ["float", "Int64", "str"])
def test_demographic_module_rejects_na_herd_id_like_r(rep):
    """R loops with ``herd_level_data[herd_id == current_herd_id]``: an NA herd
    selects no row and R stops with this message."""
    chrt = load_example("herd_simulation_input_chrt_data.csv")
    hrd = load_example("herd_simulation_input_hrd_data.csv")
    chrt = chrt[chrt["cohort_short"].isin(K.GLEAM_COHORTS_DEMOGRAPHIC) & chrt["herd_id"].isin([1, 2])]
    hrd = hrd[hrd["herd_id"].isin([1, 2])]
    c, h = _with_na_herd(chrt.reset_index(drop=True), hrd.reset_index(drop=True), rep)
    with pytest.raises(GleamValidationError, match="`parturition_rate` must be a single numeric value"):
        gleampy.run_demographic_herd_module(c, h, show_indicator=False)


def test_nondemographic_module_leaves_na_herd_rows_na_like_r():
    """R loops with ``cohort_level_results[herd_id == h]``: an NA herd is never
    simulated, its rows keep NA outputs and its total durations are 0."""
    chrt = load_example("nondemographic_herd_input_chrt_data.csv")
    hrd = load_example("nondemographic_herd_input_hrd_data.csv")
    ids = list(pd.unique(hrd["herd_id"]))[:2]
    chrt = chrt[chrt["herd_id"].isin(ids)].reset_index(drop=True)
    hrd = hrd[hrd["herd_id"].isin(ids)].reset_index(drop=True)
    c, h = _with_na_herd(chrt, hrd, "float", herd=ids[1])
    out = gleampy.run_nondemographic_herd_module(c, h, show_indicator=False)
    rows = out["cohort_level_results"]
    na_rows = rows[rows["herd_id"].isna()]
    assert len(na_rows) == 4
    for col in ("offtake_rate", "cohort_stock_size_unscaled", "partial_nondemo_phase_duration",
                "offtake_heads_unscaled", "offtake_heads_assessment_unscaled", "number_full_nondemo_cycles",
                "total_nondemo_cycle_starts_to_distribute"):
        assert na_rows[col].isna().all(), col
    assert na_rows["cohort_duration_days"].tolist() == [60.0, 110.0, 60.0, 100.0]
    herd = out["herd_level_results"]
    na_herd = herd[herd["herd_id"].isna()]
    assert na_herd["total_nondemo_fem_duration_days"].tolist() == [0.0]
    assert na_herd["total_nondemo_mal_duration_days"].tolist() == [0.0]
    # the other herd is unaffected (R: 19096.18 / 32319.24 head)
    real = rows[rows["herd_id"] == ids[0]]
    assert real["cohort_stock_size_unscaled"].to_numpy() == pytest.approx([19096.178323, 32319.236753])


# --------------------------------------------------------------------------
# validate_is_egg_producing_flag
# --------------------------------------------------------------------------


def test_egg_flag_phase_only_checked_on_laying_fn_rows():
    def obj(*v):
        return np.array(v, dtype=object)

    # a non-numeric phase on a cattle row is never looked at (R: row-wise early return)
    validate_is_egg_producing_flag(obj("CTL", "CHK"), obj("FA", "FN"), obj(None, True), obj("abc", 2))
    validate_is_egg_producing_flag(obj("CHK",), obj("FN",), obj(True,), obj("2",))  # "2" == 2 in R
    for bad in ("abc", "2.0", np.nan, None, 1, True):
        with pytest.raises(GleamValidationError, match="only when `nondemo_productive_phase_id` is 2"):
            validate_is_egg_producing_flag(obj("CHK",), obj("FN",), obj(True,), obj(bad,))
    validate_is_egg_producing_flag(
        pd.Series(["CHK"]), pd.Series(["FN"]), pd.Series([True], dtype="boolean"), pd.Series([2], dtype="Int64"))


def test_mer_run_with_non_numeric_phase_on_cattle_row():
    chrt = load_example("metabolic_energy_req_input_chrt_data.csv")
    hrd = load_example("metabolic_energy_req_input_hrd_data.csv")
    chrt["nondemo_productive_phase_id"] = chrt["nondemo_productive_phase_id"].astype(object)
    assert chrt.loc[0, "species_short"] == "CTL"
    chrt.loc[0, "nondemo_productive_phase_id"] = "abc"
    out = gleampy.run_metabolic_energy_req_module(chrt, hrd, show_indicator=False)
    with pytest.warns(GleamWarning):
        ref = gleampy.run_metabolic_energy_req_module(chrt, hrd, show_indicator=False, validate_inputs=False)
    pd.testing.assert_frame_equal(out, ref)


# --------------------------------------------------------------------------
# Warnings are attributed to the caller
# --------------------------------------------------------------------------


def test_validation_off_warning_points_at_caller():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with setup_validation(False):
            pass
    assert len(rec) == 1 and rec[0].filename == __file__


def test_module_warning_points_at_caller():
    chrt = load_example("weights_input_chrt_data.csv")
    hrd = load_example("weights_input_hrd_data.csv")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gleampy.run_weights_module(chrt, hrd, show_indicator=False, validate_inputs=False)
    assert [r.filename for r in rec if issubclass(r.category, GleamWarning)] == [__file__]


def test_validator_warning_points_at_caller():
    from gleampy.validation.all_herd_run import validate_run_all_herd_module_inputs

    cohort = load_example("herd_all_input_chrt_data.csv")
    herd = load_example("herd_all_input_hrd_data.csv")
    demo = cohort[cohort["cohort_short"].isin(K.GLEAM_COHORTS_DEMOGRAPHIC)].reset_index(drop=True)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            validate_run_all_herd_module_inputs(demo, herd, True, True)
        except GleamValidationError:
            pass
    msgs = [r for r in rec if "no non-demographic rows" in str(r.message)]
    assert msgs and all(r.filename == __file__ for r in msgs)


def test_warning_module_filter_works():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=GleamWarning, module=__name__)
        with setup_validation(False):
            pass
    assert not [r for r in rec if issubclass(r.category, GleamWarning)]
