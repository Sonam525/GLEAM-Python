"""Golden parity and behaviour of ``run_emissions_manure_module``.

The golden case mirrors ``tools/r_reference/generate_golden.R`` (case
``emissions_manure_module``). The other tests check the vectorised run module
against a literal row-by-row transcription of the R algorithm, and the
run-level validation messages against R.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from golden_utils import assert_frame_matches, assert_matches_golden, load_golden

from gleampy import (
    GleamValidationError,
    GleamWarning,
    calc_ch4_manure,
    calc_n2o_manure_direct,
    calc_n2o_manure_leaching,
    calc_n2o_manure_total,
    calc_n2o_manure_volatilization,
    calc_volatile_solids,
    load_example,
    run_emissions_manure_module,
)
from gleampy._utils import is_na, merge_dt

PHASE = "nondemo_productive_phase_id"
MMS = "manure_management_system"

R_OUTPUT_COLUMNS = [
    "volatile_solids",
    "ch4_manure_pasture", "ch4_manure_burned", "ch4_manure_other", "ch4_manure_all_noburn",
    "n2o_manure_pasture_direct", "n2o_manure_burned_direct", "n2o_manure_other_direct",
    "n2o_manure_all_noburn_direct",
    "n2o_manure_pasture_vol", "n2o_manure_burned_vol", "n2o_manure_other_vol", "n2o_manure_all_noburn_vol",
    "n2o_manure_pasture_leach", "n2o_manure_burned_leach", "n2o_manure_other_leach",
    "n2o_manure_all_noburn_leach",
    "n2o_manure_pasture_indirect", "n2o_manure_burned_indirect", "n2o_manure_other_indirect",
    "n2o_manure_pasture_total", "n2o_manure_burned_total", "n2o_manure_other_total",
]


def _inputs():
    return (
        load_example("emissions_manure_input_chrt_data.csv"),
        load_example("manure_management_system_fraction.csv"),
        load_example("manure_management_system_factors.csv"),
    )


def _run(cohort, fraction, factors, **kwargs):
    return run_emissions_manure_module(
        cohort_level_data=cohort,
        manure_management_system_fraction=fraction,
        manure_management_system_factors=factors,
        show_indicator=False,
        **kwargs,
    )


# ---- golden parity ------------------------------------------------------------


def test_emissions_manure_module_matches_r():
    cohort, fraction, factors = _inputs()
    result = _run(cohort, fraction, factors)
    assert_matches_golden(result, "emissions_manure_module", "result")


def _pipeline_inputs(kind, keep):
    from gleampy.io import example_path, read_csv

    def rd(name):
        df = read_csv(example_path(name, kind))
        return df if keep is None else df[keep(df["herd_id"])].reset_index(drop=True)

    return rd("manure_management_system_fraction.csv"), rd("manure_management_system_factors.csv")


_DIRECT_HERDS = None


def _in_direct(h):
    global _DIRECT_HERDS
    if _DIRECT_HERDS is None:
        _DIRECT_HERDS = load_example("emissions_direct_input_hrd_data.csv")["herd_id"]
    return h.isin(_DIRECT_HERDS)


@pytest.mark.parametrize(
    "case, kind, keep",
    [
        ("run_gleam_mixed_no_structure", "run_gleam_examples", lambda h: ~h.isin([14, 15])),
        ("run_gleam_nondemo_only", "run_gleam_examples", lambda h: h.isin([14, 15])),
        ("run_gleam_structure_AR6", "run_gleam_examples", None),
        ("emissions_direct_1a_no_structure", "run_modules_examples", _in_direct),
        ("emissions_direct_2b_structure_rq", "run_modules_examples", _in_direct),
    ],
)
def test_manure_step_of_r_pipelines(case, kind, keep):
    # The R pipelines run this module on the cohort table built by the upstream
    # modules; the golden cohort table holds those columns before `volatile_solids`
    # and the manure outputs right after them.
    golden = load_golden(case, "cohort_level_results")
    upstream = list(golden.columns[: list(golden.columns).index("volatile_solids")])
    fraction, factors = _pipeline_inputs(kind, keep)
    result = _run(golden[upstream], fraction, factors)
    assert_frame_matches(result, golden[upstream + R_OUTPUT_COLUMNS], label=case)


# ---- row-by-row reference (literal transcription of the R loop) -----------------


def _reference_rowwise(cohort, fraction, factors):
    """R's ``by = .I`` algorithm: per-row selection of ``mms_data`` + scalar core calls."""
    use_phase = PHASE in cohort.columns and PHASE in fraction.columns
    mms_data = merge_dt(fraction, factors, by=["herd_id", MMS])
    out = cohort.copy()
    records = []
    for _, row in cohort.iterrows():
        sel = (mms_data["herd_id"] == row["herd_id"]) & (mms_data["cohort_short"] == row["cohort_short"])
        if use_phase:
            if is_na(row[PHASE]):
                sel &= mms_data[PHASE].isna()
            else:
                sel &= mms_data[PHASE] == row[PHASE]
        rows = mms_data[sel.to_numpy()]

        def build(fields, rows=rows):
            # split() by system name (sorted levels), first row per system
            return {
                name: {f: float(rows[rows[MMS] == name].iloc[0][f]) for f in fields}
                for name in sorted(rows[MMS].unique())
            }

        rec = {}
        rec["volatile_solids"] = vs = calc_volatile_solids(
            float(row["ration_intake"]), float(row["ration_digestibility_fraction"]),
            float(row["ration_urinary_energy_fraction"]), float(row["ration_ash"]),
        )
        n = float(row["nitrogen_excretion"])
        rec.update(calc_ch4_manure(volatile_solids=vs, **build(
            ["manure_management_system_fraction", "methane_conversion_factor_mcf", "ch4_max_producing_capacity_bo"]
        )))
        rec.update(calc_n2o_manure_direct(
            nitrogen_excretion=n, **build(["manure_management_system_fraction", "n2o_ef3"])
        ))
        rec.update(calc_n2o_manure_volatilization(
            nitrogen_excretion=n, **build(["manure_management_system_fraction", "n2o_ef4", "nitrogen_fracgas"])
        ))
        rec.update(calc_n2o_manure_leaching(
            nitrogen_excretion=n, **build(["manure_management_system_fraction", "n2o_ef5", "nitrogen_fracleach"])
        ))
        rec.update(calc_n2o_manure_total(**{
            k: rec[k] for k in [
                "n2o_manure_pasture_vol", "n2o_manure_pasture_leach", "n2o_manure_burned_vol",
                "n2o_manure_burned_leach", "n2o_manure_other_vol", "n2o_manure_other_leach",
                "n2o_manure_pasture_direct", "n2o_manure_burned_direct", "n2o_manure_other_direct",
            ]
        }))
        records.append(rec)
    for col in R_OUTPUT_COLUMNS:
        out[col] = [r[col] for r in records]
    return out


def _no_phase_inputs():
    """Inputs without ``nondemo_productive_phase_id`` (matching on herd and cohort only)."""
    cohort, fraction, factors = _inputs()
    keep_c = cohort[PHASE].isna() | (cohort[PHASE] == 1)
    keep_f = fraction[PHASE].isna() | (fraction[PHASE] == 1)
    cohort = cohort[keep_c].drop(columns=PHASE).reset_index(drop=True)
    fraction = fraction[keep_f].drop(columns=PHASE).reset_index(drop=True)
    # shuffle the lookup tables: the result must not depend on their row order
    fraction = fraction.sample(frac=1, random_state=3).reset_index(drop=True)
    factors = factors.sample(frac=1, random_state=4).reset_index(drop=True)
    return cohort, fraction, factors


@pytest.mark.parametrize("case", ["with_phase", "without_phase"])
def test_vectorised_run_matches_rowwise_r_algorithm(case):
    cohort, fraction, factors = _inputs() if case == "with_phase" else _no_phase_inputs()
    result = _run(cohort, fraction, factors)
    expected = _reference_rowwise(cohort, fraction, factors)
    assert list(result.columns) == list(cohort.columns) + R_OUTPUT_COLUMNS
    # identical operations in identical order: bit-identical results
    assert_frame_matches(result, expected, rtol=0, atol=0)


def test_reversed_cohort_row_order_is_kept():
    cohort, fraction, factors = _inputs()
    rev = cohort.iloc[::-1].reset_index(drop=True)
    result = _run(rev, fraction, factors)
    expected = _run(cohort, fraction, factors).iloc[::-1].reset_index(drop=True)
    assert_frame_matches(result, expected, rtol=0, atol=0)


def test_duplicate_fraction_rows_use_first_row_without_validation():
    # Without validation, R's split() + mms_df[1, ] keeps the first matching row per system.
    cohort, fraction, factors = _inputs()
    dup = fraction.iloc[[0]].copy()
    dup["manure_management_system_fraction"] = 0.99
    fraction2 = pd.concat([fraction, dup], ignore_index=True)
    with pytest.warns(GleamWarning, match="Input validation has been turned off"):
        result = _run(cohort, fraction2, factors, validate_inputs=False)
    expected = _run(cohort, fraction, factors)
    assert_frame_matches(result, expected, rtol=0, atol=0)
    with pytest.warns(GleamWarning):
        assert_frame_matches(
            _reference_rowwise(cohort, fraction2, factors),
            _run(cohort, fraction2, factors, validate_inputs=False), rtol=0, atol=0,
        )


def test_cohort_without_mms_rows_gets_zero_without_validation():
    cohort, fraction, factors = _inputs()
    fraction2 = fraction[~((fraction["herd_id"] == 1) & (fraction["cohort_short"] == "FA"))]
    with pytest.warns(GleamWarning):
        result = _run(cohort, fraction2, factors, validate_inputs=False)
    row = result[(result["herd_id"] == 1) & (result["cohort_short"] == "FA")].iloc[0]
    for col in R_OUTPUT_COLUMNS[1:]:
        assert row[col] == 0.0
    with pytest.raises(GleamValidationError, match="Missing herd/cohort combinations"):
        _run(cohort, fraction2, factors)


def test_inputs_not_modified_and_existing_columns_overwritten_in_place():
    cohort, fraction, factors = _inputs()
    cohort["ch4_manure_other"] = -1.0  # pre-existing output column keeps its position
    copies = [cohort.copy(), fraction.copy(), factors.copy()]
    result = _run(cohort, fraction, factors)
    for before, after in zip(copies, (cohort, fraction, factors)):
        pd.testing.assert_frame_equal(before, after)
    assert list(result.columns) == list(cohort.columns) + [c for c in R_OUTPUT_COLUMNS if c != "ch4_manure_other"]
    assert (result["ch4_manure_other"] > 0).all()


def test_validate_inputs_false_gives_same_result():
    cohort, fraction, factors = _inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        fast = _run(cohort, fraction, factors, validate_inputs=False)
    assert_frame_matches(fast, _run(cohort, fraction, factors), rtol=0, atol=0)


# ---- run-level validation (messages as produced by R, minus cli markup) ----------


def _drop(df, **eq):
    mask = np.ones(len(df), dtype=bool)
    for k, v in eq.items():
        mask &= (df[k] == v).to_numpy()
    return df[~mask].reset_index(drop=True)


def test_run_validation_messages_match_r():
    cohort, fraction, factors = _inputs()

    def err(c=cohort, f=fraction, a=factors):
        with pytest.raises(GleamValidationError) as exc:
            _run(c, f, a)
        return str(exc.value)

    assert err(c=pd.concat([cohort, cohort.iloc[[0]]], ignore_index=True)) == (
        "Duplicate herd/cohort rows in `cohort_level_data` for grouping columns "
        '"herd_id", "cohort_short", and "nondemo_productive_phase_id".'
    )
    assert err(f=pd.concat([fraction, fraction.iloc[[0]]], ignore_index=True)) == (
        "Duplicate herd/cohort/manure-management rows in `manure_management_system_fraction`."
    )
    f2 = fraction.copy()
    f2.loc[0, "manure_management_system_fraction"] = 0.5
    f2.loc[59, "manure_management_system_fraction"] = 0.5
    assert err(f=f2) == (
        "For each herd/cohort group, the sum of MMS fractions in `manure_management_system_fraction` "
        'must equal 1. Invalid groups: "1 / FA / NA"'
    )
    assert err(a=pd.concat([factors, factors.iloc[[0]]], ignore_index=True)) == (
        "Duplicate herd_id + manure_management_system rows in `manure_management_system_factors`."
    )
    assert err(a=_drop(factors, herd_id=3)) == (
        "Herd IDs in `manure_management_system_fraction` not found in "
        "`manure_management_system_factors`: 3"
    )
    assert err(a=_drop(factors, herd_id=2, manure_management_system="mms_burned")) == (
        "Some `manure_management_system` values in `manure_management_system_fraction` have no "
        "matching entry in `manure_management_system_factors`. Affected herd_ids and missing "
        'systems: "2: mms_burned"'
    )
    f3 = _drop(fraction, herd_id=2, cohort_short="FA", manure_management_system="mms_burned")
    sel = (f3["herd_id"] == 2) & (f3["cohort_short"] == "FA") & (f3[MMS] == "mms_drylot")
    burned = fraction[(fraction["herd_id"] == 2) & (fraction["cohort_short"] == "FA") & (fraction[MMS] == "mms_burned")]
    f3.loc[sel, "manure_management_system_fraction"] += burned["manure_management_system_fraction"].iloc[0]
    assert err(f=f3) == (
        "Within each herd_id, manure_management_system lists must be consistent across cohorts in "
        "`manure_management_system_fraction`. Inconsistent herds: 2"
    )
    assert err(f=_drop(fraction, herd_id=1, cohort_short="MN")) == (
        'Missing herd/cohort combinations in `manure_management_system_fraction`: "1 / MN / 1"'
    )
    assert err(c=_drop(cohort, herd_id=13)) == (
        "Herd IDs in `manure_management_system_fraction` not found in `cohort_level_data`: 13"
    )
    assert err(f=_drop(fraction, herd_id=13), a=_drop(factors, herd_id=13)) == (
        'Missing herd/cohort combinations in `manure_management_system_fraction`: "13 / FA / NA"'
    )
    assert err(c=_drop(cohort, herd_id=13), f=_drop(fraction, herd_id=13)) == (
        "Herd IDs in `manure_management_system_factors` not found in `cohort_level_data`: 13"
    )
    assert err(c=_drop(cohort, herd_id=13), a=_drop(factors, herd_id=13)) == (
        "Herd IDs in `manure_management_system_fraction` not found in `manure_management_system_factors`: 13"
    )
    c2 = cohort.copy()
    c2.loc[0, "ration_ash"] = np.nan
    assert err(c=c2) == (
        "`cohort_level_data` must not contain missing ration_urinary_energy_fraction or ration_ash."
    )
    c2 = cohort.copy()
    c2.loc[0, "herd_id"] = np.nan
    assert err(c=c2) == "`cohort_level_data` must not contain missing herd_id or cohort_short."
    c2 = cohort.copy()
    c2.loc[0, "cohort_short"] = "XX"
    assert err(c=c2).startswith('Invalid `cohort_short` values in `cohort_level_data`: "XX".')
    c2 = cohort.copy()
    c2.loc[1, "ration_ash"] = 0.5
    assert "`ration_ash`" in err(c=c2) and "= 0.5 is out of range" in err(c=c2)
    a2 = factors.copy()
    a2.loc[2, "methane_conversion_factor_mcf"] = 120
    assert err(a=a2) == (
        "`methane_conversion_factor_mcf` = 120 is out of range; expected value should be >= 0 and <= 100."
    )
    a2 = factors.copy()
    a2.loc[2, "n2o_ef3"] = np.nan
    assert err(a=a2) == "MMS values must not contain missing values."
    assert err(a=factors.drop(columns="n2o_ef4")) == (
        'Missing required columns in `manure_management_system_factors`: "n2o_ef4"'
    )
    assert err(c=cohort.iloc[0:0]) == "`cohort_level_data` must contain at least one row."


def test_fraction_sums_are_sequential_like_data_table_gsum():
    # data.table's grouped sum() adds a group's values one by one
    # in row order. For these fractions that sum is 1 + 1.19e-7 and R 4.6
    # reports the group, while a compensated sum (pandas' groupby().sum())
    # gives exactly 1.
    cohort, fraction, factors = _inputs()
    f2 = fraction.copy()
    rows = f2.index[(f2["herd_id"] == 1) & (f2["cohort_short"] == "FA")]
    assert len(rows) == 7
    f2.loc[rows, "manure_management_system_fraction"] = [0.32, 7e8, 0.19, 0.49, -7e8, 0.0, 0.0]
    with pytest.raises(GleamValidationError) as exc:
        _run(cohort, f2, factors)
    assert str(exc.value) == (
        "For each herd/cohort group, the sum of MMS fractions in `manure_management_system_fraction` "
        'must equal 1. Invalid groups: "1 / FA / NA"'
    )


# ---- key and value column dtypes ------------------------------------------------


def _numeric_outputs_equal(result, expected):
    for col in R_OUTPUT_COLUMNS:
        assert result[col].dtype == np.float64, col
        np.testing.assert_array_equal(result[col].to_numpy(), expected[col].to_numpy(), err_msg=col)


def _categorical(df, cols=("herd_id", "cohort_short", MMS, PHASE)):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


@pytest.mark.parametrize("subset", [False, True])
def test_categorical_key_columns_give_the_same_result(subset):
    # On pandas 2.x, groupby on a categorical key without
    # observed=True adds a group for every unobserved category combination,
    # which crashed the fraction-sum check. Groups must be those that occur,
    # as with data.table's `by` (and pandas 3).
    cohort, fraction, factors = _inputs()
    cats = [_categorical(df) for df in (cohort, fraction, factors)]
    if subset:  # unused categories remain after selecting herds

        def keep(df):
            return df[df["herd_id"].isin([1, 9])].reset_index(drop=True)

        cohort, fraction, factors = (keep(df) for df in (cohort, fraction, factors))
        cats = [keep(df) for df in cats]
        assert len(cats[1]["herd_id"].cat.categories) == 13
    _numeric_outputs_equal(_run(*cats), _run(cohort, fraction, factors))


def test_categorical_key_columns_keep_validation_messages():
    cohort, fraction, factors = _inputs()
    f2 = fraction.copy()
    f2.loc[0, "manure_management_system_fraction"] = 0.5
    with pytest.raises(GleamValidationError, match='Invalid groups: "1 / FA / NA"'):
        _run(_categorical(cohort), _categorical(f2), _categorical(factors))
    f3 = _drop(fraction, herd_id=1, cohort_short="MN")
    with pytest.raises(GleamValidationError, match='fraction`: "1 / MN / 1"'):
        _run(_categorical(cohort), _categorical(f3), _categorical(factors))
    a2 = _drop(factors, herd_id=2, manure_management_system="mms_burned")
    with pytest.raises(GleamValidationError, match='missing systems: "2: mms_burned"'):
        _run(_categorical(cohort), _categorical(fraction), _categorical(a2))


def test_group_codes_only_count_observed_combinations():
    from gleampy.validation.emissions_manure_run import _group_codes, _mms_sets_by

    df = pd.DataFrame({
        "herd_id": pd.Categorical([2, 2, 1, 2], categories=[1, 2, 3]),
        "cohort_short": pd.Categorical(["FA", "FJ", "FA", "FA"], categories=["FA", "FJ", "MA"]),
        MMS: ["b", "a", "a", "c"],
    })
    codes, n = _group_codes(df, ["herd_id", "cohort_short"])
    assert n == 3 and codes.tolist() == [0, 1, 2, 0]
    assert _mms_sets_by(df, ["herd_id", "cohort_short"]) == {
        (2, "FA"): ("b", "c"), (2, "FJ"): ("a",), (1, "FA"): ("a",),
    }
    assert _mms_sets_by(df, ["herd_id"]) == {(2,): ("a", "b", "c"), (1,): ("a",)}


def _object_floats(s):
    """Object column of Python floats, as pd.concat of an all-None and a float column gives."""
    return pd.Series([None if pd.isna(v) else float(v) for v in s], index=s.index, dtype=object)


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("col", [PHASE, "herd_id"])
@pytest.mark.parametrize("where", ["cohort", "fraction", "both"])
def test_object_dtype_numeric_keys_match_numeric_keys(where, col, validate):
    # Object-dtype key columns holding floats were compared as
    # "1.0" against "1": a false validation error, or zero manure emissions on
    # the FN/MN rows with validation off. R holds such a column as numeric.
    cohort, fraction, factors = _inputs()
    expected = _run(cohort, fraction, factors)
    c2, f2 = cohort.copy(), fraction.copy()
    if where in ("cohort", "both"):
        c2[col] = _object_floats(c2[col])
    if where in ("fraction", "both"):
        f2[col] = _object_floats(f2[col])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        result = _run(c2, f2, factors, validate_inputs=validate)
    _numeric_outputs_equal(result, expected)


def test_concatenated_phase_column_matches_r():
    # The realistic trigger of the case above: demographic rows with an all-None
    # phase concatenated with the non-demographic rows give an object column.
    cohort, fraction, factors = _inputs()
    demo = cohort[cohort[PHASE].isna()].copy()
    demo[PHASE] = None
    with warnings.catch_warnings():
        # pandas 2.x ignores the all-None column for the result dtype (with a
        # FutureWarning) and gives float64; pandas 3 gives an object column.
        warnings.simplefilter("ignore", FutureWarning)
        c2 = pd.concat([demo, cohort[cohort[PHASE].notna()]]).sort_index()
    c2[PHASE] = c2[PHASE].astype(object)
    assert {type(v) for v in c2[PHASE].dropna()} == {float}
    _numeric_outputs_equal(_run(c2, fraction, factors), _run(cohort, fraction, factors))


def _pipeline_tables(results, prefix=""):
    for key, value in results.items():
        if isinstance(value, dict):
            yield from _pipeline_tables(value, f"{prefix}{key}/")
        elif isinstance(value, pd.DataFrame):
            yield f"{prefix}{key}", value


def test_run_gleam_with_object_dtype_phase_column():
    # The manure step handles an object-dtype phase column (tests above), but
    # run_gleam first joins the ration summary on the cohort keys, where
    # pandas alone refuses to "merge on object and float64 columns". R holds
    # such a column as numeric and joins it like the float column.
    from gleampy import run_gleam
    from gleampy.io import example_path, read_csv

    def table(name):
        return read_csv(example_path(name, "run_gleam_examples"))

    cohort = table("master_chrt_lvl_structure_data.csv")
    args = dict(
        has_herd_structure=True,
        herd_level_data=table("master_hrd_lvl_mixed_data.csv"),
        feed_rations=table("feed_rations_share_chrt.csv"),
        feed_params=table("feed_quality.csv"),
        feed_emissions=table("feed_emission_factors.csv"),
        manure_management_system_fraction=table("manure_management_system_fraction.csv"),
        manure_management_system_factors=table("manure_management_system_factors.csv"),
        show_indicator=False,
    )
    expected = dict(_pipeline_tables(run_gleam(cohort_level_data=cohort, **args)))
    c2 = cohort.copy()
    c2[PHASE] = _object_floats(c2[PHASE])
    result = dict(_pipeline_tables(run_gleam(cohort_level_data=c2, **args)))
    assert list(result) == list(expected)
    for key, exp in expected.items():
        for col in exp.columns:
            if exp[col].dtype.kind == "f":
                np.testing.assert_array_equal(
                    result[key][col].to_numpy(dtype=float), exp[col].to_numpy(), err_msg=f"{key}: {col}"
                )


# ---- key codes with missing values (string / pyarrow backends) -------------------

try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

_STRING_KINDS = ["string[python]", "string[pyarrow]", "str[python]", "str[pyarrow]"]


def _string_dtype(kind):
    """StringDtype of ``kind``: ``string[...]`` has pd.NA, ``str[...]`` (pandas 3 default) NaN.

    Skips the test where pyarrow is not installed or this pandas has no such dtype.
    """
    storage = kind[kind.index("[") + 1 : -1]
    if storage == "pyarrow" and not HAS_PYARROW:
        pytest.skip("pyarrow is not installed")
    if kind.startswith("string"):
        return pd.StringDtype(storage)
    try:
        return pd.StringDtype(storage, na_value=np.nan)
    except TypeError:  # pandas < 2.3: no NaN-variant string dtype
        pytest.skip("this pandas has no StringDtype(na_value=np.nan)")


def _string_phase(s, dtype):
    """Phase ids written as text ("p1", "p2"), missing phases kept missing."""
    return pd.Series([None if pd.isna(v) else f"p{int(v)}" for v in s], index=s.index, dtype=dtype)


@pytest.mark.parametrize(
    "kind", _STRING_KINDS + ["object", "float64", "Float64", "Int64", "category"]
)
def test_key_codes_number_missing_values_where_they_first_appear(kind):
    # The string fast path of _codes() relied on where
    # pd.factorize(use_na_sentinel=False) puts NA, which is up to the array
    # backend (python or pyarrow). _first_rows() needs every value, NA
    # included, numbered in order of first appearance, like data.table's `by`.
    from gleampy.validation.emissions_manure_run import _codes, _first_rows

    if kind in _STRING_KINDS:
        s = pd.Series(["b", None, "a", None, "b"], dtype=_string_dtype(kind))
    elif kind in ("object", "category"):
        s = pd.Series(["b", None, "a", None, "b"], dtype=kind)
    else:
        s = pd.Series([2, None, 1, None, 2], dtype=kind)
    codes, n = _codes(s)
    assert codes.tolist() == [0, 1, 2, 1, 0] and n == 3
    assert _first_rows(codes, n).tolist() == [0, 1, 2]
    no_na, n_no_na = _codes(s.iloc[[0, 2, 4]])
    assert no_na.tolist() == [0, 1, 0] and n_no_na == 2
    first_na, n_first_na = _codes(s.iloc[[1, 0, 2, 3]])
    assert first_na.tolist() == [0, 1, 2, 0] and n_first_na == 3


def test_key_codes_do_not_depend_on_where_factorize_puts_missing_values(monkeypatch):
    # pandas documents the order of the non-missing uniques, not where NA goes
    # with use_na_sentinel=False. A backend numbering NA after all other
    # values made _first_rows() drop groups: with a text phase column, valid
    # inputs failed with an IndexError in the missing-combination check.
    from gleampy.validation.emissions_manure_run import _codes, _first_rows

    factorize = pd.factorize

    def na_last(values, *args, use_na_sentinel=True, **kwargs):
        codes, uniques = factorize(values, *args, **kwargs)
        if use_na_sentinel or not (codes < 0).any():
            return codes, uniques
        codes = np.where(codes < 0, len(uniques), codes)
        return codes, list(uniques) + [None]

    cohort, fraction, factors = _inputs()
    expected = _run(cohort, fraction, factors)
    dtype = pd.StringDtype("python")
    c2 = cohort.assign(**{PHASE: _string_phase(cohort[PHASE], dtype)})
    f2 = fraction.assign(**{PHASE: _string_phase(fraction[PHASE], dtype)})
    monkeypatch.setattr(pd, "factorize", na_last)
    for s in (
        pd.Series(["b", None, "a", None], dtype=dtype),
        pd.Series(["b", None, "a", None], dtype=object),
        pd.Series([2.0, np.nan, 1.0, np.nan]),
    ):
        codes, n = _codes(s)
        assert codes.tolist() == [0, 1, 2, 1] and n == 3
        assert _first_rows(codes, n).tolist() == [0, 1, 2]
    _numeric_outputs_equal(_run(c2, f2, factors), expected)
    with pytest.raises(GleamValidationError, match='fraction`: "1 / MN / p1"'):
        _run(c2, _drop(f2, herd_id=1, cohort_short="MN"), factors)


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("kind", _STRING_KINDS)
def test_text_phase_ids_match_r(kind, validate):
    # Checked against R 4.6: with phase ids "p1" / "p2" (NA for demographic
    # cohorts) in both cohort-level tables the results are identical() to
    # those with numeric ids, and a missing combination is reported as
    # "1 / MN / p1". Under pandas 3 with pyarrow installed, text columns read
    # from CSV are pyarrow-backed ("str[pyarrow]").
    cohort, fraction, factors = _inputs()
    expected = _run(cohort, fraction, factors)
    dtype = _string_dtype(kind)
    c2 = cohort.assign(**{PHASE: _string_phase(cohort[PHASE], dtype)})
    f2 = fraction.assign(**{PHASE: _string_phase(fraction[PHASE], dtype)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        _numeric_outputs_equal(_run(c2, f2, factors, validate_inputs=validate), expected)
    if validate:
        with pytest.raises(GleamValidationError, match='fraction`: "1 / MN / p1"'):
            _run(c2, _drop(f2, herd_id=1, cohort_short="MN"), factors)


def test_non_numeric_mms_columns_are_rejected_like_r():
    # R builds each MMS with unlist(mms_df[1, fields]); one
    # character column (even with text only in a row no cohort uses) makes it
    # a character vector and R aborts. Messages checked against R 4.6.
    cohort, fraction, factors = _inputs()

    def err(c=cohort, f=fraction, a=factors):
        with pytest.raises(GleamValidationError) as exc:
            _run(c, f, a)
        return str(exc.value)

    numeric_msg = "Each MMS argument must be a numeric vector."
    unused = factors.iloc[[0]].copy()
    unused[MMS] = "mms_unused"
    a2 = pd.concat([factors, unused], ignore_index=True)
    a2["n2o_ef5"] = a2["n2o_ef5"].astype(object)
    a2.loc[len(a2) - 1, "n2o_ef5"] = "n/a"
    assert err(a=a2) == numeric_msg
    a3 = factors.copy()
    a3["n2o_ef5"] = a3["n2o_ef5"].astype(str)  # numeric strings: still a character column
    assert err(a=a3) == numeric_msg
    # Earlier steps run first, as in R: an out-of-range MCF is reported before.
    a3["methane_conversion_factor_mcf"] = 500.0
    assert err(a=a3).startswith("`methane_conversion_factor_mcf` = 500 is out of range")
    # An all-missing (logical) column joins a numeric vector: missing values.
    a4 = factors.copy()
    a4["n2o_ef5"] = None
    assert err(a=a4) == "MMS values must not contain missing values."
    # A character fraction column fails data.table's grouped sum in R.
    f2 = fraction.copy()
    f2["manure_management_system_fraction"] = f2["manure_management_system_fraction"].astype(str)
    assert err(f=f2) == (
        "`manure_management_system_fraction` in `manure_management_system_fraction` must be numeric "
        "(R: Type 'character' is not supported by GForce sum (gsum))."
    )
    # A field in both tables: R's merge() renames it to .x / .y.
    f3 = fraction.copy()
    f3["n2o_ef3"] = 0.01
    assert err(f=f3).startswith("column not found: [n2o_ef3]")


def test_logical_and_integer_mms_columns_are_numeric():
    # A logical column joined with numeric ones is numeric in R (TRUE = 1).
    cohort, fraction, factors = _inputs()
    a_bool, a_float = factors.copy(), factors.copy()
    a_bool["nitrogen_fracleach"] = a_bool["nitrogen_fracleach"] > 0
    a_float["nitrogen_fracleach"] = (a_float["nitrogen_fracleach"] > 0).astype(float)
    _numeric_outputs_equal(_run(cohort, fraction, a_bool), _run(cohort, fraction, a_float))
    # Whole-number columns as int64 / nullable Int64 compute in float64 (no truncation).
    a_round = factors.copy()
    a_round["methane_conversion_factor_mcf"] = a_round["methane_conversion_factor_mcf"].round()
    expected = _run(cohort, fraction, a_round)
    for dtype in ("int64", "Int64"):
        a_int = a_round.copy()
        a_int["methane_conversion_factor_mcf"] = a_int["methane_conversion_factor_mcf"].astype(dtype)
        _numeric_outputs_equal(_run(cohort, fraction, a_int), expected)


@pytest.mark.parametrize(
    "placeholder",
    [0, pd.array([0] * 88, dtype="Int64"), pd.array([None] * 88, dtype="Int64"), False, None],
    ids=["int64", "Int64", "Int64-NA", "bool", "object-None"],
)
def test_existing_output_columns_of_any_dtype_are_replaced_by_float64(placeholder):
    # R coerces results assigned by group into an
    # existing integer column (truncation) or logical column (error); Python
    # always writes float64 results.
    cohort, fraction, factors = _inputs()
    assert len(cohort) == 88
    c2 = cohort.copy()
    for col in ("volatile_solids", "ch4_manure_burned", "n2o_manure_other_total"):
        c2[col] = placeholder
    result = _run(c2, fraction, factors)
    assert list(result.columns) == list(c2.columns) + [c for c in R_OUTPUT_COLUMNS if c not in c2.columns]
    _numeric_outputs_equal(result, _run(cohort, fraction, factors))


def test_numpy_bool_validate_inputs_switch():
    cohort, fraction, factors = _inputs()
    expected = _run(cohort, fraction, factors)
    _numeric_outputs_equal(_run(cohort, fraction, factors, validate_inputs=np.True_), expected)
    with pytest.warns(GleamWarning, match="Input validation has been turned off"):
        result = _run(cohort, fraction, factors, validate_inputs=np.False_)
    _numeric_outputs_equal(result, expected)


# ---- vectorised MMS consistency checks -------------------------------------------


def _reference_mms_problems(fraction, factors):
    """The two MMS checks computed from R's per-herd lists (plain pandas, slow)."""

    def sets(df, by):
        return {
            k if isinstance(k, tuple) else (k,): tuple(sorted(str(v) for v in g.unique()))
            for k, g in df.groupby(by, sort=False)[MMS]
        }

    frac, fact = sets(fraction, "herd_id"), sets(factors, "herd_id")
    without = any(set(v) - set(fact.get(k, ())) for k, v in frac.items())
    by_cohort = sets(fraction, ["herd_id", "cohort_short"])
    herd_lists = {}
    for (herd, _), v in by_cohort.items():
        herd_lists.setdefault(herd, set()).add(v)
    differ = any(len(v) > 1 for v in herd_lists.values())
    return without, differ, frac, by_cohort


@pytest.mark.parametrize("seed", range(40))
def test_vectorised_mms_checks_match_per_herd_lists(seed):
    from gleampy.validation.emissions_manure_run import (
        _mms_sets_by,
        _mms_sets_differ_within_herds,
        _mms_without_factors,
    )

    _, fraction, factors = _inputs()
    rng = np.random.default_rng(seed)
    f2, a2 = fraction.copy(), factors.copy()
    kind = seed % 4
    if kind == 0:  # one fraction row moved to a system without factors
        f2.loc[int(rng.integers(len(f2))), MMS] = "mms_new"
    elif kind == 1:  # a factor row dropped
        a2 = a2.drop(index=int(rng.integers(len(a2)))).reset_index(drop=True)
    elif kind == 2:  # a fraction row moved to a system that has factors
        i = int(rng.integers(len(f2)))
        f2.loc[i, MMS] = "mms_zzz"
        extra = a2[a2["herd_id"] == f2.loc[i, "herd_id"]].iloc[[0]].copy()
        extra[MMS] = "mms_zzz"
        a2 = pd.concat([a2, extra], ignore_index=True)
    else:  # a fraction row dropped
        f2 = f2.drop(index=int(rng.integers(len(f2)))).reset_index(drop=True)
    without, differ, frac_sets, cohort_sets = _reference_mms_problems(f2, a2)
    assert _mms_without_factors(f2, a2) == without
    assert _mms_sets_differ_within_herds(f2) == differ
    assert _mms_sets_by(f2, ["herd_id"]) == frac_sets
    assert _mms_sets_by(f2, ["herd_id", "cohort_short"]) == cohort_sets
