"""Pipeline switches and malformed inputs of run_gleam() / run_emissions_direct().

* The logical switches follow R's ``isTRUE()``: a ``numpy.bool_`` (e.g. from
  ``Series.any()`` or a numpy array) must behave exactly like a Python
  ``bool``, including the step that sets the non-demographic start weights
  from the weaning weight. ``validate_inputs``
  goes through ``setup_validation`` (R: "must be TRUE or FALSE"). With
  validation off the switches are not checked and a non-logical value counts
  as FALSE, as ``isTRUE()`` does.
* Malformed inputs that R rejects, sometimes with a plain "object ... not
  found" error, raise :class:`GleamValidationError` with R-style messages,
  never a bare ``KeyError``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import gleampy
from gleampy import GleamValidationError
from gleampy.io import example_path, read_csv
from gleampy.modules.gleam import build_herd_structure
from gleampy.validation.gleam_run import check_nondemographic_presence
from golden_utils import assert_matches_golden, golden_tables

NONDEMO_HERDS = (14, 15)

#: R 4.6 run_gleam() / run_emissions_direct() on herd 1 of the mixed example
#: with live_weight_male_nondemographic_start = 100 (weaning weight 250) and
#: run_demographic = run_nondemographic = TRUE: MN rows (phase 1, phase 2).
R_HERD1_MN = {
    "live_weight_cohort_initial": [250.0, 400.0],
    "live_weight_cohort_final": [400.0, 700.0],
    "ration_intake": [11.020967873393106, 16.352314559432667],
    "ch4_enteric": [0.23077598564901174, 0.34241289453409096],
}


def _run(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_gleam_examples"))


def _mod(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_modules_examples"))


def _sub(df: pd.DataFrame, ids) -> pd.DataFrame:
    if ids is None or "herd_id" not in df.columns:
        return df
    return df[df["herd_id"].isin(ids)].reset_index(drop=True)


def _mixed_ids() -> list:
    herd = _run("master_hrd_lvl_mixed_data.csv")
    return [h for h in herd["herd_id"] if h not in NONDEMO_HERDS]


def _args(ids, structure: bool = False, direct: bool = False) -> dict:
    """Inputs of one run on the bundled run_gleam examples, restricted to ``ids``."""
    if structure:
        cohort, herd = "master_chrt_lvl_structure_data.csv", "master_hrd_lvl_structure_data.csv"
    else:
        cohort, herd = "master_chrt_lvl_no_structure_mixed_data.csv", "master_hrd_lvl_mixed_data.csv"
    args = dict(
        has_herd_structure=structure,
        cohort_level_data=_sub(_run(cohort), ids),
        herd_level_data=_sub(_run(herd), ids),
        feed_rations=_sub(_run("feed_rations_share_chrt.csv"), ids),
        feed_params=_run("feed_quality.csv"),
        manure_management_system_fraction=_sub(_run("manure_management_system_fraction.csv"), ids),
        manure_management_system_factors=_sub(_run("manure_management_system_factors.csv"), ids),
        simulation_duration=365,
        show_indicator=False,
    )
    if not direct:
        args["feed_emissions"] = _run("feed_emission_factors.csv")
    return args


def _call(direct: bool, **args) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gleampy.GleamWarning)
        return (gleampy.run_emissions_direct if direct else gleampy.run_gleam)(**args)


def _flatten(result: dict, prefix: str = "") -> dict[str, pd.DataFrame]:
    out = {}
    for k, v in result.items():
        name = f"{prefix}__{k}" if prefix else k
        if isinstance(v, pd.DataFrame):
            out[name] = v
        elif isinstance(v, dict):
            out.update(_flatten(v, name))
        else:
            out[name] = v
    return out


def _assert_identical(a: dict, b: dict) -> None:
    fa, fb = _flatten(a), _flatten(b)
    assert sorted(fa) == sorted(fb)
    for name in fa:
        if isinstance(fa[name], pd.DataFrame):
            pd.testing.assert_frame_equal(fa[name], fb[name], check_exact=True, obj=name)
        else:
            assert fa[name] is None and fb[name] is None, name


FLAG_FACTORIES = {
    "numpy.True_": lambda cohort: np.True_,
    "Series.any()": lambda cohort: cohort["cohort_short"].isin(["FN", "MN"]).any(),
    "array element": lambda cohort: np.array([True, True])[0],
}


# ---- numpy.bool_ switches ----------------------------------------------------


@pytest.mark.parametrize("make_flag", FLAG_FACTORIES.values(), ids=FLAG_FACTORIES.keys())
def test_numpy_bool_flags_match_r_golden(make_flag):
    """run_gleam_mixed_no_structure with numpy flags reproduces every R table."""
    args = _args(_mixed_ids())
    flag = make_flag(args["cohort_level_data"])
    assert isinstance(flag, np.bool_)
    res = _call(False, run_demographic=flag, run_nondemographic=flag, **args)
    tables = _flatten(res)
    case = "run_gleam_mixed_no_structure"
    assert sorted(tables) == golden_tables(case)
    for table in golden_tables(case):
        assert_matches_golden(tables[table], case, table)


@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
def test_numpy_bool_flags_set_start_weight_from_weaning_weight(direct):
    """A start weight that differs from the weaning weight is overridden, as in R."""
    args = _args([1], direct=direct)
    herd = args["herd_level_data"].copy()
    herd["live_weight_male_nondemographic_start"] = 100.0  # weaning weight is 250
    args["herd_level_data"] = herd
    py_bool = _call(direct, run_demographic=True, run_nondemographic=True, **args)
    np_bool = _call(direct, run_demographic=np.True_, run_nondemographic=np.True_, **args)
    _assert_identical(py_bool, np_bool)
    for res in (py_bool, np_bool):
        hrd = res["herd_level_results"]
        assert hrd["live_weight_male_nondemographic_start"].tolist() == [250.0]
        assert np.isnan(hrd["live_weight_female_nondemographic_start"]).all()
        mn = res["cohort_level_results"]
        mn = mn[mn["cohort_short"] == "MN"]
        for col, expected in R_HERD1_MN.items():
            np.testing.assert_allclose(mn[col].to_numpy(dtype=float), expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
def test_numpy_bool_flags_accept_absent_optional_start_weights(direct):
    """R fills absent non-demographic start weights; numpy flags must too."""
    args = _args([1], direct=direct)
    args["herd_level_data"] = args["herd_level_data"].drop(
        columns=["live_weight_female_nondemographic_start", "live_weight_male_nondemographic_start"]
    )
    py_bool = _call(direct, run_demographic=True, run_nondemographic=True, **args)
    np_bool = _call(direct, run_demographic=np.True_, run_nondemographic=np.True_, **args)
    _assert_identical(py_bool, np_bool)


def test_numpy_bool_structure_and_factor_switches_match_python_bool():
    args = _args([1, 13], structure=True, direct=True)
    for factors_only in (False, True):
        py_bool = _call(True, emission_factors_only=factors_only, **args)
        np_args = dict(args, has_herd_structure=np.True_)
        np_bool = _call(True, emission_factors_only=np.bool_(factors_only), **np_args)
        _assert_identical(py_bool, np_bool)
    assert np_bool["aggregation_results"] is None

    gleam_args = _args([1, 13], structure=True)
    py_bool = _call(False, **gleam_args)
    np_bool = _call(False, **dict(gleam_args, has_herd_structure=np.True_))
    _assert_identical(py_bool, np_bool)
    with pytest.warns(gleampy.GleamWarning, match="validation has been turned off"):
        unchecked = gleampy.run_gleam(validate_inputs=np.False_, **gleam_args)
    _assert_identical(py_bool, unchecked)


def test_numpy_false_has_herd_structure_runs_herd_simulation():
    args = _args([1])
    py_bool = _call(False, **dict(args, has_herd_structure=False))
    np_bool = _call(False, **dict(args, has_herd_structure=np.False_))
    _assert_identical(py_bool, np_bool)


@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
@pytest.mark.parametrize("arg", ["has_herd_structure", "run_demographic", "run_nondemographic"])
@pytest.mark.parametrize(
    "bad", [1, 1.0, "TRUE", None, [True], np.array([True, True])],
    ids=["int", "float", "str", "None", "list", "array2"],
)
def test_non_logical_switches_are_rejected(direct, arg, bad):
    args = _args([1], direct=direct)
    args[arg] = bad
    with pytest.raises(GleamValidationError, match=f"`{arg}` must be a single logical value"):
        _call(direct, **args)


@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
def test_one_element_bool_array_switches_match_python_bool(direct):
    """A length-1 logical vector is a valid switch in R; is_true_scalar reads it as TRUE."""
    args = _args([1], direct=direct)
    expected = _call(direct, **args)
    flagged = _call(
        direct, **dict(args, has_herd_structure=np.array([False]),
                       run_demographic=np.array([True]), run_nondemographic=np.array([True]))
    )
    _assert_identical(expected, flagged)
    if direct:
        args = _args([1, 13], structure=True, direct=True)
        expected = _call(True, emission_factors_only=True, **args)
        flagged = _call(True, **dict(args, has_herd_structure=np.array([True]),
                                     emission_factors_only=np.array([True])))
        _assert_identical(expected, flagged)


def test_switches_use_istrue_semantics_without_validation():
    """With validation off, 1 is not a logical TRUE (R's isTRUE(1) is FALSE)."""
    args = _args([1])
    with pytest.raises(GleamValidationError, match="At least one of `run_demographic`"):
        _call(False, run_demographic=1, run_nondemographic=1, validate_inputs=False, **args)


@pytest.mark.parametrize(
    "direct,value", [(False, 1), (True, "yes")], ids=["run_gleam-int", "run_emissions_direct-str"]
)
def test_unchecked_non_logical_has_herd_structure_counts_as_false(direct, value):
    """Validation off: has_herd_structure=1 or "yes" is not TRUE, so the herd is simulated.

    (The port once read these with Python truthiness, i.e. as TRUE.)
    """
    args = _args([1], direct=direct)
    expected = _call(direct, **dict(args, has_herd_structure=False, validate_inputs=False))
    got = _call(direct, **dict(args, has_herd_structure=value, validate_inputs=False))
    _assert_identical(expected, got)


@pytest.mark.parametrize("value", [1, "TRUE"], ids=["int", "str"])
def test_unchecked_non_logical_emission_factors_only_counts_as_false(value):
    """Validation off: emission_factors_only=1 is not TRUE, so the full pipeline runs."""
    args = _args([1, 13], structure=True, direct=True)
    expected = _call(True, **dict(args, emission_factors_only=False, validate_inputs=False))
    got = _call(True, **dict(args, emission_factors_only=value, validate_inputs=False))
    _assert_identical(expected, got)
    assert got["aggregation_results"] is not None


@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
def test_validate_inputs_accepts_numpy_bool(direct):
    """numpy.True_ / numpy.False_ are valid validate_inputs values, like True / False."""
    args = _args([1], direct=direct)
    run = gleampy.run_emissions_direct if direct else gleampy.run_gleam
    checked = run(validate_inputs=True, **args)
    _assert_identical(checked, run(validate_inputs=np.True_, **args))
    with pytest.warns(gleampy.GleamWarning, match="validation has been turned off"):
        unchecked = run(validate_inputs=np.False_, **args)
    _assert_identical(checked, unchecked)


@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
@pytest.mark.parametrize("bad", [1, "TRUE", None], ids=["int", "str", "None"])
def test_non_logical_validate_inputs_is_rejected(direct, bad):
    """R's setup_validation(): "`validate_inputs` must be TRUE or FALSE."."""
    args = _args([1], direct=direct)
    with pytest.raises(GleamValidationError, match="`validate_inputs` must be TRUE or FALSE"):
        _call(direct, validate_inputs=bad, **args)


def test_build_herd_structure_numpy_true_structure_copies_inputs():
    args = _args([1, 13], structure=True)
    herd = args["herd_level_data"].copy()
    herd["live_weight_male_nondemographic_start"] = 100.0
    chrt, hrd = build_herd_structure(
        np.True_, np.True_, np.True_, args["cohort_level_data"], herd, 365, False,
    )
    pd.testing.assert_frame_equal(chrt, args["cohort_level_data"])
    # With a supplied herd structure the start weights are kept, as in R.
    assert (hrd["live_weight_male_nondemographic_start"] == 100.0).all()


# ---- live_weight_at_weaning read only for diverted herds ---------------------


WEANING_MSG = 'Missing required columns in `herd_level_data`: "live_weight_at_weaning"'


@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
def test_missing_weaning_weight_without_diverted_herds_is_a_validation_error(direct):
    """No herd diverts juveniles: R skips the `:=` and the weights validator reports the column."""
    args = _args([2], direct=direct)
    args["herd_level_data"] = args["herd_level_data"].drop(columns=["live_weight_at_weaning"])
    with pytest.raises(GleamValidationError, match=WEANING_MSG):
        _call(direct, **args)


def test_missing_weaning_weight_in_direct_example_is_a_validation_error():
    """Case 1a of run_emissions_direct (herd 1, all prop_nondemo_* = 0), as R."""
    herd = _sub(_mod("emissions_direct_input_hrd_data.csv"), [1])
    cohort = _sub(_mod("emissions_direct_input_chrt_no_structure_data.csv"), [1])
    feed = _sub(_mod("feed_rations_share_chrt.csv"), [1])
    feed = feed[feed["cohort_short"].isin(["FA", "FJ", "FS", "MA", "MJ", "MS"])].reset_index(drop=True)
    with pytest.raises(GleamValidationError, match=WEANING_MSG):
        _call(
            True, has_herd_structure=False, cohort_level_data=cohort,
            herd_level_data=herd.drop(columns=["live_weight_at_weaning"]),
            feed_rations=feed, feed_params=_mod("feed_quality.csv"),
            manure_management_system_fraction=_sub(_mod("manure_management_system_fraction.csv"), [1]),
            manure_management_system_factors=_sub(_mod("manure_management_system_factors.csv"), [1]),
            show_indicator=False,
        )


@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
def test_missing_weaning_weight_with_diverted_herd(direct):
    """Herd 9 diverts juveniles: R fails with "object 'live_weight_at_weaning' not found"."""
    args = _args([9], direct=direct)
    args["herd_level_data"] = args["herd_level_data"].drop(columns=["live_weight_at_weaning"])
    for validate in (True, False):
        with pytest.raises(GleamValidationError, match=WEANING_MSG):
            _call(direct, validate_inputs=validate, **args)


# ---- columns R reads without checking them -----------------------------------


@pytest.mark.parametrize(
    "structure,direct",
    [(True, False), (False, False), (True, True), (False, True)],
    ids=["gleam-structure", "gleam-no-structure", "direct-structure", "direct-no-structure"],
)
def test_chk_herds_require_is_egg_producing(structure, direct):
    """R: "object 'is_egg_producing' not found" in the energy step."""
    args = _args([1, 13], structure=structure, direct=direct)
    args["cohort_level_data"] = args["cohort_level_data"].drop(columns=["is_egg_producing"])
    with pytest.raises(
        GleamValidationError,
        match='Missing required columns in `cohort_level_data`: "is_egg_producing"',
    ):
        _call(direct, **args)


@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
def test_is_egg_producing_is_optional_without_chk_herds(direct):
    """R fills the column with NA when no herd is CHK, so the run succeeds."""
    args = _args([1], structure=True, direct=direct)
    with_flag = _call(direct, **args)
    args["cohort_level_data"] = args["cohort_level_data"].drop(columns=["is_egg_producing"])
    without_flag = _call(direct, **args)
    for table in ("results_emissions", "results_production"):
        if with_flag["aggregation_results"] is not None:
            pd.testing.assert_frame_equal(
                with_flag["aggregation_results"][table], without_flag["aggregation_results"][table]
            )


@pytest.mark.parametrize("ids", [[1, 13], [1]], ids=["CTL+CHK", "CTL"])
@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
def test_structure_requires_cohort_stock_size(ids, direct):
    """R: "object 'cohort_stock_size' not found" (CHK) or the nitrogen validator (CTL)."""
    args = _args(ids, structure=True, direct=direct)
    args["cohort_level_data"] = args["cohort_level_data"].drop(columns=["cohort_stock_size"])
    with pytest.raises(
        GleamValidationError,
        match='Missing required columns in `cohort_level_data`: "cohort_stock_size"',
    ):
        _call(direct, **args)


@pytest.mark.parametrize("structure", [False, True], ids=["no-structure", "structure"])
def test_herd_table_without_herd_id_is_a_validation_error(structure):
    """With the prop_nondemo_* columns R fails with "object 'herd_id' not found"."""
    args = _args([1, 13], structure=structure)
    args["herd_level_data"] = args["herd_level_data"].drop(columns=["herd_id"])
    with pytest.raises(
        GleamValidationError, match='Missing required columns in `herd_level_data`: "herd_id"'
    ):
        _call(False, **args)


def test_nondemographic_presence_handles_nullable_dtypes():
    cohort = pd.DataFrame({
        "herd_id": pd.array([1, 1, 2, 2, None], dtype="Int64"),
        "cohort_short": pd.array(["FA", "FN", "FA", "MJ", pd.NA], dtype="string"),
    })
    herd = pd.DataFrame({
        "herd_id": pd.array([1, 2], dtype="Int64"),
        "prop_nondemo_fem_juv": pd.array([0.5, None], dtype="Float64"),
        "prop_nondemo_mal_juv": pd.array([0.0, 0.0], dtype="Float64"),
    })
    check_nondemographic_presence(cohort, herd)
    herd["prop_nondemo_fem_juv"] = pd.array([0.5, 0.2], dtype="Float64")
    with pytest.raises(GleamValidationError, match='Missing "FN" rows .* `herd_id` 2'):
        check_nondemographic_presence(cohort, herd)
    cohort["cohort_short"] = cohort["cohort_short"].astype("category")
    with pytest.raises(GleamValidationError, match='Missing "FN" rows .* `herd_id` 2'):
        check_nondemographic_presence(cohort, herd)


# ---- no herd module ran ------------------------------------------------------


@pytest.mark.parametrize("validate", [True, False], ids=["validated", "unchecked"])
@pytest.mark.parametrize("direct", [False, True], ids=["run_gleam", "run_emissions_direct"])
def test_nondemographic_only_without_fn_mn_rows_names_the_cause(direct, validate):
    """R crashes inside data.table on the NULL herd table; the port names the cause."""
    args = _args([2], direct=direct)
    with pytest.raises(
        GleamValidationError,
        match=r"`run_demographic = FALSE` requires non-demographic \(FN/MN\) rows",
    ):
        _call(direct, run_demographic=False, run_nondemographic=True, validate_inputs=validate, **args)
