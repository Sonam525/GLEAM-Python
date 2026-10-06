"""Golden parity tests for the herd modules (demographic, non-demographic, all-herd).

Each case runs the module on the bundled example inputs exactly as
``tools/r_reference/generate_golden.R`` does and compares every output table
with the R golden output (rtol 1e-9, exact column and row order).
"""

from __future__ import annotations

import functools

import pandas as pd
import pytest
from golden_utils import assert_matches_golden

import gleam
from gleam.core import demographic_herd as dh
from gleam.io import load_example

TABLES = ("cohort_level_results", "herd_level_results")


def _demographic():
    return gleam.run_demographic_herd_module(
        cohort_level_data=load_example("herd_simulation_input_chrt_data.csv"),
        herd_level_data=load_example("herd_simulation_input_hrd_data.csv"),
        simulation_duration=365,
        show_indicator=False,
    )


def _nondemographic():
    return gleam.run_nondemographic_herd_module(
        cohort_level_data=load_example("nondemographic_herd_input_chrt_data.csv"),
        herd_level_data=load_example("nondemographic_herd_input_hrd_data.csv"),
        simulation_duration=365,
        show_indicator=False,
    )


def _all_herd(chrt: str, hrd: str, run_demographic: bool, run_nondemographic: bool):
    return gleam.run_all_herd_module(
        cohort_level_data=load_example(chrt),
        herd_level_data=load_example(hrd),
        run_demographic=run_demographic,
        run_nondemographic=run_nondemographic,
        show_indicator=False,
    )


CASES = {
    "demographic_herd_module": _demographic,
    "nondemographic_herd_module": _nondemographic,
    "all_herd_module_demographic": functools.partial(
        _all_herd, "herd_simulation_input_chrt_data.csv", "herd_simulation_input_hrd_data.csv", True, False
    ),
    "all_herd_module_nondemographic": functools.partial(
        _all_herd, "nondemographic_herd_input_chrt_data.csv", "nondemographic_herd_input_hrd_data.csv", False, True
    ),
    "all_herd_module_both": functools.partial(
        _all_herd, "herd_all_input_chrt_data.csv", "herd_all_input_hrd_data.csv", True, True
    ),
}


@functools.lru_cache(maxsize=None)
def _result(case: str):
    return CASES[case]()


@pytest.mark.parametrize("case", list(CASES))
def test_output_tables(case):
    assert list(_result(case)) == list(TABLES)


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("case", list(CASES))
def test_parity_with_r(case, table):
    assert_matches_golden(_result(case)[table], case, table)


@pytest.mark.parametrize("table", TABLES)
def test_parity_with_r_batch_steady_state_kernel(table, monkeypatch):
    """The numpy (many-herd) steady-state path reproduces R as well."""
    monkeypatch.setattr(dh, "SCALAR_STEADY_STATE_MAX_HERDS", 0)
    res = _demographic()
    assert_matches_golden(res[table], "demographic_herd_module", table)
    pd.testing.assert_frame_equal(res[table], _result("demographic_herd_module")[table], check_exact=True)


def test_inputs_are_not_modified():
    chrt = load_example("herd_all_input_chrt_data.csv")
    hrd = load_example("herd_all_input_hrd_data.csv")
    chrt0, hrd0 = chrt.copy(), hrd.copy()
    gleam.run_all_herd_module(chrt, hrd, show_indicator=False)
    pd.testing.assert_frame_equal(chrt, chrt0)
    pd.testing.assert_frame_equal(hrd, hrd0)


def test_all_herd_without_validation_matches_r():
    with pytest.warns(gleam.GleamWarning, match="Input validation has been turned off"):
        res = gleam.run_all_herd_module(
            load_example("herd_all_input_chrt_data.csv"), load_example("herd_all_input_hrd_data.csv"),
            show_indicator=False, validate_inputs=False,
        )
    for table in TABLES:
        assert_matches_golden(res[table], "all_herd_module_both", table)
