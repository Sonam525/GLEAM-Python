"""Golden parity tests for :func:`gleam.run_metabolic_energy_req_module`.

Runs the module on the bundled example inputs exactly as
``tools/r_reference/generate_golden.R`` does (case
``metabolic_energy_req_module``, table ``result``) and compares with the R
output (``rtol=1e-9``, exact column and row order). Also checks run-level
behaviour that mirrors R: no mutation of the inputs, lazily-needed optional
columns, run-level validation messages and the validation switch.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import gleam
from gleam.io import load_example
from gleam.validation import GleamValidationError, GleamWarning
from golden_utils import assert_matches_golden

CASE = "metabolic_energy_req_module"
COMPUTED = [
    "metabolic_energy_req_maintenance",
    "metabolic_energy_req_activity",
    "metabolic_energy_req_growth",
    "metabolic_energy_req_lactation",
    "metabolic_energy_req_work",
    "metabolic_energy_req_fibre_production",
    "metabolic_energy_req_egg_deposition",
    "metabolic_energy_req_pregnancy",
    "net_energy_maintenance_digestible_energy_ratio",
    "net_energy_growth_digestible_energy_ratio",
    "metabolic_energy_req_total",
    "ration_intake",
]


def _inputs():
    return (
        load_example("metabolic_energy_req_input_chrt_data.csv"),
        load_example("metabolic_energy_req_input_hrd_data.csv"),
    )


def _run(cohort, herd, **kw):
    return gleam.run_metabolic_energy_req_module(
        cohort_level_data=cohort, herd_level_data=herd, show_indicator=False, **kw
    )


def test_parity_metabolic_energy_req_module_result():
    cohort, herd = _inputs()
    result = _run(cohort, herd)
    assert_matches_golden(result, CASE, "result")


def test_output_columns_and_order():
    cohort, herd = _inputs()
    result = _run(cohort, herd)
    assert list(result.columns) == list(cohort.columns) + COMPUTED
    pd.testing.assert_frame_equal(result[list(cohort.columns)], cohort)


def test_inputs_are_not_mutated():
    cohort, herd = _inputs()
    cohort_before, herd_before = cohort.copy(deep=True), herd.copy(deep=True)
    _run(cohort.drop(columns="is_egg_producing").iloc[:-6], herd.iloc[:-1])
    _run(cohort, herd)
    pd.testing.assert_frame_equal(cohort, cohort_before)
    pd.testing.assert_frame_equal(herd, herd_before)


def test_parity_with_validation_disabled():
    cohort, herd = _inputs()
    with pytest.warns(GleamWarning, match="Input validation has been turned off"):
        result = _run(cohort, herd, validate_inputs=False)
    assert_matches_golden(result, CASE, "result")


def test_progress_indicator(capsys):
    cohort, herd = _inputs()
    gleam.run_metabolic_energy_req_module(cohort, herd, show_indicator=True)
    err = capsys.readouterr().err
    assert "Metabolic energy requirements calculation complete." in err


def test_is_egg_producing_added_when_absent_without_chickens():
    cohort, herd = _inputs()
    no_chk_cohort = cohort[cohort["herd_id"] != 13].drop(columns="is_egg_producing")
    no_chk_herd = herd[herd["herd_id"] != 13]
    full = _run(cohort, herd)
    for validate in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GleamWarning)
            result = _run(no_chk_cohort, no_chk_herd, validate_inputs=validate)
        expected_cols = [c for c in cohort.columns if c != "is_egg_producing"] + ["is_egg_producing"] + COMPUTED
        assert list(result.columns) == expected_cols
        assert result["is_egg_producing"].isna().all()
        np.testing.assert_array_equal(
            result[COMPUTED].to_numpy(), full.loc[full["herd_id"] != 13, COMPUTED].to_numpy()
        )


def test_optional_columns_only_needed_where_used():
    # R evaluates herd look-ups lazily: CHK-only columns (temperature, egg data)
    # and cohort_stock_size are not needed when no row uses them.
    cohort, herd = _inputs()
    keep = cohort["herd_id"] != 13
    c = cohort[keep].drop(columns=["cohort_stock_size", "nondemo_productive_phase_id", "is_egg_producing"])
    h = herd[herd["herd_id"] != 13].drop(
        columns=["average_annual_temperature", "egg_output_human_consumption", "egg_average_weight"]
    )
    result = _run(c, h)
    full = _run(cohort, herd)
    np.testing.assert_array_equal(result[COMPUTED].to_numpy(), full.loc[keep, COMPUTED].to_numpy())


def test_column_needed_by_some_row_must_be_present():
    cohort, herd = _inputs()
    # cohort_stock_size is used by the laying CHK row: R fails with "object not found".
    with pytest.raises(KeyError, match="cohort_stock_size"):
        _run(cohort.drop(columns="cohort_stock_size"), herd)
    # Pig herds need lactation_duration (not a required column when validation is off).
    with pytest.warns(GleamWarning), pytest.raises(KeyError, match="lactation_duration"):
        _run(cohort, herd.drop(columns="lactation_duration"), validate_inputs=False)


def test_run_validation_required_columns():
    cohort, herd = _inputs()
    with pytest.raises(GleamValidationError, match='Missing required columns in `cohort_level_data`: "daily_weight_gain"'):
        _run(cohort.drop(columns="daily_weight_gain"), herd)
    with pytest.raises(GleamValidationError, match='Missing required columns in `herd_level_data`: "litter_size"'):
        _run(cohort, herd.drop(columns="litter_size"))
    with pytest.raises(GleamValidationError, match="average_annual_temperature"):
        _run(cohort, herd.drop(columns="average_annual_temperature"))
    with pytest.raises(GleamValidationError, match='"egg_average_weight"'):
        _run(cohort, herd.drop(columns="egg_average_weight"))


def test_run_validation_structure():
    cohort, herd = _inputs()
    with pytest.raises(GleamValidationError, match="must contain at least one row"):
        _run(cohort.iloc[:0], herd)
    with pytest.raises(GleamValidationError, match="exactly 6 rows"):
        _run(cohort.drop(index=0), herd)
    with pytest.raises(GleamValidationError, match="Each herd_id must appear exactly once"):
        _run(cohort, pd.concat([herd, herd.iloc[[0]]], ignore_index=True))
    bad_species = herd.copy()
    bad_species.loc[0, "species_short"] = "XXX"
    with pytest.raises(GleamValidationError, match="Invalid `species_short` values"):
        _run(cohort, bad_species)
    with pytest.raises(GleamValidationError, match="not found in `cohort_level_data`"):
        _run(cohort[cohort["herd_id"] != 13], herd)


def test_run_validation_numeric_consistency():
    cohort, herd = _inputs()
    c = cohort.copy()
    c.loc[1, "high_activity_fraction"] = 0.99999
    c.loc[1, "low_activity_fraction"] = 0.5
    with pytest.raises(GleamValidationError, match=r'must be <= 1\. Violation\(s\): "1 / FJ"'):
        _run(c, herd)

    c = cohort.copy()
    c.loc[2, "live_weight_cohort_initial"] = 500.0
    with pytest.raises(GleamValidationError, match=r'must hold\. Violation\(s\): "1 / FS"'):
        _run(c, herd)

    h = herd.copy()
    h.loc[1, "live_weight_at_birth"] = 300.0
    with pytest.raises(GleamValidationError, match=r"Violation\(s\) for herd_id: 2"):
        _run(cohort, h)
