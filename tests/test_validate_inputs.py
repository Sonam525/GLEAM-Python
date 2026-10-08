"""The ``validate_inputs`` switch of every ``run_*`` function.

Port of the five test_that blocks that ``feature/optional-validation-rule``
(commit 6a66d86) adds to tests/testthat/test-run_gleam.R, extended to every
exported ``run_*`` function:

* only ``TRUE`` / ``FALSE`` are accepted (numpy booleans too, like R's
  logical scalars);
* an unchecked run consults no validator: the rule table is replaced by an
  empty one (R mocks ``parameter_ranges`` with ``parameter_ranges[0L]``), so
  any range lookup would fail, yet every output table equals the checked run
  and exactly one warning is issued;
* the switch state is restored after nested runs and after errors, both in
  the pipeline and in standalone modules.
"""

from __future__ import annotations

import importlib
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import gleampy
from gleampy import GleamValidationError, GleamWarning
from gleampy.io import example_path, read_csv
from gleampy.validation import _shared
from test_run_gleam import gleam_test_data, run_gleam_default

RUN_FUNCTIONS = sorted(n for n in gleampy.__all__ if n.startswith("run_"))
OFF_WARNING = "Input validation has been turned off"


def _mod(name: str) -> pd.DataFrame:
    return read_csv(example_path(name, "run_modules_examples"))


def _state() -> tuple[bool, bool]:
    """R: ``options("gleam.validate", "gleam.validation_active")``."""
    return _shared._VALIDATE.get(), _shared._ACTIVE.get()


@pytest.fixture(autouse=True)
def _state_is_restored():
    before = _state()
    assert before == (True, False)
    yield
    assert _state() == before


@pytest.fixture(scope="module")
def d_gleam() -> dict:
    return gleam_test_data()


@pytest.fixture
def no_rules(monkeypatch):
    """Empty rule table: any ``parameter_ranges`` lookup fails, as with R's mock."""
    empty = _shared.parameter_ranges().iloc[0:0].copy()
    monkeypatch.setattr(_shared, "parameter_ranges", lambda: empty)
    for name, obj in list(vars(_shared).items()):
        if callable(obj) and hasattr(obj, "cache_clear") and hasattr(obj, "__wrapped__"):
            monkeypatch.setattr(_shared, name, obj.__wrapped__)  # uncached while mocked


def _count_off_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    gleam_warnings = [w for w in caught if issubclass(w.category, GleamWarning)]
    return result, gleam_warnings


def assert_same(actual, expected, label: str = "result") -> None:
    """Identical results: same tables (exactly), same nesting."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and list(actual) == list(expected), label
        for key in expected:
            assert_same(actual[key], expected[key], f"{label}[{key!r}]")
    elif expected is None:
        assert actual is None, label
    else:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True, obj=label)


# ---- all run functions accept only TRUE or FALSE for validate_inputs ---------


@pytest.mark.parametrize("function_name", RUN_FUNCTIONS)
@pytest.mark.parametrize(
    "flag",
    [np.nan, None, pd.NA, [], [True, False], (False,), 0, 1, 0.0, "FALSE", "TRUE"],
    ids=["NA", "NULL", "pd.NA", "logical()", "c(TRUE,FALSE)", "tuple", "0", "1", "0.0", "'FALSE'", "'TRUE'"],
)
def test_run_functions_accept_only_true_or_false(function_name, flag):
    fn = getattr(gleampy, function_name)
    required = {
        name: None for name, p in inspect.signature(fn).parameters.items()
        if p.default is inspect.Parameter.empty
    }
    with pytest.raises(GleamValidationError, match="validate_inputs.*must be TRUE or FALSE"):
        fn(**required, validate_inputs=flag)


def test_run_functions_are_all_covered():
    assert len(RUN_FUNCTIONS) == 15
    assert set(RUN_FUNCTIONS) == set(MODULE_CALLS) | {"run_gleam", "run_emissions_direct"}


# ---- unchecked runs skip validators and preserve results ----------------------


@pytest.mark.parametrize("has_structure", [False, True])
@pytest.mark.parametrize("off", [False, np.False_], ids=["False", "np.False_"])
def test_unchecked_pipeline_skips_validators_and_preserves_results(d_gleam, has_structure, off, request):
    expected = run_gleam_default(d_gleam, has_herd_structure=has_structure)
    request.getfixturevalue("no_rules")
    result, caught = _count_off_warnings(
        run_gleam_default, d_gleam, has_herd_structure=has_structure, validate_inputs=off
    )
    assert len(caught) == 1 and OFF_WARNING in str(caught[0].message)
    assert_same(result, expected)
    assert _state() == (True, False)


def test_empty_rule_table_breaks_checked_pipeline(d_gleam, no_rules):
    with warnings.catch_warnings():
        warnings.simplefilter("error", GleamWarning)
        with pytest.raises(GleamValidationError, match="expected exactly one rule"):
            run_gleam_default(d_gleam, validate_inputs=True)


def _direct_args(has_structure: bool) -> dict:
    herd = _mod("emissions_direct_input_hrd_data.csv")

    def in_direct(df):
        return df[df["herd_id"].isin(herd["herd_id"])].reset_index(drop=True)

    cohort = "structure" if has_structure else "no_structure"
    return dict(
        has_herd_structure=has_structure,
        cohort_level_data=_mod(f"emissions_direct_input_chrt_{cohort}_data.csv"),
        herd_level_data=herd,
        feed_rations=in_direct(_mod("feed_rations_share_chrt.csv")),
        feed_params=_mod("feed_quality.csv"),
        manure_management_system_fraction=in_direct(_mod("manure_management_system_fraction.csv")),
        manure_management_system_factors=in_direct(_mod("manure_management_system_factors.csv")),
        run_nondemographic=False,
        show_indicator=False,
    )


@pytest.mark.parametrize("has_structure", [False, True])
@pytest.mark.parametrize("factors_only", [False, True])
def test_unchecked_direct_emissions_skip_validators(has_structure, factors_only, request):
    args = _direct_args(has_structure)
    expected = gleampy.run_emissions_direct(**args, emission_factors_only=factors_only)
    request.getfixturevalue("no_rules")
    result, caught = _count_off_warnings(
        gleampy.run_emissions_direct, **args, emission_factors_only=factors_only, validate_inputs=False
    )
    assert len(caught) == 1 and OFF_WARNING in str(caught[0].message)
    assert_same(result, expected)


# Every standalone module on its R example inputs (tools/r_reference/generate_golden.R).
MODULE_CALLS = {
    "run_weights_module": lambda: dict(
        cohort_level_data=_mod("weights_input_chrt_data.csv"), herd_level_data=_mod("weights_input_hrd_data.csv"),
    ),
    "run_ration_quality_module": lambda: dict(
        rations_share=_mod("feed_rations_share_chrt.csv"), feed_params=_mod("feed_quality.csv"),
    ),
    "run_emissions_ration_module": lambda: dict(
        rations_share=_mod("feed_rations_share_chrt.csv"), feed_emissions=_mod("feed_emission_factors.csv"),
    ),
    "run_metabolic_energy_req_module": lambda: dict(
        cohort_level_data=_mod("metabolic_energy_req_input_chrt_data.csv"),
        herd_level_data=_mod("metabolic_energy_req_input_hrd_data.csv"),
    ),
    "run_emissions_enteric_module": lambda: dict(cohort_level_data=_mod("emissions_enteric_input_chrt_data.csv")),
    "run_nitrogen_balance_module": lambda: dict(
        cohort_level_data=_mod("nitrogen_balance_input_chrt_data.csv"),
        herd_level_data=_mod("nitrogen_balance_input_hrd_data.csv"),
    ),
    "run_emissions_manure_module": lambda: dict(
        cohort_level_data=_mod("emissions_manure_input_chrt_data.csv"),
        manure_management_system_fraction=_mod("manure_management_system_fraction.csv"),
        manure_management_system_factors=_mod("manure_management_system_factors.csv"),
    ),
    "run_production_module": lambda: dict(
        cohort_level_data=_mod("production_input_chrt_data.csv"),
        herd_level_data=_mod("production_input_hrd_data.csv"), simulation_duration=365,
    ),
    "run_allocation_module": lambda: dict(
        cohort_level_data=_mod("allocation_input_chrt_data.csv"),
        herd_level_data=_mod("allocation_input_hrd_data.csv"),
    ),
    "run_aggregation_module": lambda: dict(
        cohort_level_data=_mod("aggregation_input_chrt_data.csv"),
        allocation_herd_long=_mod("aggregation_allocation_input_data.csv"),
        simulation_duration=365, global_warming_potential_set="AR6",
    ),
    "run_demographic_herd_module": lambda: dict(
        cohort_level_data=_mod("herd_simulation_input_chrt_data.csv"),
        herd_level_data=_mod("herd_simulation_input_hrd_data.csv"), simulation_duration=365,
    ),
    "run_nondemographic_herd_module": lambda: dict(
        cohort_level_data=_mod("nondemographic_herd_input_chrt_data.csv"),
        herd_level_data=_mod("nondemographic_herd_input_hrd_data.csv"), simulation_duration=365,
    ),
    "run_all_herd_module": lambda: dict(
        cohort_level_data=_mod("herd_all_input_chrt_data.csv"),
        herd_level_data=_mod("herd_all_input_hrd_data.csv"),
        run_demographic=True, run_nondemographic=True,
    ),
}


@pytest.mark.parametrize("function_name", sorted(MODULE_CALLS))
def test_unchecked_modules_skip_validators_and_preserve_results(function_name, request):
    fn = getattr(gleampy, function_name)
    args = MODULE_CALLS[function_name]()
    expected = fn(**args, show_indicator=False)
    request.getfixturevalue("no_rules")
    result, caught = _count_off_warnings(fn, **args, show_indicator=False, validate_inputs=False)
    assert len(caught) == 1 and OFF_WARNING in str(caught[0].message)
    assert_same(result, expected)
    assert _state() == (True, False)


# ---- validation is restored after nested runs and calculation errors ---------


def test_validation_restored_after_nested_runs_and_calculation_errors(d_gleam, monkeypatch):
    pipeline = importlib.import_module("gleampy.modules.gleam")
    seen = []

    def failing_weights(*args, **kwargs):
        seen.append(gleampy.validation_enabled())
        with pytest.raises(GleamValidationError, match="simulation_duration.*positive"):
            gleampy.run_gleam(simulation_duration=0)
        seen.append(gleampy.validation_enabled())
        raise RuntimeError("Example calculation failure")

    monkeypatch.setattr(pipeline, "run_weights_module", failing_weights)
    with pytest.warns(GleamWarning, match=OFF_WARNING):
        with pytest.raises(RuntimeError, match="Example calculation failure"):
            run_gleam_default(d_gleam, has_herd_structure=True, validate_inputs=False)
    assert seen == [False, False]
    assert _state() == (True, False)
    with pytest.raises(GleamValidationError, match="cohort_short"):
        gleampy.calc_cohort_weights("CTL", "invalid")


# ---- standalone modules ---------------------------------------------------------


def test_standalone_modules_skip_validation_only_when_requested(d_gleam, request):
    expected = gleampy.run_weights_module(d_gleam["cohort_structure"], d_gleam["herd"], show_indicator=False)
    request.getfixturevalue("no_rules")
    with pytest.warns(GleamWarning, match=OFF_WARNING):
        result = gleampy.run_weights_module(
            d_gleam["cohort_structure"], d_gleam["herd"], show_indicator=False, validate_inputs=False
        )
    assert_same(result, expected)
    assert _state() == (True, False)
    with pytest.raises(GleamValidationError, match="expected exactly one rule"):
        gleampy.run_weights_module(d_gleam["cohort_structure"], d_gleam["herd"], show_indicator=False)
    assert _state() == (True, False)


def test_standalone_modules_restore_validation_after_calculation_errors(d_gleam, monkeypatch):
    weights = importlib.import_module("gleampy.modules.weights")
    seen = []

    def failing_cohort_weights(*args, **kwargs):
        seen.append(gleampy.validation_enabled())
        raise RuntimeError("Example calculation failure")

    monkeypatch.setattr(weights, "calc_cohort_weights", failing_cohort_weights)
    with pytest.warns(GleamWarning, match=OFF_WARNING):
        with pytest.raises(RuntimeError, match="Example calculation failure"):
            gleampy.run_weights_module(
                d_gleam["cohort_structure"], d_gleam["herd"], show_indicator=False, validate_inputs=False
            )
    assert seen == [False]
    assert _state() == (True, False)


def test_checked_standalone_module_restores_state_after_validation_error(d_gleam):
    bad = d_gleam["herd"].drop(columns=["live_weight_at_birth"])
    with pytest.raises(GleamValidationError):
        gleampy.run_weights_module(d_gleam["cohort_structure"], bad, show_indicator=False)
    assert _state() == (True, False)


def test_validation_disabled_context_restores_state():
    with gleampy.validation_disabled():
        assert not gleampy.validation_enabled()
        with pytest.raises(ZeroDivisionError):
            with gleampy.validation_disabled():
                1 / 0  # noqa: B018
        assert not gleampy.validation_enabled()
    assert gleampy.validation_enabled()
