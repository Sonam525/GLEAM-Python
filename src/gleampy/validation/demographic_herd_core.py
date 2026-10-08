"""Validation of the demographic herd core model (port of
``R/validate_demographic_herd_core_model.R``)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import as_float
from ._shared import (
    abort,
    validate_named_numeric_vector,
    validate_param_range,
    validate_scalar_numeric,
    validator,
)

SIX_COHORTS = K.GLEAM_COHORTS_DEMOGRAPHIC
EIGHT_COHORTS = ("FB", "FJ", "FS", "FA", "MB", "MJ", "MS", "MA")
TEN_COHORTS = ("FB", "FJ", "FS", "FA", "FC", "MB", "MJ", "MS", "MA", "MC")


def _single_numeric(x: Any, arg_name: str) -> None:
    """R ``validate_scalar_numeric()``: numeric, length 1, not NA."""
    if x is None or np.size(x) != 1:
        abort(f"`{arg_name}` must be a single numeric value.")
    validate_scalar_numeric(x, arg_name)


def param_range_named(x: Any, arg_name: str) -> None:
    """``validate_param_range()`` on a named vector (labels = names, as in R)."""
    if isinstance(x, Mapping):
        names, vals = list(x.keys()), list(x.values())
    elif isinstance(x, pd.Series):
        names, vals = list(x.index), list(x.values)
    else:
        validate_param_range(x, arg_name)
        return
    arr = np.asarray(vals, dtype=object)
    if all(isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) for v in arr):
        arr = as_float(arr)
    validate_param_range(arr, arg_name, labels=names)


@validator
def validate_fecundity_inputs(parturition_rate: Any, litter_size: Any, birth_fraction_female: Any) -> None:
    """Inputs of ``calc_fecundity_rates()`` (element-wise for arrays)."""
    validate_scalar_numeric(parturition_rate, "parturition_rate")
    validate_scalar_numeric(litter_size, "litter_size")
    validate_scalar_numeric(birth_fraction_female, "birth_fraction_female")
    validate_param_range(parturition_rate, "parturition_rate")
    validate_param_range(litter_size, "litter_size")
    validate_param_range(birth_fraction_female, "birth_fraction_female")


@validator
def validate_transition_inputs(cohort_duration_days: Any, offtake_rate: Any, death_rate: Any) -> None:
    """Inputs of ``calc_transition_probabilities()`` (named vectors of length 6)."""
    validate_named_numeric_vector(cohort_duration_days, 6, None, "cohort_duration_days")
    validate_named_numeric_vector(offtake_rate, 6, None, "offtake_rate")
    validate_named_numeric_vector(death_rate, 6, None, "death_rate")
    param_range_named(cohort_duration_days, "cohort_duration_days")
    param_range_named(death_rate, "death_rate")
    param_range_named(offtake_rate, "offtake_rate")


@validator
def validate_steady_state_inputs(
    initial_herd_structure: Any,
    max_simulation_years: Any,
    min_lambda_change: Any,
    fecundity_female: Any,
    fecundity_male: Any,
    probability_death: Any,
    probability_offtake: Any,
    probability_growth: Any,
    proportion_nondemographic: Any,
) -> None:
    """Inputs of ``calc_steady_state_structure()``."""
    validate_named_numeric_vector(initial_herd_structure, 6, SIX_COHORTS, "initial_herd_structure")
    validate_named_numeric_vector(probability_death, 10, TEN_COHORTS, "probability_death")
    validate_named_numeric_vector(probability_offtake, 10, TEN_COHORTS, "probability_offtake")
    validate_named_numeric_vector(probability_growth, 10, TEN_COHORTS, "probability_growth")
    validate_named_numeric_vector(proportion_nondemographic, 6, SIX_COHORTS, "proportion_nondemographic")
    _single_numeric(max_simulation_years, "max_simulation_years")
    _single_numeric(min_lambda_change, "min_lambda_change")
    _single_numeric(fecundity_female, "fecundity_female")
    _single_numeric(fecundity_male, "fecundity_male")
    param_range_named(proportion_nondemographic, "proportion_nondemographic")


@validator
def validate_population_size_inputs(
    herd_size_total: Any,
    fecundity_female: Any,
    fecundity_male: Any,
    probability_death: Any,
    probability_offtake: Any,
    probability_growth: Any,
    growth_rate_herd: Any,
    herd_structure: Any,
    cohort_share: Any,
    proportion_nondemographic: Any,
) -> None:
    """Inputs of ``calc_projected_population_size()``."""
    validate_named_numeric_vector(probability_death, 10, TEN_COHORTS, "probability_death")
    validate_named_numeric_vector(probability_offtake, 10, TEN_COHORTS, "probability_offtake")
    validate_named_numeric_vector(probability_growth, 10, TEN_COHORTS, "probability_growth")
    validate_named_numeric_vector(herd_structure, 8, EIGHT_COHORTS, "herd_structure")
    validate_named_numeric_vector(cohort_share, 6, SIX_COHORTS, "cohort_share")
    validate_named_numeric_vector(proportion_nondemographic, 6, SIX_COHORTS, "proportion_nondemographic")
    _single_numeric(herd_size_total, "herd_size_total")
    _single_numeric(fecundity_female, "fecundity_female")
    _single_numeric(fecundity_male, "fecundity_male")
    _single_numeric(growth_rate_herd, "growth_rate_herd")
    validate_param_range(herd_size_total, "herd_size_total")
    param_range_named(proportion_nondemographic, "proportion_nondemographic")


@validator
def validate_offtake_summary_inputs(
    cohort_stock_start: Any,
    cohort_stock_end_projected: Any,
    cohort_stock_average: Any,
    cohort_offtake_heads: Any,
    simulation_duration: Any,
) -> None:
    """Inputs of ``calc_summary_offtake()``."""
    validate_named_numeric_vector(cohort_stock_start, 6, None, "cohort_stock_start")
    validate_named_numeric_vector(cohort_stock_end_projected, 6, None, "cohort_stock_end_projected")
    validate_named_numeric_vector(cohort_stock_average, 6, None, "cohort_stock_average")
    validate_named_numeric_vector(cohort_offtake_heads, 10, None, "cohort_offtake_heads")
    _single_numeric(simulation_duration, "simulation_duration")
