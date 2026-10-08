"""Validation of the non-demographic herd core model (port of
``R/validate_nondemographic_herd_core_model.R``).

The R checks are scalar; these versions are element-wise (``if (x < 0)``
becomes "any element < 0", conditional range checks apply to the elements
meeting the condition).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .._utils import as_float, is_scalar
from ._shared import GleamValidationError, abort, validate_param_range, validate_scalar_numeric, validator


def _vals(x: Any) -> np.ndarray:
    return np.atleast_1d(as_float(x)).reshape(-1)


def _nonnegative(x: Any, arg_name: str) -> None:
    if (_vals(x) < 0).any():
        abort(f"`{arg_name}` must be greater than or equal to 0.")


def _range_where(x: Any, mask: np.ndarray, arg_name: str) -> None:
    """``if (cond) validate_param_range(x, arg_name)`` for the elements where ``cond``."""
    vals = _vals(x)
    mask = np.broadcast_to(np.asarray(mask, dtype=bool).reshape(-1), vals.shape)
    if not mask.any():
        return
    if is_scalar(x):
        validate_param_range(float(vals[0]), arg_name)
        return
    idx = np.flatnonzero(mask)
    validate_param_range(vals[idx], arg_name, labels=None if idx.size == 1 else list(idx + 1))


@validator
def validate_nondemo_cycle_geometry_inputs(
    phase1_nondemo_duration: Any, phase2_nondemo_duration: Any, rest_between_nondemo_cycles_duration: Any
) -> None:
    """Inputs of ``calc_nondemo_cycle_geometry()``."""
    validate_scalar_numeric(phase1_nondemo_duration, "phase1_nondemo_duration")
    validate_scalar_numeric(phase2_nondemo_duration, "phase2_nondemo_duration")
    validate_scalar_numeric(rest_between_nondemo_cycles_duration, "rest_between_nondemo_cycles_duration")
    _nonnegative(phase1_nondemo_duration, "phase1_nondemo_duration")
    _nonnegative(phase2_nondemo_duration, "phase2_nondemo_duration")
    _range_where(phase1_nondemo_duration, _vals(phase1_nondemo_duration) > 0, "cohort_duration_days")
    _range_where(phase2_nondemo_duration, _vals(phase2_nondemo_duration) > 0, "cohort_duration_days")
    validate_param_range(rest_between_nondemo_cycles_duration, "duration")


@validator
def validate_nondemo_start_size_inputs(
    cohort_stock_nondemo_annual_entrants: Any, total_nondemo_cycle_starts_to_distribute: Any
) -> None:
    """Inputs of ``calc_nondemo_start_sizes()``."""
    validate_scalar_numeric(cohort_stock_nondemo_annual_entrants, "cohort_stock_nondemo_annual_entrants")
    validate_scalar_numeric(total_nondemo_cycle_starts_to_distribute, "total_nondemo_cycle_starts_to_distribute")
    _nonnegative(cohort_stock_nondemo_annual_entrants, "cohort_stock_nondemo_annual_entrants")
    _nonnegative(total_nondemo_cycle_starts_to_distribute, "total_nondemo_cycle_starts_to_distribute")


@validator
def validate_nondemo_phase_inputs(
    cohort_stock_nondemo_start_by_phase: Any,
    productive_phase_nondemo_duration: Any,
    death_rate_nondemo_phase: Any,
    max_simulation_days_nondemo_phase: Any,
) -> None:
    """Inputs of ``calc_nondemo_phase()``."""
    validate_scalar_numeric(cohort_stock_nondemo_start_by_phase, "cohort_stock_nondemo_start_by_phase")
    validate_scalar_numeric(productive_phase_nondemo_duration, "productive_phase_nondemo_duration")
    validate_scalar_numeric(death_rate_nondemo_phase, "death_rate_nondemo_phase")
    validate_scalar_numeric(max_simulation_days_nondemo_phase, "max_simulation_days_nondemo_phase")
    _nonnegative(cohort_stock_nondemo_start_by_phase, "cohort_stock_nondemo_start_by_phase")
    _nonnegative(productive_phase_nondemo_duration, "productive_phase_nondemo_duration")
    _nonnegative(max_simulation_days_nondemo_phase, "max_simulation_days_nondemo_phase")
    _range_where(
        productive_phase_nondemo_duration, _vals(productive_phase_nondemo_duration) > 0, "cohort_duration_days"
    )
    validate_param_range(death_rate_nondemo_phase, "death_rate")
    _range_where(
        max_simulation_days_nondemo_phase, _vals(max_simulation_days_nondemo_phase) > 0, "simulation_duration"
    )


def _is_phase(phase: Any) -> bool:
    if not isinstance(phase, Mapping):
        return False
    stock = phase.get("cohort_stock_nondemo")
    return (
        phase.get("time_simulated_nondemographic") is not None
        and isinstance(stock, Mapping)
        and stock.get("start") is not None
        and stock.get("end") is not None
    )


@validator
def validate_nondemo_avg_stock_inputs(
    full_nondemo_phase_duration: Any, partial_nondemo_phase: Any, number_full_nondemo_cycles: Any
) -> None:
    """Inputs of ``calc_nondemo_avg_stock_phase_horizon()``."""
    for arg_name, phase in (
        ("full_nondemo_phase_duration", full_nondemo_phase_duration),
        ("partial_nondemo_phase", partial_nondemo_phase),
    ):
        if not _is_phase(phase):
            abort(f"`{arg_name}` must be a list produced by `calc_nondemo_phase()`.")
    validate_scalar_numeric(number_full_nondemo_cycles, "number_full_nondemo_cycles")
    _nonnegative(number_full_nondemo_cycles, "number_full_nondemo_cycles")


@validator
def validate_nondemo_offtake_inputs(
    cohort_stock_nondemo_end_phase1: Any,
    cohort_stock_nondemo_end_phase2: Any,
    number_full_nondemo_cycles: Any,
    partial_phase1_nondemo_duration: Any,
    partial_phase2_nondemo_duration: Any,
    phase1_nondemo_duration: Any,
    phase2_nondemo_duration: Any,
    simulation_duration: Any,
) -> None:
    """Inputs of ``calc_nondemo_offtake_total_horizon()``.

    Parameters
    ----------
    cohort_stock_nondemo_end_phase1, cohort_stock_nondemo_end_phase2 : float or array-like
        Animals left at the end of a full productive phase 1 / 2 (heads / cycle).
    number_full_nondemo_cycles : float or array-like
        Complete production cycles within the 365-day horizon (cycles / year).
    partial_phase1_nondemo_duration, partial_phase2_nondemo_duration : float or array-like
        Terminal partial phase durations (days). As in R, a positive value is
        range-checked as ``cohort_duration_days`` [1, 8000], so a partial phase
        shorter than 1 day is rejected; the message says it is a partial phase.
    phase1_nondemo_duration, phase2_nondemo_duration : float or array-like
        Full productive phase durations (days).
    simulation_duration : float or array-like
        Length of the assessment period (days).
    """
    numeric_args = {
        "cohort_stock_nondemo_end_phase1": cohort_stock_nondemo_end_phase1,
        "cohort_stock_nondemo_end_phase2": cohort_stock_nondemo_end_phase2,
        "number_full_nondemo_cycles": number_full_nondemo_cycles,
        "partial_phase1_nondemo_duration": partial_phase1_nondemo_duration,
        "partial_phase2_nondemo_duration": partial_phase2_nondemo_duration,
        "phase1_nondemo_duration": phase1_nondemo_duration,
        "phase2_nondemo_duration": phase2_nondemo_duration,
        "simulation_duration": simulation_duration,
    }
    for arg_name, value in numeric_args.items():
        validate_scalar_numeric(value, arg_name)
        _nonnegative(value, arg_name)

    validate_param_range(phase1_nondemo_duration, "cohort_duration_days")
    _range_where(phase2_nondemo_duration, _vals(phase2_nondemo_duration) > 0, "cohort_duration_days")
    for arg_name, partial in (
        ("partial_phase1_nondemo_duration", partial_phase1_nondemo_duration),
        ("partial_phase2_nondemo_duration", partial_phase2_nondemo_duration),
    ):
        _partial_phase_range(partial, arg_name)
    _range_where(simulation_duration, _vals(simulation_duration) > 0, "simulation_duration")


def _partial_phase_range(partial: Any, arg_name: str) -> None:
    """R's ``cohort_duration_days`` range check of a terminal partial phase, with its cause.

    As in R, a positive partial phase shorter than 1 day is rejected with the
    ``cohort_duration_days`` message (a replicated R bug: the partial phase is
    not an input but the remainder of the 365-day horizon after the full
    cycles, and only fractional phase or rest durations make it shorter than
    1 day). The message is extended to say so.
    """
    try:
        _range_where(partial, _vals(partial) > 0, "cohort_duration_days")
    except GleamValidationError as err:
        msg = (
            f"{err} The value is `{arg_name}`, the part of a production cycle left in the 365-day "
            "horizon after the full cycles, not an input value: like R, GLEAM rejects a partial "
            "phase shorter than 1 day, which only fractional phase or rest durations produce."
        )
    else:
        return
    abort(msg)
