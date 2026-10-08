"""Run-level input validation for the production module.

Port of ``R/validate_run_production_inputs.R``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import as_float, as_str, is_true
from ._shared import (
    _vals,
    abort,
    check_cohort_completeness,
    check_data_table,
    check_herd_id_consistency,
    check_herd_id_unique,
    check_required_columns,
    normalize_optional_is_egg_producing_column,
    validate_cohort_short_values,
    validate_is_egg_producing_flag,
    validate_scalar_numeric,
    validation_enabled,
    validator,
)

#: Columns required in the cohort-level table.
REQUIRED_COHORT_COLUMNS: tuple[str, ...] = (
    "herd_id", "cohort_short", "cohort_stock_size",
    "offtake_heads_assessment", "live_weight_cohort_at_slaughter",
)

#: Columns required in the herd-level table.
REQUIRED_HERD_COLUMNS: tuple[str, ...] = (
    "herd_id", "species_short",
    "milk_yield_day", "lactating_females_fraction",
    "milk_protein_fraction", "milk_fat_fraction", "milk_lactose_fraction",
    "milk_protein_fraction_standard", "milk_fat_fraction_standard", "milk_lactose_fraction_standard",
    "fibre_yield_year",
    "carcass_dressing_fraction", "bone_free_meat_fraction", "meat_protein_fraction",
)


@validator
def validate_run_production_module_inputs(
    cohort_level_data: pd.DataFrame, herd_level_data: pd.DataFrame
) -> None:
    """Validate the inputs of :func:`gleampy.run_production_module`.

    Checks table structure, required columns, valid cohort codes, the 6
    demographic cohorts per herd, unique herd ids and the same herd-id set in
    both tables.

    Like R, this adds ``is_egg_producing = NA`` to ``cohort_level_data`` (in
    place) when the column is absent and there are no CHK herds; the module
    runner calls it on its own working copy.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table (see :func:`gleampy.run_production_module`).
    herd_level_data : pandas.DataFrame
        Herd-level table (see :func:`gleampy.run_production_module`).
    """
    check_data_table(cohort_level_data, "cohort_level_data")
    check_data_table(herd_level_data, "herd_level_data")
    normalize_optional_is_egg_producing_column(cohort_level_data, herd_level_data)

    check_required_columns(cohort_level_data, REQUIRED_COHORT_COLUMNS, "cohort_level_data")
    check_required_columns(herd_level_data, REQUIRED_HERD_COLUMNS, "herd_level_data")

    validate_cohort_short_values(cohort_level_data["cohort_short"], data_arg="cohort_level_data")
    check_cohort_completeness(cohort_level_data, "cohort_level_data")

    check_herd_id_unique(herd_level_data, "herd_level_data")

    check_herd_id_consistency(cohort_level_data, herd_level_data, "cohort_level_data", "herd_level_data")


def check_single_value(x: Any, arg_name: str) -> None:
    """R ``length(x) == 1``: a scalar or a one-element list, array or Series.

    The ``run_*`` functions pass ``simulation_duration`` to vectorised
    ``calc_*`` functions, where a vector would silently give each cohort row
    its own value. R cannot run with a vector either: its by-row ``:=``
    assignments (and ``if (simulation_duration <= 0)``) fail. So this check
    runs even when validation is switched off.

    Parameters
    ----------
    x : Any
        Value to check.
    arg_name : str
        Argument name used in the error message.
    """
    if x is None or np.size(x) != 1:
        abort(f"`{arg_name}` must be a single numeric value.")


def validate_production_simulation_duration(simulation_duration: Any) -> None:
    """R ``validate_scalar_numeric(simulation_duration)`` plus ``> 0``.

    A single (length-1) numeric, non-missing, positive value (days). The
    length check always runs (see :func:`check_single_value`); the others
    only when validation is enabled.

    Parameters
    ----------
    simulation_duration : Any
        Length of the assessment period (days).
    """
    check_single_value(simulation_duration, "simulation_duration")
    if not validation_enabled():
        return
    validate_scalar_numeric(simulation_duration, "simulation_duration")
    if np.any(as_float(simulation_duration) <= 0):
        abort("`simulation_duration` must be positive.")


#: Herd-level columns read only for egg-producing cohorts.
EGG_HERD_COLUMNS: tuple[str, ...] = ("egg_output_human_consumption", "egg_average_weight")


def check_egg_columns(cohort_level_data: pd.DataFrame, herd_level_data: pd.DataFrame, species_short: Any) -> None:
    """Columns that R reads lazily, only for the rows that need them.

    * ``is_egg_producing`` is read for every row (R stops with "object not
      found" when it is absent, which can only happen with CHK herds since
      it is added as NA otherwise); checked whatever ``validate_inputs``.
    * ``nondemo_productive_phase_id`` is needed by the flag check of
      egg-producing CHK ``FN`` cohorts.
    * ``egg_output_human_consumption`` and ``egg_average_weight`` (herd
      level) are needed by the cohorts whose flag is a logical TRUE and
      passes the flag check (CHK ``FA``, or CHK ``FN`` in phase 2).

    The last two are checked only when validation is enabled; unvalidated
    runs read an absent column as NA, as before.

    R's ``calc_egg_production()`` validates the flag placement
    (``validate_is_egg_producing_flag()``) before it forces the herd egg
    columns, so a misplaced TRUE flag (non-CHK species, CHK cohort other
    than ``FA``/``FN``, CHK ``FN`` outside phase 2) is reported as such even
    when those columns are absent. The flag checks therefore run here first,
    over all rows (category by category, see ``docs/R_PORT_NOTES.md``); the
    absent ``nondemo_productive_phase_id`` is reported at the place of R's
    phase check, after the species and cohort checks of the flag.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table (after the optional ``is_egg_producing`` column
        has been added).
    herd_level_data : pandas.DataFrame
        Herd-level table.
    species_short : array-like
        Species code of each cohort row (already validated by the milk step).
    """
    if "is_egg_producing" not in cohort_level_data.columns:
        abort(f"Missing required columns in `cohort_level_data`: {_vals(['is_egg_producing'])}")
    if not validation_enabled():
        return
    flag = cohort_level_data["is_egg_producing"]
    if not is_true(flag).any():
        return  # no row forces the egg columns; flag errors are reported by calc_egg_production
    cohort_short = cohort_level_data["cohort_short"]
    has_phase = "nondemo_productive_phase_id" in cohort_level_data.columns
    # Flag placement first. Without the phase column every phase check passes
    # here (placeholder 2), so the earlier flag errors keep precedence.
    validate_is_egg_producing_flag(
        species_short=species_short,
        cohort_short=cohort_short,
        is_egg_producing=flag,
        nondemo_productive_phase_id=cohort_level_data["nondemo_productive_phase_id"] if has_phase else 2,
    )
    # From here every TRUE row is a CHK FA row or a CHK FN row (in phase 2
    # when the phase column exists).
    laying = is_true(flag)
    if not has_phase and (laying & np.asarray(as_str(cohort_short) == "FN", dtype=bool)).any():
        abort(f"Missing required columns in `cohort_level_data`: {_vals(['nondemo_productive_phase_id'])}")
    missing = [c for c in EGG_HERD_COLUMNS if c not in herd_level_data.columns]
    if missing:
        abort(f"Missing required columns in `herd_level_data`: {_vals(missing)}")
