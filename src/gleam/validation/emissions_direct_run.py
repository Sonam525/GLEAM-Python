"""Input validation for :func:`gleam.run_emissions_direct`.

Port of ``R/validate_run_emissions_direct_inputs.R``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import is_na
from ._shared import (
    _vals,
    abort,
    check_cohort_completeness,
    check_required_columns,
    validate_cohort_short_values,
    validate_param_range,
    validator,
)
from .gleam_run import (
    GLEAM_CALCULATED_COLUMNS,
    HERD_STRUCTURE_COLUMNS,
    check_gwp_set,
    check_herd_id_sets,
    check_logical_flag,
    check_no_calculated_columns,
    check_simulation_duration,
)

#: Primary nutritional-quality columns that replace the feed tables.
RATION_QUALITY_COLUMNS: tuple[str, ...] = (
    "ration_gross_energy",
    "ration_metabolizable_energy",
    "ration_nitrogen",
    "ration_digestibility_fraction",
    "ration_urinary_energy_fraction",
    "ration_ash",
)


@validator
def validate_run_emissions_direct_inputs(
    has_herd_structure: Any = False,
    cohort_level_data: Any = None,
    herd_level_data: Any = None,
    feed_rations: Any = None,
    feed_params: Any = None,
    manure_management_system_fraction: Any = None,
    manure_management_system_factors: Any = None,
    simulation_duration: Any = 365,
    global_warming_potential_set: Any = "AR6",
    emission_factors_only: Any = False,
) -> None:
    """Validate the inputs of :func:`gleam.run_emissions_direct`."""
    if (
        emission_factors_only is None
        or np.ndim(emission_factors_only) != 0
        or not isinstance(emission_factors_only, (bool, np.bool_))
    ):
        abort("`emission_factors_only` must be a single logical value (TRUE or FALSE).")

    if (feed_rations is None) != (feed_params is None):
        abort(
            "Provide `feed_rations` and `feed_params` together, or omit both and supply "
            "primary nutritional quality in `cohort_level_data`."
        )
    use_primary = feed_rations is None and feed_params is None
    if not use_primary:
        if not isinstance(feed_rations, pd.DataFrame):
            abort("`feed_rations` must be a data frame (e.g. data.table).")
        if not isinstance(feed_params, pd.DataFrame):
            abort("`feed_params` must be a data frame (e.g. data.table).")

    check_simulation_duration(simulation_duration)
    check_logical_flag(has_herd_structure, "has_herd_structure")
    check_gwp_set(global_warming_potential_set)

    for arg, val in (
        ("cohort_level_data", cohort_level_data),
        ("herd_level_data", herd_level_data),
        ("manure_management_system_fraction", manure_management_system_fraction),
        ("manure_management_system_factors", manure_management_system_factors),
    ):
        if val is None or not isinstance(val, pd.DataFrame):
            abort(f"`{arg}` must be a data frame (e.g. data.table).")

    required = ("herd_id", "species_short", "cohort_short")
    missing = [c for c in required if c not in cohort_level_data.columns]
    if missing:
        abort(
            f"Missing required columns in `cohort_level_data`: {_vals(missing)}. "
            "`species_short` (e.g. CTL, BFL, SHP) must be present for each cohort."
        )

    if len(cohort_level_data) == 0:
        abort("`cohort_level_data` must contain at least one row.")
    validate_cohort_short_values(cohort_level_data["cohort_short"], data_arg="cohort_level_data")
    check_cohort_completeness(cohort_level_data, "cohort_level_data")

    if not use_primary:
        check_required_columns(feed_rations, required, "feed_rations")
        req = cohort_level_data[list(required)].drop_duplicates()
        have = set(map(tuple, feed_rations[list(required)].drop_duplicates().itertuples(index=False)))
        miss = [r for r in map(tuple, req.itertuples(index=False)) if r not in have]
        if miss:
            info = [" / ".join(str(v) for v in r) for r in miss]
            abort(
                f"Missing herd_id + species_short + cohort_short combinations in `feed_rations`: "
                f"{_vals(info)}. Feed rations must cover every supplied cohort."
            )

    cohort_blocklist = (
        [c for c in GLEAM_CALCULATED_COLUMNS if c not in HERD_STRUCTURE_COLUMNS]
        if has_herd_structure
        else list(GLEAM_CALCULATED_COLUMNS)
    )
    if use_primary:
        cohort_blocklist = [c for c in cohort_blocklist if c not in RATION_QUALITY_COLUMNS]

    check_no_calculated_columns(cohort_level_data, "cohort_level_data", cohort_blocklist)
    check_no_calculated_columns(herd_level_data, "herd_level_data")
    check_no_calculated_columns(feed_rations, "feed_rations")
    check_no_calculated_columns(manure_management_system_fraction, "manure_management_system_fraction")
    check_no_calculated_columns(manure_management_system_factors, "manure_management_system_factors")

    check_herd_id_sets(
        {
            "cohort_level_data": cohort_level_data,
            "herd_level_data": herd_level_data,
            "feed_rations": feed_rations,
            "manure_management_system_fraction": manure_management_system_fraction,
            "manure_management_system_factors": manure_management_system_factors,
        }
    )

    if use_primary:
        missing_q = [c for c in RATION_QUALITY_COLUMNS if c not in cohort_level_data.columns]
        if missing_q:
            abort(
                "When `feed_rations` and `feed_params` are omitted, supply primary nutritional "
                f"quality in `cohort_level_data`. Missing required columns: {_vals(missing_q)}."
            )
        for col in RATION_QUALITY_COLUMNS:
            validate_param_range(cohort_level_data[col].to_numpy(), arg_name=col)
