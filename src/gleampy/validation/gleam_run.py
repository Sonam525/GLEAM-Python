"""Input validation for :func:`gleampy.run_gleam` (port of ``R/validate_run_gleam_inputs.R``).

Also hosts the checks shared with :func:`gleampy.run_emissions_direct`
(``R/validate_run_emissions_direct_inputs.R`` duplicates them in R).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import as_float, is_na, is_true_scalar, isin
from ._shared import _vals, abort, check_required_columns, validator

#: Columns that GLEAM computes; users must not provide them as inputs.
GLEAM_CALCULATED_COLUMNS: tuple[str, ...] = (
    # Herd simulation (cohort and herd)
    "cohort_stock_size", "offtake_heads", "offtake_heads_assessment", "growth_rate_herd",
    # Weights (cohort)
    "live_weight_mature_stage", "live_weight_cohort_initial", "live_weight_cohort_potential_final",
    "live_weight_cohort_at_slaughter", "live_weight_cohort_average", "live_weight_cohort_final",
    "daily_weight_gain",
    # Production cohort (cohort-level outputs)
    "milk_production_mass_cohort", "milk_production_protein_cohort", "milk_production_fpcm_cohort",
    "fibre_production_cohort",
    "meat_production_live_weight_cohort", "meat_production_carcass_weight_cohort",
    "meat_production_bone_free_meat_cohort", "meat_production_protein_cohort",
    # Feed rations (cohort-level outputs merged into pipeline)
    "ration_gross_energy", "ration_metabolizable_energy", "ration_nitrogen",
    "ration_digestibility_fraction", "ration_urinary_energy_fraction", "ration_ash",
    # Allocation (cohort-level energy allocation terms)
    "milk_allocation_energy", "meat_allocation_energy", "fibre_allocation_energy",
    "work_allocation_energy", "egg_allocation_energy",
    # Feed emissions (cohort-level diet emission factors)
    "co2_ration_fertilizer", "co2_ration_pesticides",
    "co2_ration_crop_activities", "co2_ration_luc_nopeat", "co2_ration_luc_peat",
    "n2o_ration_fertilizer", "n2o_ration_manure_applied", "n2o_ration_crop_residues",
    "ch4_ration_rice",
    # Energy requirements (cohort)
    "metabolic_energy_req_maintenance", "metabolic_energy_req_activity", "metabolic_energy_req_growth",
    "metabolic_energy_req_lactation", "metabolic_energy_req_work",
    "metabolic_energy_req_fibre_production", "metabolic_energy_req_pregnancy",
    "net_energy_maintenance_digestible_energy_ratio",
    "net_energy_growth_digestible_energy_ratio",
    "metabolic_energy_req_total", "ration_intake",
    # Nitrogen balance (cohort)
    "nitrogen_intake", "nitrogen_retention", "nitrogen_excretion",
    # Enteric direct emissions (cohort)
    "ch4_conversion_factor_ym", "ch4_enteric",
    # Manure direct emissions (cohort)
    "volatile_solids",
    "ch4_manure_pasture", "ch4_manure_burned", "ch4_manure_other", "ch4_manure_all_noburn",
    "n2o_manure_pasture_direct", "n2o_manure_burned_direct", "n2o_manure_other_direct",
    "n2o_manure_all_noburn_direct",
    "n2o_manure_pasture_vol", "n2o_manure_burned_vol", "n2o_manure_other_vol",
    "n2o_manure_all_noburn_vol",
    "n2o_manure_pasture_leach", "n2o_manure_burned_leach", "n2o_manure_other_leach",
    "n2o_manure_all_noburn_leach",
    "n2o_manure_pasture_indirect", "n2o_manure_burned_indirect", "n2o_manure_other_indirect",
    "n2o_manure_pasture_total", "n2o_manure_burned_total", "n2o_manure_other_total",
)

HERD_STRUCTURE_COLUMNS: tuple[str, ...] = ("cohort_stock_size", "offtake_heads", "offtake_heads_assessment")


def _is_frame(x: Any) -> bool:
    return isinstance(x, pd.DataFrame)


def check_simulation_duration(simulation_duration: Any) -> None:
    """``simulation_duration`` must be a single positive number (days).

    Parameters
    ----------
    simulation_duration : Any
        Value to check. Arrays, lists and Series are rejected even with one
        element, as are logicals, strings and missing values.
    """
    if (
        isinstance(simulation_duration, (bool, np.bool_, str))
        or simulation_duration is None
        or np.ndim(simulation_duration) != 0
        or not isinstance(simulation_duration, (int, float, np.number))
        or is_na(simulation_duration)
    ):
        abort("`simulation_duration` must be a single numeric value.")
    if simulation_duration <= 0:
        abort("`simulation_duration` must be positive (days).")


def is_single_logical(value: Any) -> bool:
    """R ``is.logical(x) && length(x) == 1 && !is.na(x)`` for a switch.

    A Python ``bool``, a ``numpy.bool_`` or a one-element bool array, the
    values :func:`gleampy._utils.is_true_scalar` reads as a logical.

    Parameters
    ----------
    value : Any
        Value to test.

    Returns
    -------
    bool
        Whether ``value`` is a single non-missing logical.
    """
    if isinstance(value, (bool, np.bool_)):
        return True
    return isinstance(value, np.ndarray) and value.dtype == bool and value.size == 1


def check_logical_flag(value: Any, arg: str) -> None:
    """``value`` must be a single non-missing logical (see :func:`is_single_logical`).

    Parameters
    ----------
    value : Any
        Value to check.
    arg : str
        Argument name used in the error message.
    """
    if is_single_logical(value):
        return
    if value is not None and np.ndim(value) == 0 and is_na(value):
        abort(f"`{arg}` must be TRUE or FALSE, not NA.")
    abort(f"`{arg}` must be a single logical value (TRUE or FALSE).")


def check_gwp_set(global_warming_potential_set: Any) -> None:
    """``global_warming_potential_set`` must be one of the supported GWP sets.

    Parameters
    ----------
    global_warming_potential_set : Any
        Value to check: a single string among
        :data:`gleampy.constants.GLOBAL_WARMING_POTENTIAL_SETS`.
    """
    if not isinstance(global_warming_potential_set, str):
        abort("`global_warming_potential_set` must be a single character value.")
    if global_warming_potential_set not in K.GLOBAL_WARMING_POTENTIAL_SETS:
        abort(
            f"`global_warming_potential_set` must be one of: {_vals(K.GLOBAL_WARMING_POTENTIAL_SETS)}. "
            f'Got: "{global_warming_potential_set}"'
        )


def check_frame(x: Any, arg: str) -> None:
    """``x`` must be a :class:`pandas.DataFrame`.

    Parameters
    ----------
    x : Any
        Value to check.
    arg : str
        Argument name used in the error message.
    """
    if x is None or not _is_frame(x):
        abort(f"`{arg}` must be a data frame (e.g. data.table).")


def check_required_cohort_columns(cohort_level_data: pd.DataFrame) -> None:
    """``cohort_level_data`` must have ``herd_id``, ``species_short`` and ``cohort_short``.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level master table.
    """
    required = ("herd_id", "species_short", "cohort_short")
    missing = [c for c in required if c not in cohort_level_data.columns]
    if missing:
        abort(
            f"Missing required columns in `cohort_level_data`: {_vals(missing)}. "
            "`species_short` (e.g. CTL, BFL, SHP) must be present for each cohort."
        )


def check_nondemographic_presence(cohort_level_data: pd.DataFrame, herd_level_data: pd.DataFrame) -> None:
    """FN / MN rows must exist for herds with non-zero juvenile diversion proportions.

    Only applies when ``herd_level_data`` has both ``prop_nondemo_fem_juv``
    and ``prop_nondemo_mal_juv``; ``herd_level_data`` then needs ``herd_id``
    (R fails there with "object 'herd_id' not found").

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort table with ``herd_id`` and ``cohort_short``.
    herd_level_data : pandas.DataFrame
        Herd table.
    """
    if not all(c in herd_level_data.columns for c in ("prop_nondemo_fem_juv", "prop_nondemo_mal_juv")):
        return
    check_required_columns(herd_level_data, ["herd_id"], "herd_level_data")
    herd_ids = herd_level_data["herd_id"].to_numpy(dtype=object)
    cohort_ids = cohort_level_data["herd_id"].to_numpy(dtype=object)
    for prop, cohort in (("prop_nondemo_fem_juv", "FN"), ("prop_nondemo_mal_juv", "MN")):
        p = as_float(herd_level_data[prop])
        required = list(pd.unique(herd_ids[~np.isnan(p) & (p != 0)]))
        present = set(pd.unique(cohort_ids[isin(cohort_level_data["cohort_short"], [cohort])]))
        missing = [h for h in required if h not in present]
        if missing:
            abort(
                f'Missing "{cohort}" rows in `cohort_level_data` for `herd_id` {_vals(missing)}. '
                f'`{prop}` is non-zero for these herds, so "{cohort}" must be present.'
            )


def check_no_calculated_columns(data: Any, source_name: str, blocklist: tuple[str, ...] | list[str] = GLEAM_CALCULATED_COLUMNS) -> None:
    """``data`` must not contain columns that GLEAM calculates.

    Parameters
    ----------
    data : pandas.DataFrame or None
        Input table; anything that is not a data frame is skipped.
    source_name : str
        Argument name used in the error message.
    blocklist : sequence of str, default GLEAM_CALCULATED_COLUMNS
        Column names that must be absent.
    """
    if data is None or not _is_frame(data):
        return
    provided = [c for c in blocklist if c in data.columns]
    if provided:
        abort(
            f"Do not provide these variables in `{source_name}`: {_vals(provided)}. "
            "GLEAM calculates them; they are not expected as inputs."
        )


def _unique_sorted_herd_ids(x: Any) -> list | None:
    if x is None or not _is_frame(x) or "herd_id" not in x.columns:
        return None
    ids = [v for v in pd.unique(x["herd_id"]) if not is_na(v)]
    if not ids:
        return None
    return sorted(ids, key=lambda v: (str(type(v)), v))


def check_herd_id_sets(sources: dict[str, Any]) -> None:
    """All inputs with a ``herd_id`` column must have the same herd-id set.

    Parameters
    ----------
    sources : dict of str to pandas.DataFrame or None
        Input tables by argument name; tables without a non-missing
        ``herd_id`` are skipped, and at least one must have one.
    """
    sets = {name: ids for name, df in sources.items() if (ids := _unique_sorted_herd_ids(df)) is not None}
    if not sets:
        abort("No pipeline input with `herd_id` found. Input tables must contain a non-empty `herd_id` column.")
    ref_name = next(iter(sets))
    ref = sets[ref_name]
    for name, cur in sets.items():
        if len(cur) != len(ref) or set(cur) != set(ref):
            abort(
                "All pipeline inputs must have the same `herd_id` set (same length and content). "
                f"Reference has {_vals(ref)}. Mismatch in `{name}`: {_vals(cur)}."
            )


@validator
def validate_run_gleam_inputs(
    has_herd_structure: Any,
    cohort_level_data: Any,
    herd_level_data: Any,
    feed_rations: Any,
    feed_params: Any,
    feed_emissions: Any,
    manure_management_system_fraction: Any,
    manure_management_system_factors: Any,
    simulation_duration: Any,
    global_warming_potential_set: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.run_gleam` (port of ``validate_run_gleam_inputs()``).

    Checks the scalar arguments, that every table is a data frame, the key
    cohort columns, FN / MN presence for herds that divert juveniles, that no
    GLEAM-calculated column is supplied, and that all tables share the same
    ``herd_id`` set. Module-specific columns are checked by each module.

    Parameters
    ----------
    has_herd_structure : bool
        Single non-missing logical (``bool``, ``numpy.bool_`` or a one-element
        bool array).
    cohort_level_data, herd_level_data : pandas.DataFrame
        Cohort- and herd-level master tables.
    feed_rations, feed_params, feed_emissions : pandas.DataFrame
        Feed ration shares (fractions), feed nutritional parameters and feed
        emission factors (g / kg DM).
    manure_management_system_fraction, manure_management_system_factors : pandas.DataFrame
        Manure management system shares (fractions) and emission factors.
    simulation_duration : float
        Single positive number (days).
    global_warming_potential_set : str
        One of ``"AR6"``, ``"AR5_excluding_carbon_feedback"``,
        ``"AR5_including_carbon_feedback"`` or ``"AR4"``.

    Raises
    ------
    GleamValidationError
        On the first failed check.
    """
    check_simulation_duration(simulation_duration)
    check_logical_flag(has_herd_structure, "has_herd_structure")
    check_gwp_set(global_warming_potential_set)

    check_frame(cohort_level_data, "cohort_level_data")
    check_frame(herd_level_data, "herd_level_data")
    check_frame(feed_rations, "feed_rations")
    check_frame(feed_params, "feed_params")
    check_frame(feed_emissions, "feed_emissions")
    check_frame(manure_management_system_fraction, "manure_management_system_fraction")
    check_frame(manure_management_system_factors, "manure_management_system_factors")

    check_required_cohort_columns(cohort_level_data)
    check_nondemographic_presence(cohort_level_data, herd_level_data)

    cohort_blocklist = (
        [c for c in GLEAM_CALCULATED_COLUMNS if c not in HERD_STRUCTURE_COLUMNS]
        if is_true_scalar(has_herd_structure)
        else list(GLEAM_CALCULATED_COLUMNS)
    )
    check_no_calculated_columns(cohort_level_data, "cohort_level_data", cohort_blocklist)
    check_no_calculated_columns(herd_level_data, "herd_level_data")
    check_no_calculated_columns(feed_rations, "feed_rations")
    check_no_calculated_columns(feed_emissions, "feed_emissions")
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
