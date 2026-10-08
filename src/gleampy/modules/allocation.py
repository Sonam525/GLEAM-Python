"""Allocation module (port of ``R/run_allocation_module.R``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import Lookup, Progress, as_float, copy_frame, isin, lookup
from ..core.allocation import (
    assign_allocation_shares,
    calc_allocation_shares,
    calc_cohort_to_herd_aggregation,
    calc_egg_allocation_energy,
    calc_fibre_allocation_energy,
    calc_meat_allocation_energy,
    calc_milk_allocation_energy,
    calc_work_allocation_energy,
)
from ..validation._shared import _vals, abort, normalize_optional_is_egg_producing_column, setup_validation
from ..validation.allocation_run import validate_run_allocation_module_inputs
from ..validation.production_run import check_single_value

#: Allocation-share columns (melt measure variables, in R order) and the
#: commodity names they are renamed to.
_SHARE_TO_COMMODITY: dict[str, str] = {
    "meat_share_allocation": "Meat",
    "milk_share_allocation": "Milk",
    "fibre_share_allocation": "Fibre",
    "work_share_allocation": "Work",
    "eggs_share_allocation": "Eggs",
    "allocation_share_other": "Other",
}

_ENERGY_COLS: tuple[str, ...] = (
    "meat_allocation_energy",
    "milk_allocation_energy",
    "fibre_allocation_energy",
    "work_allocation_energy",
    "egg_allocation_energy",
)


def run_allocation_module(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
    simulation_duration: float = 365,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the biophysical allocation module.

    Computes cohort-level energy requirements for meat, milk, fibre, work and
    eggs, sums them to herd level, derives commodity allocation shares and
    assigns them to every emission source (IDF, 2022; FAO LEAP, 2016;
    ISO 14044:2006).

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table with ``herd_id``, ``cohort_short``,
        ``milk_production_fpcm_cohort`` (kg/cohort/period),
        ``live_weight_cohort_at_slaughter`` (kg),
        ``meat_production_live_weight_cohort`` (kg/cohort/period),
        ``metabolic_energy_req_fibre_production`` (MJ/head/day),
        ``cohort_stock_size`` (heads), ``metabolic_energy_req_work``
        (MJ/head/day), ``egg_production_mass_cohort`` (kg/cohort/period),
        ``nondemo_productive_phase_id`` (required; NA for demographic
        cohorts) and ``is_egg_producing`` (required when ``CHK`` herds are
        present; added as NA otherwise).
    herd_level_data : pandas.DataFrame
        Herd-level table (one row per ``herd_id``) with ``species_short``,
        ``live_weight_at_birth`` (kg), ``milk_protein_fraction_standard``,
        ``milk_fat_fraction_standard``, ``milk_lactose_fraction_standard``
        (kg/kg milk) and ``ratio_me_to_ne``.
    simulation_duration : float
        Length of the assessment period (days); a single value (a vector is
        rejected, as R cannot run with one either).
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    dict
        ``cohort_allocation_inputs``: the cohort table plus
        ``milk_allocation_energy``, ``meat_allocation_energy``,
        ``fibre_allocation_energy``, ``work_allocation_energy`` and
        ``egg_allocation_energy`` (MJ/cohort/period);
        ``allocation_long``: herd x emission source x commodity table with
        ``herd_id``, ``species_short``, ``variable_name``,
        ``commodity_name``, ``commodity_type`` (``Edible`` / ``Non-Edible``)
        and ``allocation_share`` (fraction; zero shares dropped).

    Notes
    -----
    Emissions from manure burned for fuel and manure deposited on pasture
    are not allocated to commodities: they go 100 % to ``"Other"``
    (:func:`gleampy.assign_allocation_shares`).
    """
    with setup_validation(validate_inputs):
        # --- Steps 1-2: working copies, then validate (R validates before
        # copying but adds `is_egg_producing` to its input by reference)
        cohort_level_data = copy_frame(cohort_level_data)
        herd_level_data = copy_frame(herd_level_data)
        validate_run_allocation_module_inputs(cohort_level_data, herd_level_data)
        # Idempotent; keeps unvalidated runs working when the optional flag is absent.
        normalize_optional_is_egg_producing_column(cohort_level_data, herd_level_data)
        # R has no run-level check, but its by-row assignments fail on a vector
        check_single_value(simulation_duration, "simulation_duration")
        if "is_egg_producing" not in cohort_level_data.columns:
            # only possible with CHK herds; R's egg energy reads it for every row
            abort(f"Missing required columns in `cohort_level_data`: {_vals(['is_egg_producing'])}")

        progress = Progress(show_indicator)
        progress.status("Computing allocation shares, please wait...")

        chrt, hrd = cohort_level_data, herd_level_data
        # herd_level_data[.SD, on = "herd_id", x.<col>], joined once for all columns
        herd = Lookup(chrt, hrd, "herd_id")
        species = herd("species_short")
        ratio_me_to_ne = herd("ratio_me_to_ne")
        # Only validated, never used in the energies: R reads it lazily, so an
        # unvalidated run works without it (validation requires it).
        if "nondemo_productive_phase_id" in chrt.columns:
            phase_id = chrt["nondemo_productive_phase_id"]
        else:
            phase_id = np.full(len(chrt), np.nan)

        # --- Step 3: cohort-level energy allocations
        chrt["milk_allocation_energy"] = calc_milk_allocation_energy(
            milk_production_fpcm_cohort=chrt["milk_production_fpcm_cohort"],
            milk_protein_fraction_standard=herd("milk_protein_fraction_standard"),
            milk_fat_fraction_standard=herd("milk_fat_fraction_standard"),
            milk_lactose_fraction_standard=herd("milk_lactose_fraction_standard"),
        )
        chrt["meat_allocation_energy"] = calc_meat_allocation_energy(
            species_short=species,
            cohort_short=chrt["cohort_short"],
            nondemo_productive_phase_id=phase_id,
            is_egg_producing=chrt["is_egg_producing"],
            live_weight_cohort_at_slaughter=chrt["live_weight_cohort_at_slaughter"],
            live_weight_at_birth=herd("live_weight_at_birth"),
            meat_production_live_weight_cohort=chrt["meat_production_live_weight_cohort"],
            ratio_me_to_ne=ratio_me_to_ne,
        )
        chrt["fibre_allocation_energy"] = calc_fibre_allocation_energy(
            species_short=species,
            cohort_stock_size=chrt["cohort_stock_size"],
            metabolic_energy_req_fibre_production=chrt["metabolic_energy_req_fibre_production"],
            ratio_me_to_ne=ratio_me_to_ne,
            simulation_duration=simulation_duration,
        )
        chrt["work_allocation_energy"] = calc_work_allocation_energy(
            species_short=species,
            cohort_stock_size=chrt["cohort_stock_size"],
            metabolic_energy_req_work=chrt["metabolic_energy_req_work"],
            ratio_me_to_ne=ratio_me_to_ne,
            simulation_duration=simulation_duration,
        )
        chrt["egg_allocation_energy"] = calc_egg_allocation_energy(
            species_short=species,
            cohort_short=chrt["cohort_short"],
            nondemo_productive_phase_id=phase_id,
            egg_production_mass_cohort=chrt["egg_production_mass_cohort"],
            is_egg_producing=chrt["is_egg_producing"],
        )

        # --- Step 4: aggregate from cohort to herd level
        allocation_herd = calc_cohort_to_herd_aggregation(
            data_cohort=chrt,
            id_cols="herd_id",
            vars_to_sum=list(_ENERGY_COLS),
            cohort_short="cohort_short",
        )
        # allocation_herd[herd_level_data, species_short := i.species_short, on = "herd_id"]
        allocation_herd["species_short"] = lookup(allocation_herd, hrd, "species_short")

        # --- Step 5: allocation shares per commodity
        shares = calc_allocation_shares(
            species_short=allocation_herd["species_short"],
            meat_allocation_energy=allocation_herd["meat_allocation_energy"],
            milk_allocation_energy=allocation_herd["milk_allocation_energy"],
            fibre_allocation_energy=allocation_herd["fibre_allocation_energy"],
            work_allocation_energy=allocation_herd["work_allocation_energy"],
            egg_allocation_energy=allocation_herd["egg_allocation_energy"],
        )
        for col, values in shares.items():
            allocation_herd[col] = values
        allocation_herd["allocation_share_other"] = 0.0

        # --- Step 6: reshape to long format and rename commodities
        allocation_herd_long = _melt(
            allocation_herd,
            id_vars=["herd_id", "species_short"],
            measure_vars=list(_SHARE_TO_COMMODITY),
            variable_name="commodity_name",
            value_name="allocation_share",
        )
        allocation_herd_long["commodity_name"] = np.array(
            [_SHARE_TO_COMMODITY[v] for v in allocation_herd_long["commodity_name"]], dtype=object
        )
        commodity = allocation_herd_long["commodity_name"].to_numpy(dtype=object)
        commodity_type = np.full(len(commodity), None, dtype=object)
        commodity_type[isin(commodity, ("Meat", "Milk", "Eggs"))] = "Edible"
        commodity_type[isin(commodity, ("Work", "Fibre", "Other"))] = "Non-Edible"
        allocation_herd_long["commodity_type"] = commodity_type

        # --- Step 7: assign allocation to emission sources
        allocation_herd_long = assign_allocation_shares(
            allocation_herd_long=allocation_herd_long,
            emissions_vars=[m["emissions_source"] for m in K.GLEAM_EMISSIONS_META],
            commodities=["Other", "Milk", "Meat", "Fibre", "Work", "Eggs"],
            non_allocated_emission_sources=list(K.GLEAM_NON_ALLOCATED_EMISSIONS),
            commodity_col="commodity_name",
            allocation_col="allocation_share",
        )
        # Remove rows with allocation_share == 0 (NA comparisons drop the row too)
        share = as_float(allocation_herd_long["allocation_share"])
        keep = ~np.isnan(share) & (share != 0)
        allocation_herd_long = allocation_herd_long.loc[keep].reset_index(drop=True)

        # Reorder columns for clarity (setcolorder: listed columns first)
        first = ["herd_id", "species_short", "variable_name", "commodity_name", "commodity_type", "allocation_share"]
        allocation_herd_long = allocation_herd_long[
            first + [c for c in allocation_herd_long.columns if c not in first]
        ]

        progress.success("Allocation calculation complete.")
        return {
            "cohort_allocation_inputs": chrt,
            "allocation_long": allocation_herd_long,
        }


def _melt(
    df: pd.DataFrame,
    id_vars: Sequence[str],
    measure_vars: Sequence[str],
    variable_name: str,
    value_name: str,
) -> pd.DataFrame:
    """``data.table::melt`` for numeric measure columns.

    Rows are ordered measure variable by measure variable, each block in the
    original row order; columns are ``id_vars``, ``variable_name``,
    ``value_name``.
    """
    n = len(df)
    m = len(measure_vars)
    idx = np.tile(np.arange(n), m)
    out = {c: df[c].iloc[idx].reset_index(drop=True) for c in id_vars}
    out[variable_name] = pd.Series(np.repeat(np.array(list(measure_vars), dtype=object), n), dtype=object)
    out[value_name] = pd.Series(
        np.concatenate([as_float(df[c]) for c in measure_vars]) if m else np.array([], dtype=float)
    )
    return pd.DataFrame(out)
