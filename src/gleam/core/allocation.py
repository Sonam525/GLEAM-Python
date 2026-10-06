"""Biophysical (energy-based) allocation (port of ``R/core_model_allocation.R``).

The allocation approach implements the IDF (2022) standard for the dairy
sector, adapted from Thoma and Nemecek (2020) and consistent with the FAO LEAP
guidelines (FAO, 2016a, b, c) and ISO 14044:2006 (Section 4.3.4.2, Step 2):
shared emissions are apportioned to co-products (milk, meat, fibre, work,
eggs) in proportion to the energy needed to produce them.

The ``calc_*`` functions are vectorised: they accept scalars, lists, numpy
arrays or pandas Series (broadcast against each other) and reproduce, element
by element, the scalar species / cohort branches of the R code (which is
evaluated row by row with ``by = .I``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import (
    all_scalar,
    as_float,
    as_str,
    broadcast,
    finalize,
    finalize_dict,
    is_true,
    isin,
    merge_dt,
)
from ..validation.allocation_core import (
    validate_allocation_egg_inputs,
    validate_allocation_fibre_inputs,
    validate_allocation_meat_inputs,
    validate_allocation_milk_inputs,
    validate_allocation_work_inputs,
)

#: Poultry growth coefficient (MJ/g): mean of adult and juvenile/sub-adult values.
_CGRO_CHK = (0.0279 + 0.0202) / 2

#: Energy content of eggs (MJ/kg egg), adapted from Sakomura (2004).
_EGG_ENERGY = 10.04


def calc_milk_allocation_energy(
    milk_production_fpcm_cohort: Any,
    milk_protein_fraction_standard: Any,
    milk_fat_fraction_standard: Any,
    milk_lactose_fraction_standard: Any,
) -> Any:
    """Energy required for milk production (for biophysical allocation).

    ``energy_standard = (0.0929 * fat + 0.0547 * protein + 0.0395 * lactose) * 4.184 * 100``
    (IDF, 2022; MJ/kg milk) and
    ``milk_allocation_energy = energy_standard * milk_production_fpcm_cohort``.

    Parameters
    ----------
    milk_production_fpcm_cohort : float or array-like
        Total fat- and protein-corrected milk (FPCM) produced over the
        assessment period (kg/cohort/assessment period).
    milk_protein_fraction_standard : float or array-like
        Standard protein content of milk (kg protein/kg milk), e.g. 0.033.
    milk_fat_fraction_standard : float or array-like
        Standard fat content of milk (kg fat/kg milk), e.g. 0.04.
    milk_lactose_fraction_standard : float or array-like
        Standard lactose content of milk (kg lactose/kg milk), e.g. 0.048.

    Returns
    -------
    float or numpy.ndarray
        Energy required to produce the cohort's milk output
        (MJ/cohort/assessment period).
    """
    validate_allocation_milk_inputs(
        milk_production_fpcm_cohort,
        milk_protein_fraction_standard,
        milk_fat_fraction_standard,
        milk_lactose_fraction_standard,
    )
    scalar = all_scalar(
        milk_production_fpcm_cohort, milk_protein_fraction_standard,
        milk_fat_fraction_standard, milk_lactose_fraction_standard,
    )
    milk, prot, fat, lac = broadcast(
        as_float(milk_production_fpcm_cohort),
        as_float(milk_protein_fraction_standard),
        as_float(milk_fat_fraction_standard),
        as_float(milk_lactose_fraction_standard),
    )
    with np.errstate(invalid="ignore", over="ignore"):
        # IDF (2022) coefficients: kcal per 100 g milk per 1 % unit of fat/protein/lactose
        energy_standard = (0.0929 * fat + 0.0547 * prot + 0.0395 * lac) * 4.184 * 100
        milk_allocation_energy = energy_standard * milk
    return finalize(milk_allocation_energy, scalar)


def calc_meat_allocation_energy(
    species_short: Any,
    cohort_short: Any,
    meat_production_live_weight_cohort: Any,
    live_weight_cohort_at_slaughter: Any = np.nan,
    live_weight_at_birth: Any = np.nan,
    ratio_me_to_ne: Any = np.nan,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> Any:
    """Energy required for meat production (for biophysical allocation).

    ``meat_allocation_energy = specific_energy_meat * meat_production_live_weight_cohort``
    with the species-specific energy per kg live weight (``s`` = slaughter
    weight, ``b`` = birth weight):

    * ``PGS``: 0 (pigs are single-output; :func:`calc_allocation_shares`
      assigns 100 % to meat);
    * ``CTL``, ``BFL``: ``22.02 * ((s - b) / 2 / (g * s))^0.75 * (s - b)^1.097 / s``
      with growth efficiency ``g = 0.8`` for female cohorts, 1 otherwise;
    * ``CML``: ``(41.8 * (s - b) / s) / ratio_me_to_ne``;
    * ``SHP``: ``(s - b) * (a + 0.5 * k * (b + s)) / s`` with
      ``a, k = 2.1, 0.45`` (females) or ``4.4, 0.32`` (males);
    * ``GTS``: same with ``a, k = 5, 0.33``;
    * ``CHK``: ``cgro * 1000 * (s - b) / s`` with
      ``cgro = (0.0279 + 0.0202) / 2`` MJ/g (Sakomura, 2004).

    Parameters
    ----------
    species_short : str or array-like
        Species code (``CTL``, ``BFL``, ``SHP``, ``GTS``, ``PGS``, ``CML``, ``CHK``).
    cohort_short : str or array-like
        Cohort code (``FJ``, ``FS``, ``FA``, ``MJ``, ``MS``, ``MA``, ``FN``, ``MN``).
    meat_production_live_weight_cohort : float or array-like
        Meat produced as live weight over the assessment period
        (kg/cohort/assessment period).
    live_weight_cohort_at_slaughter : float or array-like
        Live weight at slaughter (kg); not needed for ``PGS``.
    live_weight_at_birth : float or array-like
        Live weight at birth (kg); not needed for ``PGS``.
    ratio_me_to_ne : float or array-like
        Ratio of metabolizable to net energy; used for ``CML`` only.
    nondemo_productive_phase_id : float or array-like
        Productive phase of non-demographic cohorts (validation only).
    is_egg_producing : bool or array-like
        Egg-producing ``CHK`` cohort flag (validation only).

    Returns
    -------
    float or numpy.ndarray
        Energy required for the cohort's meat output (MJ/cohort/assessment period).
    """
    validate_allocation_meat_inputs(
        species_short, cohort_short, meat_production_live_weight_cohort,
        live_weight_cohort_at_slaughter, live_weight_at_birth, ratio_me_to_ne,
        nondemo_productive_phase_id, is_egg_producing,
    )
    scalar = all_scalar(
        species_short, cohort_short, meat_production_live_weight_cohort,
        live_weight_cohort_at_slaughter, live_weight_at_birth, ratio_me_to_ne,
    )
    sp, co, meat, s, b, ratio = broadcast(
        as_str(species_short),
        as_str(cohort_short),
        as_float(meat_production_live_weight_cohort),
        as_float(live_weight_cohort_at_slaughter),
        as_float(live_weight_at_birth),
        as_float(ratio_me_to_ne),
    )
    female = isin(co, K.GLEAM_COHORTS_FEMALE)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Cattle and buffalo: growth efficiency factor based on cohort sex
        growth_efficiency_factor = np.where(female, 0.8, 1.0)
        e_large = (
            22.02 * (((s - b) / 2) / (growth_efficiency_factor * s)) ** 0.75
            * (s - b) ** 1.097
        ) / s
        # Camelids: convert ME to NE with ratio_me_to_ne
        e_cml = (41.8 * (s - b) / s) / ratio
        # Sheep: coefficients by sex; goats: fixed coefficients
        a_shp = np.where(female, 2.1, 4.4)
        k_shp = np.where(female, 0.45, 0.32)
        e_shp = ((s - b) * (a_shp + 0.5 * k_shp * (b + s))) / s
        e_gts = ((s - b) * (5 + 0.5 * 0.33 * (b + s))) / s
        # Chickens
        e_chk = (_CGRO_CHK * 1000 * (s - b)) / s

        # R leaves the energy undefined (an error) for unknown species -> NA
        specific_energy_meat = np.select(
            [sp == "PGS", isin(sp, ("CTL", "BFL")), sp == "CML", sp == "SHP", sp == "GTS", sp == "CHK"],
            [np.zeros(sp.shape), e_large, e_cml, e_shp, e_gts, e_chk],
            default=np.nan,
        )
        meat_allocation_energy = specific_energy_meat * meat
    return finalize(meat_allocation_energy, scalar)


def calc_fibre_allocation_energy(
    species_short: Any,
    cohort_stock_size: Any = np.nan,
    metabolic_energy_req_fibre_production: Any = np.nan,
    ratio_me_to_ne: Any = np.nan,
    simulation_duration: Any = np.nan,
) -> Any:
    """Energy required for fibre production (for biophysical allocation).

    * ``SHP``, ``GTS``: ``metabolic_energy_req_fibre_production * simulation_duration * cohort_stock_size``;
    * ``CML``: ``(metabolic_energy_req_fibre_production / ratio_me_to_ne) * simulation_duration * cohort_stock_size``;
    * other species: 0.

    Parameters
    ----------
    species_short : str or array-like
        Species code.
    cohort_stock_size : float or array-like
        Average cohort population (heads).
    metabolic_energy_req_fibre_production : float or array-like
        Energy for fibre synthesis (MJ/head/day; NE for SHP/GTS, ME for CML).
    ratio_me_to_ne : float or array-like
        Ratio of metabolizable to net energy; used for ``CML`` only.
    simulation_duration : float or array-like
        Length of the assessment period (days).

    Returns
    -------
    float or numpy.ndarray
        Energy required for the cohort's fibre output (MJ/cohort/assessment period).
    """
    validate_allocation_fibre_inputs(
        species_short, cohort_stock_size,
        metabolic_energy_req_fibre_production, ratio_me_to_ne, simulation_duration,
    )
    scalar = all_scalar(
        species_short, cohort_stock_size, metabolic_energy_req_fibre_production,
        ratio_me_to_ne, simulation_duration,
    )
    sp, stock, fibre, ratio, duration = broadcast(
        as_str(species_short),
        as_float(cohort_stock_size),
        as_float(metabolic_energy_req_fibre_production),
        as_float(ratio_me_to_ne),
        as_float(simulation_duration),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e_smr = fibre * duration * stock
        e_cml = (fibre / ratio) * duration * stock
        fibre_allocation_energy = np.select(
            [isin(sp, ("GTS", "SHP")), sp == "CML"], [e_smr, e_cml], default=0.0
        )
    return finalize(fibre_allocation_energy, scalar)


def calc_work_allocation_energy(
    species_short: Any,
    cohort_stock_size: Any,
    metabolic_energy_req_work: Any,
    simulation_duration: Any,
    ratio_me_to_ne: Any = np.nan,
) -> Any:
    """Energy required for draught power (for biophysical allocation).

    * ``CML``: ``metabolic_energy_req_work * simulation_duration * cohort_stock_size / ratio_me_to_ne``;
    * other species: ``metabolic_energy_req_work * simulation_duration * cohort_stock_size``.

    Parameters
    ----------
    species_short : str or array-like
        Species code.
    cohort_stock_size : float or array-like
        Cohort population (heads).
    metabolic_energy_req_work : float or array-like
        Energy for work (MJ/head/day; NE, ME for CML).
    simulation_duration : float or array-like
        Length of the assessment period (days).
    ratio_me_to_ne : float or array-like
        Ratio of metabolizable to net energy; used for ``CML`` only.

    Returns
    -------
    float or numpy.ndarray
        Energy required for the cohort's work output (MJ/cohort/assessment period).
    """
    validate_allocation_work_inputs(
        species_short, cohort_stock_size, metabolic_energy_req_work,
        simulation_duration, ratio_me_to_ne,
    )
    scalar = all_scalar(
        species_short, cohort_stock_size, metabolic_energy_req_work,
        simulation_duration, ratio_me_to_ne,
    )
    sp, stock, work, duration, ratio = broadcast(
        as_str(species_short),
        as_float(cohort_stock_size),
        as_float(metabolic_energy_req_work),
        as_float(simulation_duration),
        as_float(ratio_me_to_ne),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e_direct = work * duration * stock
        e_cml = (work * duration * stock) / ratio
        work_allocation_energy = np.where(sp == "CML", e_cml, e_direct)
    return finalize(work_allocation_energy, scalar)


def calc_egg_allocation_energy(
    species_short: Any,
    cohort_short: Any,
    egg_production_mass_cohort: Any,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> Any:
    """Energy required for egg production (for biophysical allocation).

    ``egg_allocation_energy = egg_production_mass_cohort * 10.04`` (MJ/kg egg,
    adapted from Sakomura, 2004) for cohorts flagged ``is_egg_producing``
    (``CHK`` ``FA``, or ``FN`` in phase 2); 0 otherwise.

    Parameters
    ----------
    species_short : str or array-like
        Species code.
    cohort_short : str or array-like
        Cohort code.
    egg_production_mass_cohort : float or array-like
        Egg mass produced over the assessment period (kg/cohort/assessment period).
    nondemo_productive_phase_id : float or array-like
        Productive phase of non-demographic cohorts (validation only).
    is_egg_producing : bool or array-like
        Egg-producing ``CHK`` cohort flag.

    Returns
    -------
    float or numpy.ndarray
        Energy required for the cohort's egg output (MJ/cohort/assessment period).
    """
    validate_allocation_egg_inputs(
        species_short=species_short,
        cohort_short=cohort_short,
        nondemo_productive_phase_id=nondemo_productive_phase_id,
        egg_production_mass_cohort=egg_production_mass_cohort,
        is_egg_producing=is_egg_producing,
    )
    scalar = all_scalar(species_short, cohort_short, egg_production_mass_cohort, is_egg_producing)
    sp, egg, laying = broadcast(
        as_str(species_short), as_float(egg_production_mass_cohort), is_true(is_egg_producing)
    )
    with np.errstate(invalid="ignore", over="ignore"):
        egg_allocation_energy = np.where(laying, egg * _EGG_ENERGY, 0.0)
    return finalize(egg_allocation_energy, scalar)


def calc_cohort_to_herd_aggregation(
    data_cohort: pd.DataFrame,
    id_cols: str | Sequence[str],
    vars_to_sum: str | Sequence[str],
    cohort_short: Any = None,
) -> pd.DataFrame:
    """Aggregate cohort-level data to herd level by summing variables.

    Reproduces ``data_cohort[, lapply(.SD, sum), by = id_cols, .SDcols = vars_to_sum]``:
    groups are kept in order of first appearance, ``NA`` keys form their own
    group and a group sum is ``NA`` if any of its values is ``NA``.

    Parameters
    ----------
    data_cohort : pandas.DataFrame
        Cohort-level table (one row per cohort).
    id_cols : str or sequence of str
        Grouping columns (e.g. ``"herd_id"``).
    vars_to_sum : str or sequence of str
        Numeric columns summed within each group.
    cohort_short : Any
        Unused (kept for signature parity with R).

    Returns
    -------
    pandas.DataFrame
        One row per group: the ``id_cols`` followed by the summed ``vars_to_sum``.
    """
    ids = [id_cols] if isinstance(id_cols, str) else list(id_cols)
    sums = [vars_to_sum] if isinstance(vars_to_sum, str) else list(vars_to_sum)
    df = data_cohort.reset_index(drop=True)
    missing = [c for c in ids + sums if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in `data_cohort`: {', '.join(missing)}")

    codes, ngroups = _group_codes(df, ids)
    first = np.full(ngroups, len(df), dtype=np.int64)
    np.minimum.at(first, codes, np.arange(len(df), dtype=np.int64))

    out = {c: df[c].iloc[first].reset_index(drop=True) for c in ids}
    for c in sums:
        col = df[c]
        if col.dtype.kind in "iu":
            # R keeps integer sums integer
            out[c] = pd.Series(np.bincount(codes, weights=col.to_numpy(dtype="float64"), minlength=ngroups)).astype(col.dtype)
        else:
            out[c] = pd.Series(np.bincount(codes, weights=as_float(col), minlength=ngroups))
    return pd.DataFrame(out)


def _group_codes(df: pd.DataFrame, keys: Sequence[str]) -> tuple[np.ndarray, int]:
    """Group number of each row, numbered by first appearance (``by =`` order)."""
    codes = np.zeros(len(df), dtype=np.int64)
    for k in keys:
        c, uniques = pd.factorize(df[k], use_na_sentinel=False)
        codes = codes * (len(uniques) + 1) + c
        codes, _ = pd.factorize(codes)
        codes = codes.astype(np.int64)
    ngroups = int(codes.max()) + 1 if len(codes) else 0
    return codes, ngroups


def calc_allocation_shares(
    species_short: Any,
    meat_allocation_energy: Any,
    milk_allocation_energy: Any,
    fibre_allocation_energy: Any,
    work_allocation_energy: Any,
    egg_allocation_energy: Any,
) -> dict[str, Any]:
    """Allocation shares of meat, milk, fibre, work and eggs.

    Each share is the commodity's energy divided by the total energy
    (``sum(..., na.rm = TRUE)`` of the five terms). For pigs (``PGS``) 100 %
    is allocated to meat and 0 to the other commodities.

    Parameters
    ----------
    species_short : str or array-like
        Species code.
    meat_allocation_energy, milk_allocation_energy, fibre_allocation_energy, work_allocation_energy, egg_allocation_energy : float or array-like
        Commodity energy requirements (MJ/herd/assessment period).

    Returns
    -------
    dict
        ``meat_share_allocation``, ``milk_share_allocation``,
        ``fibre_share_allocation``, ``work_share_allocation`` and
        ``eggs_share_allocation`` (fractions).
    """
    scalar = all_scalar(
        species_short, meat_allocation_energy, milk_allocation_energy,
        fibre_allocation_energy, work_allocation_energy, egg_allocation_energy,
    )
    sp, meat, milk, fibre, work, egg = broadcast(
        as_str(species_short),
        as_float(meat_allocation_energy),
        as_float(milk_allocation_energy),
        as_float(fibre_allocation_energy),
        as_float(work_allocation_energy),
        as_float(egg_allocation_energy),
    )
    # sum(c(...), na.rm = TRUE), element by element
    total_energy = np.zeros(sp.shape)
    for e in (meat, milk, fibre, work, egg):
        total_energy = total_energy + np.where(np.isnan(e), 0.0, e)

    pgs = sp == "PGS"
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = {
            "meat_share_allocation": np.where(pgs, 1.0, meat / total_energy),
            "milk_share_allocation": np.where(pgs, 0.0, milk / total_energy),
            "fibre_share_allocation": np.where(pgs, 0.0, fibre / total_energy),
            "work_share_allocation": np.where(pgs, 0.0, work / total_energy),
            "eggs_share_allocation": np.where(pgs, 0.0, egg / total_energy),
        }
    return finalize_dict(shares, scalar)


def assign_allocation_shares(
    allocation_herd_long: pd.DataFrame,
    emissions_vars: Sequence[str],
    commodities: Sequence[str],
    non_allocated_emission_sources: Sequence[str],
    commodity_col: str,
    allocation_col: str,
) -> pd.DataFrame:
    """Expand commodity allocation shares over emission sources.

    Builds every ``variable_name`` x ``commodity_name`` combination
    (``data.table::CJ``: unique, sorted), merges it with
    ``allocation_herd_long`` on ``commodity_col`` (many-to-many, result
    sorted by commodity) and applies the allocation rules:

    * sources in ``non_allocated_emission_sources`` (manure burned for fuel,
      manure deposited on pasture) go 100 % to ``"Other"`` and 0 to every
      other commodity (cut-off / avoidance of double counting; IDF, 2022);
    * all other sources get 0 for ``"Other"`` (other commodities unchanged).

    Parameters
    ----------
    allocation_herd_long : pandas.DataFrame
        Long-format herd-level allocation shares (one row per herd x commodity).
    emissions_vars : sequence of str
        Emission variables to expand over.
    commodities : sequence of str
        Commodity categories, e.g. ``["Other", "Milk", "Meat", "Fibre", "Work", "Eggs"]``.
    non_allocated_emission_sources : sequence of str
        Emission sources not allocated to commodities.
    commodity_col : str
        Commodity column of ``allocation_herd_long``.
    allocation_col : str
        Allocation-share column of ``allocation_herd_long``.

    Returns
    -------
    pandas.DataFrame
        ``allocation_herd_long`` expanded over ``emissions_vars`` with the
        rules above applied.
    """
    # 1) All combinations: commodity x emission variable (CJ sorts and de-duplicates)
    grid_vars = sorted(set(as_str(list(emissions_vars)).tolist()), key=_sort_key)
    grid_comm = sorted(set(as_str(list(commodities)).tolist()), key=_sort_key)
    grid = pd.DataFrame(
        {
            "variable_name": np.repeat(np.array(grid_vars, dtype=object), len(grid_comm)),
            "commodity_name": np.tile(np.array(grid_comm, dtype=object), len(grid_vars)),
        }
    )

    # 2) Expand allocation table by emission variables (many-to-many)
    out = merge_dt(allocation_herd_long, grid, by=commodity_col)

    # 3) Non-allocated emission sources -> 100 % to Other, 0 % to others;
    # 4) allocated emission sources -> Other = 0. Rows where the condition is
    #    NA (missing commodity) are not assigned, as in data.table.
    non_alloc = isin(as_str(out["variable_name"]), list(non_allocated_emission_sources))
    comm = as_str(out[commodity_col])
    is_other = np.array([v is not None and v == "Other" for v in comm], dtype=bool)
    not_other = np.array([v is not None and v != "Other" for v in comm], dtype=bool)

    share = as_float(out[allocation_col]).copy()
    share[non_alloc & is_other] = 1.0
    share[non_alloc & not_other] = 0.0
    share[~non_alloc & is_other] = 0.0
    out[allocation_col] = share
    return out


def _sort_key(v: Any) -> tuple[int, str]:
    """``forder`` order for character vectors: NA first, then C-locale order."""
    return (0, "") if v is None else (1, v)
