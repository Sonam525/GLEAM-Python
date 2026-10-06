"""Manure management emissions module (port of ``R/run_emissions_manure_module.R``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._utils import Progress, as_float, copy_frame
from ..core.emissions_manure import (
    CH4_MMS_FIELDS,
    N2O_DIRECT_MMS_FIELDS,
    N2O_LEACHING_MMS_FIELDS,
    N2O_VOLATILIZATION_MMS_FIELDS,
    calc_ch4_manure,
    calc_n2o_manure_direct,
    calc_n2o_manure_leaching,
    calc_n2o_manure_total,
    calc_n2o_manure_volatilization,
    calc_volatile_solids,
)
from ..validation._shared import abort, setup_validation, validation_enabled
from ..validation.emissions_manure_run import (
    _aligned_keys,
    _as_character,
    validate_run_emissions_manure_module_inputs,
)

_PHASE = "nondemo_productive_phase_id"
_MMS = "manure_management_system"


def run_emissions_manure_module(
    cohort_level_data: pd.DataFrame,
    manure_management_system_fraction: pd.DataFrame,
    manure_management_system_factors: pd.DataFrame,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> pd.DataFrame:
    """Run the manure management emissions module.

    Computes cohort-level CH4 and N2O emissions from manure management
    systems (MMS) with the IPCC (2006, 2019) Tier 2 methodology, reported by
    MMS group: manure deposited on pasture (``mms_pasture``), burned for fuel
    (``mms_burned``) and all other systems.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table with ``herd_id``, ``cohort_short``,
        ``ration_intake`` (kg DM/head/day), ``ration_digestibility_fraction``
        (fraction), ``ration_urinary_energy_fraction`` (fraction),
        ``ration_ash`` (kg ash/kg DM), ``nitrogen_excretion``
        (kg N/head/day) and optionally ``nondemo_productive_phase_id``.
    manure_management_system_fraction : pandas.DataFrame
        Fraction of manure handled in each system per herd and cohort:
        ``herd_id``, ``cohort_short``, ``manure_management_system``,
        ``manure_management_system_fraction`` (summing to 1 per herd /
        cohort) and optionally ``nondemo_productive_phase_id``.
    manure_management_system_factors : pandas.DataFrame
        Emission factors per herd and system: ``herd_id``,
        ``manure_management_system``, ``methane_conversion_factor_mcf`` (%),
        ``ch4_max_producing_capacity_bo`` (m3 CH4/kg VS), ``n2o_ef3``,
        ``n2o_ef4``, ``n2o_ef5``, ``nitrogen_fracgas``,
        ``nitrogen_fracleach``. (A ``ratio_m3CH4_to_kgCH4`` column is not
        used, as in R: the default 0.67 always applies.)
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    pandas.DataFrame
        The input cohort table (same rows and order) with the columns
        ``volatile_solids``; ``ch4_manure_{pasture,burned,other,all_noburn}``;
        ``n2o_manure_{pasture,burned,other,all_noburn}_direct``;
        ``n2o_manure_{pasture,burned,other,all_noburn}_vol``;
        ``n2o_manure_{pasture,burned,other,all_noburn}_leach``;
        ``n2o_manure_{pasture,burned,other}_indirect`` and
        ``n2o_manure_{pasture,burned,other}_total`` (all per head per day).
        Columns already present are overwritten in place, others appended.

    Notes
    -----
    R joins fractions and factors on ``herd_id`` and
    ``manure_management_system`` (inner join) and, for each cohort row,
    selects the joined rows with the same ``herd_id`` and ``cohort_short``
    and, when ``nondemo_productive_phase_id`` is in both cohort-level
    tables, the same phase id (a missing phase matches missing phases);
    missing ``herd_id`` / ``cohort_short`` never match (R ``==``). The first
    row per system is used and systems are passed to the core functions
    sorted by name (``split()``). Here this selection is done for all rows at
    once (see :class:`_CohortMMS`), and the core functions are evaluated
    vectorised over cohorts with per-cohort system sets given as masked
    arrays.

    Steps: :func:`gleam.calc_volatile_solids`, :func:`gleam.calc_ch4_manure`,
    :func:`gleam.calc_n2o_manure_direct`,
    :func:`gleam.calc_n2o_manure_volatilization`,
    :func:`gleam.calc_n2o_manure_leaching` and
    :func:`gleam.calc_n2o_manure_total`.
    """
    with setup_validation(validate_inputs):
        # --- Step 1: validate inputs
        validate_run_emissions_manure_module_inputs(
            cohort_level_data=cohort_level_data,
            manure_management_system_fraction=manure_management_system_fraction,
            manure_management_system_factors=manure_management_system_factors,
        )

        progress = Progress(show_indicator)
        progress.status("Calculating emissions from manure management systems...")

        # --- Step 2: prepare inputs
        cohort = copy_frame(cohort_level_data)
        mms_fraction = copy_frame(manure_management_system_fraction)
        mms_factors = copy_frame(manure_management_system_factors)
        use_phase_id = _PHASE in cohort.columns and _PHASE in mms_fraction.columns

        # New columns, in R's `:=` order (assigned to the table at the end).
        out: dict[str, Any] = {}

        # --- Step 3: volatile solids (VS)
        out["volatile_solids"] = volatile_solids = calc_volatile_solids(
            ration_intake=cohort["ration_intake"],
            ration_digestibility_fraction=cohort["ration_digestibility_fraction"],
            ration_urinary_energy_fraction=cohort["ration_urinary_energy_fraction"],
            ration_ash=cohort["ration_ash"],
        )

        # Fractions joined with factors, matched to every cohort row
        # (R: merge() + per-row mms_rows + build_mms_list()).
        mms_sets = _CohortMMS(cohort, mms_fraction, mms_factors, use_phase_id)
        nitrogen_excretion = cohort["nitrogen_excretion"]

        # --- Step 4: CH4 from manure (pasture, burned, other, total non-burned)
        out.update(
            calc_ch4_manure(volatile_solids=volatile_solids, **mms_sets.arguments(CH4_MMS_FIELDS))
        )

        # --- Step 5: direct N2O
        out.update(
            calc_n2o_manure_direct(
                nitrogen_excretion=nitrogen_excretion, **mms_sets.arguments(N2O_DIRECT_MMS_FIELDS)
            )
        )

        # --- Step 6: indirect N2O from volatilization
        out.update(
            calc_n2o_manure_volatilization(
                nitrogen_excretion=nitrogen_excretion,
                **mms_sets.arguments(N2O_VOLATILIZATION_MMS_FIELDS),
            )
        )

        # --- Step 7: indirect N2O from leaching/runoff
        out.update(
            calc_n2o_manure_leaching(
                nitrogen_excretion=nitrogen_excretion, **mms_sets.arguments(N2O_LEACHING_MMS_FIELDS)
            )
        )

        # --- Step 8: total N2O (direct + indirect)
        out.update(
            calc_n2o_manure_total(
                n2o_manure_pasture_vol=out["n2o_manure_pasture_vol"],
                n2o_manure_pasture_leach=out["n2o_manure_pasture_leach"],
                n2o_manure_burned_vol=out["n2o_manure_burned_vol"],
                n2o_manure_burned_leach=out["n2o_manure_burned_leach"],
                n2o_manure_other_vol=out["n2o_manure_other_vol"],
                n2o_manure_other_leach=out["n2o_manure_other_leach"],
                n2o_manure_pasture_direct=out["n2o_manure_pasture_direct"],
                n2o_manure_burned_direct=out["n2o_manure_burned_direct"],
                n2o_manure_other_direct=out["n2o_manure_other_direct"],
            )
        )

        cohort = _assign(cohort, out)

        progress.success("Emissions from manure management calculation complete.")
        return cohort


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


def _assign(df: pd.DataFrame, values: dict[str, Any]) -> pd.DataFrame:
    """Successive ``DT[, col := value]``: existing columns overwritten in place, new ones appended in order."""
    new = {}
    for name, v in values.items():
        if name in df.columns:
            df[name] = v
        else:
            new[name] = v
    if not new:
        return df
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


class _CohortMMS:
    """The MMS records used by each cohort row, in wide (system x cohort) form.

    R builds ``mms_data <- merge(fraction, factors, by = c("herd_id",
    "manure_management_system"))`` (inner join, ordered by the keys, then by
    fraction row and factor row) and, for each cohort row, selects
    ``mms_data[herd_id == current_herd_id & cohort_short == current_cohort
    (& phase match)]``, splits it by system name (sorted) and keeps the first
    row per system. Because the selection keys plus the system name refine
    the join keys, that first row is the first fraction row (in table order)
    with the cohort's keys and that system, combined with the first factor
    row (in table order) for its herd and system; fraction rows without
    factors are dropped. This class performs that selection for all rows at
    once with two hash joins.

    System names are sorted with Python's code-point ``sorted()``; R's
    ``split()`` uses the locale collation. They agree for lower-case names
    such as the bundled ``mms_*`` codes; otherwise only the summation order
    of the "other" systems (last-bit differences) would change.
    """

    def __init__(
        self,
        cohort: pd.DataFrame,
        fraction: pd.DataFrame,
        factors: pd.DataFrame,
        use_phase_id: bool,
    ) -> None:
        n = len(cohort)
        herd_c, herd_f, herd_a = _aligned_keys(cohort["herd_id"], fraction["herd_id"], factors["herd_id"])
        coh_c, coh_f = _aligned_keys(cohort["cohort_short"], fraction["cohort_short"])
        mms_f, mms_a = _aligned_keys(fraction[_MMS], factors[_MMS])

        # Factors: first row per herd and system (NA keys never selected / dropped by split()).
        fa = pd.DataFrame({"__h__": herd_a[0], "__m__": mms_a[0], "__fpos__": np.arange(len(factors))})
        fa = fa[~(herd_a[1] | mms_a[1])].drop_duplicates(["__h__", "__m__"], keep="first")

        # Fractions: first row per selection key and system, joined with the factors.
        sel = {"__h__": herd_f[0], "__c__": coh_f[0]}
        left = {"__h__": herd_c[0], "__c__": coh_c[0]}
        if use_phase_id:
            # is.na(current_phase_id) selects the rows with a missing phase id.
            ph_c, ph_f = _aligned_keys(cohort[_PHASE], fraction[_PHASE])
            filler = 0.0 if ph_c[0].dtype.kind == "f" else ""
            left["__pna__"], sel["__pna__"] = ph_c[1], ph_f[1]
            left["__p__"] = np.where(ph_c[1], filler, ph_c[0])
            sel["__p__"] = np.where(ph_f[1], filler, ph_f[0])
        on = list(sel)
        fr = pd.DataFrame({**sel, "__m__": mms_f[0], "__rpos__": np.arange(len(fraction))})
        # `==` with NA is NA and data.table does not select those rows.
        fr = fr[~(herd_f[1] | coh_f[1] | mms_f[1])].drop_duplicates(on + ["__m__"], keep="first")
        fr = fr.merge(fa, on=["__h__", "__m__"], how="inner", sort=False)

        lt = pd.DataFrame({**left, "__row__": np.arange(n)})[~(herd_c[1] | coh_c[1])]
        matched = lt.merge(fr, on=on, how="inner", sort=False)

        names = matched["__m__"].to_numpy(dtype=object)
        if names.size and not isinstance(names[0], str):
            names = np.array([_as_character(v) for v in names], dtype=object)
        self.n = n
        self.names: list[str] = sorted(set(names.tolist()))
        codes = pd.Categorical(names, categories=self.names).codes.astype(np.intp)
        rows = matched["__row__"].to_numpy(dtype=np.intp)
        self._index = (codes, rows)
        self._sources = {
            "fraction": (fraction, matched["__rpos__"].to_numpy(dtype=np.intp)),
            "factors": (factors, matched["__fpos__"].to_numpy(dtype=np.intp)),
        }
        self.present = np.zeros((len(self.names), n), dtype=bool)
        self.present[codes, rows] = True
        self._values: dict[str, np.ndarray] = {}

    def values(self, field: str) -> np.ndarray:
        """``(n_systems, n_cohorts)`` array of ``field`` (``nan`` where not supplied).

        Taken from whichever of the two tables has the column. Like R (where
        ``merge()`` would rename a clashing column to ``.x`` / ``.y``), a column
        present in both tables or in neither is an error.
        """
        if field not in self._values:
            found = [src for src in self._sources.values() if field in src[0].columns]
            if len(found) != 1:
                raise KeyError(
                    f"column '{field}' must be in exactly one of `manure_management_system_fraction` "
                    "and `manure_management_system_factors`"
                )
            table, pos = found[0]
            try:
                col = as_float(table[field])
            except (TypeError, ValueError):
                if validation_enabled():
                    abort("Each MMS argument must be a numeric vector.")
                raise
            out = np.full((len(self.names), self.n), np.nan)
            out[self._index] = col[pos]
            self._values[field] = out
        return self._values[field]

    def arguments(self, fields: Sequence[str]) -> dict[str, dict[str, np.ma.MaskedArray]]:
        """MMS keyword arguments for the core functions, sorted by system name.

        Each system maps to ``{field: masked array}``; masked elements are the
        cohorts for which the system is not supplied.
        """
        values = {f: self.values(f) for f in fields}
        return {
            name: {f: np.ma.MaskedArray(values[f][i], mask=~self.present[i]) for f in fields}
            for i, name in enumerate(self.names)
        }
