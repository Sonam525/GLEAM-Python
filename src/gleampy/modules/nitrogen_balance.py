"""Nitrogen balance module (port of ``R/run_nitrogen_balance_module.R``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import Lookup, Progress, as_str, copy_frame, is_true, isin, isna
from ..core.nitrogen_balance import (
    calc_nitrogen_excretion,
    calc_nitrogen_intake,
    calc_nitrogen_retention,
)
from ..validation._shared import (
    abort,
    normalize_optional_is_egg_producing_column,
    setup_validation,
    validation_enabled,
)
from ..validation.nitrogen_balance_run import validate_run_nitrogen_balance_module_inputs


class _Columns:
    """Input columns, with R's lazy evaluation of absent columns.

    In R every argument of ``calc_nitrogen_retention()`` is a promise
    (``herd_level_data[.SD, on = "herd_id", x.col]`` or a cohort column) that
    is only evaluated for the rows whose species / cohort branch, or
    validator, uses it. An absent column is therefore an error only when some
    row needs it. :meth:`require` raises a :class:`GleamValidationError`
    ("Missing required columns in ...") naming the absent columns that some
    row needs (where R stops with "object not found" or with data.table's
    "column name ... is not found"), with or without validation; the
    accessors then return missing values only for absent columns that no row
    needs. A needed column is never silently replaced by ``NA``.
    """

    def __init__(self, cohort: pd.DataFrame, herd: pd.DataFrame) -> None:
        self.cohort = cohort
        self.herd = herd
        self.n = len(cohort)
        self._join: Lookup | None = None

    def require(self, arg: str, needs: dict[str, Any]) -> None:
        """Abort if a column of ``needs`` (``{column: rows needing it}``) is absent but needed."""
        table = self.herd if arg == "herd_level_data" else self.cohort
        missing = [c for c, need in needs.items() if c not in table.columns and np.any(need)]
        if missing:
            abort(f"Missing required columns in `{arg}`: " + ", ".join(f'"{c}"' for c in missing))

    def _absent(self, fill: Any) -> np.ndarray:
        return np.full(self.n, fill, dtype=object if fill is None else "float64")

    def herd_col(self, col: str, fill: Any = np.nan) -> np.ndarray:
        """``herd_level_data[.SD, on = "herd_id", x.<col>]`` for every cohort row."""
        if col not in self.herd.columns:
            return self._absent(fill)
        if self._join is None:
            self._join = Lookup(self.cohort, self.herd, "herd_id")
        return self._join(col)

    def cohort_col(self, col: str, fill: Any = np.nan) -> Any:
        """A cohort-level column (the Series itself, so validation sees its type)."""
        if col not in self.cohort.columns:
            return self._absent(fill)
        return self.cohort[col]


def run_nitrogen_balance_module(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> pd.DataFrame:
    """Run the nitrogen balance module.

    Computes cohort-level daily nitrogen intake, retention and excretion
    (kg N/head/day) following the IPCC Tier 2 structure.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        Cohort-level table: ``herd_id``, ``cohort_short``, ``ration_intake``
        (kg DM/head/day), ``ration_nitrogen`` (kg N/kg DM),
        ``daily_weight_gain`` (kg/head/day), ``cohort_duration_days`` (days),
        ``cohort_stock_size`` (heads) and, optionally,
        ``nondemo_productive_phase_id`` (1 or 2 for ``FN`` / ``MN`` rows) and
        ``is_egg_producing`` (logical; required when there are CHK herds).
    herd_level_data : pandas.DataFrame
        Herd-level table (one row per ``herd_id``): ``species_short``,
        ``milk_protein_fraction`` (kg protein/kg milk), ``milk_yield_day``
        (kg/head/day), ``fibre_yield_year`` (kg/head/year), ``litter_size``
        (# offspring/parturition), ``parturition_rate`` (# parturitions/adult
        female/year; reproductive eggs/hen/year for CHK),
        ``live_weight_at_weaning`` and ``live_weight_at_birth`` (kg),
        ``pregnancy_duration`` (days) and, for laying CHK cohorts,
        ``egg_output_human_consumption`` (eggs/year) and
        ``egg_average_weight`` (kg/egg). As in R, a column is needed when a
        row uses it, whatever its cohort: the milk and fibre columns for every
        row of a milk-producing species (CTL, BFL, SHP, GTS, CML, ``FN`` /
        ``MN`` included), the PGS reproduction columns for PGS ``FA`` /
        ``FS``, the egg columns for laying CHK rows and, with validation on,
        ``live_weight_at_birth`` (and ``live_weight_at_weaning`` where the
        birth weight is given) for every non-CHK row. Values may be NA.
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    pandas.DataFrame
        The cohort-level input columns (plus ``is_egg_producing = NA`` when it
        was absent and there are no CHK herds) and ``nitrogen_intake``,
        ``nitrogen_retention`` and ``nitrogen_excretion`` (kg N/head/day).

    Raises
    ------
    GleamValidationError
        For invalid inputs (with validation on) and, with or without
        validation, when a column that some row needs is absent ("Missing
        required columns in `herd_level_data`: ...").

    Notes
    -----
    Species are taken from ``herd_level_data`` (joined by ``herd_id``).
    Steps: :func:`gleampy.calc_nitrogen_intake`,
    :func:`gleampy.calc_nitrogen_retention`, :func:`gleampy.calc_nitrogen_excretion`.
    """
    with setup_validation(validate_inputs):
        # as.data.table() + copy(): never modify the caller's tables.
        cohort = copy_frame(cohort_level_data)
        herd = copy_frame(herd_level_data)

        # --- Step 1: validate inputs (adds the optional is_egg_producing column)
        validate_run_nitrogen_balance_module_inputs(cohort, herd)
        normalize_optional_is_egg_producing_column(cohort, herd)

        progress = Progress(show_indicator)
        progress.status("Calculating nitrogen balance, please wait...")

        # --- Rows that evaluate each input in R (validator and calc body)
        V = validation_enabled()
        every = np.ones(len(cohort), dtype=bool)
        cols = _Columns(cohort, herd)
        cols.require("herd_level_data", {"species_short": every})
        cols.require("cohort_level_data", {"cohort_short": every})
        species = cols.herd_col("species_short", fill=None)
        cohort_short = cols.cohort_col("cohort_short", fill=None)
        sp, co = as_str(species), as_str(cohort_short)
        milk = isin(sp, K.GLEAM_SPECIES_MILK_PRODUCERS)
        pgs = sp == "PGS"
        chk = sp == "CHK"
        non_chk = ~isna(sp) & ~chk
        fa, fs, fn = co == "FA", co == "FS", co == "FN"
        pgs_repro = pgs & (fa | fs)
        # R's validator reads the flag on every row, the calc body on CHK rows.
        cols.require("cohort_level_data", {"is_egg_producing": V | chk})
        is_egg_producing = cols.cohort_col("is_egg_producing", fill=None)
        laying = chk & is_true(is_egg_producing)
        live_weight_at_birth = cols.herd_col("live_weight_at_birth")
        cols.require(
            "cohort_level_data",
            {
                "ration_intake": every,
                "ration_nitrogen": every,
                "daily_weight_gain": ~(pgs & fa),
                "cohort_duration_days": pgs & fs,
                "cohort_stock_size": laying,
                "nondemo_productive_phase_id": V & laying & fn,
            },
        )
        cols.require(
            "herd_level_data",
            {
                "milk_protein_fraction": milk,
                "milk_yield_day": milk,
                "fibre_yield_year": milk,
                "litter_size": pgs_repro,
                "parturition_rate": (pgs & (fa | (V & fs))) | laying,
                # birth < weaning check of R's validator: every non-CHK row
                "live_weight_at_birth": pgs_repro | (V & non_chk),
                "live_weight_at_weaning": (
                    (pgs & (fa | (V & fs))) | (V & non_chk & ~isna(live_weight_at_birth))
                ),
                "pregnancy_duration": pgs & fs,
                "egg_output_human_consumption": laying,
                "egg_average_weight": laying,
            },
        )

        # --- Step 3: intake - N consumed per head/day
        cohort["nitrogen_intake"] = calc_nitrogen_intake(
            ration_intake=cols.cohort_col("ration_intake"),
            ration_nitrogen=cols.cohort_col("ration_nitrogen"),
        )

        # --- Step 4: retention - N in growth, milk, reproduction, fibre, eggs
        cohort["nitrogen_retention"] = calc_nitrogen_retention(
            species_short=species,
            cohort_short=cohort_short,
            nondemo_productive_phase_id=cols.cohort_col("nondemo_productive_phase_id"),
            milk_protein_fraction=cols.herd_col("milk_protein_fraction"),
            milk_yield_day=cols.herd_col("milk_yield_day"),
            daily_weight_gain=cols.cohort_col("daily_weight_gain"),
            fibre_yield_year=cols.herd_col("fibre_yield_year"),
            litter_size=cols.herd_col("litter_size"),
            parturition_rate=cols.herd_col("parturition_rate"),
            live_weight_at_weaning=cols.herd_col("live_weight_at_weaning"),
            live_weight_at_birth=live_weight_at_birth,
            pregnancy_duration=cols.herd_col("pregnancy_duration"),
            cohort_duration_days=cols.cohort_col("cohort_duration_days"),
            cohort_stock_size=cols.cohort_col("cohort_stock_size"),
            egg_output_human_consumption=cols.herd_col("egg_output_human_consumption"),
            egg_average_weight=cols.herd_col("egg_average_weight"),
            is_egg_producing=is_egg_producing,
        )

        # --- Step 5: excretion - N lost (intake - retention)
        cohort["nitrogen_excretion"] = calc_nitrogen_excretion(
            species_short=species,
            nitrogen_intake=cohort["nitrogen_intake"],
            nitrogen_retention=cohort["nitrogen_retention"],
        )

        progress.success("Nitrogen balance calculation complete.")
        return cohort
