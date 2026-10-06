"""Metabolic energy requirements and dry matter intake module.

Port of ``R/run_metabolic_energy_req_module.R``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import Progress, as_str, copy_frame, is_true, lookup
from ..core.metabolic_energy_req import (
    calc_metabolic_energy_req_activity,
    calc_metabolic_energy_req_eggs,
    calc_metabolic_energy_req_fibre,
    calc_metabolic_energy_req_growth,
    calc_metabolic_energy_req_lactation,
    calc_metabolic_energy_req_maintenance,
    calc_metabolic_energy_req_pregnancy,
    calc_metabolic_energy_req_work,
    calc_ration_intake,
    calc_reg_growth,
    calc_rem_maintenance,
    calc_total_metabolic_energy_req,
)
from ..validation._shared import normalize_optional_is_egg_producing_column, setup_validation, validation_enabled
from ..validation.metabolic_energy_req_run import validate_run_metabolic_energy_req_module_inputs

__all__ = ["run_metabolic_energy_req_module"]


class _LazyInputs:
    """Column access that mimics R's lazy argument evaluation.

    In R every ``calc_*`` argument (``cohort_stock_size``,
    ``herd_level_data[.SD, on = "herd_id", x.col]``, ...) is a promise that is
    only evaluated for the rows whose species / cohort branch (or validator)
    uses it. A column that is absent from the input is therefore an error only
    if some row needs it. Here an absent column is replaced by missing values,
    and a :class:`KeyError` is raised when ``need`` marks a row that would have
    evaluated it in R.
    """

    def __init__(self, cohort: pd.DataFrame, herd: pd.DataFrame) -> None:
        self.cohort_data = cohort
        self.herd_data = herd
        self.n = len(cohort)
        self._cache: dict[str, np.ndarray] = {}

    def _missing(self, kind: str, col: str, need: Any, fill: Any) -> np.ndarray:
        if np.any(need):
            if kind == "herd":
                raise KeyError(
                    f"column 'x.{col}' is not found in `herd_level_data` "
                    "(needed by run_metabolic_energy_req_module)"
                )
            raise KeyError(f"object '{col}' not found in `cohort_level_data`")
        if fill is None:
            return np.full(self.n, None, dtype=object)
        return np.full(self.n, fill, dtype="float64")

    def herd_col(self, col: str, need: Any = True, fill: Any = np.nan) -> np.ndarray:
        """``herd_level_data[.SD, on = "herd_id", x.<col>]`` for every cohort row."""
        if col not in self.herd_data.columns:
            return self._missing("herd", col, need, fill)
        if col not in self._cache:
            self._cache[col] = lookup(self.cohort_data, self.herd_data, col)
        return self._cache[col]

    def cohort_col(self, col: str, need: Any = True, fill: Any = np.nan) -> np.ndarray:
        """A cohort-level column (as raw values, so validation still sees its type)."""
        if col not in self.cohort_data.columns:
            return self._missing("cohort", col, need, fill)
        return self.cohort_data[col].to_numpy()


def _in(x: np.ndarray, values: tuple[str, ...]) -> np.ndarray:
    out = np.zeros(np.shape(x), dtype=bool)
    for v in values:
        out = out | (x == v)
    return out


def run_metabolic_energy_req_module(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> pd.DataFrame:
    """Run the metabolic energy requirements and dry matter intake module.

    Computes cohort-level daily energy requirements (MJ/head/day) and ration
    dry matter intake (kg DM/head/day) with the IPCC Tier 2 energy partition
    functions: maintenance, activity, growth, lactation, work, fibre, egg
    deposition, pregnancy, REM / REG ratios (ruminants), total requirement and
    intake. Energy is net energy for CTL, BFL, SHP, GTS and metabolizable
    energy for CML, PGS and CHK.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        One row per herd / cohort (and non-demographic phase) with
        ``herd_id``, ``cohort_short``, ``live_weight_cohort_average``,
        ``offtake_rate``, ``low_activity_fraction``, ``high_activity_fraction``,
        ``live_weight_cohort_initial``, ``live_weight_cohort_final``,
        ``live_weight_mature_stage``, ``daily_weight_gain``,
        ``cohort_duration_days``, ``ration_digestibility_fraction``,
        ``ration_gross_energy``, ``ration_metabolizable_energy`` and, when
        used, ``cohort_stock_size``, ``nondemo_productive_phase_id`` and
        ``is_egg_producing`` (CHK).
    herd_level_data : pandas.DataFrame
        One row per herd with ``herd_id``, ``species_short`` and the
        reproduction, milk, draught, fibre and (CHK) temperature / egg
        parameters (see the R documentation).
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default). ``False`` skips all checks.

    Returns
    -------
    pandas.DataFrame
        ``cohort_level_data`` (plus ``is_egg_producing`` when it had to be
        added) with the new columns ``metabolic_energy_req_maintenance``,
        ``metabolic_energy_req_activity``, ``metabolic_energy_req_growth``,
        ``metabolic_energy_req_lactation``, ``metabolic_energy_req_work``,
        ``metabolic_energy_req_fibre_production``,
        ``metabolic_energy_req_egg_deposition``,
        ``metabolic_energy_req_pregnancy``,
        ``net_energy_maintenance_digestible_energy_ratio``,
        ``net_energy_growth_digestible_energy_ratio``,
        ``metabolic_energy_req_total`` and ``ration_intake``; row order unchanged.
    """
    with setup_validation(validate_inputs):
        # Working copies (R validates the input table itself, which may add the
        # optional is_egg_producing column by reference; the copy avoids that).
        cohort = copy_frame(cohort_level_data)
        herd = copy_frame(herd_level_data)

        # --- Step 1: Validate inputs ---
        validate_run_metabolic_energy_req_module_inputs(cohort, herd)
        # Done inside the validator in R; repeated (idempotent) so the output
        # has the same columns when validation is switched off.
        normalize_optional_is_egg_producing_column(cohort, herd)

        progress = Progress(show_indicator)
        progress.status("Calculating metabolic energy requirements and ration, please wait...")

        V = validation_enabled()
        n = len(cohort)
        every = np.ones(n, dtype=bool)
        cols = _LazyInputs(cohort, herd)

        species = cols.herd_col("species_short", every, fill=None)
        cohort_short = cols.cohort_col("cohort_short", every, fill=None)
        sp, co = as_str(species), as_str(cohort_short)

        cattle = _in(sp, ("CTL", "BFL"))
        camel = sp == "CML"
        sheep = sp == "SHP"
        goat = sp == "GTS"
        pig = sp == "PGS"
        chk = sp == "CHK"
        ruminant = _in(sp, K.GLEAM_SPECIES_RUMINANTS)
        milk_species = cattle | camel | sheep | goat
        fa, fs, fn = co == "FA", co == "FS", co == "FN"
        ma, ms = co == "MA", co == "MS"
        growing4 = _in(co, ("FS", "FJ", "MS", "MJ"))
        growing6 = _in(co, ("FS", "FJ", "MS", "MJ", "FN", "MN"))

        egg_flag = cols.cohort_col("is_egg_producing", V | (chk & fn), fill=None)
        egg = is_true(egg_flag)
        egg_rows = (V & egg) | (chk & (fa | (fn & egg)))  # rows evaluating the egg inputs
        phase = cols.cohort_col("nondemo_productive_phase_id", V & chk & egg & fn)

        lw_avg = cols.cohort_col("live_weight_cohort_average", every)
        offtake = cols.cohort_col("offtake_rate", every)
        parturition_rate = cols.herd_col(
            "parturition_rate",
            (fa & milk_species) | egg_rows | (V & cattle & fs),
        )
        pregnancy_duration = cols.herd_col(
            "pregnancy_duration",
            (fa & (cattle | sheep | goat | pig)) | (fs & (milk_species | pig)),
        )
        litter_size = cols.herd_col("litter_size", (fa & (sheep | goat)) | (pig & (fa | fs)))
        non_productive_duration = cols.herd_col("non_productive_duration", pig & (fa | (V & fs)))
        lactation_duration = cols.herd_col("lactation_duration", pig & (fa | (V & fs)))
        cohort_duration_days = cols.cohort_col(
            "cohort_duration_days",
            (V & (cattle | sheep | goat) & growing4) | sheep | goat | (fs & (milk_species | pig)),
        )

        # --- Step 3: Maintenance energy (MJ/day) ---
        cohort["metabolic_energy_req_maintenance"] = calc_metabolic_energy_req_maintenance(
            species_short=species,
            cohort_short=cohort_short,
            live_weight_cohort_average=lw_avg,
            nondemo_productive_phase_id=phase,
            lactating_females_fraction=cols.herd_col("lactating_females_fraction", fa & milk_species),
            offtake_rate=offtake,
            age_first_parturition=cols.herd_col("age_first_parturition", sheep & (fs | ms)),
            average_annual_temperature=cols.herd_col("average_annual_temperature", chk),
            lower_critical_temperature=np.where(chk, K.GLEAM_CHK_LOWER_CRITICAL_TEMPERATURE, np.nan),
            is_egg_producing=egg_flag,
        )
        maintenance = cohort["metabolic_energy_req_maintenance"].to_numpy()

        # --- Step 4: Activity energy (MJ/day) ---
        cohort["metabolic_energy_req_activity"] = calc_metabolic_energy_req_activity(
            species_short=species,
            cohort_short=cohort_short,
            metabolic_energy_req_maintenance=maintenance,
            live_weight_cohort_average=lw_avg,
            low_activity_fraction=cols.cohort_col("low_activity_fraction", every),
            high_activity_fraction=cols.cohort_col("high_activity_fraction", every),
        )

        # --- Step 5: Growth energy (MJ/day) ---
        weights_need = (V & (cattle | sheep | goat) & growing4) | sheep | goat
        cohort["metabolic_energy_req_growth"] = calc_metabolic_energy_req_growth(
            species_short=species,
            cohort_short=cohort_short,
            nondemo_productive_phase_id=phase,
            live_weight_cohort_average=lw_avg,
            live_weight_cohort_final=cols.cohort_col("live_weight_cohort_final", weights_need),
            live_weight_cohort_initial=cols.cohort_col("live_weight_cohort_initial", weights_need),
            live_weight_mature_stage=cols.cohort_col(
                "live_weight_mature_stage", (V & cattle & growing4) | (cattle & growing6)
            ),
            daily_weight_gain=cols.cohort_col(
                "daily_weight_gain", ((cattle | camel | pig) & growing6) | chk
            ),
            offtake_rate=offtake,
            cohort_duration_days=cohort_duration_days,
            is_egg_producing=egg_flag,
        )

        # --- Step 6: Lactation energy (MJ/day) ---
        cohort["metabolic_energy_req_lactation"] = calc_metabolic_energy_req_lactation(
            species_short=species,
            cohort_short=cohort_short,
            lactating_females_fraction=cols.herd_col("lactating_females_fraction", fa & milk_species),
            milk_yield_day=cols.herd_col("milk_yield_day", fa & milk_species),
            milk_fat_fraction=cols.herd_col("milk_fat_fraction", fa & ((V & milk_species) | cattle)),
            non_productive_duration=non_productive_duration,
            pregnancy_duration=pregnancy_duration,
            litter_size=litter_size,
            death_rate_juvenile=cols.herd_col("death_rate_juvenile", fa & pig),
            live_weight_at_birth=cols.herd_col("live_weight_at_birth", fa & (milk_species | pig)),
            live_weight_at_weaning=cols.herd_col("live_weight_at_weaning", fa & (milk_species | pig)),
            lactation_duration=lactation_duration,
            parturition_rate=parturition_rate,
        )

        # --- Step 7: Work energy (MJ/day) ---
        draught = cattle | camel
        cohort["metabolic_energy_req_work"] = calc_metabolic_energy_req_work(
            species_short=species,
            cohort_short=cohort_short,
            metabolic_energy_req_maintenance=maintenance,
            draught_work_hours_female=cols.herd_col("draught_work_hours_female", draught & fa),
            draught_work_hours_male=cols.herd_col("draught_work_hours_male", draught & ma),
            draught_fraction_female=cols.herd_col("draught_fraction_female", draught & fa),
            draught_fraction_male=cols.herd_col("draught_fraction_male", draught & ma),
        )

        # --- Step 8: Fibre production energy (MJ/day) ---
        cohort["metabolic_energy_req_fibre_production"] = calc_metabolic_energy_req_fibre(
            species_short=species,
            cohort_short=cohort_short,
            fibre_yield_year=cols.herd_col(
                "fibre_yield_year",
                (sheep | goat | camel) & _in(co, ("FA", "FS", "MA", "MS", "FN", "MN")),
            ),
        )

        # --- Step 10: Egg deposition energy (MJ/day) ---
        cohort["metabolic_energy_req_egg_deposition"] = calc_metabolic_energy_req_eggs(
            species_short=species,
            cohort_short=cohort_short,
            cohort_stock_size=cols.cohort_col("cohort_stock_size", egg_rows),
            egg_output_human_consumption=cols.herd_col("egg_output_human_consumption", egg_rows),
            egg_average_weight=cols.herd_col("egg_average_weight", egg_rows),
            parturition_rate=parturition_rate,
            nondemo_productive_phase_id=phase,
            is_egg_producing=egg_flag,
        )

        # --- Step 11: Pregnancy energy (MJ/day) ---
        cohort["metabolic_energy_req_pregnancy"] = calc_metabolic_energy_req_pregnancy(
            species_short=species,
            cohort_short=cohort_short,
            metabolic_energy_req_maintenance=maintenance,
            parturition_rate=parturition_rate,
            litter_size=litter_size,
            pregnancy_duration=pregnancy_duration,
            non_productive_duration=non_productive_duration,
            lactation_duration=lactation_duration,
            cohort_duration_days=cohort_duration_days,
            offtake_rate=offtake,
        )

        # --- Step 12: Diet NE fractions ---
        digestibility = cols.cohort_col("ration_digestibility_fraction", V | ruminant)
        cohort["net_energy_maintenance_digestible_energy_ratio"] = calc_rem_maintenance(
            species_short=species, ration_digestibility_fraction=digestibility
        )
        cohort["net_energy_growth_digestible_energy_ratio"] = calc_reg_growth(
            species_short=species, ration_digestibility_fraction=digestibility
        )

        # --- Step 13: Total requirement (MJ/day) ---
        cohort["metabolic_energy_req_total"] = calc_total_metabolic_energy_req(
            species_short=species,
            metabolic_energy_req_maintenance=maintenance,
            metabolic_energy_req_activity=cohort["metabolic_energy_req_activity"].to_numpy(),
            metabolic_energy_req_lactation=cohort["metabolic_energy_req_lactation"].to_numpy(),
            metabolic_energy_req_work=cohort["metabolic_energy_req_work"].to_numpy(),
            metabolic_energy_req_pregnancy=cohort["metabolic_energy_req_pregnancy"].to_numpy(),
            net_energy_maintenance_digestible_energy_ratio=cohort[
                "net_energy_maintenance_digestible_energy_ratio"
            ].to_numpy(),
            metabolic_energy_req_growth=cohort["metabolic_energy_req_growth"].to_numpy(),
            metabolic_energy_req_fibre_production=cohort["metabolic_energy_req_fibre_production"].to_numpy(),
            metabolic_energy_req_egg_deposition=cohort["metabolic_energy_req_egg_deposition"].to_numpy(),
            net_energy_growth_digestible_energy_ratio=cohort[
                "net_energy_growth_digestible_energy_ratio"
            ].to_numpy(),
            ration_digestibility_fraction=digestibility,
        )

        # --- Step 14: Dry matter intake (kg DM/day) ---
        cohort["ration_intake"] = calc_ration_intake(
            species_short=species,
            metabolic_energy_req_total=cohort["metabolic_energy_req_total"].to_numpy(),
            ration_gross_energy=cols.cohort_col("ration_gross_energy", V | ruminant),
            ration_metabolizable_energy=cols.cohort_col(
                "ration_metabolizable_energy", V | pig | camel | chk
            ),
        )

        progress.success("Metabolic energy requirements calculation complete.")
        return cohort
