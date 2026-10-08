"""Input validation for the metabolic energy requirement core model.

Port of ``R/validate_metabolic_energy_req_core_model.R``. The R validators are
called once per row with scalars and only check the arguments that the given
species / cohort branch actually uses. Here every validator receives whole
(broadcastable) arrays: each scalar ``if (species ... && cohort ...)`` guard
becomes a boolean mask and the shared checks are applied to the masked
elements only.

Error messages follow the R wording. For an all-scalar call the messages are
identical to R; for vector calls :func:`validate_param_range` reports the
(1-based) position of the first offending element, e.g. ``offtake_rate[3]``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import as_float, as_str, is_true, isna
from ._shared import (
    abort,
    validate_animal_species,
    validate_cohort_code,
    validate_fraction,
    validate_is_egg_producing_flag,
    validate_nonnegative_numeric,
    validate_param_range,
    validate_positive_numeric,
    validate_scalar_numeric,
    validator,
)

_CTL_BFL = ("CTL", "BFL")
_GROWING_4 = ("FS", "FJ", "MS", "MJ")

# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


def _raw(x: Any) -> np.ndarray:
    """Raw input as a numpy array (no type coercion, so type checks still work)."""
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy()
    if x is None:
        return np.array(None, dtype=object)
    return np.asarray(x)


def _prepare(*xs: Any) -> list[np.ndarray]:
    """Broadcast raw inputs to their common shape (``()`` for an all-scalar call)."""
    arrays = [_raw(x) for x in xs]
    shape = np.broadcast_shapes(*(a.shape for a in arrays))
    return [np.broadcast_to(a, shape) for a in arrays]


def _in(x: np.ndarray, values: Iterable[str]) -> np.ndarray:
    """Element-wise ``x %in% values`` for string-code arrays (``None`` never matches)."""
    out = np.zeros(np.shape(x), dtype=bool)
    for v in values:
        out = out | (x == v)
    return out


def _pick(a: np.ndarray, mask: np.ndarray) -> Any:
    """The scalar itself for an all-scalar call, otherwise the masked elements."""
    return a[()] if a.ndim == 0 else a[mask]


def _range(a: np.ndarray, mask: np.ndarray, name: str) -> None:
    """``validate_param_range(x)`` on the elements selected by ``mask``."""
    if not np.any(mask):
        return
    if a.ndim == 0:
        validate_param_range(a[()], name)
    else:
        labels = [int(i) + 1 for i in np.flatnonzero(mask)]
        validate_param_range(a[mask], name, labels=labels)


def _apply(fn: Callable[[Any, str], None], a: np.ndarray, mask: np.ndarray, name: str) -> None:
    """Apply a shared scalar check (``validate_positive_numeric`` ...) to the masked elements."""
    if not np.any(mask):
        return
    fn(_pick(a, mask), name)


def _everywhere(a: np.ndarray) -> np.ndarray:
    return np.ones(a.shape, dtype=bool)


def _birth_before_weaning(birth: np.ndarray, weaning: np.ndarray, mask: np.ndarray) -> None:
    if not np.any(mask):
        return
    if np.any(as_float(_pick(birth, mask)) >= as_float(_pick(weaning, mask))):
        abort("`live_weight_at_birth` must be strictly less than `live_weight_at_weaning`.")


# --------------------------------------------------------------------------
# Validators (one per calc_* function)
# --------------------------------------------------------------------------


@validator
def validate_maintenance_inputs(
    species_short: Any,
    cohort_short: Any,
    live_weight_cohort_average: Any,
    lactating_females_fraction: Any = np.nan,
    offtake_rate: Any = np.nan,
    age_first_parturition: Any = np.nan,
    average_annual_temperature: Any = np.nan,
    lower_critical_temperature: Any = np.nan,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> None:
    """Validate inputs of ``calc_metabolic_energy_req_maintenance``.

    Live weight is always required; the optional arguments are checked only
    for the species / cohort branches that use them. For ``CHK`` the
    egg-producing flag is validated only when ``lower_critical_temperature``
    is supplied (as in R).
    """
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    sp, co, lw, lff, otr, afp, aat, lct, phase, egg = _prepare(
        species_short, cohort_short, live_weight_cohort_average, lactating_females_fraction,
        offtake_rate, age_first_parturition, average_annual_temperature,
        lower_critical_temperature, nondemo_productive_phase_id, is_egg_producing,
    )
    s, c = as_str(sp), as_str(co)
    _apply(validate_positive_numeric, lw, _everywhere(lw), "live_weight_cohort_average")

    cattle = _in(s, _CTL_BFL)
    _range(lff, cattle & (c == "FA"), "lactating_females_fraction")
    _range(otr, cattle & _in(c, ("MA", "MS")), "offtake_rate")

    sheep = s == "SHP"
    _range(afp, sheep & (c == "FS"), "age_first_parturition")
    _range(otr, sheep & _in(c, K.GLEAM_COHORTS_MALE), "offtake_rate")

    chk = s == "CHK"
    _apply(validate_scalar_numeric, aat, chk, "average_annual_temperature")
    with_lct = chk & ~isna(lct)
    _apply(validate_scalar_numeric, lct, with_lct, "lower_critical_temperature")
    if np.any(with_lct):
        validate_is_egg_producing_flag(
            species_short=_pick(sp, with_lct),
            cohort_short=_pick(co, with_lct),
            is_egg_producing=_pick(egg, with_lct),
            nondemo_productive_phase_id=_pick(phase, with_lct),
        )


@validator
def validate_activity_inputs(
    species_short: Any,
    cohort_short: Any,
    metabolic_energy_req_maintenance: Any,
    live_weight_cohort_average: Any,
    low_activity_fraction: Any,
    high_activity_fraction: Any,
) -> None:
    """Validate inputs of ``calc_metabolic_energy_req_activity`` (both fractions and their sum)."""
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    _sp, _co, maint, lw, low, high = _prepare(
        species_short, cohort_short, metabolic_energy_req_maintenance,
        live_weight_cohort_average, low_activity_fraction, high_activity_fraction,
    )
    every = _everywhere(maint)
    _apply(validate_positive_numeric, maint, every, "metabolic_energy_req_maintenance")
    _apply(validate_positive_numeric, lw, every, "live_weight_cohort_average")
    _range(low, every, "low_activity_fraction")
    _range(high, every, "high_activity_fraction")

    activity_sum = as_float(low) + as_float(high)
    if np.any((activity_sum < 0) | (activity_sum > 1)):
        abort("Sum of `low_activity_fraction` + `high_activity_fraction` must be >= 0 and <= 1.")


@validator
def validate_growth_inputs(
    species_short: Any,
    cohort_short: Any,
    live_weight_cohort_average: Any,
    live_weight_cohort_final: Any,
    live_weight_cohort_initial: Any,
    live_weight_mature_stage: Any,
    daily_weight_gain: Any,
    offtake_rate: Any,
    cohort_duration_days: Any,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> None:
    """Validate inputs of ``calc_metabolic_energy_req_growth``.

    Only the arguments used by the species / cohort branch are checked; adult
    cohorts (and FN / MN outside CHK) are not validated.
    """
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    sp, co, lwa, lwf, lwi, lwm, dwg, otr, cdd, phase, egg = _prepare(
        species_short, cohort_short, live_weight_cohort_average, live_weight_cohort_final,
        live_weight_cohort_initial, live_weight_mature_stage, daily_weight_gain, offtake_rate,
        cohort_duration_days, nondemo_productive_phase_id, is_egg_producing,
    )
    validate_is_egg_producing_flag(
        species_short=sp, cohort_short=co, is_egg_producing=egg, nondemo_productive_phase_id=phase
    )
    s, c = as_str(sp), as_str(co)
    growing = _in(c, _GROWING_4)
    male_growing = _in(c, ("MS", "MJ"))

    # --- Cattle and buffalo ---
    m = _in(s, _CTL_BFL) & growing
    _range(lwa, m, "live_weight_cohort_average")
    _range(lwf, m, "live_weight_cohort_final")
    _range(lwi, m, "live_weight_cohort_initial")
    _range(lwm, m, "live_weight_mature_stage")
    _range(dwg, m, "daily_weight_gain")
    _range(cdd, m, "cohort_duration_days")
    _range(otr, m & male_growing, "offtake_rate")
    if np.any(m):
        if np.any(as_float(_pick(lwi, m)) > as_float(_pick(lwa, m))):
            abort("live_weight_cohort_average cannot be lower than live_weight_cohort_initial.")
        if np.any(as_float(_pick(lwa, m)) > as_float(_pick(lwf, m))):
            abort("live_weight_cohort_average cannot be higher than live_weight_cohort_final.")

    # --- Camels: only daily_weight_gain is used ---
    _range(dwg, (s == "CML") & growing, "daily_weight_gain")

    # --- Sheep and goats: linear formula (offtake only for sheep males) ---
    for species, use_offtake in (("SHP", True), ("GTS", False)):
        m = (s == species) & growing
        _range(lwf, m, "live_weight_cohort_final")
        _range(lwi, m, "live_weight_cohort_initial")
        _range(cdd, m, "cohort_duration_days")
        if use_offtake:
            _range(otr, m & male_growing, "offtake_rate")
        if np.any(m) and np.any(as_float(_pick(lwi, m)) > as_float(_pick(lwf, m))):
            abort("live_weight_cohort_final cannot be lower than live_weight_cohort_initial.")

    # --- Pigs: only daily_weight_gain is used ---
    _range(dwg, (s == "PGS") & growing, "daily_weight_gain")

    # --- Chickens: growth can apply to all cohorts ---
    _apply(validate_scalar_numeric, dwg, s == "CHK", "daily_weight_gain")


@validator
def validate_egg_inputs(
    species_short: Any,
    cohort_short: Any,
    cohort_stock_size: Any = np.nan,
    egg_output_human_consumption: Any = np.nan,
    egg_average_weight: Any = np.nan,
    parturition_rate: Any = np.nan,
    nondemo_productive_phase_id: Any = np.nan,
    is_egg_producing: Any = False,
) -> None:
    """Validate inputs of ``calc_metabolic_energy_req_eggs`` (egg inputs only where flagged)."""
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    sp, co, css, eoh, eaw, pr, phase, egg = _prepare(
        species_short, cohort_short, cohort_stock_size, egg_output_human_consumption,
        egg_average_weight, parturition_rate, nondemo_productive_phase_id, is_egg_producing,
    )
    validate_is_egg_producing_flag(
        species_short=sp, cohort_short=co, is_egg_producing=egg, nondemo_productive_phase_id=phase
    )
    laying = is_true(egg)
    _apply(validate_nonnegative_numeric, css, laying, "cohort_stock_size")
    _apply(validate_positive_numeric, eaw, laying, "egg_average_weight")
    _apply(validate_nonnegative_numeric, eoh, laying, "egg_output_human_consumption")
    _apply(validate_nonnegative_numeric, pr, laying, "parturition_rate")


@validator
def validate_lactation_inputs(
    species_short: Any,
    cohort_short: Any,
    lactating_females_fraction: Any,
    milk_yield_day: Any,
    milk_fat_fraction: Any,
    non_productive_duration: Any,
    pregnancy_duration: Any,
    litter_size: Any,
    death_rate_juvenile: Any,
    live_weight_at_birth: Any,
    live_weight_at_weaning: Any,
    lactation_duration: Any,
    parturition_rate: Any,
) -> None:
    """Validate inputs of ``calc_metabolic_energy_req_lactation`` (cohort ``FA`` only)."""
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    sp, co, lff, myd, mff, npd, pd_, ls, drj, lwb, lww, ld, pr = _prepare(
        species_short, cohort_short, lactating_females_fraction, milk_yield_day, milk_fat_fraction,
        non_productive_duration, pregnancy_duration, litter_size, death_rate_juvenile,
        live_weight_at_birth, live_weight_at_weaning, lactation_duration, parturition_rate,
    )
    s, c = as_str(sp), as_str(co)
    fa = c == "FA"

    # --- Cattle, buffalo, camels ---
    m = fa & _in(s, ("CTL", "BFL", "CML"))
    _range(lff, m, "lactating_females_fraction")
    _range(myd, m, "milk_yield_day")
    _range(mff, m, "milk_fat_fraction")
    _range(pr, m, "parturition_rate")
    _range(lwb, m, "live_weight_at_birth")
    _range(lww, m, "live_weight_at_weaning")
    _birth_before_weaning(lwb, lww, m)

    # --- Sheep and goats: plus litter size ---
    m = fa & _in(s, ("SHP", "GTS"))
    _range(lff, m, "lactating_females_fraction")
    _range(myd, m, "milk_yield_day")
    _range(mff, m, "milk_fat_fraction")
    _range(pr, m, "parturition_rate")
    _range(ls, m, "litter_size")
    _range(lwb, m, "live_weight_at_birth")
    _range(lww, m, "live_weight_at_weaning")
    _birth_before_weaning(lwb, lww, m)

    # --- Pigs ---
    m = fa & (s == "PGS")
    _range(ls, m, "litter_size")
    _apply(validate_fraction, drj, m, "death_rate_juvenile")
    _range(lwb, m, "live_weight_at_birth")
    _range(lww, m, "live_weight_at_weaning")
    _apply(validate_positive_numeric, ld, m, "lactation_duration")
    _apply(validate_positive_numeric, npd, m, "non_productive_duration")
    _apply(validate_positive_numeric, pd_, m, "pregnancy_duration")
    _birth_before_weaning(lwb, lww, m)


@validator
def validate_work_inputs(
    species_short: Any,
    cohort_short: Any,
    metabolic_energy_req_maintenance: Any,
    draught_work_hours_female: Any,
    draught_work_hours_male: Any,
    draught_fraction_female: Any,
    draught_fraction_male: Any,
) -> None:
    """Validate inputs of ``calc_metabolic_energy_req_work`` (CTL / BFL / CML adults only)."""
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    sp, co, maint, hours_f, hours_m, frac_f, frac_m = _prepare(
        species_short, cohort_short, metabolic_energy_req_maintenance, draught_work_hours_female,
        draught_work_hours_male, draught_fraction_female, draught_fraction_male,
    )
    s, c = as_str(sp), as_str(co)
    draught = _in(s, ("CTL", "BFL", "CML")) & _in(c, ("MA", "FA"))
    _apply(validate_positive_numeric, maint, draught & _in(s, _CTL_BFL), "metabolic_energy_req_maintenance")
    male = draught & (c == "MA")
    female = draught & (c != "MA")
    _range(hours_m, male, "draught_work_hours_male")
    _range(frac_m, male, "draught_fraction_male")
    _range(hours_f, female, "draught_work_hours_female")
    _range(frac_f, female, "draught_fraction_female")


@validator
def validate_fibre_inputs(species_short: Any, cohort_short: Any, fibre_yield_year: Any) -> None:
    """Validate inputs of ``calc_metabolic_energy_req_fibre``.

    ``fibre_yield_year`` is checked only for SHP / GTS / CML and cohorts
    FA, FS, MA, MS (R does not check FN / MN although they produce fibre).
    """
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    sp, co, fyy = _prepare(species_short, cohort_short, fibre_yield_year)
    s, c = as_str(sp), as_str(co)
    _range(fyy, _in(s, ("SHP", "GTS", "CML")) & _in(c, ("FA", "FS", "MA", "MS")), "fibre_yield_year")


@validator
def validate_pregnancy_inputs(
    species_short: Any,
    cohort_short: Any,
    metabolic_energy_req_maintenance: Any,
    parturition_rate: Any,
    litter_size: Any,
    pregnancy_duration: Any,
    non_productive_duration: Any,
    lactation_duration: Any,
    cohort_duration_days: Any,
    offtake_rate: Any,
) -> None:
    """Validate inputs of ``calc_metabolic_energy_req_pregnancy`` (female cohorts FA / FS)."""
    validate_animal_species(species_short)
    validate_cohort_code(cohort_short)
    sp, co, maint, pr, ls, pd_, npd, ld, cdd, otr = _prepare(
        species_short, cohort_short, metabolic_energy_req_maintenance, parturition_rate,
        litter_size, pregnancy_duration, non_productive_duration, lactation_duration,
        cohort_duration_days, offtake_rate,
    )
    s, c = as_str(sp), as_str(co)
    fa = c == "FA"
    fs = c == "FS"
    females = fa | fs

    # --- Cattle and buffalo ---
    m = females & _in(s, _CTL_BFL)
    _apply(validate_positive_numeric, maint, m, "metabolic_energy_req_maintenance")
    _range(pr, m, "parturition_rate")
    _apply(validate_positive_numeric, pd_, m, "pregnancy_duration")
    _range(cdd, m & fs, "cohort_duration_days")
    _range(otr, m & fs, "offtake_rate")

    # --- Camels ---
    m = females & (s == "CML")
    _apply(validate_positive_numeric, maint, m, "metabolic_energy_req_maintenance")
    _range(pr, m & fa, "parturition_rate")
    _apply(validate_positive_numeric, pd_, m & fs, "pregnancy_duration")
    _range(cdd, m & fs, "cohort_duration_days")
    _range(otr, m & fs, "offtake_rate")

    # --- Sheep and goats ---
    m = females & _in(s, ("SHP", "GTS"))
    _apply(validate_positive_numeric, maint, m, "metabolic_energy_req_maintenance")
    _range(pr, m & fa, "parturition_rate")
    _range(ls, m & fa, "litter_size")
    _apply(validate_positive_numeric, pd_, m, "pregnancy_duration")
    _range(cdd, m & fs, "cohort_duration_days")
    _range(otr, m & fs, "offtake_rate")

    # --- Pigs ---
    m = females & (s == "PGS")
    _range(ls, m, "litter_size")
    _apply(validate_positive_numeric, pd_, m, "pregnancy_duration")
    _apply(validate_positive_numeric, npd, m, "non_productive_duration")
    _apply(validate_positive_numeric, ld, m, "lactation_duration")
    _range(cdd, m & fs, "cohort_duration_days")
    _range(otr, m & fs, "offtake_rate")


@validator
def validate_rem_inputs(species_short: Any, ration_digestibility_fraction: Any) -> None:
    """Validate inputs of ``calc_rem_maintenance`` (digestibility needed for ruminants only)."""
    validate_animal_species(species_short)
    sp, rdf = _prepare(species_short, ration_digestibility_fraction)
    _range(rdf, _in(as_str(sp), K.GLEAM_SPECIES_RUMINANTS), "ration_digestibility_fraction")


@validator
def validate_reg_inputs(species_short: Any, ration_digestibility_fraction: Any) -> None:
    """Validate inputs of ``calc_reg_growth`` (digestibility needed for ruminants only)."""
    validate_animal_species(species_short)
    sp, rdf = _prepare(species_short, ration_digestibility_fraction)
    _range(rdf, _in(as_str(sp), K.GLEAM_SPECIES_RUMINANTS), "ration_digestibility_fraction")


@validator
def validate_total_energy_inputs(
    species_short: Any,
    metabolic_energy_req_maintenance: Any,
    metabolic_energy_req_activity: Any,
    metabolic_energy_req_lactation: Any,
    metabolic_energy_req_work: Any,
    metabolic_energy_req_pregnancy: Any,
    net_energy_maintenance_digestible_energy_ratio: Any,
    metabolic_energy_req_growth: Any,
    metabolic_energy_req_fibre_production: Any,
    metabolic_energy_req_egg_deposition: Any,
    net_energy_growth_digestible_energy_ratio: Any,
    ration_digestibility_fraction: Any,
) -> None:
    """Validate inputs of ``calc_total_metabolic_energy_req``."""
    validate_animal_species(species_short)
    sp, maint, act, lact, work, preg, rem, growth, fibre, egg, reg, rdf = _prepare(
        species_short, metabolic_energy_req_maintenance, metabolic_energy_req_activity,
        metabolic_energy_req_lactation, metabolic_energy_req_work, metabolic_energy_req_pregnancy,
        net_energy_maintenance_digestible_energy_ratio, metabolic_energy_req_growth,
        metabolic_energy_req_fibre_production, metabolic_energy_req_egg_deposition,
        net_energy_growth_digestible_energy_ratio, ration_digestibility_fraction,
    )
    every = _everywhere(maint)
    for a, name in (
        (maint, "metabolic_energy_req_maintenance"),
        (act, "metabolic_energy_req_activity"),
        (lact, "metabolic_energy_req_lactation"),
        (work, "metabolic_energy_req_work"),
        (preg, "metabolic_energy_req_pregnancy"),
        (growth, "metabolic_energy_req_growth"),
        (fibre, "metabolic_energy_req_fibre_production"),
        (egg, "metabolic_energy_req_egg_deposition"),
    ):
        _apply(validate_scalar_numeric, a, every, name)
    _range(rdf, every, "ration_digestibility_fraction")

    ruminant = _in(as_str(sp), K.GLEAM_SPECIES_RUMINANTS)
    _apply(validate_scalar_numeric, rem, ruminant, "net_energy_maintenance_digestible_energy_ratio")
    _apply(validate_scalar_numeric, reg, ruminant, "net_energy_growth_digestible_energy_ratio")


@validator
def validate_dmi_inputs(
    species_short: Any,
    metabolic_energy_req_total: Any,
    ration_gross_energy: Any,
    ration_metabolizable_energy: Any,
) -> None:
    """Validate inputs of ``calc_ration_intake``."""
    validate_animal_species(species_short)
    _sp, total, ge, me = _prepare(
        species_short, metabolic_energy_req_total, ration_gross_energy, ration_metabolizable_energy
    )
    every = _everywhere(total)
    _apply(validate_positive_numeric, total, every, "metabolic_energy_req_total")
    _range(ge, every, "ration_gross_energy")
    _range(me, every, "ration_metabolizable_energy")
