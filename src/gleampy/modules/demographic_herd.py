"""Demographic herd module runner (port of ``R/run_demographic_herd_module.R``).

Estimates, for every herd, a steady-state sex-age herd structure on a daily
time step (Dynmod *STEADY1*-like), the annual herd growth rate, cohort stocks,
offtake and the annual number of animals diverted to the non-demographic
block. The R function loops over herds; here every step is evaluated for all
herds at once (see :mod:`gleampy.core.demographic_herd`), reproducing R's
results, columns and row order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .._utils import Lookup, Progress, as_float, copy_frame, is_na, lookup, order_by
from ..core.all_herd import rescale_x_to_y
from ..core.demographic_herd import (
    EIGHT_COHORTS,
    SIX_COHORTS,
    TEN_COHORTS,
    _bump_zero_hazard,
    _NAConditionError,
    _get,
    _projection_kernel,
    _steady_state_kernel,
    _summary_offtake_kernel,
    _transition_kernel,
    calc_fecundity_rates,
)
from ..validation._shared import (
    GleamValidationError,
    abort,
    setup_validation,
    validate_named_numeric_vector,
    validate_param_range,
    validate_scalar_numeric,
    validation_enabled,
)
from ..validation.demographic_herd_core import (
    _single_numeric,
    validate_fecundity_inputs,
    validate_population_size_inputs,
    validate_steady_state_inputs,
    validate_transition_inputs,
)
from ..validation.demographic_herd_run import validate_run_demographic_herd_module_inputs
from ..validation.nondemographic_herd_run import is_numeric_column

#: Default ``initial_herd_structure`` of the R function.
DEFAULT_INITIAL_HERD_STRUCTURE: dict[str, float] = {
    "FJ": 100.0, "FS": 50.0, "FA": 30.0, "MJ": 100.0, "MS": 50.0, "MA": 30.0,
}


def _safe_float(x: Any) -> np.ndarray:
    """``as_float`` that maps non-numeric entries to NaN (they fail validation later)."""
    try:
        return as_float(x)
    except (TypeError, ValueError):
        arr = np.asarray(x, dtype=object)
        out = np.full(arr.shape, np.nan)
        for i, v in enumerate(arr.ravel()):
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)):
                out.ravel()[i] = float(v)
        return out


def _c_numeric(x: Any) -> np.ndarray | None:
    """A herd column as an element of R's ``c(FJ = x, FS = 0, FA = 0, ...)``.

    ``c()`` with the numeric zeros keeps a numeric column, coerces a logical
    one to numeric (``TRUE`` -> 1, ``NA`` -> ``NA``; fread reads an all-empty
    column as logical) and turns the whole vector into character for a
    character column. Returns the float64 values, or ``None`` for a
    character (non-numeric) column.
    """
    arr = np.asarray(x)
    if arr.dtype.kind in "iufb":
        return arr.astype("float64")
    flat = arr.reshape(-1)
    if all(is_na(v) or isinstance(v, (bool, int, float, np.bool_, np.number)) for v in flat):
        return np.array([np.nan if is_na(v) else float(v) for v in flat], dtype="float64").reshape(arr.shape)
    return None


def _fmt_id(v: Any) -> str:
    """A herd id as R prints it (``1``, not ``np.int64(1)`` or ``1.0``)."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _check_stage(batch: Callable[[], Any], per_herd: Callable[[int], None], n: int) -> Any:
    """Run a vectorised validation (or validated computation) for all herds.

    On failure the per-herd validator is replayed herd by herd (R's processing
    order) so the raised message is the one R gives for the first failing herd.
    """
    try:
        return batch()
    except GleamValidationError:
        for i in range(n):
            per_herd(i)
        raise


def run_demographic_herd_module(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
    initial_herd_structure: Mapping[str, float] | pd.Series | None = None,
    max_simulation_years: float = 100,
    min_lambda_change: float = 1e-9,
    show_indicator: bool = True,
    simulation_duration: float = 365,
    validate_inputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the demographic herd module.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        One row per herd and demographic cohort with ``herd_id``, ``cohort_short``
        (``FJ, FS, FA, MJ, MS, MA``), ``cohort_duration_days`` (days),
        ``offtake_rate`` and ``death_rate`` (annual fractions).
    herd_level_data : pandas.DataFrame
        One row per herd with ``herd_id``, ``parturition_rate``, ``litter_size``,
        ``birth_fraction_female``, ``herd_size_total`` (heads at the start of the
        year), ``prop_nondemo_fem_juv`` and ``prop_nondemo_mal_juv`` (fractions
        of juveniles diverted to the non-demographic block at FJ->FS / MJ->MS).
    initial_herd_structure : dict or pandas.Series, optional
        Starting heads for the steady-state iteration (default
        ``FJ = 100, FS = 50, FA = 30, MJ = 100, MS = 50, MA = 30``).
    max_simulation_years : float
        Maximum number of simulated years (years, default 100).
    min_lambda_change : float
        Convergence threshold on the day-to-day change of the cohort growth
        ratios (dimensionless, default 1e-9).
    show_indicator : bool
        Print progress messages.
    simulation_duration : float
        Length of the assessment period (days); a single value.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    dict
        ``cohort_level_results``: the input cohort rows sorted by ``herd_id`` and
        ``cohort_short`` plus ``cohort_stock_size_unscaled`` (average stock),
        ``cohort_stock_size_scaled`` (start stock), ``cohort_stock_annual_nondemographic``,
        ``offtake_heads_unscaled``, ``offtake_heads_assessment_unscaled``,
        ``offtake_heads_scaled`` and ``offtake_heads_assessment_scaled``;
        ``herd_level_results``: the herd rows sorted by ``herd_id`` plus
        ``growth_rate_herd``. All results are float64 (never truncated to the
        dtype of a pre-existing integer column).

    Raises
    ------
    GleamValidationError
        On invalid inputs, and (like R's "missing value where TRUE/FALSE
        needed" stop) when a herd's steady-state iteration hits a NaN growth
        ratio: a cohort holding zero animals, or a juvenile (FJ / MJ)
        ``cohort_duration_days`` of 1 day. The message names the cause and the herd.
    """
    if initial_herd_structure is None:
        initial_herd_structure = dict(DEFAULT_INITIAL_HERD_STRUCTURE)

    with setup_validation(validate_inputs):
        validate_run_demographic_herd_module_inputs(cohort_level_data, herd_level_data)

        progress = Progress(show_indicator)
        progress.status("Simulating the herd structure, please wait…")

        cohort_level_results = order_by(copy_frame(cohort_level_data), ["herd_id", "cohort_short"])
        herd_level_results = order_by(copy_frame(herd_level_data), ["herd_id"])

        herd_ids = pd.unique(cohort_level_results["herd_id"])
        n = len(herd_ids)
        keys = pd.DataFrame({"herd_id": herd_ids})
        # one join for all herd columns; R selects herd_level_data[herd_id == h],
        # so an NA herd id matches no herd row (R then aborts on parturition_rate)
        herd_lookup = Lookup(keys, herd_level_results, na_matches=False)

        def herd_col(col: str) -> np.ndarray:
            v = herd_lookup(col)
            return v if v.dtype == object and validation_enabled() else as_float(v)

        parturition_rate = herd_col("parturition_rate")
        litter_size = herd_col("litter_size")
        birth_fraction_female = herd_col("birth_fraction_female")
        herd_size_total = herd_col("herd_size_total")
        prop_fem = herd_col("prop_nondemo_fem_juv")
        prop_mal = herd_col("prop_nondemo_mal_juv")
        # proportion_nondemographic is c(FJ = prop_fem, FS = 0, FA = 0, MJ = prop_mal, ...) in R
        prop_fem_num = _c_numeric(prop_fem)
        prop_mal_num = _c_numeric(prop_mal)

        # One join of the (cohort x herd) grid into the cohort rows, for all cohort columns
        grid = pd.DataFrame({
            "herd_id": pd.concat([keys["herd_id"]] * len(SIX_COHORTS), ignore_index=True),
            "cohort_short": np.repeat(np.array(SIX_COHORTS, dtype=object), n),
        })
        cohort_lookup = Lookup(grid, cohort_level_results, on=["herd_id", "cohort_short"])

        def cohort_col(col: str) -> dict[str, np.ndarray]:
            vals = as_float(cohort_lookup(col)).reshape(len(SIX_COHORTS), n)
            return {c: vals[j] for j, c in enumerate(SIX_COHORTS)}

        # ---- Fecundity -----------------------------------------------------
        fec = _check_stage(
            lambda: calc_fecundity_rates(parturition_rate, litter_size, birth_fraction_female),
            lambda i: validate_fecundity_inputs(parturition_rate[i], litter_size[i], birth_fraction_female[i]),
            n,
        )
        fec_f = np.atleast_1d(as_float(fec["fecundity_female"]))
        fec_m = np.atleast_1d(as_float(fec["fecundity_male"]))

        # ---- Transition probabilities ---------------------------------------
        herd_pos = pd.Index(herd_ids).get_indexer(cohort_level_results["herd_id"])

        def herd_vector(i: int, col: str) -> pd.Series:
            rows = cohort_level_results[herd_pos == i]
            return pd.Series(rows[col].to_numpy(), index=rows["cohort_short"].to_numpy())

        if validation_enabled():
            def batch_transition() -> None:
                counts = np.bincount(herd_pos[herd_pos >= 0], minlength=n)
                for col in ("cohort_duration_days", "offtake_rate", "death_rate"):
                    if (counts != 6).any() or not is_numeric_column(cohort_level_results[col]):
                        abort(f"`{col}` must be a numeric vector of length 6 with names.")
                labels = cohort_level_results["cohort_short"].tolist()
                for col in ("cohort_duration_days", "death_rate", "offtake_rate"):
                    validate_param_range(as_float(cohort_level_results[col]), col, labels=labels)

            _check_stage(
                batch_transition,
                lambda i: validate_transition_inputs(
                    herd_vector(i, "cohort_duration_days"), herd_vector(i, "offtake_rate"),
                    herd_vector(i, "death_rate"),
                ),
                n,
            )

        duration = cohort_col("cohort_duration_days")
        offtake = cohort_col("offtake_rate")
        death = cohort_col("death_rate")
        death = {c: _bump_zero_hazard(offtake[c], death[c]) for c in SIX_COHORTS}
        trans = _transition_kernel(duration, offtake, death)
        p_death = trans["probability_death"]
        p_off = trans["probability_offtake"]
        p_growth = trans["probability_growth"]

        zeros = np.zeros(n)
        prop_nondemo = {
            "FJ": _safe_float(prop_fem) if prop_fem_num is None else prop_fem_num, "FS": zeros, "FA": zeros,
            "MJ": _safe_float(prop_mal) if prop_mal_num is None else prop_mal_num, "MS": zeros, "MA": zeros,
        }

        def herd_named(d: Mapping[str, np.ndarray], i: int, names) -> pd.Series:
            return pd.Series([float(np.broadcast_to(d[c], (n,))[i]) for c in names], index=list(names))

        def prop_named(i: int) -> pd.Series:
            # raw values of a character column, so the per-herd check sees R's character vector
            fem = prop_fem[i] if prop_fem_num is None else prop_fem_num[i]
            mal = prop_mal[i] if prop_mal_num is None else prop_mal_num[i]
            return pd.Series([fem, 0.0, 0.0, mal, 0.0, 0.0], index=list(SIX_COHORTS), dtype=object)

        # ---- Steady-state structure -----------------------------------------
        if validation_enabled():
            def batch_steady() -> None:
                validate_named_numeric_vector(initial_herd_structure, 6, SIX_COHORTS, "initial_herd_structure")
                if prop_fem_num is None or prop_mal_num is None:
                    abort("`proportion_nondemographic` must be a numeric vector of length 6 with names.")
                _single_numeric(max_simulation_years, "max_simulation_years")
                _single_numeric(min_lambda_change, "min_lambda_change")
                validate_scalar_numeric(fec_f, "fecundity_female")
                validate_scalar_numeric(fec_m, "fecundity_male")
                pn_mat = np.stack([np.broadcast_to(as_float(prop_nondemo[c]), (n,)) for c in SIX_COHORTS], axis=1)
                validate_param_range(pn_mat.reshape(-1), "proportion_nondemographic", labels=list(SIX_COHORTS) * n)

            _check_stage(
                batch_steady,
                lambda i: validate_steady_state_inputs(
                    initial_herd_structure, max_simulation_years, min_lambda_change, fec_f[i], fec_m[i],
                    herd_named(p_death, i, TEN_COHORTS), herd_named(p_off, i, TEN_COHORTS),
                    herd_named(p_growth, i, TEN_COHORTS), prop_named(i),
                ),
                n,
            )

        try:
            structure = _steady_state_kernel(
                {c: _get(initial_herd_structure, c) for c in SIX_COHORTS},
                float(np.asarray(max_simulation_years).reshape(-1)[0]),
                float(np.asarray(min_lambda_change).reshape(-1)[0]),
                fec_f, fec_m, p_death, p_off, p_growth, prop_nondemo,
            )
        except _NAConditionError as err:
            if err.herd_index is None:
                raise
            msg = f"{str(err).rstrip('.')} (herd_id {_fmt_id(herd_ids[err.herd_index])})."
            raise _NAConditionError(msg, err.herd_index) from None
        growth_rate_herd = structure["growth_rate_herd"]
        herd_structure = {c: structure["herd_structure"][:, j] for j, c in enumerate(EIGHT_COHORTS)}

        # ---- One-year projection --------------------------------------------
        if validation_enabled():
            def batch_projection() -> None:
                validate_scalar_numeric(herd_size_total, "herd_size_total")
                validate_scalar_numeric(fec_f, "fecundity_female")
                validate_scalar_numeric(fec_m, "fecundity_male")
                validate_scalar_numeric(growth_rate_herd, "growth_rate_herd")
                validate_param_range(herd_size_total, "herd_size_total")

            _check_stage(
                batch_projection,
                lambda i: validate_population_size_inputs(
                    herd_size_total[i], fec_f[i], fec_m[i],
                    herd_named(p_death, i, TEN_COHORTS), herd_named(p_off, i, TEN_COHORTS),
                    herd_named(p_growth, i, TEN_COHORTS), growth_rate_herd[i],
                    herd_named(herd_structure, i, EIGHT_COHORTS),
                    pd.Series(structure["cohort_share"][i], index=list(SIX_COHORTS)),
                    prop_named(i),
                ),
                n,
            )

        popsize = _projection_kernel(
            as_float(herd_size_total), fec_f, fec_m, p_death, p_off, p_growth,
            growth_rate_herd, herd_structure, structure["cohort_share"], prop_nondemo,
        )

        # ---- Offtake summary --------------------------------------------------
        if validation_enabled():
            _single_numeric(simulation_duration, "simulation_duration")
        offtake_heads10 = {c: popsize["cohort_offtake_heads"][:, j] for j, c in enumerate(TEN_COHORTS)}
        summary = _summary_offtake_kernel(
            popsize["cohort_stock_start"], popsize["cohort_stock_end_projected"],
            popsize["cohort_stock_average"], offtake_heads10,
            float(np.asarray(simulation_duration, dtype="float64").reshape(-1)[0]),
        )

        # ---- Map results back to cohort rows ------------------------------------
        cohort_short = cohort_level_results["cohort_short"].to_numpy(dtype=object)
        cohort_pos = np.array([SIX_COHORTS.index(c) if c in SIX_COHORTS else -1 for c in cohort_short], dtype=int)
        assigned = (herd_pos >= 0) & (cohort_pos >= 0)
        hp = np.where(assigned, herd_pos, 0)
        cp = np.where(assigned, cohort_pos, 0)

        def put(col: str, mat: np.ndarray) -> None:
            vals = np.asarray(mat, dtype="float64")[hp, cp] if n else np.full(len(hp), np.nan)
            if col in cohort_level_results.columns:
                old = as_float(cohort_level_results[col])
                cohort_level_results[col] = np.where(assigned, vals, old)
            else:
                cohort_level_results[col] = np.where(assigned, vals, np.nan)

        put("cohort_stock_size_unscaled", popsize["cohort_stock_average"])
        put("cohort_stock_size_scaled", popsize["cohort_stock_start"])
        put("cohort_stock_annual_nondemographic", popsize["cohort_stock_annual_nondemographic"])
        put("offtake_heads_unscaled", summary["offtake_heads"])
        put("offtake_heads_assessment_unscaled", summary["offtake_heads_assessment"])

        cohort_level_results["offtake_heads_scaled"] = rescale_x_to_y(
            x_scaled_variable=cohort_level_results["offtake_heads_unscaled"],
            x_reference_from=cohort_level_results["cohort_stock_size_unscaled"],
            y_scaling_variable=cohort_level_results["cohort_stock_size_scaled"],
        )
        cohort_level_results["offtake_heads_assessment_scaled"] = rescale_x_to_y(
            x_scaled_variable=cohort_level_results["offtake_heads_assessment_unscaled"],
            x_reference_from=cohort_level_results["cohort_stock_size_unscaled"],
            y_scaling_variable=cohort_level_results["cohort_stock_size_scaled"],
        )

        # ---- Herd-level results ---------------------------------------------------
        if n:
            growth_tab = pd.DataFrame({"herd_id": herd_ids, "growth_rate_herd": np.asarray(growth_rate_herd, float)})
            g = as_float(lookup(herd_level_results, growth_tab, "growth_rate_herd"))
            matched = pd.Index(herd_ids).get_indexer(herd_level_results["herd_id"]) >= 0
        else:
            g = np.full(len(herd_level_results), np.nan)
            matched = np.zeros(len(herd_level_results), dtype=bool)
        if "growth_rate_herd" in herd_level_results.columns:
            herd_level_results["growth_rate_herd"] = np.where(
                matched, g, as_float(herd_level_results["growth_rate_herd"]))
        else:
            herd_level_results["growth_rate_herd"] = np.where(matched, g, np.nan)

        progress.success("Demographic herd simulation complete.")

    return {
        "cohort_level_results": cohort_level_results,
        "herd_level_results": herd_level_results,
    }
