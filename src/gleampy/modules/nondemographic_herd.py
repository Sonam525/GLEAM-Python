"""Non-demographic herd module runner (port of ``R/run_nondemographic_herd_module.R``).

Simulates the production cycles of the non-demographic cohort blocks ``FN`` /
``MN`` of every herd over a fixed 365-day horizon: cycle geometry, entrants
per cycle start, phase-level survival, average stock per productive phase and
terminal offtake. The R function loops over herds and cohort blocks; here all
blocks are evaluated at once with the vectorised core functions (on a
validation error the R loop is replayed block by block so the error raised is
the one R raises first).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils import Lookup, Progress, as_float, as_str, copy_frame, lookup_columns
from ..core.nondemographic_herd import (
    assign_nondemographic_phase_durations,
    calc_nondemo_avg_stock_phase_horizon,
    calc_nondemo_cycle_geometry,
    calc_nondemo_offtake_total_horizon,
    calc_nondemo_phase,
    calc_nondemo_start_sizes,
)
from ..validation._shared import GleamValidationError, setup_validation, validation_enabled
from ..validation.demographic_herd_core import _single_numeric
from ..validation.nondemographic_herd_run import (
    PHASE_DURATION_COLS,
    validate_run_nondemographic_herd_module_inputs,
)
from .demographic_herd import _fmt_id

_NO_CYCLE_START_MSG = (
    "No non-demographic cycle start for herd {herd}, cohort block {block} "
    "(phase 1 duration is 0): calc_nondemo_start_sizes() returns 0 instead of a list "
    "and R fails with '$ operator is invalid for atomic vectors'."
)
_MISSING_BLOCK_INPUT_MSG = (
    "No non-demographic cycle start for herd {herd}, cohort block {block}: {what} is missing, "
    "so the production cycle cannot be built."
)


def _no_cycle_start_error(b: _Blocks, i: int) -> GleamValidationError:
    """The error for a block without any cycle start, naming the input at fault.

    A phase 1 of 0 days is the replicated R crash; with ``validate_inputs=False``
    a missing herd row or a missing phase / rest duration leads to the same
    "no cycle start" state, and the message names that input instead.
    """
    herd, block = _fmt_id(b.herd_id[i]), _fmt_id(b.block[i])
    phase1, rest = b.phase1[i], b.rest[i]
    if b.herd_missing[i]:
        what = "the herd's row in `herd_level_data`"
    elif np.isnan(phase1):
        what = (
            "the phase 1 duration (`cohort_duration_days` of the phase 1 row or the herd-level "
            "`phase1_nondemo_*_duration_days`)"
        )
    elif np.isnan(rest):
        what = "`rest_between_nondemo_cycles_duration`"
    elif phase1 <= 0:
        return GleamValidationError(_NO_CYCLE_START_MSG.format(herd=herd, block=block))
    else:
        return GleamValidationError(
            f"No non-demographic cycle start for herd {herd}, cohort block {block}: the cycle "
            f"length (phase 1 = {phase1:g}, phase 2 = {b.phase2[i]:g}, rest = {rest:g} days) "
            "does not fit a cycle start in the 365-day horizon."
        )
    return GleamValidationError(_MISSING_BLOCK_INPUT_MSG.format(herd=herd, block=block, what=what))


class _Blocks:
    """Per (herd, cohort block) inputs, in R's processing order."""

    def __init__(self, cohort: pd.DataFrame, herd_level_data: pd.DataFrame) -> None:
        herd_ids = cohort["herd_id"]
        # R loops over herds with cohort_level_results[herd_id == h]: an NA herd
        # selects no row, so its blocks are never simulated (outputs stay NA).
        herd_na = herd_ids.isna().to_numpy()
        herd_order = pd.unique(herd_ids[~herd_na])
        herd_rank = pd.Index(herd_order).get_indexer(herd_ids)
        herd_rank[herd_na] = -1
        cohort_short = as_str(cohort["cohort_short"])
        block = np.array([None if v is None else v[:2] for v in cohort_short], dtype=object)
        phase = as_float(cohort["nondemo_productive_phase_id"])

        tab = pd.DataFrame({"rank": herd_rank, "block": block, "row": np.arange(len(cohort))})
        first = (
            tab[tab["rank"] >= 0]
            .drop_duplicates(["rank", "block"], keep="first")
            .sort_values(["rank", "row"], kind="mergesort")
        )
        self.herd_rank = first["rank"].to_numpy()
        self.herd_id = herd_order[self.herd_rank] if len(first) else np.array([], dtype=object)
        self.block = first["block"].to_numpy(dtype=object)
        n = len(first)
        pair_of = {(r, b): i for i, (r, b) in enumerate(zip(self.herd_rank, self.block))}
        # block of each row; -1 for the rows of an NA herd (in no block)
        self.row_pair = np.array([pair_of.get((r, b), -1) for r, b in zip(herd_rank, block)], dtype=int)

        # first phase-1 / phase-2 row of each block (setorder() is stable)
        self.r1 = np.full(n, -1)
        self.r2 = np.full(n, -1)
        for i in range(len(cohort) - 1, -1, -1):
            p = self.row_pair[i]
            if p < 0:
                continue
            if phase[i] == 1:
                self.r1[p] = i
            elif phase[i] == 2:
                self.r2[p] = i
        self.has2 = self.r2 >= 0

        duration = as_float(cohort["cohort_duration_days"])
        death = as_float(cohort["death_rate"])

        def at(vals: np.ndarray, rows: np.ndarray) -> np.ndarray:
            return np.where(rows >= 0, vals[np.maximum(rows, 0)], np.nan)

        self.phase1 = at(duration, self.r1)
        self.death1 = at(death, self.r1)
        self.phase2 = np.where(self.has2, at(duration, self.r2), 0.0)
        self.death2 = np.where(self.has2, at(death, self.r2), 0.0)
        self.r1_short = np.array([cohort_short[r] if r >= 0 else None for r in self.r1], dtype=object)
        self.r2_short = np.array([cohort_short[r] if r >= 0 else None for r in self.r2], dtype=object)

        keys = pd.DataFrame({"herd_id": self.herd_id})
        herd = Lookup(keys, herd_level_data)
        fem = herd("cohort_stock_fem_annual_nondemo")
        mal = herd("cohort_stock_mal_annual_nondemo")
        self.entrants = np.where(self.block == "FN", as_float(fem), as_float(mal))
        self.rest = as_float(herd("rest_between_nondemo_cycles_duration"))
        self.herd_missing = ~herd.hit
        self.n = n


def _replay_r_loop(b: _Blocks, simulation_duration: Any) -> None:
    """The R loop with scalar calls; raises the first error R would raise."""
    for i in range(b.n):
        phase1, phase2 = b.phase1[i], b.phase2[i]
        death1, death2 = b.death1[i], b.death2[i]
        entrants = b.entrants[i]
        geom = calc_nondemo_cycle_geometry(phase1, phase2, b.rest[i])
        start_size = calc_nondemo_start_sizes(entrants, geom["total_nondemo_cycle_starts_to_distribute"])
        if not isinstance(start_size, dict):
            raise _no_cycle_start_error(b, i)
        start_cycle = start_size["cohort_stock_nondemo_start_cycle"]
        full1 = calc_nondemo_phase(start_cycle, phase1, death1, phase1)
        part1 = calc_nondemo_phase(start_cycle, phase1, death1, geom["partial_phase1_nondemo_duration"])
        start2 = full1["cohort_stock_nondemo"]["end"]
        full2 = calc_nondemo_phase(start2, phase2, death2, phase2)
        part2 = calc_nondemo_phase(start2, phase2, death2, geom["partial_phase2_nondemo_duration"])
        calc_nondemo_avg_stock_phase_horizon(full1, part1, geom["number_full_nondemo_cycles"])
        calc_nondemo_avg_stock_phase_horizon(full2, part2, geom["number_full_nondemo_cycles"])
        calc_nondemo_offtake_total_horizon(
            full1["cohort_stock_nondemo"]["end"], full2["cohort_stock_nondemo"]["end"], entrants,
            start_cycle, geom["number_full_nondemo_cycles"], geom["partial_phase1_nondemo_duration"],
            geom["partial_phase2_nondemo_duration"], phase1, phase2, simulation_duration,
        )


def _simulate_blocks(b: _Blocks, simulation_duration: Any) -> dict[str, np.ndarray]:
    """Vectorised version of the R loop body for all blocks."""
    geom = calc_nondemo_cycle_geometry(b.phase1, b.phase2, b.rest)
    starts = np.atleast_1d(geom["total_nondemo_cycle_starts_to_distribute"])
    start_size = calc_nondemo_start_sizes(b.entrants, starts)
    if (starts <= 0).any():
        raise _NoCycleStart
    start_cycle = start_size["cohort_stock_nondemo_start_cycle"]
    partial1 = geom["partial_phase1_nondemo_duration"]
    partial2 = geom["partial_phase2_nondemo_duration"]
    number_full = geom["number_full_nondemo_cycles"]

    full1 = calc_nondemo_phase(start_cycle, b.phase1, b.death1, b.phase1)
    part1 = calc_nondemo_phase(start_cycle, b.phase1, b.death1, partial1)
    start2 = full1["cohort_stock_nondemo"]["end"]
    full2 = calc_nondemo_phase(start2, b.phase2, b.death2, b.phase2)
    part2 = calc_nondemo_phase(start2, b.phase2, b.death2, partial2)

    avg1 = calc_nondemo_avg_stock_phase_horizon(full1, part1, number_full)["cohort_stock_size_unscaled"]
    avg2 = calc_nondemo_avg_stock_phase_horizon(full2, part2, number_full)["cohort_stock_size_unscaled"]
    off = calc_nondemo_offtake_total_horizon(
        full1["cohort_stock_nondemo"]["end"], full2["cohort_stock_nondemo"]["end"], b.entrants,
        start_cycle, number_full, partial1, partial2, b.phase1, b.phase2, simulation_duration,
    )
    shape = (b.n,)
    f = lambda v: np.broadcast_to(np.asarray(v, dtype="float64"), shape)  # noqa: E731
    return {
        "avg1": f(avg1), "avg2": f(avg2), "partial1": f(partial1), "partial2": f(partial2),
        "off1": f(off["offtake_heads_nondemo_phase1"]), "off2": f(off["offtake_heads_nondemo_phase2"]),
        "offa1": f(off["offtake_heads_assessment_nondemo_phase1"]),
        "offa2": f(off["offtake_heads_assessment_nondemo_phase2"]),
        "number_full": f(number_full), "starts": f(starts),
    }


class _NoCycleStart(Exception):
    pass


def run_nondemographic_herd_module(
    cohort_level_data: pd.DataFrame,
    herd_level_data: pd.DataFrame,
    simulation_duration: float = 365,
    show_indicator: bool = True,
    validate_inputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the non-demographic herd module.

    Parameters
    ----------
    cohort_level_data : pandas.DataFrame
        One row per herd, cohort block (``FN`` / ``MN``) and productive phase with
        ``herd_id``, ``cohort_short``, ``nondemo_productive_phase_id`` (1, optional 2),
        ``death_rate`` (phase mortality, fraction) and, unless the herd table
        gives phase durations, ``cohort_duration_days`` (days).
    herd_level_data : pandas.DataFrame
        One row per herd with ``herd_id``, ``cohort_stock_fem_annual_nondemo``,
        ``cohort_stock_mal_annual_nondemo`` (annual entrants, heads),
        ``rest_between_nondemo_cycles_duration`` (days) and optionally the four
        ``phase{1,2}_nondemo_{fem,mal}_duration_days`` columns (days).
    simulation_duration : float
        Length of the reporting period (days); a single value.
    show_indicator : bool
        Print progress messages.
    validate_inputs : bool
        Validate inputs (default ``True``).

    Returns
    -------
    dict
        ``cohort_level_results``: input rows (in input order, rows with zero
        average stock dropped) with ``cohort_duration_days`` filled (float64,
        whatever the input dtype: fractional herd-level phase durations are
        never truncated, unlike R with an integer column),
        ``offtake_rate = 1`` and ``cohort_stock_size_unscaled``,
        ``partial_nondemo_phase_duration``, ``offtake_heads_unscaled``,
        ``offtake_heads_assessment_unscaled``, ``number_full_nondemo_cycles``,
        ``total_nondemo_cycle_starts_to_distribute``; ``herd_level_results``:
        herd rows with the phase-duration columns and
        ``total_nondemo_fem_duration_days`` / ``total_nondemo_mal_duration_days``.
    """
    with setup_validation(validate_inputs):
        validate_run_nondemographic_herd_module_inputs(cohort_level_data, herd_level_data)

        progress = Progress(show_indicator)
        progress.status("Simulating the non-demographic herd, please wait…")

        cohort_level_results = copy_frame(cohort_level_data)
        herd_level_results = copy_frame(herd_level_data)
        herd_level_input = copy_frame(herd_level_data)

        for nm in PHASE_DURATION_COLS:
            if nm not in herd_level_results.columns:
                herd_level_results[nm] = np.nan
        if "cohort_duration_days" not in cohort_level_results.columns:
            cohort_level_results["cohort_duration_days"] = np.nan

        # Pre-assign the non-demographic phase durations to cohort_duration_days.
        # Always float64, whatever the dtype of the input column: R's grouped
        # `:=` truncates fractional herd-level phase durations when fread typed
        # cohort_duration_days as integer (an R bug that is not replicated, see
        # docs/R_PORT_NOTES.md).
        durations = np.atleast_1d(assign_nondemographic_phase_durations(
            cohort_short=cohort_level_results["cohort_short"],
            nondemo_productive_phase_id=cohort_level_results["nondemo_productive_phase_id"],
            cohort_duration_days=cohort_level_results["cohort_duration_days"],
            **lookup_columns(cohort_level_results, herd_level_results, PHASE_DURATION_COLS),
        )).astype("float64", copy=False)
        cohort_level_results["cohort_duration_days"] = durations

        # Total productive duration of the FN / MN blocks of each herd
        herd_index = pd.Index(pd.unique(herd_level_results["herd_id"]))
        row_herd = herd_index.get_indexer(cohort_level_results["herd_id"])
        row_herd[cohort_level_results["herd_id"].isna().to_numpy()] = -1  # herd_id == NA selects no row
        durations = as_float(cohort_level_results["cohort_duration_days"])
        cohort_short = as_str(cohort_level_results["cohort_short"])
        totals = {}
        for col, code in (("total_nondemo_fem_duration_days", "FN"), ("total_nondemo_mal_duration_days", "MN")):
            acc = np.zeros(len(herd_index))
            sel = (row_herd >= 0) & (cohort_short == code) & ~np.isnan(durations)
            np.add.at(acc, row_herd[sel], durations[sel])  # sequential, as R's sum()
            totals[col] = acc[herd_index.get_indexer(herd_level_results["herd_id"])]
        for col, vals in totals.items():
            herd_level_results[col] = vals

        # Simulate every herd x cohort block. R checks the scalar simulation_duration
        # in calc_nondemo_offtake_total_horizon(); the vectorised core would broadcast
        # a vector over the blocks instead, so it is checked once here.
        if validation_enabled():
            _single_numeric(simulation_duration, "simulation_duration")
        blocks = _Blocks(cohort_level_results, herd_level_input)
        try:
            sim = _simulate_blocks(blocks, simulation_duration)
        except (GleamValidationError, _NoCycleStart) as err:
            _replay_r_loop(blocks, simulation_duration)
            if isinstance(err, _NoCycleStart):  # pragma: no cover - replay raises first
                raise ValueError("no non-demographic cycle start") from None
            raise

        # Write back phase 1 and phase 2 rows
        in_block = blocks.row_pair >= 0  # False on the rows of an NA herd (never simulated)
        pair = np.where(in_block, blocks.row_pair, 0)
        phase = as_float(cohort_level_results["nondemo_productive_phase_id"])
        if blocks.n:
            is1 = in_block & (phase == 1) & (cohort_short == blocks.r1_short[pair])
            is2 = in_block & (phase == 2) & blocks.has2[pair] & (cohort_short == blocks.r2_short[pair])
        else:  # no block to simulate: no rows, or only NA herd ids
            is1 = is2 = np.zeros(len(pair), dtype=bool)
        assigned = is1 | is2

        def put(col: str, v1: np.ndarray, v2: np.ndarray) -> None:
            if blocks.n:
                vals = np.where(is1, np.asarray(v1)[pair], np.where(is2, np.asarray(v2)[pair], np.nan))
            else:
                vals = np.full(len(pair), np.nan)
            if col in cohort_level_results.columns:
                old = cohort_level_results[col]
                if old.dtype.kind in "iuf" or old.isna().all():
                    cohort_level_results[col] = np.where(assigned, vals, as_float(old))
                else:
                    new = old.astype(object).copy()
                    new[assigned] = vals[assigned]
                    cohort_level_results[col] = new
            else:
                cohort_level_results[col] = np.where(assigned, vals, np.nan)

        ones = np.ones(blocks.n)
        put("offtake_rate", ones, ones)
        put("cohort_stock_size_unscaled", sim["avg1"], sim["avg2"])
        put("partial_nondemo_phase_duration", sim["partial1"], sim["partial2"])
        put("offtake_heads_unscaled", sim["off1"], sim["off2"])
        put("offtake_heads_assessment_unscaled", sim["offa1"], sim["offa2"])
        put("number_full_nondemo_cycles", sim["number_full"], sim["number_full"])
        put("total_nondemo_cycle_starts_to_distribute", sim["starts"], sim["starts"])

        if "cohort_stock_size_unscaled" in cohort_level_results.columns:
            size = as_float(cohort_level_results["cohort_stock_size_unscaled"])
            with np.errstate(invalid="ignore"):
                keep = np.isnan(size) | (size > 0)
            cohort_level_results = cohort_level_results[keep].reset_index(drop=True)

        progress.success("Non-demographic herd simulation complete.")

    return {
        "cohort_level_results": cohort_level_results,
        "herd_level_results": herd_level_results,
    }
