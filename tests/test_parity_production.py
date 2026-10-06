"""Golden parity: ``run_production_module`` vs the R package.

Mirrors ``tools/r_reference/generate_golden.R`` (case ``production_module``).
"""

from __future__ import annotations

from golden_utils import assert_matches_golden

from gleam import load_example, run_production_module


def test_production_module_matches_r():
    result = run_production_module(
        cohort_level_data=load_example("production_input_chrt_data.csv"),
        herd_level_data=load_example("production_input_hrd_data.csv"),
        simulation_duration=365,
        show_indicator=False,
    )
    assert_matches_golden(result, "production_module", "result")
