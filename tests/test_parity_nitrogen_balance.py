"""Golden parity: ``run_nitrogen_balance_module`` vs the R package.

Mirrors ``tools/r_reference/generate_golden.R`` (case ``nitrogen_balance_module``).
"""

from __future__ import annotations

from golden_utils import assert_matches_golden

from gleam import load_example, run_nitrogen_balance_module


def test_nitrogen_balance_module_matches_r():
    result = run_nitrogen_balance_module(
        cohort_level_data=load_example("nitrogen_balance_input_chrt_data.csv"),
        herd_level_data=load_example("nitrogen_balance_input_hrd_data.csv"),
        show_indicator=False,
    )
    assert_matches_golden(result, "nitrogen_balance_module", "result")
