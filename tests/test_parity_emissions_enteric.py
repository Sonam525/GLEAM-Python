"""Golden parity: ``run_emissions_enteric_module`` vs the R package.

Mirrors ``tools/r_reference/generate_golden.R`` (case ``emissions_enteric_module``).
"""

from __future__ import annotations

from golden_utils import assert_matches_golden

from gleampy import load_example, run_emissions_enteric_module


def test_emissions_enteric_module_matches_r():
    result = run_emissions_enteric_module(
        cohort_level_data=load_example("emissions_enteric_input_chrt_data.csv"),
        show_indicator=False,
    )
    assert_matches_golden(result, "emissions_enteric_module", "result")
