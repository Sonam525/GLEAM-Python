"""Sweep one management lever through the full GLEAM pipeline.

Varies the daily milk yield of the example dairy cattle herd (herd 1) and
reports total emissions and emission intensity per kg of fat-and-protein-
corrected milk (FPCM). The same pattern (perturb inputs, run, collect herd-
level outputs) produces training data for a surrogate model.

Run: python examples/parameter_sweep.py
"""

from __future__ import annotations

import pandas as pd

import gleampy
from gleampy.io import load_example


def ex(name: str) -> pd.DataFrame:
    return load_example(name, "run_gleam_examples")


def herd_one(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["herd_id"] == 1].reset_index(drop=True)


base = dict(
    has_herd_structure=True,
    cohort_level_data=herd_one(ex("master_chrt_lvl_structure_data.csv")),
    herd_level_data=herd_one(ex("master_hrd_lvl_structure_data.csv")),
    feed_rations=herd_one(ex("feed_rations_share_chrt.csv")),
    feed_params=ex("feed_quality.csv"),
    feed_emissions=ex("feed_emission_factors.csv"),
    manure_management_system_fraction=herd_one(ex("manure_management_system_fraction.csv")),
    manure_management_system_factors=herd_one(ex("manure_management_system_factors.csv")),
    simulation_duration=365,
    global_warming_potential_set="AR6",
    show_indicator=False,
)

rows = []
for milk_yield in (15.0, 20.0, 25.0, 30.0, 35.0):
    herd = base["herd_level_data"].copy()
    herd["milk_yield_day"] = milk_yield
    res = gleampy.run_gleam(**{**base, "herd_level_data": herd})

    emissions = res["aggregation_results"]["results_emissions"]
    production = res["aggregation_results"]["results_production"]
    milk = emissions[emissions["commodity_name"] == "Milk"]["value_total_allocated_co2eq"].sum()
    fpcm = production.loc[production["label"] == "MilkFatProteinCorrected", "value_total"].sum()
    rows.append(
        {
            "milk_yield_day": milk_yield,
            "total_kg_co2eq": emissions["value_total_allocated_co2eq"].sum(),
            "milk_kg_co2eq": milk,
            "fpcm_kg": fpcm,
            "kg_co2eq_per_kg_fpcm": milk / fpcm,
        }
    )

print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
