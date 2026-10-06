"""GLEAM package constants.

Port of ``R/gleam_constants.R``. Centralised lookup objects for valid codes,
biologically meaningful species/cohort sub-groups, and pipeline variable
metadata. Any change to supported codes or groupings should be made here only;
all other modules reference these objects.
"""

from __future__ import annotations

# --- Valid input codes -------------------------------------------------------

#: Species code -> full name mapping.
GLEAM_SPECIES_NAMES: dict[str, str] = {
    "CTL": "Cattle",
    "BFL": "Buffalo",
    "SHP": "Sheep",
    "GTS": "Goats",
    "PGS": "Pigs",
    "CML": "Camels",
    "CHK": "Chickens",
}

#: Valid species short codes (in R order).
GLEAM_SPECIES: tuple[str, ...] = tuple(GLEAM_SPECIES_NAMES)

# --- Cohort codes ------------------------------------------------------------

#: Cohort code -> full name mapping.
GLEAM_COHORTS_NAMES: dict[str, str] = {
    "FJ": "Juvenile female",
    "FS": "Sub-adult female",
    "FA": "Adult female",
    "MJ": "Juvenile male",
    "MS": "Sub-adult male",
    "MA": "Adult male",
    "FN": "Non-demographic female",
    "MN": "Non-demographic male",
}

#: Valid cohort short codes (in R order).
GLEAM_COHORTS: tuple[str, ...] = tuple(GLEAM_COHORTS_NAMES)

#: The 6 sex- and age-class cohort codes used by the demographic herd module.
GLEAM_COHORTS_DEMOGRAPHIC: tuple[str, ...] = ("FJ", "FS", "FA", "MJ", "MS", "MA")

#: Non-demographic cohort codes (``setdiff(gleam_cohorts, gleam_cohorts_demographic)``).
GLEAM_COHORTS_NONDEMOGRAPHIC: tuple[str, ...] = tuple(
    c for c in GLEAM_COHORTS if c not in GLEAM_COHORTS_DEMOGRAPHIC
)

# --- Species sub-groups ------------------------------------------------------

#: Ruminant species (NE-based energy system): CTL, BFL, SHP, GTS.
GLEAM_SPECIES_RUMINANTS: tuple[str, ...] = tuple(
    s for s in GLEAM_SPECIES if s not in ("PGS", "CML", "CHK")
)

#: Milk-producing species: ruminants plus camels.
GLEAM_SPECIES_MILK_PRODUCERS: tuple[str, ...] = tuple(
    s for s in GLEAM_SPECIES if s not in ("PGS", "CHK")
)

#: Poultry species code (R stores this as a length-1 character vector).
GLEAM_SPECIES_POULTRY: str = "CHK"

#: All supported species except poultry.
GLEAM_SPECIES_NON_POULTRY: tuple[str, ...] = tuple(
    s for s in GLEAM_SPECIES if s != GLEAM_SPECIES_POULTRY
)

#: Default lower critical temperature for chickens (degrees C).
GLEAM_CHK_LOWER_CRITICAL_TEMPERATURE: float = 18.89

# --- Cohort sub-groups -------------------------------------------------------

#: Male cohort codes (M-prefix), in R order: MJ, MS, MA, MN.
GLEAM_COHORTS_MALE: tuple[str, ...] = tuple(c for c in GLEAM_COHORTS if c.startswith("M"))

#: Female cohort codes (complement of male): FJ, FS, FA, FN.
GLEAM_COHORTS_FEMALE: tuple[str, ...] = tuple(
    c for c in GLEAM_COHORTS if c not in GLEAM_COHORTS_MALE
)

# --- Pipeline variable metadata ----------------------------------------------
#
# Each list below describes the cohort-level variables that the aggregation
# module pivots into long form, and that the allocation module assigns
# emission shares to. Keeping them here ensures both modules stay in sync.

#: Feed intake variable metadata.
GLEAM_FEED_META: list[dict[str, str]] = [
    {"feed_source": "ration_intake", "label": "DryMatterIntake", "unit": "kg dry matter"},
]

#: Nitrogen balance variable metadata.
GLEAM_NITROGEN_BALANCE_META: list[dict[str, str]] = [
    {"nitrogen_balance_source": "nitrogen_intake", "label": "NitrogenIntake", "unit": "kg N"},
    {"nitrogen_balance_source": "nitrogen_retention", "label": "NitrogenRetention", "unit": "kg N"},
    {"nitrogen_balance_source": "nitrogen_excretion", "label": "NitrogenExcretion", "unit": "kg N"},
]


def _prod(source: str, label: str, unit: str, commodity: str) -> dict[str, str]:
    return {
        "production_source": source,
        "label": label,
        "unit": unit,
        "commodity_name": commodity,
        "commodity_type": "Edible",
    }


#: Production variable metadata.
GLEAM_PRODUCTION_META: list[dict[str, str]] = [
    _prod("milk_production_mass_cohort", "MilkRaw", "kg", "Milk"),
    _prod("milk_production_protein_cohort", "MilkProtein", "kg protein", "Milk"),
    _prod("milk_production_fpcm_cohort", "MilkFatProteinCorrected", "kg fat-protein corrected", "Milk"),
    _prod("meat_production_live_weight_cohort", "MeatLiveWeight", "kg live weight", "Meat"),
    _prod("meat_production_carcass_weight_cohort", "MeatCarcassWeight", "kg carcass weight", "Meat"),
    _prod("meat_production_bone_free_meat_cohort", "MeatBoneFree", "kg bone-free meat", "Meat"),
    _prod("meat_production_protein_cohort", "MeatProtein", "kg protein", "Meat"),
    _prod("fibre_production_cohort", "Fibre", "kg", "Fibre"),
    _prod("egg_production_number_cohort", "EggNumber", "eggs", "Eggs"),
    _prod("egg_production_mass_cohort", "EggMass", "kg", "Eggs"),
    _prod("egg_production_protein_cohort", "EggProtein", "kg protein", "Eggs"),
]

#: Emissions variable metadata: all direct and indirect emission sources
#: handled by the allocation and aggregation modules.
GLEAM_EMISSIONS_META: list[dict[str, str]] = [
    {"emissions_source": "ch4_enteric", "label": "Enteric_CH4"},
    {"emissions_source": "ch4_manure_pasture", "label": "Manure-Pasture_CH4"},
    {"emissions_source": "ch4_manure_burned", "label": "Manure-Burned_CH4"},
    {"emissions_source": "ch4_manure_other", "label": "Manure-Other_CH4"},
    {"emissions_source": "n2o_manure_pasture_direct", "label": "ManureDirect-Pasture_N2O"},
    {"emissions_source": "n2o_manure_burned_direct", "label": "ManureDirect-Burned_N2O"},
    {"emissions_source": "n2o_manure_other_direct", "label": "ManureDirect-Other_N2O"},
    {"emissions_source": "n2o_manure_burned_indirect", "label": "ManureIndirect-Burned_N2O"},
    {"emissions_source": "n2o_manure_pasture_indirect", "label": "ManureIndirect-Pasture_N2O"},
    {"emissions_source": "n2o_manure_other_indirect", "label": "ManureIndirect-Other_N2O"},
    {"emissions_source": "co2_ration_fertilizer", "label": "Feed-Fertilizer_CO2"},
    {"emissions_source": "co2_ration_pesticides", "label": "Feed-Pesticides_CO2"},
    {"emissions_source": "co2_ration_crop_activities", "label": "Feed-CropActivities_CO2"},
    {"emissions_source": "co2_ration_luc_nopeat", "label": "Feed-LandUseChange_CO2"},
    {"emissions_source": "co2_ration_luc_peat", "label": "Feed-PeatDrainage_CO2"},
    {"emissions_source": "n2o_ration_fertilizer", "label": "Feed-Fertilizer_N2O"},
    {"emissions_source": "n2o_ration_manure_applied", "label": "Feed-ManureApplication_N2O"},
    {"emissions_source": "n2o_ration_crop_residues", "label": "Feed-CropResidues_N2O"},
    {"emissions_source": "ch4_ration_rice", "label": "Feed-Rice_CH4"},
]

#: Feed-related emission sources, expressed per kg dry matter intake (g/kg DM).
GLEAM_FEED_EMISSIONS_META: list[dict[str, str]] = [
    m for m in GLEAM_EMISSIONS_META if m["emissions_source"].split("_")[1] == "ration"
]

#: Emission sources excluded from commodity allocation.
GLEAM_NON_ALLOCATED_EMISSIONS: tuple[str, ...] = (
    "ch4_manure_pasture",
    "ch4_manure_burned",
    "n2o_manure_pasture_direct",
    "n2o_manure_burned_direct",
    "n2o_manure_burned_indirect",
    "n2o_manure_pasture_indirect",
)

#: Supported global-warming-potential sets (GWP-100) used for CO2-eq conversion.
GLOBAL_WARMING_POTENTIAL_SETS: tuple[str, ...] = (
    "AR6",
    "AR5_excluding_carbon_feedback",
    "AR5_including_carbon_feedback",
    "AR4",
)
