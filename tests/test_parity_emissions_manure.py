"""Golden parity and behaviour of ``run_emissions_manure_module``.

The golden case mirrors ``tools/r_reference/generate_golden.R`` (case
``emissions_manure_module``). The other tests check the vectorised run module
against a literal row-by-row transcription of the R algorithm, and the
run-level validation messages against R.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from golden_utils import assert_frame_matches, assert_matches_golden, load_golden

from gleam import (
    GleamValidationError,
    GleamWarning,
    calc_ch4_manure,
    calc_n2o_manure_direct,
    calc_n2o_manure_leaching,
    calc_n2o_manure_total,
    calc_n2o_manure_volatilization,
    calc_volatile_solids,
    load_example,
    run_emissions_manure_module,
)
from gleam._utils import is_na, merge_dt

PHASE = "nondemo_productive_phase_id"
MMS = "manure_management_system"

R_OUTPUT_COLUMNS = [
    "volatile_solids",
    "ch4_manure_pasture", "ch4_manure_burned", "ch4_manure_other", "ch4_manure_all_noburn",
    "n2o_manure_pasture_direct", "n2o_manure_burned_direct", "n2o_manure_other_direct",
    "n2o_manure_all_noburn_direct",
    "n2o_manure_pasture_vol", "n2o_manure_burned_vol", "n2o_manure_other_vol", "n2o_manure_all_noburn_vol",
    "n2o_manure_pasture_leach", "n2o_manure_burned_leach", "n2o_manure_other_leach",
    "n2o_manure_all_noburn_leach",
    "n2o_manure_pasture_indirect", "n2o_manure_burned_indirect", "n2o_manure_other_indirect",
    "n2o_manure_pasture_total", "n2o_manure_burned_total", "n2o_manure_other_total",
]


def _inputs():
    return (
        load_example("emissions_manure_input_chrt_data.csv"),
        load_example("manure_management_system_fraction.csv"),
        load_example("manure_management_system_factors.csv"),
    )


def _run(cohort, fraction, factors, **kwargs):
    return run_emissions_manure_module(
        cohort_level_data=cohort,
        manure_management_system_fraction=fraction,
        manure_management_system_factors=factors,
        show_indicator=False,
        **kwargs,
    )


# ---- golden parity ------------------------------------------------------------


def test_emissions_manure_module_matches_r():
    cohort, fraction, factors = _inputs()
    result = _run(cohort, fraction, factors)
    assert_matches_golden(result, "emissions_manure_module", "result")


def _pipeline_inputs(kind, keep):
    from gleam.io import example_path, read_csv

    def rd(name):
        df = read_csv(example_path(name, kind))
        return df if keep is None else df[keep(df["herd_id"])].reset_index(drop=True)

    return rd("manure_management_system_fraction.csv"), rd("manure_management_system_factors.csv")


_DIRECT_HERDS = None


def _in_direct(h):
    global _DIRECT_HERDS
    if _DIRECT_HERDS is None:
        _DIRECT_HERDS = load_example("emissions_direct_input_hrd_data.csv")["herd_id"]
    return h.isin(_DIRECT_HERDS)


@pytest.mark.parametrize(
    "case, kind, keep",
    [
        ("run_gleam_mixed_no_structure", "run_gleam_examples", lambda h: ~h.isin([14, 15])),
        ("run_gleam_nondemo_only", "run_gleam_examples", lambda h: h.isin([14, 15])),
        ("run_gleam_structure_AR6", "run_gleam_examples", None),
        ("emissions_direct_1a_no_structure", "run_modules_examples", _in_direct),
        ("emissions_direct_2b_structure_rq", "run_modules_examples", _in_direct),
    ],
)
def test_manure_step_of_r_pipelines(case, kind, keep):
    # The R pipelines run this module on the cohort table built by the upstream
    # modules; the golden cohort table holds those columns before `volatile_solids`
    # and the manure outputs right after them.
    golden = load_golden(case, "cohort_level_results")
    upstream = list(golden.columns[: list(golden.columns).index("volatile_solids")])
    fraction, factors = _pipeline_inputs(kind, keep)
    result = _run(golden[upstream], fraction, factors)
    assert_frame_matches(result, golden[upstream + R_OUTPUT_COLUMNS], label=case)


# ---- row-by-row reference (literal transcription of the R loop) -----------------


def _reference_rowwise(cohort, fraction, factors):
    """R's ``by = .I`` algorithm: per-row selection of ``mms_data`` + scalar core calls."""
    use_phase = PHASE in cohort.columns and PHASE in fraction.columns
    mms_data = merge_dt(fraction, factors, by=["herd_id", MMS])
    out = cohort.copy()
    records = []
    for _, row in cohort.iterrows():
        sel = (mms_data["herd_id"] == row["herd_id"]) & (mms_data["cohort_short"] == row["cohort_short"])
        if use_phase:
            if is_na(row[PHASE]):
                sel &= mms_data[PHASE].isna()
            else:
                sel &= mms_data[PHASE] == row[PHASE]
        rows = mms_data[sel.to_numpy()]

        def build(fields, rows=rows):
            # split() by system name (sorted levels), first row per system
            return {
                name: {f: float(rows[rows[MMS] == name].iloc[0][f]) for f in fields}
                for name in sorted(rows[MMS].unique())
            }

        rec = {}
        rec["volatile_solids"] = vs = calc_volatile_solids(
            float(row["ration_intake"]), float(row["ration_digestibility_fraction"]),
            float(row["ration_urinary_energy_fraction"]), float(row["ration_ash"]),
        )
        n = float(row["nitrogen_excretion"])
        rec.update(calc_ch4_manure(volatile_solids=vs, **build(
            ["manure_management_system_fraction", "methane_conversion_factor_mcf", "ch4_max_producing_capacity_bo"]
        )))
        rec.update(calc_n2o_manure_direct(
            nitrogen_excretion=n, **build(["manure_management_system_fraction", "n2o_ef3"])
        ))
        rec.update(calc_n2o_manure_volatilization(
            nitrogen_excretion=n, **build(["manure_management_system_fraction", "n2o_ef4", "nitrogen_fracgas"])
        ))
        rec.update(calc_n2o_manure_leaching(
            nitrogen_excretion=n, **build(["manure_management_system_fraction", "n2o_ef5", "nitrogen_fracleach"])
        ))
        rec.update(calc_n2o_manure_total(**{
            k: rec[k] for k in [
                "n2o_manure_pasture_vol", "n2o_manure_pasture_leach", "n2o_manure_burned_vol",
                "n2o_manure_burned_leach", "n2o_manure_other_vol", "n2o_manure_other_leach",
                "n2o_manure_pasture_direct", "n2o_manure_burned_direct", "n2o_manure_other_direct",
            ]
        }))
        records.append(rec)
    for col in R_OUTPUT_COLUMNS:
        out[col] = [r[col] for r in records]
    return out


def _no_phase_inputs():
    """Inputs without ``nondemo_productive_phase_id`` (matching on herd and cohort only)."""
    cohort, fraction, factors = _inputs()
    keep_c = cohort[PHASE].isna() | (cohort[PHASE] == 1)
    keep_f = fraction[PHASE].isna() | (fraction[PHASE] == 1)
    cohort = cohort[keep_c].drop(columns=PHASE).reset_index(drop=True)
    fraction = fraction[keep_f].drop(columns=PHASE).reset_index(drop=True)
    # shuffle the lookup tables: the result must not depend on their row order
    fraction = fraction.sample(frac=1, random_state=3).reset_index(drop=True)
    factors = factors.sample(frac=1, random_state=4).reset_index(drop=True)
    return cohort, fraction, factors


@pytest.mark.parametrize("case", ["with_phase", "without_phase"])
def test_vectorised_run_matches_rowwise_r_algorithm(case):
    cohort, fraction, factors = _inputs() if case == "with_phase" else _no_phase_inputs()
    result = _run(cohort, fraction, factors)
    expected = _reference_rowwise(cohort, fraction, factors)
    assert list(result.columns) == list(cohort.columns) + R_OUTPUT_COLUMNS
    # identical operations in identical order: bit-identical results
    assert_frame_matches(result, expected, rtol=0, atol=0)


def test_reversed_cohort_row_order_is_kept():
    cohort, fraction, factors = _inputs()
    rev = cohort.iloc[::-1].reset_index(drop=True)
    result = _run(rev, fraction, factors)
    expected = _run(cohort, fraction, factors).iloc[::-1].reset_index(drop=True)
    assert_frame_matches(result, expected, rtol=0, atol=0)


def test_duplicate_fraction_rows_use_first_row_without_validation():
    # Without validation, R's split() + mms_df[1, ] keeps the first matching row per system.
    cohort, fraction, factors = _inputs()
    dup = fraction.iloc[[0]].copy()
    dup["manure_management_system_fraction"] = 0.99
    fraction2 = pd.concat([fraction, dup], ignore_index=True)
    with pytest.warns(GleamWarning, match="Input validation has been turned off"):
        result = _run(cohort, fraction2, factors, validate_inputs=False)
    expected = _run(cohort, fraction, factors)
    assert_frame_matches(result, expected, rtol=0, atol=0)
    with pytest.warns(GleamWarning):
        assert_frame_matches(
            _reference_rowwise(cohort, fraction2, factors),
            _run(cohort, fraction2, factors, validate_inputs=False), rtol=0, atol=0,
        )


def test_cohort_without_mms_rows_gets_zero_without_validation():
    cohort, fraction, factors = _inputs()
    fraction2 = fraction[~((fraction["herd_id"] == 1) & (fraction["cohort_short"] == "FA"))]
    with pytest.warns(GleamWarning):
        result = _run(cohort, fraction2, factors, validate_inputs=False)
    row = result[(result["herd_id"] == 1) & (result["cohort_short"] == "FA")].iloc[0]
    for col in R_OUTPUT_COLUMNS[1:]:
        assert row[col] == 0.0
    with pytest.raises(GleamValidationError, match="Missing herd/cohort combinations"):
        _run(cohort, fraction2, factors)


def test_inputs_not_modified_and_existing_columns_overwritten_in_place():
    cohort, fraction, factors = _inputs()
    cohort["ch4_manure_other"] = -1.0  # pre-existing output column keeps its position
    copies = [cohort.copy(), fraction.copy(), factors.copy()]
    result = _run(cohort, fraction, factors)
    for before, after in zip(copies, (cohort, fraction, factors)):
        pd.testing.assert_frame_equal(before, after)
    assert list(result.columns) == list(cohort.columns) + [c for c in R_OUTPUT_COLUMNS if c != "ch4_manure_other"]
    assert (result["ch4_manure_other"] > 0).all()


def test_validate_inputs_false_gives_same_result():
    cohort, fraction, factors = _inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        fast = _run(cohort, fraction, factors, validate_inputs=False)
    assert_frame_matches(fast, _run(cohort, fraction, factors), rtol=0, atol=0)


# ---- run-level validation (messages as produced by R, minus cli markup) ----------


def _drop(df, **eq):
    mask = np.ones(len(df), dtype=bool)
    for k, v in eq.items():
        mask &= (df[k] == v).to_numpy()
    return df[~mask].reset_index(drop=True)


def test_run_validation_messages_match_r():
    cohort, fraction, factors = _inputs()

    def err(c=cohort, f=fraction, a=factors):
        with pytest.raises(GleamValidationError) as exc:
            _run(c, f, a)
        return str(exc.value)

    assert err(c=pd.concat([cohort, cohort.iloc[[0]]], ignore_index=True)) == (
        "Duplicate herd/cohort rows in `cohort_level_data` for grouping columns "
        '"herd_id", "cohort_short", and "nondemo_productive_phase_id".'
    )
    assert err(f=pd.concat([fraction, fraction.iloc[[0]]], ignore_index=True)) == (
        "Duplicate herd/cohort/manure-management rows in `manure_management_system_fraction`."
    )
    f2 = fraction.copy()
    f2.loc[0, "manure_management_system_fraction"] = 0.5
    f2.loc[59, "manure_management_system_fraction"] = 0.5
    assert err(f=f2) == (
        "For each herd/cohort group, the sum of MMS fractions in `manure_management_system_fraction` "
        'must equal 1. Invalid groups: "1 / FA / NA"'
    )
    assert err(a=pd.concat([factors, factors.iloc[[0]]], ignore_index=True)) == (
        "Duplicate herd_id + manure_management_system rows in `manure_management_system_factors`."
    )
    assert err(a=_drop(factors, herd_id=3)) == (
        "Herd IDs in `manure_management_system_fraction` not found in "
        "`manure_management_system_factors`: 3"
    )
    assert err(a=_drop(factors, herd_id=2, manure_management_system="mms_burned")) == (
        "Some `manure_management_system` values in `manure_management_system_fraction` have no "
        "matching entry in `manure_management_system_factors`. Affected herd_ids and missing "
        'systems: "2: mms_burned"'
    )
    f3 = _drop(fraction, herd_id=2, cohort_short="FA", manure_management_system="mms_burned")
    sel = (f3["herd_id"] == 2) & (f3["cohort_short"] == "FA") & (f3[MMS] == "mms_drylot")
    burned = fraction[(fraction["herd_id"] == 2) & (fraction["cohort_short"] == "FA") & (fraction[MMS] == "mms_burned")]
    f3.loc[sel, "manure_management_system_fraction"] += burned["manure_management_system_fraction"].iloc[0]
    assert err(f=f3) == (
        "Within each herd_id, manure_management_system lists must be consistent across cohorts in "
        "`manure_management_system_fraction`. Inconsistent herds: 2"
    )
    assert err(f=_drop(fraction, herd_id=1, cohort_short="MN")) == (
        'Missing herd/cohort combinations in `manure_management_system_fraction`: "1 / MN / 1"'
    )
    assert err(c=_drop(cohort, herd_id=13)) == (
        "Herd IDs in `manure_management_system_fraction` not found in `cohort_level_data`: 13"
    )
    assert err(f=_drop(fraction, herd_id=13), a=_drop(factors, herd_id=13)) == (
        'Missing herd/cohort combinations in `manure_management_system_fraction`: "13 / FA / NA"'
    )
    assert err(c=_drop(cohort, herd_id=13), f=_drop(fraction, herd_id=13)) == (
        "Herd IDs in `manure_management_system_factors` not found in `cohort_level_data`: 13"
    )
    assert err(c=_drop(cohort, herd_id=13), a=_drop(factors, herd_id=13)) == (
        "Herd IDs in `manure_management_system_fraction` not found in `manure_management_system_factors`: 13"
    )
    c2 = cohort.copy()
    c2.loc[0, "ration_ash"] = np.nan
    assert err(c=c2) == (
        "`cohort_level_data` must not contain missing ration_urinary_energy_fraction or ration_ash."
    )
    c2 = cohort.copy()
    c2.loc[0, "herd_id"] = np.nan
    assert err(c=c2) == "`cohort_level_data` must not contain missing herd_id or cohort_short."
    c2 = cohort.copy()
    c2.loc[0, "cohort_short"] = "XX"
    assert err(c=c2).startswith('Invalid `cohort_short` values in `cohort_level_data`: "XX".')
    c2 = cohort.copy()
    c2.loc[1, "ration_ash"] = 0.5
    assert "`ration_ash`" in err(c=c2) and "= 0.5 is out of range" in err(c=c2)
    a2 = factors.copy()
    a2.loc[2, "methane_conversion_factor_mcf"] = 120
    assert err(a=a2) == (
        "`methane_conversion_factor_mcf` = 120 is out of range; expected value should be >= 0 and <= 100."
    )
    a2 = factors.copy()
    a2.loc[2, "n2o_ef3"] = np.nan
    assert err(a=a2) == "MMS values must not contain missing values."
    assert err(a=factors.drop(columns="n2o_ef4")) == (
        'Missing required columns in `manure_management_system_factors`: "n2o_ef4"'
    )
    assert err(c=cohort.iloc[0:0]) == "`cohort_level_data` must contain at least one row."
