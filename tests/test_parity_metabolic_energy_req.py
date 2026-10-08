"""Golden parity tests for :func:`gleampy.run_metabolic_energy_req_module`.

Runs the module on the bundled example inputs exactly as
``tools/r_reference/generate_golden.R`` does (case
``metabolic_energy_req_module``, table ``result``) and compares with the R
output (``rtol=1e-9``, exact column and row order). Also checks run-level
behaviour that mirrors R: no mutation of the inputs, lazily-needed optional
columns, run-level validation messages and the validation switch.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import gleampy
from gleampy.io import load_example
from gleampy.validation import GleamValidationError, GleamWarning
from golden_utils import assert_matches_golden

CASE = "metabolic_energy_req_module"
COMPUTED = [
    "metabolic_energy_req_maintenance",
    "metabolic_energy_req_activity",
    "metabolic_energy_req_growth",
    "metabolic_energy_req_lactation",
    "metabolic_energy_req_work",
    "metabolic_energy_req_fibre_production",
    "metabolic_energy_req_egg_deposition",
    "metabolic_energy_req_pregnancy",
    "net_energy_maintenance_digestible_energy_ratio",
    "net_energy_growth_digestible_energy_ratio",
    "metabolic_energy_req_total",
    "ration_intake",
]


def _inputs():
    return (
        load_example("metabolic_energy_req_input_chrt_data.csv"),
        load_example("metabolic_energy_req_input_hrd_data.csv"),
    )


def _run(cohort, herd, **kw):
    return gleampy.run_metabolic_energy_req_module(
        cohort_level_data=cohort, herd_level_data=herd, show_indicator=False, **kw
    )


def test_parity_metabolic_energy_req_module_result():
    cohort, herd = _inputs()
    result = _run(cohort, herd)
    assert_matches_golden(result, CASE, "result")


def test_output_columns_and_order():
    cohort, herd = _inputs()
    result = _run(cohort, herd)
    assert list(result.columns) == list(cohort.columns) + COMPUTED
    pd.testing.assert_frame_equal(result[list(cohort.columns)], cohort)


def test_inputs_are_not_mutated():
    cohort, herd = _inputs()
    cohort_before, herd_before = cohort.copy(deep=True), herd.copy(deep=True)
    _run(cohort.drop(columns="is_egg_producing").iloc[:-6], herd.iloc[:-1])
    _run(cohort, herd)
    pd.testing.assert_frame_equal(cohort, cohort_before)
    pd.testing.assert_frame_equal(herd, herd_before)


def test_parity_with_validation_disabled():
    cohort, herd = _inputs()
    with pytest.warns(GleamWarning, match="Input validation has been turned off"):
        result = _run(cohort, herd, validate_inputs=False)
    assert_matches_golden(result, CASE, "result")


def test_progress_indicator(capsys):
    cohort, herd = _inputs()
    gleampy.run_metabolic_energy_req_module(cohort, herd, show_indicator=True)
    err = capsys.readouterr().err
    assert "Metabolic energy requirements calculation complete." in err


def test_is_egg_producing_added_when_absent_without_chickens():
    cohort, herd = _inputs()
    no_chk_cohort = cohort[cohort["herd_id"] != 13].drop(columns="is_egg_producing")
    no_chk_herd = herd[herd["herd_id"] != 13]
    full = _run(cohort, herd)
    for validate in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GleamWarning)
            result = _run(no_chk_cohort, no_chk_herd, validate_inputs=validate)
        expected_cols = [c for c in cohort.columns if c != "is_egg_producing"] + ["is_egg_producing"] + COMPUTED
        assert list(result.columns) == expected_cols
        assert result["is_egg_producing"].isna().all()
        np.testing.assert_array_equal(
            result[COMPUTED].to_numpy(), full.loc[full["herd_id"] != 13, COMPUTED].to_numpy()
        )


def test_optional_columns_only_needed_where_used():
    # R evaluates herd look-ups lazily: CHK-only columns (temperature, egg data)
    # and cohort_stock_size are not needed when no row uses them.
    cohort, herd = _inputs()
    keep = cohort["herd_id"] != 13
    c = cohort[keep].drop(columns=["cohort_stock_size", "nondemo_productive_phase_id", "is_egg_producing"])
    h = herd[herd["herd_id"] != 13].drop(
        columns=["average_annual_temperature", "egg_output_human_consumption", "egg_average_weight"]
    )
    result = _run(c, h)
    full = _run(cohort, herd)
    np.testing.assert_array_equal(result[COMPUTED].to_numpy(), full.loc[keep, COMPUTED].to_numpy())


def test_column_needed_by_some_row_must_be_present():
    cohort, herd = _inputs()
    # cohort_stock_size is used by the laying CHK row: R fails with "object
    # not found"; the port raises GleamValidationError.
    with pytest.raises(
        GleamValidationError, match='Missing required columns in `cohort_level_data`: "cohort_stock_size"'
    ):
        _run(cohort.drop(columns="cohort_stock_size"), herd)
    # Pig herds need lactation_duration (not a required column when validation is off).
    with pytest.warns(GleamWarning), pytest.raises(
        GleamValidationError, match='Missing required columns in `herd_level_data`: "lactation_duration"'
    ):
        _run(cohort, herd.drop(columns="lactation_duration"), validate_inputs=False)


def test_run_validation_required_columns():
    cohort, herd = _inputs()
    with pytest.raises(GleamValidationError, match='Missing required columns in `cohort_level_data`: "daily_weight_gain"'):
        _run(cohort.drop(columns="daily_weight_gain"), herd)
    with pytest.raises(GleamValidationError, match='Missing required columns in `herd_level_data`: "litter_size"'):
        _run(cohort, herd.drop(columns="litter_size"))
    with pytest.raises(GleamValidationError, match="average_annual_temperature"):
        _run(cohort, herd.drop(columns="average_annual_temperature"))
    with pytest.raises(GleamValidationError, match='"egg_average_weight"'):
        _run(cohort, herd.drop(columns="egg_average_weight"))


def test_run_validation_structure():
    cohort, herd = _inputs()
    with pytest.raises(GleamValidationError, match="must contain at least one row"):
        _run(cohort.iloc[:0], herd)
    with pytest.raises(GleamValidationError, match="exactly 6 rows"):
        _run(cohort.drop(index=0), herd)
    with pytest.raises(GleamValidationError, match="Each herd_id must appear exactly once"):
        _run(cohort, pd.concat([herd, herd.iloc[[0]]], ignore_index=True))
    bad_species = herd.copy()
    bad_species.loc[0, "species_short"] = "XXX"
    with pytest.raises(GleamValidationError, match="Invalid `species_short` values"):
        _run(cohort, bad_species)
    with pytest.raises(GleamValidationError, match="not found in `cohort_level_data`"):
        _run(cohort[cohort["herd_id"] != 13], herd)


def test_run_validation_numeric_consistency():
    cohort, herd = _inputs()
    c = cohort.copy()
    c.loc[1, "high_activity_fraction"] = 0.99999
    c.loc[1, "low_activity_fraction"] = 0.5
    with pytest.raises(GleamValidationError, match=r'must be <= 1\. Violation\(s\): "1 / FJ"'):
        _run(c, herd)

    c = cohort.copy()
    c.loc[2, "live_weight_cohort_initial"] = 500.0
    with pytest.raises(GleamValidationError, match=r'must hold\. Violation\(s\): "1 / FS"'):
        _run(c, herd)

    h = herd.copy()
    h.loc[1, "live_weight_at_birth"] = 300.0
    with pytest.raises(GleamValidationError, match=r"Violation\(s\) for herd_id: 2"):
        _run(cohort, h)


# --------------------------------------------------------------------------
# R-backed regression: non-demographic branches of every species and the CHK
# cold-temperature branch
# --------------------------------------------------------------------------
#
# The golden cases only reach CTL MN, PGS FN / MN and CHK FN / MN, and every
# bundled CHK herd is at 20 C, above the 18.89 C lower critical temperature.
# This scenario is built from the module example: one herd per species
# (1 CTL, 3 SHP, 5 GTS, 7 BFL, 9 PGS, 11 CML, 13 CHK), the existing FN / MN
# rows removed, FN / MN rows for phases 1 and 2 copied from the FS / MS rows
# (CHK FN phase 2 laying), the CHK FA flag set to FALSE (R still gives FA egg
# energy) and the CHK herd at 10 C (juvenile maintenance
# 0.3866 + 0.0282 * (18.89 - 10)). The reference values were computed with R
# (run_metabolic_energy_req_module on the same tables, written with
# sprintf("%a")): herd, cohort, phase ("-" for NA), then the 12 outputs.

_R_NONDEMO_COLD = """
13 FA - 0x1.d90432cfa7caep-1 0x1.7a69c23fb96f2p-5 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x1.a0c89bca602bcp-2 0x0p+0 NA NA 0x1.60878e6c69bbep+0 0x1.8da2ead22cba4p-4
13 FJ - 0x1.464bec679cc75p-1 0x1.050989ec7d6c4p-5 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.569c8506649e1p-1 0x1.c6eb71f55e4cp-6
13 FS - 0x1.464bec679cc75p-1 0x1.050989ec7d6c4p-5 0x1.b3fdb1c897f7p-3 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.c39bf1788a9bdp-1 0x1.fd64bad2dc696p-5
13 MA - 0x1.2575fe0113093p+0 0x1.d58996681e752p-5 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.34224ab453fcep+0 0x1.5b8f7196f9662p-4
13 MJ - 0x1.464bec679cc75p-1 0x1.050989ec7d6c4p-5 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.569c8506649e1p-1 0x1.c6eb71f55e4cp-6
13 MS - 0x1.464bec679cc75p-1 0x1.050989ec7d6c4p-5 0x1.244fe8a67d291p-2 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.e8c47959a332ap-1 0x1.13a734494afc2p-4
1 FN 1 0x1.f7352f2b1e3aep+4 0x1.b4b44835d2fdfp-12 0x1.6078b7c03d6c9p+3 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x1.100ef035467b7p-1 0x1.58b3afd116efep-2 0x1.0312a9ac13c65p+7 0x1.b941bdf4d976dp+2
1 FN 2 0x1.f7352f2b1e3aep+4 0x1.b4b44835d2fdfp-12 0x1.6078b7c03d6c9p+3 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x1.100ef035467b7p-1 0x1.58b3afd116efep-2 0x1.0312a9ac13c65p+7 0x1.b941bdf4d976dp+2
1 MN 1 0x1.eb182b20bc445p+4 0x1.aa31180ff46c2p-12 0x1.780f071f6b223p+3 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x1.100ef035467b7p-1 0x1.58b3afd116efep-2 0x1.053a98cf26c7bp+7 0x1.bcedcded70de6p+2
1 MN 2 0x1.eb182b20bc445p+4 0x1.aa31180ff46c2p-12 0x1.780f071f6b223p+3 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x1.100ef035467b7p-1 0x1.58b3afd116efep-2 0x1.053a98cf26c7bp+7 0x1.bcedcded70de6p+2
3 FN 1 0x1.c40b076da848bp+1 0x1.028db2a9b7a93p-2 0x1.25cf08e72f4edp-1 0x0p+0 0x0p+0 0x1.93fd31cc193fdp-4 0x0p+0 0x0p+0 0x1.05d2d97e91528p-1 0x1.37e3118e2bb1ep-2 0x1.de3b7796e9b08p+3 0x1.a419d0a39a351p-1
3 FN 2 0x1.c40b076da848bp+1 0x1.028db2a9b7a93p-2 0x1.25cf08e72f4edp-1 0x0p+0 0x0p+0 0x1.93fd31cc193fdp-4 0x0p+0 0x0p+0 0x1.05d2d97e91528p-1 0x1.37e3118e2bb1ep-2 0x1.de3b7796e9b08p+3 0x1.a419d0a39a351p-1
3 MN 1 0x1.a50ca91724298p+1 0x1.d66201fa417f4p-3 0x1.ea60f8000d9d1p-3 0x0p+0 0x0p+0 0x1.93fd31cc193fdp-4 0x0p+0 0x0p+0 0x1.05d2d97e91528p-1 0x1.37e3118e2bb1ep-2 0x1.8dcae1432b245p+3 0x1.5d705b495e424p-1
3 MN 2 0x1.a50ca91724298p+1 0x1.d66201fa417f4p-3 0x1.ea60f8000d9d1p-3 0x0p+0 0x0p+0 0x1.93fd31cc193fdp-4 0x0p+0 0x0p+0 0x1.05d2d97e91528p-1 0x1.37e3118e2bb1ep-2 0x1.8dcae1432b245p+3 0x1.5d705b495e424p-1
5 FN 1 0x1.cf24ad13d2085p+1 0x1.a6080ce9c85b2p-3 0x1.6d17c2df6cfc6p-1 0x0p+0 0x0p+0 0x1.e4c96ef4eb197p-4 0x0p+0 0x0p+0 0x1.0ca7e87a435abp-1 0x1.4dbba3e61204ep-2 0x1.cbaed821deb81p+3 0x1.90a38acf9f9cbp-1
5 FN 2 0x1.cf24ad13d2085p+1 0x1.a6080ce9c85b2p-3 0x1.6d17c2df6cfc6p-1 0x0p+0 0x0p+0 0x1.e4c96ef4eb197p-4 0x0p+0 0x0p+0 0x1.0ca7e87a435abp-1 0x1.4dbba3e61204ep-2 0x1.cbaed821deb81p+3 0x1.90a38acf9f9cbp-1
5 MN 1 0x1.8d047451df4acp+1 0x1.57aa989a6b25cp-3 0x1.6877faad0b111p-2 0x0p+0 0x0p+0 0x1.e4c96ef4eb197p-4 0x0p+0 0x0p+0 0x1.0ca7e87a435abp-1 0x1.4dbba3e61204ep-2 0x1.6686633cdcabep+3 0x1.38795d8c16a56p-1
5 MN 2 0x1.8d047451df4acp+1 0x1.57aa989a6b25cp-3 0x1.6877faad0b111p-2 0x0p+0 0x0p+0 0x1.e4c96ef4eb197p-4 0x0p+0 0x0p+0 0x1.0ca7e87a435abp-1 0x1.4dbba3e61204ep-2 0x1.6686633cdcabep+3 0x1.38795d8c16a56p-1
7 FN 1 0x1.783ba8ddd5d25p+4 0x1.a28a29b5c087fp-3 0x1.c338a8e810fa4p+2 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x1.0c9001029a92ep-1 0x1.4d6ecfd98e9cep-2 0x1.86df7b567a949p+6 0x1.4b1ba4dfc14a8p+2
7 FN 2 0x1.783ba8ddd5d25p+4 0x1.a28a29b5c087fp-3 0x1.c338a8e810fa4p+2 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x1.0c9001029a92ep-1 0x1.4d6ecfd98e9cep-2 0x1.86df7b567a949p+6 0x1.4b1ba4dfc14a8p+2
7 MN 1 0x1.6cbac0e019598p+4 0x1.95be1b9f557ecp-3 0x1.b7fa0c1dacbbep+2 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x1.0c9001029a92ep-1 0x1.4d6ecfd98e9cep-2 0x1.7ba33615e1e6cp+6 0x1.419728b8372cap+2
7 MN 2 0x1.6cbac0e019598p+4 0x1.95be1b9f557ecp-3 0x1.b7fa0c1dacbbep+2 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x1.0c9001029a92ep-1 0x1.4d6ecfd98e9cep-2 0x1.7ba33615e1e6cp+6 0x1.419728b8372cap+2
9 FN 1 0x1.4d03dc8634088p+3 0x0p+0 0x1.dd0041b7f2fe3p+3 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.95020f1f13836p+4 0x1.c8d4646f36396p+0
9 FN 2 0x1.4d03dc8634088p+3 0x0p+0 0x1.dd0041b7f2fe3p+3 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.95020f1f13836p+4 0x1.c8d4646f36396p+0
9 MN 1 0x1.4a3b68c4d9215p+3 0x0p+0 0x1.1a431672b8e43p+4 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.bf60cad52574ep+4 0x1.f89efcc9cc392p+0
9 MN 2 0x1.4a3b68c4d9215p+3 0x0p+0 0x1.1a431672b8e43p+4 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.bf60cad52574ep+4 0x1.f89efcc9cc392p+0
11 FN 1 0x1.a313914f59d72p+4 0x1.4f42daa5e178fp+1 0x1.17550d8357dc8p+2 0x0p+0 0x0p+0 0x1.392b77f784b08p-3 0x0p+0 0x0p+0 NA NA 0x1.0aa1c37a6d836p+5 0x1.e249400ebd9d6p+1
11 FN 2 0x1.a313914f59d72p+4 0x1.4f42daa5e178fp+1 0x1.17550d8357dc8p+2 0x0p+0 0x0p+0 0x1.392b77f784b08p-3 0x0p+0 0x0p+0 NA NA 0x1.0aa1c37a6d836p+5 0x1.e249400ebd9d6p+1
11 MN 1 0x1.b6e620d96fe6fp+4 0x1.5f1e80adf31f3p+1 0x1.3b73e9be421d9p+2 0x0p+0 0x0p+0 0x1.392b77f784b08p-3 0x0p+0 0x0p+0 NA NA 0x1.1a0ca12756eddp+5 0x1.fe2c605d62683p+1
11 MN 2 0x1.b6e620d96fe6fp+4 0x1.5f1e80adf31f3p+1 0x1.3b73e9be421d9p+2 0x0p+0 0x0p+0 0x1.392b77f784b08p-3 0x0p+0 0x0p+0 NA NA 0x1.1a0ca12756eddp+5 0x1.fe2c605d62683p+1
13 FN 1 0x1.464bec679cc75p-1 0x1.050989ec7d6c4p-5 0x1.b3fdb1c897f7p-3 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.c39bf1788a9bdp-1 0x1.fd64bad2dc696p-5
13 FN 2 0x1.1b21553d5ea03p-1 0x1.c50221fbca99fp-6 0x1.2d17c11ce7adcp-2 0x0p+0 0x0p+0 0x0p+0 0x1.bffb24134c025p-3 0x0p+0 NA NA 0x1.17ea07f041e64p+0 0x1.3bbad2e71d4dbp-4
13 MN 1 0x1.464bec679cc75p-1 0x1.050989ec7d6c4p-5 0x1.244fe8a67d291p-2 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.e8c47959a332ap-1 0x1.13a734494afc2p-4
13 MN 2 0x1.464bec679cc75p-1 0x1.050989ec7d6c4p-5 0x1.244fe8a67d291p-2 0x0p+0 0x0p+0 0x0p+0 0x0p+0 0x0p+0 NA NA 0x1.e8c47959a332ap-1 0x1.13a734494afc2p-4
"""


def _nondemo_cold_inputs():
    cohort, herd = _inputs()
    keep = [1, 3, 5, 7, 9, 11, 13]
    demo = cohort[cohort["herd_id"].isin(keep) & ~cohort["cohort_short"].isin(["FN", "MN"])]
    rows = []
    for hid in keep:
        for src, dst in (("FS", "FN"), ("MS", "MN")):
            for phase in (1, 2):
                r = demo[(demo["herd_id"] == hid) & (demo["cohort_short"] == src)].copy()
                r["cohort_short"] = dst
                r["nondemo_productive_phase_id"] = phase
                r["is_egg_producing"] = hid == 13 and dst == "FN" and phase == 2
                rows.append(r)
    nondemo = pd.concat(rows, ignore_index=True)
    full = pd.concat([demo, nondemo], ignore_index=True)
    full.loc[(full["herd_id"] == 13) & (full["cohort_short"] == "FA"), "is_egg_producing"] = False
    herd = herd[herd["herd_id"].isin(keep)].reset_index(drop=True)
    herd.loc[herd["herd_id"] == 13, "average_annual_temperature"] = 10.0
    return full, nondemo, herd


def _r_reference():
    keys, values = [], []
    for line in _R_NONDEMO_COLD.strip().splitlines():
        hid, co, phase, *vals = line.split()
        keys.append((int(hid), co, np.nan if phase == "-" else float(phase)))
        values.append([np.nan if v == "NA" else float.fromhex(v) for v in vals])
    return keys, np.array(values)


def _assert_matches_r(result, keys, expected):
    got_keys = list(
        zip(result["herd_id"], result["cohort_short"], result["nondemo_productive_phase_id"].astype(float))
    )
    assert len(got_keys) == len(keys)
    for (h1, c1, p1), (h2, c2, p2) in zip(got_keys, keys):
        assert (h1, c1) == (h2, c2) and (p1 == p2 or (np.isnan(p1) and np.isnan(p2)))
    # R's ^ uses powl on this platform and numpy uses pow: results agree to a
    # few ulp (largest relative difference seen: 1.1e-15), hence rtol=1e-13.
    np.testing.assert_allclose(result[COMPUTED].to_numpy(dtype=float), expected, rtol=1e-13, atol=0)


@pytest.mark.parametrize("validate", [True, False])
def test_nondemographic_branches_and_chk_cold_branch_match_r(validate):
    full, _, herd = _nondemo_cold_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        result = _run(full, herd, validate_inputs=validate)
    keys, expected = _r_reference()
    sel = result["nondemo_productive_phase_id"].notna().to_numpy() | (result["herd_id"] == 13).to_numpy()
    _assert_matches_r(result[sel], keys, expected)
    # The CHK juveniles are in the cold branch.
    fj = result[(result["herd_id"] == 13) & (result["cohort_short"] == "FJ")].iloc[0]
    assert fj["metabolic_energy_req_maintenance"] == pytest.approx(0.3866 + 0.0282 * (18.89 - 10), rel=1e-15)


@pytest.mark.parametrize("validate", [True, False])
def test_nondemographic_only_herds_with_minimal_herd_table_match_r(validate):
    # Only FN / MN rows and a herd table with just the columns those rows read
    # (R gives exactly the values of the full scenario's FN / MN rows).
    _, nondemo, herd = _nondemo_cold_inputs()
    minimal = herd[
        [
            "herd_id", "species_short", "average_annual_temperature", "fibre_yield_year",
            "parturition_rate", "egg_output_human_consumption", "egg_average_weight",
        ]
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GleamWarning)
        result = _run(nondemo, minimal, validate_inputs=validate)
    keys, expected = _r_reference()
    keep = [i for i, k in enumerate(keys) if not np.isnan(k[2])]
    _assert_matches_r(result, [keys[i] for i in keep], expected[keep])
    # fibre_yield_year is read by the SHP / GTS / CML FN / MN rows, as in R.
    with pytest.raises(GleamValidationError, match='Missing required columns in `herd_level_data`: "fibre_yield_year"'):
        _run(nondemo, minimal.drop(columns="fibre_yield_year"), validate_inputs=validate)
