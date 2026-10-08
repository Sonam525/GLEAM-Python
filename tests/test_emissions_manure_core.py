"""Port of ``tests/testthat/test-emissions_manure_core.R`` plus vectorisation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gleampy import (
    GleamValidationError,
    calc_ch4_manure,
    calc_n2o_manure_direct,
    calc_n2o_manure_leaching,
    calc_n2o_manure_total,
    calc_n2o_manure_volatilization,
    calc_volatile_solids,
    validation_disabled,
)

REL = 1.5e-8  # testthat's default tolerance


# ---- test calc_volatile_solids ----------------------------------------------


def test_calc_volatile_solids_produces_expected_results_1():
    result = calc_volatile_solids(
        ration_intake=5,
        ration_digestibility_fraction=0.6,
        ration_urinary_energy_fraction=0.04,
        ration_ash=0.08,
    )
    assert isinstance(result, float)  # expect_length(result, 1)
    assert result >= 0
    assert result == pytest.approx(5 * (1 - 0.6 + 0.04) * (1 - 0.08), rel=REL)


def test_calc_volatile_solids_produces_expected_results_2():
    result = calc_volatile_solids(
        ration_intake=4,
        ration_digestibility_fraction=0.7,
        ration_urinary_energy_fraction=0.02,
        ration_ash=0.06,
    )
    assert result == pytest.approx(4 * (1 - 0.7 + 0.02) * (1 - 0.06), rel=REL)


@pytest.mark.parametrize(
    "args, pattern",
    [
        ((-1, 0.6, 0.04, 0.08), "`ration_intake`"),
        ((1, -0.1, 0.04, 0.08), "`ration_digestibility_fraction`"),
        ((1, 1.1, 0.04, 0.08), "`ration_digestibility_fraction`"),
        ((1, 0.6, -0.01, 0.08), "`ration_urinary_energy_fraction`"),
        ((1, 0.6, 1.1, 0.08), "`ration_urinary_energy_fraction`"),
        ((1, 0.6, 0.04, -0.01), "`ration_ash`"),
        ((1, 0.6, 0.04, 1.1), "`ration_ash`"),
    ],
)
def test_calc_volatile_solids_validates_inputs(args, pattern):
    with pytest.raises(GleamValidationError, match=pattern + ".*out of range"):
        calc_volatile_solids(*args)


# ---- test calc_ch4_manure -----------------------------------------------------


def test_calc_ch4_manure_computes_methane_by_mms_group():
    volatile_solids = 2
    ratio = 0.67
    mms_burned = dict(
        manure_management_system_fraction=0.2,
        methane_conversion_factor_mcf=10,
        ch4_max_producing_capacity_bo=0.13,
    )
    mms_pasture = dict(
        manure_management_system_fraction=0.3,
        methane_conversion_factor_mcf=0.47,
        ch4_max_producing_capacity_bo=0.19,
    )
    mms_drylot = dict(
        manure_management_system_fraction=0.5,
        methane_conversion_factor_mcf=2,
        ch4_max_producing_capacity_bo=0.13,
    )

    result = calc_ch4_manure(
        ratio_m3CH4_to_kgCH4=ratio,
        volatile_solids=volatile_solids,
        mms_burned=mms_burned,
        mms_pasture=mms_pasture,
        mms_drylot=mms_drylot,
    )

    expected_pasture = volatile_solids * ratio * 0.3 * (0.47 / 100) * 0.19
    expected_burned = volatile_solids * ratio * 0.2 * (10 / 100) * 0.13
    expected_other = volatile_solids * ratio * (0.5 * 2 * 0.13) / 100

    assert list(result) == [
        "ch4_manure_pasture", "ch4_manure_burned", "ch4_manure_other", "ch4_manure_all_noburn",
    ]
    assert result["ch4_manure_pasture"] == pytest.approx(expected_pasture, rel=REL)
    assert result["ch4_manure_burned"] == pytest.approx(expected_burned, rel=REL)
    assert result["ch4_manure_other"] == pytest.approx(expected_other, rel=REL)
    assert result["ch4_manure_all_noburn"] == pytest.approx(expected_pasture + expected_other, rel=REL)


def test_calc_ch4_manure_errors_when_mms_fractions_do_not_sum_to_1():
    with pytest.raises(GleamValidationError, match=r"sum of all MMS fractions must be equal to 1 \(current sum: 0.8\)"):
        calc_ch4_manure(
            volatile_solids=1,
            mms_pasture=dict(
                manure_management_system_fraction=0.6,
                methane_conversion_factor_mcf=2,
                ch4_max_producing_capacity_bo=0.13,
            ),
            mms_solid=dict(
                manure_management_system_fraction=0.2,
                methane_conversion_factor_mcf=5,
                ch4_max_producing_capacity_bo=0.13,
            ),
        )


def test_calc_ch4_manure_validates_mms_inputs():
    with pytest.raises(GleamValidationError, match="`manure_management_system_fraction` = 1.2 is out of range"):
        calc_ch4_manure(volatile_solids=1, mms_pasture=dict(
            manure_management_system_fraction=1.2,
            methane_conversion_factor_mcf=2,
            ch4_max_producing_capacity_bo=0.13,
        ))
    with pytest.raises(GleamValidationError, match="`methane_conversion_factor_mcf` = 120 is out of range"):
        calc_ch4_manure(volatile_solids=1, mms_pasture=dict(
            manure_management_system_fraction=1,
            methane_conversion_factor_mcf=120,
            ch4_max_producing_capacity_bo=0.13,
        ))
    with pytest.raises(GleamValidationError, match="Each MMS must contain exactly these named values"):
        calc_ch4_manure(volatile_solids=1, mms_pasture=dict(
            manure_management_system_fraction=1, methane_conversion_factor_mcf=2
        ))


# ---- test calc_n2o_manure_direct ------------------------------------------------


def test_calc_n2o_manure_direct_computes_direct_n2o_by_mms_group():
    n_excretion = 0.9
    mms_burned = dict(manure_management_system_fraction=0.2, n2o_ef3=0)
    mms_pasture = dict(manure_management_system_fraction=0.3, n2o_ef3=0.02)
    mms_drylot = dict(manure_management_system_fraction=0.5, n2o_ef3=0.01)

    result = calc_n2o_manure_direct(
        nitrogen_excretion=n_excretion,
        mms_burned=mms_burned,
        mms_pasture=mms_pasture,
        mms_drylot=mms_drylot,
    )

    expected_pasture = n_excretion * (44 / 28) * 0.3 * 0.02
    expected_burned = n_excretion * (44 / 28) * 0.2 * 0
    expected_other = n_excretion * (44 / 28) * (0.5 * 0.01)

    assert result["n2o_manure_pasture_direct"] == pytest.approx(expected_pasture, rel=REL)
    assert result["n2o_manure_burned_direct"] == pytest.approx(expected_burned, rel=REL)
    assert result["n2o_manure_other_direct"] == pytest.approx(expected_other, rel=REL)
    assert result["n2o_manure_all_noburn_direct"] == pytest.approx(expected_pasture + expected_other, rel=REL)


def test_calc_n2o_manure_direct_validates_mms_inputs():
    with pytest.raises(GleamValidationError, match="`manure_management_system_fraction` = 1.1"):
        calc_n2o_manure_direct(nitrogen_excretion=0.9, mms_pasture=dict(
            manure_management_system_fraction=1.1, n2o_ef3=0.02
        ))
    with pytest.raises(GleamValidationError, match="`n2o_ef3` = -0.01"):
        calc_n2o_manure_direct(nitrogen_excretion=0.9, mms_pasture=dict(
            manure_management_system_fraction=1, n2o_ef3=-0.01
        ))


# ---- test calc_n2o_manure_volatilization ---------------------------------------


def test_calc_n2o_manure_volatilization_computes_indirect_n2o_by_mms_group():
    n_excretion = 0.9
    mms_burned = dict(manure_management_system_fraction=0.2, n2o_ef4=0.14, nitrogen_fracgas=0)
    mms_pasture = dict(manure_management_system_fraction=0.3, n2o_ef4=0.14, nitrogen_fracgas=0.21)
    mms_drylot = dict(manure_management_system_fraction=0.5, n2o_ef4=0.14, nitrogen_fracgas=0.3)

    result = calc_n2o_manure_volatilization(
        nitrogen_excretion=n_excretion,
        mms_burned=mms_burned,
        mms_pasture=mms_pasture,
        mms_drylot=mms_drylot,
    )

    expected_pasture = n_excretion * (44 / 28) * 0.3 * 0.21 * 0.14
    expected_burned = n_excretion * (44 / 28) * 0.2 * 0 * 0.14
    expected_other = n_excretion * (44 / 28) * (0.5 * 0.3 * 0.14)

    assert result["n2o_manure_pasture_vol"] == pytest.approx(expected_pasture, rel=REL)
    assert result["n2o_manure_burned_vol"] == pytest.approx(expected_burned, rel=REL)
    assert result["n2o_manure_other_vol"] == pytest.approx(expected_other, rel=REL)
    assert result["n2o_manure_all_noburn_vol"] == pytest.approx(expected_pasture + expected_other, rel=REL)


def test_calc_n2o_manure_volatilization_validates_mms_inputs():
    with pytest.raises(GleamValidationError, match="`manure_management_system_fraction` = 1.1"):
        calc_n2o_manure_volatilization(nitrogen_excretion=0.9, mms_pasture=dict(
            manure_management_system_fraction=1.1, n2o_ef4=0.14, nitrogen_fracgas=0.21
        ))
    with pytest.raises(GleamValidationError, match="`n2o_ef4` = -0.01"):
        calc_n2o_manure_volatilization(nitrogen_excretion=0.9, mms_pasture=dict(
            manure_management_system_fraction=1, n2o_ef4=-0.01, nitrogen_fracgas=0.21
        ))
    with pytest.raises(GleamValidationError, match="`nitrogen_fracgas` = 1.2"):
        calc_n2o_manure_volatilization(nitrogen_excretion=0.9, mms_pasture=dict(
            manure_management_system_fraction=1, n2o_ef4=0.14, nitrogen_fracgas=1.2
        ))


# ---- test calc_n2o_manure_leaching ----------------------------------------------


def test_calc_n2o_manure_leaching_computes_indirect_n2o_by_mms_group():
    n_excretion = 0.9
    mms_burned = dict(manure_management_system_fraction=0.2, n2o_ef5=0.011, nitrogen_fracleach=0)
    mms_pasture = dict(manure_management_system_fraction=0.3, n2o_ef5=0.011, nitrogen_fracleach=0.24)
    mms_drylot = dict(manure_management_system_fraction=0.5, n2o_ef5=0.011, nitrogen_fracleach=0.035)

    result = calc_n2o_manure_leaching(
        nitrogen_excretion=n_excretion,
        mms_burned=mms_burned,
        mms_pasture=mms_pasture,
        mms_drylot=mms_drylot,
    )

    expected_pasture = n_excretion * (44 / 28) * 0.3 * 0.24 * 0.011
    expected_burned = n_excretion * (44 / 28) * 0.2 * 0 * 0.011
    expected_other = n_excretion * (44 / 28) * (0.5 * 0.035 * 0.011)

    assert result["n2o_manure_pasture_leach"] == pytest.approx(expected_pasture, rel=REL)
    assert result["n2o_manure_burned_leach"] == pytest.approx(expected_burned, rel=REL)
    assert result["n2o_manure_other_leach"] == pytest.approx(expected_other, rel=REL)
    assert result["n2o_manure_all_noburn_leach"] == pytest.approx(expected_pasture + expected_other, rel=REL)


def test_calc_n2o_manure_leaching_validates_mms_inputs():
    with pytest.raises(GleamValidationError, match="`manure_management_system_fraction` = 1.1"):
        calc_n2o_manure_leaching(nitrogen_excretion=0.9, mms_pasture=dict(
            manure_management_system_fraction=1.1, n2o_ef5=0.011, nitrogen_fracleach=0.24
        ))
    with pytest.raises(GleamValidationError, match="`n2o_ef5` = -0.01"):
        calc_n2o_manure_leaching(nitrogen_excretion=0.9, mms_pasture=dict(
            manure_management_system_fraction=1, n2o_ef5=-0.01, nitrogen_fracleach=0.24
        ))
    with pytest.raises(GleamValidationError, match="`nitrogen_fracleach` = 1.2"):
        calc_n2o_manure_leaching(nitrogen_excretion=0.9, mms_pasture=dict(
            manure_management_system_fraction=1, n2o_ef5=0.011, nitrogen_fracleach=1.2
        ))


# ---- test calc_n2o_manure_total -------------------------------------------------


def test_calc_n2o_manure_total_aggregates_direct_and_indirect_n2o():
    result = calc_n2o_manure_total(
        n2o_manure_pasture_vol=0.0129,
        n2o_manure_pasture_leach=0.0012,
        n2o_manure_burned_vol=0,
        n2o_manure_burned_leach=0,
        n2o_manure_other_vol=0.052,
        n2o_manure_other_leach=0.00027,
        n2o_manure_pasture_direct=0.009,
        n2o_manure_burned_direct=0,
        n2o_manure_other_direct=0.01033,
    )

    assert result["n2o_manure_pasture_indirect"] == pytest.approx(0.0129 + 0.0012, rel=REL)
    assert result["n2o_manure_burned_indirect"] == pytest.approx(0, rel=REL)
    assert result["n2o_manure_other_indirect"] == pytest.approx(0.052 + 0.00027, rel=REL)
    assert result["n2o_manure_pasture_total"] == pytest.approx(0.0129 + 0.0012 + 0.009, rel=REL)
    assert result["n2o_manure_burned_total"] == pytest.approx(0, rel=REL)
    assert result["n2o_manure_other_total"] == pytest.approx(0.052 + 0.00027 + 0.01033, rel=REL)


def test_calc_n2o_manure_total_validates_scalar_numeric_inputs():
    with pytest.raises(GleamValidationError, match="`n2o_manure_pasture_vol` must be a single numeric value"):
        calc_n2o_manure_total(
            n2o_manure_pasture_vol="0.01",
            n2o_manure_pasture_leach=0.0012,
            n2o_manure_burned_vol=0,
            n2o_manure_burned_leach=0,
            n2o_manure_other_vol=0.052,
            n2o_manure_other_leach=0.00027,
            n2o_manure_pasture_direct=0.009,
            n2o_manure_burned_direct=0,
            n2o_manure_other_direct=0.01033,
        )


# ---- additional checks (Python port) ---------------------------------------------


def test_other_mms_validation_messages_match_r():
    with pytest.raises(GleamValidationError, match="^At least one manure management system must be provided.$"):
        calc_ch4_manure(volatile_solids=1)
    with pytest.raises(GleamValidationError, match="^Each MMS argument must be a numeric vector.$"):
        calc_ch4_manure(volatile_solids=1, mms_pasture="a")
    with pytest.raises(GleamValidationError, match="^Each MMS argument must be a numeric vector.$"):
        calc_n2o_manure_direct(nitrogen_excretion=1, mms_pasture=None)
    with pytest.raises(GleamValidationError, match="^MMS values must not contain missing values.$"):
        calc_n2o_manure_direct(
            nitrogen_excretion=1, mms_pasture=dict(manure_management_system_fraction=1, n2o_ef3=np.nan)
        )
    with pytest.raises(
        GleamValidationError,
        match=r"^Each MMS must contain exactly these named values: \* manure_management_system_fraction "
        r"and n2o_ef3$",
    ):
        calc_n2o_manure_direct(nitrogen_excretion=1, mms_pasture=[1, 0.02])
    with pytest.raises(GleamValidationError, match=r"current sum: 0.8000000001\)"):
        calc_n2o_manure_direct(
            nitrogen_excretion=1,
            mms_pasture=dict(manure_management_system_fraction=0.6000000001, n2o_ef3=0.02),
            mms_solid=dict(manure_management_system_fraction=0.2, n2o_ef3=0.02),
        )
    with pytest.raises(GleamValidationError, match="`nitrogen_excretion` must be a single numeric value"):
        calc_n2o_manure_direct(
            nitrogen_excretion=np.nan, mms_pasture=dict(manure_management_system_fraction=1, n2o_ef3=0.02)
        )
    with pytest.raises(GleamValidationError, match="`ratio_m3CH4_to_kgCH4`"):
        calc_ch4_manure(
            0, 1, mms_pasture=dict(
                manure_management_system_fraction=1,
                methane_conversion_factor_mcf=2,
                ch4_max_producing_capacity_bo=0.13,
            )
        )


def test_named_series_mms_and_defaults():
    # R named numeric vectors map naturally to pandas Series.
    mms = pd.Series({"manure_management_system_fraction": 1.0, "n2o_ef3": 0.02})
    res = calc_n2o_manure_direct(nitrogen_excretion=0.9, mms_solid=mms)
    assert res["n2o_manure_other_direct"] == pytest.approx(0.9 * (44 / 28) * 1.0 * 0.02, rel=1e-15)
    assert res["n2o_manure_pasture_direct"] == 0.0
    assert res["n2o_manure_burned_direct"] == 0.0
    # missing required argument
    with pytest.raises(TypeError, match="nitrogen_excretion"):
        calc_n2o_manure_direct(mms_solid=mms)


def test_absent_groups_give_zero_and_validation_off_propagates_na():
    with validation_disabled():
        res = calc_ch4_manure(volatile_solids=np.nan)
        assert res == {
            "ch4_manure_pasture": 0.0, "ch4_manure_burned": 0.0,
            "ch4_manure_other": 0.0, "ch4_manure_all_noburn": 0.0,
        }
        res = calc_n2o_manure_leaching(
            nitrogen_excretion=1.0,
            mms_pasture=dict(manure_management_system_fraction=np.nan, n2o_ef5=0.011, nitrogen_fracleach=0.2),
        )
        assert np.isnan(res["n2o_manure_pasture_leach"])
        assert res["n2o_manure_other_leach"] == 0.0


# ---- vectorisation: arrays (with per-row MMS sets) == element-wise scalar calls ----


def _random_case(rng, n):
    """Random per-row MMS sets as masked arrays, plus the per-row scalar MMS dicts."""
    names = ["mms_burned", "mms_drylot", "mms_lagoon", "mms_pasture", "mms_solid"]
    fields = {
        "manure_management_system_fraction": None,
        "methane_conversion_factor_mcf": (0, 100),
        "ch4_max_producing_capacity_bo": (0.05, 0.5),
        "n2o_ef3": (0, 0.05),
        "n2o_ef4": (0, 0.05),
        "n2o_ef5": (0, 0.05),
        "nitrogen_fracgas": (0, 1),
        "nitrogen_fracleach": (0, 1),
    }
    present = rng.random((len(names), n)) < 0.6
    present[rng.integers(0, len(names), n), np.arange(n)] = True  # at least one system per row
    present[:, 0] = [False, False, False, True, False]  # a row with only mms_pasture
    present[:, 1] = [False, True, True, False, True]  # a row with only "other" systems
    raw = rng.random((len(names), n)) * present
    frac = raw / raw.sum(axis=0)
    values = {"manure_management_system_fraction": frac}
    for f, rng_ in fields.items():
        if rng_ is not None:
            values[f] = rng.uniform(*rng_, size=(len(names), n))
    wide = {
        name: {f: np.ma.MaskedArray(values[f][i], mask=~present[i]) for f in values}
        for i, name in enumerate(names)
    }
    rows = [
        {name: {f: float(values[f][i, j]) for f in values} for i, name in enumerate(names) if present[i, j]}
        for j in range(n)
    ]
    return wide, rows


def _subset(mms, fields):
    return {name: {f: d[f] for f in fields} for name, d in mms.items()}


@pytest.mark.parametrize(
    "func, scalar_arg, fields",
    [
        (calc_ch4_manure, "volatile_solids",
         ("manure_management_system_fraction", "methane_conversion_factor_mcf", "ch4_max_producing_capacity_bo")),
        (calc_n2o_manure_direct, "nitrogen_excretion", ("manure_management_system_fraction", "n2o_ef3")),
        (calc_n2o_manure_volatilization, "nitrogen_excretion",
         ("manure_management_system_fraction", "n2o_ef4", "nitrogen_fracgas")),
        (calc_n2o_manure_leaching, "nitrogen_excretion",
         ("manure_management_system_fraction", "n2o_ef5", "nitrogen_fracleach")),
    ],
)
def test_mms_functions_vectorised_match_scalar_calls(func, scalar_arg, fields):
    rng = np.random.default_rng(42)
    n = 60
    wide, rows = _random_case(rng, n)
    x = rng.uniform(0, 5, n)
    vec = func(**{scalar_arg: x}, **_subset(wide, fields))
    for key, arr in vec.items():
        assert isinstance(arr, np.ndarray) and arr.shape == (n,)
    for j in range(n):
        sc = func(**{scalar_arg: float(x[j])}, **_subset(rows[j], fields))
        for key, value in sc.items():
            assert isinstance(value, float)
            # Same operations in the same order: bit-identical.
            assert vec[key][j] == value, (key, j)
    # rows 0 (pasture only) and 1 (no pasture / burned) exercise the zero branches
    keys = list(vec)
    assert vec[keys[1]][0] == 0.0 and vec[keys[2]][0] == 0.0
    assert vec[keys[0]][1] == 0.0


def test_mms_functions_broadcast_scalar_fields():
    mms = dict(
        mms_pasture=dict(manure_management_system_fraction=0.4, n2o_ef3=0.02),
        mms_solid=dict(manure_management_system_fraction=0.6, n2o_ef3=[0.005, 0.01, 0.02]),
    )
    vec = calc_n2o_manure_direct(nitrogen_excretion=0.5, **mms)
    for j, ef3 in enumerate([0.005, 0.01, 0.02]):
        sc = calc_n2o_manure_direct(
            nitrogen_excretion=0.5,
            mms_pasture=mms["mms_pasture"],
            mms_solid=dict(manure_management_system_fraction=0.6, n2o_ef3=ef3),
        )
        for k in sc:
            assert vec[k][j] == sc[k]


def test_vectorised_validation_is_row_wise():
    frac = np.ma.MaskedArray([1.0, 0.5], mask=[False, False])
    ok = dict(manure_management_system_fraction=frac, n2o_ef3=0.01)
    other = dict(manure_management_system_fraction=np.ma.MaskedArray([0.0, 0.5], mask=[True, False]), n2o_ef3=0.01)
    res = calc_n2o_manure_direct(nitrogen_excretion=[1.0, 2.0], mms_drylot=ok, mms_solid=other)
    assert res["n2o_manure_other_direct"] == pytest.approx([1.0 * (44 / 28) * 0.01, 2.0 * (44 / 28) * 0.01])
    # second row's fractions no longer sum to one once mms_solid is absent there
    other_absent = dict(
        manure_management_system_fraction=np.ma.MaskedArray([0.0, 0.5], mask=[True, True]), n2o_ef3=0.01
    )
    with pytest.raises(GleamValidationError, match=r"current sum: 0.5\)"):
        calc_n2o_manure_direct(nitrogen_excretion=[1.0, 2.0], mms_drylot=ok, mms_solid=other_absent)
    # a row without any supplied system
    none_row = dict(manure_management_system_fraction=np.ma.MaskedArray([1.0, 1.0], mask=[False, True]), n2o_ef3=0.01)
    with pytest.raises(GleamValidationError, match="At least one manure management system"):
        calc_n2o_manure_direct(nitrogen_excretion=[1.0, 2.0], mms_drylot=none_row)
    # out-of-range / missing values are only checked where the system is supplied
    drylot = dict(manure_management_system_fraction=[0.5, 1.0], n2o_ef3=0.01)
    solid = dict(
        manure_management_system_fraction=np.ma.MaskedArray([0.5, np.nan], mask=[False, True]),
        n2o_ef3=np.ma.MaskedArray([0.02, 5.0], mask=[False, True]),
    )
    res = calc_n2o_manure_direct(nitrogen_excretion=[1.0, 2.0], mms_drylot=drylot, mms_solid=solid)
    assert res["n2o_manure_other_direct"][1] == 2.0 * (44 / 28) * (1.0 * 0.01)
    solid["n2o_ef3"] = np.ma.MaskedArray([5.0, 0.02], mask=[False, True])
    with pytest.raises(GleamValidationError, match="^`n2o_ef3` = 5 is out of range"):
        calc_n2o_manure_direct(nitrogen_excretion=[1.0, 2.0], mms_drylot=drylot, mms_solid=solid)


@pytest.mark.parametrize("seed", [0, 1])
def test_volatile_solids_and_total_vectorised_match_scalar_calls(seed):
    rng = np.random.default_rng(seed)
    n = 40
    intake, dig = rng.uniform(0, 20, n), rng.uniform(0, 1, n)
    ue, ash = rng.uniform(0, 0.2, n), rng.uniform(0, 0.2, n)
    vec = calc_volatile_solids(intake, dig, ue, pd.Series(ash))
    assert isinstance(vec, np.ndarray) and vec.shape == (n,)
    for j in range(n):
        assert vec[j] == calc_volatile_solids(float(intake[j]), float(dig[j]), float(ue[j]), float(ash[j]))

    args = rng.uniform(0, 0.01, (9, n))
    names = [
        "n2o_manure_pasture_vol", "n2o_manure_pasture_leach", "n2o_manure_burned_vol",
        "n2o_manure_burned_leach", "n2o_manure_other_vol", "n2o_manure_other_leach",
        "n2o_manure_pasture_direct", "n2o_manure_burned_direct", "n2o_manure_other_direct",
    ]
    vec = calc_n2o_manure_total(**dict(zip(names, args)))
    for j in range(n):
        sc = calc_n2o_manure_total(**{k: float(a[j]) for k, a in zip(names, args)})
        for k in sc:
            assert vec[k][j] == sc[k]
