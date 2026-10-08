"""Golden parity tests for the herd modules (demographic, non-demographic, all-herd).

Each case runs the module on the bundled example inputs exactly as
``tools/r_reference/generate_golden.R`` does and compares every output table
with the R golden output (rtol 1e-9, exact column and row order).

Further R-backed cases (reference values computed with the R package and
written below as 17-significant-digit literals) cover what the examples never
reach: the zero-hazard bump, negative offtake, zero mortality, herds that do
not converge within ``max_simulation_years``, zero initial cohorts, the numpy
batch steady-state kernel used for more than ``SCALAR_STEADY_STATE_MAX_HERDS``
herds, and fractional herd-level phase durations.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
from golden_utils import assert_matches_golden, golden_tables

import gleampy
from gleampy.core import demographic_herd as dh
from gleampy.io import example_path, load_example, read_csv
from gleampy.validation import GleamValidationError

TABLES = ("cohort_level_results", "herd_level_results")


def _demographic():
    return gleampy.run_demographic_herd_module(
        cohort_level_data=load_example("herd_simulation_input_chrt_data.csv"),
        herd_level_data=load_example("herd_simulation_input_hrd_data.csv"),
        simulation_duration=365,
        show_indicator=False,
    )


def _nondemographic():
    return gleampy.run_nondemographic_herd_module(
        cohort_level_data=load_example("nondemographic_herd_input_chrt_data.csv"),
        herd_level_data=load_example("nondemographic_herd_input_hrd_data.csv"),
        simulation_duration=365,
        show_indicator=False,
    )


def _all_herd(chrt: str, hrd: str, run_demographic: bool, run_nondemographic: bool):
    return gleampy.run_all_herd_module(
        cohort_level_data=load_example(chrt),
        herd_level_data=load_example(hrd),
        run_demographic=run_demographic,
        run_nondemographic=run_nondemographic,
        show_indicator=False,
    )


CASES = {
    "demographic_herd_module": _demographic,
    "nondemographic_herd_module": _nondemographic,
    "all_herd_module_demographic": functools.partial(
        _all_herd, "herd_simulation_input_chrt_data.csv", "herd_simulation_input_hrd_data.csv", True, False
    ),
    "all_herd_module_nondemographic": functools.partial(
        _all_herd, "nondemographic_herd_input_chrt_data.csv", "nondemographic_herd_input_hrd_data.csv", False, True
    ),
    "all_herd_module_both": functools.partial(
        _all_herd, "herd_all_input_chrt_data.csv", "herd_all_input_hrd_data.csv", True, True
    ),
}


@functools.lru_cache(maxsize=None)
def _result(case: str):
    return CASES[case]()


@pytest.mark.parametrize("case", list(CASES))
def test_output_tables(case):
    assert list(_result(case)) == list(TABLES)


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("case", list(CASES))
def test_parity_with_r(case, table):
    assert_matches_golden(_result(case)[table], case, table)


@pytest.mark.parametrize("table", TABLES)
def test_parity_with_r_batch_steady_state_kernel(table, monkeypatch):
    """The numpy (many-herd) steady-state path reproduces R as well."""
    monkeypatch.setattr(dh, "SCALAR_STEADY_STATE_MAX_HERDS", 0)
    res = _demographic()
    assert_matches_golden(res[table], "demographic_herd_module", table)
    pd.testing.assert_frame_equal(res[table], _result("demographic_herd_module")[table], check_exact=True)


def test_inputs_are_not_modified():
    chrt = load_example("herd_all_input_chrt_data.csv")
    hrd = load_example("herd_all_input_hrd_data.csv")
    chrt0, hrd0 = chrt.copy(), hrd.copy()
    gleampy.run_all_herd_module(chrt, hrd, show_indicator=False)
    pd.testing.assert_frame_equal(chrt, chrt0)
    pd.testing.assert_frame_equal(hrd, hrd0)


def test_all_herd_without_validation_matches_r():
    with pytest.warns(gleampy.GleamWarning, match="Input validation has been turned off"):
        res = gleampy.run_all_herd_module(
            load_example("herd_all_input_chrt_data.csv"), load_example("herd_all_input_hrd_data.csv"),
            show_indicator=False, validate_inputs=False,
        )
    for table in TABLES:
        assert_matches_golden(res[table], "all_herd_module_both", table)


# ---------------------------------------------------------------------------
# R-backed cases for branches of the demographic herd model that the bundled
# examples never reach. The reference values were computed with the R package
# (integration/r-reference): run_demographic_herd_module() on the five herds of
# _edge_inputs(5), and calc_steady_state_structure() for the iteration counts.
# ---------------------------------------------------------------------------

SIX = ("FJ", "FS", "FA", "MJ", "MS", "MA")
EDGE_BASE = {
    "cohort_duration_days": {"FJ": 365, "FS": 700, "FA": 2000, "MJ": 365, "MS": 700, "MA": 1500},
    "death_rate": {"FJ": 0.1, "FS": 0.05, "FA": 0.05, "MJ": 0.1, "MS": 0.05, "MA": 0.05},
    "offtake_rate": {"FJ": 0.05, "FS": 0.1, "FA": 0.2, "MJ": 0.2, "MS": 0.3, "MA": 0.4},
}
EDGE_HERD = {
    "parturition_rate": 0.8, "litter_size": 1.0, "birth_fraction_female": 0.5, "herd_size_total": 1000.0,
    "prop_nondemo_fem_juv": 0.0, "prop_nondemo_mal_juv": 0.0,
}
EDGE_OVERRIDES = {
    # death and offtake both 0: R bumps death_rate to 1e-12
    1: {"death_rate": {"FS": 0.0, "MS": 0.0}, "offtake_rate": {"FS": 0.0, "MS": 0.0}},
    # negative offtake (allowed down to -2): animals brought in
    2: {"offtake_rate": {"FA": -0.5, "MA": -1.2}},
    # death 0 with offtake > 0
    3: {"death_rate": {"FJ": 0.0, "MA": 0.0}},
    # short (CHK-like) cohorts with diversion to the non-demographic block
    4: {"cohort_duration_days": {"FJ": 3, "FS": 140, "FA": 420, "MJ": 3, "MS": 140, "MA": 420},
        "herd": {"parturition_rate": 6.0, "litter_size": 1.5, "prop_nondemo_fem_juv": 0.3,
                 "prop_nondemo_mal_juv": 0.6}},
    # shortest juvenile cohorts that run (1 day crashes in R and Python)
    5: {"cohort_duration_days": {"FJ": 2, "MJ": 2}},
}
DEFAULT_INIT = {"FJ": 100, "FS": 50, "FA": 30, "MJ": 100, "MS": 50, "MA": 30}
EDGE_CONFIGS = {
    "default": {},
    # no herd converges within 2 years (731 days)
    "years2": {"max_simulation_years": 2},
    # no initial males in MS / MA: Python's plain-float kernel divides by zero and
    # falls back to the numpy kernel, which converges like R
    "zeroMA": {"initial_herd_structure": {"FJ": 100, "FS": 50, "FA": 30, "MJ": 100, "MS": 0, "MA": 0}},
}


def _edge_inputs(n_herds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Herds 1-5 are the five edge cases; further herds repeat them with rates scaled by 1.01, 1.02, ..."""
    rows, herds = [], []
    for k in range(1, n_herds + 1):
        over = EDGE_OVERRIDES[(k - 1) % 5 + 1]
        scale = 1 + 0.01 * ((k - 1) // 5)
        for c in SIX:
            row = {"herd_id": k, "cohort_short": c}
            for col in ("cohort_duration_days", "death_rate", "offtake_rate"):
                v = float(over.get(col, {}).get(c, EDGE_BASE[col][c]))
                row[col] = v if col == "cohort_duration_days" else v * scale
            rows.append(row)
        herds.append({"herd_id": k, **EDGE_HERD, **over.get("herd", {})})
    return pd.DataFrame(rows), pd.DataFrame(herds)


R_EDGE_COHORT_ORDER = ["FA", "FJ", "FS", "MA", "MJ", "MS"] * 5
R_EDGE = {
    "default": {
        "cohort_stock_size_unscaled": [
            292.5374159040939, 109.00779595291925, 195.87272253316831, 135.44160606510053, 99.86647493153525,
            163.19184056535335, 362.5315922580065, 119.96247430583756, 150.86236950728778, 218.94787653076054,
            110.94480403026989, 104.98896531509122, 312.07501179719407, 126.97912803694433, 220.1496624523873,
            81.2265471144709, 109.77441641219349, 128.77420613019058, 638.4176221372543, 22.351775041398362,
            475.3760057623972, 219.58561795555568, 21.1105494453869, 205.48783958290127, 418.1396781260053,
            0.8814390984829672, 267.6115986683835, 104.35790381960835, 0.8446294989892815, 173.37640462161892
        ],
        "offtake_heads_unscaled": [
            89.53259411989235, 5.895218986502688, 0, 79.28779174941106, 23.565962502474132, 0,
            2.236021425611483, 6.477649064429111, 16.29252820997495, -3.406461071565417, 26.13978602044683,
            38.463030952400416, 95.50255743001505, 6.512102747214268, 23.80961195879905, 47.24495118217363,
            25.901381541062126, 47.244930943735454, 586.7137971397358, 136.3970946152144, 127.5857781805829,
            237.8334133387868, 545.675222231849, 186.9803481306902, 127.93022198751501, 8.360788832845031,
            28.9357316596798, 61.07140179440903, 33.44300562735482, 63.593653887309664
        ],
        "cohort_stock_annual_nondemographic": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 696.751655074316, 0, 0, 1161.7728553103177,
            0, 0, 0, 0, 0, 0, 0
        ],
        "growth_rate_herd": [
            -0.008164288095658945, 0.13647616389450734, -0.04204205611323886, 1.1646588198497874,
            -0.06957669233382335
        ],
        "days_to_steady_state": [
            5654, 9024, 5996, 2057, 5812
        ],
    },
    "years2": {
        "cohort_stock_size_unscaled": [
            205.35316645084146, 90.66737775123636, 211.2198249718036, 142.77821914408332, 81.47527438704168,
            187.20391099644934, 249.67960449469564, 98.82209305809857, 192.7377949542925, 193.51709858481956,
            89.34727564688207, 140.99118398912287, 206.44067070210224, 104.280077086899, 231.9766123003315,
            108.15847836496451, 87.35401299924212, 156.69567246461827, 634.6783259098305, 22.220853118332514,
            472.6071769943157, 227.53538727530474, 20.986898065289193, 204.36707279871047, 333.88659429813936,
            0.7037155903429546, 308.8441438144157, 163.56195591361413, 0.6743331962711786, 220.89273589911696
        ],
        "offtake_heads_unscaled": [
            73.1789774707859, 5.0928884217491674, 0, 91.43400926421032, 20.06609383434771, 0,
            1.8118025545077927, 5.594248819162833, 20.428492389602265, -3.427889547003872, 22.217405922542117,
            48.88790261932623, 74.23737712481389, 5.535096588154413, 25.821731030612884, 66.40752519838779,
            21.4508290164798, 56.26039242311541, 583.2583259954959, 135.5937752229323, 126.8349468266777,
            240.33637718193359, 542.4614341567403, 185.8937336297236, 100.52231748907238, 6.568844309749305,
            29.365513466785483, 84.6996115679124, 26.275395594440464, 68.75493166087703
        ],
        "cohort_stock_annual_nondemographic": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 692.6480806380039, 0, 0, 1154.930502974661,
            0, 0, 0, 0, 0, 0, 0
        ],
        "growth_rate_herd": [
            -0.16260445259708844, -0.06980989854417774, -0.2101889521636846, 1.1647914283235665,
            0.057126957423800695
        ],
        "days_to_steady_state": [
            731, 731, 731, 731, 731
        ],
    },
    "zeroMA": {
        "cohort_stock_size_unscaled": [
            294.04020737046943, 109.53651827394273, 196.8028060239982, 132.7160811414198, 100.36336937193232,
            162.61230844108223, 377.1598587258399, 124.787981391577, 156.9378415564559, 184.91726063316787,
            115.41204117309763, 109.10655610652294, 312.70551026908265, 127.19041666729598, 220.46950531819047,
            80.17710482980041, 109.98203501522458, 128.62354998995912, 639.6626736956384, 22.395364874993124,
            476.3062116981574, 216.96336689109742, 21.15171872713102, 205.86330895076972, 418.4132115701589,
            0.8820156080493928, 267.86687308552075, 103.91316325454866, 0.8451819375495692, 173.33114133025913
        ],
        "offtake_heads_unscaled": [
            89.97360711937762, 5.923456265916084, 0, 78.04500528082251, 23.680715691494488, 0,
            2.326047585199234, 6.738018574518478, 16.947293889069147, -2.9289561477262964, 27.19112545100903,
            39.982534372454964, 95.67197277356208, 6.522520151940859, 23.84172154231922, 46.75812378317278,
            25.946645450780306, 47.22281770707435, 587.8541904535076, 136.6622083101973, 127.83388475670404,
            237.00599560945014, 546.7358460035036, 187.33909472676004, 128.01280076422628, 8.366185063087,
            28.960364120791656, 60.8681633647502, 33.46459057383873, 63.59072446365141
        ],
        "cohort_stock_annual_nondemographic": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 698.1059221983977, 0, 0,
            1164.0309783997905, 0, 0, 0, 0, 0, 0, 0
        ],
        "growth_rate_herd": [
            -0.00785741875431012, 0.13664307917332308, -0.04170375582089402, 1.1646852896755746,
            -0.06949682642782684
        ],
        "days_to_steady_state": [
            3289, 3501, 3320, 839, 3913
        ],
    },
}


def _assert_matches_r_edge(res: dict, config: str) -> None:
    ref = R_EDGE[config]
    clr = res["cohort_level_results"]
    clr = clr[clr["herd_id"] <= 5]
    assert clr["cohort_short"].tolist() == R_EDGE_COHORT_ORDER
    for col in ("cohort_stock_size_unscaled", "offtake_heads_unscaled", "cohort_stock_annual_nondemographic"):
        np.testing.assert_allclose(clr[col].to_numpy(float), ref[col], rtol=1e-9, atol=1e-10, err_msg=col)
    hlr = res["herd_level_results"]
    np.testing.assert_allclose(hlr.loc[hlr["herd_id"] <= 5, "growth_rate_herd"].to_numpy(float),
                               ref["growth_rate_herd"], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("kernel", ["scalar", "batch"])
@pytest.mark.parametrize("config", list(EDGE_CONFIGS))
def test_edge_herds_match_r(config, kernel, monkeypatch):
    if kernel == "batch":
        monkeypatch.setattr(dh, "SCALAR_STEADY_STATE_MAX_HERDS", 0)
    chrt, hrd = _edge_inputs(5)
    res = gleampy.run_demographic_herd_module(chrt, hrd, show_indicator=False, **EDGE_CONFIGS[config])
    _assert_matches_r_edge(res, config)


@pytest.mark.parametrize("kernel", ["scalar", "batch"])
@pytest.mark.parametrize("config", list(EDGE_CONFIGS))
def test_edge_herds_iteration_counts_match_r(config, kernel, monkeypatch):
    if kernel == "batch":
        monkeypatch.setattr(dh, "SCALAR_STEADY_STATE_MAX_HERDS", 0)
    cfg = EDGE_CONFIGS[config]
    chrt, hrd = _edge_inputs(5)
    days = []
    for k in range(1, 6):
        ck = chrt[chrt["herd_id"] == k].set_index("cohort_short").loc[list(SIX)]
        hk = hrd[hrd["herd_id"] == k].iloc[0]
        tr = gleampy.calc_transition_probabilities(ck["cohort_duration_days"], ck["offtake_rate"], ck["death_rate"])
        fec = gleampy.calc_fecundity_rates(hk["parturition_rate"], hk["litter_size"], hk["birth_fraction_female"])
        pn = {"FJ": hk["prop_nondemo_fem_juv"], "FS": 0.0, "FA": 0.0, "MJ": hk["prop_nondemo_mal_juv"],
              "MS": 0.0, "MA": 0.0}
        res = gleampy.calc_steady_state_structure(
            cfg.get("initial_herd_structure", DEFAULT_INIT), cfg.get("max_simulation_years", 100), 1e-9,
            fec["fecundity_female"], fec["fecundity_male"], tr["probability_death"], tr["probability_offtake"],
            tr["probability_growth"], pn,
        )
        days.append(res["days_to_steady_state"])
    assert days == R_EDGE[config]["days_to_steady_state"]


@pytest.mark.parametrize("config", list(EDGE_CONFIGS))
def test_batch_kernel_on_many_herds_matches_scalar_kernel_and_r(config):
    """More than SCALAR_STEADY_STATE_MAX_HERDS herds use the numpy kernel (the RL batch case).

    It must give bit-identical results to the per-herd plain-float kernel
    (including its ZeroDivisionError fallback) and match R on herds 1-5,
    also when no herd converges within max_simulation_years.
    """
    chrt, hrd = _edge_inputs(60)
    assert 60 > dh.SCALAR_STEADY_STATE_MAX_HERDS
    batch = gleampy.run_demographic_herd_module(chrt, hrd, show_indicator=False, **EDGE_CONFIGS[config])
    _assert_matches_r_edge(batch, config)
    # herds are simulated independently: the first 10 herds (the five cases at two
    # rate scales) through the plain-float kernel give exactly the batch values
    assert 10 <= dh.SCALAR_STEADY_STATE_MAX_HERDS
    scalar = gleampy.run_demographic_herd_module(*_edge_inputs(10), show_indicator=False, **EDGE_CONFIGS[config])
    for table, rows in (("cohort_level_results", 60), ("herd_level_results", 10)):
        pd.testing.assert_frame_equal(batch[table].iloc[:rows], scalar[table], check_exact=True)


# ---------------------------------------------------------------------------
# Fractional herd-level phase durations (never truncated, whatever the dtype)
# ---------------------------------------------------------------------------

#: R run_all_herd_module() on herd 1 of herd_all_input_*.csv with
#: phase1_nondemo_mal_duration_days = 60.7, cohort_duration_days converted to
#: double. On the column as fread types it (integer: whole numbers plus blank
#: FN/MN cells) R truncates 60.7 to 60 with a warning and gives MN stocks
#: 691119.66271039948 / 1169682.2068753834: an R (data.table coercion) bug the
#: port does not replicate.
R_FRACTIONAL_PHASE = {
    "cohort_short": ["FA", "FJ", "FS", "MA", "MJ", "MS", "MN", "MN"],
    "cohort_duration_days": [989, 60, 710, 823, 60, 710, 60.700000000000003, 120],
    "cohort_stock_size": [
        12076014.85996552, 566311.05238195963, 4917790.1686125798, 2306.4367651363746,
        796244.31221819902, 79552.397211794669, 699145.60711161816, 1162635.1657331886,
    ],
    "offtake_heads": [
        4251095.1356901834, 2468768.7787572849, 1412859.9365378616, 982.54158952312741,
        0, 248465.22406720795, 0, 3584538.2596470928,
    ],
    "total_nondemo_mal_duration_days": 180.69999999999999,
}


def _herd1_inputs(dtype: str, **phases: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    chrt = load_example("herd_all_input_chrt_data.csv")
    hrd = load_example("herd_all_input_hrd_data.csv")
    chrt = chrt[chrt["herd_id"] == 1].reset_index(drop=True)
    hrd = hrd[hrd["herd_id"] == 1].reset_index(drop=True)
    for col, value in phases.items():
        hrd[col] = value
    assert chrt["cohort_duration_days"].isna().any()  # blank FN/MN durations, as in the bundled layout
    if dtype == "nullable":  # Int64 cohort_duration_days, string cohort codes
        chrt, hrd = chrt.convert_dtypes(), hrd.convert_dtypes()
        assert str(chrt["cohort_duration_days"].dtype) == "Int64"
    elif dtype == "int64":
        chrt["cohort_duration_days"] = chrt["cohort_duration_days"].fillna(0).astype("int64")
    return chrt, hrd


@pytest.mark.parametrize("dtype", ["float64", "nullable", "int64"])
def test_fractional_phase_duration_is_never_truncated(dtype, recwarn):
    chrt, hrd = _herd1_inputs(dtype, phase1_nondemo_mal_duration_days=60.7)
    res = gleampy.run_all_herd_module(chrt, hrd, show_indicator=False)
    assert not [w for w in recwarn if issubclass(w.category, gleampy.GleamWarning)]
    clr = res["cohort_level_results"]
    assert clr["cohort_short"].tolist() == R_FRACTIONAL_PHASE["cohort_short"]
    for col in ("cohort_duration_days", "cohort_stock_size", "offtake_heads"):
        np.testing.assert_allclose(clr[col].to_numpy(float), R_FRACTIONAL_PHASE[col], rtol=1e-9, atol=1e-10,
                                   err_msg=col)
    assert res["herd_level_results"]["total_nondemo_mal_duration_days"].tolist() == pytest.approx(
        [R_FRACTIONAL_PHASE["total_nondemo_mal_duration_days"]], rel=1e-12)


@pytest.mark.parametrize(("value", "message"), [
    (0.5, r"^`cohort_duration_days` = 0\.5 is out of range; expected value should be >= 1 and <= 8000\.$"),
    (-0.5, r"^`phase2_nondemo_duration` must be greater than or equal to 0\.$"),
])
def test_sub_day_phase2_duration_is_rejected_like_r_with_a_double_column(value, message):
    # R with an integer cohort_duration_days column truncates 0.5 / -0.5 to 0 and runs
    chrt, hrd = _herd1_inputs("float64", phase2_nondemo_mal_duration_days=value)
    with pytest.raises(GleamValidationError, match=message):
        gleampy.run_all_herd_module(chrt, hrd, show_indicator=False)


def test_all_herd_with_nullable_dtypes_matches_r():
    """convert_dtypes() inputs (Int64 with NA, string codes) give the R golden results."""
    res = gleampy.run_all_herd_module(
        load_example("herd_all_input_chrt_data.csv").convert_dtypes(),
        load_example("herd_all_input_hrd_data.csv").convert_dtypes(),
        show_indicator=False,
    )
    for table in TABLES:
        assert_matches_golden(res[table], "all_herd_module_both", table)


def _flatten(result: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in result.items():
        name = f"{prefix}__{k}" if prefix else k
        if isinstance(v, pd.DataFrame):
            out[name] = v
        elif isinstance(v, dict):
            out.update(_flatten(v, name))
    return out


def test_run_gleam_without_herd_structure_accepts_nullable_dtypes():
    def run(name: str) -> pd.DataFrame:
        return read_csv(example_path(name, "run_gleam_examples"))

    def nondemo(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["herd_id"].isin((14, 15))].reset_index(drop=True)

    res = gleampy.run_gleam(
        has_herd_structure=False, run_demographic=False, run_nondemographic=True,
        cohort_level_data=run("master_chrt_lvl_no_structure_nondemo_data.csv").convert_dtypes(),
        herd_level_data=run("master_hrd_lvl_nondemo_data.csv").convert_dtypes(),
        feed_rations=nondemo(run("feed_rations_share_chrt.csv")),
        feed_params=run("feed_quality.csv"),
        feed_emissions=run("feed_emission_factors.csv"),
        manure_management_system_fraction=nondemo(run("manure_management_system_fraction.csv")),
        manure_management_system_factors=nondemo(run("manure_management_system_factors.csv")),
        simulation_duration=365, show_indicator=False,
    )
    tables = _flatten(res)
    for table in golden_tables("run_gleam_nondemo_only"):
        assert_matches_golden(tables[table], "run_gleam_nondemo_only", table)


# ---------------------------------------------------------------------------
# Switches and missing columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [np.True_, np.array([True]), np.array(True)])
def test_all_herd_switches_accept_numpy_logicals(flag):
    res = gleampy.run_all_herd_module(
        load_example("herd_all_input_chrt_data.csv"), load_example("herd_all_input_hrd_data.csv"),
        run_demographic=flag, run_nondemographic=np.True_, show_indicator=False,
    )
    for table in TABLES:
        assert_matches_golden(res[table], "all_herd_module_both", table)


@pytest.mark.parametrize("flag", [1, "TRUE", None, np.array([True, True])])
def test_all_herd_switches_reject_non_logical_values(flag):
    with pytest.raises(GleamValidationError, match="`run_demographic` must be a single logical value"):
        gleampy.run_all_herd_module(
            load_example("herd_all_input_chrt_data.csv"), load_example("herd_all_input_hrd_data.csv"),
            run_demographic=flag, show_indicator=False,
        )


@pytest.mark.parametrize("table", ["cohort_level_data", "herd_level_data"])
def test_nondemographic_all_herd_reports_a_missing_herd_id(table):
    inputs = {
        "cohort_level_data": load_example("nondemographic_herd_input_chrt_data.csv"),
        "herd_level_data": load_example("nondemographic_herd_input_hrd_data.csv"),
    }
    inputs[table] = inputs[table].drop(columns="herd_id")
    with pytest.raises(GleamValidationError, match=rf'^Missing required columns in `{table}`: "herd_id"$'):
        gleampy.run_all_herd_module(**inputs, run_demographic=False, show_indicator=False)


@pytest.mark.parametrize(("module", "argument"), [
    ("demographic", "simulation_duration"),
    ("demographic", "max_simulation_years"),
    ("demographic", "min_lambda_change"),
    ("nondemographic", "simulation_duration"),
    ("all_herd", "simulation_duration"),
])
def test_scalar_arguments_reject_vectors_like_r(module, argument):
    run = {
        "demographic": lambda **kw: gleampy.run_demographic_herd_module(
            load_example("herd_simulation_input_chrt_data.csv"), load_example("herd_simulation_input_hrd_data.csv"),
            show_indicator=False, **kw),
        "nondemographic": lambda **kw: gleampy.run_nondemographic_herd_module(
            load_example("nondemographic_herd_input_chrt_data.csv"),
            load_example("nondemographic_herd_input_hrd_data.csv"), show_indicator=False, **kw),
        "all_herd": lambda **kw: gleampy.run_all_herd_module(
            load_example("herd_all_input_chrt_data.csv"), load_example("herd_all_input_hrd_data.csv"),
            show_indicator=False, **kw),
    }[module]
    with pytest.raises(GleamValidationError, match=rf"^`{argument}` must be a single numeric value\.$"):
        run(**{argument: np.array([1.0, 2.0])})
