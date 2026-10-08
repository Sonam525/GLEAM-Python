"""Port of tests/testthat/test-demographic_herd_core.R (plus a few checks of the port)."""

import numpy as np
import pandas as pd
import pytest

import gleampy
from gleampy.core import demographic_herd as dh
from gleampy.validation import GleamValidationError

COHORTS = ["FB", "FJ", "FS", "FA", "FC", "MB", "MJ", "MS", "MA", "MC"]
SHARE_COHORTS = ["FJ", "FS", "FA", "MJ", "MS", "MA"]
PROP_NONDEMO_ZERO = {"FJ": 0, "FS": 0, "FA": 0, "MJ": 0, "MS": 0, "MA": 0}
TOL = 1.5e-8  # testthat's default tolerance


def set_names(values, names):
    return pd.Series(np.asarray(values, dtype=float), index=names)


def _transitions():
    return gleampy.calc_transition_probabilities(
        cohort_duration_days=set_names([365] * 6, SHARE_COHORTS),
        offtake_rate=set_names([0.1] * 6, SHARE_COHORTS),
        death_rate=set_names([0.05] * 6, SHARE_COHORTS),
    )


def _steady(fec, trans):
    return gleampy.calc_steady_state_structure(
        initial_herd_structure={"FJ": 100, "FS": 50, "FA": 30, "MJ": 100, "MS": 50, "MA": 30},
        max_simulation_years=5,
        min_lambda_change=1e-6,
        fecundity_female=fec["fecundity_female"],
        fecundity_male=fec["fecundity_male"],
        probability_death=set_names(trans["probability_death"], COHORTS),
        probability_offtake=set_names(trans["probability_offtake"], COHORTS),
        probability_growth=set_names(trans["probability_growth"], COHORTS),
        proportion_nondemographic=PROP_NONDEMO_ZERO,
    )


# ---- test calc_fecundity_rates ----
def test_calc_fecundity_rates_returns_expected_output():
    res = gleampy.calc_fecundity_rates(parturition_rate=0.8, litter_size=2, birth_fraction_female=0.5)
    assert isinstance(res, dict)
    assert list(res) == ["fecundity_female", "fecundity_male"]
    assert res["fecundity_female"] == pytest.approx(0.8 * 2 * 0.5 / 365, rel=TOL)
    assert res["fecundity_male"] == pytest.approx(0.8 * 2 * 0.5 / 365, rel=TOL)  # symmetrical case


# ---- test calc_transition_probabilities ----
def test_calc_transition_probabilities_returns_named_list_with_correct_lengths():
    res = _transitions()
    assert isinstance(res, dict)
    assert list(res) == [
        "hazard_death", "hazard_offtake", "probability_death",
        "probability_offtake", "probability_survival", "probability_growth",
    ]
    assert len(res["hazard_death"]) == 6
    assert len(res["probability_death"]) == 10


# ---- test calc_steady_state_structure ----
def test_calc_steady_state_structure_converges_and_returns_valid_structure():
    fec = gleampy.calc_fecundity_rates(0.8, 2, 0.5)
    result = _steady(fec, _transitions())
    assert list(result) == [
        "days_to_steady_state",
        "herd_structure",
        "cohort_share",
        "growth_rate_herd",
        "size_unscaled",
        "herd_size_total_demographic",
    ]
    assert result["days_to_steady_state"] <= 5 * 365
    assert result["herd_structure"].sum() == pytest.approx(1, abs=1e-6)


# ---- test calc_projected_population_size ----
def test_calc_projected_population_size_runs_and_returns_list_with_expected_elements():
    fec = gleampy.calc_fecundity_rates(0.8, 2, 0.5)
    trans = _transitions()
    steady = _steady(fec, trans)
    res = gleampy.calc_projected_population_size(
        herd_size_total=1000,
        fecundity_female=fec["fecundity_female"],
        fecundity_male=fec["fecundity_male"],
        probability_death=set_names(trans["probability_death"], COHORTS),
        probability_offtake=set_names(trans["probability_offtake"], COHORTS),
        probability_growth=set_names(trans["probability_growth"], COHORTS),
        growth_rate_herd=steady["growth_rate_herd"],
        herd_structure=steady["herd_structure"],
        cohort_share=steady["cohort_share"],
        proportion_nondemographic=PROP_NONDEMO_ZERO,
    )
    assert list(res) == [
        "cohort_stock_start",
        "cohort_stock_end_projected",
        "cohort_stock_end_exact_simulated",
        "cohort_stock_average",
        "cohort_stock_annual_nondemographic",
        "cohort_offtake_heads",
    ]
    assert len(res["cohort_stock_start"]) == 6


# ---- test calc_summary_offtake ----
def test_calc_summary_offtake_returns_all_expected_components():
    res = gleampy.calc_summary_offtake(
        cohort_stock_start=set_names([100] * 6, SHARE_COHORTS),
        cohort_stock_end_projected=set_names([105] * 6, SHARE_COHORTS),
        cohort_stock_average=set_names([102] * 6, SHARE_COHORTS),
        cohort_offtake_heads=set_names([0.01] * 10, COHORTS),
        simulation_duration=200,
    )
    assert list(res) == [
        "stock_variation_heads",
        "offtake_heads",
        "offtake_heads_assessment",
        "offtake_rate_to_stock_start",
        "offtake_rate_to_stock_average",
        "offtake_stock_variation_heads",
        "offtake_stock_plus_variation_rate_to_stock_start",
        "offtake_stock_plus_variation_rate_to_stock_average",
    ]
    assert len(res["offtake_heads"]) == 6


# ---- additional checks of the Python port ----
def test_transition_inputs_are_validated_with_cohort_labels():
    with pytest.raises(GleamValidationError, match=r"`death_rate`\[FS\] = 1.5 is out of range"):
        gleampy.calc_transition_probabilities(
            cohort_duration_days=set_names([365] * 6, SHARE_COHORTS),
            offtake_rate=set_names([0.1] * 6, SHARE_COHORTS),
            death_rate=set_names([0.05, 1.5, 0.05, 0.05, 0.05, 0.05], SHARE_COHORTS),
        )
    with pytest.raises(GleamValidationError, match="must be a numeric vector of length 6 with names"):
        gleampy.calc_transition_probabilities(
            cohort_duration_days=set_names([365] * 5, SHARE_COHORTS[:5]),
            offtake_rate=set_names([0.1] * 6, SHARE_COHORTS),
            death_rate=set_names([0.05] * 6, SHARE_COHORTS),
        )


def test_steady_state_scalar_and_batch_kernels_are_identical():
    """The plain-float and numpy steady-state kernels evaluate the same IEEE operations."""
    herds = [1, 9, 13]  # example herds (CTL, PGS with diversion, CHK)
    chrt = gleampy.load_example("herd_simulation_input_chrt_data.csv")
    hrd = gleampy.load_example("herd_simulation_input_hrd_data.csv").set_index("herd_id").loc[herds]
    chrt = chrt[chrt["herd_id"].isin(herds)].set_index(["cohort_short", "herd_id"])

    def by_cohort(col):
        return {c: chrt[col].xs(c).loc[herds].to_numpy(dtype=float) for c in SHARE_COHORTS}

    fec = gleampy.calc_fecundity_rates(
        hrd["parturition_rate"].to_numpy(), hrd["litter_size"].to_numpy(), hrd["birth_fraction_female"].to_numpy()
    )
    off = by_cohort("offtake_rate")
    death = {c: dh._bump_zero_hazard(off[c], v) for c, v in by_cohort("death_rate").items()}
    tr = dh._transition_kernel(by_cohort("cohort_duration_days"), off, death)
    zeros = np.zeros(len(herds))
    pn = {"FJ": hrd["prop_nondemo_fem_juv"].to_numpy(float), "FS": zeros, "FA": zeros,
          "MJ": hrd["prop_nondemo_mal_juv"].to_numpy(float), "MS": zeros, "MA": zeros}
    init = {"FJ": 100.0, "FS": 50.0, "FA": 30.0, "MJ": 100.0, "MS": 50.0, "MA": 30.0}
    args = (init, 100, 1e-9, fec["fecundity_female"], fec["fecundity_male"],
            tr["probability_death"], tr["probability_offtake"], tr["probability_growth"], pn)
    scalar = dh._steady_state_kernel(*args)
    old = dh.SCALAR_STEADY_STATE_MAX_HERDS
    try:
        dh.SCALAR_STEADY_STATE_MAX_HERDS = 0
        batch = dh._steady_state_kernel(*args)
    finally:
        dh.SCALAR_STEADY_STATE_MAX_HERDS = old
    for key, val in scalar.items():
        np.testing.assert_array_equal(val, batch[key], err_msg=key)
    assert list(scalar["days_to_steady_state"]) == [21889, 1419, 894]  # R iteration counts


#: R calc_steady_state_structure() with the _transitions() probabilities and
#: fecundity (0.8, 2, 0.5): `stop_at_max_years` stops after 1:(0.01 * 365 + 1)
#: days without converging (threshold 1e-12); `zero_initial_MA` starts with no
#: adult males and converges (in Python the plain-float kernel divides by zero
#: and falls back to the numpy kernel).
R_STEADY_STATE = {
    "stop_at_max_years": {
        "init": {"FJ": 100, "FS": 50, "FA": 30, "MJ": 100, "MS": 50, "MA": 30}, "years": 0.01, "thr": 1e-12,
        "days_to_steady_state": 4, "growth_rate_herd": -0.56989504125775543,
        "herd_structure": [0.00018353873797857576, 0.27616305904223966, 0.13991385301705652, 0.083739549202725172,
                           0.00018353873797857576, 0.27616305904223966, 0.13991385301705652, 0.083739549202725172],
        "herd_size_total_demographic": 359.59995343693299,
    },
    "zero_initial_MA": {
        "init": {"FJ": 100, "FS": 50, "FA": 30, "MJ": 100, "MS": 50, "MA": 0}, "years": 100, "thr": 1e-9,
        "days_to_steady_state": 2587, "growth_rate_herd": -0.18321863463690679,
        "herd_structure": [0.00038293467775133919, 0.15839520555810882, 0.16658742602935881, 0.17471394672404852,
                           0.00038293467775133919, 0.15839520555810882, 0.16658742602935881, 0.17455492074551358],
        "herd_size_total_demographic": 87.62331127786689,
    },
}


@pytest.mark.parametrize("kernel", ["scalar", "batch"])
@pytest.mark.parametrize("case", list(R_STEADY_STATE))
def test_steady_state_edge_cases_match_r(case, kernel, monkeypatch):
    ref = R_STEADY_STATE[case]
    if kernel == "batch":
        monkeypatch.setattr(dh, "SCALAR_STEADY_STATE_MAX_HERDS", 0)
    fec = gleampy.calc_fecundity_rates(0.8, 2, 0.5)
    trans = _transitions()
    res = gleampy.calc_steady_state_structure(
        ref["init"], ref["years"], ref["thr"],
        fec["fecundity_female"], fec["fecundity_male"], trans["probability_death"],
        trans["probability_offtake"], trans["probability_growth"], PROP_NONDEMO_ZERO,
    )
    assert res["days_to_steady_state"] == ref["days_to_steady_state"]
    assert res["growth_rate_herd"] == pytest.approx(ref["growth_rate_herd"], rel=1e-9)
    np.testing.assert_allclose(res["herd_structure"].to_numpy(), ref["herd_structure"], rtol=1e-9)
    assert res["herd_size_total_demographic"] == pytest.approx(ref["herd_size_total_demographic"], rel=1e-9)


def test_steady_state_fails_like_r_when_a_lambda_change_is_nan():
    # males never present (no male births, no initial males): 0/0 lambda -> R's `if (NA)` error
    fec = gleampy.calc_fecundity_rates(0.8, 2, 1)
    trans = _transitions()
    with pytest.raises(GleamValidationError, match="^missing value where TRUE/FALSE needed: .* zero animals"):
        gleampy.calc_steady_state_structure(
            {"FJ": 100, "FS": 50, "FA": 30, "MJ": 0, "MS": 0, "MA": 0}, 100, 1e-9,
            fec["fecundity_female"], fec["fecundity_male"], trans["probability_death"],
            trans["probability_offtake"], trans["probability_growth"], PROP_NONDEMO_ZERO,
        )


def _one_day_juvenile_herds(n_herds, bad_herd):
    rows, herds = [], []
    off = {"FJ": 0.05, "FS": 0.1, "FA": 0.15, "MJ": 0.2, "MS": 0.3, "MA": 0.4}
    death = {"FJ": 0.1, "FS": 0.05, "FA": 0.05, "MJ": 0.1, "MS": 0.05, "MA": 0.05}
    for h in range(1, n_herds + 1):
        dur = {"FJ": 1 if h == bad_herd else 300, "FS": 700, "FA": 1500, "MJ": 60, "MS": 700, "MA": 1500}
        rows += [{"herd_id": h, "cohort_short": c, "cohort_duration_days": dur[c], "offtake_rate": off[c],
                  "death_rate": death[c]} for c in SHARE_COHORTS]
        herds.append({"herd_id": h, "parturition_rate": 0.8, "litter_size": 1.0, "birth_fraction_female": 0.5,
                      "herd_size_total": 1000, "prop_nondemo_fem_juv": 0.0, "prop_nondemo_mal_juv": 0.0})
    return pd.DataFrame(rows), pd.DataFrame(herds)


@pytest.mark.parametrize("n_herds", [1, 60])  # scalar and numpy (batch) steady-state kernels
def test_one_day_juvenile_cohort_fails_like_r_and_names_the_cause(n_herds):
    """R stops with "missing value where TRUE/FALSE needed" (a replicated R crash).

    FJ = 1 day passes the [1, 8000] range check, but the juvenile sub-class
    lasts 1 - 1 = 0 days, so its growth probability is infinite.
    """
    trans = gleampy.calc_transition_probabilities(
        set_names([1, 700, 1500, 60, 700, 1500], SHARE_COHORTS), set_names([0.1] * 6, SHARE_COHORTS),
        set_names([0.05] * 6, SHARE_COHORTS),
    )
    assert trans["probability_growth"]["FJ"] == np.inf  # as in R
    chrt, hrd = _one_day_juvenile_herds(n_herds, bad_herd=n_herds)
    with pytest.raises(GleamValidationError, match=(
        r"^missing value where TRUE/FALSE needed: .*`probability_growth` is not finite \(FJ = Inf\): "
        rf"a juvenile \(FJ or MJ\) `cohort_duration_days` of 1 day .* \(herd_id {n_herds}\)\.$"
    )):
        gleampy.run_demographic_herd_module(chrt, hrd, show_indicator=False)
    chrt.loc[chrt["cohort_duration_days"] == 1, "cohort_duration_days"] = 2
    gleampy.run_demographic_herd_module(chrt, hrd, show_indicator=False)  # 2 days run in R and Python


#: R run_demographic_herd_module on the 2 first herds of the herd simulation
#: example, with a logical prop_nondemo_fem_juv column (FALSE, TRUE): R's
#: c(FJ = TRUE, FS = 0, ...) coerces it to 1.
R_GROWTH_LOGICAL_PROP = [-0.20884596469812178, -0.096637286859885041]


def _two_example_herds():
    chrt = gleampy.load_example("herd_simulation_input_chrt_data.csv")
    hrd = gleampy.load_example("herd_simulation_input_hrd_data.csv")
    return (chrt[chrt["herd_id"].isin([1, 2])].reset_index(drop=True),
            hrd[hrd["herd_id"].isin([1, 2])].reset_index(drop=True))


@pytest.mark.parametrize("bad_herd", [0, 1])
def test_non_numeric_prop_nondemo_column_gives_the_r_message(bad_herd):
    chrt, hrd = _two_example_herds()
    values = ["0.0", "0.0"]
    values[bad_herd] = "abc"
    hrd["prop_nondemo_mal_juv"] = pd.Series(values, dtype=object)
    with pytest.raises(GleamValidationError,
                       match=r"^`proportion_nondemographic` must be a numeric vector of length 6 with names\.$"):
        gleampy.run_demographic_herd_module(chrt, hrd, show_indicator=False)


@pytest.mark.parametrize("dtype", ["bool", "boolean", "object"])
def test_logical_prop_nondemo_column_is_coerced_like_r(dtype):
    chrt, hrd = _two_example_herds()
    hrd["prop_nondemo_fem_juv"] = pd.Series([False, True], dtype=dtype)
    res = gleampy.run_demographic_herd_module(chrt, hrd, show_indicator=False)
    np.testing.assert_allclose(res["herd_level_results"]["growth_rate_herd"], R_GROWTH_LOGICAL_PROP, rtol=1e-9)
    # an all-empty column (fread: logical NA) is a missing proportion, as in R
    hrd["prop_nondemo_fem_juv"] = pd.Series([None, None], dtype=object)
    with pytest.raises(GleamValidationError, match=r"^`proportion_nondemographic` must not contain missing values\.$"):
        gleampy.run_demographic_herd_module(chrt, hrd, show_indicator=False)


def test_run_demographic_herd_module_reports_r_validation_message():
    chrt = gleampy.load_example("herd_simulation_input_chrt_data.csv")
    hrd = gleampy.load_example("herd_simulation_input_hrd_data.csv")
    chrt.loc[(chrt["herd_id"] == 2) & (chrt["cohort_short"] == "FS"), "death_rate"] = 1.2
    with pytest.raises(GleamValidationError, match=r"^`death_rate`\[FS\] = 1.2 is out of range"):
        gleampy.run_demographic_herd_module(chrt, hrd, show_indicator=False)
    hrd.loc[hrd["herd_id"] == 3, "prop_nondemo_mal_juv"] = 1.5
    chrt.loc[(chrt["herd_id"] == 2) & (chrt["cohort_short"] == "FS"), "death_rate"] = 0.05
    with pytest.raises(GleamValidationError, match=r"^`proportion_nondemographic`\[MJ\] = 1.5 is out of range"):
        gleampy.run_demographic_herd_module(chrt, hrd, show_indicator=False)


def test_rescale_x_to_y_keeps_zeros():
    out = gleampy.rescale_x_to_y(np.array([0.0, 2.0, np.nan]), np.array([np.nan, 4.0, 1.0]), 10.0)
    np.testing.assert_array_equal(out, [0.0, 5.0, np.nan])
    assert gleampy.rescale_x_to_y(3.0, 6.0, 2.0) == 1.0
