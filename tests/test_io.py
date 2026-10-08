"""Tests of gleampy.io.read_csv against data.table::fread (data.table 1.18.6.1).

Every expected value below was produced by running ``fread`` with default
arguments on the same bytes (R 4.6).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gleampy
from gleampy.io import example_dir, example_path, load_example, read_csv
from gleampy.validation import GleamValidationError
from gleampy.validation._shared import GleamWarning


def _write(tmp_path: Path, name: str, content: str | bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    return p


def _read(tmp_path: Path, content: str | bytes, name: str = "t.csv", **kw) -> pd.DataFrame:
    return read_csv(_write(tmp_path, name, content), **kw)


def _kind(s: pd.Series) -> str:
    """fread class of a column read by read_csv: i(nteger), n(umeric), l(ogical), c(haracter)."""
    if s.dtype.kind in "iu":
        return "i"
    if s.dtype.kind == "f":
        return "n"
    vals = [v for v in s.tolist() if not (v is None or (isinstance(v, float) and math.isnan(v)))]
    if all(isinstance(v, (bool, np.bool_)) for v in vals):
        return "l"
    return "c"


def _values(s: pd.Series) -> list:
    """Column values with every missing value as None."""
    return [None if (v is None or (isinstance(v, float) and math.isnan(v) and s.dtype.kind != "f")) else v
            for v in s.tolist()]


# --------------------------------------------------------------------------
# White space
# --------------------------------------------------------------------------


def test_unquoted_fields_are_stripped_quoted_fields_kept(tmp_path):
    df = _read(
        tmp_path,
        'id,name,val,flag,q,num2,blank\n'
        '1, mms_pasture ,  1.5 , TRUE ," kept ", 3 , \n'
        '2,mms_burned ,2,FALSE, x ,4,  \n'
        '3,  CTL,3, NA ," NA ",  NA ,NA\n',
    )
    # fread: name chr, val num, flag logical, q chr (quoted kept), num2 int with NA, blank logical NA
    assert [_kind(df[c]) for c in df.columns] == ["i", "c", "n", "l", "c", "n", "l"]
    assert df["name"].tolist() == ["mms_pasture", "mms_burned", "CTL"]
    assert df["val"].tolist() == [1.5, 2.0, 3.0]
    assert df["flag"].tolist() == [True, False, None]
    assert df["q"].tolist() == [" kept ", "x", " NA "]
    assert _values(df["num2"])[:2] == [3.0, 4.0] and math.isnan(df["num2"][2])
    assert df["blank"].tolist() == [None, None, None]


def test_whitespace_only_field_is_empty_string_in_character_column(tmp_path):
    df = _read(tmp_path, 'id\tname\tval\tflag\tq\n1\t mms_pasture \t 1.5 \t TRUE \t" kept "\n'
                         '2\tmms_burned \t2\tFALSE\t x \n3\t  \t 3 \t  \t""\n', name="t.tsv")
    assert df["name"].tolist() == ["mms_pasture", "mms_burned", ""]
    assert df["val"].tolist() == [1.5, 2.0, 3.0]
    assert df["flag"].tolist() == [True, False, None]
    assert df["q"].tolist() == [" kept ", "x", ""]


def test_padded_logical_column_is_logical(tmp_path):
    df = _read(tmp_path, "a,b\n TRUE ,1\nFALSE ,2\n  ,3\n")
    assert df["a"].tolist() == [True, False, None]


def test_comma_space_file_equals_tab_original(tmp_path):
    """The weights example rewritten with ', ' separators (sed 's/\\t/, /g') reads identically."""
    frames = {}
    for name in ("weights_input_chrt_data.csv", "weights_input_hrd_data.csv"):
        text = example_path(name).read_text(encoding="utf-8")
        padded = _write(tmp_path, name, text.replace("\t", ", "))
        frames[name] = (load_example(name), read_csv(padded))
        pd.testing.assert_frame_equal(frames[name][1], frames[name][0])
    (c0, c1), (h0, h1) = frames.values()
    out0 = gleampy.run_weights_module(c0, h0, show_indicator=False)
    out1 = gleampy.run_weights_module(c1, h1, show_indicator=False)
    assert list(out1) == list(out0)
    for key in out0:
        pd.testing.assert_frame_equal(out1[key], out0[key])


@pytest.mark.parametrize("system", ["mms_pasture", "mms_burned"])
def test_padded_manure_system_names_keep_their_group(tmp_path, system):
    """'mms_pasture ' must still be pasture (fread strips it), not 'other'."""
    cohort = load_example("emissions_manure_input_chrt_data.csv")
    fraction, factors = (load_example(n) for n in (
        "manure_management_system_fraction.csv", "manure_management_system_factors.csv"))
    padded = []
    for name in ("manure_management_system_fraction.csv", "manure_management_system_factors.csv"):
        text = example_path(name).read_text(encoding="utf-8").replace(system, system + "  ")
        padded.append(read_csv(_write(tmp_path, name, text)))
    assert (padded[0]["manure_management_system"] == system).sum() == (
        fraction["manure_management_system"] == system).sum() > 0
    expected = gleampy.run_emissions_manure_module(cohort, fraction, factors, show_indicator=False)
    actual = gleampy.run_emissions_manure_module(cohort, padded[0], padded[1], show_indicator=False)
    pd.testing.assert_frame_equal(actual, expected)


# --------------------------------------------------------------------------
# Missing values and empty fields
# --------------------------------------------------------------------------


def test_empty_field_is_na_in_numbers_and_empty_string_in_text(tmp_path):
    df = _read(tmp_path, "a,b,c\n1,,x\n2,3,\n")
    assert df["b"].dtype.kind in "if" and math.isnan(df["b"][0]) and df["b"][1] == 3
    assert df["c"].tolist() == ["x", ""]


def test_fread_missing_value_rules(tmp_path):
    df = _read(tmp_path, 'a,b,c,d,e,f,g\nTRUE,True,true,T,"",1,x\nFALSE,False,false,F,y,NA,\nNA,True,false,T,z,3,NA\n')
    assert [_kind(df[c]) for c in df.columns] == ["l", "l", "l", "c", "c", "n", "c"]
    assert df["a"].tolist() == [True, False, None]
    assert df["d"].tolist() == ["T", "F", "T"]  # T/F are not logical for fread
    assert df["e"].tolist() == ["", "y", "z"]
    assert _values(df["g"]) == ["x", "", None]  # unquoted NA is missing, empty is ""


def test_quoted_na_is_text_and_quoted_empty_is_missing_in_numbers(tmp_path):
    df = _read(tmp_path, 'a,b,c\n1,"",""\n2,"","NA"\n3,"5","x"\n')
    assert _kind(df["b"]) == "n" and math.isnan(df["b"][0]) and df["b"][2] == 5
    assert df["c"].tolist() == ["", "NA", "x"]
    df = _read(tmp_path, 'a,b\n1,"NA"\n2,"y"\n3,NA\n')
    assert _values(df["b"]) == ["NA", "y", None]
    df = _read(tmp_path, 'a,b\n1,"NA"\n2,3\n')  # a quoted NA makes the column character
    assert df["b"].tolist() == ["NA", "3"]
    df = _read(tmp_path, 'a,b\n1," 2 "\n2,3\n')  # quoted blanks are kept
    assert df["b"].tolist() == [" 2 ", "3"]


def test_quoted_numbers_are_numbers(tmp_path):
    df = _read(tmp_path, 'a,b,c\n"1.5","1",x\n"2.5","2",y\n')
    assert df["a"].tolist() == [1.5, 2.5] and df["b"].dtype == np.int64 and df["b"].tolist() == [1, 2]


def test_blank_feed_name_behaves_like_r(tmp_path):
    """fread reads a blank feed_name as "", so R accepts it in both tables."""
    ef = ("feed_id,feed_name,co2_feed_fertilizer,co2_feed_pesticides,co2_feed_crop_activities,"
          "co2_feed_luc_nopeat,co2_feed_luc_peat,n2o_feed_fertilizer,n2o_feed_manure_applied,"
          "n2o_feed_crop_residues,ch4_feed_rice\n")
    rations = _read(tmp_path, "herd_id,species_short,feed_name,feed_id,cohort_short,feed_ration_fraction\n"
                              "1,PGS,Maize,F1,FA,0.6\n1,PGS,,F2,FA,0.4\n", name="r.csv")
    one_blank = _read(tmp_path, ef + "F1,Maize,1,0,0,0,0,0,0,0,0\nF2,,2,0,0,0,0,0,0,0,0\n", name="e1.csv")
    two_blank = _read(tmp_path, ef + "F1,Maize,1,0,0,0,0,0,0,0,0\nF2,,2,0,0,0,0,0,0,0,0\n"
                                     "F3,,3,0,0,0,0,0,0,0,0\n", name="e2.csv")
    assert rations["feed_name"].tolist() == ["Maize", ""]
    out = gleampy.run_emissions_ration_module(rations, one_blank, show_indicator=False)
    assert out["co2_ration_fertilizer"].tolist() == pytest.approx([1.4])  # R: 1.4
    with pytest.raises(GleamValidationError, match="must be unique"):  # R: duplicated feed_name ""
        gleampy.run_emissions_ration_module(rations, two_blank, show_indicator=False)


# --------------------------------------------------------------------------
# All-empty columns are logical NA
# --------------------------------------------------------------------------


def test_all_empty_column_is_logical_na(tmp_path):
    df = _read(tmp_path, "a,b,c\n1,,NA\n2,,NA\n3,,NA\n")
    for c in ("b", "c"):
        assert df[c].dtype == object and df[c].tolist() == [None, None, None]
    df = _read(tmp_path, 'a,b\n1,""\n2,""\n')
    assert df["b"].tolist() == [None, None]


def test_all_empty_numeric_column_is_rejected_like_r(tmp_path):
    """R rejects an empty ch4_feed_rice column (fread types it logical)."""
    rations = load_example("feed_rations_share_chrt.csv")
    text = example_path("feed_emission_factors.csv").read_text(encoding="utf-8").splitlines()
    header = text[0].split("\t")
    j = header.index("ch4_feed_rice")
    rows = [line.split("\t") for line in text[1:]]
    for r in rows:
        r[j] = ""
    ef = read_csv(_write(tmp_path, "ef.tsv", "\n".join("\t".join(r) for r in [header] + rows) + "\n"))
    assert ef["ch4_feed_rice"].tolist() == [None] * len(ef)
    with pytest.raises(GleamValidationError, match="`ch4_feed_rice` must be a single numeric"):
        gleampy.run_emissions_ration_module(rations, ef, show_indicator=False)


def test_all_empty_mn_phase_columns_rejected_like_r(tmp_path):
    """R stops with "... must be numeric." for an empty MN phase column."""
    chrt = _write(tmp_path, "c.tsv", "herd_id\tspecies_short\tcohort_short\tnondemo_productive_phase_id\tdeath_rate\n"
                                    "14\tCHK\tFN\t1\t0.08\n14\tCHK\tFN\t2\t0.1\n")
    hrd = _write(tmp_path, "h.tsv",
                 "herd_id\tspecies_short\tcohort_stock_fem_annual_nondemo\tcohort_stock_mal_annual_nondemo\t"
                 "rest_between_nondemo_cycles_duration\tphase1_nondemo_fem_duration_days\t"
                 "phase2_nondemo_fem_duration_days\tphase1_nondemo_mal_duration_days\tphase2_nondemo_mal_duration_days\n"
                 "14\tCHK\t468457.2\t0\t14\t60\t365\t\t\n")
    with pytest.raises(GleamValidationError,
                       match=r"`herd_level_data\$phase1_nondemo_mal_duration_days` must be numeric"):
        gleampy.run_nondemographic_herd_module(read_csv(chrt), read_csv(hrd), show_indicator=False)


def test_bundled_all_empty_columns_still_run():
    """master_hrd_lvl_nondemo_data.csv has 20 all-empty columns that R (and Python) accept."""
    hrd = read_csv(example_path("master_hrd_lvl_nondemo_data.csv", "run_gleam_examples"))
    empty = [c for c in hrd.columns if hrd[c].isna().all()]
    assert len(empty) == 20 and all(hrd[c].dtype == object for c in empty)


# --------------------------------------------------------------------------
# Numeric tokens
# --------------------------------------------------------------------------

_NAN = "nan"
_NA = "na"
FREAD_TOKENS = {
    # token -> value fread gives in a numeric column ("nan"/"na" = NaN/NA, None = character column)
    "NaN": _NAN, "nan": _NAN, "NAN": _NAN, "-nan": _NAN, "-NaN": _NAN, "+NaN": _NAN, "NAN123": _NAN,
    "NaNQ": _NAN, "NaNS": _NAN, "NaN%": _NAN, "NaN1": _NAN, "qNaN": _NAN, "sNaN": _NAN,
    "1.#QNAN": _NAN, "-1.#QNAN": _NAN, "1.#SNAN": _NAN, "1.#IND": _NAN, "-1.#IND": _NAN,
    "#DIV/0!": _NAN, "-#DIV/0!": _NAN, "#VALUE!": _NAN,
    "#N/A": _NA, "+#N/A": _NA, "#NUM!": _NA, "#REF!": _NA, "#NAME?": _NA, "#NULL!": _NA,
    "NA": _NA, " NA ": _NA, "": _NA, " ": _NA,
    "Inf": math.inf, "+Inf": math.inf, "inf": math.inf, "+inf": math.inf, "INF": math.inf,
    "Infinity": math.inf, "1.#INF": math.inf,
    "-Inf": -math.inf, "-inf": -math.inf, "-INF": -math.inf, "-Infinity": -math.inf, "-1.#INF": -math.inf,
    "1e5": 1e5, "1E-3": 1e-3, ".5": 0.5, "5.": 5.0, "+3": 3.0, "00.5": 0.5, "-.5": -0.5, "+.5e-3": 0.5e-3,
    "1.5E+05": 1.5e5, "-0.0": -0.0, "0.30000000000000004": 0.30000000000000004,
    # character in fread
    "N/A": None, "n/a": None, "na": None, "null": None, "NULL": None, "None": None, "infinity": None,
    "INFINITY": None, "Infinit": None, "nan123": None, "nan%": None, "nAn": None, "NAn": None, "iNF": None,
    "InF": None, "1.#INFINITY": None, "1.#INF0": None, "-1.#IND00": None, "0x1.8p+0": None, "1d5": None,
    "1 000": None, "1.5e": None, "e5": None, "--1": None, "1_000": None, "0.1f": None, "TRUE": None,
    "T": None, "1.5.2": None, "1e400": None, "-1e400": None,
}


@pytest.mark.parametrize("token", list(FREAD_TOKENS))
def test_numeric_tokens_like_fread(tmp_path, token):
    df = _read(tmp_path, f"a,b\n1.5,x\n{token},y\n2.5,z\n")
    expected = FREAD_TOKENS[token]
    if expected is None:
        assert _kind(df["a"]) == "c"
        assert df["a"].tolist() == ["1.5", token.strip(), "2.5"]
        return
    assert df["a"].dtype == np.float64
    v = df["a"][1]
    if expected in (_NAN, _NA):
        assert math.isnan(v)
    else:
        assert v == expected and math.copysign(1, v) == math.copysign(1, expected)


def test_correctly_rounded_parsing(tmp_path):
    df = _read(tmp_path, "a\n0.1\n1.00000000000000011102230246251565404236316680908203125\n1e23\n"
                         "2.2250738585072011e-308\n")
    assert df["a"].tolist() == [0.1, 1.0, 1e23, 2.2250738585072011e-308]


def test_nan_tokens_keep_production_example_numeric(tmp_path):
    """NaN written for the empty milk_yield_day cells (e.g. by numpy.savetxt)."""
    chrt = load_example("production_input_chrt_data.csv")
    hrd = load_example("production_input_hrd_data.csv")
    lines = example_path("production_input_hrd_data.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    j = header.index("milk_yield_day")
    out_lines = [lines[0]]
    n_tok = 0
    for i, line in enumerate(lines[1:]):
        f = line.split("\t")
        if f[j] == "":
            f[j] = ("NaN", "nan", "#N/A")[n_tok % 3]
            n_tok += 1
        out_lines.append("\t".join(f))
    assert n_tok > 0
    hrd_nan = read_csv(_write(tmp_path, "h.tsv", "\n".join(out_lines) + "\n"))
    assert hrd_nan["milk_yield_day"].dtype == np.float64
    pd.testing.assert_frame_equal(hrd_nan, hrd)
    pd.testing.assert_frame_equal(
        gleampy.run_production_module(chrt, hrd_nan, show_indicator=False),
        gleampy.run_production_module(chrt, hrd, show_indicator=False),
    )


# --------------------------------------------------------------------------
# Logical columns, integers
# --------------------------------------------------------------------------


def test_logical_columns_need_one_spelling(tmp_path):
    df = _read(tmp_path, "a,b,c,d,e,f,g,h\nTRUE,True,true,T,t,yes,1,TRUE\nFALSE,False,false,F,f,no,0,False\n"
                         "NA,True,,T,t,yes,1,true\n")
    assert [_kind(df[c]) for c in df.columns] == ["l", "l", "l", "c", "c", "c", "i", "c"]
    assert df["c"].tolist() == [True, False, None]
    assert df["h"].tolist() == ["TRUE", "False", "true"]
    df = _read(tmp_path, 'a,b\n"TRUE",1\nFALSE,2\n')  # quoted TRUE is still logical
    assert df["a"].tolist() == [True, False]


def test_integer_columns(tmp_path):
    df = _read(tmp_path, "a,b,c\n+3,1.0,1\n007,2.0,NA\n-0,3,2\n")
    assert df["a"].dtype == np.int64 and df["a"].tolist() == [3, 7, 0]
    assert df["b"].dtype == np.float64
    assert df["c"].dtype == np.float64 and math.isnan(df["c"][1])  # integer with NA


# --------------------------------------------------------------------------
# Encodings and separators
# --------------------------------------------------------------------------


def test_non_utf8_files_fall_back_to_cp1252(tmp_path):
    df = _read(tmp_path, "feed_id,feed_name,value\n1,Ma\xefs grain,0.5\n2,Bl\xe9,0.25\n".encode("latin-1"))
    assert df["feed_name"].tolist() == ["Ma\xefs grain", "Bl\xe9"] and df["value"].tolist() == [0.5, 0.25]
    df = _read(tmp_path, "feed_id,feed_name,value\n1,caf\u00e9 \u20ac,0.5\n".encode("cp1252"))
    assert df["feed_name"].tolist() == ["caf\u00e9 \u20ac"]
    df = _read(tmp_path, b"\xef\xbb\xbfa,b\n1,x\n")  # BOM
    assert list(df.columns) == ["a", "b"]
    df = _read(tmp_path, "a,b\n1,Ma\u00efs\n".encode())
    assert df["b"].tolist() == ["Ma\u00efs"]


@pytest.mark.parametrize(
    "content, expected",
    [
        ("herd_id;value;x\n1;2,5;a\n2;3,5;b\n", [2.5, 3.5]),  # Excel (European locale) export
        ("herd_id;value;x\n1;2.5;a\n2;3.5;b\n", [2.5, 3.5]),
        ("herd_id|value|x\n1|2.5|a\n2|3.5|b\n", [2.5, 3.5]),
        ("herd_id\tvalue\tx\n1\t2,5\ta\n2\t3,5\tb\n", [2.5, 3.5]),
    ],
)
def test_separator_and_decimal_detection(tmp_path, content, expected):
    df = _read(tmp_path, content)
    assert list(df.columns) == ["herd_id", "value", "x"]
    assert df["herd_id"].tolist() == [1, 2] and df["value"].tolist() == expected
    assert df["x"].tolist() == ["a", "b"]


@pytest.mark.parametrize(
    "content, b",
    [
        # fread's dec="auto" balance: fields parsed with "," minus fields parsed with "."
        ("a;b\n1;2,5\n2;4\n", ["2,5", "4"]),      # balance 0 -> "." -> character
        ("a;b\n1;4\n2;2,5\n", [4.0, 2.5]),        # balance 1 -> ","
        ("a;b\n1;2,5\n2;4\n3;4,5\n", [2.5, 4.0, 4.5]),
        ("a;b\n1;2,5\n2;3,5\n3;1.5\n", ["2,5", "3,5", "1.5"]),  # "," chosen, 1.5 is then text
        ("a;b\n1;1,000\n2;2,500\n", [1.0, 2.5]),
    ],
)
def test_decimal_comma_balance_like_fread(tmp_path, content, b):
    assert _read(tmp_path, content)["b"].tolist() == b


@pytest.mark.parametrize(
    "content, col, expected",
    [
        # Once a column is double, its empty / NA fields and its NaN / Inf /
        # #N/A-style tokens count as fields parsed with ".".
        ("herd_id;cohort_short;offtake_rate;cohort_duration_days\n1;FA;0,2;\n1;FN;;\n1;MN;;\n",
         "offtake_rate", ["0,2", "", ""]),                       # 1 - 2 < 0 -> "." -> character
        ("herd_id;milk_yield_day;fibre_yield_year\n1;12,5;NaN\n2;8,25;NaN\n3;10,0;NaN\n",
         "milk_yield_day", ["12,5", "8,25", "10,0"]),            # 3 - 3 = 0 -> "."
        ("a;f\n1,5;1.5\n2,25;\n3,0;\n4,5;\n", "a", ["1,5", "2,25", "3,0", "4,5"]),  # 4 - 1 - 3 = 0
        ("a;b;c\n1,5;Inf;Inf\n2,5;Inf;Inf\n", "a", ["1,5", "2,5"]),
        ("a;b\n1,5;#N/A\n2,5;#N/A\n3,5;1.0\n", "a", ["1,5", "2,5", "3,5"]),
        ('a;b\n1,5;"1.5"\n2,5;""\n3,5;""\n4,5;""\n', "a", ["1,5", "2,5", "3,5", "4,5"]),
        ("a;b\n1,5;1,5\n2,5;NA\n3,5;NA\n4,5;NA\n", "a", [1.5, 2.5, 3.5, 4.5]),  # 4 + 1 - 3 > 0 -> ","
        # Missing fields of a column that is still logical or integer do not count.
        ("a;b;c\n1,5;1;1\n2,5;NA;NA\n3,5;NA;NA\n4,5;NA;NA\n", "a", [1.5, 2.5, 3.5, 4.5]),
        ("a;b;c\n1,5;NA;1\n2,5;NA;1,5\n3,5;NA;\n", "a", [1.5, 2.5, 3.5]),  # 3 + 1 - 1 > 0 -> ","
    ],
)
def test_decimal_balance_counts_missing_and_special_fields_like_fread(tmp_path, content, col, expected):
    assert _read(tmp_path, content)[col].tolist() == expected


def test_separator_preference(tmp_path):
    assert list(_read(tmp_path, "a,b;c\n1,2;3\n4,5;6\n").columns) == ["a", "b;c"]  # tie -> comma
    df = _read(tmp_path, "a;b;c,d\n1;2;3,4\n5;6;7,8\n")  # more fields -> semicolon (and dec ",")
    assert list(df.columns) == ["a", "b", "c,d"] and df["c,d"].tolist() == [3.4, 7.8]
    df = _read(tmp_path, "feed_id\tfeed_name\tvalue\n1\tBarley, straw\t0.5\n2\tMaize, grain\t0.25\n")
    assert df["feed_name"].tolist() == ["Barley, straw", "Maize, grain"]


def test_quoted_fields_with_separators_quotes_and_newlines(tmp_path):
    df = _read(tmp_path, 'a,b,c\n1,"x\ty",2\n3,"z, w",4\n5,"line1\nline2",6\n7,"q""q",8\n')
    assert df["b"].tolist() == ["x\ty", "z, w", "line1\nline2", 'q"q']
    assert df["c"].tolist() == [2, 4, 6, 8]


def test_explicit_sep_and_dec(tmp_path):
    df = _read(tmp_path, "a;b\n1;2,5\n2;4\n", sep=";", dec=",")
    assert df["b"].tolist() == [2.5, 4.0]
    with pytest.raises(ValueError):
        _read(tmp_path, "a\n1\n", dec="x")


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------


def test_line_endings_header_and_blank_lines(tmp_path):
    df = _read(tmp_path, "a,b,c\r\n1, x ,TRUE\r\n2,y ,FALSE\r\n")
    assert df["b"].tolist() == ["x", "y"] and df["c"].tolist() == [True, False]
    assert list(_read(tmp_path, " a , b ,c\n1,2,3\n").columns) == ["a", "b", "c"]
    assert list(_read(tmp_path, "a,,c\n1,2,3\n").columns) == ["a", "V2", "c"]
    df = _read(tmp_path, "\n\na,b\n1,x\n2,y\n\n\n")  # leading blank lines skipped, trailing ignored
    assert df["a"].tolist() == [1, 2]
    df = _read(tmp_path, "a\n1.5\n\n2.5\n\n\n")  # one column: blank lines are NA rows
    assert len(df) == 5 and math.isnan(df["a"][1]) and math.isnan(df["a"][4])
    df = _read(tmp_path, "a,b\n1,x\n,\n2,y\n")  # a line of empty fields is a row
    assert _values(df["b"]) == ["x", "", "y"] and math.isnan(df["a"][1])


def test_blank_line_ends_data_with_warning(tmp_path):
    with pytest.warns(GleamWarning, match="Stopped early on line 3"):
        df = _read(tmp_path, "a,b\n1,x\n\n2,y\n3,z\n")
    assert df["a"].tolist() == [1]
    with pytest.warns(GleamWarning, match="Expected 2 fields but found 3"):
        df = _read(tmp_path, "a,b\n1,x\n2,y,z\n")
    assert df["a"].tolist() == [1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(_read(tmp_path, "a,b\n1,x\n2,y\n\n\n")) == 2


def test_empty_file(tmp_path):
    assert _read(tmp_path, "").empty
    df = _read(tmp_path, "a,b\n")
    assert list(df.columns) == ["a", "b"] and len(df) == 0


# --------------------------------------------------------------------------
# nrows / usecols, and rejection of other pandas.read_csv keywords
# --------------------------------------------------------------------------

T1 = "a,b,c\n1,TRUE,x\n2,FALSE,y\nz,3.5,\n"


def test_nrows_types_from_rows_read_like_fread(tmp_path):
    # fread(nrows = 1 / 2): a integer, b logical, c character; nrows = 3: all character
    for n in (1, 2):
        df = _read(tmp_path, T1, nrows=n)
        assert len(df) == n and [_kind(df[c]) for c in "abc"] == ["i", "l", "c"]
    df = _read(tmp_path, T1, nrows=3)
    assert [_kind(df[c]) for c in "abc"] == ["c", "c", "c"] and _values(df["c"]) == ["x", "y", ""]
    assert _read(tmp_path, T1, nrows=np.int64(2)).equals(_read(tmp_path, T1, nrows=2))
    pd.testing.assert_frame_equal(_read(tmp_path, T1, nrows=99), _read(tmp_path, T1))
    df = _read(tmp_path, T1, nrows=0)
    assert list(df.columns) == ["a", "b", "c"] and len(df) == 0
    df = _read(tmp_path, "a\n1\n\n3\n", nrows=2)  # one column: a blank line is an NA row
    assert len(df) == 2 and df["a"][0] == 1 and math.isnan(df["a"][1])
    for bad in (-1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="nrows"):
            _read(tmp_path, T1, nrows=bad)


def test_nrows_stops_before_malformed_lines_like_fread(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # fread(nrows = 2) does not warn about line 4
        df = _read(tmp_path, "a,b\n1,2\n3,4\n5,6,7\n", nrows=2)
        assert df["a"].tolist() == [1, 3]
        assert len(_read(tmp_path, "a,b\n1,2\n3,4\n\n5,6\n", nrows=2)) == 2
        assert len(_read(tmp_path, "a,b\n1,2\n3,4\n5,6,7\n", nrows=0)) == 0
    # fread(nrows = 3) reaches the bad line, stops there and warns
    with pytest.warns(GleamWarning, match="Expected 2 fields but found 3"):
        df = _read(tmp_path, "a,b\n1,2\n3,4\n5,6,7\n", nrows=3)
    assert df["a"].tolist() == [1, 3]
    with pytest.warns(GleamWarning, match="Stopped early on line 4"):
        df = _read(tmp_path, "a,b\n1,2\n3,4\n\n5,6\n", nrows=3)
    assert df["a"].tolist() == [1, 3]


def test_usecols_like_pandas(tmp_path):
    full = _read(tmp_path, T1)
    for usecols in (["c", "a"], ("a", "c"), [2, 0], np.array([0, 2]), lambda name: name != "b"):
        df = _read(tmp_path, T1, usecols=usecols)
        pd.testing.assert_frame_equal(df, full[["a", "c"]])  # file order, same types
    df = _read(tmp_path, T1, usecols=["b"], nrows=2)
    assert list(df.columns) == ["b"] and _kind(df["b"]) == "l"
    df = _read(tmp_path, T1, usecols=[])
    assert df.shape == (3, 0)
    with pytest.raises(ValueError, match=r"not found in the header: \['zz'\]"):
        _read(tmp_path, T1, usecols=["zz", "a"])
    with pytest.raises(ValueError, match="out of range"):
        _read(tmp_path, T1, usecols=[3])
    with pytest.raises(ValueError):
        _read(tmp_path, T1, usecols="a")
    with pytest.raises(ValueError, match="not a mix"):
        _read(tmp_path, T1, usecols=["a", 1])


def test_usecols_on_bundled_example():
    path = example_path("weights_input_hrd_data.csv")
    full = read_csv(path)
    df = read_csv(path, usecols=["herd_id", "species_short"], nrows=5)
    pd.testing.assert_frame_equal(df, full.loc[:4, ["herd_id", "species_short"]])


def test_other_pandas_keywords_are_rejected_clearly(tmp_path):
    with pytest.raises(TypeError, match=r"unsupported keyword argument\(s\) 'skiprows'.*pandas.read_csv"):
        _read(tmp_path, T1, skiprows=1)
    with pytest.raises(TypeError, match="'dtype', 'na_values'"):
        _read(tmp_path, T1, dtype=str, na_values=["x"])


# --------------------------------------------------------------------------
# Bundled examples: column classes and row counts as fread reads them
# --------------------------------------------------------------------------

# i = integer, n = numeric (or integer with NA), l = logical, c = character
FREAD_CLASSES = {
    ("run_gleam_examples", "feed_emission_factors.csv"): ("cccnnnnnnnnn", 43),
    ("run_gleam_examples", "feed_quality.csv"): ("cccnnnnnnnnni", 45),
    ("run_gleam_examples", "feed_rations_share_chrt.csv"): ("iccnccn", 585),
    ("run_gleam_examples", "manure_management_system_factors.csv"): ("icnnnnnnn", 81),
    ("run_gleam_examples", "manure_management_system_fraction.csv"): ("icncn", 520),
    ("run_gleam_examples", "master_chrt_lvl_no_structure_mixed_data.csv"): ("iccnnnnnnl", 106),
    ("run_gleam_examples", "master_chrt_lvl_no_structure_nondemo_data.csv"): ("iccnnnnnl", 18),
    ("run_gleam_examples", "master_chrt_lvl_structure_data.csv"): ("iccninnnnlnnn", 94),
    ("run_gleam_examples", "master_hrd_lvl_mixed_data.csv"): ("icnnnnnnnnnnnnnnnnnnnnninnnnnnnnnnnnnnnnnnnnnnnn", 15),
    ("run_gleam_examples", "master_hrd_lvl_nondemo_data.csv"): ("icllnllliiiinnnnnnllllllllllinilllllnnnnii", 2),
    ("run_gleam_examples", "master_hrd_lvl_structure_data.csv"): ("icnnnnnnnnnnnnnninnnnnnnnnnnnnnnnnnnnnnnnnn", 15),
    ("run_modules_examples", "aggregation_allocation_input_data.csv"): ("icccn", 481),
    ("run_modules_examples", "aggregation_input_chrt_data.csv"): ("iccnnnnnnnniiinnnnnnnnnnnnnnnnnnnninnnin", 88),
    ("run_modules_examples", "allocation_input_chrt_data.csv"): ("iccnnnnnninl", 88),
    ("run_modules_examples", "allocation_input_hrd_data.csv"): ("icnnnnn", 13),
    ("run_modules_examples", "emissions_direct_input_chrt_no_structure_data.csv"): ("iccinnnn", 72),
    ("run_modules_examples", "emissions_direct_input_chrt_no_structure_ration_quality_data.csv"): ("iccinnnnnnnnnn", 72),
    ("run_modules_examples", "emissions_direct_input_chrt_structure_data.csv"): ("iccninnnn", 72),
    ("run_modules_examples", "emissions_direct_input_chrt_structure_ration_quality_data.csv"): ("iccninnnnnnnnnn", 72),
    ("run_modules_examples", "emissions_direct_input_hrd_data.csv"): ("icnnnnnnnnniinnnninnnnnnnnnnnnnnnniii", 12),
    ("run_modules_examples", "emissions_enteric_input_chrt_data.csv"): ("iccnnn", 88),
    ("run_modules_examples", "emissions_manure_input_chrt_data.csv"): ("icnnnnnn", 88),
    ("run_modules_examples", "feed_emission_factors.csv"): ("cnnnnnnnnn", 43),
    ("run_modules_examples", "feed_quality.csv"): ("cnnnnnnnni", 45),
    ("run_modules_examples", "feed_rations_share_chrt.csv"): ("iccccnn", 536),
    ("run_modules_examples", "herd_all_input_chrt_data.csv"): ("inccnnn", 88),
    ("run_modules_examples", "herd_all_input_hrd_data.csv"): ("innnnnnnnnnn", 13),
    ("run_modules_examples", "herd_simulation_input_chrt_data.csv"): ("iccinn", 78),
    ("run_modules_examples", "herd_simulation_input_hrd_data.csv"): ("icnnnnnn", 13),
    ("run_modules_examples", "manure_management_system_factors.csv"): ("icnnnnnnn", 62),
    ("run_modules_examples", "manure_management_system_fraction.csv"): ("iccnn", 462),
    ("run_modules_examples", "metabolic_energy_req_input_chrt_data.csv"): ("iccnnnnnnnninnnnnl", 88),
    ("run_modules_examples", "metabolic_energy_req_input_hrd_data.csv"): ("icinnnnnnnnnnnnnnnnnnn", 13),
    ("run_modules_examples", "nitrogen_balance_input_chrt_data.csv"): ("iccnnninnl", 88),
    ("run_modules_examples", "nitrogen_balance_input_hrd_data.csv"): ("icnnnnnnnnnn", 13),
    ("run_modules_examples", "nondemographic_herd_input_chrt_data.csv"): ("iccin", 16),
    ("run_modules_examples", "nondemographic_herd_input_hrd_data.csv"): ("icnninnnn", 5),
    ("run_modules_examples", "production_input_chrt_data.csv"): ("iccnnnnl", 88),
    ("run_modules_examples", "production_input_hrd_data.csv"): ("icnnnnnnnnnnnnnn", 13),
    ("run_modules_examples", "weights_input_chrt_data.csv"): ("iccinn", 88),
    ("run_modules_examples", "weights_input_hrd_data.csv"): ("icnnnnnnnnnnnnnn", 13),
}


def test_every_bundled_example_is_listed():
    found = {(k, p.name) for k in ("run_modules_examples", "run_gleam_examples")
             for p in example_dir(k).glob("*.csv")}
    assert found == set(FREAD_CLASSES)


@pytest.mark.parametrize("kind, name", list(FREAD_CLASSES))
def test_bundled_examples_have_fread_column_classes(kind, name):
    classes, nrow = FREAD_CLASSES[(kind, name)]
    df = load_example(name, kind)
    assert len(df) == nrow
    assert "".join(_kind(df[c]) for c in df.columns) == classes
