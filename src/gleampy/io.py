"""Reading GLEAM input tables and the bundled example data.

:func:`read_csv` reads a delimited text file the way ``data.table::fread``
(data.table 1.18, default arguments) does for the files GLEAM uses. The rules
below were checked against ``fread`` field by field:

* **Encoding.** UTF-8 (a byte-order mark is dropped). A file that is not valid
  UTF-8 is decoded as Windows-1252, or as Latin-1 if it uses one of the five
  bytes cp1252 leaves undefined. (``fread`` passes such bytes through
  undecoded.)
* **Separator.** Detected among comma, tab, pipe and semicolon: the one that
  splits the header line into more than one field and gives the longest run
  of lines with that same number of fields; ties go to the larger number of
  fields, then to the order comma, tab, pipe, semicolon.
* **Decimal mark.** ``"."``, or ``","`` when the separator is not a comma and
  more fields parse as numbers with a decimal comma than with a decimal point
  (``fread``'s ``dec = "auto"`` balance, computed here over all rows instead
  of ``fread``'s sample of rows). As in ``fread``, once a column has been
  typed double, its empty and ``NA`` fields and its ``NaN`` / ``Inf`` /
  ``#N/A``-style tokens count as decimal-point fields, so a decimal-comma
  file whose numeric columns are mostly blank can be read with ``"."`` (and
  those columns as text). Pass ``dec=","`` for such files.
* **White space.** Leading and trailing blanks of unquoted fields are removed
  (``strip.white = TRUE``); quoted fields are kept verbatim.
* **Missing values.** An unquoted ``NA`` is missing in every column. An empty
  field is missing in numeric and logical columns but is the empty string
  ``""`` in character columns. A quoted ``"NA"`` is the text ``NA``.
* **Column types**, tried in this order:

  - *logical*: every value is ``TRUE``/``FALSE``, or every value is
    ``True``/``False``, or every value is ``true``/``false`` (mixed spellings
    or ``T``/``F`` give a character column). Logical columns are object
    arrays of ``True`` / ``False`` / ``None``;
  - *integer*: ``int64``, or ``float64`` when the column has missing values;
  - *double*: ``float64``, parsed with correct rounding. The special tokens
    ``fread`` accepts in numeric columns are numbers too: ``NaN``, ``nan``,
    ``NAN``, ``1.#QNAN``, ``1.#IND``, ``#DIV/0!``, ``#VALUE!`` (NaN),
    ``#N/A``, ``#NUM!``, ``#REF!``, ``#NAME?``, ``#NULL!`` (NA) and ``Inf``,
    ``inf``, ``INF``, ``Infinity``, ``1.#INF`` (all with an optional sign);
  - *character* otherwise (strings such as ``None``, ``N/A`` or ``null``
    stay text, like in ``fread``).

  A column without any value (every field empty or ``NA``) is a logical NA
  column, an object array of ``None``, because ``fread`` types it logical.
* **Blank lines.** Leading blank lines are skipped and trailing ones ignored.
  In a one-column file a blank line is a missing value. In a file with
  several columns a blank line, or a line with a different number of fields,
  ends the data: the rest of the file is discarded with a
  :class:`~gleampy.validation._shared.GleamWarning`, as ``fread`` stops early
  with a warning.
* **Subsets.** ``nrows`` reads only the first rows, like ``fread``'s
  ``nrows``: column types come from those rows, and a malformed line after
  them is not looked at (``nrows=0`` returns the empty columns typed from the
  whole file). ``usecols`` keeps some columns, with the meaning it has in
  :func:`pandas.read_csv` (names or positions, or a callable on the names;
  columns stay in file order). Other :func:`pandas.read_csv` options are not
  accepted.

Not reproduced: ``fread``'s header auto-detection (the first line is always
the header), its row sampling for large files, its quote-rule fallbacks for
malformed quoting (``""`` inside a quoted field is read as one quote),
``integer64`` columns (whole numbers are ``int64``), and its rejection of
numbers outside its parser's range (more than 18 integer digits, decimal
exponents beyond about +-350). Python parses those, except values that
overflow to infinity, which give a character column as in ``fread``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["example_dir", "example_path", "load_example", "read_csv"]

#: Separator candidates, in ``fread``'s order of preference.
_SEPS = (",", "\t", "|", ";")

#: Number of lines used to detect the separator (``fread``'s JUMPLINES).
_SEP_SAMPLE_LINES = 100

#: Logical spellings -> (family, value); a logical column uses one family only.
_LOGICAL_FAMILY = {
    "TRUE": (0, True), "FALSE": (0, False),
    "True": (1, True), "False": (1, False),
    "true": (2, True), "false": (2, False),
}

_INT_RE = re.compile(r"[+-]?\d+")
_FLOAT_RE = {
    ".": re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"),
    ",": re.compile(r"[+-]?(?:\d+,?\d*|,\d+)(?:[eE][+-]?\d+)?"),
}
_NAN_RE = re.compile(r"[+-]?(?:nan|NaN[%QS]?\d*|NAN\d*|[qs]NaN\d*|1\.#(?:QNAN|SNAN|IND)|#DIV/0!|#VALUE!)")
_NA_RE = re.compile(r"[+-]?(?:#N/A|#NUM!|#REF!|#NAME\?|#NULL!)")
_INF_RE = re.compile(r"([+-]?)(?:Inf(?:inity)?|inf|INF|1\.#INF)")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def read_csv(
    path: str | os.PathLike,
    *,
    sep: str = "auto",
    dec: str = "auto",
    encoding: str | None = None,
    nrows: int | None = None,
    usecols: Iterable[str] | Iterable[int] | Callable[[str], bool] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a GLEAM input table the way ``data.table::fread`` does.

    See the :mod:`gleampy.io` module documentation for the exact rules
    (separator and decimal detection, white space, missing values, column
    types, blank lines). This is not a wrapper of :func:`pandas.read_csv`:
    only the options below are accepted.

    Parameters
    ----------
    path : str or os.PathLike
        Delimited text file whose first line holds the column names.
    sep : str, default "auto"
        Field separator. ``"auto"`` detects it among ``","``, ``"\\t"``,
        ``"|"`` and ``";"``.
    dec : {"auto", ".", ","}, default "auto"
        Decimal mark. ``"auto"`` uses ``"."`` with a comma separator and
        otherwise decides from the data, like ``fread``.
    encoding : str, optional
        Text encoding. By default UTF-8 (with or without byte-order mark),
        falling back to Windows-1252 / Latin-1 when the file is not valid
        UTF-8.
    nrows : int, optional
        Number of data rows to read (``fread``'s ``nrows``); all rows by
        default. Column types are decided from the rows read. ``0`` returns
        the column names with empty columns typed from the whole file
        (without the warning about discarded lines).
    usecols : list of str, list of int or callable, optional
        Columns to keep, as in :func:`pandas.read_csv`: header names (all
        must exist), 0-based positions, or a function called with each
        column name. The columns keep their file order.
    **kwargs
        Not accepted: any other keyword raises ``TypeError``. Read the file
        with :func:`pandas.read_csv` if you need its other options (its
        parsing rules differ from ``fread``'s).

    Returns
    -------
    pandas.DataFrame
        One column per header field, with a default ``RangeIndex``. Numeric
        columns are ``int64`` / ``float64``; logical columns (and columns
        without any value) are object arrays of ``True`` / ``False`` /
        ``None``; character columns hold ``str`` values with missing values
        as NA.
    """
    if kwargs:
        raise TypeError(
            f"read_csv() got unsupported keyword argument(s) {', '.join(map(repr, kwargs))}: "
            "gleampy.io.read_csv reads files like data.table::fread and accepts only sep, dec, "
            "encoding, nrows and usecols. Use pandas.read_csv for other options."
        )
    if dec not in ("auto", ".", ","):
        raise ValueError("`dec` must be 'auto', '.' or ','.")
    if nrows is not None and (
        isinstance(nrows, (bool, np.bool_)) or not isinstance(nrows, (int, np.integer)) or nrows < 0
    ):
        raise ValueError("`nrows` must be a non-negative integer or None.")
    text = _decode(Path(path).read_bytes(), encoding)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.removesuffix("\n")  # terminator of the last line

    # Leading blank lines are skipped, as fread skips them before the header.
    start = 0
    skipped_lines = 0
    while True:
        nl = text.find("\n", start)
        if nl < 0 or text[start:nl].strip():
            break
        start = nl + 1
        skipped_lines += 1
    text = text[start:]
    if not text.strip():
        return pd.DataFrame()

    if sep == "auto":
        sep = _detect_sep(text)
    elif len(sep) != 1 or sep in ('"', "\n", "\r"):
        raise ValueError("`sep` must be a single character other than a quote or a newline.")

    records, quoted, line_no = _tokenize(text, sep)
    header = records[0]
    ncol = len(header)
    names = [v.strip() or f"V{j + 1}" for j, v in enumerate(header)]
    selected = _usecols_positions(usecols, names)
    # nrows = 0: the types come from every row (silently), then no row is kept
    rows, rows_quoted = _data_rows(
        records, quoted, line_no, ncol, skipped_lines, sep,
        nrows=None if nrows == 0 else nrows, warn_discarded=nrows != 0,
    )

    blanks = " " if sep == "\t" else " \t"
    raw_cols = [list(c) for c in zip(*rows)] if rows else [[] for _ in range(ncol)]
    if rows_quoted is None:
        columns = [[v.strip(blanks) for v in col] for col in raw_cols]
        col_quoted: list[list[bool] | None] = [None] * ncol
    else:
        col_quoted = [list(c) for c in zip(*rows_quoted)] if rows else [[] for _ in range(ncol)]
        columns = [
            [v if q else v.strip(blanks) for v, q in zip(col, qcol)]
            for col, qcol in zip(raw_cols, col_quoted)
        ]

    if dec == "auto":
        dec = "." if sep == "," else _detect_dec(columns, col_quoted)

    data = {j: _convert_column(columns[j], col_quoted[j], dec) for j in selected}
    df = pd.DataFrame(data, index=pd.RangeIndex(len(rows)))
    df.columns = [names[j] for j in selected]
    if nrows == 0:
        df = df.iloc[:0]
    return df


def example_dir(kind: str = "run_modules_examples") -> Path:
    """Directory of the bundled example inputs.

    Equivalent of ``system.file("extdata/<kind>", package = "gleam")``.

    Parameters
    ----------
    kind : {"run_modules_examples", "run_gleam_examples"}
        Example set: inputs of the individual ``run_*_module`` functions, or
        of the ``run_gleam`` pipeline.

    Returns
    -------
    pathlib.Path
        Directory holding the example CSV files.
    """
    if kind not in ("run_modules_examples", "run_gleam_examples"):
        raise ValueError("kind must be 'run_modules_examples' or 'run_gleam_examples'")
    return Path(str(resources.files("gleampy.data").joinpath(kind)))


def example_path(name: str, kind: str = "run_modules_examples") -> Path:
    """Path of one bundled example file.

    Parameters
    ----------
    name : str
        File name, e.g. ``"weights_input_chrt_data.csv"``.
    kind : {"run_modules_examples", "run_gleam_examples"}
        Example set (see :func:`example_dir`).

    Returns
    -------
    pathlib.Path
        Path of the file. Raises ``FileNotFoundError`` if there is no such
        example.
    """
    p = example_dir(kind) / name
    if not p.exists():
        raise FileNotFoundError(f"No bundled example named {name!r} in {kind}")
    return p


def load_example(name: str, kind: str = "run_modules_examples") -> pd.DataFrame:
    """Load one bundled example table with :func:`read_csv`.

    Parameters
    ----------
    name : str
        File name, e.g. ``"weights_input_chrt_data.csv"``.
    kind : {"run_modules_examples", "run_gleam_examples"}
        Example set (see :func:`example_dir`).

    Returns
    -------
    pandas.DataFrame
        The table as ``fread`` reads it.
    """
    return read_csv(example_path(name, kind))


def _usecols_positions(usecols: Any, names: list[str]) -> list[int]:
    """0-based positions of the columns kept by ``usecols`` (pandas meaning), in file order."""
    if usecols is None:
        return list(range(len(names)))
    if callable(usecols):
        return [j for j, nm in enumerate(names) if usecols(nm)]
    if isinstance(usecols, (str, bytes)) or not isinstance(usecols, Iterable):
        raise ValueError("`usecols` must be a list of column names, a list of positions or a callable.")
    wanted = list(usecols)
    if all(isinstance(c, str) for c in wanted):
        missing = [c for c in wanted if c not in names]
        if missing:
            raise ValueError(f"`usecols` names not found in the header: {missing}")
        keep = set(wanted)
        return [j for j, nm in enumerate(names) if nm in keep]
    if all(isinstance(c, (int, np.integer)) and not isinstance(c, (bool, np.bool_)) for c in wanted):
        bad = [int(c) for c in wanted if not 0 <= c < len(names)]
        if bad:
            raise ValueError(f"`usecols` positions out of range for {len(names)} columns: {bad}")
        return sorted({int(c) for c in wanted})
    raise ValueError("`usecols` must be all column names or all positions, not a mix.")


# --------------------------------------------------------------------------
# Decoding and tokenising
# --------------------------------------------------------------------------


def _decode(raw: bytes, encoding: str | None) -> str:
    if encoding is not None:
        return raw.decode(encoding).removeprefix("\ufeff")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


_FIELD_PATTERNS: dict[str, re.Pattern[str]] = {}


def _field_pattern(sep: str) -> re.Pattern[str]:
    """Regex for one field plus its terminator (separator, newline or end of text).

    Groups: 1 opening quote (``""`` when unquoted), 2 quoted content, 3 unquoted
    text, 4 terminator. A quoted field may be surrounded by blanks and may span
    lines; ``""`` inside it is an escaped quote. A field that starts with a
    quote but is not a well-formed quoted field is read as unquoted text.
    """
    pat = _FIELD_PATTERNS.get(sep)
    if pat is None:
        s = re.escape(sep)
        ws = "[ ]*" if sep == "\t" else r"[ \t]*"
        pat = re.compile(rf'(?:{ws}(")([^"]*(?:""[^"]*)*)"{ws}(?={s}|\n|\Z)|([^{s}\n]*))({s}|\n|\Z)')
        _FIELD_PATTERNS[sep] = pat
    return pat


def _tokenize(text: str, sep: str) -> tuple[list[list[str]], list[list[bool]] | None, list[int]]:
    """Split ``text`` (no final newline) into records of fields.

    Returns the records, per-field "was quoted" flags (``None`` when the text
    contains no quote character) and the 1-based line number of each record.
    """
    if '"' not in text:
        lines = text.split("\n")
        return [ln.split(sep) for ln in lines], None, list(range(1, len(lines) + 1))
    records: list[list[str]] = []
    quoted: list[list[bool]] = []
    line_no: list[int] = []
    rec: list[str] = []
    qrec: list[bool] = []
    line = rec_line = 1
    for q, content, raw, term in _field_pattern(sep).findall(text):
        if q:
            rec.append(content.replace('""', '"'))
            qrec.append(True)
            line += content.count("\n")
        else:
            rec.append(raw)
            qrec.append(False)
        if term == sep:
            continue
        records.append(rec)
        quoted.append(qrec)
        line_no.append(rec_line)
        if term == "":  # end of text
            break
        rec, qrec = [], []
        line += 1
        rec_line = line
    return records, quoted, line_no


def _detect_sep(text: str) -> str:
    """Separator that splits the header into >1 field with the longest consistent run."""
    sample = "\n".join(text.split("\n", _SEP_SAMPLE_LINES)[:_SEP_SAMPLE_LINES])
    best, best_score = ",", None
    for sep in _SEPS:
        if sep not in sample:
            continue
        records = _tokenize(sample, sep)[0]
        nf = len(records[0])
        if nf < 2:
            continue
        run = next((i for i, rec in enumerate(records) if len(rec) != nf), len(records))
        score = (run, nf)
        if best_score is None or score > best_score:
            best, best_score = sep, score
    return best


def _is_blank(rec: list[str]) -> bool:
    return len(rec) == 1 and not rec[0].strip()


def _data_rows(
    records: list[list[str]],
    quoted: list[list[bool]] | None,
    line_no: list[int],
    ncol: int,
    skipped_lines: int,
    sep: str,
    nrows: int | None = None,
    warn_discarded: bool = True,
) -> tuple[list[list[str]], list[list[bool]] | None]:
    """Data records after the header, cut where ``fread`` stops reading.

    With ``nrows``, at most ``nrows`` records are kept and, when they are all
    complete, the lines after them are not examined (no warning about them),
    as ``fread`` stops once it has read ``nrows`` rows.
    """
    body = records[1:]
    qbody = quoted[1:] if quoted is not None else None
    if ncol == 1:
        # One column: a blank line is a missing value; every record has 1 field.
        if nrows is not None:
            body = body[:nrows]
            qbody = qbody[:nrows] if qbody is not None else None
        return body, qbody
    limit = len(body) if nrows is None else min(nrows, len(body))
    end = next((i for i in range(limit) if len(body[i]) != ncol), None)
    if end is None:  # every record up to the limit is complete
        end = limit
    else:  # fread stops at the first incomplete line and warns about the rest
        rest = [r for r in body[end:] if not _is_blank(r)]
        if rest and warn_discarded:
            from .validation._shared import warn

            bad = body[end]
            found = 0 if _is_blank(bad) else len(bad)
            warn(
                f"Stopped early on line {line_no[end + 1] + skipped_lines}. Expected {ncol} "
                f"fields but found {found}. Discarded {len(rest)} non-empty line(s), the "
                f"first being <<{sep.join(rest[0])}>>."
            )
    if end < len(body):
        body = body[:end]
        if qbody is not None:
            qbody = qbody[:end]
    return body, qbody


# --------------------------------------------------------------------------
# Column typing
# --------------------------------------------------------------------------


def _missing(vals: list[str], quoted: list[bool] | None) -> list[bool]:
    """Missing in a numeric / logical column: empty, or an unquoted ``NA``."""
    if quoted is None:
        return [v == "" or v == "NA" for v in vals]
    return [v == "" or (v == "NA" and not q) for v, q in zip(vals, quoted)]


def _is_special(v: str) -> bool:
    return bool(_NAN_RE.fullmatch(v) or _NA_RE.fullmatch(v) or _INF_RE.fullmatch(v))


def _detect_dec(columns: list[list[str]], quoted: list[list[bool] | None]) -> str:
    """``fread``'s ``dec = "auto"``: fields parsed with "," minus fields parsed with ".".

    Each column is followed through ``fread``'s type ladder (logical, integer,
    double, character). While it is double, a field that parses with a
    decimal point counts -1 (whole numbers, empty and ``NA`` fields and the
    ``NaN`` / ``Inf`` / ``#N/A``-style tokens included) and one that parses
    only with a decimal comma counts +1. ``","`` wins on a positive balance.
    """
    balance = 0
    dot, comma = _FLOAT_RE["."], _FLOAT_RE[","]
    for vals, q in zip(columns, quoted):
        state = 0  # 0 logical, 1 integer, 2 double
        family = None
        for v, m in zip(vals, _missing(vals, q)):
            if m:
                if state == 2:
                    balance -= 1  # fread parses an empty / NA field of a double column with "."
                continue
            if state == 0:
                fam = _LOGICAL_FAMILY.get(v)
                if fam is not None and family in (None, fam[0]):
                    family = fam[0]
                    continue
                state = 1
            if state == 1:
                if _INT_RE.fullmatch(v):
                    continue
                state = 2
            if dot.fullmatch(v):
                balance -= 1
            elif comma.fullmatch(v):
                balance += 1
            elif _is_special(v):
                balance -= 1  # NaN / Inf / #N/A-style token, also parsed with "."
            else:
                break  # character column: fread stops parsing it as a number
    return "," if balance > 0 else "."


def _parse_float(v: str, dec: str) -> float | None:
    """One non-missing field as a double, or ``None`` if it is not a number."""
    if _FLOAT_RE[dec].fullmatch(v):
        x = float(v if dec == "." else v.replace(",", "."))
        return None if np.isinf(x) else x  # overflow: not a number for fread
    m = _INF_RE.fullmatch(v)
    if m:
        return -np.inf if m.group(1) == "-" else np.inf
    if _NAN_RE.fullmatch(v) or _NA_RE.fullmatch(v):
        return np.nan
    return None


def _convert_column(vals: list[str], quoted: list[bool] | None, dec: str):
    """Type one column like ``fread``: logical, integer, double, else character."""
    n = len(vals)
    miss = _missing(vals, quoted)
    present = [v for v, m in zip(vals, miss) if not m]
    if not present:  # logical NA column
        out = np.empty(n, dtype=object)
        out[:] = None
        return out

    fam = _LOGICAL_FAMILY.get(present[0])
    if fam is not None and all(_LOGICAL_FAMILY.get(v, (-1,))[0] == fam[0] for v in present):
        out = np.empty(n, dtype=object)
        out[:] = [None if m else _LOGICAL_FAMILY[v][1] for v, m in zip(vals, miss)]
        return out

    if all(_INT_RE.fullmatch(v) for v in present):
        ints = [0 if m else int(v) for v, m in zip(vals, miss)]
        if all(-(2**63) <= i < 2**63 for i in ints):
            arr = np.array(ints, dtype=np.int64)
            if any(miss):
                out = arr.astype("float64")
                out[np.asarray(miss)] = np.nan
                return out
            return arr

    parsed = [np.nan if m else _parse_float(v, dec) for v, m in zip(vals, miss)]
    if all(x is not None for x in parsed):
        return np.array(parsed, dtype="float64")

    # Character: an unquoted NA is missing; empty fields stay "".
    if quoted is None:
        chars = [None if v == "NA" else v for v in vals]
    else:
        chars = [None if (v == "NA" and not q) else v for v, q in zip(vals, quoted)]
    # pandas' default string representation ("str" dtype on pandas 3, object before)
    return pd.Series(chars, index=pd.RangeIndex(n))
