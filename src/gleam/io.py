"""Reading GLEAM input tables and the bundled example data.

:func:`read_csv` mimics ``data.table::fread`` as used by the R package:

* the separator (tab or comma) is detected from the header line;
* only empty fields and ``"NA"`` are missing values (pandas would otherwise
  also turn strings such as ``"None"``, a valid ``commodity_name``, into NaN);
* ``TRUE`` / ``FALSE`` columns become logical columns (object dtype holding
  ``True`` / ``False`` / ``None``).
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from ._utils import is_na

_LOGICAL = {"TRUE": True, "FALSE": False, "True": True, "False": False, "true": True, "false": False}


def read_csv(path: str | os.PathLike, **kwargs) -> pd.DataFrame:
    """Read a GLEAM table the way ``data.table::fread`` does."""
    path = Path(path)
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = "\t" if "\t" in header else ","
    df = pd.read_csv(
        path,
        sep=sep,
        keep_default_na=False,
        na_values=["", "NA"],
        true_values=None,
        false_values=None,
        encoding="utf-8-sig",
        **kwargs,
    )
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        s = df[col]
        if s.dtype == bool:
            df[col] = s.astype(object)
        elif s.dtype.kind not in "iuf":
            non_na = [v for v in s.tolist() if not is_na(v)]
            if non_na and all((isinstance(v, (bool, np.bool_))) or (isinstance(v, str) and v in _LOGICAL) for v in non_na):
                df[col] = pd.Series(
                    [None if is_na(v) else (bool(v) if isinstance(v, (bool, np.bool_)) else _LOGICAL[v]) for v in s.tolist()],
                    dtype=object,
                    index=s.index,
                )
    return df


def example_dir(kind: str = "run_modules_examples") -> Path:
    """Directory of bundled example inputs (``system.file("extdata/<kind>", package = "gleam")``).

    ``kind`` is ``"run_modules_examples"`` or ``"run_gleam_examples"``.
    """
    if kind not in ("run_modules_examples", "run_gleam_examples"):
        raise ValueError("kind must be 'run_modules_examples' or 'run_gleam_examples'")
    return Path(str(resources.files("gleam.data").joinpath(kind)))


def example_path(name: str, kind: str = "run_modules_examples") -> Path:
    """Path of one bundled example file."""
    p = example_dir(kind) / name
    if not p.exists():
        raise FileNotFoundError(f"No bundled example named {name!r} in {kind}")
    return p


def load_example(name: str, kind: str = "run_modules_examples") -> pd.DataFrame:
    """Load one bundled example table, e.g. ``load_example("weights_input_chrt_data.csv")``."""
    return read_csv(example_path(name, kind))
