"""Input validation for the ration quality core model.

Port of ``R/validate_ration_quality_core_model.R``. The R validators are
called row by row on scalars; these versions check whole vectors at once
(element ``i`` is checked exactly as R checks row ``i``) and are no-ops when
validation is disabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .. import constants as K
from .._utils import all_scalar, as_float, broadcast, is_na, is_scalar
from ._shared import (
    abort,
    validate_animal_species,
    validate_param_range,
    validate_scalar_character,
    validate_scalar_numeric,
    validator,
)

# --------------------------------------------------------------------------
# Private helpers (vectorised counterparts of R scalar checks)
# --------------------------------------------------------------------------


def _raw_values(x: Any) -> np.ndarray:
    """Raw elements of ``x`` as a 1-d object array (no type coercion)."""
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy(dtype=object)
    return np.atleast_1d(np.asarray(x, dtype=object)).ravel()


def _as_codes(x: Any) -> np.ndarray:
    """Code vector (species) as an object array with ``None`` for NA.

    Vectorised equivalent of :func:`gleampy._utils.as_str` for validated code
    vectors (``as_str`` loops in Python, which dominates run time on large
    tables); values are not converted to ``str``.
    """
    if x is None:
        return np.array(None, dtype=object)
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy(dtype=object)
    else:
        arr = np.asarray(x, dtype=object)
    if arr.ndim == 0:
        return np.array(None if is_na(arr.item()) else arr.item(), dtype=object)
    na = pd.isna(arr)
    if na.any():
        arr = arr.copy()
        arr[na] = None
    return arr


def _codes_in(codes: np.ndarray, values: Sequence[str]) -> np.ndarray:
    """R ``codes %in% values`` for an :func:`_as_codes` array (``NA`` -> ``FALSE``)."""
    arr = np.asarray(codes, dtype=object)
    if arr.ndim == 0:
        v = arr.item()
        return np.array(v is not None and v in set(values))
    return pd.Series(arr.ravel(), dtype=object).isin(list(values)).to_numpy().reshape(arr.shape)


def _unique_values(x: Any) -> Any:
    """Distinct raw values of a vector (scalars unchanged).

    The shared element-wise checks (``validate_scalar_character``,
    ``validate_animal_species``) loop in Python and their messages do not
    depend on the position of the offending value, so running them on the
    distinct values is equivalent and much faster.
    """
    if is_scalar(x):
        return x
    return pd.unique(_raw_values(x))


def _check_lengths(args: dict[str, Any], char_args: Sequence[str] = ()) -> None:
    """Vectorised replacement of R's ``length(x) != 1`` checks.

    Every argument must be a scalar or a 1-d vector with the common length of
    the other vector arguments.
    """
    sizes = {}
    for name, v in args.items():
        if is_scalar(v):
            sizes[name] = 1
        elif np.ndim(v) == 1:
            sizes[name] = int(np.size(v))
        else:
            sizes[name] = -1
    n = max(sizes.values()) if sizes else 1
    for name, s in sizes.items():
        if s == -1 or (s != 1 and s != n):
            kind = "character value" if name in char_args else "numeric (scalar)"
            abort(
                f"`{name}` must be a single {kind} or a vector with the same length "
                f"as the other inputs."
            )


def _is_numeric_or_na(x: Any) -> bool:
    """R ``is.numeric(x)`` for a value or a column, with missing numbers allowed.

    A scalar ``None`` / ``nan`` counts as a missing number. A column (pandas
    Series or array) of object dtype whose values are all missing does not:
    it is what :func:`gleampy.io.read_csv` returns for an all-empty column,
    which ``fread`` types as logical, and R's ``is.numeric()`` is ``FALSE``
    for a logical ``NA`` (R rejects such a column with "must be numeric" /
    "must be a single numeric (scalar)").
    """
    if x is None:
        return True
    if isinstance(x, (pd.Series, pd.Index)):
        if x.dtype.kind in "iuf":
            return True
        is_column = True
    else:
        if not isinstance(x, (str, bytes)) and np.asarray(x).dtype.kind in "iuf":
            return True
        is_column = np.ndim(x) > 0
    values = _raw_values(x)
    present = 0
    for v in values:
        if is_na(v):
            continue
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)):
            return False
        present += 1
    return present > 0 or not is_column or values.size == 0


def _validate_range_where(values: np.ndarray, arg_name: str, mask: np.ndarray, scalar: bool) -> None:
    """``validate_param_range`` on the elements selected by ``mask``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    if scalar:
        validate_param_range(values[idx], arg_name)
    else:
        validate_param_range(values[idx], arg_name, labels=idx + 1)


def _validate_species_specific(
    label: str,
    species_short: Any,
    feed_ration_fraction: Any,
    ruminant_name: str,
    ruminant: Any,
    pigs_name: str,
    pigs: Any,
    species_first: bool,
) -> None:
    """Shared body of the digestibility / ME / urinary-energy validators.

    Ruminant and pig parameters must be numeric (NA allowed) and within range
    when given; the one used for the species (ruminant parameter for
    ``gleam_species_milk_producers``, pig parameter otherwise) must be
    non-missing.
    """
    _check_lengths(
        {
            "species_short": species_short,
            "feed_ration_fraction": feed_ration_fraction,
            ruminant_name: ruminant,
            pigs_name: pigs,
        },
        char_args=("species_short",),
    )
    species_values = _unique_values(species_short)
    if species_first:
        validate_animal_species(species_values)
    else:
        validate_scalar_character(species_values, "species_short")
    validate_scalar_numeric(feed_ration_fraction, "feed_ration_fraction")
    validate_param_range(feed_ration_fraction, "feed_ration_fraction")

    args = {ruminant_name: ruminant, pigs_name: pigs}
    for name, val in args.items():
        if not _is_numeric_or_na(val):
            abort(f"`{name}` must be a single numeric (scalar). NA is allowed.")
    scalar = all_scalar(species_short, ruminant, pigs)
    sp, rum, pig = (
        np.atleast_1d(a) for a in broadcast(_as_codes(species_short), as_float(ruminant), as_float(pigs))
    )
    _validate_range_where(rum, ruminant_name, ~np.isnan(rum), scalar)
    _validate_range_where(pig, pigs_name, ~np.isnan(pig), scalar)

    if not species_first:
        validate_animal_species(species_values)

    uses_ruminant = _codes_in(sp, K.GLEAM_SPECIES_MILK_PRODUCERS)
    missing = np.where(uses_ruminant, np.isnan(rum), np.isnan(pig))
    if missing.any():
        i = int(np.flatnonzero(missing)[0])
        name = ruminant_name if uses_ruminant[i] else pigs_name
        abort(f'Missing required {label} inputs for species_short "{sp[i]}": "{name}"')


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------


@validator
def validate_diet_digestibility_inputs(
    species_short: Any,
    feed_ration_fraction: Any,
    feed_digestibility_fraction_ruminant: Any,
    feed_digestibility_fraction_pigs: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.calc_ration_digestibility`."""
    _validate_species_specific(
        "digestibility",
        species_short,
        feed_ration_fraction,
        "feed_digestibility_fraction_ruminant",
        feed_digestibility_fraction_ruminant,
        "feed_digestibility_fraction_pigs",
        feed_digestibility_fraction_pigs,
        species_first=False,
    )


@validator
def validate_ration_metabolizable_energy_inputs(
    species_short: Any,
    feed_ration_fraction: Any,
    feed_metabolizable_energy_ruminant: Any,
    feed_metabolizable_energy_pigs: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.calc_ration_metabolizable_energy`."""
    _validate_species_specific(
        "metabolizable energy",
        species_short,
        feed_ration_fraction,
        "feed_metabolizable_energy_ruminant",
        feed_metabolizable_energy_ruminant,
        "feed_metabolizable_energy_pigs",
        feed_metabolizable_energy_pigs,
        species_first=True,
    )


@validator
def validate_feed_digestibility_inputs(
    feed_digestible_energy_ruminant: Any,
    feed_digestible_energy_pigs: Any,
    feed_gross_energy: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.calc_feed_digestibility_fraction`.

    All arguments must be numeric (digestible energies may be NA);
    ``feed_gross_energy`` must be non-missing and within its bounds.
    """
    args = {
        "feed_digestible_energy_ruminant": feed_digestible_energy_ruminant,
        "feed_digestible_energy_pigs": feed_digestible_energy_pigs,
        "feed_gross_energy": feed_gross_energy,
    }
    _check_lengths(args)
    for name, val in args.items():
        if not _is_numeric_or_na(val):
            abort(f"`{name}` must be numeric.")
    if np.isnan(as_float(feed_gross_energy)).any():
        abort("`feed_gross_energy` must not contain missing values.")

    validate_param_range(as_float(feed_gross_energy), "feed_gross_energy")


@validator
def validate_ration_gross_energy_inputs(feed_ration_fraction: Any, feed_gross_energy: Any) -> None:
    """Validate the inputs of :func:`gleampy.calc_ration_gross_energy`."""
    _check_lengths({"feed_ration_fraction": feed_ration_fraction, "feed_gross_energy": feed_gross_energy})
    validate_scalar_numeric(feed_ration_fraction, "feed_ration_fraction")
    validate_scalar_numeric(feed_gross_energy, "feed_gross_energy")

    validate_param_range(feed_ration_fraction, "feed_ration_fraction")
    validate_param_range(feed_gross_energy, "feed_gross_energy")


@validator
def validate_ration_nitrogen_inputs(feed_ration_fraction: Any, feed_nitrogen_content: Any) -> None:
    """Validate the inputs of :func:`gleampy.calc_ration_nitrogen_content`."""
    _check_lengths(
        {"feed_ration_fraction": feed_ration_fraction, "feed_nitrogen_content": feed_nitrogen_content}
    )
    validate_scalar_numeric(feed_ration_fraction, "feed_ration_fraction")
    validate_scalar_numeric(feed_nitrogen_content, "feed_nitrogen_content")

    validate_param_range(feed_ration_fraction, "feed_ration_fraction")
    validate_param_range(feed_nitrogen_content, "feed_nitrogen_content")


@validator
def validate_urinary_energy_inputs(
    species_short: Any,
    feed_ration_fraction: Any,
    feed_urinary_energy_ruminant: Any,
    feed_urinary_energy_pigs: Any,
) -> None:
    """Validate the inputs of :func:`gleampy.calc_ration_urinary_energy_fraction`."""
    _validate_species_specific(
        "urinary energy",
        species_short,
        feed_ration_fraction,
        "feed_urinary_energy_ruminant",
        feed_urinary_energy_ruminant,
        "feed_urinary_energy_pigs",
        feed_urinary_energy_pigs,
        species_first=False,
    )


@validator
def validate_ration_ash_inputs(feed_ration_fraction: Any, feed_ash: Any) -> None:
    """Validate the inputs of :func:`gleampy.calc_ration_ash`."""
    _check_lengths({"feed_ration_fraction": feed_ration_fraction, "feed_ash": feed_ash})
    validate_scalar_numeric(feed_ration_fraction, "feed_ration_fraction")
    validate_scalar_numeric(feed_ash, "feed_ash")

    validate_param_range(feed_ration_fraction, "feed_ration_fraction")
    validate_param_range(feed_ash, "feed_ash")
