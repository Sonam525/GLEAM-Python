"""Enteric methane emissions (port of ``R/core_model_emissions_enteric.R``).

Both functions are vectorised: they accept scalars, lists, numpy arrays or
pandas Series (broadcast against each other) and reproduce, element by
element, the scalar species / cohort branches of the R code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import constants as K
from .._utils import all_scalar, as_float, as_str, broadcast, finalize, isin
from ..validation._shared import abort
from ..validation.emissions_enteric_core import (
    validate_enteric_emission_inputs,
    validate_ym_inputs,
)

_LARGE_RUMINANTS = ("CTL", "BFL")
_SMALL_RUMINANTS_CAMELS = ("SHP", "GTS", "CML")
_JUVENILE = ("FJ", "MJ")
_SUBADULT_NONDEMO = ("FS", "MS", "FN", "MN")


def calc_conversion_factor_ym(
    species_short: Any,
    cohort_short: Any,
    ration_digestibility_fraction: Any,
) -> Any:
    """Methane conversion factor ym (% of gross energy intake converted to CH4).

    Species- and cohort-specific default relationships with diet digestibility
    (Opio et al., 2013; IPCC 2006/2019, Eq. 10.21), with
    ``DE = ration_digestibility_fraction * 100``:

    * ``CTL``, ``BFL``: ``ym = 9.75 - 0.05 * DE``;
    * ``SHP``, ``GTS``, ``CML``: ``FA``/``MA``: ``9.75 - 0.05 * DE``;
      ``FS``/``MS``/``FN``/``MN``: ``7.75 - 0.05 * DE``;
    * ``PGS``: ``FA``/``MA``: ``1.01``; ``FS``/``MS``/``FN``/``MN``: ``0.39``
      (Jorgensen et al., 2011);
    * ``CHK``: ``0``.

    Juvenile cohorts (``FJ``, ``MJ``) get ``ym = 0`` (negligible enteric
    fermentation before weaning).

    Parameters
    ----------
    species_short : str or array-like
        Species code (``CTL``, ``BFL``, ``SHP``, ``GTS``, ``PGS``, ``CML``, ``CHK``).
    cohort_short : str or array-like
        Cohort code (``FJ``, ``FS``, ``FA``, ``MJ``, ``MS``, ``MA``, ``FN``, ``MN``).
    ration_digestibility_fraction : float or array-like
        Ration digestibility, digestible (metabolizable for poultry) over gross
        energy (fraction).

    Returns
    -------
    float or numpy.ndarray
        Methane conversion factor ym (percentage of gross energy).
    """
    validate_ym_inputs(species_short, cohort_short, ration_digestibility_fraction)

    sp, co, dig = _vec(
        as_str(species_short), as_str(cohort_short), as_float(ration_digestibility_fraction)
    )

    large = isin(sp, _LARGE_RUMINANTS)
    small = isin(sp, _SMALL_RUMINANTS_CAMELS)
    pgs = sp == "PGS"
    chk = sp == "CHK"
    juvenile = isin(co, _JUVENILE)
    subadult = isin(co, _SUBADULT_NONDEMO)

    # R leaves the result undefined (and errors) for any other species.
    _abort_unknown_species(sp, large | small | pgs | chk)

    with np.errstate(invalid="ignore", over="ignore"):
        ym_975 = 9.75 - 0.05 * dig * 100
        ym_775 = 7.75 - 0.05 * dig * 100

    out = np.select(
        [
            large & juvenile, large,
            small & juvenile, small & subadult, small,
            pgs & juvenile, pgs & subadult, pgs,
            chk,
        ],
        [
            0.0, ym_975,
            0.0, ym_775, ym_975,
            0.0, 0.39, 1.01,
            0.0,
        ],
        default=np.nan,
    )
    return finalize(out, all_scalar(species_short, cohort_short, ration_digestibility_fraction))


def calc_ch4_enteric(
    species_short: Any,
    ch4_conversion_factor_ym: Any,
    ch4_mitigation_factor: Any,
    ration_gross_energy: Any,
    ration_intake: Any,
) -> Any:
    """Daily enteric methane emissions (kg CH4/head/day).

    IPCC (2006/2019) Tier 2, Eq. 10.21::

        ch4_enteric = ration_gross_energy * ration_intake
                      * (ch4_conversion_factor_ym / 100) * ch4_mitigation_factor / 55.65

    where 55.65 MJ/kg is the energy content of methane.

    Parameters
    ----------
    species_short : str or array-like
        Species code (only validated).
    ch4_conversion_factor_ym : float or array-like
        Methane conversion factor ym (% of gross energy), see
        :func:`calc_conversion_factor_ym`.
    ch4_mitigation_factor : float or array-like
        Multiplicative mitigation factor on baseline enteric CH4
        (dimensionless; 1 = no mitigation, 0.9 = 10 % reduction).
    ration_gross_energy : float or array-like
        Average gross energy content of the ration (MJ/kg DM).
    ration_intake : float or array-like
        Average daily dry matter intake (kg DM/head/day).

    Returns
    -------
    float or numpy.ndarray
        Enteric methane emissions (kg CH4/head/day).
    """
    validate_enteric_emission_inputs(
        species_short, ch4_conversion_factor_ym, ch4_mitigation_factor,
        ration_gross_energy, ration_intake,
    )

    _, ym, mitigation, ge, dmi = _vec(
        as_str(species_short), as_float(ch4_conversion_factor_ym), as_float(ch4_mitigation_factor),
        as_float(ration_gross_energy), as_float(ration_intake),
    )
    with np.errstate(invalid="ignore", over="ignore"):
        ch4_enteric = ge * dmi * (ym / 100) * mitigation / 55.65

    return finalize(
        ch4_enteric,
        all_scalar(species_short, ch4_conversion_factor_ym, ch4_mitigation_factor,
                   ration_gross_energy, ration_intake),
    )


def _vec(*xs: Any) -> list[np.ndarray]:
    """Broadcast the converted inputs against each other as 1-d arrays."""
    return [np.atleast_1d(a) for a in broadcast(*xs)]


def _abort_unknown_species(sp: np.ndarray, known: np.ndarray) -> None:
    """Error for species outside every R branch (R fails with "object not found")."""
    if not np.all(known):
        bad = sp[~known].ravel()[0]
        abort(
            f"`species_short` must be one of: {', '.join(K.GLEAM_SPECIES)} "
            f"(got {bad!r})."
        )
