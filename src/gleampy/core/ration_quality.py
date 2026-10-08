"""Ration (diet) quality (port of ``R/core_model_ration_quality.R``).

Each function computes the contribution of one feed component to a
diet-level nutritional metric; the run module sums the contributions over the
feed components of a cohort's ration. All functions are vectorised: they
accept scalars, lists, numpy arrays or pandas Series (broadcast against each
other) and reproduce, element by element, the scalar species branches of the
R code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import constants as K
from .._utils import all_scalar, as_float, broadcast, finalize, finalize_dict
from ..validation.ration_quality_core import (
    _as_codes,
    _codes_in,
    validate_diet_digestibility_inputs,
    validate_feed_digestibility_inputs,
    validate_ration_ash_inputs,
    validate_ration_gross_energy_inputs,
    validate_ration_metabolizable_energy_inputs,
    validate_ration_nitrogen_inputs,
    validate_urinary_energy_inputs,
)


def _by_species(species_short: Any, feed_ration_fraction: Any, ruminant: Any, pigs: Any) -> Any:
    """``feed_ration_fraction * <ruminant or pig parameter>`` by species.

    R: ``if (species_short %in% gleam_species_milk_producers)`` use the
    ruminant parameter, ``else`` the pig parameter (also for ``CHK``).
    """
    scalar = all_scalar(species_short, feed_ration_fraction, ruminant, pigs)
    sp, frac, rum, pig = broadcast(
        _as_codes(species_short), as_float(feed_ration_fraction), as_float(ruminant), as_float(pigs)
    )
    with np.errstate(invalid="ignore", over="ignore"):
        out = np.where(_codes_in(sp, K.GLEAM_SPECIES_MILK_PRODUCERS), frac * rum, frac * pig)
    return finalize(out, scalar)


def calc_feed_digestibility_fraction(
    feed_digestible_energy_ruminant: Any,
    feed_digestible_energy_pigs: Any,
    feed_gross_energy: Any,
) -> dict[str, Any]:
    """Digestibility of a feed component for ruminants and pigs (fraction).

    ``feed_digestibility_fraction = digestible_energy / feed_gross_energy``;
    a missing digestible energy gives a digestibility of 0.

    Parameters
    ----------
    feed_digestible_energy_ruminant : float or array-like
        Digestible energy of the feed for ruminants (MJ/kg DM); may be NA.
    feed_digestible_energy_pigs : float or array-like
        Digestible energy of the feed for pigs (MJ/kg DM); may be NA.
    feed_gross_energy : float or array-like
        Gross energy of the feed (MJ/kg DM).

    Returns
    -------
    dict
        ``feed_digestibility_fraction_ruminant`` and
        ``feed_digestibility_fraction_pigs`` (fraction).
    """
    validate_feed_digestibility_inputs(
        feed_digestible_energy_ruminant,
        feed_digestible_energy_pigs,
        feed_gross_energy,
    )
    scalar = all_scalar(feed_digestible_energy_ruminant, feed_digestible_energy_pigs, feed_gross_energy)
    de_rum, de_pig, ge = broadcast(
        as_float(feed_digestible_energy_ruminant),
        as_float(feed_digestible_energy_pigs),
        as_float(feed_gross_energy),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rum = np.where(np.isnan(de_rum), 0.0, de_rum / ge)
        pig = np.where(np.isnan(de_pig), 0.0, de_pig / ge)
    return finalize_dict(
        {
            "feed_digestibility_fraction_ruminant": rum,
            "feed_digestibility_fraction_pigs": pig,
        },
        scalar,
    )


def calc_ration_digestibility(
    species_short: Any,
    feed_ration_fraction: Any,
    feed_digestibility_fraction_ruminant: Any = np.nan,
    feed_digestibility_fraction_pigs: Any = np.nan,
) -> Any:
    """Contribution of a feed component to diet digestibility (fraction).

    * ruminants and camels (``CTL``, ``BFL``, ``SHP``, ``GTS``, ``CML``):
      ``feed_ration_fraction * feed_digestibility_fraction_ruminant``;
    * other species (``PGS``, and ``CHK``):
      ``feed_ration_fraction * feed_digestibility_fraction_pigs``.

    Parameters
    ----------
    species_short : str or array-like
        Species code.
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    feed_digestibility_fraction_ruminant, feed_digestibility_fraction_pigs : float or array-like
        Feed digestibility for ruminants / pigs (fraction); the one not used
        for the species may be NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to diet digestibility (fraction).
    """
    validate_diet_digestibility_inputs(
        species_short,
        feed_ration_fraction,
        feed_digestibility_fraction_ruminant,
        feed_digestibility_fraction_pigs,
    )
    return _by_species(
        species_short, feed_ration_fraction,
        feed_digestibility_fraction_ruminant, feed_digestibility_fraction_pigs,
    )


def calc_ration_metabolizable_energy(
    species_short: Any,
    feed_ration_fraction: Any,
    feed_metabolizable_energy_ruminant: Any = np.nan,
    feed_metabolizable_energy_pigs: Any = np.nan,
) -> Any:
    """Contribution of a feed component to diet metabolizable energy (MJ/kg DM).

    * ruminants and camels: ``feed_ration_fraction * feed_metabolizable_energy_ruminant``;
    * other species (``PGS``, and ``CHK``):
      ``feed_ration_fraction * feed_metabolizable_energy_pigs``.

    Parameters
    ----------
    species_short : str or array-like
        Species code.
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    feed_metabolizable_energy_ruminant, feed_metabolizable_energy_pigs : float or array-like
        Feed metabolizable energy for ruminants / pigs (MJ/kg DM); the one not
        used for the species may be NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to diet metabolizable energy (MJ/kg DM).
    """
    validate_ration_metabolizable_energy_inputs(
        species_short,
        feed_ration_fraction,
        feed_metabolizable_energy_ruminant,
        feed_metabolizable_energy_pigs,
    )
    return _by_species(
        species_short, feed_ration_fraction,
        feed_metabolizable_energy_ruminant, feed_metabolizable_energy_pigs,
    )


def calc_ration_gross_energy(feed_ration_fraction: Any, feed_gross_energy: Any) -> Any:
    """Contribution of a feed component to diet gross energy (MJ/kg DM).

    ``ration_gross_energy = feed_ration_fraction * feed_gross_energy``

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    feed_gross_energy : float or array-like
        Gross energy of the feed (MJ/kg DM).

    Returns
    -------
    float or numpy.ndarray
        Contribution to diet gross energy (MJ/kg DM).
    """
    validate_ration_gross_energy_inputs(feed_ration_fraction, feed_gross_energy)
    scalar = all_scalar(feed_ration_fraction, feed_gross_energy)
    with np.errstate(invalid="ignore", over="ignore"):
        out = as_float(feed_ration_fraction) * as_float(feed_gross_energy)
    return finalize(out, scalar)


def calc_ration_nitrogen_content(feed_ration_fraction: Any, feed_nitrogen_content: Any) -> Any:
    """Contribution of a feed component to diet nitrogen content (kg N/kg DM).

    ``ration_nitrogen = feed_ration_fraction * feed_nitrogen_content``

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    feed_nitrogen_content : float or array-like
        Nitrogen content of the feed (kg N/kg DM).

    Returns
    -------
    float or numpy.ndarray
        Contribution to diet nitrogen content (kg N/kg DM).
    """
    validate_ration_nitrogen_inputs(feed_ration_fraction, feed_nitrogen_content)
    scalar = all_scalar(feed_ration_fraction, feed_nitrogen_content)
    with np.errstate(invalid="ignore", over="ignore"):
        out = as_float(feed_ration_fraction) * as_float(feed_nitrogen_content)
    return finalize(out, scalar)


def calc_ration_urinary_energy_fraction(
    species_short: Any,
    feed_ration_fraction: Any,
    feed_urinary_energy_ruminant: Any = np.nan,
    feed_urinary_energy_pigs: Any = np.nan,
) -> Any:
    """Contribution of a feed component to the urinary energy fraction (fraction).

    * ruminants and camels: ``feed_ration_fraction * feed_urinary_energy_ruminant``;
    * other species (``PGS``, and ``CHK``):
      ``feed_ration_fraction * feed_urinary_energy_pigs``.

    Parameters
    ----------
    species_short : str or array-like
        Species code.
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    feed_urinary_energy_ruminant, feed_urinary_energy_pigs : float or array-like
        Fraction of feed gross energy excreted in urine by ruminants / pigs;
        the one not used for the species may be NA.

    Returns
    -------
    float or numpy.ndarray
        Contribution to the fraction of diet gross energy lost in urine.
    """
    validate_urinary_energy_inputs(
        species_short,
        feed_ration_fraction,
        feed_urinary_energy_ruminant,
        feed_urinary_energy_pigs,
    )
    return _by_species(
        species_short, feed_ration_fraction,
        feed_urinary_energy_ruminant, feed_urinary_energy_pigs,
    )


def calc_ration_ash(feed_ration_fraction: Any, feed_ash: Any) -> Any:
    """Contribution of a feed component to diet ash content (kg ash/kg DM).

    ``ration_ash = feed_ration_fraction * feed_ash / 100`` (``feed_ash`` in
    g ash/100 g DM).

    Parameters
    ----------
    feed_ration_fraction : float or array-like
        Share of the feed component in ration dry matter (fraction).
    feed_ash : float or array-like
        Ash content of the feed (g ash/100 g DM).

    Returns
    -------
    float or numpy.ndarray
        Contribution to diet ash content (kg ash/kg DM).
    """
    validate_ration_ash_inputs(feed_ration_fraction, feed_ash)
    scalar = all_scalar(feed_ration_fraction, feed_ash)
    with np.errstate(invalid="ignore", over="ignore"):
        out = as_float(feed_ration_fraction) * as_float(feed_ash) / 100
    return finalize(out, scalar)
