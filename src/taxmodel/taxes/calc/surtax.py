"""Scenario AGI surtax."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import capped_at, floored_at

Array = jnp.ndarray


@dataclass(frozen=True)
class SurtaxLaw:
    threshold1: Array
    rate1: Array
    threshold2: Array
    rate2: Array
    threshold3: Array
    rate3: Array


def calc_agi_surtax(records: Mapping[str, Array], law: SurtaxLaw) -> dict[str, Array]:
    """Return AGI surtax integrated across three bands."""
    # Each segment is taxed only between its two thresholds, leaving the final
    # rate to apply solely to AGI above the highest threshold.
    band1 = floored_at(
        capped_at(records["agi"], law.threshold2) - law.threshold1, 0.0
    )
    band2 = floored_at(
        capped_at(records["agi"], law.threshold3) - law.threshold2, 0.0
    )
    band3 = floored_at(records["agi"] - law.threshold3, 0.0)
    return {"liab_surtax": band1 * law.rate1 + band2 * law.rate2 + band3 * law.rate3}
