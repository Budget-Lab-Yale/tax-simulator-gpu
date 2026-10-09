"""Treatment of a decedent's unrealized gain, averaged over an age cell.

Tax law can do one of three things to the gain on an asset its owner dies
holding, and today's law and every reform in the corpus pick one of the three
for each asset class:

**Step-up in basis.** The heir takes the asset at its value on the date of
death, so the gain the decedent built up is never taxed by anyone.  This is
current law for every class, and it is how gains leave the stock without ever
being taxed.

**Carryover basis.** The heir takes the decedent's own basis, so the gain
survives the death and is taxed when the heir eventually sells.  Nothing is
collected in the year of death; the gain moves into the heirs' age cells and is
realized from there in later years.

**Deemed realization at death.** The gain is treated as realized on the final
return and taxed there.  It is collected in the year of death and nothing moves
to the heir.

The model works on age cells, not on assets, so the three treatments have to
be averaged within a cell: a cell whose gains are three-quarters in equities
under carryover and one-quarter in a primary home under step-up sends
three-quarters of its dying dollars to heirs and forgives one-quarter.  This
module does that averaging, weighting each asset class by its share of the
cell's gain.

The forgiven share is measured against the cell's *whole* gain, while the
routed and taxed shares are measured against the gain the two death-gain
exclusions leave.  An exclusion is modeled as a step-up in basis up to its cap,
so the excluded dollars are forgiven whichever regime the class is under.
They count in the forgiven share and are not also counted as routed to an heir
or taxed at death, so the three shares add to one for every cell that holds
gain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .applier import DEEMED_ASSET_CLASSES

Array = jnp.ndarray

# What tax law does to the gain on an asset its owner dies holding, by the code
# the law parameter `pref.kg_death_regime_{class}` carries.
STEP_UP = 0
CARRYOVER_BASIS = 1
DEEMED_REALIZATION = 2

# For each code, what becomes of a dollar of the decedent's gain: forgiven,
# carried over to the heir, or realized and taxed on the final return.  Exactly
# one of the three is a whole dollar and the other two are nothing, so the cell
# average is a weighted sum of shares.
_WHAT_BECOMES_OF_A_DOLLAR: dict[int, tuple[float, float, float]] = {
    STEP_UP: (1.0, 0.0, 0.0),
    CARRYOVER_BASIS: (0.0, 1.0, 0.0),
    DEEMED_REALIZATION: (0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class DeathShares:
    """How a cell's dying gain splits three ways.

    Every field is shaped [age cell, year].  The three add to one for every
    cell that holds gain; the dollars the exclusions take are counted as
    forgiven along with the stepped-up ones.
    """

    forgiven: Array
    routed_to_heirs: Array
    taxed_at_death: Array


def what_becomes_of_a_dollar(
    regime_code_by_class: Sequence[int],
) -> np.ndarray:
    """For each asset class, the three shares its death regime implies.

    One row per class in :data:`~.applier.DEEMED_ASSET_CLASSES` order, and
    three columns: forgiven, carried over to the heir, realized and taxed on
    the final return. Exactly one of the three is one and the other two zero,
    so the cell average is a weighted sum.

    The codes are turned into weights on the host, so the arithmetic below
    reads weights.  Many scenarios' death regimes can then enter one compiled
    program as data.
    """

    rows = []
    for position, code in enumerate(regime_code_by_class):
        becomes = _WHAT_BECOMES_OF_A_DOLLAR.get(int(code))
        if becomes is None:
            raise ValueError(
                f"the year's law puts {DEEMED_ASSET_CLASSES[position]} under "
                f"death regime {code}, and there are three: {STEP_UP} for a "
                f"stepped-up basis, {CARRYOVER_BASIS} for carryover basis, "
                f"{DEEMED_REALIZATION} for deemed realization at death"
            )
        rows.append(becomes)
    return np.asarray(rows, dtype=np.float64)


def death_shares_from_regime_weights(
    whole: Array,
    at_stake: Array,
    becomes_by_class: Array,
) -> tuple[Array, Array, Array]:
    """Average the treatment of gains at death over each age cell.

    ``whole`` and ``at_stake`` are shaped [..., asset class] in
    :data:`~.applier.DEEMED_ASSET_CLASSES` order: the cell's whole gain in each
    class, and the same gain after the section 121 and per-decedent
    exclusions. ``becomes_by_class`` is what :func:`what_becomes_of_a_dollar`
    returns: [asset class, 3], the shares forgiven, routed to heirs and taxed
    at death.
    """

    cell_gain = jnp.sum(whole, axis=-1)
    holds_something = cell_gain > 0.0
    safe_gain = jnp.where(holds_something, cell_gain, 1.0)

    def share_of_the_cell(amount: Array) -> Array:
        return jnp.where(holds_something, amount / safe_gain, 0.0)

    forgiven = jnp.zeros_like(cell_gain)
    routed = jnp.zeros_like(cell_gain)
    taxed = jnp.zeros_like(cell_gain)

    for position in range(len(DEEMED_ASSET_CLASSES)):
        is_forgiven = becomes_by_class[position, 0]
        goes_to_heirs = becomes_by_class[position, 1]
        is_taxed = becomes_by_class[position, 2]

        # A step-up forgives the whole gain. Under the other two regimes the
        # exclusions forgive the part they take, and only the gain at stake
        # is routed or taxed.
        whole_share = share_of_the_cell(whole[..., position])
        at_stake_share = share_of_the_cell(at_stake[..., position])
        excluded_share = whole_share - at_stake_share

        forgiven = (
            forgiven
            + is_forgiven * whole_share
            + (goes_to_heirs + is_taxed) * excluded_share
        )
        routed = routed + goes_to_heirs * at_stake_share
        taxed = taxed + is_taxed * at_stake_share

    return forgiven, routed, taxed


def realized_at_death_share_by_class(
    regime_code_by_class: Sequence[int],
) -> tuple[float, ...]:
    """The share deemed realized at death, one number per asset class.

    The household applier needs this rather than the cell average, because the
    gain it puts on a decedent's final return is that household's own gain in
    each class.
    """

    return tuple(
        float(taxed_at_death)
        for _forgiven, _to_heirs, taxed_at_death in what_becomes_of_a_dollar(
            regime_code_by_class
        )
    )
