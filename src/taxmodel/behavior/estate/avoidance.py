"""Estate-tax avoidance: the share of the gross estate not reported.

This module sets one column, `estate_concealed_frac`, which the estate
calculator subtracts from the taxable base as a fraction of the reported gross
estate. Three things contribute to it, and each is zero when the module it
comes from did not run:

  1. Wealth the wealth-avoidance module concealed. Concealed wealth is not in
     the estate either. Wealth that was only valued lower stays visible to the
     estate tax, since the estate's own valuation is a separate exercise.
  2. Income the evasion module hid. The evaded share of a household's closely
     held income removes the matching share of its closely held assets from
     the reported estate.
  3. The estate tax's own effect on reporting. The reported estate has an
     elasticity of 0.16 with respect to the net-of-estate-tax rate (Kopczuk
     and Slemrod 2001), so the share of the estate still reported after a
     reform is

         ((1 - estate rate under reform) / (1 - estate rate under baseline))
             ^ 0.16

The estate rate is the rate on one more dollar of the taxable estate: each
household's own next-dollar rate through the full calculation, zero below the
exemption. The per-asset rates the wealth-tax module uses have the SOI
valuation bridge applied, and the dollars this module hides are dollars that
would otherwise appear on the estate return after that bridge.

The own-rate response uses the change in the estate rate, not its level.
Avoidance at today's rate is already in the valuation bridge, which was fitted
to reported estates in the SOI data. Where estate law does not change, the
ratio is one and this term contributes nothing.

The three contributions multiply on the still-reported share, so the hidden
share cannot exceed the whole estate.

This reduces the taxable estate only. The asset columns and what heirs inherit
are unchanged.

Two caveats on the source. The 0.16 estimate
bundles timing, avoidance, valuation, and some real change in accumulation,
and part of what it measures is charitable planning. It is an elasticity of
the reported taxable estate, applied here to the reported gross estate, which
overstates the response slightly above the exemption.

For scale: at this elasticity, raising the top estate rate from 40% to 55%
hides about 4.5% of the top bracket's reported estate.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior.wealth.avoidance import (
    CONCEALED_SHARE_CLOSELY_HELD_COL,
    CONCEALED_SHARE_MARKETABLE_COL,
)
from taxmodel.frame.steps_jax import ESTATE_ASSET_COLS

Array = jnp.ndarray
Records = Mapping[str, Array]

# The elasticity of the reported estate with respect to the net-of-estate-tax
# rate. Kopczuk and Slemrod (2001) report pooled estimates between 0.10 and
# 0.22; 0.16 is the middle of that band.
ESTATE_REPORTING_ELASTICITY = 0.16

# A marginal estate rate of exactly 100% would make the net-of-tax rate zero
# and the ratio undefined, so both rates are held just below one. Negative
# rates are held at zero.
HIGHEST_USABLE_RATE = 1.0 - 1e-6

# A rate cut makes the concealed fraction negative: estate that was not being
# reported comes back into view. The floor says a rate cut cannot more than
# double the reported base. It is a modeling limit, not an estimate.
CONCEALED_FRACTION_FLOOR = -1.0

DEFAULTS: dict[str, float] = {
    "estate_reporting_elasticity": ESTATE_REPORTING_ELASTICITY,
}

# Columns the wealth module writes, with the value used when it did not run.
# A concealed share of zero means nothing was concealed.
WEALTH_CONCEALED_SHARE_COLS: dict[str, float] = {
    CONCEALED_SHARE_MARKETABLE_COL: 0.0,
    CONCEALED_SHARE_CLOSELY_HELD_COL: 0.0,
}

# This module runs after the evasion and wealth modules, whose columns it
# reads. FAMILY_ORDER in behavior/loader.py sets the order.
MANIFEST: dict[str, object] = {
    "name": "estate_avoidance",
    # The function in this file that is the response.
    "entry_point": "estate_avoidance",
    # The reported estate scales by the elasticity applied to the change in
    # the estate rate, so a zero elasticity reports the estate unchanged.
    "zero_response": {"estate_reporting_elasticity": 0.0},
    "family": "estate",
    "reads": (
        *ESTATE_ASSET_COLS,
        "sole_prop1",
        "sole_prop2",
        "part_active",
        "part_passive",
        "scorp_active",
        "scorp_passive",
        "rent",
    ),
    # Columns the evasion and wealth modules write, if they ran. Where evasion
    # did not run each of its three factors reads as one; where wealth
    # concealment did not run each of its two shares reads as zero.
    "optional_reads": (*bs.EVASION_FACTOR_COLS, *WEALTH_CONCEALED_SHARE_COLS),
    "writes": ("estate_concealed_frac",),
    "prices": (bs.ESTATE_TAXABLE_BASE_RATE_KEY,),
    "discrete_prices": (),
    "draws": (),
}


def _capped_estate_rate(prices: Mapping[str, Array]) -> Array:
    # The tax on one more dollar of the taxable estate, measured after the SOI
    # measurement bridge. The Kopczuk-Slemrod elasticity was estimated on
    # reported estates against statutory-style rates, which this rate matches.
    rate = jnp.asarray(prices[bs.ESTATE_TAXABLE_BASE_RATE_KEY], dtype=float)
    return jnp.clip(rate, 0.0, HIGHEST_USABLE_RATE)


def concealed_fraction_parts(
    records: Records,
    prices_reform: Mapping[str, Array],
    prices_baseline: Mapping[str, Array],
    params: Mapping[str, float],
) -> dict[str, Array]:
    """The pieces the concealed fraction is assembled from, per household.

    `carried_in` is the share of the estate already out of sight because of
    wealth concealment and income evasion. `still_reported_after_own_rate` is
    the share of what remains that survives the estate tax's own reporting
    response. `concealed_fraction` is the column the estate calculator reads.
    """
    marketable = bs.total(records, bs.MARKETABLE_ASSET_COLS)
    closely_held = bs.total(records, bs.CLOSELY_HELD_ASSET_COLS)
    gross_estate = marketable + closely_held

    concealed_marketable = jnp.clip(
        jnp.asarray(records.get(CONCEALED_SHARE_MARKETABLE_COL, 0.0), dtype=float),
        0.0,
        1.0,
    )
    concealed_closely_held = jnp.clip(
        jnp.asarray(records.get(CONCEALED_SHARE_CLOSELY_HELD_COL, 0.0), dtype=float),
        0.0,
        1.0,
    )
    evaded = bs.evaded_closely_held_income_share(records)

    # Concealment and evasion can hide the same closely held asset. Wealth is
    # concealed first, and evasion hides a share of what is still visible, so
    # the overlap is not counted twice.
    closely_held_out_of_sight = (
        concealed_closely_held + (1.0 - concealed_closely_held) * evaded
    )
    carried_in = bs.divide_where_positive(
        concealed_marketable * marketable + closely_held_out_of_sight * closely_held,
        gross_estate,
    )

    rate_reform = _capped_estate_rate(prices_reform)
    rate_baseline = _capped_estate_rate(prices_baseline)
    still_reported = ((1.0 - rate_reform) / (1.0 - rate_baseline)) ** params[
        "estate_reporting_elasticity"
    ]

    has_estate = gross_estate > 0.0
    # A household with no estate conceals nothing. The rate on the taxable
    # base exists for every household, so that case is masked out here.
    concealed_fraction = jnp.where(
        has_estate,
        jnp.maximum(
            1.0 - (1.0 - carried_in) * still_reported, CONCEALED_FRACTION_FLOOR
        ),
        0.0,
    )
    return {
        "gross_estate": gross_estate,
        "evaded_income_share": evaded,
        "closely_held_out_of_sight_share": closely_held_out_of_sight,
        "carried_in": carried_in,
        "estate_rate_reform": rate_reform,
        "estate_rate_baseline": rate_baseline,
        "still_reported_after_own_rate": still_reported,
        "concealed_fraction": concealed_fraction,
    }


def estate_avoidance(
    records: Records,
    prices_reform: Mapping[str, Array],
    prices_baseline: Mapping[str, Array],
    params: Mapping[str, float],
    draws: Mapping[str, Array] | None = None,
) -> dict[str, Array]:
    """Set the share of the reported gross estate hidden from the estate tax.

    Returns one changed column, `estate_concealed_frac`. The estate calculator
    multiplies it by the reported gross estate and takes that out of the
    taxable base. What heirs receive is unchanged.

    Runs after the income-evasion and wealth-avoidance modules, whose columns
    it reads. Where neither ran and estate law does not change, it returns
    zero for every household. `draws` is unused.
    """
    del draws
    parts = concealed_fraction_parts(records, prices_reform, prices_baseline, params)
    return {"estate_concealed_frac": parts["concealed_fraction"]}
