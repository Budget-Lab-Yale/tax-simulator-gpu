"""Income tax noncompliance, on the DeBacker, Heim and Yuskavage elasticities.

Some income is reported only because reporting it is cheaper than hiding it.
When the marginal rate on a dollar rises, the net-of-tax return to reporting
that dollar falls, and some of it stops being reported. This module applies
that response to the income the Internal Revenue Service cannot see through a
third-party information return: sole proprietorship and farm income,
pass-through and S corporation income, and rent.

Wages, interest and dividends do not respond here, because a third party
reports them and they are already visible. The authors find audit adjustments
on roughly 7 percent of wages against roughly 74 percent of sole proprietor
income, and the tax gap literature puts wage misreporting near 1 percent.

Two margins are left out: overstated losses and deductions, where the same
authors find a response but only positive income legs move here, and recovery
through enforcement, which the model does not represent.

This module represents income leaving the tax system. Income moving between
tax bases is the conversion and entity shifting modules. Running this module
alongside a general elasticity of taxable income would count noncompliance
twice.

The three factor columns `evasion_g_schc`, `evasion_g_pt` and `evasion_g_rent`
are part of this module's contract. The wealth-avoidance and estate-avoidance
modules read them, because a household that hides the income from an asset
does not report the asset either.

The Schedule C and farm responses are conditioned on the household total,
which is the sum of the two earners' columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from taxmodel.behavior import balance_sheet as bs

Array = jnp.ndarray

# The elasticities of reported income with respect to the net-of-tax rate,
# from DeBacker, Heim and Yuskavage, "Marginal Tax Rates and Income Tax
# Noncompliance" (National Tax Association meetings, November 2025), estimated
# on random audit data from 2006 to 2017. The findings are the authors' own
# and not the Treasury's. They treat noncompliance as one component of the
# elasticity of taxable income, alongside avoidance, and that is how the
# values are used here.
#
# The three groups are the three levels of visibility to the IRS:
#
#   sole_prop_and_farm  Schedule C and Schedule F. The authors' pooled
#                       cross-section, sole proprietor subsample, federal
#                       marginal rate. Alternative anchors, if a band rather
#                       than a central value is wanted: 0.09 (a Kansas
#                       difference-in-differences on Schedule C) and 0.26
#                       (bunching at the earned income credit's kinks, a
#                       low-income population).
#   pass_through        Partnership and S corporation income. Same pooled
#                       cross-section, partnership subsample.
#   rent                Schedule E excluding pass-through income. The Kansas
#                       difference-in-differences. Not statistically
#                       significant, and the weakest-identified value here.
#
# The top-end multiplier scales all three, for evasion that random audits do
# not detect: they miss sophisticated arrangements at the top, such as
# offshore structures and tiered partnerships (Guyton, Langetieg, Reck, Risch
# and Zucman 2021), so an elasticity measured from detected noncompliance is a
# floor there. At 1 there is no adjustment; a high variant sets 1.5 to 2. None
# of these is a calibrated value.
DEFAULTS: dict[str, float] = {
    "elasticity_sole_prop_and_farm": 0.046,
    "elasticity_pass_through": 0.052,
    "elasticity_rent": 0.040,
    "top_end_multiplier": 1.0,
    # The largest proportional change in reported income allowed, either way.
    # At 1 the factor is bounded between 0 and 2: income can vanish entirely
    # or double, and no further.
    "max_adjustment": 1.0,
}

# The response is a ratio of net-of-tax rates, so it is undefined when the
# baseline net-of-tax rate is zero (a baseline marginal rate of exactly 100
# percent). Below this size the denominator is treated as zero and the
# household does not respond.
NET_OF_TAX_RATE_FLOOR = 1e-12

MANIFEST: dict[str, Any] = {
    "name": "evasion_debacker",
    # The function in this file that is the response.
    "entry_point": "evasion",
    # Each class of income is scaled by its elasticity applied to the change
    # in the net-of-tax rate, so zero elasticities leave reported income
    # unchanged and the three retained-reporting factors at one.
    "zero_response": {
        "elasticity_sole_prop_and_farm": 0.0,
        "elasticity_pass_through": 0.0,
        "elasticity_rent": 0.0,
    },
    "family": "evasion",
    "reads": (
        "sole_prop1",
        "sole_prop2",
        "farm1",
        "farm2",
        "part_active",
        "part_se1",
        "part_se2",
        "part_passive",
        "scorp_active",
        "scorp_passive",
        "rent",
    ),
    "writes": (
        "sole_prop1",
        "sole_prop2",
        "farm1",
        "farm2",
        "part_active",
        "part_se1",
        "part_se2",
        "part_passive",
        "scorp_active",
        "scorp_passive",
        "rent",
        *bs.EVASION_FACTOR_COLS,
    ),
    # One representative rate per visibility group. Each is the income-tax
    # rate on the next dollar, with no payroll tax.
    "prices": ("mtr_sole_prop1", "mtr_part_active", "mtr_rent"),
    "discrete_prices": (),
    "draws": (),
}


def response_factor(
    rate_reform: Array,
    rate_baseline: Array,
    elasticity: float,
    top_end_multiplier: float,
    max_adjustment: float,
) -> Array:
    """How much of this group's reported income survives the rate change.

    The proportional change in reported income is the elasticity times the
    proportional change in the net-of-tax rate,

        elasticity x top_end_multiplier x [(1 - reform rate)/(1 - baseline rate) - 1],

    clamped to plus or minus `max_adjustment`, and the returned factor is one
    plus that. A household whose baseline net-of-tax rate is zero, or whose
    rate is not a finite number, gets a factor of one.
    """
    net_of_tax_baseline = 1.0 - jnp.asarray(rate_baseline)
    net_of_tax_reform = 1.0 - jnp.asarray(rate_reform)

    usable = (
        jnp.isfinite(net_of_tax_reform)
        & jnp.isfinite(net_of_tax_baseline)
        & (jnp.abs(net_of_tax_baseline) > NET_OF_TAX_RATE_FLOOR)
    )
    # Divide by one where the denominator is unusable, so no infinity is
    # produced; that result is discarded below.
    safe_denominator = jnp.where(usable, net_of_tax_baseline, 1.0)

    proportional_change = (
        elasticity * top_end_multiplier * (net_of_tax_reform / safe_denominator - 1.0)
    )
    proportional_change = jnp.clip(proportional_change, -max_adjustment, max_adjustment)
    return jnp.where(usable, 1.0 + proportional_change, 1.0)


def evasion(
    records: Mapping[str, Array],
    prices_reform: Mapping[str, Array],
    prices_baseline: Mapping[str, Array],
    params: Mapping[str, float],
    draws: Mapping[str, Array] | None = None,
) -> dict[str, Array]:
    """Reduce reported low-visibility income where its marginal rate rises.

    `records` are this year's household columns, `prices_reform` and
    `prices_baseline` the marginal rates under the two laws keyed by the
    stored rate column's name (`mtr_sole_prop1`, `mtr_part_active`,
    `mtr_rent`), `params` this module's parameter values, and `draws` the
    pre-committed random numbers, which this module does not use.

    Only positive income legs move; losses and the section 179 expensing legs
    are unchanged. Each earner's share of a household total is scaled by the
    same factor as the total, so the self-employment tax base stays consistent
    with the income it comes from.

    Returns the changed income columns and the three concealment factors.
    """

    factor_sole_prop_and_farm = response_factor(
        prices_reform["mtr_sole_prop1"],
        prices_baseline["mtr_sole_prop1"],
        params["elasticity_sole_prop_and_farm"],
        params["top_end_multiplier"],
        params["max_adjustment"],
    )
    factor_pass_through = response_factor(
        prices_reform["mtr_part_active"],
        prices_baseline["mtr_part_active"],
        params["elasticity_pass_through"],
        params["top_end_multiplier"],
        params["max_adjustment"],
    )
    factor_rent = response_factor(
        prices_reform["mtr_rent"],
        prices_baseline["mtr_rent"],
        params["elasticity_rent"],
        params["top_end_multiplier"],
        params["max_adjustment"],
    )

    sole_prop1 = jnp.asarray(records["sole_prop1"])
    sole_prop2 = jnp.asarray(records["sole_prop2"])
    farm1 = jnp.asarray(records["farm1"])
    farm2 = jnp.asarray(records["farm2"])
    part_active = jnp.asarray(records["part_active"])
    part_se1 = jnp.asarray(records["part_se1"])
    part_se2 = jnp.asarray(records["part_se2"])
    part_passive = jnp.asarray(records["part_passive"])
    scorp_active = jnp.asarray(records["scorp_active"])
    scorp_passive = jnp.asarray(records["scorp_passive"])
    rent = jnp.asarray(records["rent"])

    # Whether the household reports positive income of each kind. The test is
    # on the household total for proprietorship and farm income and on the
    # single column for everything else, so an earner's share moves with its
    # household total.
    has_sole_prop = (sole_prop1 + sole_prop2) > 0
    has_farm = (farm1 + farm2) > 0
    has_part_active = part_active > 0
    has_part_passive = part_passive > 0
    has_scorp_active = scorp_active > 0
    has_scorp_passive = scorp_passive > 0
    has_rent = rent > 0

    def scaled(amount: Array, keep: Array, factor: Array) -> Array:
        return jnp.where(keep, amount * factor, amount)

    changed: dict[str, Array] = {
        # Schedule C and Schedule F, with each earner's self-employment share.
        "sole_prop1": scaled(sole_prop1, has_sole_prop, factor_sole_prop_and_farm),
        "sole_prop2": scaled(sole_prop2, has_sole_prop, factor_sole_prop_and_farm),
        "farm1": scaled(farm1, has_farm, factor_sole_prop_and_farm),
        "farm2": scaled(farm2, has_farm, factor_sole_prop_and_farm),
        # Partnership and S corporation income. The two partnership
        # self-employment columns are the self-employment portion of active
        # partnership income and move with it.
        "part_active": scaled(part_active, has_part_active, factor_pass_through),
        "part_se1": scaled(part_se1, has_part_active, factor_pass_through),
        "part_se2": scaled(part_se2, has_part_active, factor_pass_through),
        "part_passive": scaled(part_passive, has_part_passive, factor_pass_through),
        "scorp_active": scaled(scorp_active, has_scorp_active, factor_pass_through),
        "scorp_passive": scaled(scorp_passive, has_scorp_passive, factor_pass_through),
        # Rent. The rent loss column is unchanged.
        "rent": scaled(rent, has_rent, factor_rent),
        # Written for the wealth and estate avoidance modules.
        "evasion_g_schc": factor_sole_prop_and_farm,
        "evasion_g_pt": factor_pass_through,
        "evasion_g_rent": factor_rent,
    }
    return changed
