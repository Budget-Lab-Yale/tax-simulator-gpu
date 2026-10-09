"""More pay taken as tips when tips are taxed at a lower rate.

An exemption for tips lowers the marginal rate on a tipped dollar, so pay
moves toward tips. Two margins, both from the Yale Budget Lab's 2024 "no tax
on tips" report, low fringe-benefit scenario:

- workers who already receive tips receive more of them (the intensive
  margin), and
- some workers who are paid only wages start receiving tips (the extensive
  margin).

Every dollar of tips gained is a dollar of base wages lost: pay is relabeled,
and total pay per earner is conserved. Where an earner does not have enough
base wages to give up, the tips gain is capped at the base wages available,
and the household is flagged in `tips_wage_floor_binding`.

The response applies from its first year onward, and only if at least one
household's tips deduction limit is nonzero, in which case every tipped worker
responds whatever that household's own limit is. See `deduction_applies`
below. Population aggregates are weighted sums over every record with a mask
for each filter, so array shapes do not depend on the data. An earner's total
pay is base wages plus tips plus overtime.

The prices this module reads are `mtr_tips1` and `mtr_tips2`, the marginal tax
rate on a dollar of each earner's tipped pay.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior import loader
from taxmodel.frame.steps_jax import FILING_JOINT

Array = jnp.ndarray

DEFAULTS: dict[str, float] = {
    # The intensive margin: the elasticity of an already-tipped worker's tips
    # with respect to the marginal rate on a tipped dollar. Yale Budget Lab,
    # "No Tax on Tips: Budgetary, Distributional, and Tax Avoidance
    # Considerations" (2024), low fringe-benefit scenario.
    "e_intensive": -1.05,
    # The extensive margin, from the same report: -0.357 is the response and
    # 0.91 the share it is expressed against in the report's arithmetic.
    "e_extensive": -0.357 / 0.91,
    # Tips for an already-tipped worker cannot grow faster than that worker's
    # cumulative nominal wage growth since the exemption took effect: 3
    # percent a year, compounded. Relabeling pay as tips faster than pay
    # itself grows would mean cutting someone's cash wage.
    "annual_wage_growth": 1.03,
    # The year the tips exemption takes effect, from which the cumulative wage
    # growth above is counted.
    "first_year": 2025,
}

# The reform law's cap on deductible tips, as the frame names it. The frame
# joins every compiled law parameter on under its section-prefixed name.
TIPS_DEDUCTION_LIMIT = "below.tip_ded_limit"

# The current year and the reform law's tips deduction limit come from the
# run, on the frame; `params` stands in for a caller testing the module on a
# hand-built frame. See `loader.run_fact`.
MANIFEST: dict[str, Any] = {
    "name": "tips_fringe_low",
    # The function in this file that is the response.
    "entry_point": "tips",
    # With both elasticities at zero, no pay is relabeled on either margin.
    "zero_response": {"e_intensive": 0.0, "e_extensive": 0.0},
    "family": "tips",
    "reads": (
        "year",
        "weight",
        "filing_status",
        "wages_base1",
        "wages_base2",
        "tips1",
        "tips2",
        "ot1",
        "ot2",
        TIPS_DEDUCTION_LIMIT,
    ),
    "writes": (
        "tips1",
        "tips2",
        "wages_base1",
        "wages_base2",
        # 1.0 for a household where at least one earner's tips gain was larger
        # than the base wages available, so base wages stopped at zero and
        # total pay rose. 0.0 everywhere else.
        "tips_wage_floor_binding",
    ),
    # The income-tax rate on the next dollar of each earner's tips, with no
    # payroll tax.
    "prices": ("mtr_tips1", "mtr_tips2"),
    "discrete_prices": (),
    # The two per-household uniform draws the frame carries, one per earner.
    # They decide which untipped workers become tipped.
    "draws": loader.MODULE_DRAW_COLUMNS,
}


def _weighted_mean(values: Array, weights: Array) -> Array:
    """A weighted mean that is zero when the weights sum to zero.

    Filtered-out records carry a weight of zero. When no record passes the
    filter there is no mean to take, and zero is the right answer everywhere
    this is used: no tipped income means no rate change to average, and no
    tipped worker means no tips-to-pay share to copy.
    """
    total_weight = jnp.sum(weights)
    return jnp.where(total_weight > 0, jnp.sum(values * weights) / total_weight, 0.0)


def tips(
    records: Mapping[str, Array],
    prices_reform: Mapping[str, Array],
    prices_baseline: Mapping[str, Array],
    params: Mapping[str, float],
    draws: Mapping[str, Array] | None = None,
) -> dict[str, Array]:
    """Move pay into tips in response to a lower marginal rate on tips.

    `records` are this year's household columns, `prices_reform` and
    `prices_baseline` the marginal rates on a tipped dollar for each earner
    under the two laws (keyed `mtr_tips1` and `mtr_tips2`), `params` this
    module's parameter values, and `draws` the pre-committed uniform random
    numbers, one per earner.

    Returns both earners' tips, both earners' base wages, and the flag saying
    whether base wages hit zero.
    """
    if draws is None:
        raise ValueError(
            "The tips module needs the per-household draws r.behavior1 and "
            "r.behavior2 to decide which untipped workers become tipped."
        )

    weight = jnp.asarray(records["weight"])
    wages_base1 = jnp.asarray(records["wages_base1"])
    wages_base2 = jnp.asarray(records["wages_base2"])
    tips1 = jnp.asarray(records["tips1"])
    tips2 = jnp.asarray(records["tips2"])
    overtime1 = jnp.asarray(records["ot1"])
    overtime2 = jnp.asarray(records["ot2"])
    filing_status = jnp.asarray(records["filing_status"])

    # Total pay is the three non-overlapping pieces added up.
    pay1 = wages_base1 + tips1 + overtime1
    pay2 = wages_base2 + tips2 + overtime2
    household_pay = pay1 + pay2
    household_tips = tips1 + tips2

    year = loader.run_fact(records, params, "year", "tips")
    deduction_applies = loader.provision_applies(
        records, params, TIPS_DEDUCTION_LIMIT, "tips"
    )

    # The price change each earner sees on a tipped dollar. Negative when the
    # reform exempts tips. An ordinary-rate reform cannot cause a tips response
    # where tips have no deduction.
    rate_change1 = jnp.where(
        deduction_applies,
        jnp.asarray(prices_reform["mtr_tips1"])
        - jnp.asarray(prices_baseline["mtr_tips1"]),
        0.0,
    )
    rate_change2 = jnp.where(
        deduction_applies,
        jnp.asarray(prices_reform["mtr_tips2"])
        - jnp.asarray(prices_baseline["mtr_tips2"]),
        0.0,
    )

    # ------------------------------------------------------------------
    # The intensive margin: bigger tips for workers who already have them
    # ------------------------------------------------------------------
    years_since_reform = jnp.maximum(0.0, year - params["first_year"])
    cumulative_wage_growth = params["annual_wage_growth"] ** years_since_reform

    growth1 = jnp.minimum(
        cumulative_wage_growth, 1.0 + params["e_intensive"] * rate_change1
    )
    growth2 = jnp.minimum(
        cumulative_wage_growth, 1.0 + params["e_intensive"] * rate_change2
    )
    intensive_gain1 = tips1 * (growth1 - 1.0)
    intensive_gain2 = tips2 * (growth2 - 1.0)

    # ------------------------------------------------------------------
    # The extensive margin: workers who had no tips start receiving them
    # ------------------------------------------------------------------
    # How much aggregate tipped income grows, in proportion: the elasticity
    # times the average rate change facing a tipped dollar. The average is
    # taken over both earners of every household, weighted by the tipped
    # income each one has, so an earner with no tips does not count.
    average_rate_change = _weighted_mean(
        jnp.concatenate([rate_change1, rate_change2]),
        jnp.concatenate(
            [
                jnp.where(deduction_applies, weight * tips1, 0.0),
                jnp.where(deduction_applies, weight * tips2, 0.0),
            ]
        ),
    )
    aggregate_tips_growth = params["e_extensive"] * average_rate_change

    # The dollars of new tips that growth asks for, at the population level.
    total_tips = jnp.sum(jnp.where(deduction_applies, weight * household_tips, 0.0))
    new_tips_needed = aggregate_tips_growth * total_tips

    # What share of a tipped household's pay comes in as tips. A newly tipped
    # worker is assumed to look like the tipped workers already out there.
    is_tipped_household = (household_tips > 0) & deduction_applies
    share_tips_new_tipped = _weighted_mean(
        bs.divide_where_positive(household_tips, household_pay),
        jnp.where(is_tipped_household, weight, 0.0),
    )

    # The pay held by households that receive no tips today, which is the pool
    # the new tips have to come out of.
    could_become_tipped = (
        (household_tips <= 0) & (household_pay > 0) & deduction_applies
    )
    untipped_pay = jnp.sum(jnp.where(could_become_tipped, weight * household_pay, 0.0))

    # The chance an untipped worker becomes tipped: the share of untipped pay
    # that has to turn into tipped pay, divided by the share of a tipped
    # worker's pay that is tips.
    becoming_tipped = bs.divide_where_positive(
        bs.divide_where_positive(new_tips_needed, untipped_pay), share_tips_new_tipped
    )

    # Who becomes tipped: an earner with no tips, positive pay, and a
    # pre-committed draw below that chance. The second earner exists only on a
    # joint return. The draws are the same numbers in every scenario of the
    # run.
    became_tipped1 = (
        (tips1 <= 0)
        & (pay1 > 0)
        & deduction_applies
        & (draws["r.behavior1"] < becoming_tipped)
    )
    became_tipped2 = (
        (tips2 <= 0)
        & (pay2 > 0)
        & deduction_applies
        & (draws["r.behavior2"] < becoming_tipped)
        & (filing_status == FILING_JOINT)
    )
    extensive_gain1 = jnp.where(became_tipped1, pay1 * share_tips_new_tipped, 0.0)
    extensive_gain2 = jnp.where(became_tipped2, pay2 * share_tips_new_tipped, 0.0)

    # ------------------------------------------------------------------
    # Relabeling: tips gained are base wages given up
    # ------------------------------------------------------------------
    # The gain is capped at the base wages available, so total pay is
    # conserved. The flag marks households where the cap bound. The cap is
    # the positive part of base wages: a few records per year whose Tax-Data
    # tips exceed their wages carry negative base wages, and a minimum
    # against a negative number would move pay out of tips at a zero
    # elasticity. A blank base wage relabels nothing.
    wanted1 = intensive_gain1 + extensive_gain1
    wanted2 = intensive_gain2 + extensive_gain2
    relabelable1 = jnp.maximum(jnp.nan_to_num(wages_base1, nan=0.0), 0.0)
    relabelable2 = jnp.maximum(jnp.nan_to_num(wages_base2, nan=0.0), 0.0)
    gain1 = jnp.minimum(wanted1, relabelable1)
    gain2 = jnp.minimum(wanted2, relabelable2)

    new_wages_base1 = wages_base1 - gain1
    new_wages_base2 = wages_base2 - gain2
    floor_binding = (wanted1 > relabelable1) | (wanted2 > relabelable2)

    return {
        "tips1": tips1 + gain1,
        "tips2": tips2 + gain2,
        "wages_base1": new_wages_base1,
        "wages_base2": new_wages_base2,
        "tips_wage_floor_binding": jnp.where(floor_binding, 1.0, 0.0),
    }
