"""Allocation of the corporate tax across households for the distribution table.

The allocation is written in JAX because its caller, the distribution table,
runs inside the compiled projection year.

The three-way split
-------------------
A national corporate amount ``A`` is divided in three, following the Harberger
convention on corporate tax incidence:

* The part of the tax falling on **above-normal returns** (economic rent)
  stays on the owners of corporate equity. That is ``(1 - normal share) * A``,
  spread over the above-normal base, less the share borne by foreign owners.
* The part falling on the **normal return** to the marginal investment is
  split half onto labor and half onto capital generally. So ``normal share *
  A`` splits fifty-fifty: the labor half is spread over labor income, the
  capital half over the all-capital base less the foreign share.

The labor half takes no foreign haircut.

The foreign-borne part of the two capital legs is dropped, not reallocated.
The distribution tables therefore sum to less than the corporate revenue line,
and the gap is the foreign share. This follows the Joint Committee on
Taxation; the Congressional Budget Office, the Office of Tax Analysis and the
Tax Policy Center allocate the whole amount to US households.

Allocation bases
----------------
Capital income is both the above-normal base and the normal-capital base.

Filers-only corporate line
--------------------------
The corporate line in the revenue tables counts filers only, because the
household-level burden is summed by the same code that summarizes Form 1040
totals, and that sum multiplies by the filing flag. This is a known defect.
Nothing in this module filters on the
filing flag; every household in the frame gets its share.

The national corporate figure arrives here in dollars. The caller reads
``rev_corp`` through ``reporting/receipts_inputs.py``, which has already
converted it from billions.
"""

from __future__ import annotations

import jax.numpy as jnp

from taxmodel.behavior import balance_sheet as bs

# The normal return to capital splits half onto labor and half onto capital
# (the Harberger convention).
LABOR_SHARE_OF_NORMAL_RETURN = 0.5

# A change in cost recovery (depreciation schedules, expensing) is a change in
# the tax on the marginal investment, so all of it falls on the normal return,
# whatever the configured normal-return share is for a rate change.
COST_RECOVERY_NORMAL_RETURN_SHARE = 1.0

# Pass-through business income is treated as eighty percent pay for work and
# twenty percent return on capital. The twenty percent is the same number as
# the corporate channel's `wealth.cap_flows_pt_weight`, hard-coded in a second
# place.
PASS_THROUGH_CAPITAL_SHARE = 0.20

Array = jnp.ndarray


def capital_income_base(
    wages: Array,
    business_income: Array,
    interest_dividends_and_gains: Array,
) -> tuple[Array, Array]:
    """The labor and capital-income bases of the `capital_income` convention.

    Both of the tax's capital legs are spread over capital income under this
    convention, so the second array returned serves as both the above-normal
    base and the normal-capital base.

    Args:
        wages: The household's wage and salary income.
        business_income: Sole proprietorship, farm, and the positive half of
            netted partnership and S corporation income, added together. Eighty
            percent of it is treated as pay for work done and twenty percent as
            a return on capital invested.
        interest_dividends_and_gains: Taxable and tax-exempt interest, ordinary
            and preferred-rate dividends, and short- and long-term realized
            gains, added together.

    Returns:
        Labor income and capital income, each floored at zero because a
        household cannot hold a negative share of a national tax.
    """
    labor = jnp.maximum(
        0.0, wages + business_income * (1.0 - PASS_THROUGH_CAPITAL_SHARE)
    )
    capital = jnp.maximum(
        0.0,
        business_income * PASS_THROUGH_CAPITAL_SHARE + interest_dividends_and_gains,
    )
    return labor, capital


def _spread(amount: Array, base: Array, weight: Array) -> Array:
    """Give each household `amount` times its share of the weighted base.

    When the weighted base is zero, the leg contributes nothing.
    """
    return bs.divide_where_positive(amount * base, jnp.sum(base * weight))


def allocate_corporate_dollars(
    amount_dollars: Array,
    normal_return_share: float,
    foreign_share: float,
    weight: Array,
    labor_income: Array,
    base_above_normal: Array,
    base_normal_capital: Array,
) -> Array:
    """Spread one national corporate dollar figure across households.

    Writing `A` for the amount, `s` for the share of it falling on the normal
    return and `f` for the share borne by foreign owners:

        above-normal leg   = (1 - s) * A * (1 - f), over the above-normal base
        labor leg          =       s * A * 0.5,     over labor income
        normal capital leg =       s * A * 0.5 * (1 - f), over the normal base

    The weighted total of the answer is less than `A` because the
    foreign-borne part of the two capital legs is dropped. With a
    normal-return share of 0.375 and a foreign share of 0.40, households
    receive 0.675 of the amount.

    Args:
        amount_dollars: The national figure in dollars. May be traced.
        normal_return_share: Share of the amount falling on the normal return.
        foreign_share: Share of the capital legs borne by foreign owners.
        weight: Population weight per household.
        labor_income: The base the labor leg is spread over.
        base_above_normal: The base the above-normal leg is spread over.
        base_normal_capital: The base the normal capital leg is spread over.

    Returns:
        Dollars per household.
    """
    amount = jnp.asarray(amount_dollars, dtype=jnp.float64)
    amount = jnp.where(jnp.isfinite(amount), amount, 0.0)
    weight = jnp.asarray(weight, dtype=jnp.float64)

    above_normal = _spread(
        (1.0 - normal_return_share) * amount * (1.0 - foreign_share),
        base_above_normal,
        weight,
    )
    on_labor = _spread(
        normal_return_share * amount * LABOR_SHARE_OF_NORMAL_RETURN,
        labor_income,
        weight,
    )
    on_capital = _spread(
        normal_return_share
        * amount
        * LABOR_SHARE_OF_NORMAL_RETURN
        * (1.0 - foreign_share),
        base_normal_capital,
        weight,
    )
    return above_normal + on_labor + on_capital


__all__ = [
    "COST_RECOVERY_NORMAL_RETURN_SHARE",
    "LABOR_SHARE_OF_NORMAL_RETURN",
    "PASS_THROUGH_CAPITAL_SHARE",
    "allocate_corporate_dollars",
    "capital_income_base",
]
