"""Relabeling compensation as capital gain, and the stock of gain that creates.

A business owner taxed more heavily on wages than on the gain in the value of
their own firm can take less pay and leave the money in the business, where it
shows up later as a capital gain.  The response is priced one leg of income at
a time.  For each leg, the change in the advantage of converting is the change
in that leg's own marginal rate less the change in the household's marginal
rate on one more dollar of realized long-term gain:

    wedge = (scenario rate on the leg - baseline rate on the leg)
            - (scenario rate on long-term gain - baseline rate on long-term gain)

Both rates are measured on the mechanical world's frame for the same
household, under reform law and under baseline law.  The amount converted is
the response parameter sigma times that wedge times the convertible part of
the leg, clamped so no leg can be turned negative.  Only the labor content of
pass-through income is convertible, since the rest was never compensation to
begin with; wages are convertible in full.

Sigma and the pass-through labor share are read from the conversion file the
behavior leg binds (``config/calibrations/kg/conversion.yaml`` by default)
and passed in.

The converted dollars raise the stock of unrealized gain from the following
year.  Neither rate depends on the stock, so the conversions are priced once,
before the stock is stepped (``years/gains_solve.py``).  The clamp of the
conversion to the household's live income happens later, in the in-year
module that applies these numbers.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .stock_inputs import (
    CONVERSION_LEG_IS_ACTIVE_BUSINESS,
    KG_N_AGES,
    ConversionPool,
)

Array = jnp.ndarray


class YearConversions(NamedTuple):
    """One year of conversions: per household per leg, and summed to age cells."""

    by_household_leg: Array
    inflow_by_age: Array


def _zero_where_missing(values: Array) -> Array:
    """Treat a missing number as zero."""

    return jnp.where(jnp.isnan(values), 0.0, values)


def convertible_share_by_leg(pass_through_labor_share: Array | float) -> Array:
    """How much of each leg is compensation that could have been paid as equity.

    All of a wage, and the labor share of active pass-through income.
    """

    share = jnp.asarray(pass_through_labor_share, dtype=jnp.float64)
    is_business = jnp.asarray(CONVERSION_LEG_IS_ACTIVE_BUSINESS)
    return jnp.where(is_business, share, jnp.ones_like(share))


def price_conversions_one_year(
    pool: ConversionPool,
    sigma: Array | float,
    pass_through_labor_share: Array | float,
    n_ages: int = KG_N_AGES,
) -> YearConversions:
    """Price one year's conversions from that year's rates.

    Parameters
    ----------
    pool
        One year's slice of the conversion inputs: arrays over households, with
        the leg arrays carrying a second axis over the five legs.
    sigma
        Dollars relabeled per dollar of convertible income per unit of wedge.
    pass_through_labor_share
        The labor content of active pass-through income.
    n_ages
        The number of age cells in the stock of gain.

    Returns the conversion for every household and leg, and those conversions
    summed by the household's age into the inflow the stock of gains receives.
    """

    valid_age = (pool.age_index >= 0) & (pool.age_index < n_ages)
    lookup_age = jnp.where(valid_age, pool.age_index, 0)
    change_in_gains_rate = _zero_where_missing(
        pool.gains_rate_scenario - pool.gains_rate_baseline
    )

    leg = _zero_where_missing(pool.leg_amount)
    is_business = jnp.asarray(CONVERSION_LEG_IS_ACTIVE_BUSINESS)
    has_active_business = jnp.any(jnp.where(is_business, leg > 0, False), axis=-1)

    taxable_income = pool.taxable_income
    threshold = pool.top_bracket_threshold
    income_known = ~jnp.isnan(taxable_income) & ~jnp.isnan(threshold)
    in_pool = (
        valid_age
        & has_active_business
        & income_known
        & (_zero_where_missing(taxable_income) >= _zero_where_missing(threshold))
    )

    convertible = convertible_share_by_leg(pass_through_labor_share) * leg
    convertible = jnp.where(in_pool[:, None] & (leg > 0), convertible, 0.0)

    wedge = (
        pool.marginal_rate_scenario - pool.marginal_rate_baseline
    ) - change_in_gains_rate[:, None]
    wedge = _zero_where_missing(wedge)

    converted = jnp.asarray(sigma, dtype=jnp.float64) * wedge * convertible
    converted = jnp.where(convertible > 0, converted, 0.0)
    # Conversion cannot exceed the leg it comes out of.  The clamp is on the
    # whole leg, not on its convertible part.
    leg_size = jnp.abs(leg)
    converted = jnp.clip(converted, -leg_size, leg_size)

    household_total = jnp.sum(converted, axis=-1)
    # Households with no usable age are sent to an extra cell past the top of
    # the grid, which is dropped from the answer.
    destination = jnp.where(valid_age, lookup_age, n_ages)
    inflow = jax.ops.segment_sum(
        pool.weight * household_total,
        destination,
        num_segments=n_ages + 1,
    )[:n_ages]

    return YearConversions(by_household_leg=converted, inflow_by_age=inflow)


def price_conversions_all_years(
    pool: ConversionPool,
    sigma: Array | float,
    pass_through_labor_share: Array | float,
    n_ages: int = KG_N_AGES,
) -> YearConversions:
    """Price every year's conversions at once.

    Conversion in year t reads year t's rates and enters the stock at the end
    of year t.  No year's conversion depends on any other year's, so all
    years are priced in one calculation.  The pool's arrays are shaped
    [year, household], with the leg axis last on the leg arrays.
    """

    return jax.vmap(
        lambda one_year: price_conversions_one_year(
            one_year, sigma, pass_through_labor_share, n_ages
        )
    )(pool)
