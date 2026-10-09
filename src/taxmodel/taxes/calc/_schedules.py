"""Branchless integrations of statutory rate schedules."""

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import capped_at, floored_at

Array = jnp.ndarray


def live_slots(brackets: Array, rates: Array) -> tuple[Array, Array]:
    """Neutralize the slots of a schedule that hold no bracket.

    The compiled law gives every schedule MAX_BRACKETS slots and fills the
    unused ones with NaN (law/compile.py). An empty slot gets a lower edge of
    infinity, so no income
    ever falls in it, and a rate of zero, so it contributes nothing to the sum.
    """
    brackets = jnp.asarray(brackets)
    rates = jnp.asarray(rates)
    live = ~(jnp.isnan(brackets) | jnp.isnan(rates))
    return (
        jnp.where(live, brackets, jnp.inf),
        jnp.where(live, rates, 0.0),
    )


def last_live_rate(rates: Array) -> Array:
    """Return the rate in the last slot that holds one.

    The top rate is the rate in the last filled slot; NaN padding is skipped.
    """
    rates = jnp.asarray(rates)
    last = _last_live_slot(rates)
    return jnp.take_along_axis(rates, last[..., None], axis=-1)[..., 0]


def top_bracket_only(rates: Array, rate: Array) -> Array:
    """A schedule with the same brackets as `rates` that charges `rate` in the
    top bracket and nothing in the brackets below it.

    Unused slots stay NaN so that `live_slots` still switches them off.
    """
    rates = jnp.asarray(rates)
    rate = jnp.asarray(rate)[..., None]
    in_top = jnp.arange(rates.shape[-1]) == _last_live_slot(rates)[..., None]
    return jnp.where(jnp.isnan(rates), rates, jnp.where(in_top, rate, 0.0))


def _last_live_slot(rates: Array) -> Array:
    """The index of the last slot of a NaN-padded schedule that holds a rate."""
    live = ~jnp.isnan(rates)
    return rates.shape[-1] - 1 - jnp.argmax(live[..., ::-1], axis=-1)


def upper_edges(brackets: Array) -> Array:
    """Append the unbounded upper edge of the final statutory bracket."""
    infinity = jnp.full(brackets.shape[:-1] + (1,), jnp.inf, dtype=brackets.dtype)
    return jnp.concatenate((brackets[..., 1:], infinity), axis=-1)


def schedule_tax(income: Array, brackets: Array, rates: Array) -> Array:
    """Integrate a marginal rate schedule over income."""
    # Each rate applies only to the slice between its statutory lower and upper
    # bracket edges; the final bracket has no upper bound. Income sitting exactly
    # on a bracket edge has filled the bracket below and started the one above,
    # so the rate it faces on its next dollar is the higher one.
    income = jnp.asarray(income)
    brackets, rates = live_slots(brackets, rates)
    taxable_in_bracket = floored_at(
        capped_at(income[..., None], upper_edges(brackets)) - brackets, 0.0
    )
    return jnp.sum(taxable_in_bracket * rates, axis=-1)


def stacked_tax_exclusive(
    income: Array,
    lower_income: Array,
    brackets: Array,
    rates: Array,
) -> Array:
    """Tax income stacked above an amount excluded from the supplied income."""
    # This is the Schedule D convention for a component whose lower-priority
    # income has already filled the ordinary-rate brackets.
    income = jnp.asarray(income)
    lower_income = jnp.asarray(lower_income)
    brackets, rates = live_slots(brackets, rates)
    # The room left in each bracket shrinks as the lower-priority income
    # grows, so the corner convention runs the other way: without
    # sinks_against_the_limit, the derivative for a household with an empty bracket
    # reports tax on income it does not have.
    lower = floored_at(brackets - lower_income[..., None], 0.0, sinks_against_the_limit=True)
    upper = floored_at(
        upper_edges(brackets) - lower_income[..., None], 0.0, sinks_against_the_limit=True
    )
    taxable_in_bracket = floored_at(capped_at(income[..., None], upper) - lower, 0.0)
    return jnp.sum(taxable_in_bracket * rates, axis=-1)


def stacked_tax_inclusive(
    income: Array,
    total_income: Array,
    brackets: Array,
    rates: Array,
) -> Array:
    """Tax the top portion of total income represented by income."""
    # This form taxes the upper slice of a combined income measure, as required
    # when preferred income is stacked on top of ordinary income.
    income = jnp.asarray(income)
    total_income = jnp.asarray(total_income)
    brackets, rates = live_slots(brackets, rates)
    bottom = floored_at(total_income - income, 0.0)
    # A bracket's slice starts wherever the lower-priority income already
    # reaches, or at the bracket's own lower edge, whichever is higher.
    slice_starts_at = floored_at(bottom[..., None], brackets)
    taxable_in_bracket = floored_at(
        capped_at(total_income[..., None], upper_edges(brackets)) - slice_starts_at,
        0.0,
    )
    return jnp.sum(taxable_in_bracket * rates, axis=-1)
