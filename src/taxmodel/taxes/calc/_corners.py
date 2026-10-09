"""Floors and caps written so the slope exactly at the edge is the right one.

Every threshold in the tax code is a corner: the marginal rate just below it
differs from the marginal rate just above it, and exactly on it there is no
single slope. This model reads marginal tax rates off a gradient taken through
these expressions, so which of the two slopes autodiff reports at an exact tie
is decided by nothing more than how the expression was written.
`jnp.maximum` and `jnp.minimum` report the average of the two slopes, which is
a rate no household ever faces.

The convention: at an exact threshold the household is treated as having
crossed it in the direction of more income. So the test to apply at
each site is: one more dollar of the household's income -- which branch does
that land it in? That branch takes the tie.

What decides that is whether the amount moves up or down **relative to the limit
it is being tested against**, and both cases come up constantly in the tax code:

- An amount that rises against its limit -- taxable income against a bracket
  edge, the excess over a phase-out threshold, a bracket's filled width -- is
  tested with `>=`, so a tie lands above the edge. That is the default.
- An amount that sinks against its limit -- a credit being phased out toward
  zero, a cap phasing down, an expense measured against a floor that is a
  percentage of AGI -- is tested with `>`, so a tie lands below the edge. Pass
  `sinks_against_the_limit=True`.

Where the limit is a fixed dollar amount
in the law, or zero, the only thing that can move is the amount, so the question
is just whether the amount itself falls as income rises. Where the limit is
itself a tax computation that grows with income, what matters is which of the
two grows faster: an amount that rises with income but rises more slowly than
its limit is sinking against it, and the tie belongs to the same branch as if it
had fallen outright. The Schedule D limit in `tax.py` is an example: tax at
preferred rates rises with income, but it rises more slowly than ordinary tax on
all taxable income, which is exactly why the preferred rates are worth having.

An amount that does not move with income at all, or that moves either way
depending on what kind of income the extra dollar is, has no more-income side.
Leave those at the default and say so in a comment. CONTRIBUTING.md has the
rule in full.

One caution. These return the same number as `jnp.maximum` and `jnp.minimum`
for every ordinary input, but not for a NaN: `jnp.maximum(0.0, nan)` is NaN
while `floored_at(nan, 0.0)` is 0.0. The compiled law pads unused schedule
slots with NaN, so a law array goes through `live_slots` in `_schedules.py`
before it reaches a floor or a cap here.
"""

import jax.numpy as jnp

Array = jnp.ndarray


def floored_at(
    amount: Array, floor: Array, *, sinks_against_the_limit: bool = False
) -> Array:
    """`amount`, but never below `floor`.

    An amount rising against its floor and sitting exactly on it counts as above
    it, so the tie gets the sloped branch. An amount sinking against its floor
    has already reached it, so the tie gets the flat branch.
    """
    if sinks_against_the_limit:
        return jnp.where(amount > floor, amount, floor)
    return jnp.where(amount >= floor, amount, floor)


def capped_at(
    amount: Array, cap: Array, *, sinks_against_the_limit: bool = False
) -> Array:
    """`amount`, but never above `cap`.

    An amount rising against its cap and sitting exactly on it counts as having
    reached it, so the tie gets the flat branch. An amount sinking against its
    cap is on its way back under it, so the tie gets the sloped branch.
    """
    if sinks_against_the_limit:
        return jnp.where(amount > cap, cap, amount)
    return jnp.where(amount >= cap, cap, amount)


def phased_out(amount: Array, income: Array, threshold: Array, rate: Array) -> Array:
    """`amount`, reduced by `rate` for each dollar of `income` above `threshold`,
    and never below zero.

    This is the common shape of a phase-out: a credit or a deduction shrinks as
    income rises past a threshold, until nothing is left. The excess over the
    threshold rises with income, so a household exactly on the threshold is on
    the sloped branch.

    Two different kinds of household sit exactly on the zero floor, and the
    more-income rule sends them to opposite branches.

    A household whose reduction has taken the whole amount away is sinking
    against the floor. More income takes nothing further, so it belongs on the
    flat branch.

    A household with no reduction at all is on the floor because it holds none
    of the thing being phased out. That amount rises with the next dollar of
    whatever it is measured on -- a dollar of tips earns a dollar of tip
    deduction -- so it belongs on the sloped branch.

    A household on the phase-out threshold while holding none of what is being
    phased out is on both floors at once and has no more-income side. It takes
    the default spelling.

    Every reading returns the same amount; only the slope differs.
    """
    excess = floored_at(income - threshold, 0.0)
    reduction = excess * rate
    remaining = amount - reduction
    return jnp.where(
        reduction > 0.0,
        floored_at(remaining, 0.0, sinks_against_the_limit=True),
        floored_at(remaining, 0.0),
    )


def phase_out_share(income: Array, threshold: Array, width: Array) -> Array:
    """The share of an amount lost to a phase-out that starts at `threshold`
    and runs over `width` dollars of `income`: zero below the threshold, one
    past the end, and a straight line between.

    The share rises with income, so a household exactly on the threshold is
    on the sloped branch and one exactly at the end has reached the cap.
    """
    return capped_at(floored_at(income - threshold, 0.0) / width, 1.0)
