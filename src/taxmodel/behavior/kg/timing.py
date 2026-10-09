"""Retiming: realizations moved between years by a rate change.

The selling choice in ``bellman_age.py`` is a long-run margin — how much of its
portfolio a cell turns over per year at the rate it faces.  Separately, a
taxpayer who knows the rate rises next year sells this year instead.  This
module is that short-run response, laid on top of the long-run one as an
overlay.

A calibrated share of each cell-year's baseline realizations is treated as
moveable.  Each moveable bucket looks at the years within a window either side
and moves toward whichever has the lowest change in the marginal rate.  The
share of the bucket that moves is the rate saving divided by the reference
wedge, capped at one.

The comparison is on the change in the rate rather than its level, so a run
where law does not change retimes nothing at all.  The reform rate compared is
the law-only rate: the reform law's rate on the mechanical households, with the
corporate, payroll and wealth-drawdown adjustments included, measured before
the frozen gains pass.

Non-differentiable point
------------------------
Choosing the destination year is a hard select among a small fixed set of
candidates.  The amount moved is smooth in the rate saving and in the timeable
share, so a gradient with respect to either calibrated parameter is fine
almost everywhere; but the gradient does not see the destination flipping from
one year to another when two years' rate changes cross.  It is the only known
non-differentiable point in the schedule solver.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .inputs import TIMING_TIE_TOLERANCE

Array = jnp.ndarray


class RetimingOverlay(NamedTuple):
    """Where the moveable realizations sit before and after retiming.

    Both are dollars of realizations, shaped ``[age, year]``.

    moveable_baseline
        The timeable share of each cell-year's baseline realizations, sitting
        in the year it was earned.
    moveable_reform
        The same dollars after each bucket moved toward its cheapest year in
        the window.  Summed over the whole window this equals
        ``moveable_baseline`` exactly: retiming moves dollars, it never creates
        or destroys them.
    net_shift
        ``moveable_reform - moveable_baseline``: the net dollars a cell-year
        gains or loses to retiming.  This is what is added to the long-run
        response.
    """

    moveable_baseline: Array
    moveable_reform: Array
    net_shift: Array


def _window_offsets(window: int) -> list[int]:
    """Candidate years, ordered the way ties are broken.

    A tie goes to the nearest year, and among equally near years to the earlier
    one.  Ordering the offsets by distance and then by year, and taking the
    first that attains the minimum, gives that.  Staying put is offset zero,
    which comes first.
    """
    return sorted(range(-window, window + 1), key=lambda d: (abs(d), d))


def build_retiming_overlay(
    baseline_realizations: Array,
    rate_change: Array,
    timeable_share: Array,
    timing_reference_wedge: Array,
    timing_window: int,
) -> RetimingOverlay:
    """Move the moveable realizations toward their cheapest nearby year.

    Parameters
    ----------
    baseline_realizations
        ``[age, year]`` dollars of long-term gains realized under baseline law.
    rate_change
        ``[age, year]`` reform law-only marginal rate minus baseline marginal
        rate.  Exactly zero everywhere in a run where law does not change, so
        the overlay moves nothing there.
    timeable_share
        Share of each cell-year's baseline realizations that can move.
    timing_reference_wedge
        The rate saving at which a whole bucket moves.
    timing_window
        Years either side to consider.  Zero means no retiming.
    """
    moveable_baseline = timeable_share * baseline_realizations

    if timing_window == 0:
        return RetimingOverlay(
            moveable_baseline=moveable_baseline,
            moveable_reform=moveable_baseline,
            net_shift=jnp.zeros_like(moveable_baseline),
        )

    n_ages, n_years = baseline_realizations.shape
    offsets = _window_offsets(timing_window)

    # Years outside the simulation are not candidates.  Padding with positive
    # infinity keeps them from ever being the cheapest.
    padded = jnp.pad(
        rate_change,
        ((0, 0), (timing_window, timing_window)),
        constant_values=jnp.inf,
    )
    candidates = jnp.stack(
        [
            padded[:, timing_window + d : timing_window + d + n_years]
            for d in offsets
        ],
        axis=0,
    )

    cheapest = jnp.min(candidates, axis=0)
    attains_minimum = candidates <= cheapest + TIMING_TIE_TOLERANCE

    # argmax on a boolean returns the first True, which with the ordering above
    # is the nearest and then earliest of the cheapest years.
    chosen_index = jnp.argmax(attains_minimum, axis=0)
    offset_values = jnp.asarray(offsets)
    chosen_offset = offset_values[chosen_index]

    staying_here = candidates[0]
    rate_at_destination = jnp.take_along_axis(
        candidates, chosen_index[None, :, :], axis=0
    )[0]
    rate_saving = staying_here - rate_at_destination

    share_that_moves = jnp.clip(rate_saving / timing_reference_wedge, 0.0, 1.0)
    share_that_moves = jnp.where(chosen_offset == 0, 0.0, share_that_moves)
    moved = moveable_baseline * share_that_moves

    ages = jnp.arange(n_ages)[:, None]
    years = jnp.arange(n_years)[None, :]
    moveable_reform = (
        jnp.zeros_like(moveable_baseline)
        .at[ages, years]
        .add(moveable_baseline - moved)
        .at[ages, years + chosen_offset]
        .add(moved)
    )

    return RetimingOverlay(
        moveable_baseline=moveable_baseline,
        moveable_reform=moveable_reform,
        net_shift=moveable_reform - moveable_baseline,
    )


def retiming_turnover_shift(
    baseline_gain_stock: Array, overlay: RetimingOverlay
) -> Array:
    """The overlay's net dollars turned into a change in turnover.

    The overlay is in dollars of realized gains.  Dividing by the cell's stock
    of gains turns it into turnover: under pro-rata selling, realizing a dollar
    of gain out of a stock of G is turning over 1/G of the portfolio.  A cell
    holding no gain has no turnover to shift.
    """
    holds_gains = baseline_gain_stock > 0.0
    safe_stock = jnp.where(holds_gains, baseline_gain_stock, 1.0)
    moveable_turnover_baseline = jnp.where(
        holds_gains, overlay.moveable_baseline / safe_stock, 0.0
    )
    moveable_turnover_reform = jnp.where(
        holds_gains, overlay.moveable_reform / safe_stock, 0.0
    )
    return moveable_turnover_reform - moveable_turnover_baseline
