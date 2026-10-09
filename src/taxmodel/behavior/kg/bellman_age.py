"""The selling choice when a household is described by its age and gain share.

A household is a dollar of assets held at a given age, with a gain share g:
the unrealized gain it carries divided by the market value of the assets
holding it. Each year it chooses turnover s, the share of those assets it
sells. Selling a dollar of assets realizes g dollars of gain and pays tax of
tau times g on it. Everything in this module is measured per dollar of the
household's own assets.

The path of the household's assets is fixed from outside: it does not respond
to selling or to taxes. Every tax is a flow charged against consumption in
the year it is paid.

The value of being at age a in year t with gain share g is

    V[t,a,g] = max over s in [0,1] of
                 B(s) - tau[a,t] g s
                 + beta[t] (1 + p + d[a]) (1 - m[a,t]) V[t+1,a+1,g']
                 + beta[t] m[a,t] W[t+1,a,(1 - s) g]

where B is the nontax benefit of selling (liquidity, consumption,
rebalancing), p is the yearly appreciation rate of assets, d[a] is net new
saving as a share of opening assets, m is the chance of dying during the
year, beta is one over one plus the nominal ten-year Treasury rate, and W is
the value per dollar of assets of dying with a given gain share. Sales are
churn, not decumulation: the household keeps its assets and they grow by
(1 + p + d[a]), so next year's gain share is

    g' = ((1 - s) g + p) / (1 + p + d[a]).

Order of events within a year: sales, then death, then appreciation. A
decedent leaves its whole dollar of assets to the estate, carrying the gain
share left after the year's sales, (1 - s) g, before that year's
appreciation. With c the share of the estate that goes to charity and f, r
and x the shares of a decedent's gain that are stepped up, carried to heirs
at the decedent's basis, and deemed realized and taxed:

    W[t+1,a,gd] = - (1 - c) x tau[a,t] gd
                  + (1 - c) (1 + p) sum_h w[h|a] [
                        (f + x) V[t+1,h,p/(1+p)]
                        + r V[t+1,h,(gd+p)/(1+p)] ]

A death during year t is priced under year t's law: the tax rate, the three
shares and the charity share are all read from year t's column. The heirs'
value is next year's and is discounted by beta.

The share that goes to charity leaves the dynasty and is worth nothing to the
household. The stepped-up share and the deemed share leave the heir with a
clean portfolio; only the routed share carries the gain. The heirs' assets
appreciate before the heirs' own next choice, which is the factor (1 + p) on
their value. The household weights the heirs' value in full.

The estate tax does not enter the selling choice. The asset path is fixed, so
the estate is the same dollar whatever the household sells.

Normalization of the value function
-----------------------------------
The continuation weight (1 + p + d[a]) / (1 + r) exceeds one wherever assets
grow faster than the discount rate, so the level of the value per dollar of
assets has no finite stationary value. Each year's value function is
therefore measured against a clean portfolio of the same age: V[a,0] is
subtracted from every row, so V[a,0] = 0 in every year. The terminal year is
solved as a stationary problem on that normalized value.

The two passes
--------------
The baseline pass fits desired turnover s0[a,t], the turnover a household
would choose with no tax and no continuation value, so that a household at
the cell's baseline gain share chooses the cell's baseline turnover. Baseline
selling is then matched by construction and carries no information about the
elasticity. The reform pass holds s0 fixed and solves the same problem under
reform law.

Ages run from 18 to 119. Rows past the top-coded age 80 come in already
extended: life-table mortality, every other input repeated from the age-80
cell. A household that survives age 119 has no continuation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from .inputs import GAIN_SHARE_GRID_POINTS, N_STOCK_AGES

# Floor under the fitted baseline gain share in the saving profile.
SMALLEST_FITTED_GAIN_SHARE = 1e-3

Array = jnp.ndarray

# Bisection steps on the turnover that solves the first-order condition. The
# bracket is the whole unit interval and each step halves it, so 50 steps
# reach the precision of the arithmetic.
CHOICE_STEPS = 50

# Bisection steps on desired turnover when it is being fitted to baseline
# selling, and the smallest value the bracket starts from.
DESIRED_TURNOVER_STEPS = 60
SMALLEST_DESIRED_TURNOVER = 1e-12

# Doublings allowed while pushing the upper end of that bracket high enough
# for the household to choose at least the baseline turnover.
BRACKET_DOUBLINGS = 30

# Times the terminal year's desired turnover is refitted against the
# stationary value function being solved at the same time.
TERMINAL_FIT_ROUNDS = 20

# Denominators are floored here so that a division never produces an infinite
# value or an infinite derivative.
DENOMINATOR_FLOOR = 1e-300


class AgeGainInputs(NamedTuple):
    """One law's inputs to the selling choice.

    Arrays are ``[n_extended_ages, n_years]`` on the long age grid, 18 to 119,
    unless stated otherwise. Rows past age 80 repeat the age-80 cell except
    mortality, which is the life table.

    mortality
        Chance that the holder of a dollar of unrealized gain dies during the
        year.
    tax_rate
        Marginal tax rate on gains realized by a sale under this law.
    death_share_forgiven
        Share of a decedent's gain that vanishes because the heir takes a
        stepped-up basis.
    death_share_routed_to_heirs
        Share that keeps the decedent's basis and moves to the heirs.
    death_share_taxed_at_death
        Share deemed realized and taxed in the year of death.
    charity_probability
        Share of a decedent's gain that goes to charity and is never taxed.
    baseline_gain_share
        Unrealized gain over asset value under baseline law, in [0, 1]. This
        is the gain share the reported turnover is read at.
    baseline_turnover
        Share of assets sold per year under baseline law.
    heir_operator
        ``[n_stock_ages, n_stock_ages]``, row per decedent age and column
        per heir age, each row summing to one, as ``stepper.py`` uses it.
        Decedents older than 80 are given the age-80 row.
    discount_factor_by_year
        ``[n_years]``, one over one plus the nominal ten-year Treasury rate.
    appreciation
        Yearly appreciation rate of assets, nominal.
    saving_profile
        ``[n_extended_ages]``, net new saving as a share of opening assets.
    """

    mortality: Array
    tax_rate: Array
    death_share_forgiven: Array
    death_share_routed_to_heirs: Array
    death_share_taxed_at_death: Array
    charity_probability: Array
    baseline_gain_share: Array
    baseline_turnover: Array
    heir_operator: Array
    discount_factor_by_year: Array
    appreciation: float
    saving_profile: Array


class AgeGainSolution(NamedTuple):
    """One pass of the selling choice, every matrix ``[n_extended_ages, n_years]``.

    turnover
        The share of assets the cell sells, read at its baseline gain share.
    desired_turnover
        ``s0``: the turnover the cell would choose with no tax and no
        continuation value. The baseline pass fits it; the reform pass is
        handed it and returns it unchanged.
    value_of_gain
        ``V(gbar) - V(0)``: what carrying the cell's gain share is worth
        against a clean portfolio of the same age, per dollar of assets. It is
        negative wherever gains are taxed.
    marginal_tax_on_deferred_gain
        ``K = -dV/dg`` at the baseline gain share: the present value of tax on
        one more dollar of unrealized gain, per dollar of gain.
    policy_by_gain_share
        ``[n_years, n_extended_ages, gain_grid_points]``: the turnover the cell
        would choose at every gain share on the grid, not only at its baseline
        share. The stock ledger reads this at the gain share its cells actually
        carry.
    marginal_tax_by_gain_share
        ``[n_years, n_extended_ages, gain_grid_points]``: K at every gain share
        on the grid. Read at the baseline gain share it is
        ``marginal_tax_on_deferred_gain``.
    terminal_residual
        Scalar: the largest change in the value function over the last sweep
        of the stationary terminal year.
    terminal_desired_turnover_residual
        Scalar: the largest change in the terminal year's desired turnover
        over the last refit round. Zero on a pass that is handed its desired
        turnover instead of fitting it.
    """

    turnover: Array
    desired_turnover: Array
    value_of_gain: Array
    marginal_tax_on_deferred_gain: Array
    policy_by_gain_share: Array
    marginal_tax_by_gain_share: Array
    terminal_residual: Array
    terminal_desired_turnover_residual: Array


class _YearInputs(NamedTuple):
    """One year's column of everything a year's choice reads, one per age.

    Every field is the year's own: a household that dies during the year is
    taxed under that year's law.
    """

    tax_rate: Array
    mortality: Array
    death_share_forgiven: Array
    death_share_routed_to_heirs: Array
    death_share_taxed_at_death: Array
    charity_probability: Array
    discount_factor: Array


# -----------------------------------------------------------------------------
# The saving profile behind the transition
# -----------------------------------------------------------------------------


def saving_profile(
    baseline_gain_share: Array,
    baseline_turnover: Array,
    appreciation: float,
    degree: int = 3,
) -> Array:
    """Net new saving by age, backed out of the baseline gain shares.

    The baseline gain share and turnover are averaged over calendar years at
    each age and a polynomial of the given degree in age is fitted through the
    gain shares by least squares. Saving d[a] is then set so that a household
    sitting on the fitted curve at age a, selling at the year-averaged
    baseline turnover, arrives at the fitted curve at age a + 1:

        d[a] = ((1 - sbar[a]) gfit[a] + p) / gfit[a+1] - 1 - p

    The fit runs over ages 18 to 80, the ages the data observes. Age 80
    repeats the value below it, and every age past the top code repeats age
    80. Both arguments are ``[n_ages, n_years]`` and the result is
    ``[n_ages]``. The fitted gain share is floored at a small positive value
    and saving at zero, so that the transition keeps the gain share inside
    [0, 1]; the function runs inside the compiled program and cannot raise.
    """
    gain_share = jnp.asarray(baseline_gain_share, dtype=jnp.float64)
    turnover = jnp.asarray(baseline_turnover, dtype=jnp.float64)
    n_ages = gain_share.shape[0]
    n_fitted = min(n_ages, N_STOCK_AGES)
    ages = jnp.arange(n_fitted, dtype=jnp.float64)
    observed = gain_share[:n_fitted].mean(axis=1)
    sold = turnover[:n_fitted].mean(axis=1)
    # Least-squares polynomial fit in age. Ages are scaled to [0, 1] so the
    # powers are well conditioned.
    scaled = ages / max(n_fitted - 1, 1)
    powers = jnp.stack([scaled**k for k in range(int(degree) + 1)], axis=1)
    coefficients, _, _, _ = jnp.linalg.lstsq(powers, observed, rcond=None)
    fitted = jnp.maximum(powers @ coefficients, SMALLEST_FITTED_GAIN_SHARE)
    p = float(appreciation)
    saving_below_top = ((1.0 - sold[:-1]) * fitted[:-1] + p) / fitted[1:] - 1.0 - p
    saving_below_top = jnp.maximum(saving_below_top, 0.0)
    saving = jnp.concatenate([saving_below_top, saving_below_top[-1:]])
    return jnp.concatenate([saving, jnp.repeat(saving[-1], n_ages - n_fitted)])


# -----------------------------------------------------------------------------
# Reading the value function on the gain grid
# -----------------------------------------------------------------------------


def _segment(points: Array, n_points: int) -> tuple[Array, Array, float]:
    """Which interval of the gain grid each point falls in, and how far along.

    The grid is evenly spaced from zero to one, so the interval is arithmetic
    rather than a search.
    """
    step = 1.0 / (n_points - 1)
    position = jnp.clip(points, 0.0, 1.0) / step
    index = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, n_points - 2)
    return index, position - index, step


def _interpolate(rows: Array, points: Array) -> Array:
    """Linear interpolation of each age's row at that age's points.

    ``rows`` is ``[n_ages, n_grid]`` and ``points`` is ``[n_ages, n_points]``.
    """
    index, weight, _ = _segment(points, rows.shape[1])
    low = jnp.take_along_axis(rows, index, axis=1)
    high = jnp.take_along_axis(rows, index + 1, axis=1)
    return low + weight * (high - low)


def _slope(rows: Array, points: Array) -> Array:
    """The derivative of that interpolation, constant within each interval."""
    index, _, step = _segment(points, rows.shape[1])
    low = jnp.take_along_axis(rows, index, axis=1)
    high = jnp.take_along_axis(rows, index + 1, axis=1)
    return (high - low) / step


# -----------------------------------------------------------------------------
# One year's choice
# -----------------------------------------------------------------------------


def _selling_benefit(turnover: Array, desired: Array, eta: Array) -> Array:
    """B(s) = s - eta/(eta+1) s0 (s/s0)^(1 + 1/eta), the nontax benefit of selling.

    It is concave in turnover and its marginal benefit, 1 - (s/s0)^(1/eta),
    reaches zero at the desired turnover s0.
    """
    safe_desired = jnp.maximum(desired, DENOMINATOR_FLOOR)
    ratio = turnover / safe_desired
    sells = ratio > 0.0
    safe_ratio = jnp.where(sells, ratio, 1.0)
    powered = jnp.where(sells, safe_ratio ** (1.0 + 1.0 / eta), 0.0)
    return turnover - eta / (eta + 1.0) * safe_desired * powered


def _death_value(
    gains: Array, year: _YearInputs, heir_value: Array, appreciation: float
) -> Array:
    """W: what a decedent's dollar of assets is worth, before appreciation.

    ``heir_value`` is ``[n_ages, n_grid]``: next year's value function mixed
    over the heirs of each decedent age. ``gains`` is ``[n_ages, n_points]``,
    the gain share the decedent carries after the year's sales, and the result
    has its shape.
    """
    charity = jnp.clip(year.charity_probability, 0.0, 1.0)[:, None]
    forgiven = year.death_share_forgiven[:, None]
    routed = year.death_share_routed_to_heirs[:, None]
    taxed = year.death_share_taxed_at_death[:, None]
    rate = year.tax_rate[:, None]

    tax_at_death = (1.0 - charity) * taxed * rate * gains

    clean_gain_share = jnp.full(
        (gains.shape[0], 1), appreciation / (1.0 + appreciation)
    )
    clean = _interpolate(heir_value, clean_gain_share)
    carried = _interpolate(heir_value, (gains + appreciation) / (1.0 + appreciation))
    bequest = (
        (1.0 - charity)
        * (1.0 + appreciation)
        * ((forgiven + taxed) * clean + routed * carried)
    )
    return bequest - tax_at_death


def _death_value_slope(
    gains: Array, year: _YearInputs, heir_value: Array, appreciation: float
) -> Array:
    """The slope of W in the gain share the decedent carries.

    Only the deemed tax and the gain routed to the heirs depend on that gain
    share. The factor (1 + p) on the heirs' value and the (1 + p) in the
    denominator of their gain share cancel.
    """
    charity = jnp.clip(year.charity_probability, 0.0, 1.0)[:, None]
    routed = year.death_share_routed_to_heirs[:, None]
    taxed = year.death_share_taxed_at_death[:, None]
    rate = year.tax_rate[:, None]
    carried_slope = _slope(heir_value, (gains + appreciation) / (1.0 + appreciation))
    return (1.0 - charity) * (routed * carried_slope - taxed * rate)


def _choose(
    gains: Array,
    desired: Array,
    eta: Array,
    year: _YearInputs,
    survivor_rows: Array,
    heir_value: Array,
    saving: Array,
    appreciation: float,
) -> tuple[Array, Array]:
    """Value and turnover at each gain share, for every age at once.

    ``gains`` is ``[n_ages, n_points]``, ``desired`` is one number per age,
    and ``survivor_rows`` is next year's value function at each age's next age,
    zero at the last age. Returns value and turnover, both shaped like
    ``gains``.

    The turnover is the one that solves

        s = s0 (1 - tau g + dF/ds)^eta

    where F is the discounted continuation, survivor branch and death branch
    together. The right-hand side is bisected against s over the unit
    interval, which lands on the corner s = 0 or s = 1 when the condition has
    no interior solution. Both corners are then scored against the interior
    point on the objective itself, so the answer is the best of the three.
    """
    tax_rate = year.tax_rate[:, None]
    mortality = year.mortality[:, None]
    growth = 1.0 + appreciation + saving[:, None]
    survivor_weight = year.discount_factor * growth * (1.0 - mortality)
    death_weight = year.discount_factor * mortality
    desired_by_age = desired[:, None]

    def next_gain_share(turnover: Array) -> Array:
        return ((1.0 - turnover) * gains + appreciation) / growth

    def gain_share_at_death(turnover: Array) -> Array:
        return (1.0 - turnover) * gains

    def objective(turnover: Array) -> Array:
        at_death = _death_value(
            gain_share_at_death(turnover), year, heir_value, appreciation
        )
        continuation = (
            survivor_weight * _interpolate(survivor_rows, next_gain_share(turnover))
            + death_weight * at_death
        )
        return (
            _selling_benefit(turnover, desired_by_age, eta)
            - tax_rate * gains * turnover
            + continuation
        )

    def first_order_condition(turnover: Array) -> Array:
        """The turnover the condition asks for, given the turnover priced at."""
        # Selling one more unit lowers the gain share the decedent carries by
        # g, so the death branch contributes -g times the slope of W.
        death_slope = -gains * _death_value_slope(
            gain_share_at_death(turnover), year, heir_value, appreciation
        )
        continuation_slope = (
            -survivor_weight
            * gains
            / growth
            * _slope(survivor_rows, next_gain_share(turnover))
            + death_weight * death_slope
        )
        base = 1.0 - tax_rate * gains + continuation_slope
        positive = base > 0.0
        safe_base = jnp.where(positive, base, 1.0)
        asked = jnp.where(positive, desired_by_age * safe_base**eta, 0.0)
        return jnp.clip(asked, 0.0, 1.0)

    def narrow(_, bracket):
        low, high = bracket
        middle = 0.5 * (low + high)
        sell_more = first_order_condition(middle) > middle
        return jnp.where(sell_more, middle, low), jnp.where(sell_more, high, middle)

    low, high = lax.fori_loop(
        0, CHOICE_STEPS, narrow, (jnp.zeros_like(gains), jnp.ones_like(gains))
    )
    turnover = 0.5 * (low + high)
    value = objective(turnover)
    for corner in (0.0, 1.0):
        at_corner = jnp.full_like(gains, corner)
        corner_value = objective(at_corner)
        better = corner_value > value
        turnover = jnp.where(better, at_corner, turnover)
        value = jnp.where(better, corner_value, value)
    return value, turnover


def _survivor_rows(next_value: Array) -> Array:
    """Next year's value at each age's next age, zero past the last age."""
    return jnp.concatenate(
        [next_value[1:], jnp.zeros((1, next_value.shape[1]), next_value.dtype)]
    )


def _year_layer(
    next_value: Array,
    year: _YearInputs,
    desired: Array,
    eta: Array,
    saving: Array,
    appreciation: float,
    heirs: Array,
    grid_points: int,
) -> tuple[Array, Array]:
    """One year's value function and policy on the whole gain grid.

    The returned value is measured against a clean portfolio of the same age,
    so its first column is zero.
    """
    n_ages = next_value.shape[0]
    grid = jnp.linspace(0.0, 1.0, grid_points)
    gains = jnp.broadcast_to(grid, (n_ages, grid_points))
    value, turnover = _choose(
        gains,
        desired,
        eta,
        year,
        _survivor_rows(next_value),
        heirs @ next_value,
        saving,
        appreciation,
    )
    return value - value[:, :1], turnover


def _fit_desired_turnover(
    next_value: Array,
    year: _YearInputs,
    gain_share: Array,
    target_turnover: Array,
    eta: Array,
    saving: Array,
    appreciation: float,
    heirs: Array,
    grid_points: int,
) -> Array:
    """Desired turnover that makes each age choose its baseline turnover.

    Chosen turnover rises with desired turnover, so the answer is bracketed
    and bisected. The turnover compared against the target is read the way the
    solution reports it: the policy is solved at the two grid points around the
    baseline gain share and interpolated between them.
    """
    survivor = _survivor_rows(next_value)
    heir_value = heirs @ next_value
    index, weight, step = _segment(gain_share[:, None], grid_points)
    points = jnp.concatenate([index * step, (index + 1) * step], axis=1)

    def chosen(desired: Array) -> Array:
        _, policy = _choose(
            points, desired, eta, year, survivor, heir_value, saving, appreciation
        )
        return ((1.0 - weight) * policy[:, :1] + weight * policy[:, 1:])[:, 0]

    def widen(_, high):
        return jnp.where(chosen(high) < target_turnover, 2.0 * high, high)

    low = jnp.full_like(target_turnover, SMALLEST_DESIRED_TURNOVER)
    high = lax.fori_loop(
        0,
        BRACKET_DOUBLINGS,
        widen,
        jnp.maximum(4.0 * target_turnover, 0.05),
    )

    def narrow(_, bracket):
        low, high = bracket
        middle = 0.5 * (low + high)
        above = chosen(middle) > target_turnover
        return jnp.where(above, low, middle), jnp.where(above, middle, high)

    low, high = lax.fori_loop(0, DESIRED_TURNOVER_STEPS, narrow, (low, high))
    return 0.5 * (low + high)


# -----------------------------------------------------------------------------
# The terminal year and the backward pass
# -----------------------------------------------------------------------------


def _terminal_year(
    year: _YearInputs,
    gain_share: Array,
    target_turnover: Array,
    desired: Array | None,
    eta: Array,
    saving: Array,
    appreciation: float,
    heirs: Array,
    grid_points: int,
    terminal_sweeps: int,
) -> tuple[Array, Array, Array, Array]:
    """The last year solved as if it repeated forever.

    Returns its value function, its desired turnover, the largest change in
    the value function over the last sweep, and the largest change in desired
    turnover over the last refit round. When desired turnover is being fitted,
    it and the stationary value function are found together: the fit is
    repeated against the value function the sweeps have reached so far. A pass
    that is handed its desired turnover does not refit, so its second residual
    is zero.
    """
    n_ages = gain_share.shape[0]
    fitting = desired is None
    sweeps_per_round = max(1, terminal_sweeps // TERMINAL_FIT_ROUNDS)

    def sweep(_, state):
        value, _, desired_now, desired_change = state
        swept, _ = _year_layer(
            value, year, desired_now, eta, saving, appreciation, heirs, grid_points
        )
        return swept, jnp.max(jnp.abs(swept - value)), desired_now, desired_change

    def one_round(_, state):
        value, _, desired_before, _ = state
        if fitting:
            desired_now = _fit_desired_turnover(
                value,
                year,
                gain_share,
                target_turnover,
                eta,
                saving,
                appreciation,
                heirs,
                grid_points,
            )
        else:
            desired_now = desired_before
        return lax.fori_loop(
            0,
            sweeps_per_round,
            sweep,
            (
                value,
                jnp.asarray(jnp.inf),
                desired_now,
                jnp.max(jnp.abs(desired_now - desired_before)),
            ),
        )

    start_desired = jnp.maximum(4.0 * target_turnover, 0.05) if fitting else desired
    value, residual, desired_now, desired_residual = lax.fori_loop(
        0,
        TERMINAL_FIT_ROUNDS,
        one_round,
        (
            jnp.zeros((n_ages, grid_points)),
            jnp.asarray(jnp.inf),
            start_desired,
            jnp.asarray(jnp.inf),
        ),
    )
    return value, desired_now, residual, desired_residual


def _extended_heir_operator(heir_operator: Array, n_ages: int) -> Array:
    """The heir operator on the long age grid, row per decedent age.

    Heirs are never older than the top-coded age, so the columns past it are
    zero; a decedent older than the top code is given the top-coded age's row.
    """
    heirs = jnp.asarray(heir_operator)
    if heirs.shape == (n_ages, n_ages):
        return heirs
    n_short = heirs.shape[0]
    padded = jnp.pad(heirs, ((0, 0), (0, n_ages - n_short)))
    rows = jnp.minimum(jnp.arange(n_ages), n_short - 1)
    return padded[rows]


def _year_columns(inputs: AgeGainInputs) -> _YearInputs:
    """Every year's column of the inputs, stacked year first for the scan."""

    def by_year(values: Array) -> Array:
        return jnp.asarray(values).T

    return _YearInputs(
        tax_rate=by_year(inputs.tax_rate),
        mortality=by_year(inputs.mortality),
        death_share_forgiven=by_year(inputs.death_share_forgiven),
        death_share_routed_to_heirs=by_year(inputs.death_share_routed_to_heirs),
        death_share_taxed_at_death=by_year(inputs.death_share_taxed_at_death),
        charity_probability=by_year(inputs.charity_probability),
        discount_factor=jnp.asarray(inputs.discount_factor_by_year),
    )


def marginal_tax_on_grid(value: Array) -> Array:
    """K, the present value of tax on one more dollar of unrealized gain, at
    every point of the gain grid.

    ``value`` is one year's value function, ``[n_ages, gain_grid_points]``.
    K is ``-dV/dg``, taken as a central difference over the two neighbouring
    grid points, and as a one-sided difference at the two ends of the grid.
    Read between grid points by linear interpolation, it equals the central
    difference of the interpolated value function one grid step either side
    of the reading point.
    """
    grid_points = value.shape[1]
    step = 1.0 / (grid_points - 1)
    above = jnp.concatenate([value[:, 1:], value[:, -1:]], axis=1)
    below = jnp.concatenate([value[:, :1], value[:, :-1]], axis=1)
    # Two grid steps apart in the interior, one at either end.
    width = jnp.full((grid_points,), 2.0 * step).at[0].set(step).at[-1].set(step)
    return -(above - below) / width


def _readings(
    value: Array, policy: Array, marginal_tax_grid: Array, gain_share: Array
) -> tuple[Array, Array, Array]:
    """What a year's solution reports at the cell's baseline gain share.

    The turnover, the value of the gain and K are each read off their grid by
    linear interpolation.
    """
    at = gain_share[:, None]
    turnover = _interpolate(policy, at)[:, 0]
    value_of_gain = _interpolate(value, at)[:, 0]
    marginal_tax = _interpolate(marginal_tax_grid, at)[:, 0]
    return turnover, value_of_gain, marginal_tax


def at_gain_share(rows: Array, gain_share: Array) -> Array:
    """A schedule solved on the gain grid, read at a gain share it was not
    solved at.

    ``rows`` is one year of a schedule, ``[n_ages, gain_grid_points]``, such
    as the turnover policy or K, and ``gain_share`` is one number per age.
    The schedule is interpolated linearly between grid points, the same
    reading the solve itself uses at the baseline gain share.
    """
    return _interpolate(rows, jnp.asarray(gain_share)[:, None])[:, 0]


def _solve_all_years(
    inputs: AgeGainInputs,
    eta: Array,
    desired_turnover: Array | None,
    grid_points: int,
    terminal_sweeps: int,
):
    """Solve every year and age, backward from the stationary terminal year.

    Returns the readings at each cell's baseline gain share, the fitted or
    given desired turnover, the two terminal residuals, and the value function,
    policy and K of every year, each ``[n_years, n_ages, n_grid]``. The value
    function is returned for the tests; the production caller reads
    :func:`solve_selling_choice`.
    """
    n_ages, n_years = jnp.shape(inputs.mortality)
    heirs = _extended_heir_operator(inputs.heir_operator, n_ages)
    years = _year_columns(inputs)
    appreciation = float(inputs.appreciation)
    saving = jnp.asarray(inputs.saving_profile)
    gain_share = jnp.asarray(inputs.baseline_gain_share).T
    baseline_turnover = jnp.asarray(inputs.baseline_turnover).T
    given = None if desired_turnover is None else jnp.asarray(desired_turnover).T

    terminal_value, terminal_desired, residual, desired_residual = _terminal_year(
        _year_column(years, n_years - 1),
        gain_share[n_years - 1],
        baseline_turnover[n_years - 1],
        None if given is None else given[n_years - 1],
        eta,
        saving,
        appreciation,
        heirs,
        grid_points,
        terminal_sweeps,
    )

    def one_year(next_value, year_state):
        year, this_gain_share, this_turnover, this_desired = year_state
        if given is None:
            desired = _fit_desired_turnover(
                next_value,
                year,
                this_gain_share,
                this_turnover,
                eta,
                saving,
                appreciation,
                heirs,
                grid_points,
            )
        else:
            desired = this_desired
        value, policy = _year_layer(
            next_value, year, desired, eta, saving, appreciation, heirs, grid_points
        )
        marginal_tax_grid = marginal_tax_on_grid(value)
        turnover, value_of_gain, marginal_tax = _readings(
            value, policy, marginal_tax_grid, this_gain_share
        )
        return value, (
            turnover,
            desired,
            value_of_gain,
            marginal_tax,
            value,
            policy,
            marginal_tax_grid,
        )

    scanned = (
        years,
        gain_share,
        baseline_turnover,
        jnp.zeros_like(gain_share) if given is None else given,
    )
    _, (
        turnover,
        desired,
        value_of_gain,
        marginal_tax,
        value,
        policy,
        marginal_tax_grid,
    ) = lax.scan(one_year, terminal_value, scanned, reverse=True)
    return (
        turnover.T,
        desired.T,
        value_of_gain.T,
        marginal_tax.T,
        policy,
        marginal_tax_grid,
        residual,
        desired_residual,
        value,
        terminal_desired,
    )


def _year_column(year_inputs: _YearInputs, index: int) -> _YearInputs:
    """One year's column out of the stacked year-first inputs."""
    return _YearInputs(*(field[index] for field in year_inputs))


def solve_selling_choice(
    inputs: AgeGainInputs,
    eta: Array,
    desired_turnover: Array | None = None,
    *,
    gain_grid_points: int = GAIN_SHARE_GRID_POINTS,
    terminal_sweeps: int = 400,
) -> AgeGainSolution:
    """Solve the selling choice backward over year and age.

    With ``desired_turnover`` left as None this is the baseline pass: desired
    turnover is fitted so that a household at the cell's baseline gain share
    chooses the cell's baseline turnover, and the returned turnover reproduces
    the baseline to the tolerance of that fit. Supplying it solves a reform
    against the same desired turnover, held fixed.

    ``eta`` is the curvature of the nontax benefit of selling, common to every
    cell. ``gain_grid_points`` and ``terminal_sweeps`` are static: the gain
    grid is evenly spaced from zero to one, and the terminal year takes that
    many sweeps however close it is to standing still.
    """
    (
        turnover,
        desired,
        value_of_gain,
        marginal_tax,
        policy,
        marginal_tax_grid,
        residual,
        desired_residual,
        _value,
        _terminal,
    ) = _solve_all_years(
        inputs, eta, desired_turnover, gain_grid_points, terminal_sweeps
    )
    return AgeGainSolution(
        turnover=turnover,
        desired_turnover=desired,
        value_of_gain=value_of_gain,
        marginal_tax_on_deferred_gain=marginal_tax,
        policy_by_gain_share=policy,
        marginal_tax_by_gain_share=marginal_tax_grid,
        terminal_residual=residual,
        terminal_desired_turnover_residual=desired_residual,
    )
