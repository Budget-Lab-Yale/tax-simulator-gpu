"""The capital-gains schedule solver.

A caller enters the package here.  Given age-cell aggregates, matrices of
marginal rates by age and year, and the calibrated parameters, it returns the
schedule bundle: how fast each cell sells, the factor by which the reform moves
that pace, both policies' priced tax on a deferred dollar, and the treatment of
gains at death.  Every field is an age-by-year array on the stock's
age grid, 18 to 80.

The pace of selling is called turnover throughout: the share of the market value
of a cell's assets sold during the year.  Under pro-rata selling out of a
well-mixed portfolio that is the same number as the share of the cell's
unrealized gains realized, which is how the model computes it and how the
emitted cell table labels it.

Nothing here touches per-household data.  The selling choice is solved on
the baseline stock of gains only.  It returns reform law's turnover and priced
tax on a deferred dollar at every gain share, and the stock recurrence reads
both at the gain share each cell actually carries, so the whole solve runs
before any year is stepped.

The order of work:

1. Solve the baseline selling choice, which recovers the desired turnover
   behind the observed baseline pace for every age and year.
2. Solve the reform's selling choice against that desired turnover, held
   fixed.
3. Build the retiming overlay from the law-only rates and add it to the
   long-run response.

The baseline and reform solves also return the present value of tax on one
more dollar of unrealized gain, which is the priced tax the bundle carries.

``solve_frozen_schedule()`` returns the same shape of answer with turnover
pinned to baseline and no selling choice at all.  It runs before the static
pass, so the pure law arithmetic of changing the treatment of gains at death
lands in the static world and the behavioral effect stays equal to the
conventional estimate minus the static one.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from . import bellman_age, timing
from .inputs import (
    GAIN_SHARE_GRID_POINTS,
    SMALLEST_SAFE_DENOMINATOR,
    TERMINAL_RESIDUAL_TOLERANCE,
    Calibration,
    ScheduleInputs,
    extend_to_choice_grid,
)

Array = jnp.ndarray

# What a bundle carries for a terminal residual it was not given: one
# assembled by hand, or the frozen pass, which solves nothing.
NO_RESIDUAL = 0.0


class ScheduleBundle(NamedTuple):
    """Everything the solver hands to the stock recurrence and the appliers.

    Every array is ``[age, year]`` on the stock's age grid (18 to 80, with 80 a
    pooled top-coded cell) except where said otherwise.

    How fast each cell sells, and what moves it:

    baseline_turnover
        Share of the market value of its assets the cell sells during the year
        under baseline law, which under pro-rata selling is also the share of
        its unrealized gains it realizes.  Passed through from the inputs.
    reform_turnover
        The same share under reform law, read at the cell's baseline gain
        share: the long-run response from the selling choice plus the net
        retiming, clipped to the unit interval.  The stock ledger does not read
        this; it reads the policy schedule below at the gain share its cells
        actually carry.  The two agree wherever a cell holds exactly its
        baseline stock.
    rate_factor
        ``reform_turnover / baseline_turnover``, and one where the cell sells
        nothing in the baseline.  The applier multiplies each household's
        realized long-term gains by it.

    What the stock ledger reads to price a cell at its own gain share.  A cell that has inherited gain under carryover, or that has
    held back gain under a rate rise, carries a higher gain share than the
    baseline one the schedule was read at, and sells at the reform policy's
    rate for that share:

    reform_policy_by_gain_share
        ``[year, age, gain-share grid point]``: the turnover the reform pass
        chooses at every gain share on the solver's grid, evenly spaced from
        zero to one.
    long_run_turnover_scale
        ``[age, year]``: the observed baseline turnover over the baseline
        re-solve's turnover at the baseline gain share.  Multiplying the
        reform policy by it takes the baseline fit error out.  One where the
        re-solve sells nothing.
    retiming_turnover_shift
        ``[age, year]``: the retiming overlay's net dollars as a change in
        turnover, added on top of the long-run response.
    baseline_gain_stock, baseline_asset_value
        ``[age, year]``: the numerator and denominator of the cell's gain
        share, passed through from the inputs so the ledger can form the
        scenario gain share from the stock it is tracking.

    The selling choice's own quantities:

    value_baseline, value_reform
        The value of holding a dollar of unrealized gain under each law.

    The priced tax on a deferred dollar, which the conversion and entity
    shifting modules read:

    tax_on_deferred_gain_baseline, tax_on_deferred_gain_reform
        Expected present value of tax on one more dollar of unrealized gain,
        per dollar of gain, under each law.  Each is the selling choice's own
        ``K = -dV/dg``, read at the cell's baseline gain share at the start of
        the year, before that year's sales.  The retiming overlay is not in
        it; the priced tax is a long-run price on deferring a dollar.  The
        baseline one is what the other modules read for baseline law.  The
        reform one is a diagnostic: the stock ledger reads the grid below at
        the gain share its cells actually carry.  The two readings of reform
        law agree wherever a cell holds exactly its baseline stock.
    reform_tax_on_deferred_gain_by_gain_share
        ``[year, age, gain-share grid point]``: reform law's K at every gain
        share on the solver's grid.  The stock ledger reads it at the scenario
        gain share, the same way it reads the reform policy.

    The treatment of gains at death, passed through so that the stock
    recurrence and the appliers read one object:

    death_share_forgiven, death_share_routed_to_heirs,
    death_share_taxed_at_death
        Shares of a decedent's gains that vanish under a stepped-up basis,
        move to the heirs' cells, and are taxed in the year of death.

    Diagnostics:

    terminal_residual_baseline, terminal_residual_reform,
    terminal_residual_baseline_resolve
        Scalars: the largest change the last sweep of each solve's terminal
        year was still making in the value of holding a gain.  There are three
        because the baseline law is solved twice: once to fit the desired
        turnover, and once against the fitted desired turnover.
    terminal_desired_turnover_residual_baseline,
    terminal_desired_turnover_residual_reform,
    terminal_desired_turnover_residual_baseline_resolve
        Scalars: the largest change the last refit round of each solve's
        terminal year was still making in the desired turnover.  Zero on the
        two solves that are handed a desired turnover rather than fitting one.
        ``check_schedule()`` stops a run whose terminal year has not settled on
        either count.
    baseline_turnover_fit_gap
        Scalar: the largest gap, over every cell and year of the selling
        choice's age grid, between the turnover the baseline pass chooses and
        the observed baseline turnover it was fitted to.  Zero from the frozen
        pass, which fits nothing.  ``check_schedule()`` stops a run whose gap
        is above the fit tolerance.
    frozen
        True when the bundle came from ``solve_frozen_schedule()``, in which
        case reform turnover is baseline turnover everywhere and both priced
        taxes on a deferred dollar are zero.  See that function's docstring.
    """

    baseline_turnover: Array
    reform_turnover: Array
    rate_factor: Array

    reform_policy_by_gain_share: Array
    long_run_turnover_scale: Array
    retiming_turnover_shift: Array
    baseline_gain_stock: Array
    baseline_asset_value: Array

    value_baseline: Array
    value_reform: Array

    tax_on_deferred_gain_baseline: Array
    tax_on_deferred_gain_reform: Array
    reform_tax_on_deferred_gain_by_gain_share: Array

    death_share_forgiven: Array
    death_share_routed_to_heirs: Array
    death_share_taxed_at_death: Array

    terminal_residual_baseline: Array
    terminal_residual_reform: Array

    frozen: bool
    baseline_turnover_fit_gap: Array = NO_RESIDUAL
    terminal_residual_baseline_resolve: Array = NO_RESIDUAL
    terminal_desired_turnover_residual_baseline: Array = NO_RESIDUAL
    terminal_desired_turnover_residual_reform: Array = NO_RESIDUAL
    terminal_desired_turnover_residual_baseline_resolve: Array = NO_RESIDUAL


def rate_factor(reform_turnover: Array, baseline_turnover: Array) -> Array:
    """The applier's multiplier on realized gains.

    A cell that sold nothing in the baseline has no ratio to take, so the
    factor is one there and that cell's households are left alone.
    """
    sells = baseline_turnover > 0.0
    safe_baseline = jnp.where(sells, baseline_turnover, 1.0)
    return jnp.where(sells, reform_turnover / safe_baseline, 1.0)


def solve_schedule(inputs: ScheduleInputs, calibration: Calibration) -> ScheduleBundle:
    """Solve the schedules for one scenario.

    Differentiable end to end with respect to the calibrated parameters, with
    one documented exception: the retiming overlay's choice of destination year
    is a hard select, so the gradient does not see a bucket switching years.
    See ``timing.py``'s header.
    """
    inputs.validate()
    calibration.validate()

    n_stock = inputs.n_stock_ages
    n_extended = inputs.n_extended_ages

    # The selling choice reads the cell aggregates on its longer age grid,
    # where every row past the top-coded age of 80 repeats the age-80 row.
    def extended(values: Array) -> Array:
        return extend_to_choice_grid(values, n_choice_ages=n_extended)

    charity_extended = extended(inputs.charity_probability)
    baseline_turnover_extended = extended(inputs.baseline_turnover)

    # The share of the cell's asset value that is unrealized gain. Selling a
    # dollar of assets brings this much gain into income, so the marginal rate
    # times this share is the tax cost of raising a dollar of cash.
    gain_share = jnp.clip(
        inputs.baseline_gain_stock
        / jnp.maximum(inputs.baseline_asset_value, SMALLEST_SAFE_DENOMINATOR),
        0.0,
        1.0,
    )
    gain_share_extended = extended(gain_share)

    zeros_extended = jnp.zeros((n_extended, inputs.n_years), dtype=gain_share.dtype)
    ones_extended = jnp.ones_like(zeros_extended)

    saving = bellman_age.saving_profile(
        gain_share_extended,
        baseline_turnover_extended,
        float(calibration.appreciation),
    )

    def leg(
        tax_rate: Array,
        forgiven: Array,
        routed: Array,
        taxed_at_death: Array,
    ) -> bellman_age.AgeGainInputs:
        return bellman_age.AgeGainInputs(
            mortality=inputs.gain_weighted_mortality,
            tax_rate=tax_rate,
            death_share_forgiven=forgiven,
            death_share_routed_to_heirs=routed,
            death_share_taxed_at_death=taxed_at_death,
            charity_probability=charity_extended,
            baseline_gain_share=gain_share_extended,
            baseline_turnover=baseline_turnover_extended,
            heir_operator=inputs.heir_operator,
            discount_factor_by_year=inputs.discount_factor_by_year,
            appreciation=float(calibration.appreciation),
            saving_profile=saving,
        )

    # Pass 1: the baseline solve, which recovers the nontax benefit of selling
    # from the observed baseline pace. Current law gives every asset a
    # stepped-up basis at death, so nothing is taxed at death here.
    baseline_pass = bellman_age.solve_selling_choice(
        leg(
            inputs.marginal_rate_baseline,
            ones_extended,
            zeros_extended,
            zeros_extended,
        ),
        calibration.elasticity,
        gain_grid_points=GAIN_SHARE_GRID_POINTS,
    )

    # Pass 2: the reform solve, against the nontax benefit pass 1 recovered.
    reform_pass = bellman_age.solve_selling_choice(
        leg(
            inputs.marginal_rate_reform,
            extended(inputs.death_share_forgiven),
            extended(inputs.death_share_routed_to_heirs),
            extended(inputs.death_share_taxed_at_death),
        ),
        calibration.elasticity,
        baseline_pass.desired_turnover,
        gain_grid_points=GAIN_SHARE_GRID_POINTS,
    )

    # The long-run reform turnover is the observed baseline turnover scaled by
    # the ratio of the reform pass to a baseline re-solve at the fitted
    # desired turnover. The fit lands within a tolerance rather than exactly,
    # and the ratio takes that fit error out: the re-solve follows the same
    # path as the reform pass, so a reform identical to baseline law leaves
    # every cell's turnover exactly where it was.
    baseline_resolve = bellman_age.solve_selling_choice(
        leg(
            inputs.marginal_rate_baseline,
            ones_extended,
            zeros_extended,
            zeros_extended,
        ),
        calibration.elasticity,
        baseline_pass.desired_turnover,
        gain_grid_points=GAIN_SHARE_GRID_POINTS,
    )
    chosen_baseline = jnp.where(
        baseline_resolve.turnover > 0.0, baseline_resolve.turnover, 1.0
    )
    long_run_turnover_scale = jnp.where(
        baseline_resolve.turnover > 0.0,
        baseline_turnover_extended / chosen_baseline,
        1.0,
    )[:n_stock, :]
    long_run_reform_turnover = (
        long_run_turnover_scale * reform_pass.turnover[:n_stock, :]
    )

    # The retiming overlay compares the law-only reform rate against the
    # baseline rate, and nothing else in the model uses the law-only rate.
    rate_change = (
        inputs.marginal_rate_reform_law_only - inputs.marginal_rate_baseline
    )[:n_stock, :]
    overlay = timing.build_retiming_overlay(
        baseline_realizations=inputs.baseline_realizations,
        rate_change=rate_change,
        timeable_share=calibration.timeable_share,
        timing_reference_wedge=calibration.timing_reference_wedge,
        timing_window=calibration.timing_window,
    )

    retiming_turnover_shift = timing.retiming_turnover_shift(
        inputs.baseline_gain_stock, overlay
    )
    reform_turnover = jnp.clip(
        long_run_reform_turnover + retiming_turnover_shift, 0.0, 1.0
    )

    # The baseline pass fits the desired turnover so that a cell at its
    # baseline gain share chooses its observed baseline turnover. The gap the
    # fit leaves is carried out and checked.
    baseline_turnover_fit_gap = jnp.max(
        jnp.abs(baseline_pass.turnover - baseline_turnover_extended)
    )

    # The priced tax on a deferred dollar comes out of the two selling-choice
    # solves, one per law.  Rows past the top-coded age of 80 are dropped:
    # everything downstream works on the stock's own age grid.
    tax_on_deferred_gain_baseline = baseline_pass.marginal_tax_on_deferred_gain[:n_stock, :]
    tax_on_deferred_gain_reform = reform_pass.marginal_tax_on_deferred_gain[:n_stock, :]

    return ScheduleBundle(
        baseline_turnover=inputs.baseline_turnover,
        reform_turnover=reform_turnover,
        rate_factor=rate_factor(reform_turnover, inputs.baseline_turnover),
        reform_policy_by_gain_share=reform_pass.policy_by_gain_share[:, :n_stock, :],
        long_run_turnover_scale=long_run_turnover_scale,
        retiming_turnover_shift=retiming_turnover_shift,
        baseline_gain_stock=inputs.baseline_gain_stock,
        baseline_asset_value=inputs.baseline_asset_value,
        value_baseline=baseline_pass.value_of_gain[:n_stock, :],
        value_reform=reform_pass.value_of_gain[:n_stock, :],
        tax_on_deferred_gain_baseline=tax_on_deferred_gain_baseline,
        tax_on_deferred_gain_reform=tax_on_deferred_gain_reform,
        reform_tax_on_deferred_gain_by_gain_share=(
            reform_pass.marginal_tax_by_gain_share[:, :n_stock, :]
        ),
        death_share_forgiven=inputs.death_share_forgiven,
        death_share_routed_to_heirs=inputs.death_share_routed_to_heirs,
        death_share_taxed_at_death=inputs.death_share_taxed_at_death,
        terminal_residual_baseline=baseline_pass.terminal_residual,
        terminal_residual_reform=reform_pass.terminal_residual,
        frozen=False,
        baseline_turnover_fit_gap=baseline_turnover_fit_gap,
        terminal_residual_baseline_resolve=baseline_resolve.terminal_residual,
        terminal_desired_turnover_residual_baseline=(
            baseline_pass.terminal_desired_turnover_residual
        ),
        terminal_desired_turnover_residual_reform=(
            reform_pass.terminal_desired_turnover_residual
        ),
        terminal_desired_turnover_residual_baseline_resolve=(
            baseline_resolve.terminal_desired_turnover_residual
        ),
    )


def solve_frozen_schedule(inputs: ScheduleInputs) -> ScheduleBundle:
    """The schedule with turnover pinned to baseline and no selling choice.

    This is what the static and mechanical worlds read.  Freezing the pace of
    selling switches off every behavioral margin by construction, and what
    survives is the policy's mechanical content: under carryover the routed
    death stock accumulates in the stock of gains and is realized at the heir
    cell's baseline pace, and under deemed realization the appliers' death term
    delivers the deemed tax on the baseline stock.

    Both priced taxes on a deferred dollar are zero rather than solved, and
    ``frozen`` is set to True so a consumer reading them can refuse.
    """
    inputs.validate()

    zeros = jnp.zeros_like(inputs.baseline_gain_stock)
    ones = jnp.ones_like(inputs.baseline_gain_stock)

    # A policy that does not vary with the gain share: whatever share a cell
    # turns out to carry, it sells at the baseline pace.
    frozen_policy = jnp.repeat(
        inputs.baseline_turnover.T[:, :, None], GAIN_SHARE_GRID_POINTS, axis=2
    )

    return ScheduleBundle(
        baseline_turnover=inputs.baseline_turnover,
        reform_turnover=inputs.baseline_turnover,
        rate_factor=ones,
        reform_policy_by_gain_share=frozen_policy,
        long_run_turnover_scale=ones,
        retiming_turnover_shift=zeros,
        baseline_gain_stock=inputs.baseline_gain_stock,
        baseline_asset_value=inputs.baseline_asset_value,
        value_baseline=zeros,
        value_reform=zeros,
        tax_on_deferred_gain_baseline=zeros,
        tax_on_deferred_gain_reform=zeros,
        reform_tax_on_deferred_gain_by_gain_share=jnp.zeros_like(frozen_policy),
        death_share_forgiven=inputs.death_share_forgiven,
        death_share_routed_to_heirs=inputs.death_share_routed_to_heirs,
        death_share_taxed_at_death=inputs.death_share_taxed_at_death,
        terminal_residual_baseline=jnp.asarray(0.0, dtype=zeros.dtype),
        terminal_residual_reform=jnp.asarray(0.0, dtype=zeros.dtype),
        terminal_residual_baseline_resolve=jnp.asarray(0.0, dtype=zeros.dtype),
        terminal_desired_turnover_residual_baseline=jnp.asarray(
            0.0, dtype=zeros.dtype
        ),
        terminal_desired_turnover_residual_reform=jnp.asarray(0.0, dtype=zeros.dtype),
        terminal_desired_turnover_residual_baseline_resolve=jnp.asarray(
            0.0, dtype=zeros.dtype
        ),
        frozen=True,
        baseline_turnover_fit_gap=jnp.asarray(0.0, dtype=zeros.dtype),
    )


# -----------------------------------------------------------------------------
# The checks a caller runs after the solve
# -----------------------------------------------------------------------------


class ScheduleCheckFailure(Exception):
    """A solved schedule failed one of the checks run after the solve."""


def check_schedule(bundle: ScheduleBundle, inputs: ScheduleInputs) -> None:
    """Refuse a schedule that failed one of five plausibility checks.

    A compiled program cannot raise, so the solve writes NaN and the caller
    runs these checks afterwards:

    1. No turnover came out non-finite.
    2. The baseline pass reproduced the observed baseline turnover, within
       the fit tolerance, in every cell and year.
    3. The terminal year's stationary solve settled, on the baseline pass,
       the reform pass and the baseline re-solve.  The last sweep may still be
       moving the value of holding a gain by at most
       ``TERMINAL_RESIDUAL_TOLERANCE``, and the last refit round may still be
       moving the desired turnover by at most the same.
    4. The priced tax on a deferred dollar is plausible: never negative, and
       never much above the highest marginal rate in any cell.  A dollar of
       gain is taxed at most once, at that rate, so anything materially above
       it is broken arithmetic.  The 5% of slack covers the numerical
       difference the priced tax is read as.
    5. The value of holding a dollar of unrealized gain is worth at most a
       dollar, on both legs.  Deferral can only reduce a bill that starts at
       the marginal rate, so the value is bounded near that rate; measured
       maxima sit near 0.23.  A GPU compiler bug has produced values near 1e8,
       which this check refuses.

    Raises ``ScheduleCheckFailure``.  Host-side, so it runs outside any
    compiled region.
    """
    for name in ("reform_turnover", "rate_factor"):
        values = getattr(bundle, name)
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ScheduleCheckFailure(
                f"The solved schedule's {name} has non-finite values."
            )

    gap = float(jnp.max(jnp.abs(jnp.asarray(bundle.baseline_turnover_fit_gap))))
    if not (gap <= BASELINE_TURNOVER_FIT_TOLERANCE):
        raise ScheduleCheckFailure(
            f"The baseline pass left a cell selling {gap:.4g} away from its "
            "observed baseline turnover, above the fit tolerance of "
            f"{BASELINE_TURNOVER_FIT_TOLERANCE}. The baseline pass fits the "
            "desired turnover so that every cell reproduces the pace it is "
            "observed to sell at, and the reform response is read against "
            "that fit."
        )

    for solve, quantity, residual in (
        ("baseline pass", "value of holding a gain", bundle.terminal_residual_baseline),
        ("reform pass", "value of holding a gain", bundle.terminal_residual_reform),
        (
            "baseline re-solve",
            "value of holding a gain",
            bundle.terminal_residual_baseline_resolve,
        ),
        (
            "baseline pass",
            "desired turnover",
            bundle.terminal_desired_turnover_residual_baseline,
        ),
        (
            "reform pass",
            "desired turnover",
            bundle.terminal_desired_turnover_residual_reform,
        ),
        (
            "baseline re-solve",
            "desired turnover",
            bundle.terminal_desired_turnover_residual_baseline_resolve,
        ),
    ):
        worst = float(jnp.max(jnp.abs(jnp.asarray(residual))))
        if not (worst <= TERMINAL_RESIDUAL_TOLERANCE):
            raise ScheduleCheckFailure(
                f"The {solve}'s terminal year had not settled: the last "
                f"round still moved the {quantity} by {worst:.4g}, above the "
                f"tolerance of {TERMINAL_RESIDUAL_TOLERANCE}. Past the last "
                "simulation year the model holds that year's inputs constant "
                "forever, so the terminal year has to be a fixed point."
            )

    for leg in ("value_baseline", "value_reform"):
        worst = float(jnp.nanmax(jnp.abs(getattr(bundle, leg))))
        if worst > VALUE_OF_HOLDING_CEILING:
            raise ScheduleCheckFailure(
                f"The solved schedule's {leg} reaches {worst:.4g} per dollar "
                "of unrealized gain. A dollar of gain can never be worth "
                "more than a dollar — deferral only reduces a tax bill that "
                "starts at the rate — so numbers past the ceiling of "
                f"{VALUE_OF_HOLDING_CEILING} mean broken arithmetic, not "
                "strong lock-in."
            )

    _check_tax_on_deferred_gain(
        bundle.tax_on_deferred_gain_reform,
        float(jnp.max(inputs.marginal_rate_reform)),
        "reform",
    )
    _check_tax_on_deferred_gain(
        bundle.reform_tax_on_deferred_gain_by_gain_share,
        float(jnp.max(inputs.marginal_rate_reform)),
        "reform (across the gain-share grid)",
    )
    _check_tax_on_deferred_gain(
        bundle.tax_on_deferred_gain_baseline,
        float(jnp.max(inputs.marginal_rate_baseline)),
        "baseline",
    )


# How far a cell's chosen baseline turnover may sit from the turnover it was
# observed to sell at. It is the tolerance the baseline fit works to.
BASELINE_TURNOVER_FIT_TOLERANCE = 1e-5

# The ceiling on the value of holding a dollar of unrealized gain, both legs.
# The economic bound is roughly the highest tax rate the gain might escape,
# about 0.4 with every wedge stacked.
VALUE_OF_HOLDING_CEILING = 1.0

# How far above the highest marginal rate the priced tax may sit before the
# check fails.
DEFERRED_TAX_OVERLAP_ALLOWANCE = 0.05

# Slack for the solve's own floating-point error.
DEFERRED_TAX_NUMERICAL_TOLERANCE = 1e-9


def _check_tax_on_deferred_gain(values: Array, highest_marginal_rate: float, leg: str) -> None:
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ScheduleCheckFailure(
            f"The {leg} priced tax on a deferred dollar has non-finite values."
        )
    smallest = float(jnp.min(values))
    if smallest < -DEFERRED_TAX_NUMERICAL_TOLERANCE:
        raise ScheduleCheckFailure(
            f"The {leg} priced tax on a deferred dollar is negative somewhere "
            f"(smallest {smallest})."
        )
    cap = (
        highest_marginal_rate * (1.0 + DEFERRED_TAX_OVERLAP_ALLOWANCE)
        + DEFERRED_TAX_NUMERICAL_TOLERANCE
    )
    largest = float(jnp.max(values))
    if largest > cap:
        raise ScheduleCheckFailure(
            f"The {leg} priced tax on a deferred dollar reaches {largest}, above "
            f"the highest marginal rate of {highest_marginal_rate} plus the "
            "allowance."
        )
