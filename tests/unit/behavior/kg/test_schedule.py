"""The capital-gains schedule solver, on a five-age four-year toy economy.

The fixtures are made up: five age cells and four years, with round numbers,
so that each check below is about one property of the solver.

Each test states the property in its name and checks that one thing.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.behavior.kg import (
    BASELINE_TURNOVER_FIT_TOLERANCE,
    Calibration,
    ScheduleCheckFailure,
    ScheduleInputs,
    bellman_age,
    check_schedule,
    solve_frozen_schedule,
    solve_schedule,
    timing,
)
from taxmodel.behavior.kg.inputs import GAIN_SHARE_GRID_POINTS

# Five age cells standing in for 18 through 80, and two more standing in for
# the tail out to 119 that only the selling choice sees.
N_AGES = 5
N_TAIL_AGES = 2
N_EXTENDED_AGES = N_AGES + N_TAIL_AGES
N_YEARS = 4

# Baseline law's rate on long-term gains: 20% plus the 3.8% net investment
# income tax.
BASELINE_GAINS_RATE = 0.238

# The reform in these fixtures raises that rate by five points from the second
# year on, the same size of shock the calibration moments use.
REFORM_RATE_RISE = 0.05

# Nominal discount factor, roughly one over one plus a two per cent ten-year
# Treasury rate.
BETA = 0.978

# Half of each cell's asset value is unrealized gain.
GAIN_SHARE = 0.5

# The appreciation rate in the settings file, read off Tax-Data's accruals.
APPRECIATION = 0.0677


def _by_age_and_year(by_age: list[float], n_years: int = N_YEARS) -> jnp.ndarray:
    """Repeat one column of age values across every year."""
    return jnp.tile(jnp.asarray(by_age, dtype=jnp.float64)[:, None], (1, n_years))


def toy_inputs(
    *,
    reform_rate_rise: float = REFORM_RATE_RISE,
    announced_a_year_early: bool = False,
    carryover_at_death: bool = True,
) -> ScheduleInputs:
    """A small economy the checks below can reason about.

    ``reform_rate_rise`` of zero makes reform law identical to baseline law,
    the setting the "nothing moves" checks use. ``announced_a_year_early``
    shifts the law-only rate rise one year earlier than the rate the selling
    choice sees, as for a reform announced before it takes effect:
    realizations retime into the last year at the lower rate.
    """
    stock = _by_age_and_year([100.0, 200.0, 300.0, 400.0, 500.0])
    asset_value = stock / GAIN_SHARE
    baseline_turnover = _by_age_and_year([0.050, 0.045, 0.040, 0.035, 0.030])
    realizations = baseline_turnover * stock

    # Mortality rises with age, and the two tail cells stand in for the very
    # old, where death is close to certain.
    mortality_by_age = [0.002, 0.005, 0.010, 0.030, 0.080]
    gain_weighted_mortality = _by_age_and_year(
        mortality_by_age + [0.200, 0.400], N_YEARS
    )

    charity = _by_age_and_year([0.005, 0.005, 0.010, 0.020, 0.040])

    baseline_tax_rate = jnp.full(
        (N_EXTENDED_AGES, N_YEARS), BASELINE_GAINS_RATE, dtype=jnp.float64
    )
    # The rise takes effect in the second year and stays.
    takes_effect = (jnp.arange(N_YEARS) >= 1).astype(jnp.float64)
    reform_tax_rate = baseline_tax_rate + reform_rate_rise * takes_effect[None, :]
    if announced_a_year_early:
        announced = (jnp.arange(N_YEARS) >= 2).astype(jnp.float64)
        law_only_rate = baseline_tax_rate + reform_rate_rise * announced[None, :]
    else:
        law_only_rate = reform_tax_rate

    zeros_by_age = jnp.zeros((N_AGES, N_YEARS), dtype=jnp.float64)

    if carryover_at_death and reform_rate_rise != 0.0:
        # Half of each cell's gains keep their basis and go to the heirs; the
        # rest still get a stepped-up basis.
        routed = jnp.full((N_AGES, N_YEARS), 0.5, dtype=jnp.float64)
        forgiven = 1.0 - routed
        taxed_at_death = zeros_by_age
    else:
        routed = zeros_by_age
        forgiven = jnp.ones((N_AGES, N_YEARS), dtype=jnp.float64)
        taxed_at_death = zeros_by_age

    # Each age's stock moves to the next age; the top cell pools and points at
    # itself.
    aging = jnp.eye(N_AGES, k=1, dtype=jnp.float64).at[N_AGES - 1, N_AGES - 1].set(1.0)
    # Heirs are the two youngest cells, in equal parts, whoever dies.
    heir_row = jnp.asarray([0.5, 0.5, 0.0, 0.0, 0.0], dtype=jnp.float64)
    heirs = jnp.tile(heir_row[None, :], (N_AGES, 1))

    return ScheduleInputs(
        baseline_gain_stock=stock,
        baseline_realizations=realizations,
        baseline_turnover=baseline_turnover,
        charity_probability=charity,
        baseline_asset_value=asset_value,
        death_share_forgiven=forgiven,
        death_share_routed_to_heirs=routed,
        death_share_taxed_at_death=taxed_at_death,
        gain_weighted_mortality=gain_weighted_mortality,
        marginal_rate_baseline=baseline_tax_rate,
        marginal_rate_reform=reform_tax_rate,
        marginal_rate_reform_law_only=law_only_rate,
        aging_operator=aging,
        heir_operator=heirs,
        discount_factor_by_year=jnp.full((N_YEARS,), BETA, dtype=jnp.float64),
    )


def toy_calibration(**overrides) -> Calibration:
    """Made-up values in the shape of the real calibration.

    Nothing in this file asserts a level that depends on them.
    """
    base = {
        "elasticity": jnp.asarray(1.6342),
        "timeable_share": jnp.asarray(0.2452),
        "appreciation": APPRECIATION,
        "timing_reference_wedge": jnp.asarray(0.05),
        "timing_window": 1,
    }
    base.update(overrides)
    return Calibration(**base)


# -----------------------------------------------------------------------------
# Pass 1: recovering the benefit of realizing
# -----------------------------------------------------------------------------


def test_the_baseline_fit_gap_is_small_and_a_large_one_stops_the_run():
    """The baseline pass fits the desired turnover so that each cell chooses
    the pace it is observed to sell at.  The bundle carries the largest gap
    the fit left, and a gap above the fit tolerance stops the run."""
    inputs = toy_inputs()
    bundle = solve_schedule(inputs, toy_calibration())

    assert float(bundle.baseline_turnover_fit_gap) <= BASELINE_TURNOVER_FIT_TOLERANCE
    check_schedule(bundle, inputs)

    missed = bundle._replace(baseline_turnover_fit_gap=jnp.asarray(1e-3))
    with pytest.raises(ScheduleCheckFailure, match="observed baseline turnover"):
        check_schedule(missed, inputs)


def test_the_policy_schedule_read_at_the_baseline_share_is_the_bundle_turnover():
    """The two readings agree where the ledger holds its baseline stock.

    The bundle carries the reform pass's turnover at every gain share on the
    solver's grid, because the ledger reads it at the share its cells actually
    carry. Read instead at the baseline gain share, scaled and with the
    retiming added, it has to come back as the bundle's own
    ``reform_turnover``.
    """
    inputs = toy_inputs()
    bundle = solve_schedule(inputs, toy_calibration())

    n_ages, n_years = inputs.baseline_turnover.shape
    assert bundle.reform_policy_by_gain_share.shape == (
        n_years,
        n_ages,
        GAIN_SHARE_GRID_POINTS,
    )

    for year in range(n_years):
        at_baseline = bellman_age.at_gain_share(
            bundle.reform_policy_by_gain_share[year],
            jnp.full((n_ages,), GAIN_SHARE),
        )
        rebuilt = np.clip(
            np.asarray(bundle.long_run_turnover_scale)[:, year] * np.asarray(at_baseline)
            + np.asarray(bundle.retiming_turnover_shift)[:, year],
            0.0,
            1.0,
        )
        np.testing.assert_allclose(
            rebuilt,
            np.asarray(bundle.reform_turnover)[:, year],
            rtol=0.0,
            atol=1e-15,
        )


def test_the_solver_refuses_an_asset_value_of_the_wrong_shape():
    """The cells' asset value is required, on the stock's own age grid."""
    inputs = toy_inputs()
    broken = dataclasses.replace(
        inputs, baseline_asset_value=inputs.baseline_asset_value[:-1, :]
    )
    with pytest.raises(ValueError, match="baseline_asset_value"):
        solve_schedule(broken, toy_calibration())


# -----------------------------------------------------------------------------
# Pass 2: the reform's turnover
# -----------------------------------------------------------------------------


def test_reform_identical_to_baseline_leaves_every_rate_alone():
    """A reform that changes no law must leave turnover exactly where the
    baseline put it, because the non-tax value of selling was recovered to make
    that pace optimal in the first place."""
    inputs = toy_inputs(reform_rate_rise=0.0)
    bundle = solve_schedule(inputs, toy_calibration())
    check_schedule(bundle, inputs)

    np.testing.assert_allclose(
        np.asarray(bundle.rate_factor), 1.0, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(bundle.reform_turnover),
        np.asarray(inputs.baseline_turnover),
        rtol=0.0,
        atol=1e-12,
    )


def test_raising_the_gains_rate_lowers_realizations():
    """Realizing costs more when the rate is higher, so cells realize less of
    their stock."""
    inputs = toy_inputs()
    bundle = solve_schedule(inputs, toy_calibration())
    check_schedule(bundle, inputs)

    # Read the years in which the higher rate is in force.
    in_force = np.asarray(bundle.rate_factor)[:, 1:]
    assert np.all(in_force < 1.0)


def test_a_terminal_year_that_has_not_settled_stops_the_run():
    """Past the last simulation year the model holds that year's inputs
    constant forever, so the terminal year has to be a fixed point in both the
    value of holding a gain and the fitted desired turnover.  The check covers
    all three solves: the baseline pass, the reform pass, and the baseline
    re-solve the reform's turnover is measured against."""
    inputs = toy_inputs()
    bundle = solve_schedule(inputs, toy_calibration())
    check_schedule(bundle, inputs)

    for name in (
        "terminal_residual_baseline",
        "terminal_residual_reform",
        "terminal_residual_baseline_resolve",
        "terminal_desired_turnover_residual_baseline",
        "terminal_desired_turnover_residual_reform",
        "terminal_desired_turnover_residual_baseline_resolve",
    ):
        unsettled = bundle._replace(**{name: jnp.asarray(1e-3)})
        with pytest.raises(ScheduleCheckFailure, match="had not settled"):
            check_schedule(unsettled, inputs)


def test_death_shares_that_do_not_add_to_one_stop_the_run():
    """Every dollar of a decedent's gain is stepped up, carried to the heirs,
    or taxed at death, so the three shares have to account for the whole
    dollar."""
    inputs = toy_inputs()
    broken = dataclasses.replace(
        inputs,
        death_share_routed_to_heirs=inputs.death_share_routed_to_heirs + 0.1,
    )
    with pytest.raises(ValueError, match="add up to one"):
        solve_schedule(broken, toy_calibration())


# -----------------------------------------------------------------------------
# The retiming overlay
# -----------------------------------------------------------------------------


def _retiming_overlay(inputs):
    """The retiming overlay the solver builds from these inputs."""
    calibration = toy_calibration()
    return timing.build_retiming_overlay(
        baseline_realizations=inputs.baseline_realizations,
        rate_change=(
            inputs.marginal_rate_reform_law_only - inputs.marginal_rate_baseline
        )[:N_AGES, :],
        timeable_share=calibration.timeable_share,
        timing_reference_wedge=calibration.timing_reference_wedge,
        timing_window=calibration.timing_window,
    )


def test_retiming_moves_dollars_without_creating_them():
    """Retiming is a reallocation across years. Summed over the window, the
    dollars after must equal the dollars before, age by age."""
    overlay = _retiming_overlay(toy_inputs(announced_a_year_early=True))

    before = np.asarray(overlay.moveable_baseline).sum(axis=1)
    after = np.asarray(overlay.moveable_reform).sum(axis=1)
    np.testing.assert_allclose(after, before, rtol=0.0, atol=1e-9)


def test_a_reform_announced_early_pulls_realizations_forward():
    """When a rate rise is announced a year before it takes effect, the year
    before has the lower rate, so the moveable bucket moves back into it."""
    overlay = _retiming_overlay(toy_inputs(announced_a_year_early=True))

    shift = np.asarray(overlay.moveable_reform) - np.asarray(
        overlay.moveable_baseline
    )
    # The rate rises in year index 2, so year index 1 is the last year at the
    # lower rate.
    assert np.all(shift[:, 1] > 0.0)
    assert np.all(shift[:, 2] < 0.0)


def test_unchanged_law_retimes_nothing():
    """The retiming comparison is on the change in the rate, not its level, so
    a run in which law does not move must leave every dollar where it was."""
    inputs = toy_inputs(reform_rate_rise=0.0)
    overlay = timing.build_retiming_overlay(
        baseline_realizations=inputs.baseline_realizations,
        rate_change=jnp.zeros_like(inputs.baseline_realizations),
        timeable_share=jnp.asarray(0.2452),
        timing_reference_wedge=jnp.asarray(0.05),
        timing_window=1,
    )
    np.testing.assert_array_equal(
        np.asarray(overlay.net_shift), np.zeros((N_AGES, N_YEARS))
    )


# -----------------------------------------------------------------------------
# The priced tax on a deferred dollar
# -----------------------------------------------------------------------------


def test_priced_tax_stays_between_zero_and_the_top_marginal_rate():
    """A dollar of unrealized gain is taxed at most once, so the present value
    of tax on one more dollar of it cannot be negative and cannot exceed the
    highest marginal rate. ``check_schedule`` stops a run where it does."""
    inputs = toy_inputs()
    bundle = solve_schedule(inputs, toy_calibration())
    check_schedule(bundle, inputs)

    for priced, highest in (
        (bundle.tax_on_deferred_gain_baseline, float(jnp.max(inputs.marginal_rate_baseline))),
        (bundle.tax_on_deferred_gain_reform, float(jnp.max(inputs.marginal_rate_reform))),
    ):
        values = np.asarray(priced)
        assert values.min() >= 0.0
        assert values.max() <= highest


def test_a_reform_that_changes_no_law_prices_a_deferred_dollar_the_same():
    """The priced tax is a present value of tax under one law, so two legs
    facing the same law have to price a deferred dollar alike."""
    inputs = toy_inputs(reform_rate_rise=0.0)
    bundle = solve_schedule(inputs, toy_calibration())

    np.testing.assert_allclose(
        np.asarray(bundle.tax_on_deferred_gain_reform),
        np.asarray(bundle.tax_on_deferred_gain_baseline),
        rtol=0.0,
        atol=1e-9,
    )


def test_a_higher_gains_rate_raises_the_priced_tax_on_a_deferred_dollar():
    """The conversion channel reads the two priced taxes as the price of
    holding a dollar of gain under each law. A reform that taxes gains more
    heavily has to raise it."""
    inputs = toy_inputs()
    bundle = solve_schedule(inputs, toy_calibration())

    # The years in which the higher rate is in force.
    assert np.all(
        np.asarray(bundle.tax_on_deferred_gain_reform)[:, 1:]
        > np.asarray(bundle.tax_on_deferred_gain_baseline)[:, 1:]
    )


# -----------------------------------------------------------------------------
# The frozen pass
# -----------------------------------------------------------------------------


def test_the_frozen_pass_pins_every_rate_to_baseline():
    """The static and mechanical worlds see no behavioral response at all:
    turnover is the baseline turnover, the applier's multiplier is one
    everywhere, and the fit leaves no gap because nothing is fitted."""
    inputs = toy_inputs()
    bundle = solve_frozen_schedule(inputs)

    assert bundle.frozen
    np.testing.assert_array_equal(
        np.asarray(bundle.reform_turnover),
        np.asarray(inputs.baseline_turnover),
    )
    np.testing.assert_array_equal(
        np.asarray(bundle.rate_factor), np.ones((N_AGES, N_YEARS))
    )
    assert float(bundle.baseline_turnover_fit_gap) == 0.0
    # Carryover still moves gains to heirs.
    np.testing.assert_array_equal(
        np.asarray(bundle.death_share_routed_to_heirs),
        np.asarray(inputs.death_share_routed_to_heirs),
    )


# -----------------------------------------------------------------------------
# Response to the calibrated parameters
# -----------------------------------------------------------------------------


def test_the_solve_responds_to_both_calibrated_parameters():
    """The calibrate command fits the elasticity and the timeable share by
    finite differences through this solver, so a small move in either has to
    move the reform's realizations by a finite amount."""
    inputs = toy_inputs()

    def total_reform_realizations(elasticity, timeable_share):
        calibration = toy_calibration(
            elasticity=elasticity, timeable_share=timeable_share
        )
        bundle = solve_schedule(inputs, calibration)
        return float(jnp.sum(bundle.reform_turnover * inputs.baseline_gain_stock))

    toy = toy_calibration()
    elasticity = float(toy.elasticity)
    timeable_share = float(toy.timeable_share)
    at = total_reform_realizations(elasticity, timeable_share)
    slope_elasticity = (
        total_reform_realizations(elasticity * 1.01, timeable_share) - at
    ) / (elasticity * 0.01)
    slope_timeable = (
        total_reform_realizations(elasticity, timeable_share * 1.01) - at
    ) / (timeable_share * 0.01)
    assert np.isfinite(slope_elasticity) and np.isfinite(slope_timeable)
    # A larger elasticity means a larger response to a rate rise, so total
    # realizations fall.
    assert slope_elasticity < 0.0


def test_an_impossible_value_of_holding_stops_the_run():
    """A dollar of gain worth more than a dollar is broken arithmetic.

    The one observed cause is a GPU compiler bug. The check stops the run
    rather than scoring the values it produces.
    """
    inputs = toy_inputs()
    bundle = solve_schedule(inputs, toy_calibration())
    check_schedule(bundle, inputs)

    poisoned = bundle._replace(
        value_reform=bundle.value_reform.at[0, 0].set(4.0e8)
    )
    with pytest.raises(ScheduleCheckFailure, match="per dollar"):
        check_schedule(poisoned, inputs)
