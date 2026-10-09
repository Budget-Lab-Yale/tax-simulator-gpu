"""Measuring the gains model's two moments, on a toy economy.

The real measurement runs on the whole population priced under current
law. These tests run the same code on the five-age economy the gains
model's own tests use, stretched to the thirty-one years of the calibration
window, so that every step between "here is a baseline economy" and "here
are two elasticities Newton can differentiate" is exercised.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

# The gains model's own test economy. tests/unit is the directory pytest
# puts on the path, so these are plain imports from the sibling package.
from behavior.kg.test_schedule import toy_inputs
from behavior.kg.test_solver_to_stepper import baseline_cells_from

from taxmodel.calibration import baseline_economy, kg_moments, newton, targets

# A parameter point to evaluate the moments at.
TEST_ELASTICITY = 1.6342
TEST_TIMEABLE_SHARE = 0.2452

# The calibration window: the year it opens plus the thirty after it.
WINDOW_YEARS = targets.CALIBRATION_WINDOW_YEARS + 1


def long_baseline_economy(n_years: int = WINDOW_YEARS):
    """The toy economy under current law, stretched to a full window.

    Every year is a copy of the toy's first year, and reform law equals
    current law. Holding the years
    identical is deliberate: it means anything the moments see moving came
    from the shock this module applies and not from the fixture.
    """

    short = toy_inputs(reform_rate_rise=0.0)

    def stretch(value):
        if value is None:
            return None  # optional inputs the toy economy leaves out
        value = jnp.asarray(value)
        if value.ndim == 2 and value.shape[0] == value.shape[1]:
            return value  # the aging and heir operators are square, not yearly
        if value.ndim == 2:
            return jnp.tile(value[:, :1], (1, n_years))
        return jnp.tile(value[:1], (n_years,))

    stretched = {
        field.name: stretch(getattr(short, field.name))
        for field in dataclasses.fields(short)
    }
    stretched["aging_operator"] = short.aging_operator
    stretched["heir_operator"] = short.heir_operator
    return dataclasses.replace(short, **stretched)


def toy_scenarios(**overrides) -> kg_moments.MomentScenarios:
    """Baseline, an immediate five-point rise, and an announced two-point one."""

    baseline = long_baseline_economy()
    return kg_moments.moment_scenarios_from_baseline(
        baseline, baseline_cells_from(baseline), **overrides
    )


def test_the_builder_raises_the_rate_from_the_third_year():
    """Both legs raise the rate from the third year of the window.

    The first year carries no change and the second is the announcement year.
    """

    scenarios = toy_scenarios()
    baseline_rate = scenarios.permanent_shock.marginal_rate_baseline
    permanent_rate = scenarios.permanent_shock.marginal_rate_reform

    rise = np.asarray(permanent_rate - baseline_rate)[0]
    assert rise[0] == pytest.approx(0.0)
    assert rise[1] == pytest.approx(0.0)
    assert rise[2] == pytest.approx(targets.PERMANENT_SHOCK_SIZE)
    assert rise[-1] == pytest.approx(targets.PERMANENT_SHOCK_SIZE)

    assert scenarios.announcement_year_index == 1
    assert scenarios.effective_year_index == 2
    assert scenarios.permanent_year_index == targets.PERMANENT_HORIZON_YEARS


def test_a_window_too_short_for_the_long_run_reading_stops():
    baseline = long_baseline_economy(n_years=targets.PERMANENT_HORIZON_YEARS)
    with pytest.raises(ValueError, match="years after the window opens"):
        kg_moments.moment_scenarios_from_baseline(
            baseline, baseline_cells_from(baseline)
        )


def test_an_economy_that_already_has_a_reform_in_it_stops():
    """The calibration measures against current law, so it must be given it."""

    baseline = long_baseline_economy()
    already_reformed = dataclasses.replace(
        baseline,
        marginal_rate_reform=baseline.marginal_rate_baseline + 0.05,
    )
    with pytest.raises(ValueError, match="baseline economy"):
        kg_moments.moment_scenarios_from_baseline(
            already_reformed, baseline_cells_from(baseline)
        )


def test_the_two_moments_have_the_signs_the_targets_have():
    scenarios = toy_scenarios()
    moments = np.asarray(
        kg_moments.measure_moments(
            scenarios, TEST_ELASTICITY, TEST_TIMEABLE_SHARE
        )
    )

    assert moments.shape == (2,)
    assert np.all(np.isfinite(moments))
    # A permanent rise in the rate on gains lowers long-run realizations, so
    # the permanent moment is negative like the -0.6 target. A rise people
    # see coming raises realizations in the year before it, so the transitory
    # moment is positive like the +1.2 target.
    assert moments[0] < 0.0
    assert moments[1] > 0.0


def test_a_bigger_timeable_share_pulls_more_forward_in_the_announcement_year():
    scenarios = toy_scenarios()

    def announcement_moment(share):
        return kg_moments.measure_moments(scenarios, TEST_ELASTICITY, share)[1]

    small = float(announcement_moment(jnp.asarray(0.10)))
    large = float(announcement_moment(jnp.asarray(0.40)))

    # The rate rises next year, so this year is the cheap year to sell in:
    # a larger timeable bucket means more realizations pulled into it.
    assert large > small


def test_the_long_run_moment_barely_notices_the_timeable_share():
    """The timeable share leaves the long-run moment almost unchanged.

    Retiming moves dollars between adjacent years and nets to zero over a
    permanent change, so ten years out the timeable share should leave
    the long-run moment where it was. That property is what makes the two
    targets separately identified: the slope is read off the long-run
    moment alone, and the share off the short-run one given the slope.
    """

    scenarios = toy_scenarios()

    def permanent_moment(share):
        return float(
            kg_moments.measure_moments(scenarios, TEST_ELASTICITY, share)[0]
        )

    low = permanent_moment(0.15)
    high = permanent_moment(0.40)
    assert abs(high - low) < 0.05 * abs(low)


def test_dividing_by_the_measured_rate_change_makes_shock_size_almost_cancel():
    """A five-point rise on half the cells reads about the same as on all.

    This is why the moment divides by the average rate change the realized
    dollars faced rather than by the statutory move: a shock that raises
    the rate on only some cells and one that raises it on all still give
    nearly the same long-run number.
    """

    baseline = long_baseline_economy()
    cells = baseline_cells_from(baseline)

    everyone = kg_moments.moment_scenarios_from_baseline(baseline, cells)

    # The same five points, but only on the three oldest cells — the ones
    # holding most of the stock, standing in for a top-bracket-only reform.
    ages = jnp.arange(baseline.marginal_rate_baseline.shape[0])[:, None]
    years = jnp.arange(baseline.marginal_rate_baseline.shape[1])[None, :]
    top_only = baseline.marginal_rate_baseline + jnp.where(
        (ages >= 2) & (years >= kg_moments.EFFECTIVE_YEAR_OFFSET),
        targets.PERMANENT_SHOCK_SIZE,
        0.0,
    )
    narrow = kg_moments.moment_scenarios_from_baseline(
        baseline, cells, permanent_rate_on_gains=top_only
    )

    broad_moment = float(
        kg_moments.measure_moments(
            everyone, TEST_ELASTICITY, TEST_TIMEABLE_SHARE
        )[0]
    )
    narrow_moment = float(
        kg_moments.measure_moments(
            narrow, TEST_ELASTICITY, TEST_TIMEABLE_SHARE
        )[0]
    )
    assert narrow_moment == pytest.approx(broad_moment, rel=0.15)


def test_the_measurement_has_an_invertible_finite_difference_sensitivity():
    """The 2x2 sensitivity matrix Newton uses comes from moving each parameter a little.

    The selling choice is found by bisection, so the sensitivities are central
    finite differences rather than derivatives.
    """

    scenarios = toy_scenarios()
    evaluate = kg_moments.moment_evaluation(scenarios)
    at_the_test_point = np.asarray([TEST_ELASTICITY, TEST_TIMEABLE_SHARE])

    sensitivity = np.empty((2, 2))
    for column in range(2):
        step = abs(at_the_test_point[column]) * 1e-3
        up, down = at_the_test_point.copy(), at_the_test_point.copy()
        up[column] += step
        down[column] -= step
        higher = np.asarray(evaluate(jnp.asarray(up)), dtype=float)
        lower = np.asarray(evaluate(jnp.asarray(down)), dtype=float)
        sensitivity[:, column] = (higher - lower) / (2.0 * step)

    assert np.all(np.isfinite(sensitivity))
    # Newton needs an invertible matrix to take a step.
    assert abs(float(np.linalg.det(sensitivity))) > 0.0
    # Nearly triangular: the long-run moment hardly moves with the timeable
    # share, since retiming nets to zero over a permanent rate change.
    assert abs(sensitivity[0, 1]) < 0.05 * abs(sensitivity[0, 0])


def test_newton_reaches_targets_the_toy_economy_can_actually_produce():
    """Aim at moments this toy model does produce, and check we get back.

    The declared -0.6 and +1.2 are targets for the real population; a
    five-cell toy cannot reach them, and asking it to would test the
    fixture rather than the solver. So take the moments the toy gives at a
    known pair of parameters and check Newton recovers that pair.
    """

    scenarios = toy_scenarios()
    evaluate = kg_moments.moment_evaluation(scenarios)
    answer = jnp.asarray([1.40, 0.30])
    reachable = tuple(float(value) for value in evaluate(answer))

    result = newton.solve(
        evaluate,
        targets=reachable,
        start=(TEST_ELASTICITY, TEST_TIMEABLE_SHARE),
        settings=newton.NewtonSettings(tolerance=1e-6, max_iterations=12),
        parameter_names=targets.PARAMETER_NAMES,
        moment_names=targets.MOMENT_NAMES,
        sensible_ranges=((0.0, 20.0), (0.0, 1.0)),
    )

    # The selling choice is found by bisection, so the moments carry noise of
    # the order of the fit tolerance; the recovered pair is checked to that.
    assert result.parameters[0] == pytest.approx(1.40, abs=1e-3)
    assert result.parameters[1] == pytest.approx(0.30, abs=1e-3)
    assert result.iterations <= 8


def test_the_window_is_the_opening_year_plus_thirty():
    """The window is thirty-one years long and the long-run moment is read
    ten years in."""

    years = baseline_economy.window_years(2025)

    assert years[0] == 2025
    assert len(years) == targets.CALIBRATION_WINDOW_YEARS + 1
    assert years[-1] == 2055
    assert years[targets.PERMANENT_HORIZON_YEARS] == 2035
