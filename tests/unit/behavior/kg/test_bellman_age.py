"""The age and gain-share selling choice.

A hand-sized economy of five ages and four years, with the first-order
condition restated here in NumPy and checked against what the JAX solver
chose.
"""

from __future__ import annotations

import numpy as np

from taxmodel.behavior.kg.bellman_age import (
    AgeGainInputs,
    at_gain_share,
    saving_profile,
    solve_selling_choice,
)

N_AGES = 5
N_YEARS = 4
GRID_POINTS = 101
ETA = 5.5
APPRECIATION = 0.0677
DISCOUNT_FACTOR = 1.0 / 1.04


def _matrix(value):
    """One number, or one number per age, spread over every year."""
    return np.broadcast_to(
        np.asarray(value, dtype=np.float64).reshape(-1, 1), (N_AGES, N_YEARS)
    ).copy()


def _economy(**changes) -> AgeGainInputs:
    """Five ages and four years: gains mostly held by the young, all stepped up.

    The gain share falls with age and baseline selling slows with age, which
    is what makes net new saving positive at every age.
    """
    gain_share = _matrix(np.linspace(0.55, 0.45, N_AGES))
    turnover = _matrix(np.linspace(0.09, 0.05, N_AGES))
    fields = {
        "mortality": _matrix(np.linspace(0.01, 0.06, N_AGES)),
        "tax_rate": _matrix(0.25),
        "death_share_forgiven": _matrix(1.0),
        "death_share_routed_to_heirs": _matrix(0.0),
        "death_share_taxed_at_death": _matrix(0.0),
        "charity_probability": _matrix(0.05),
        "baseline_gain_share": gain_share,
        "baseline_turnover": turnover,
        "heir_operator": np.full((N_AGES, N_AGES), 1.0 / N_AGES),
        "discount_factor_by_year": np.full(N_YEARS, DISCOUNT_FACTOR),
        "appreciation": APPRECIATION,
        "saving_profile": np.asarray(
            saving_profile(gain_share, turnover, APPRECIATION)
        ),
    }
    fields.update(changes)
    return AgeGainInputs(**fields)


# -----------------------------------------------------------------------------
# The first-order condition, restated
# -----------------------------------------------------------------------------


def _interpolate(row, point):
    """Linear interpolation of one row of the value function at one gain share."""
    grid = np.linspace(0.0, 1.0, len(row))
    return float(np.interp(np.clip(point, 0.0, 1.0), grid, row))


def _slope(row, point):
    """The derivative of that interpolation, constant within a grid interval."""
    step = 1.0 / (len(row) - 1)
    index = min(int(np.clip(point, 0.0, 1.0) / step), len(row) - 2)
    return (row[index + 1] - row[index]) / step


def _turnover_the_condition_asks_for(
    inputs: AgeGainInputs, next_value, age, year, gain_share, desired, turnover
):
    """s0 (1 - tau g + dF/ds)^eta, written out from the model's own arrays.

    ``next_value`` is next year's value function, ``[n_ages, n_grid]``,
    measured against a clean portfolio of the same age. F is the discounted
    continuation, survivor branch and death branch together, and its
    derivative is taken at the turnover the solver chose. Selling one more
    unit lowers the gain share the decedent carries by the gain share itself,
    which is where the death branch's slope comes from.
    """
    n_ages = N_AGES
    appreciation = inputs.appreciation
    growth = 1.0 + appreciation + inputs.saving_profile[age]
    next_gain = ((1.0 - turnover) * gain_share + appreciation) / growth
    survivor = (
        np.zeros(next_value.shape[1]) if age == n_ages - 1 else next_value[age + 1]
    )
    survivor_weight = (
        inputs.discount_factor_by_year[year]
        * growth
        * (1.0 - inputs.mortality[age, year])
    )
    death_weight = inputs.discount_factor_by_year[year] * inputs.mortality[age, year]

    # A household that dies during the year is taxed under that year's law.
    charity = inputs.charity_probability[age, year]
    routed = inputs.death_share_routed_to_heirs[age, year]
    taxed = inputs.death_share_taxed_at_death[age, year]
    death_rate = inputs.tax_rate[age, year]

    heir_value = np.asarray(inputs.heir_operator)[age] @ next_value
    gain_share_at_death = (1.0 - turnover) * gain_share
    carried_slope = _slope(
        heir_value, (gain_share_at_death + appreciation) / (1.0 + appreciation)
    )
    death_slope = (
        gain_share * (1.0 - charity) * (taxed * death_rate - routed * carried_slope)
    )

    continuation_slope = (
        -survivor_weight * gain_share / growth * _slope(survivor, next_gain)
        + death_weight * death_slope
    )
    base = 1.0 - inputs.tax_rate[age, year] * gain_share + continuation_slope
    return desired * max(base, 0.0) ** ETA


def _check_first_order_condition(inputs, solution, value_by_year):
    """Every cell sells what the first-order condition asks for.

    The check runs over the whole gain grid, where the policy is solved,
    rather than at the baseline gain share, where the reported turnover is an
    interpolation between two solved points. Cells that sell nothing or
    everything are corners and have no first-order condition.
    """
    import jax.numpy as jnp

    from taxmodel.behavior.kg.bellman_age import (
        _year_column,
        _year_columns,
        _year_layer,
    )

    desired = np.asarray(solution.desired_turnover)
    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    columns = _year_columns(inputs)
    worst = 0.0
    for year in range(N_YEARS - 1):
        _, policy = _year_layer(
            jnp.asarray(value_by_year[year + 1]),
            _year_column(columns, year),
            jnp.asarray(desired[:, year]),
            ETA,
            jnp.asarray(inputs.saving_profile),
            inputs.appreciation,
            jnp.asarray(inputs.heir_operator),
            GRID_POINTS,
        )
        policy = np.asarray(policy)
        for age in range(N_AGES):
            for point, gain_share in enumerate(grid):
                chosen = policy[age, point]
                if chosen <= 0.0 or chosen >= 1.0:
                    continue
                asked = _turnover_the_condition_asks_for(
                    inputs,
                    value_by_year[year + 1],
                    age,
                    year,
                    gain_share,
                    desired[age, year],
                    chosen,
                )
                worst = max(worst, abs(asked - chosen) / chosen)
    assert worst < 1e-4, f"worst relative gap in the first-order condition {worst:.3g}"


def _value_by_year(inputs, eta, desired, grid_points, sweeps):
    """The value function of every year, which the restated condition reads."""
    from taxmodel.behavior.kg.bellman_age import _solve_all_years

    *_, value, _terminal_desired = _solve_all_years(
        inputs, eta, desired, grid_points, sweeps
    )
    return np.asarray(value)


def test_first_order_condition_holds_in_both_passes():
    inputs = _economy()
    baseline = solve_selling_choice(inputs, ETA, gain_grid_points=GRID_POINTS)
    _check_first_order_condition(
        inputs, baseline, _value_by_year(inputs, ETA, None, GRID_POINTS, 400)
    )

    reform = _economy(tax_rate=_matrix(0.35))
    solved = solve_selling_choice(
        reform, ETA, baseline.desired_turnover, gain_grid_points=GRID_POINTS
    )
    _check_first_order_condition(
        reform,
        solved,
        _value_by_year(reform, ETA, baseline.desired_turnover, GRID_POINTS, 400),
    )

    # A reform that both taxes gains at death and carries some of them to the
    # heirs, so that the death branch's slope in the turnover is not zero.
    at_death = _economy(
        death_share_forgiven=_matrix(0.0),
        death_share_routed_to_heirs=_matrix(0.5),
        death_share_taxed_at_death=_matrix(0.5),
    )
    solved_at_death = solve_selling_choice(
        at_death, ETA, baseline.desired_turnover, gain_grid_points=GRID_POINTS
    )
    _check_first_order_condition(
        at_death,
        solved_at_death,
        _value_by_year(at_death, ETA, baseline.desired_turnover, GRID_POINTS, 400),
    )


def test_baseline_pass_reproduces_baseline_turnover():
    inputs = _economy()
    solution = solve_selling_choice(inputs, ETA, gain_grid_points=GRID_POINTS)
    error = np.abs(
        np.asarray(solution.turnover) - np.asarray(inputs.baseline_turnover)
    ).max()
    assert error < 1e-6
    assert float(solution.terminal_residual) <= 1e-7
    assert float(solution.terminal_desired_turnover_residual) <= 1e-7
    assert np.all(np.asarray(solution.desired_turnover) > 0.0)


def test_value_of_gain_and_marginal_tax_have_the_right_sign():
    """Carrying a gain is a cost, and the cost rises with the gain share."""
    solution = solve_selling_choice(_economy(), ETA, gain_grid_points=GRID_POINTS)
    assert np.all(np.asarray(solution.value_of_gain) < 0.0)
    assert np.all(np.asarray(solution.marginal_tax_on_deferred_gain) > 0.0)


def test_marginal_tax_read_off_baseline_matches_the_grid():
    """K at a gain share other than the baseline one comes from the K grid.

    Read at the baseline gain share, the grid gives the solution's own K.
    Read 0.3 higher, as a cell holding inherited gain would be, it gives a
    different K, and that K is the central difference of the value function
    one grid step either side of the new share.
    """
    inputs = _economy()
    solution = solve_selling_choice(inputs, ETA, gain_grid_points=GRID_POINTS)
    value = _value_by_year(inputs, ETA, None, GRID_POINTS, 400)
    grid = np.asarray(solution.marginal_tax_by_gain_share)
    baseline_share = np.asarray(inputs.baseline_gain_share)
    step = 1.0 / (GRID_POINTS - 1)

    year = 2
    at_baseline = at_gain_share(grid[year], baseline_share[:, year])
    np.testing.assert_allclose(
        at_baseline,
        np.asarray(solution.marginal_tax_on_deferred_gain)[:, year],
        atol=1e-12,
    )

    shifted = baseline_share[:, year] + 0.3
    assert np.all(shifted + step <= 1.0)
    at_shifted = np.asarray(at_gain_share(grid[year], shifted))
    central_difference = np.array(
        [
            -(
                _interpolate(value[year, age], shifted[age] + step)
                - _interpolate(value[year, age], shifted[age] - step)
            )
            / (2.0 * step)
            for age in range(N_AGES)
        ]
    )
    np.testing.assert_allclose(at_shifted, central_difference, atol=1e-10)
    assert np.min(np.abs(at_shifted - np.asarray(at_baseline))) > 1e-4


# -----------------------------------------------------------------------------
# Closed forms and comparative statics
# -----------------------------------------------------------------------------


def test_certain_death_into_a_clean_heir_gives_the_closed_form():
    """s = s0 (1 - tau g (1 - beta))^eta when death is certain and deemed.

    The household dies for certain during the first year and whatever gain it
    still carries is deemed realized and taxed then, under that year's law and
    at the same rate as a sale. Its heirs are worth nothing at the margin:
    nothing is routed to them, so the gain share they start with does not
    depend on what the household sold. Selling a dollar of gain now costs tau;
    holding it costs tau one year later, which is beta tau. The household is
    indifferent up to that one year of discounting.
    """
    years = 3
    tax_rate = 0.3
    gain_shares = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    desired = 0.2

    def by_year(first, later):
        return np.column_stack(
            [np.full(N_AGES, first)] + [np.full(N_AGES, later)] * (years - 1)
        )

    inputs = AgeGainInputs(
        # Nobody after the first year dies, so the value function in those
        # years does not depend on the gain share: nothing is taxed there.
        mortality=by_year(1.0, 0.0),
        tax_rate=np.full((N_AGES, years), tax_rate),
        death_share_forgiven=np.zeros((N_AGES, years)),
        death_share_routed_to_heirs=np.zeros((N_AGES, years)),
        death_share_taxed_at_death=np.ones((N_AGES, years)),
        charity_probability=np.zeros((N_AGES, years)),
        baseline_gain_share=np.tile(gain_shares[:, None], (1, years)),
        baseline_turnover=np.full((N_AGES, years), 0.05),
        heir_operator=np.full((N_AGES, N_AGES), 1.0 / N_AGES),
        discount_factor_by_year=np.full(years, DISCOUNT_FACTOR),
        appreciation=APPRECIATION,
        saving_profile=np.full(N_AGES, 0.05),
    )
    # The first year sells at a desired pace of 0.2; every later year is held
    # at a pace of effectively zero, so its value is flat.
    desired_turnover = np.column_stack(
        [np.full(N_AGES, desired)] + [np.full(N_AGES, 1e-12)] * (years - 1)
    )
    solution = solve_selling_choice(
        inputs, ETA, desired_turnover, gain_grid_points=GRID_POINTS
    )
    expected = desired * (1.0 - tax_rate * gain_shares * (1.0 - DISCOUNT_FACTOR)) ** ETA
    assert np.allclose(np.asarray(solution.turnover)[:, 0], expected, atol=1e-9)


def test_selling_responds_to_the_tax_rate_and_to_the_treatment_at_death():
    """Higher rates slow selling; taxing or carrying the gain at death speeds it."""
    baseline_inputs = _economy()
    baseline = solve_selling_choice(baseline_inputs, ETA, gain_grid_points=GRID_POINTS)
    fitted = baseline.desired_turnover

    def reform(**changes):
        return np.asarray(
            solve_selling_choice(
                _economy(**changes), ETA, fitted, gain_grid_points=GRID_POINTS
            ).turnover
        )

    step_up = reform()
    higher_rate = reform(tax_rate=_matrix(0.35))
    deemed = reform(
        death_share_forgiven=_matrix(0.0), death_share_taxed_at_death=_matrix(1.0)
    )
    carryover = reform(
        death_share_forgiven=_matrix(0.0), death_share_routed_to_heirs=_matrix(1.0)
    )

    assert np.allclose(
        step_up, np.asarray(baseline_inputs.baseline_turnover), atol=1e-6
    )
    assert np.all(higher_rate < step_up)
    assert np.all(deemed > step_up)
    assert np.all(carryover > step_up)


def test_the_death_branch_is_priced_under_the_law_of_the_year_of_death():
    """A death in the first year is taxed under the first year's law.

    Gains are stepped up in the first year and deemed realized in the second,
    and the household dies for certain in the first. Under a step-up the
    decedent leaves the heir a clean portfolio however much it sold, so the
    death branch does not move with turnover and the household sells
    s0 (1 - tau g)^eta. If the first year's death were priced under the second
    year's law, the deemed tax would make the same household sell
    s0 (1 - tau g (1 - beta))^eta instead.
    """
    years = 2
    tax_rate = 0.3
    gain_shares = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    desired = 0.2

    def by_year(first, later):
        return np.column_stack([np.full(N_AGES, first), np.full(N_AGES, later)])

    inputs = AgeGainInputs(
        mortality=by_year(1.0, 0.0),
        tax_rate=np.full((N_AGES, years), tax_rate),
        death_share_forgiven=by_year(1.0, 0.0),
        death_share_routed_to_heirs=np.zeros((N_AGES, years)),
        death_share_taxed_at_death=by_year(0.0, 1.0),
        charity_probability=np.zeros((N_AGES, years)),
        baseline_gain_share=np.tile(gain_shares[:, None], (1, years)),
        baseline_turnover=np.full((N_AGES, years), 0.05),
        heir_operator=np.full((N_AGES, N_AGES), 1.0 / N_AGES),
        discount_factor_by_year=np.full(years, DISCOUNT_FACTOR),
        appreciation=APPRECIATION,
        saving_profile=np.full(N_AGES, 0.05),
    )
    desired_turnover = by_year(desired, 1e-12)
    solution = solve_selling_choice(
        inputs, ETA, desired_turnover, gain_grid_points=GRID_POINTS
    )
    stepped_up = desired * (1.0 - tax_rate * gain_shares) ** ETA
    deemed = desired * (1.0 - tax_rate * gain_shares * (1.0 - DISCOUNT_FACTOR)) ** ETA
    chosen = np.asarray(solution.turnover)[:, 0]
    assert np.allclose(chosen, stepped_up, atol=1e-9)
    assert np.all(np.abs(chosen - deemed) > 1e-4)


# -----------------------------------------------------------------------------
# The saving profile
# -----------------------------------------------------------------------------


def test_saving_profile_keeps_the_household_on_the_fitted_gain_share_curve():
    gain_share = _matrix(np.linspace(0.55, 0.45, N_AGES))
    turnover = _matrix(np.linspace(0.09, 0.05, N_AGES))
    saving = np.asarray(saving_profile(gain_share, turnover, APPRECIATION))
    assert saving.shape == (N_AGES,)
    assert np.all(saving >= 0.0)

    fitted = np.polyval(
        np.polyfit(np.arange(N_AGES, dtype=float), gain_share.mean(axis=1), 3),
        np.arange(N_AGES, dtype=float),
    )
    landed = ((1.0 - turnover.mean(axis=1)[:-1]) * fitted[:-1] + APPRECIATION) / (
        1.0 + APPRECIATION + saving[:-1]
    )
    assert np.allclose(landed, fitted[1:])


def test_saving_profile_floors_saving_at_zero_where_the_gain_share_climbs():
    """A gain share that climbs with age would need net dissaving; saving stops at zero."""
    climbing = _matrix(np.linspace(0.35, 0.55, N_AGES))
    saving = np.asarray(saving_profile(climbing, _matrix(0.05), APPRECIATION))
    assert saving.min() == 0.0
    assert np.all(saving >= 0.0)
