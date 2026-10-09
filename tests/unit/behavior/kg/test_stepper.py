"""The stock recurrence, on a five-cell four-year fixture.

Accounting identities are checked to a cent; claims that a number
is exactly zero or exactly one are checked exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.behavior.kg.schedule import ScheduleBundle
from taxmodel.behavior.kg.stepper import (
    ScheduleYear,
    cell_table_column,
    effective_mortality,
    run_stock_recurrence,
    schedule_by_year,
    step_stock,
    turnover_this_year,
)
from taxmodel.behavior.kg.stock_inputs import BaselineCells
from tests.tolerances import CENT

from .conftest import (
    N_YEARS,
)


def test_mortality_follows_the_dollars_not_the_people(baseline):
    """The oldest holding cell dies at the rate its gains face, not its people.

    The fixture gives cell 3 a headcount mortality of 8% but a gain-weighted
    rate of 2%, because within that cell the people most likely to die hold the
    least.  Using the headcount rate would push nearly four times as much gain
    through death.  A cell holding nothing has no dollars to weight by and
    falls back to the headcount rate.
    """

    rates = np.asarray(effective_mortality(jax.tree.map(lambda x: x[0], baseline)))
    assert rates[3] == 0.020
    assert rates[4] == 0.100


def test_one_step_by_hand(baseline, schedules_carryover, aging, heirs):
    """Every flow in the first year, worked out against the fixture's numbers."""

    year = jax.tree.map(lambda x: x[0], baseline)
    schedules = jax.tree.map(lambda x: x[0], schedule_by_year(schedules_carryover))
    entering = jnp.zeros(baseline.stock.shape[1])

    step = step_stock(
        stock_change_entering=entering,
        baseline=year,
        scenario_turnover=turnover_this_year(entering, schedules)[0],
        routed_to_heirs_share=schedules.death_share_routed_to_heirs,
        conversion_inflow=jnp.zeros_like(entering),
        aging=aging,
        heirs=heirs,
    )

    stock = np.asarray(year.stock)
    baseline_turnover = np.asarray(year.turnover)
    scenario_turnover = 0.8 * baseline_turnover
    mortality = np.asarray(effective_mortality(year))
    charitable = np.asarray(year.charitable_bequest_share)

    # Entering with no deviation, the only thing the survivors contribute is
    # the sale they did not make: the stock times the fall in the rate.
    after_sales = stock * (baseline_turnover - scenario_turnover)
    survivors = (1.0 - mortality) * after_sales

    # Everyone ages one cell, and the top cell keeps its own and takes in the
    # cell below it.  The youngest cell receives nothing, because nobody ages
    # into it.
    from_survivors = np.asarray(step.from_survivors)
    assert from_survivors[0] == 0.0
    assert np.allclose(from_survivors[1:4], survivors[0:3], atol=CENT)
    assert abs(from_survivors[4] - (survivors[3] + survivors[4])) < CENT

    # Half of each decedent's taxable gains carries over to heirs, and every
    # inherited dollar lands in the two youngest cells, three to one.  What a
    # decedent holds is what the year's sales left, so the cell's stock is net
    # of the scenario's turnover.
    held_at_death = (1.0 - scenario_turnover) * stock
    inherited = 0.5 * ((1.0 - charitable) * mortality * held_at_death).sum()
    assert abs(float(step.from_inheritance[0]) - 0.75 * inherited) < CENT
    assert abs(float(step.from_inheritance[1]) - 0.25 * inherited) < CENT
    assert float(step.from_inheritance[2:].sum()) == 0.0


def test_the_cell_table_is_built_from_the_stock_entering_the_year(
    baseline, schedules_carryover, aging, heirs
):
    """The year's table reports the stock on 1 January, not on 31 December.

    Under carryover the deviation grows every year, so reading the wrong end of
    the year would shift the whole response forward by one.
    """

    result = run_stock_recurrence(baseline, schedules_carryover, aging, heirs)
    entering = np.asarray(
        cell_table_column(result.cell_tables, "stock_change_entering")
    )
    leaving = np.asarray(result.stock_change_end_of_year)

    assert np.array_equal(entering[0], np.zeros_like(entering[0]))
    for t in range(1, N_YEARS):
        assert np.allclose(entering[t], leaving[t - 1], atol=CENT)

    # If the table were built from the end of the year instead, the first
    # year's extra realizations would already be nonzero.  They are not.
    extra = np.asarray(cell_table_column(result.cell_tables, "extra_realizations"))
    assert np.array_equal(extra[0], np.zeros_like(extra[0]))
    assert abs(leaving[0]).sum() > 1e6


def test_extra_realizations_and_the_deemed_factor_follow_the_entering_stock(
    baseline, schedules_carryover, aging, heirs
):
    result = run_stock_recurrence(baseline, schedules_carryover, aging, heirs)
    tables = result.cell_tables

    entering = np.asarray(cell_table_column(tables, "stock_change_entering"))
    scenario_turnover = np.asarray(cell_table_column(tables, "scenario_realization_rate"))
    stock = np.asarray(baseline.stock)

    expected_extra = scenario_turnover * np.maximum(entering, -stock)
    assert np.allclose(
        np.asarray(cell_table_column(tables, "extra_realizations")),
        expected_extra,
        atol=CENT,
    )

    deemed = np.asarray(cell_table_column(tables, "deemed_factor"))
    holds_something = stock > 0
    after_sales = (1.0 - scenario_turnover) * (stock + entering)
    assert np.allclose(
        deemed[holds_something],
        (after_sales / np.where(holds_something, stock, 1.0))[holds_something],
        atol=1e-12,
    )
    # A cell holding no gain has nothing to scale, so its factor is one.
    assert np.array_equal(
        deemed[~holds_something], np.ones(int((~holds_something).sum()))
    )


def one_cell_fixture(turnover: float):
    """One age cell, one year: $100 of gain, certain death, no charity.

    The cell's gain-weighted mortality is one, so everyone holding the gain
    dies during the year, and the aging and heir operators both keep the
    dollars in the single cell there is.
    """

    stock = jnp.asarray([[100.0]])
    baseline = BaselineCells(
        stock=stock,
        realizations=turnover * stock,
        turnover=jnp.asarray([[turnover]]),
        mortality_headcount=jnp.asarray([[1.0]]),
        mortality_gain_weighted=stock,
        charitable_bequest_share=jnp.asarray([[0.0]]),
    )
    one = jnp.ones((1, 1))
    zero = jnp.zeros((1, 1))
    schedules = ScheduleBundle(
        baseline_turnover=jnp.asarray([[turnover]]),
        reform_turnover=jnp.asarray([[turnover]]),
        rate_factor=one,
        # The cell sells at the same pace whatever gain share it carries.
        reform_policy_by_gain_share=jnp.full((1, 1, 11), turnover),
        long_run_turnover_scale=one,
        retiming_turnover_shift=zero,
        baseline_gain_stock=stock,
        baseline_asset_value=jnp.asarray([[1_000.0]]),
        value_baseline=zero,
        value_reform=zero,
        tax_on_deferred_gain_baseline=zero,
        tax_on_deferred_gain_reform=zero,
        reform_tax_on_deferred_gain_by_gain_share=jnp.zeros((1, 1, 11)),
        death_share_forgiven=zero,
        # Carryover on everything: the heirs inherit the decedent's basis.
        death_share_routed_to_heirs=one,
        death_share_taxed_at_death=zero,
        terminal_residual_baseline=jnp.asarray(0.0),
        terminal_residual_reform=jnp.asarray(0.0),
        frozen=False,
    )
    return baseline, schedules, jnp.ones((1, 1)), jnp.ones((1, 1))


def test_a_decedent_leaves_what_is_left_after_the_years_sales():
    """Sales come before death: $80 of a $100 cell, not the whole $100.

    The cell holds $100 of gain and sells a fifth of it during the year, and
    everyone in it dies afterwards. What the decedents hold is the $80 the
    sale left, so that is what carryover routes to the heirs, and the factor
    the deemed gain of a decedent is scaled by is 0.8 rather than 1.
    """

    baseline, schedules, aging, heirs = one_cell_fixture(turnover=0.2)
    result = run_stock_recurrence(baseline, schedules, aging, heirs)
    tables = result.cell_tables

    assert float(cell_table_column(tables, "decedent_stock")[0, 0]) == pytest.approx(
        80.0, abs=CENT
    )
    assert float(
        cell_table_column(tables, "taxable_death_stock")[0, 0]
    ) == pytest.approx(80.0, abs=CENT)
    assert float(
        cell_table_column(tables, "stock_change_from_inheritance")[0, 0]
    ) == pytest.approx(80.0, abs=CENT)
    assert float(cell_table_column(tables, "deemed_factor")[0, 0]) == pytest.approx(
        0.8, abs=1e-12
    )


def falling_policy(n_years: int, n_ages: int, grid_points: int = 11):
    """A reform policy that sells more slowly the more gain a cell carries.

    Turnover runs from 10 percent at a portfolio that is all basis down to 2
    percent at one that is all gain, the same in every cell and year. It is the
    shape lock-in has: the more of a sale is taxable gain, the dearer selling
    is.
    """

    grid = jnp.linspace(0.0, 1.0, grid_points)
    return jnp.broadcast_to(0.10 - 0.08 * grid, (n_years, n_ages, grid_points))


def rising_tax(n_years: int, n_ages: int, grid_points: int = 11):
    """Reform law's tax on a deferred dollar, rising with the gain share.

    Runs from 10 percent at a portfolio that is all basis to 30 percent at one
    that is all gain, the same in every cell and year.
    """

    grid = jnp.linspace(0.0, 1.0, grid_points)
    return jnp.broadcast_to(0.10 + 0.20 * grid, (n_years, n_ages, grid_points))


def test_a_cell_holding_inherited_gain_sells_more_slowly():
    """The pace is read at the gain share the cell carries, not the baseline one.

    A cell holding $100 of gain against $1,000 of assets has a gain share of
    0.1 and sells at 9.2 percent. Raise what it holds by $400 without touching
    its assets and its gain share is 0.5, where the same policy sells at 6
    percent: inherited gain makes the heirs sell more slowly.
    """

    schedules = ScheduleYear(
        reform_policy_by_gain_share=falling_policy(1, 1)[0],
        reform_tax_on_deferred_gain_by_gain_share=jnp.zeros((1, 11)),
        long_run_turnover_scale=jnp.ones(1),
        retiming_turnover_shift=jnp.zeros(1),
        baseline_gain_stock=jnp.asarray([100.0]),
        baseline_asset_value=jnp.asarray([1_000.0]),
        tax_on_deferred_gain_baseline=jnp.zeros(1),
        death_share_routed_to_heirs=jnp.zeros(1),
        death_share_forgiven=jnp.zeros(1),
        death_share_taxed_at_death=jnp.zeros(1),
    )

    at_baseline, baseline_share = turnover_this_year(jnp.zeros(1), schedules)
    inherited, inherited_share = turnover_this_year(jnp.asarray([400.0]), schedules)

    assert float(baseline_share[0]) == pytest.approx(0.10)
    assert float(inherited_share[0]) == pytest.approx(0.50)
    assert float(at_baseline[0]) == pytest.approx(0.092)
    assert float(inherited[0]) == pytest.approx(0.060)


def test_carryover_slows_the_heirs_year_by_year():
    """The same reading, through the recurrence rather than on its own.

    One cell, everyone dying each year and every heir landing back in the same
    cell with the decedent's basis. The cell enters the first year holding its
    baseline stock, so it sells at the policy's baseline-share rate; the
    inherited gain it enters the second year with raises its gain share, and it
    sells more slowly.
    """

    n_years = 3
    stock = jnp.full((n_years, 1), 100.0)
    baseline = BaselineCells(
        stock=stock,
        realizations=0.092 * stock,
        turnover=jnp.full((n_years, 1), 0.092),
        mortality_headcount=jnp.full((n_years, 1), 0.5),
        mortality_gain_weighted=0.5 * stock,
        charitable_bequest_share=jnp.zeros((n_years, 1)),
    )
    by_age = jnp.zeros((1, n_years))
    schedules = ScheduleBundle(
        baseline_turnover=jnp.full((1, n_years), 0.092),
        reform_turnover=jnp.full((1, n_years), 0.092),
        rate_factor=jnp.ones((1, n_years)),
        reform_policy_by_gain_share=falling_policy(n_years, 1),
        long_run_turnover_scale=jnp.ones((1, n_years)),
        retiming_turnover_shift=by_age,
        baseline_gain_stock=jnp.full((1, n_years), 100.0),
        baseline_asset_value=jnp.full((1, n_years), 1_000.0),
        value_baseline=by_age,
        value_reform=by_age,
        tax_on_deferred_gain_baseline=by_age,
        tax_on_deferred_gain_reform=by_age,
        reform_tax_on_deferred_gain_by_gain_share=rising_tax(n_years, 1),
        death_share_forgiven=by_age,
        # Carryover on everything: the heirs take the decedent's basis.
        death_share_routed_to_heirs=jnp.ones((1, n_years)),
        death_share_taxed_at_death=by_age,
        terminal_residual_baseline=jnp.asarray(0.0),
        terminal_residual_reform=jnp.asarray(0.0),
        frozen=False,
    )

    result = run_stock_recurrence(
        baseline, schedules, jnp.ones((1, 1)), jnp.ones((1, 1))
    )
    pace = np.asarray(
        cell_table_column(result.cell_tables, "scenario_realization_rate")
    )[:, 0]
    share = np.asarray(cell_table_column(result.cell_tables, "scenario_gain_share"))[
        :, 0
    ]

    assert pace[0] == pytest.approx(0.092)
    assert share[0] == pytest.approx(0.10)
    assert pace[1] < pace[0]
    assert share[1] > share[0]


def test_reform_tax_on_a_deferred_dollar_is_read_at_the_scenario_gain_share():
    """The heirs' tax on a deferred dollar follows their inherited gain.

    Same economy as the test above, with reform law's tax on a deferred
    dollar rising along the gain grid. In the first year the cell holds its
    baseline stock and the column is the grid's value at the baseline share of
    0.1. From the second year the cell holds inherited gain, and the column is
    the grid's value at the higher scenario share, not at 0.1.
    """

    n_years = 3
    stock = jnp.full((n_years, 1), 100.0)
    baseline = BaselineCells(
        stock=stock,
        realizations=0.092 * stock,
        turnover=jnp.full((n_years, 1), 0.092),
        mortality_headcount=jnp.full((n_years, 1), 0.5),
        mortality_gain_weighted=0.5 * stock,
        charitable_bequest_share=jnp.zeros((n_years, 1)),
    )
    by_age = jnp.zeros((1, n_years))
    tax_grid = rising_tax(n_years, 1)
    at_baseline_share = 0.10 + 0.20 * 0.10
    schedules = ScheduleBundle(
        baseline_turnover=jnp.full((1, n_years), 0.092),
        reform_turnover=jnp.full((1, n_years), 0.092),
        rate_factor=jnp.ones((1, n_years)),
        reform_policy_by_gain_share=falling_policy(n_years, 1),
        long_run_turnover_scale=jnp.ones((1, n_years)),
        retiming_turnover_shift=by_age,
        baseline_gain_stock=jnp.full((1, n_years), 100.0),
        baseline_asset_value=jnp.full((1, n_years), 1_000.0),
        value_baseline=by_age,
        value_reform=by_age,
        tax_on_deferred_gain_baseline=jnp.full((1, n_years), 0.15),
        tax_on_deferred_gain_reform=jnp.full((1, n_years), at_baseline_share),
        reform_tax_on_deferred_gain_by_gain_share=tax_grid,
        death_share_forgiven=by_age,
        death_share_routed_to_heirs=jnp.ones((1, n_years)),
        death_share_taxed_at_death=by_age,
        terminal_residual_baseline=jnp.asarray(0.0),
        terminal_residual_reform=jnp.asarray(0.0),
        frozen=False,
    )

    result = run_stock_recurrence(
        baseline, schedules, jnp.ones((1, 1)), jnp.ones((1, 1))
    )
    tables = result.cell_tables
    share = np.asarray(cell_table_column(tables, "scenario_gain_share"))[:, 0]
    scenario_tax = np.asarray(
        cell_table_column(tables, "equivalent_tax_rate_scenario")
    )[:, 0]
    baseline_tax = np.asarray(
        cell_table_column(tables, "equivalent_tax_rate_baseline")
    )[:, 0]

    assert scenario_tax[0] == pytest.approx(at_baseline_share, abs=1e-12)
    assert share[1] > 0.10
    assert scenario_tax[1] > at_baseline_share + 1e-3
    assert scenario_tax[1] == pytest.approx(0.10 + 0.20 * share[1], abs=1e-12)
    # Baseline law's tax stays at the baseline gain share.
    np.testing.assert_allclose(baseline_tax, 0.15)
