"""A hand-sized stock of unrealized gain to test the recurrence against.

Five age cells and four years, with round numbers, so every answer these tests
check can be worked out on paper.  The model's own grid is 63 cells from age 18
to 80 over thirty years; nothing in the recurrence depends on those sizes.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.behavior.kg.conversion_inflow import (
    YearConversions,
    price_conversions_all_years,
)
from taxmodel.behavior.kg.schedule import ScheduleBundle
from taxmodel.behavior.kg.stepper import cell_table_column
from taxmodel.behavior.kg.stock_inputs import (
    BaselineCells,
    ConversionPool,
    aging_operator,
    heir_operator,
)
from tests.tolerances import CENT

N_AGES = 5
N_YEARS = 4
N_LEGS = 5


def _by_year(row):
    """Repeat one year's numbers for every year of the fixture, [year, age]."""

    return jnp.tile(jnp.asarray(row, dtype=jnp.float64), (N_YEARS, 1))


def _by_age(row):
    """The same, laid out the way the schedule solver returns it, [age, year]."""

    return jnp.tile(jnp.asarray(row, dtype=jnp.float64)[:, None], (1, N_YEARS))


# One number per age, year and point of the gain-share grid; the fixtures use a
# short grid because their policies do not vary along it.
GRID_POINTS = 11


def flat_policy(turnover_by_age_and_year):
    """A policy that sells at the given pace whatever gain share a cell carries.

    Shaped ``[year, age, gain-share grid point]``, as the bundle carries it.
    Most fixtures here are about the recurrence rather than the selling choice,
    so they hand it a pace and want that pace applied.
    """

    return jnp.repeat(
        jnp.asarray(turnover_by_age_and_year).T[:, :, None], GRID_POINTS, axis=2
    )


def make_schedules(
    *,
    reform_turnover,
    tax_on_deferred_gain_baseline,
    tax_on_deferred_gain_reform,
    death_share_routed_to_heirs,
    death_share_forgiven,
    death_share_taxed_at_death,
    frozen: bool = False,
    reform_policy_by_gain_share=None,
    baseline_gain_stock=None,
    baseline_asset_value=None,
) -> ScheduleBundle:
    """A schedule bundle carrying only the fields the recurrence reads.

    The ones the recurrence never touches are filled with zeros here so the
    fixtures stay about the recurrence.  The end-to-end test in
    ``test_solver_to_stepper.py`` uses a bundle the solver actually produced.

    The policy defaults to one that does not vary with the gain share, so a
    fixture that hands in a reform turnover gets that turnover applied in every
    cell.  Reform law's tax on a deferred dollar does not vary with the gain
    share either.  A fixture testing the reading itself supplies its own policy, and
    the gain stock and asset value that set the share it is read at.
    """

    unused = jnp.zeros((N_AGES, N_YEARS))
    ones = jnp.ones((N_AGES, N_YEARS))
    if reform_policy_by_gain_share is None:
        reform_policy_by_gain_share = flat_policy(reform_turnover)
    return ScheduleBundle(
        baseline_turnover=unused,
        reform_turnover=reform_turnover,
        rate_factor=unused,
        reform_policy_by_gain_share=reform_policy_by_gain_share,
        long_run_turnover_scale=ones,
        retiming_turnover_shift=unused,
        baseline_gain_stock=unused if baseline_gain_stock is None else baseline_gain_stock,
        baseline_asset_value=ones if baseline_asset_value is None else baseline_asset_value,
        value_baseline=unused,
        value_reform=unused,
        tax_on_deferred_gain_baseline=tax_on_deferred_gain_baseline,
        tax_on_deferred_gain_reform=tax_on_deferred_gain_reform,
        reform_tax_on_deferred_gain_by_gain_share=flat_policy(
            tax_on_deferred_gain_reform
        ),
        death_share_forgiven=death_share_forgiven,
        death_share_routed_to_heirs=death_share_routed_to_heirs,
        death_share_taxed_at_death=death_share_taxed_at_death,
        terminal_residual_baseline=jnp.asarray(0.0),
        terminal_residual_reform=jnp.asarray(0.0),
        frozen=frozen,
    )


@pytest.fixture
def baseline() -> BaselineCells:
    """Five cells holding a round billion each, one cell holding nothing.

    Mortality rises with age.  The gain-weighted numerator is set to the plain
    rate times the stock in every cell but the oldest, where it is deliberately
    lower than the headcount rate: within that cell the people most likely to
    die hold the least.
    """

    stock = np.array([1_000e6, 2_000e6, 4_000e6, 8_000e6, 0.0])
    turnover = np.array([0.04, 0.03, 0.02, 0.01, 0.05])
    realizations = stock * turnover
    mortality_headcount = np.array([0.005, 0.010, 0.030, 0.080, 0.100])
    gain_weighted = mortality_headcount * stock
    gain_weighted[3] = 0.020 * stock[3]

    return BaselineCells(
        stock=_by_year(stock),
        realizations=_by_year(realizations),
        turnover=_by_year(turnover),
        mortality_headcount=_by_year(mortality_headcount),
        mortality_gain_weighted=_by_year(gain_weighted),
        charitable_bequest_share=_by_year([0.0, 0.0, 0.10, 0.20, 0.0]),
    )


@pytest.fixture
def schedules_carryover(baseline) -> ScheduleBundle:
    """Realization slows a little, and half the death stock goes to heirs.

    Turnover falls by a fifth of itself in every cell — people hold
    on longer when selling is taxed more — and the scenario replaces step-up
    with carryover on half of each cell's assets.
    """

    return make_schedules(
        reform_turnover=0.8 * baseline.turnover.T,
        tax_on_deferred_gain_baseline=_by_age([0.15] * N_AGES),
        tax_on_deferred_gain_reform=_by_age([0.20] * N_AGES),
        death_share_routed_to_heirs=_by_age([0.5] * N_AGES),
        death_share_forgiven=_by_age([0.5] * N_AGES),
        death_share_taxed_at_death=jnp.zeros((N_AGES, N_YEARS)),
    )


@pytest.fixture
def aging():
    return aging_operator(N_AGES)


@pytest.fixture
def heirs():
    """Everything inherited lands in the two youngest cells, three to one."""

    return heir_operator(jnp.asarray([0.75, 0.25, 0.0, 0.0, 0.0]))


# The fixture's baseline marginal rate on realized long-term gain: the 20
# percent top rate plus the 3.8 percent net investment income tax.
BASELINE_GAINS_RATE = 0.238


def build_pool(
    *,
    marginal_rate_change: float,
    gains_rate_change: float = 0.0,
    n_households: int = 3,
) -> ConversionPool:
    """Three households, one of them ineligible, facing the same rate changes.

    ``marginal_rate_change`` moves the rate on every ordinary-income leg and
    ``gains_rate_change`` the rate on realized long-term gain.

    Household 0 is an employee with wages only — no active business, so it
    cannot convert.  Household 1 has wages and partnership income and is over
    the top-bracket threshold.  Household 2 has partnership income but taxable
    income below the threshold.
    """

    threshold = 600_000.0
    leg_amount = np.zeros((n_households, N_LEGS))
    # Legs, in order: wages1, wages2, partnership, S corporation, sole trader.
    leg_amount[0, 0] = 900_000.0
    leg_amount[1, 0] = 500_000.0
    leg_amount[1, 2] = 400_000.0
    leg_amount[2, 2] = 300_000.0

    taxable_income = np.array([900_000.0, 900_000.0, 300_000.0])
    age_index = np.array([1, 2, 3])
    weight = np.array([10.0, 20.0, 30.0])

    baseline_rates = np.full((n_households, N_LEGS), 0.35)
    scenario_rates = baseline_rates + marginal_rate_change

    return ConversionPool(
        weight=jnp.tile(jnp.asarray(weight), (N_YEARS, 1)),
        age_index=jnp.tile(jnp.asarray(age_index), (N_YEARS, 1)),
        taxable_income=jnp.tile(jnp.asarray(taxable_income), (N_YEARS, 1)),
        top_bracket_threshold=jnp.full((N_YEARS, n_households), threshold),
        leg_amount=jnp.tile(jnp.asarray(leg_amount), (N_YEARS, 1, 1)),
        marginal_rate_scenario=jnp.tile(jnp.asarray(scenario_rates), (N_YEARS, 1, 1)),
        marginal_rate_baseline=jnp.tile(jnp.asarray(baseline_rates), (N_YEARS, 1, 1)),
        gains_rate_scenario=jnp.full(
            (N_YEARS, n_households), BASELINE_GAINS_RATE + gains_rate_change
        ),
        gains_rate_baseline=jnp.full((N_YEARS, n_households), BASELINE_GAINS_RATE),
    )


def priced_conversions(
    pool: ConversionPool,
    sigma,
    pass_through_labor_share,
) -> YearConversions:
    """What the pool converts, summed into the fixture's age cells.

    A run prices conversions outside the stock recurrence, because the
    conversion channel works household by household while the recurrence runs
    on age cells (``years/gains_solve.py``). The tests price them the same way
    and hand the answer to ``run_stock_recurrence``.
    """

    return price_conversions_all_years(
        pool, sigma, pass_through_labor_share, n_ages=N_AGES
    )


def assert_stock_flow_identity(baseline, result) -> None:
    """Check the recurrence's accounting identity, year by year.

    Everything here is a deviation from baseline. What the cells hold above
    baseline at the end of a year has to equal what they held above baseline at
    the start, less the sales above what baseline law would have produced, less
    the gain the year's decedents took with them, plus the gain their heirs
    received, plus the compensation converted into deferred gain at year end.

    The two callers hand in populations built different ways; each keeps its own
    check that the fixture is really moving money, since an identity between
    zeros holds trivially.
    """

    tables = result.cell_tables
    entering = np.asarray(cell_table_column(tables, "stock_change_entering"))
    leaving = np.asarray(result.stock_change_end_of_year)
    stock = np.asarray(baseline.stock)
    baseline_turnover = np.asarray(baseline.turnover)
    scenario_turnover = np.asarray(
        cell_table_column(tables, "scenario_realization_rate")
    )
    mortality = np.asarray(cell_table_column(tables, "effective_mortality"))
    routed = np.asarray(cell_table_column(tables, "routed_to_heirs_share"))
    taxable_death = np.asarray(cell_table_column(tables, "taxable_death_stock"))
    inflow = np.asarray(cell_table_column(tables, "conversion_inflow"))

    for year in range(len(leaving)):
        extra_sales = (
            scenario_turnover[year] * (stock[year] + entering[year])
            - baseline_turnover[year] * stock[year]
        )
        after_sales = entering[year] - extra_sales
        taken_by_decedents = mortality[year] * after_sales
        received_by_heirs = routed[year] * taxable_death[year]

        expected = (
            entering[year].sum()
            - extra_sales.sum()
            - taken_by_decedents.sum()
            + received_by_heirs.sum()
            + inflow[year].sum()
        )
        assert abs(leaving[year].sum() - expected) < CENT
