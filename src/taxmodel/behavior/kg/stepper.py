"""The law of motion for policy-induced changes in unrealized capital gains.

The state variable is a vector with one element per age from 18 to 80. Each
element is the change in that age cohort's stock of unrealized gains relative
to the baseline projection from Tax-Data.

Cohorts choose turnover, the share of asset value sold in a year. Under
pro-rata selling from a mixed portfolio, turnover also equals the share of
unrealized gains realized that year.

Everything else — the baseline stock itself, baseline turnover, mortality — is
re-read from the data for each year. The recurrence never accumulates a level,
only a deviation: the projected baseline carries the level and the policy
carries the difference.

Asset value is never carried here. It stays on the baseline data path in every
scenario, because a sale in this model is a sale and a rebuy: the gain is
realized and the basis resets while the wealth stays on the path the projection
gives it.

The pace a cell sells at, and reform law's tax on a deferred dollar, are read
off the schedules the selling choice solved on its gain grid, at the gain share
the cell carries in the scenario rather than at its baseline gain share. The scenario gain share is the baseline stock
plus what the ledger is carrying above it, over the baseline asset value,
measured before the year's sales. Under carryover a cell that has inherited
gain sells at the rate the policy sets for that higher share, and faces the
tax on a deferred dollar for that share.

Three things happen to the stock in a year.

1. Survivors sell less, or more, than they would have. A cell entering the
   year with a deviation of dG holds ``baseline stock + dG`` in total. It sells
   the scenario share of that, where baseline law would have sold the baseline
   share of the baseline stock, so the deviation coming out of the sale is
   ``(1 - scenario turnover) * dG + baseline stock * (baseline turnover -
   scenario turnover)``. What is left ages one year.
2. Decedents' gains leave, and under carryover some arrive at heirs. What the
   people who die that year still hold after the year's sales exits. Where the
   scenario moves an asset class from step-up to carryover, the heir inherits
   the gain rather than a fresh basis, so the routed share of the taxable death
   stock lands in the heirs' cells and starts being realized there.
3. Converted compensation arrives at year end. Pay relabeled as equity enters
   the stock after the year's realizations and deaths, so it starts realizing
   and dying only from the following year, the same timing an inheritance gets.

The year's cell table is built from the stock entering the year rather than the
stock leaving it, because the household applier spends the year's gains against
the stock the household had on 1 January. Using the end-of-year stock would
shift every response a year early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from . import bellman_age, schedule
from .conversion_inflow import YearConversions
from .inputs import SMALLEST_SAFE_DENOMINATOR
from .schedule import ScheduleBundle
from .stock_inputs import BaselineCells, no_household_conversions

Array = jnp.ndarray


class ScheduleYear(NamedTuple):
    """The schedule numbers the recurrence reads, laid out year by age.

    The schedule solver returns one bundle of age-by-year matrices for the
    whole run (``schedule.ScheduleBundle``).  The recurrence scans over years,
    so it needs the same numbers with the year axis first, and only the fields
    the step or the cell table uses.  :func:`schedule_by_year` does that
    transposition once, before the scan.  ``reform_policy_by_gain_share`` and
    ``reform_tax_on_deferred_gain_by_gain_share`` already have the year axis
    first and are passed through.

    The names are the solver's: ``reform`` is the scenario being scored, and
    ``tax_on_deferred_gain`` is the present value of tax on one more dollar of
    unrealized gain at the start of the year, before that year's sales.
    Baseline law's is read at the cell's baseline gain share; reform law's is
    read from its grid at the scenario gain share.
    """

    reform_policy_by_gain_share: Array
    reform_tax_on_deferred_gain_by_gain_share: Array
    long_run_turnover_scale: Array
    retiming_turnover_shift: Array
    baseline_gain_stock: Array
    baseline_asset_value: Array
    tax_on_deferred_gain_baseline: Array
    death_share_routed_to_heirs: Array
    death_share_forgiven: Array
    death_share_taxed_at_death: Array


def schedule_by_year(schedules: ScheduleBundle) -> ScheduleYear:
    """Turn the solver's age-by-year bundle into the year-by-age slices."""

    return ScheduleYear(
        reform_policy_by_gain_share=schedules.reform_policy_by_gain_share,
        reform_tax_on_deferred_gain_by_gain_share=(
            schedules.reform_tax_on_deferred_gain_by_gain_share
        ),
        long_run_turnover_scale=schedules.long_run_turnover_scale.T,
        retiming_turnover_shift=schedules.retiming_turnover_shift.T,
        baseline_gain_stock=schedules.baseline_gain_stock.T,
        baseline_asset_value=schedules.baseline_asset_value.T,
        tax_on_deferred_gain_baseline=schedules.tax_on_deferred_gain_baseline.T,
        death_share_routed_to_heirs=schedules.death_share_routed_to_heirs.T,
        death_share_forgiven=schedules.death_share_forgiven.T,
        death_share_taxed_at_death=schedules.death_share_taxed_at_death.T,
    )


def scenario_gain_share(
    stock_change_entering: Array, schedules: ScheduleYear
) -> Array:
    """The share of a cell's asset value that is unrealized gain this year.

    Measured before the year's sales, on the stock the scenario has actually
    put in the cell: the baseline stock plus whatever the ledger is carrying
    above it.  Asset value stays on the baseline path in every scenario, so
    only the numerator moves.
    """

    return jnp.clip(
        (schedules.baseline_gain_stock + stock_change_entering)
        / jnp.maximum(schedules.baseline_asset_value, SMALLEST_SAFE_DENOMINATOR),
        0.0,
        1.0,
    )


def turnover_this_year(
    stock_change_entering: Array, schedules: ScheduleYear
) -> tuple[Array, Array]:
    """The pace each cell sells at this year, and the gain share it is read at.

    The selling choice solved the reform policy at every gain share on its
    grid.  Each cell reads that policy at the gain share it actually carries,
    which under carryover is above the baseline share because the cell has
    inherited gain, and under a rate rise is above it because the cell held
    back.  The reading is scaled by the observed baseline turnover
    over the baseline re-solve's turnover at the baseline gain share, which
    takes the baseline fit error out, and the retiming overlay is added on top.

    A cell carrying exactly its baseline stock gets exactly the schedule's
    ``reform_turnover``.
    """

    gain_share = scenario_gain_share(stock_change_entering, schedules)
    at_scenario_share = bellman_age.at_gain_share(
        schedules.reform_policy_by_gain_share, gain_share
    )
    long_run = schedules.long_run_turnover_scale * at_scenario_share
    turnover = jnp.clip(long_run + schedules.retiming_turnover_shift, 0.0, 1.0)
    return turnover, gain_share


# The columns of the cell table, in the order the emitted stack stores them.
# The first block describes the cell, the second the year's deaths, the third
# the prices the other modules read, and the last three are what the household
# applier spends.
#
# These names are read outside this package: they become the column headings of
# the run's gains_cells.csv.  The two rate columns say "realization rate" where
# the code inside this package says turnover.  They are the same number as long
# as selling is pro-rata, which the model assumes everywhere.
CELL_TABLE_FIELDS = (
    "baseline_stock",
    "baseline_realizations",
    "baseline_realization_rate",
    "scenario_realization_rate",
    "scenario_gain_share",
    "effective_mortality",
    "stock_change_entering",
    "corporate_gain_debit",
    "charitable_bequest_share",
    "forgiven_share",
    "routed_to_heirs_share",
    "realized_at_death_share",
    "decedent_stock",
    "charitable_death_stock",
    "taxable_death_stock",
    "stock_change_from_survivors",
    "stock_change_from_inheritance",
    "equivalent_tax_rate_baseline",
    "equivalent_tax_rate_scenario",
    "conversion_inflow",
    "rate_factor",
    "extra_realizations",
    "deemed_factor",
)

CELL_TABLE_FIELD_INDEX = {name: i for i, name in enumerate(CELL_TABLE_FIELDS)}


def cell_table_column(stack: Array, field: str) -> Array:
    """Pull one named column out of an emitted cell-table stack."""

    if field not in CELL_TABLE_FIELD_INDEX:
        raise KeyError(f"no such cell table column: {field}")
    return stack[..., CELL_TABLE_FIELD_INDEX[field]]


def effective_mortality(baseline: BaselineCells) -> Array:
    """The mortality rate the change in the stock of gains faces, by age cell.

    Averaging mortality over the people in a cell would overstate the flow of
    gains at death by about a factor of 2.7 in this data, because within an age
    cell the people most likely to die hold the least.  Weighting by holdings
    instead (sum each household's mortality times what it holds, then divide by
    what the cell holds) is exact rather than approximate, because the change
    in the stock is spread over households in proportion to holdings in the
    first place.  A cell holding nothing falls back to the plain per-person
    rate.
    """

    holds_something = baseline.stock > 0
    safe_stock = jnp.where(holds_something, baseline.stock, 1.0)
    rate = jnp.where(
        holds_something,
        baseline.mortality_gain_weighted / safe_stock,
        baseline.mortality_headcount,
    )
    return jnp.clip(rate, 0.0, 1.0)


class StockStep(NamedTuple):
    """One year of the recurrence: the new deviation and the flows behind it."""

    stock_change_next: Array
    scenario_turnover: Array
    effective_mortality: Array
    from_survivors: Array
    from_inheritance: Array
    decedent_stock: Array
    charitable_death_stock: Array
    taxable_death_stock: Array


def step_stock(
    stock_change_entering: Array,
    baseline: BaselineCells,
    scenario_turnover: Array,
    routed_to_heirs_share: Array,
    conversion_inflow: Array,
    aging: Array,
    heirs: Array,
) -> StockStep:
    """Move the change in the stock of unrealized gain forward one year.

    All arrays are one number per age cell.  ``aging`` and ``heirs`` are the
    two operators from :mod:`.stock_inputs`, applied transposed so that each
    cell gathers what flows into it.

    """

    mortality = effective_mortality(baseline)
    turnover = jnp.clip(scenario_turnover, 0.0, 1.0)

    # What the cell holds beyond baseline after this year's sales.  Selling a
    # share of a well-mixed portfolio takes that same share of its gains out of
    # the stock, so turnover multiplies the gains here.
    after_sales = (1.0 - turnover) * stock_change_entering + baseline.stock * (
        baseline.turnover - turnover
    )
    surviving = (1.0 - mortality) * after_sales
    from_survivors = aging.T @ surviving

    # The year's sales come before the year's deaths, so a decedent leaves the
    # stock the cell holds less the share of it sold during the year.
    decedent_stock = (
        mortality * (1.0 - turnover) * (baseline.stock + stock_change_entering)
    )
    charitable_share = jnp.clip(baseline.charitable_bequest_share, 0.0, 1.0)
    charitable_death_stock = charitable_share * decedent_stock
    taxable_death_stock = (1.0 - charitable_share) * decedent_stock
    # The routed share is per cell, so a cell whose assets all get a step-up
    # sends nothing to heirs even where neighbouring cells do.
    from_inheritance = heirs.T @ (routed_to_heirs_share * taxable_death_stock)

    return StockStep(
        stock_change_next=from_survivors + from_inheritance + conversion_inflow,
        scenario_turnover=turnover,
        effective_mortality=mortality,
        from_survivors=from_survivors,
        from_inheritance=from_inheritance,
        decedent_stock=decedent_stock,
        charitable_death_stock=charitable_death_stock,
        taxable_death_stock=taxable_death_stock,
    )


def extra_realizations_from_stock_change(
    scenario_turnover: Array,
    stock_change_entering: Array,
    corporate_gain_debit: Array,
    baseline_stock: Array,
) -> Array:
    """A cell's realizations beyond the rate channel, from the change in its stock.

    The scenario turnover applied to the change in the stock entering the
    year, less the gain the corporate equity markdown has claimed, with the
    change floored at minus the cell's whole baseline stock. The gains applier
    also calls this with no debit, to record how much the markdown lowered
    each household's realized gains.
    """

    return scenario_turnover * jnp.maximum(
        stock_change_entering - corporate_gain_debit,
        -baseline_stock,
    )


def _build_cell_table(
    stock_change_entering: Array,
    baseline: BaselineCells,
    schedules: ScheduleYear,
    step: StockStep,
    scenario_gain_share: Array,
    conversion_inflow: Array,
    corporate_gain_debit: Array,
) -> Array:
    """Assemble one year's cell table, shaped [age, field].

    Three of the columns are what the household applier spends; the rest are
    the prices other modules read and the diagnostics the wealth output keeps.

    - ``rate_factor`` scales each household's realized gains from the baseline
      pace of selling to the scenario's: scenario turnover over baseline
      turnover.
    - ``extra_realizations`` is the cell's realizations on top of that, from
      the change in the stock: the scenario turnover applied to the change,
      less any gain the corporate side has already claimed.  The change is
      floored at minus the cell's whole baseline stock, because under a
      permanent rate increase the stock can fall far enough that selling at
      the scenario's pace would take out more than the cell holds.
    - ``deemed_factor`` scales the gain a decedent is deemed to realize.  The
      cell's stock at death is the baseline stock plus the change, less the
      share of it sold earlier in the year.

    All three are built from ``stock_change_entering``, the stock on 1 January
    of this year, and not from the stock the step just produced.
    """

    holds_something = baseline.stock > 0
    safe_stock = jnp.where(holds_something, baseline.stock, 1.0)

    rate_factor = schedule.rate_factor(step.scenario_turnover, baseline.turnover)

    extra_realizations = extra_realizations_from_stock_change(
        step.scenario_turnover,
        stock_change_entering,
        corporate_gain_debit,
        baseline.stock,
    )
    deemed_factor = jnp.where(
        holds_something,
        jnp.maximum(
            0.0,
            (1.0 - step.scenario_turnover)
            * (baseline.stock + stock_change_entering)
            / safe_stock,
        ),
        1.0,
    )

    columns = {
        "baseline_stock": baseline.stock,
        "baseline_realizations": baseline.realizations,
        "baseline_realization_rate": baseline.turnover,
        "scenario_realization_rate": step.scenario_turnover,
        "scenario_gain_share": scenario_gain_share,
        "effective_mortality": step.effective_mortality,
        "stock_change_entering": stock_change_entering,
        "corporate_gain_debit": corporate_gain_debit,
        "charitable_bequest_share": baseline.charitable_bequest_share,
        "forgiven_share": schedules.death_share_forgiven,
        "routed_to_heirs_share": schedules.death_share_routed_to_heirs,
        "realized_at_death_share": schedules.death_share_taxed_at_death,
        "decedent_stock": step.decedent_stock,
        "charitable_death_stock": step.charitable_death_stock,
        "taxable_death_stock": step.taxable_death_stock,
        "stock_change_from_survivors": step.from_survivors,
        "stock_change_from_inheritance": step.from_inheritance,
        "equivalent_tax_rate_baseline": schedules.tax_on_deferred_gain_baseline,
        "equivalent_tax_rate_scenario": bellman_age.at_gain_share(
            schedules.reform_tax_on_deferred_gain_by_gain_share,
            scenario_gain_share,
        ),
        "conversion_inflow": conversion_inflow,
        "rate_factor": rate_factor,
        "extra_realizations": extra_realizations,
        "deemed_factor": deemed_factor,
    }
    return jnp.stack([columns[name] for name in CELL_TABLE_FIELDS], axis=-1)


@dataclass(frozen=True)
class StockRecurrenceResult:
    """What the stock recurrence hands to the rest of the model.

    ``cell_tables`` is shaped [year, age, field] over
    :data:`CELL_TABLE_FIELDS`; ``stock_change_end_of_year`` is shaped
    [year, age] and holds the deviation each year finishes with, which is the
    deviation the next year begins with.  ``conversions_by_household_leg`` is
    shaped [year, household, leg] over
    :data:`~.stock_inputs.CONVERSION_LEGS`, and is what the in-year
    conversion module clamps and applies.
    """

    cell_tables: Array
    stock_change_end_of_year: Array
    conversions_by_household_leg: Array


def run_stock_recurrence(
    baseline: BaselineCells,
    schedules: ScheduleBundle,
    aging: Array,
    heirs: Array,
    corporate_gain_debit: Array | None = None,
    priced_conversions: YearConversions | None = None,
) -> StockRecurrenceResult:
    """Run the whole stock recurrence for one scenario.

    The stock is the only thing that has to be computed a year at a time, so it
    is the only thing in this function's loop: one scan over years carrying a
    length-63 vector, starting from no deviation at all.

    ``priced_conversions`` is the compensation relabeled as capital gain,
    already priced year by year by ``conversion_inflow``.  It is priced outside
    this function because the conversion channel works household by household,
    which the compiled batched solve cannot do.  Its price, reform law's tax on
    a deferred dollar, is read at the scenario gain share this function
    produces, so the caller prices and steps in rounds until the two agree.
    Left out, nothing is converted, as the frozen pass and the calibration
    moments require.

    A bundle from ``schedule.solve_frozen_schedule()`` runs the pass the static
    and mechanical worlds need.  In a frozen bundle the scenario sells at
    exactly the baseline pace, so the rate factor comes out at exactly one, and
    both tax rates on a deferred dollar are zero because no schedule was
    solved.  What survives is the pure law arithmetic of changing how gains at
    death are treated, since the routing of gains to heirs stays live.

    The result is differentiable with respect to every array in the schedule
    bundle and in the priced conversions.
    """

    # The model's own grid is the 63 cells from age 18 to 80; the tests run the
    # same code on a handful of cells, so the size comes from the inputs.
    n_years, n_ages = baseline.stock.shape
    if aging.shape != (n_ages, n_ages) or heirs.shape != (n_ages, n_ages):
        raise ValueError(
            f"the aging and heir operators must both be {n_ages} by {n_ages} "
            f"to match the {n_ages} age cells the baseline carries"
        )
    if schedules.reform_turnover.shape != (n_ages, n_years):
        raise ValueError(
            "the schedule bundle is solved over "
            f"{schedules.reform_turnover.shape[0]} ages and "
            f"{schedules.reform_turnover.shape[1]} years, but the "
            f"baseline cells cover {n_ages} ages and {n_years} years"
        )
    policy = schedules.reform_policy_by_gain_share
    if policy.ndim != 3 or policy.shape[:2] != (n_years, n_ages):
        raise ValueError(
            f"the reform policy is shaped {policy.shape}; it must be "
            f"({n_years}, {n_ages}, gain-share grid points), one turnover per "
            "year, age cell and point of the gain-share grid"
        )

    if corporate_gain_debit is None:
        corporate_gain_debit = jnp.zeros((n_years, n_ages))

    by_year = schedule_by_year(schedules)

    if priced_conversions is not None:
        conversions = priced_conversions.by_household_leg
        conversion_inflow = priced_conversions.inflow_by_age
    else:
        # No households to report a conversion for, and no dollars entering
        # any cell's stock at year end.
        conversions = no_household_conversions(n_years)
        conversion_inflow = jnp.zeros((n_years, n_ages))

    def one_year(stock_change_entering, year):
        baseline_year, schedules_year, inflow_year, debit_year = year
        turnover, gain_share = turnover_this_year(
            stock_change_entering, schedules_year
        )
        step = step_stock(
            stock_change_entering=stock_change_entering,
            baseline=baseline_year,
            scenario_turnover=turnover,
            routed_to_heirs_share=schedules_year.death_share_routed_to_heirs,
            conversion_inflow=inflow_year,
            aging=aging,
            heirs=heirs,
        )
        table = _build_cell_table(
            stock_change_entering=stock_change_entering,
            baseline=baseline_year,
            schedules=schedules_year,
            step=step,
            scenario_gain_share=gain_share,
            conversion_inflow=inflow_year,
            corporate_gain_debit=debit_year,
        )
        return step.stock_change_next, (table, step.stock_change_next)

    start = jnp.zeros(n_ages)
    _, (cell_tables, stock_change_end_of_year) = lax.scan(
        one_year,
        start,
        (baseline, by_year, conversion_inflow, corporate_gain_debit),
    )

    return StockRecurrenceResult(
        cell_tables=cell_tables,
        stock_change_end_of_year=stock_change_end_of_year,
        conversions_by_household_leg=conversions,
    )
