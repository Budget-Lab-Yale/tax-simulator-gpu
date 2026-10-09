"""Measurement of the gains model's two realization moments.

This module is the function from two parameters to two moments that Newton
(:mod:`taxmodel.calibration.newton`) solves.

Both moments compare realizations under a raised tax rate on long-term gains
with realizations under current law: the log change in realizations divided
by the realization-weighted change in the marginal rate, multiplied by the
0.238 top-rate divisor to give the elasticity units of
:mod:`taxmodel.calibration.targets`.

The permanent moment raises the rate on gains by five percentage points,
permanently, and reads the year ten years after the window opens
(``targets.PERMANENT_HORIZON_YEARS``). This moment identifies the realization
elasticity.

The transitory moment raises the rate by two percentage points, permanently,
with the rise taking effect one year after the announcement year. It reads
the announcement year. Law is unchanged that year, so realizations move only
by retiming: dollars that would have been realized the next year at the
higher rate are sold in the announcement year instead. This moment identifies the timeable
share.

Both moments divide by the realization-weighted average change in the
marginal rate on gains, not by the statutory change. A shock that raises
only the top bracket changes the rate on the average realized dollar by less
than the statutory amount, and dividing by the statutory amount would
understate the response.

The permanent moment measures the average rate change in the year it is read.
The transitory moment measures it in the year the rate rises, because in the
announcement year the change is zero.

Realizations are the cell totals the gains model produces. The applier spreads
a cell's realizations across the households in it without changing the cell
total, so the moment measured on cells equals the moment measured after the
applier. Measuring on cells also keeps the calibration conditioned on nothing
outside the gains machinery.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.behavior.kg import (
    BaselineCells,
    ScheduleInputs,
    cell_table_column,
    run_stock_recurrence,
    solve_schedule,
)
from taxmodel.behavior.kg.inputs import Calibration

from . import targets as target_definitions

Array = jnp.ndarray


# Which year of the window each moment uses, counted from the first year of
# the window. The window opens in 2025: the rise is announced in 2026, takes
# effect in 2027, and the permanent moment is read ten years after the window
# opens (2035). Both legs put the higher rate in force in the same year.
ANNOUNCEMENT_YEAR_OFFSET = 1
EFFECTIVE_YEAR_OFFSET = 2


@dataclass(frozen=True)
class MomentScenarios:
    """The three prepared economies the two moments are measured from.

    One set of baseline cells, and two sets of schedule inputs that differ
    from it only in the tax rate on gains. Newton changes only the two
    calibrated parameters, so the economies are built once and reused at
    every step.
    """

    # Stocks, realizations and mortality by [year, age]. Shared by both legs.
    baseline_cells: BaselineCells

    # Prices and stocks by [age, year] for the permanent leg, where the rate
    # on gains is five points higher from the effective year on.
    permanent_shock: ScheduleInputs

    # The same for the announced leg, two points higher from the same
    # effective year.
    announcement_shock: ScheduleInputs

    # The two operators the stock recurrence uses: ageing forward, and
    # routing an inherited gain to the heirs' age cells.
    aging_operator: Array
    heir_operator: Array

    # Which year of the window each moment reads, as an index into the
    # window. The permanent moment reads ten years out; the transitory
    # moment reads the announcement year; the average rate change the
    # transitory moment divides by is read in the year the rate rises.
    permanent_year_index: int
    announcement_year_index: int
    effective_year_index: int

    # The gains machinery's settings (config/calibrations/kg/settings.yaml),
    # carried so that the fit runs under the same settings as the runs.
    appreciation: float = 0.0
    timing_reference_wedge: float = 0.05
    timing_window: int = 1


# Building the two shocked economies from a baseline one


def raise_the_rate_on_gains(
    baseline_rate: Array,
    shock_size: float,
    first_year_in_force: int,
) -> Array:
    """Add a flat rate increase from one year of the window onwards.

    ``baseline_rate`` is the cell marginal rate on realized long-term gains
    under current law, shaped [age, year]. Every cell gets the same increase.
    Because the moment divides by the average rate change the realized
    dollars saw, an even increase and a top-bracket increase give the same
    answer up to the curvature of the response.

    A caller that can price the population under the shocked law can pass
    the repriced matrix to :func:`moment_scenarios_from_baseline` instead.
    """

    baseline_rate = jnp.asarray(baseline_rate)
    year_index = jnp.arange(baseline_rate.shape[1])
    return baseline_rate + jnp.where(year_index >= first_year_in_force, shock_size, 0.0)


def _with_reform_rate(inputs: ScheduleInputs, rate_on_gains: Array) -> ScheduleInputs:
    """The same economy with a different reform-law rate on gains.

    Both reform rate matrices are set to the same values. In a run they
    differ, one measured before the frozen state is injected into the frame
    and one after; here the reform is only a change in the rate on gains, so
    the two coincide.
    """

    return dataclasses.replace(
        inputs,
        marginal_rate_reform=rate_on_gains,
        marginal_rate_reform_law_only=rate_on_gains,
    )


def moment_scenarios_from_baseline(
    baseline_inputs: ScheduleInputs,
    baseline_cells: BaselineCells,
    *,
    permanent_rate_on_gains: Array | None = None,
    announcement_rate_on_gains: Array | None = None,
    appreciation: float = 0.0,
    timing_reference_wedge: float = 0.05,
    timing_window: int = 1,
) -> MomentScenarios:
    """Build the three economies the two moments are measured from.

    ``baseline_inputs`` is the baseline economy: the prices and stocks the
    schedule solver reads under current law. Its reform-law rate on gains
    must equal its baseline-law rate, and this function checks that.

    ``baseline_cells`` is the population's age cells, the fields the stock
    recurrence reads (``years/gains_solve.population_cells``), from the same
    solve group a normal run builds.

    The two ``*_rate_on_gains`` arguments let a caller that has priced the
    population under the shocked law pass in the measured cell marginal rate,
    shaped like the baseline one. Left out, the rate is raised evenly by the
    leg's shock size from the effective year on
    (:func:`raise_the_rate_on_gains`).

    The window must be long enough for the permanent moment to be read ten
    years out.
    """

    baseline_inputs.validate()

    n_years = baseline_inputs.n_years
    permanent_year_index = target_definitions.PERMANENT_HORIZON_YEARS
    if n_years <= permanent_year_index:
        raise ValueError(
            f"The calibration window is {n_years} years long, but the permanent "
            f"moment is read {permanent_year_index} years after the window opens. "
            "Give the calibration at least "
            f"{permanent_year_index + 1} years of priced population."
        )

    for name in ("marginal_rate_reform", "marginal_rate_reform_law_only"):
        reform_rate = getattr(baseline_inputs, name)
        largest_gap = float(
            jnp.max(jnp.abs(reform_rate - baseline_inputs.marginal_rate_baseline))
        )
        if largest_gap > 0.0:
            raise ValueError(
                "The calibration must be handed a baseline economy, where reform "
                f"law is current law, but {name} differs from the baseline rate "
                f"on gains by up to {largest_gap:.4g} in some cell. Price the "
                "baseline leg on its own and pass that."
            )

    effective_year_index = EFFECTIVE_YEAR_OFFSET
    baseline_rate = baseline_inputs.marginal_rate_baseline

    if permanent_rate_on_gains is None:
        permanent_rate_on_gains = raise_the_rate_on_gains(
            baseline_rate,
            target_definitions.PERMANENT_SHOCK_SIZE,
            effective_year_index,
        )
    if announcement_rate_on_gains is None:
        announcement_rate_on_gains = raise_the_rate_on_gains(
            baseline_rate,
            target_definitions.ANNOUNCEMENT_SHOCK_SIZE,
            effective_year_index,
        )

    return MomentScenarios(
        baseline_cells=baseline_cells,
        permanent_shock=_with_reform_rate(baseline_inputs, permanent_rate_on_gains),
        announcement_shock=_with_reform_rate(
            baseline_inputs, announcement_rate_on_gains
        ),
        aging_operator=baseline_inputs.aging_operator,
        heir_operator=baseline_inputs.heir_operator,
        permanent_year_index=permanent_year_index,
        announcement_year_index=ANNOUNCEMENT_YEAR_OFFSET,
        effective_year_index=effective_year_index,
        appreciation=appreciation,
        timing_reference_wedge=timing_reference_wedge,
        timing_window=timing_window,
    )


# The measurement


def _realizations_by_cell(
    scenario_inputs: ScheduleInputs,
    scenarios: MomentScenarios,
    calibration: Calibration,
) -> Array:
    """Each cell's realizations in each year under one scenario, [year, age].

    A cell's realizations are its baseline realizations scaled by the rate
    factor (the cell's scenario turnover over its baseline turnover), plus the
    extra realizations from the cell holding a different stock than under
    baseline. These are the two columns the household applier spreads, so
    their sum is what the simulation reports.
    """

    bundle = solve_schedule(scenario_inputs, calibration)
    result = run_stock_recurrence(
        scenarios.baseline_cells,
        bundle,
        scenarios.aging_operator,
        scenarios.heir_operator,
    )
    cells = result.cell_tables
    return (
        cell_table_column(cells, "baseline_realizations")
        * cell_table_column(cells, "rate_factor")
        + cell_table_column(cells, "extra_realizations")
    )


def _realization_weighted_rate(
    rate_by_age_and_year: Array,
    realizations_by_year_and_age: Array,
    year_index: int,
) -> Array:
    """The marginal rate on gains the average realized dollar faced.

    The weights are realizations, because the moment needs the rate change on
    the dollars that moved, and most realized gains belong to a small number
    of taxpayers at the top rate.
    """

    weights = realizations_by_year_and_age[year_index]
    rates = rate_by_age_and_year[: weights.shape[0], year_index]
    return jnp.sum(weights * rates) / jnp.sum(weights)


def _measure_one_leg(
    scenarios: MomentScenarios,
    scenario_inputs: ScheduleInputs,
    scenario_realizations: Array,
    read_year_index: int,
    rate_change_year_index: int,
) -> Array:
    """One moment, in the declared elasticity units.

    ``read_year_index`` is the year whose realizations are compared with the
    baseline's, and ``rate_change_year_index`` is the year the average change
    in the rate on gains is measured in. They are the same year for the
    permanent moment and different for the transitory one, where the rate has
    not moved yet in the year that is read.
    """

    baseline_realizations = scenarios.baseline_cells.realizations

    scenario_total = jnp.sum(scenario_realizations[read_year_index])
    baseline_total = jnp.sum(baseline_realizations[read_year_index])

    rate_under_the_shock = _realization_weighted_rate(
        scenario_inputs.marginal_rate_reform,
        scenario_realizations,
        rate_change_year_index,
    )
    rate_under_current_law = _realization_weighted_rate(
        scenario_inputs.marginal_rate_baseline,
        baseline_realizations,
        rate_change_year_index,
    )
    average_rate_change = rate_under_the_shock - rate_under_current_law

    semi_elasticity = jnp.log(scenario_total / baseline_total) / average_rate_change
    return semi_elasticity * target_definitions.TOP_RATE_DIVISOR


def measure_moments(
    scenarios: MomentScenarios,
    realization_elasticity: Array,
    timeable_share: Array,
) -> Array:
    """The two moments, in the same units and order as the declared targets."""

    calibration = Calibration(
        elasticity=jnp.asarray(realization_elasticity),
        timeable_share=jnp.asarray(timeable_share),
        appreciation=scenarios.appreciation,
        timing_reference_wedge=jnp.asarray(scenarios.timing_reference_wedge),
        timing_window=scenarios.timing_window,
    )

    permanent = _measure_one_leg(
        scenarios,
        scenarios.permanent_shock,
        _realizations_by_cell(scenarios.permanent_shock, scenarios, calibration),
        read_year_index=scenarios.permanent_year_index,
        rate_change_year_index=scenarios.permanent_year_index,
    )
    transitory = _measure_one_leg(
        scenarios,
        scenarios.announcement_shock,
        _realizations_by_cell(scenarios.announcement_shock, scenarios, calibration),
        read_year_index=scenarios.announcement_year_index,
        rate_change_year_index=scenarios.effective_year_index,
    )
    return jnp.stack([permanent, transitory])


def moment_evaluation(scenarios: MomentScenarios):
    """The measurement as a function of one parameter array.

    Newton passes one array holding the realization elasticity and the
    timeable share, in that order, and receives one array holding the
    permanent and transitory moments, in that order.
    """

    def evaluate(parameters: Array) -> Array:
        return measure_moments(scenarios, parameters[0], parameters[1])

    return evaluate

