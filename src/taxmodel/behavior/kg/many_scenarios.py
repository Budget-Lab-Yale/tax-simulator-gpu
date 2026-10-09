"""Batched solution of the realization schedule across scenarios.

The host-side capital gains calculations are: averaging households into age
cells, averaging the treatment of gains at death within each cell, solving the
realization choice, and updating the stock of unrealized gains. This module
runs them for all scenarios in one compiled program. Population cell fields
that do not depend on the reform are computed once and shared. Years and
scenarios are array dimensions rather than loops.

Each of those calculations is small, about a hundred ages by thirty years, but
there is a great deal of it. One scenario, one year and one cell at a time in
ordinary Python, it becomes a very large number of separate calls to the
device, each paying the fixed cost of Python and the device driver while the
arithmetic itself is too small to matter and the GPU sits idle. That overhead,
not the arithmetic, is what the batching is for.

Only two of the ten age-cell fields depend on the reform: the gain still at
stake at death, which the death-gain exclusions cut back, and the corporate
equity standing behind the cell's gains. The other eight are facts about the
population (``prepass.population_age_cells``). Each reform's law arrives as
data — the two exclusions by filing status, the death regime as the three
shares it implies, the corporate exposure — so one compiled program serves
every scenario in the group.

The arithmetic itself does not depend on how many scenarios are in the batch:
a group of four is checked against four groups of one, cell by cell
(``tests/unit/behavior/kg/test_many_scenarios.py``).

The conversion channel stays outside this program. Relabeling pay as equity is
priced household by household and year by year (about a gigabyte a scenario)
and it already runs as a handful of wide array operations rather than as many
small ones, so batching it would cost memory and buy nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.law.compile import FILING_STATUSES

from . import life_table, prepass, regime, stepper
from .applier import DEEMED_ASSET_CLASSES, UNMARRIED_FILING_STATUSES
from .assemble import (
    mortality_above_the_top_code_for_tax_units,
    nominal_discount_factor_by_year,
    price_deflator_to_base_year,
    schedule_inputs_for_run,
    solve_schedules,
    stock_operators,
)
from .conversion_inflow import YearConversions
from .inputs import Calibration, ScheduleInputs
from .schedule import ScheduleBundle
from .stock_inputs import (
    KG_N_AGES,
    BaselineCells,
    no_household_conversions,
)

Array = jnp.ndarray

# The household columns the batched program reads. Everything else the
# population block carries stays on the host.
RECORD_COLUMNS: tuple[str, ...] = (
    "weight",
    "kg_lt",
    "age1",
    "age2",
    "filing_status",
    "m_household",
    "p_widow",
    "estate_economic_gross",
    *(f"value.{name}" for name in DEEMED_ASSET_CLASSES),
    *(f"basis.{name}" for name in DEEMED_ASSET_CLASSES),
)


# ---------------------------------------------------------------------------
# What goes into the program
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedSolveInputs:
    """What every scenario's solve reads and no scenario changes.

    ``records`` is the run's households stacked over years, one array per
    column shaped [year, household], indexed by the population block's own
    column names. ``population_cells`` holds the age-cell fields no reform can
    move, each shaped [year, age].

    The rest is what the stock of unrealized gain is carried forward with, and
    no reform changes any of it either: the operator that ages a cell's stock
    into next year's cell, the operator that passes what a decedent leaves to
    the ages of the heirs who inherit it, the nominal discount factor for each
    year, the mortality rate for ages past the top-coded 80, and the run's
    years.  That mortality rate is on the same tax-unit convention as the
    cells' own (``assemble.mortality_above_the_top_code_for_tax_units``).
    """

    records: dict[str, Array]
    population_cells: prepass.PopulationAgeCells
    aging_operator: Array
    heir_operator: Array
    discount_factor_by_year: Array
    mortality_above_top_code: Array
    years: Array


@dataclass(frozen=True)
class ScenarioLaw:
    """One scenario's law on gains at death, in the form the program reads.

    Every field is stacked before the program runs: ten reforms become ten
    rows of the same array, one row each, so one program prices them all.

    The fields carried per year are the §121 exclusion on a primary home and
    the per-decedent death-gain exclusion, both by filing status; the joint
    filing status's death-gain exclusion on its own, which an unmarried filer
    would claim as a surviving spouse; the death regime code for each of the
    five tracked asset classes; and the three shares that regime implies
    (``regime.what_becomes_of_a_dollar``).

    The corporate channel adds two more: how much of each asset class is
    corporate equity, and the proportional markdown on fully exposed equity by
    year. A scenario with no corporate channel carries zeros for both.
    """

    section_121_by_status: Array
    death_gain_by_status: Array
    joint_death_gain: Array
    regime_code_by_class: Array
    becomes_by_class: Array
    corporate_exposure_by_class: Array
    equity_markdown: Array


@dataclass(frozen=True)
class ScenarioPrices:
    """One scenario's age-cell prices and calibrated numbers.

    The rate matrices are shaped [cell, year] on the stock's own 18-to-80 age
    grid; they are the compiled program's exported cell sums, divided out
    (``years/gains_solve.py``). The frozen solve reads none of them and
    passes zeros.

    The last three are the realization response's calibrated numbers. The
    retiming window and the appreciation rate decide which code runs, so they
    are the same for every scenario in one compiled program and are carried by
    :class:`ProgramShapingSettings` instead.
    """

    marginal_rate_baseline: Array
    marginal_rate_reform: Array
    marginal_rate_reform_law_only: Array

    elasticity: Array
    timeable_share: Array
    timing_reference_wedge: Array


@dataclass(frozen=True)
class SolvedSchedules:
    """What the batched schedule solve hands back, stacked over scenarios.

    Both fields hold one row per scenario. ``inputs`` comes back with
    ``bundle`` because ``schedule.check_schedule`` reads both, and it runs on
    the host after the program.
    """

    bundle: ScheduleBundle
    inputs: ScheduleInputs


for _registered in (
    SharedSolveInputs,
    ScenarioLaw,
    ScenarioPrices,
    prepass.PopulationAgeCells,
    prepass.ScenarioAgeCells,
):
    jax.tree_util.register_dataclass(
        _registered,
        data_fields=[f.name for f in _registered.__dataclass_fields__.values()],
        meta_fields=[],
    )


# ---------------------------------------------------------------------------
# Building the shared half, once per run
# ---------------------------------------------------------------------------


def _stack_records(
    records_by_year: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, Array]:
    """The run's households as one array per column, shaped [year, household].

    The host records arrive as one frame of plain numpy per year. Stacking them
    once and putting them on the device once avoids the tens of thousands of
    small transfers a year-by-year, scenario-by-scenario path makes.
    """

    return {
        name: jnp.asarray(np.stack([records[name] for records in records_by_year]))
        for name in RECORD_COLUMNS
    }


def shared_solve_inputs(
    records_by_year: Sequence[Mapping[str, np.ndarray]],
    years: Sequence[int],
    *,
    macro_root: str | Path,
    charity: prepass.CharitableBequestSettings,
    n_ages: int = KG_N_AGES,
) -> SharedSolveInputs:
    """Everything the batched solve reads from the population and the economy.

    This is built once per run, and the population half of the age cells is
    worked out in its own small compiled program over all thirty years at once.
    """

    stacked = _stack_records(records_by_year)
    deflator = price_deflator_to_base_year(macro_root, years, charity.base_year)

    cells = _population_cells_program(
        stacked, jnp.asarray(deflator), charity=charity, n_ages=n_ages
    )

    heir_age_shares = prepass.read_heir_age_distribution()
    operators = stock_operators(heir_age_shares, n_ages)

    return SharedSolveInputs(
        records=stacked,
        population_cells=cells,
        aging_operator=operators[0],
        heir_operator=operators[1],
        discount_factor_by_year=jnp.asarray(
            nominal_discount_factor_by_year(macro_root, years), dtype=jnp.float64
        ),
        mortality_above_top_code=mortality_above_the_top_code_for_tax_units(
            life_table.mortality_above_the_top_code(years),
            _joint_share_of_top_cell_program(stacked, n_ages=n_ages),
        ),
        years=jnp.asarray(list(years)),
    )


def baseline_cells(population: prepass.PopulationAgeCells) -> BaselineCells:
    """The six population cell fields the stock recurrence reads."""

    return BaselineCells(
        stock=population.stock,
        realizations=population.realizations,
        turnover=population.turnover,
        mortality_headcount=population.mortality_headcount,
        mortality_gain_weighted=population.mortality_gain_weighted,
        charitable_bequest_share=population.charitable_bequest_share,
    )


@partial(jax.jit, static_argnames=("n_ages",))
def _joint_share_of_top_cell_program(
    records: dict[str, Array], *, n_ages: int
) -> Array:
    """The joint-return share of the top cell's gain, every year at once."""

    def one_year(year_records: dict[str, Array]) -> Array:
        return prepass.joint_share_of_top_cell_gain(year_records, n_ages=n_ages)

    return jax.vmap(one_year)(records)


@partial(jax.jit, static_argnames=("charity", "n_ages"))
def _population_cells_program(
    records: dict[str, Array],
    deflator_by_year: Array,
    *,
    charity: prepass.CharitableBequestSettings,
    n_ages: int,
) -> prepass.PopulationAgeCells:
    """The eight age-cell fields no reform changes, every year at once."""

    def one_year(
        year_records: dict[str, Array], deflator: Array
    ) -> prepass.PopulationAgeCells:
        return prepass.population_age_cells(
            year_records,
            deflator_to_base_year=deflator,
            charity=charity,
            n_ages=n_ages,
        )

    return jax.vmap(one_year)(records, deflator_by_year)


# ---------------------------------------------------------------------------
# The law-dependent age cells, one program for every scenario
# ---------------------------------------------------------------------------


def _exclusions_for_year(
    records: Mapping[str, Array],
    section_121_by_status: Array,
    death_gain_by_status: Array,
    joint_death_gain: Array,
    regime_code_by_class: Array,
) -> prepass.DeathGainExclusions:
    """One year's death-gain exclusions, household by household.

    Done inside the compiled program, so each reform's exclusion amounts are
    an argument the program reads rather than a number compiled into it and
    ten reforms share one program.

    Each household takes the death-gain exclusion its own filing status is
    allowed. An unmarried filer carries the joint amount as well, because a
    surviving spouse may claim the exclusion their late spouse did not use.
    """

    codes = jnp.asarray(records["filing_status"]).astype(jnp.int32)
    row = jnp.clip(codes - min(FILING_STATUSES), 0, len(FILING_STATUSES) - 1)
    unmarried = jnp.isin(
        jnp.asarray(records["filing_status"]), jnp.asarray(UNMARRIED_FILING_STATUSES)
    )
    own_death_gain = death_gain_by_status[row]
    return prepass.DeathGainExclusions(
        section_121=section_121_by_status[row],
        per_decedent=own_death_gain,
        per_decedent_if_widowed=jnp.where(unmarried, joint_death_gain, own_death_gain),
        regime_code_by_class=regime_code_by_class,
    )


@partial(jax.jit, static_argnames=("n_ages",))
def scenario_cells(
    shared: SharedSolveInputs,
    laws: ScenarioLaw,
    *,
    n_ages: int = KG_N_AGES,
) -> prepass.ScenarioAgeCells:
    """The two law-dependent age-cell fields, for every scenario and year.

    Both of a scenario's two passes (the frozen one before any year is
    simulated and the behavioral one between the two executions) read the same
    cells, because neither the exclusions nor the corporate exposure depends on
    a rate, so this one compiled program runs once per run.
    """

    def one_year(
        year_records: dict[str, Array],
        section_121: Array,
        death_gain: Array,
        joint: Array,
        regime_code: Array,
        exposure: Array,
    ) -> prepass.ScenarioAgeCells:
        return prepass.scenario_age_cells(
            year_records,
            exclusions=_exclusions_for_year(
                year_records, section_121, death_gain, joint, regime_code
            ),
            corporate_asset_exposure=exposure,
            n_ages=n_ages,
        )

    def one_scenario(law: ScenarioLaw) -> prepass.ScenarioAgeCells:
        return jax.vmap(one_year, in_axes=(0, 0, 0, 0, 0, None))(
            shared.records,
            law.section_121_by_status,
            law.death_gain_by_status,
            law.joint_death_gain,
            law.regime_code_by_class,
            law.corporate_exposure_by_class,
        )

    # The scenarios run one at a time inside the program rather than all ten
    # side by side, because the intermediate that carries every household's
    # gain by asset class is a quarter of a gigabyte for one scenario's thirty
    # years, and ten of those at once is memory a laptop does not have. Running
    # them in sequence on the device still costs no Python.
    return jax.lax.map(one_scenario, laws)


# ---------------------------------------------------------------------------
# The solve itself, one program for every scenario
# ---------------------------------------------------------------------------


class ProgramShapingSettings(NamedTuple):
    """The calibration settings that decide which code the selling response runs.

    These two settings decide which arithmetic the solve does rather than
    what numbers come out of it: the retiming window, and the appreciation
    rate the selling choice ages a portfolio's gain share with.  They are
    compile-time constants, so every scenario in one compiled program has to
    agree on them, and they travel together as one object.
    """

    appreciation: float
    timing_window: int = 1

    @classmethod
    def from_calibration(cls, calibration: Calibration) -> ProgramShapingSettings:
        """The two settings as a scenario's calibration states them."""

        return cls(
            appreciation=calibration.appreciation,
            timing_window=calibration.timing_window,
        )


def scenario_schedule_inputs(
    shared: SharedSolveInputs,
    law: ScenarioLaw,
    cells: prepass.ScenarioAgeCells,
    prices: ScenarioPrices,
) -> ScheduleInputs:
    """One scenario's solver inputs, laid out from the shared population.

    Everything the schedule solver reads for one scenario and nothing solved
    yet: the mix of death regimes averaged over each cell's asset classes, the
    cell prices, and the population's own cells, turned age by year and
    stretched onto the solver's longer age grid. The batched solve calls this
    for every scenario inside its compiled program; the calibrate command
    calls it once, on the host, because it measures its moments on the inputs
    and solves the selling choice itself under the numbers it is fitting.
    """

    population = shared.population_cells

    forgiven, routed, taxed = jax.vmap(
        regime.death_shares_from_regime_weights
    )(
        population.gain_by_asset_class,
        cells.gain_at_stake_by_asset_class,
        law.becomes_by_class,
    )
    death_shares = regime.DeathShares(
        forgiven=forgiven.T,
        routed_to_heirs=routed.T,
        taxed_at_death=taxed.T,
    )

    prepass_result = prepass.PrepassResult(
        cells=baseline_cells(population),
        marginal_rate_baseline=prices.marginal_rate_baseline,
        marginal_rate_reform=prices.marginal_rate_reform,
        marginal_rate_reform_law_only=prices.marginal_rate_reform_law_only,
        gain_by_asset_class=population.gain_by_asset_class,
        tracked_asset_value=population.tracked_asset_value,
        gain_at_stake_by_asset_class=cells.gain_at_stake_by_asset_class,
        corporate_exposed_value=cells.corporate_exposed_value,
    )

    return schedule_inputs_for_run(
        prepass_result,
        death_shares,
        discount_factor_by_year=shared.discount_factor_by_year,
        mortality_above_top_code=shared.mortality_above_top_code,
        operators=(shared.aging_operator, shared.heir_operator),
    )


def _one_scenario_schedule(
    shared: SharedSolveInputs,
    law: ScenarioLaw,
    cells: prepass.ScenarioAgeCells,
    prices: ScenarioPrices,
    *,
    frozen: bool,
    shaping: ProgramShapingSettings,
) -> tuple[dict[str, Array], ScheduleInputs]:
    """One scenario's solved realization schedule, from the shared population.

    The bundle comes back as its array fields alone, because whether the
    schedule was solved frozen is one fact about the whole batch rather than a
    number per scenario, and a batched output cannot carry a plain Python flag.
    """

    schedule_inputs = scenario_schedule_inputs(shared, law, cells, prices)
    # The program-shaping settings carry the same names on the calibration as
    # they do on the bundle they arrived in.
    calibration = Calibration(
        elasticity=prices.elasticity,
        timeable_share=prices.timeable_share,
        timing_reference_wedge=prices.timing_reference_wedge,
        **shaping._asdict(),
    )
    return (
        _bundle_arrays(solve_schedules(schedule_inputs, calibration, frozen)),
        schedule_inputs,
    )


@partial(jax.jit, static_argnames=("frozen", "shaping"))
def _solve_schedules_program(
    shared: SharedSolveInputs,
    laws: ScenarioLaw,
    cells: prepass.ScenarioAgeCells,
    prices: ScenarioPrices,
    *,
    frozen: bool,
    shaping: ProgramShapingSettings,
) -> tuple[dict[str, Array], ScheduleInputs]:
    """The compiled program: every scenario's schedule, side by side."""

    def one(law, cell, price):
        return _one_scenario_schedule(
            shared, law, cell, price, frozen=frozen, shaping=shaping
        )

    return jax.vmap(one)(laws, cells, prices)


def solve_schedules_for_scenarios(
    shared: SharedSolveInputs,
    laws: ScenarioLaw,
    cells: prepass.ScenarioAgeCells,
    prices: ScenarioPrices,
    *,
    frozen: bool,
    shaping: ProgramShapingSettings,
) -> SolvedSchedules:
    """Every scenario's realization schedule, from one compiled program.

    How much of each age cell's stock of unrealized gain is realized in each
    year, solved for all the scenarios at once. The compiled program returns
    one set of arrays with the scenarios stacked first, so `result.bundle`
    holds ten scenarios' schedules stacked along a leading axis and the rest of
    the model reads it as an ordinary schedule bundle. Beyond calling the
    program, this function restores the flag saying whether this was the frozen
    pass: a plain yes or no cannot be carried inside a compiled program's
    arrays.

    ``shaping`` carries the settings that decide which code the selling
    response runs, which are the same for every scenario in one program.
    """

    arrays, inputs = _solve_schedules_program(
        shared, laws, cells, prices, frozen=frozen, shaping=shaping
    )
    return SolvedSchedules(
        bundle=ScheduleBundle(**arrays, frozen=frozen), inputs=inputs
    )


# ---------------------------------------------------------------------------
# The stock recurrence, one program for every scenario
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("frozen",))
def _step_stock_program(
    shared: SharedSolveInputs,
    bundle_arrays: dict[str, Array],
    conversion_inflow: Array,
    corporate_gain_debit: Array,
    *,
    frozen: bool,
) -> tuple[Array, Array]:
    """The compiled program: every scenario's thirty years of the stock."""

    population = shared.population_cells
    baseline = baseline_cells(population)
    n_years = population.stock.shape[0]
    empty_conversions = no_household_conversions(n_years)

    def one(
        fields: dict[str, Array], inflow: Array, debit: Array
    ) -> tuple[Array, Array]:
        bundle = ScheduleBundle(**fields, frozen=frozen)
        stepped = stepper.run_stock_recurrence(
            baseline,
            bundle,
            shared.aging_operator,
            shared.heir_operator,
            corporate_gain_debit=debit,
            priced_conversions=YearConversions(
                by_household_leg=empty_conversions,
                inflow_by_age=inflow,
            ),
        )
        return stepped.cell_tables, stepped.stock_change_end_of_year

    return jax.vmap(one)(bundle_arrays, conversion_inflow, corporate_gain_debit)


def step_stock_for_scenarios(
    shared: SharedSolveInputs,
    bundles: ScheduleBundle,
    conversion_inflow: Array,
    corporate_gain_debit: Array,
    *,
    frozen: bool,
) -> stepper.StockRecurrenceResult:
    """Every scenario's cell tables, from one compiled program.

    ``conversion_inflow`` is the dollars of relabeled pay entering each age
    cell, [scenario, year, age], priced outside this program. A frozen solve
    converts nothing and passes zeros. What comes back carries a leading
    scenario axis: the cell tables [scenario, year, age, field] over
    ``stepper.CELL_TABLE_FIELDS``, and the deviation each year finishes with.
    The conversions household by household are the caller's, because this
    program never touches a household.
    """

    cell_tables, stock_change_end_of_year = _step_stock_program(
        shared,
        _bundle_arrays(bundles),
        conversion_inflow,
        corporate_gain_debit,
        frozen=frozen,
    )
    n_years = int(shared.population_cells.stock.shape[0])
    return stepper.StockRecurrenceResult(
        cell_tables=cell_tables,
        stock_change_end_of_year=stock_change_end_of_year,
        conversions_by_household_leg=no_household_conversions(n_years),
    )


def _bundle_arrays(bundle: ScheduleBundle) -> dict[str, Array]:
    """A schedule bundle's array fields, so the batch axis has no bool in it.

    ``frozen`` says which solver produced the bundle, which is one fact about
    the whole batch rather than a number per scenario, so it travels as a
    plain Python flag and everything else travels as data.
    """

    return {
        name: getattr(bundle, name) for name in ScheduleBundle._fields if name != "frozen"
    }
