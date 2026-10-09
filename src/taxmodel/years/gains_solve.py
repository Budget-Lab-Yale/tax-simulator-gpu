"""The host-side capital-gains solve that runs between the two executions of the year loop.

A household's choice of how much gain to realize depends on the tax rates it
will face in future years, and those rates are known only once the years have
been simulated. So the run executes the compiled years once to export the
rates, solves the realization choice on the host, and executes the years a
second time with the solved tables passed in as data. The applier that reads
those tables inside the year, the pricers that export the rates, and the
settings are in `behavior/gains/`; this module is the solve in between.

The gains model decides how much of its stock of unrealized gain each age
cell realizes, from the marginal tax rates the cell's households face under
baseline law and under the reform. The solve runs on the host: a realization
schedule solved backwards over ages and years, and a stock of gain stepped
forwards.

**Host, up front.** The age cells are built from the data: Tax-Data columns,
mortality, the heir distribution, and the law's exclusions
(`scenario/gains_inputs.py`). The frozen recurrence then computes the law
arithmetic of the reform's treatment of gains at death, with turnover held at
baseline. It reads no rate, so it runs before anything is simulated
(:func:`solve_frozen_tables_for_group`). It produces two tables: one for the
first-order world, without the corporate markdown of share prices, and one
for the mechanical worlds, with it.

**Execution one.** The years run with the frozen tables in and the behavioral
tables at neutral values. The baseline, first-order and mechanical worlds are
final here, and the program measures three rates the host solve reads: the
reform's rates on the mechanical frame (after the frozen pass, so a carryover
reform's heirs move the measured rate), the same frame's rates under baseline
law, and the reform's rate on the frame before the frozen pass (the law-only
rate the retiming overlay reads). Every rate is the derivative of net income
tax with respect to realized long-term gain.

The gains model reads each rate as an age-cell average, so the rates are
summed into cells on the device (:func:`average_rates_into_cells`). The
exports are weighted sums, one per age cell per year.

**The host solve.** The exported sums are divided out and the realization
choice is solved (:func:`solve_behavioral_tables_for_group`): the two selling
choice passes, the priced tax on a deferred dollar, retiming, and the stock
recurrence. Both legs of the selling choice are priced on the mechanical
frame, so the comparison isolates the change in law.

**Execution two.** The same compiled program runs again with the behavioral
tables filled in. Its conventional world is the result; its other worlds do
not read the behavioral tables and reproduce execution one.

Everything that varies between scenarios (the cell tables, the death-regime
codes, the exclusions) is runtime data (:class:`GainsTables`), so one compiled
program serves every gains scenario of the same shape. The shape depends on
whether the scenario runs the gains model at all. Step-up, carryover and
deemed realization run the same program on different numbers.

A scenario whose behavior leg does not bind the ``kg_dynamics`` section runs
once, with the behavior chain's gains step at neutral values, like a scenario
with no gains model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.behavior.gains.builders import (
    BASELINE_ON_MECHANICAL_LEG,
    LAW_ONLY_LEG,
    REFORM_LEG,
    cell_rate_on_gains,
)
from taxmodel.behavior.gains.settings import (
    CONVERSION_PIECE,
    DEFAULT_SETTINGS_FILE,
    JOINT_FILING_STATUS_ROW,
    GainsLaw,
    build_conversion_pool,
    read_calibration,
    read_conversion_settings,
    read_gains_law,
)
from taxmodel.behavior.kg import (
    conversion_inflow,
    many_scenarios,
    prepass,
    regime,
    schedule,
    stepper,
    stock_inputs,
)
from taxmodel.behavior.kg.applier import DEEMED_ASSET_CLASSES
from taxmodel.behavior.kg.inputs import Calibration, ScheduleInputs
from taxmodel.behavior.kg.stock_inputs import KG_N_AGES
from taxmodel.years.stacking import stack_lanes

Array = jnp.ndarray

# ---------------------------------------------------------------------------
# Both solves, for a whole group of scenarios at once
# ---------------------------------------------------------------------------

# A run solves the gains scenarios of a group in one compiled program
# (`behavior/kg/many_scenarios.py`). A single scenario, such as the calibrate
# command's current-law pass, is a group of one.


@dataclass(frozen=True)
class GainsSolveGroup:
    """The inputs a group of gains scenarios shares, prepared once per run.

    ``shared`` is the population and economy inputs the batched program reads,
    and ``cells`` each scenario's two law-dependent age-cell fields. The cells
    are solved once and read by both of a scenario's passes, because neither
    the death-gain exclusions nor the corporate exposure depends on a rate.

    ``laws`` is the group's laws stacked over scenarios for the program;
    ``laws_by_scenario`` is the same laws unstacked, read on the host for the
    death-gain exclusions each scenario's state carries. ``compiled_laws`` are
    the scenarios' compiled law objects, kept for the conversion pool's
    top-bracket threshold. ``conversion_settings`` holds each scenario's sigma
    and pass-through labor share, read from the conversion file its behavior
    leg binds, or None for a scenario that prices no conversions.

    The equity markdown in ``laws`` comes from the corporate paths built with
    the tax base eroding and feeds the behavioral solve.
    ``mechanical_equity_markdown`` is the markdown from the paths built with
    the base fixed, [scenario, year], and feeds the mechanical worlds' frozen
    pass.
    """

    scenario_ids: tuple[str, ...]
    years: tuple[int, ...]
    shared: many_scenarios.SharedSolveInputs
    laws: many_scenarios.ScenarioLaw
    cells: prepass.ScenarioAgeCells
    calibrations: tuple[Calibration, ...]
    laws_by_scenario: tuple[GainsLaw, ...]
    compiled_laws: tuple[Any, ...]
    mechanical_equity_markdown: Array
    conversion_settings: tuple[tuple[float, float] | None, ...]


def scenario_law_for_solve(
    law: GainsLaw,
    *,
    corporate_asset_exposure: Mapping[str, float] | None = None,
    equity_markdown_by_year: Sequence[float] | None = None,
) -> many_scenarios.ScenarioLaw:
    """One scenario's law on gains at death, in the batched program's form.

    The death regimes become the three shares each implies, the two exclusions
    keep their filing-status axis, and the corporate channel becomes five
    exposure shares and one markdown per year. A scenario with no corporate
    channel carries zeros.
    """
    n_years = len(law.death_regime_by_class)
    exposure = np.asarray(
        [
            (corporate_asset_exposure or {}).get(f"value.{name}", 0.0)
            for name in DEEMED_ASSET_CLASSES
        ],
        dtype=np.float64,
    )
    markdown = (
        np.zeros(n_years, dtype=np.float64)
        if equity_markdown_by_year is None
        else np.asarray(equity_markdown_by_year, dtype=np.float64)
    )
    if markdown.shape != (n_years,):
        raise ValueError(
            "the equity markdown needs one value per run year: got "
            f"{markdown.shape} for {n_years} years"
        )
    return many_scenarios.ScenarioLaw(
        section_121_by_status=jnp.asarray(law.section_121_exclusion),
        death_gain_by_status=jnp.asarray(law.death_gain_exclusion),
        joint_death_gain=jnp.asarray(
            np.asarray(law.death_gain_exclusion)[:, JOINT_FILING_STATUS_ROW]
        ),
        regime_code_by_class=jnp.asarray(law.death_regime_by_class),
        becomes_by_class=jnp.asarray(
            np.stack(
                [
                    regime.what_becomes_of_a_dollar(row)
                    for row in law.death_regime_by_class
                ]
            )
        ),
        corporate_exposure_by_class=jnp.asarray(exposure),
        equity_markdown=jnp.asarray(markdown),
    )


def prepare_gains_solve_group(
    host_records: Sequence[Mapping[str, Any]],
    years: Sequence[int],
    *,
    scenario_ids: Sequence[str],
    compiled_laws: Sequence[Any],
    behavior_legs: Sequence[Any],
    macro_root: str | Path,
    corporate_asset_exposures: Sequence[Mapping[str, float] | None],
    equity_markdowns: Sequence[Sequence[float] | None],
    mechanical_equity_markdowns: Sequence[Sequence[float] | None] | None = None,
    charity: prepass.CharitableBequestSettings | None = None,
    settings_file: Path | str = DEFAULT_SETTINGS_FILE,
    calibrations: Sequence[Calibration] | None = None,
) -> GainsSolveGroup:
    """Read and build the inputs to a group's two batched solves.

    Two passes over the population's households: one for the age cells no
    reform changes, one for the two fields each reform changes.

    ``equity_markdowns`` are the markdowns from the eroded-base corporate
    paths and ``mechanical_equity_markdowns`` those from the fixed-base paths.
    Left out, the mechanical worlds use ``equity_markdowns``.
    """
    n_years = len(years)
    if charity is None:
        charity = prepass.read_charitable_bequest_settings(settings_file)
    if calibrations is None:
        calibrations = tuple(
            read_calibration(leg, settings_file) for leg in behavior_legs
        )
    else:
        calibrations = tuple(calibrations)
    # Every scenario in one batched solve must agree on the settings that
    # decide which code the realization response runs. Checked before anything
    # is built.
    _shared_program_settings(calibrations)
    laws_by_scenario = tuple(read_gains_law(law, n_years) for law in compiled_laws)
    shared = many_scenarios.shared_solve_inputs(
        host_records,
        years,
        macro_root=macro_root,
        charity=charity,
    )
    laws = stack_lanes(
        [
            scenario_law_for_solve(
                law,
                corporate_asset_exposure=exposure,
                equity_markdown_by_year=markdown,
            )
            for law, exposure, markdown in zip(
                laws_by_scenario,
                corporate_asset_exposures,
                equity_markdowns,
                strict=True,
            )
        ]
    )
    if mechanical_equity_markdowns is None:
        mechanical_equity_markdowns = equity_markdowns
    mechanical_markdown = jnp.stack(
        [
            jnp.zeros(n_years)
            if markdown is None
            else jnp.asarray(markdown, dtype=jnp.float64)
            for markdown in mechanical_equity_markdowns
        ]
    )
    if mechanical_markdown.shape != (len(scenario_ids), n_years):
        raise ValueError(
            "the mechanical equity markdown needs one value per scenario and "
            f"run year: got {mechanical_markdown.shape}"
        )
    return GainsSolveGroup(
        scenario_ids=tuple(scenario_ids),
        years=tuple(int(year) for year in years),
        shared=shared,
        laws=laws,
        cells=many_scenarios.scenario_cells(shared, laws),
        calibrations=calibrations,
        laws_by_scenario=laws_by_scenario,
        compiled_laws=tuple(compiled_laws),
        mechanical_equity_markdown=mechanical_markdown,
        conversion_settings=tuple(
            read_conversion_settings(leg.kg_dynamics[CONVERSION_PIECE])
            if leg is not None and leg.converts
            else None
            for leg in behavior_legs
        ),
    )


def _shared_program_settings(
    calibrations: Sequence[Calibration],
) -> many_scenarios.ProgramShapingSettings:
    """The settings every scenario in a group must share to share a program.

    The retiming window and the appreciation rate decide which code the
    realization response runs. A group whose scenarios disagree is refused.
    Returns the shared settings.
    """
    settings = {
        many_scenarios.ProgramShapingSettings.from_calibration(c)
        for c in calibrations
    }
    if len(settings) != 1:
        raise ValueError(
            "the scenarios in one batched capital-gains solve must agree on "
            "the selling response's retiming window and the appreciation "
            "rate; this group disagrees on at least one of them"
        )
    return settings.pop()


def _stacked_prices(
    calibrations: Sequence[Calibration],
    cell_prices: Sequence[Mapping[str, Array]],
) -> many_scenarios.ScenarioPrices:
    """The group's cell prices and calibrated numbers, stacked over scenarios."""
    return stack_lanes(
        [
            many_scenarios.ScenarioPrices(
                marginal_rate_baseline=prices["baseline"],
                marginal_rate_reform=prices["reform"],
                marginal_rate_reform_law_only=prices["law_only"],
                elasticity=jnp.asarray(calibration.elasticity, dtype=jnp.float64),
                timeable_share=jnp.asarray(
                    calibration.timeable_share, dtype=jnp.float64
                ),
                timing_reference_wedge=jnp.asarray(
                    calibration.timing_reference_wedge, dtype=jnp.float64
                ),
            )
            for calibration, prices in zip(calibrations, cell_prices, strict=True)
        ]
    )


_CELL_PRICE_NAMES: tuple[str, ...] = ("baseline", "reform", "law_only")


def _no_cell_prices(n_years: int) -> dict[str, Array]:
    """The cell prices a frozen solve reads: all zeros.

    The frozen pass holds turnover at baseline and reads no rate. The solver
    shares one input structure with the behavioral solve, so the rates it does
    not read are zeros.
    """
    zero = jnp.zeros((KG_N_AGES, n_years))
    return dict.fromkeys(_CELL_PRICE_NAMES, zero)


def _cell_prices_from_exports(rate_exports: Mapping[str, Array]) -> dict[str, Array]:
    """One scenario's age-cell prices, from its first execution's exports.

    The three rate matrices handed to the schedule solver, transposed to the
    [age, year] layout it reads. Each is an exported weighted sum divided by
    its weight.
    """
    return {
        "baseline": cell_rate_on_gains(rate_exports, BASELINE_ON_MECHANICAL_LEG).T,
        "reform": cell_rate_on_gains(rate_exports, REFORM_LEG).T,
        "law_only": cell_rate_on_gains(rate_exports, LAW_ONLY_LEG).T,
    }


def current_law_cell_prices(rate_exports: Mapping[str, Array]) -> dict[str, Array]:
    """The cell prices of an economy where reform law is current law.

    The calibrate command measures its two shocks against these. Both legs of
    the realization choice read the baseline leg's rate on gains, so the
    schedule solver's reform rate matrix equals its baseline one, which the
    moment builder checks (``calibration/kg_moments.py``). Shaped like the
    result of :func:`_cell_prices_from_exports`.
    """
    on_gains = cell_rate_on_gains(rate_exports, BASELINE_ON_MECHANICAL_LEG).T
    return {"baseline": on_gains, "reform": on_gains, "law_only": on_gains}


def schedule_inputs_for_scenario(
    group: GainsSolveGroup,
    position: int,
    cell_prices: Mapping[str, Array],
) -> ScheduleInputs:
    """One scenario's solver inputs from the group, not yet solved.

    The calibrate command measures its two moments on these inputs and solves
    the realization choice itself under the parameters it is fitting. The
    inputs are built by the same function the batched program runs for every
    scenario, on the host, for the scenario at ``position``.
    """

    def at(leaf):
        return leaf[position]

    prices = _stacked_prices([group.calibrations[position]], [cell_prices])
    return many_scenarios.scenario_schedule_inputs(
        group.shared,
        jax.tree.map(at, group.laws),
        jax.tree.map(at, group.cells),
        jax.tree.map(at, prices),
    )


def population_cells(group: GainsSolveGroup) -> stock_inputs.BaselineCells:
    """The six population cell fields the stock recurrence reads, [year, cell]."""
    return many_scenarios.baseline_cells(group.shared.population_cells)


def _check_every_schedule(
    solved: many_scenarios.SolvedSchedules, scenario_ids: Sequence[str]
) -> None:
    """Check each scenario's solved schedule, and report the baseline fit.

    Runs on the host, one scenario at a time, because a compiled program
    cannot raise. The worst gap the baseline fit left in the group is printed
    once: the baseline pass fits the desired turnover so that every cell
    reproduces the pace it is observed to sell at, and the reform response is
    read against that fit.
    """
    for position, scenario_id in enumerate(scenario_ids):
        # A field left at None has no per-scenario slice and stays None.
        bundle = solved.bundle._replace(
            **{
                name: getattr(solved.bundle, name)[position]
                for name in solved.bundle._fields
                if name != "frozen" and getattr(solved.bundle, name) is not None
            }
        )
        inputs = jax.tree.map(lambda leaf, at=position: leaf[at], solved.inputs)
        try:
            schedule.check_schedule(bundle, inputs)
        except schedule.ScheduleCheckFailure as failure:
            raise schedule.ScheduleCheckFailure(
                f"scenario {scenario_id}: {failure}"
            ) from failure

    worst_fit_gap = float(
        jnp.max(jnp.abs(jnp.asarray(solved.bundle.baseline_turnover_fit_gap)))
    )
    print(
        f"gains model: the baseline fit reproduces every cell's observed "
        f"turnover to within {worst_fit_gap:.3g} across "
        f"{len(scenario_ids)} scenarios."
    )


class FrozenCellTables(NamedTuple):
    """One scenario's two frozen cell tables, each [year, age, field].

    ``first_order`` leaves out the corporate channel's markdown of share
    prices, because the first-order world holds every corporate effect out.
    ``mechanical`` subtracts the fixed-base paths' markdown from the stock of
    unrealized gain and feeds the mechanical worlds. The two tables are
    identical for a scenario with no corporate markdown.
    """

    first_order: Array
    mechanical: Array


def corporate_gain_debit_for_group(
    group: GainsSolveGroup, equity_markdown: Array
) -> Array:
    """The corporate markdown of each cell's gains, [scenario, year, age].

    The markdown is a stock: that year's proportional price decline,
    ``equity_markdown`` [scenario, year], times the market value of the
    corporate equity behind the cell's gains.
    """
    return equity_markdown[:, :, None] * group.cells.corporate_exposed_value


def solve_frozen_tables_for_group(group: GainsSolveGroup) -> list[FrozenCellTables]:
    """Every scenario's frozen cell tables, from one compiled program.

    The law arithmetic of changing the treatment of gains at death, solved
    before any year is simulated because it reads no rate. Turnover is held at
    baseline, a decedent's gain moves to the heirs under carryover, and the
    deemed tax is added to the final return under deemed realization. The
    stock is stepped twice, once without and once with the corporate
    markdown (:class:`FrozenCellTables`).
    """
    n_scenarios = len(group.scenario_ids)
    n_years = len(group.years)
    shared_settings = _shared_program_settings(group.calibrations)

    solved = many_scenarios.solve_schedules_for_scenarios(
        group.shared,
        group.laws,
        group.cells,
        _stacked_prices(
            group.calibrations,
            [_no_cell_prices(n_years)] * n_scenarios,
        ),
        frozen=True,
        shaping=shared_settings,
    )
    # A frozen pass prices no deferred dollar, so there is no salary-to-equity
    # conversion.
    nothing = jnp.zeros((n_scenarios, n_years, KG_N_AGES))
    first_order = many_scenarios.step_stock_for_scenarios(
        group.shared, solved.bundle, nothing, nothing, frozen=True
    )
    mechanical = many_scenarios.step_stock_for_scenarios(
        group.shared,
        solved.bundle,
        nothing,
        corporate_gain_debit_for_group(group, group.mechanical_equity_markdown),
        frozen=True,
    )
    return [
        FrozenCellTables(
            first_order=first_order.cell_tables[position],
            mechanical=mechanical.cell_tables[position],
        )
        for position in range(n_scenarios)
    ]


def solve_behavioral_tables_for_group(
    group: GainsSolveGroup,
    rate_exports_by_scenario: Sequence[Mapping[str, Array]],
) -> list[stepper.StockRecurrenceResult]:
    """Every scenario's behavioral cell tables, from one compiled program.

    The realization choice is solved for the whole group at once. The
    conversion channel works household by household, so it is priced
    separately, one scenario at a time in its own compiled program, and its
    inflows enter the stock recurrence.
    """
    n_scenarios = len(group.scenario_ids)
    n_years = len(group.years)
    shared_settings = _shared_program_settings(group.calibrations)

    solved = many_scenarios.solve_schedules_for_scenarios(
        group.shared,
        group.laws,
        group.cells,
        _stacked_prices(
            group.calibrations,
            [
                _cell_prices_from_exports(exports)
                for exports in rate_exports_by_scenario
            ],
        ),
        frozen=False,
        shaping=shared_settings,
    )
    _check_every_schedule(solved, group.scenario_ids)

    no_conversions = jnp.zeros((n_years, 0, len(stock_inputs.CONVERSION_LEGS)))
    inflows: list[Array] = []
    conversions: list[Array] = []
    for position, exports in enumerate(rate_exports_by_scenario):
        conversion_settings = group.conversion_settings[position]
        if conversion_settings is None:
            inflows.append(jnp.zeros((n_years, KG_N_AGES)))
            conversions.append(no_conversions)
            continue
        sigma, pass_through_labor_share = conversion_settings
        priced = _price_conversions(
            build_conversion_pool(
                group.shared.records,
                exports,
                group.compiled_laws[position],
                n_years,
            ),
            sigma,
            pass_through_labor_share,
        )
        inflows.append(priced.inflow_by_age)
        conversions.append(priced.by_household_leg)

    stepped = many_scenarios.step_stock_for_scenarios(
        group.shared,
        solved.bundle,
        jnp.stack(inflows),
        corporate_gain_debit_for_group(group, group.laws.equity_markdown),
        frozen=False,
    )
    return [
        stepper.StockRecurrenceResult(
            cell_tables=stepped.cell_tables[position],
            stock_change_end_of_year=stepped.stock_change_end_of_year[position],
            conversions_by_household_leg=conversions[position],
        )
        for position in range(n_scenarios)
    ]


_price_conversions = jax.jit(
    conversion_inflow.price_conversions_all_years, static_argnames=("n_ages",)
)
