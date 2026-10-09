"""Host-side arrangement of one scenario's results into tables.

Everything here runs once per scenario after its years are done. It does no
arithmetic on the model's numbers; it rearranges what the device handed back
into the structures the published tables read.

The scenario loop (`years/loop.py`) calls the year
loop, which calls `years/one_year.one_year_step` once per year; the step
computes the worlds and then the report sums (`reporting/device_sums.py`).
When a scenario's years are done, `assemble_simulation_run` lays its
device output into a `SimulationRun`, and `projection_from_simulation`
turns that into the `ProjectionRun` the output directory writes.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from taxmodel.reporting import detail as detail_output
from taxmodel.reporting import (
    distribution,
    estate_allocator_jax,
    mtr_table,
    revenue,
    wealth_diagnostics,
)
from taxmodel.reporting import tax_totals as tax_totals_output
from taxmodel.reporting.device_sums import (
    EVALUATED_CELLS,
    EXPECTED_ESTATE_TAX,
    TAX_AT_DEATH_LADDERS,
    _put_yearly_totals,
)
from taxmodel.years import wealth_arithmetic, world_names, worlds

if TYPE_CHECKING:
    # Only for annotations. Both of these modules import this one, so
    # importing them back at run time would be a circle.
    from taxmodel.scenario.prepare import PreparedScenario
    from taxmodel.years.one_year import DeviceOutputs

Array = jnp.ndarray

@dataclass(frozen=True)
class SimulationRun:
    """A projection's population totals, distributions, and wealth history.

    A run keeps no per-household results unless it was asked for particular
    columns. Every result column is multiplied by the household weights and
    added up in its own compiled program
    (`device_sums.population_totals_and_counts`), one number per column per
    year per (frame, law) cell, because thirty years of five worlds of full per-household detail
    is about 35 GB and does not fit on a device. A detail request names a
    handful of columns instead, and ``detail`` holds them.

    ``grid`` holds those totals, and beside each one the weighted count of
    households with a nonzero value, plus the counts of tax units and returns
    the published totals files report. Each cell's year entry is a dictionary
    of one-element arrays already carrying the population weights. The
    reporting layer reads these totals directly and subtracts them to form
    each channel's effect.

    ``wealth_channel_history`` is the first sweep's per-year wealth-bathtub
    state, in the names ``reporting/wealth_diagnostics.py`` reads.

    ``distribution_table`` is the distribution table, [year, corporate spread,
    tax row, group, column] as `reporting/device_sums.distribution_table_for_year`
    lays out each year, or None when the run formed none.
    ``mtr_table_sums`` holds the two dollar-weighted marginal-rate accumulators
    when the run calculated ordinary marginal rates.

    ``detail`` is the per-household columns the run was asked for, for the
    years it was asked for them, or None when nothing was asked for.
    """

    grid: world_names.ResultGrid
    years: tuple[int, ...]
    states_after_year: worlds.SimulationState
    wealth_channel_history: dict[str, Array]
    mtr_table_sums: dict[str, Array] | None
    detail: detail_output.HouseholdDetail | None = None
    # The capital-gains model's exports from the device: one [year, age cell]
    # array of weighted rate sums per name in
    # `behavior/gains/builders.GainsRatePricers`, plus the conversion
    # channel's per-household columns. Empty for a run without the gains model.
    gains_rate_exports: dict[str, Array] = field(default_factory=dict)
    distribution_table: np.ndarray | None = None


def assemble_simulation_run(
    outputs: DeviceOutputs,
    years: tuple[int, ...],
    detail_request: detail_output.DetailRequest | None = None,
    raw_by_year: Mapping[str, Array] | None = None,
) -> SimulationRun:
    """Lay one scenario's device output into the reporting structures.

    Host work only: the year-stacked arrays are rearranged into the result
    grid.

    When the run carried a detail request, the household identifier, weight
    and validity flag come off ``raw_by_year`` here, and the years the request
    did not ask for are dropped.
    """
    grid = world_names.ResultGrid()
    _put_yearly_totals(grid, outputs.totals_by_cell, years)
    check_estate_allocation(outputs.estate_allocation_by_year, years)
    household_detail = None
    if detail_request is not None:
        if raw_by_year is None:
            raise ValueError(
                "a detail request needs the population arrays the run read, "
                "because every detail file carries the household identifier "
                "and weight"
            )
        household_detail = detail_output.assemble(
            detail_request,
            years,
            raw_by_year["id"],
            raw_by_year["weight"],
            raw_by_year["valid"],
            outputs.detail_by_year,
        )
    return SimulationRun(
        grid=grid,
        years=years,
        states_after_year=outputs.states_after_year,
        wealth_channel_history=outputs.wealth_channel_history,
        mtr_table_sums=outputs.mtr_table_sums_by_year,
        detail=household_detail,
        gains_rate_exports=dict(outputs.gains_rate_exports_by_year),
        distribution_table=(
            None
            if outputs.distribution_table_by_year is None
            else np.asarray(outputs.distribution_table_by_year, dtype=np.float64)
        ),
    )


def simulation_run_with_only_the_gains_rates(
    gains_rate_exports: Mapping[str, Array], years: tuple[int, ...]
) -> SimulationRun:
    """One scenario's results from a pass that exported the gains rates only.

    A capital-gains scenario runs its years twice. The first pass is read only
    for the per-household rate columns the host-side selling solve needs, and
    the second pass produces every reported number, so the first pass forms no
    totals and no tables. What it returns carries the rate exports and nothing
    else.
    """
    return SimulationRun(
        grid=world_names.ResultGrid(),
        years=years,
        states_after_year=None,
        wealth_channel_history={},
        mtr_table_sums=None,
        gains_rate_exports=dict(gains_rate_exports),
    )


def check_estate_allocation(
    diagnostics: Mapping[str, Array], years: tuple[int, ...]
) -> None:
    """Check the heir allocation's totals once the run is back on the host.

    The compiled allocator cannot raise or warn, so it reports totals and they
    are read here, for every year, every world, and each of the two taxes
    charged at death on its own ladder (`TAX_AT_DEATH_LADDERS`). The two
    ladders are different lengths and can run out of heirs at different times.

    Four conditions are checked, in this order:

    *Nobody inherits anything.* There is no one for the tax at death to be
    charged to, so all of it is left unallocated and a warning says so.

    *The ladder has run out of heirs.* The reform taxes more bequest dollars
    than the heir data has inheritance dollars to receive, so the smallest
    estates past the crossing point are matched to nobody. On the deemed-gains
    ladder this is a known approximation in the late years of a long window,
    and it warns. On the estate tax's own
    ladder it stops the run.

    *The adding up.* Every dollar of tax that went onto the line comes off it
    again, to the cent. This is checked after the exhaustion test, so that a
    ladder that ran out of heirs reports the cause rather than the symptom.

    *Tax on estates with nothing to distribute*, which has no bequest dollars
    to be matched to, is reported.
    """
    if not diagnostics:
        return
    totals = np.asarray(diagnostics["estate_allocation"], dtype=np.float64)
    position = {
        name: index
        for index, name in enumerate(estate_allocator_jax.DIAGNOSTIC_NAMES)
    }
    a_cent = 0.01
    for year_index, year in enumerate(years):
        for world_index, (world, _frame, _law) in enumerate(EVALUATED_CELLS):
            for ladder_index, ladder in enumerate(TAX_AT_DEATH_LADDERS):
                cell = totals[year_index, world_index, ladder_index]
                _check_one_ladder(
                    year=year,
                    world=world,
                    ladder=ladder,
                    is_deemed_gains_ladder=ladder_index == 1,
                    owed=float(cell[position["tax_dollars"]]),
                    allocated=float(cell[position["allocated_dollars"]]),
                    inheritances=float(cell[position["inheritance_dollars"]]),
                    bequests=float(cell[position["bequest_dollars"]]),
                    unallocated=float(cell[position["unallocated_dollars"]]),
                    a_cent=a_cent,
                )


def _check_one_ladder(
    *,
    year: int,
    world: str,
    ladder: str,
    is_deemed_gains_ladder: bool,
    owed: float,
    allocated: float,
    inheritances: float,
    bequests: float,
    unallocated: float,
    a_cent: float,
) -> None:
    """One year, one world, one of the two taxes at death. See the caller."""
    if inheritances <= 0.0:
        # Nobody inherits anything this year, so all of the tax at death is
        # left unallocated. This happens in a year before the heir publication
        # starts, which the loader fills with zeros.
        if owed > a_cent:
            warnings.warn(
                f"${owed / 1e9:,.1f} billion of {ladder} in {year} in the "
                f"{world} world is charged to nobody, because the heir data "
                "has no inheritances at all that year",
                stacklevel=2,
            )
        return

    if owed <= a_cent:
        # This ladder carries no tax this year, as in a scenario that deems no
        # gains realized at death, or in a baseline. The checks below would
        # otherwise report the ladder outgrowing the heir data.
        return

    if bequests > inheritances:
        # The ladder has outgrown the heir data. Every dying estate goes on the
        # deemed-gains ladder, and the bequest dollars on that ladder grow
        # faster than the heir file's inheritance dollars, so in a long enough
        # window the two cross. Past the crossing the smallest estates are
        # matched to nobody.
        if is_deemed_gains_ladder:
            warnings.warn(
                f"in {year} the {world} world puts ${bequests / 1e9:,.1f} "
                f"billion of bequests on the ladder for {ladder}, but the heir "
                f"data carries only ${inheritances / 1e9:,.1f} billion of "
                "inheritances, so the smallest estates past that point are "
                f"matched to no heir: ${(owed - allocated) / 1e6:,.1f} million "
                f"of the ${owed / 1e9:,.1f} billion owed is charged to nobody. "
                "The totals and the revenue estimate count the tax where it "
                "is owed and are unaffected",
                stacklevel=2,
            )
            return
        raise ValueError(
            f"in {year} the {world} world taxes ${bequests / 1e9:,.1f} billion "
            f"of bequests through {ladder} but the heir data carries only "
            f"${inheritances / 1e9:,.1f} billion of inheritances, so there is "
            "no one left to match the largest estates to"
        )

    if abs(allocated - owed) > a_cent:
        raise ValueError(
            f"{ladder} owed in {year} by the {world} world does not add up "
            f"over the heirs it was spread across: ${allocated:,.2f} was "
            f"spread and ${owed:,.2f} was owed"
        )

    if unallocated > 0.0:
        warnings.warn(
            f"${unallocated / 1e6:,.1f} million of {ladder} in {year} sits on "
            f"estates with nothing left to distribute in the {world} world, so "
            "there are no bequest dollars to match it to; it is left "
            "unallocated",
            stacklevel=2,
        )


# The result columns the revenue tables report on. Each is a liability one
# household owes, so a population total is the household weight times the
# column, added up.
#
# The first three are the calendar-year liabilities the revenue tables report.
# The rest are what the fiscal-year receipts files read: the split of income
# tax into what is withheld during the year and what is settled at filing, the
# same split for refundable credits, the corporate tax change from income
# shifting, and the expected estate tax. The estate figure is formed per
# household as the run reduces (`reporting/device_sums.py`).
REVENUE_COLUMNS: tuple[str, ...] = (
    "liab_iit_net",
    "liab_pr",
    "liab_wealth",
    "pmt_iit_withheld",
    "pmt_iit_nonwithheld",
    "pmt_refund_withheld",
    "pmt_refund_nonwithheld",
    "corp_tax_change",
    EXPECTED_ESTATE_TAX,
)


@dataclass(frozen=True)
class ProjectionRun:
    """Everything one projection produces.

    ``totals`` holds every result column's population total, by year, for each
    (frame, law) cell. ``channel_totals`` is the reform's revenue effect split
    into the four channels, and ``total_effect`` is the whole of it; the four
    add up to the whole.
    ``wealth_history`` is the bathtub's per-year state, ready for the
    diagnostics tables. ``distribution_table`` is the distribution table,
    [year, corporate spread, tax row, group, column], or None when the run
    formed none.
    ``mtr_table`` is the
    host-side dollar-weighted marginal-rate table from the mechanical frame.
    ``detail`` is the per-household columns the run was asked for, or None when
    it was asked for none.

    ``tax_totals`` holds, for each evaluated (frame, law) cell, the weighted
    sums and counts the four per-tax totals files report, as
    ``reporting/tax_totals.py`` describes.

    ``compiled_law`` and ``compiled_baseline_law`` are the two
    `law.compile.CompiledLaw` objects the scenario ran, which become
    `tax_law.csv`.

    ``corporate_rate_score_by_world`` is what the scenario's change in the
    corporate statutory rate raises or loses in each world, in dollars by year.
    It is worked out before the run starts, and `run/output_directory.py`
    reads it to publish the corporate revenue line. Every world's series is
    zeros when the scenario leaves the corporate rate alone.
    """

    years: tuple[int, ...]
    grid: world_names.ResultGrid
    totals: dict[tuple[world_names.Frame, world_names.Law], dict[str, np.ndarray]]
    channel_totals: dict[revenue.CellDifference, dict[str, np.ndarray]]
    total_effect: dict[str, np.ndarray]
    wealth_history: wealth_diagnostics.WealthShortfallHistory
    mtr_table: list[dict[str, object]]
    detail: detail_output.HouseholdDetail | None = None
    tax_totals: dict[
        tuple[world_names.Frame, world_names.Law], tax_totals_output.WorldTotals
    ] = field(default_factory=dict)
    compiled_law: object = None
    compiled_baseline_law: object = None
    # The capital-gains model's cell tables: one per year, shaped
    # [year, age cell, field] over `behavior/kg/stepper.CELL_TABLE_FIELDS`.
    # None for a scenario without the gains model, which writes no gains file.
    gains_cell_tables: Array | None = None
    corporate_rate_score_by_world: dict[str, np.ndarray] | None = None
    distribution_table: np.ndarray | None = None


def assemble_wealth_history(
    simulation: SimulationRun,
    haircut_cap: float,
) -> wealth_diagnostics.WealthShortfallHistory:
    """Arrange the run's wealth-channel output as the diagnostics tables read it.

    The accumulated shortfall standing in each cell at the end of a year is
    ``states_after_year``. Everything else comes from the per-year state the
    annual step recorded.
    """
    per_year = simulation.wealth_channel_history
    history = wealth_diagnostics.WealthShortfallHistory(
        years=np.asarray(simulation.years),
        ages=np.asarray(wealth_arithmetic.age_grid()),
        cell_shortfall=np.asarray(simulation.states_after_year.mechanical_wealth),
        cell_inflow=np.asarray(per_year["cell_inflow"]),
        growth_factor=np.asarray(per_year["growth_factor"]),
        capital_income_per_dollar_of_assets=np.asarray(
            per_year["capital_income_per_dollar_of_assets"]
        ),
        cell_net_worth=np.asarray(per_year["cell_net_worth"]),
        allocated_dollars=np.asarray(per_year["allocated_dollars"]),
        haircut_fraction=np.asarray(per_year["haircut_fraction"]),
        weight=np.asarray(per_year["weight"]),
        haircut_cap=float(haircut_cap),
    )
    history.check_shapes()
    return history


def revenue_tables(
    simulation: SimulationRun,
) -> tuple[
    dict[tuple[world_names.Frame, world_names.Law], dict[str, np.ndarray]],
    dict[revenue.CellDifference, dict[str, np.ndarray]],
    dict[str, np.ndarray],
]:
    """Per-cell totals, the four revenue channels, and the whole reform effect.

    The grid's cells already hold population totals. The differences are defined
    in `reporting/revenue.py`; this chooses which ones to ask for.
    """
    years = simulation.years
    available = set(simulation.grid.get(world_names.Frame.RAW, world_names.Law.BASELINE, years[0]))
    wanted = [name for name in REVENUE_COLUMNS if name in available]
    per_cell = {
        (frame, law): revenue.totals_by_year(
            simulation.grid, frame, law, years, wanted
        )
        for _name, frame, law in EVALUATED_CELLS
    }
    channels = revenue.channel_totals_by_year(
        simulation.grid, years, wanted
    )
    whole = revenue.difference_totals_by_year(
        simulation.grid, revenue.TOTAL_DIFFERENCE, years, wanted
    )
    return per_cell, channels, whole


def tax_totals_by_cell(
    simulation: SimulationRun,
) -> dict[tuple[world_names.Frame, world_names.Law], tax_totals_output.WorldTotals]:
    """Arrange the run's weighted totals the way the four totals files read them.

    The run already produced, for every world and year, the weighted sum of
    each reported column, the weighted count of households with a nonzero
    value under that name with an `n_` in front, and the counts of tax units
    and returns. This puts them under the names `reporting/tax_totals.py`
    looks them up by.

    A column the run did not total is left out, so a file that needs it
    reports which one is missing.
    """
    years = simulation.years
    by_cell: dict[tuple[world_names.Frame, world_names.Law], tax_totals_output.WorldTotals] = {}
    for _name, frame, law in EVALUATED_CELLS:
        totals = revenue.totals_by_year(simulation.grid, frame, law, years)
        present = [
            name
            for name in tax_totals_output.every_totalled_column()
            if name in totals
        ]
        by_cell[(frame, law)] = tax_totals_output.WorldTotals(
            sums={name: totals[name] for name in present},
            nonzero_counts={
                name: totals[f"n_{name}"]
                for name in present
                if f"n_{name}" in totals
            },
            n_tax_units=totals["n_tax_units"],
            n_returns=totals["n_returns"],
            n_returns_dep=totals["n_returns_dep"],
            n_dep=totals["n_dep"],
        )
    return by_cell


def projection_from_simulation(
    prepared: PreparedScenario, simulation: SimulationRun
) -> ProjectionRun:
    """Arrange one scenario's finished device work into the reported tables.

    Host work only. ``simulation`` is the lane `years/loop.py` gave back for
    this scenario.
    """
    per_cell, channels, whole = revenue_tables(simulation)
    marginal_rate_rows = (
        []
        if simulation.mtr_table_sums is None
        else mtr_table.mtr_table(
            simulation.mtr_table_sums,
            years=simulation.years,
            group_names=distribution.QUINTILE_LABELS,
        )
    )
    return ProjectionRun(
        years=prepared.years,
        grid=simulation.grid,
        totals=per_cell,
        channel_totals=channels,
        total_effect=whole,
        wealth_history=assemble_wealth_history(simulation, prepared.haircut_cap),
        mtr_table=marginal_rate_rows,
        detail=simulation.detail,
        tax_totals=tax_totals_by_cell(simulation),
        compiled_law=prepared.compiled_law,
        compiled_baseline_law=prepared.compiled_baseline_law,
        corporate_rate_score_by_world=prepared.corporate_rate_score_by_world,
        distribution_table=simulation.distribution_table,
    )
