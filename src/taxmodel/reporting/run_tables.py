"""The tables a finished scenario publishes, built from its projection.

Each function here turns one finished projection (`reporting/assemble.ProjectionRun`)
into the DataFrame that one file in the run directory holds: revenue totals by
year and channel, the distribution table, the wealth channel's diagnostics,
the capital-gains cell tables, the resolved tax law, the receipts and revenue
estimates, and the average marginal rate table.
The stacked tables at the end read across several finished scenarios, so they
work from the small record each scenario leaves behind once its own files are
on disk (`ScenarioAggregates`).

`run/output_directory.py` decides which file each table lands in and the
folder layout around it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from taxmodel.config import law_stacks
from taxmodel.reporting import (
    distribution,
    mtr_table,
    receipts,
    receipts_inputs,
    revenue,
    stacked,
    tax_law_table,
    wealth_diagnostics,
)
from taxmodel.years import world_names

if TYPE_CHECKING:  # type names only
    from taxmodel.reporting.assemble import ProjectionRun
    from taxmodel.run.plan import RunPlan, ScenarioSpec

# The revenue column the stacked table reports. The payroll tax and the wealth
# tax are in each scenario's own receipts file.
STACKED_TABLE_COLUMN = "liab_iit_net"


def totals_table(projection: ProjectionRun) -> pd.DataFrame:
    """Revenue by year: each world's level, and each channel's effect.

    One row per year, per reported line, per revenue column. A `level` row is
    what one world collected, and its `line` is that world's name. A
    `difference` row is one channel of the reform's effect, or the whole of it,
    and its `line` names the two worlds being subtracted — for instance
    `mechanical_minus_mechanical_ex_saving`. The `description` column says the
    same thing in a sentence. The four channels add up to the whole, and both
    are here so a reader can check it.

    Amounts are in dollars.
    """
    rows: list[dict[str, Any]] = []
    years = list(projection.years)
    for cell, totals in projection.totals.items():
        world = world_names.world_name(*cell)
        rows.extend(
            _totals_rows(years, "level", world, f"the {world} world", totals)
        )
    for difference, totals in projection.channel_totals.items():
        rows.extend(
            _totals_rows(
                years,
                "difference",
                revenue.channel_name(difference),
                difference.description,
                totals,
            )
        )
    rows.extend(
        _totals_rows(
            years,
            "difference",
            revenue.channel_name(revenue.TOTAL_DIFFERENCE),
            revenue.TOTAL_DIFFERENCE.description,
            projection.total_effect,
        )
    )
    return pd.DataFrame(
        rows,
        columns=["year", "kind", "line", "description", "revenue_column", "dollars"],
    )


def _totals_rows(
    years: list[int],
    kind: str,
    line: str,
    description: str,
    totals: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "year": int(year),
            "kind": kind,
            "line": line,
            "description": description,
            "revenue_column": column,
            "dollars": float(value),
        }
        for column, values in totals.items()
        for year, value in zip(years, np.asarray(values), strict=True)
    ]


def distribution_table(projection: ProjectionRun) -> pd.DataFrame | None:
    """The distribution table, one row per year, corporate spread, group and tax.

    Rows are ordered by year, then corporate spread in
    `distribution.CORPORATE_SPREADS` order, then group in `distribution.GROUPS`
    order, then tax in `distribution.TAX_ROWS` order. The columns are
    `distribution.table_columns()`. Returns None when the run formed no table.
    """
    if projection.distribution_table is None:
        return None
    values = np.asarray(projection.distribution_table, dtype=np.float64)
    years = np.asarray(projection.years, dtype=np.int64)
    value_columns = distribution.value_column_names()
    spreads = np.array(distribution.CORPORATE_SPREADS)
    expected_shape = (
        len(years),
        len(spreads),
        len(distribution.TAX_ROWS),
        len(distribution.GROUPS),
        len(value_columns),
    )
    if values.shape != expected_shape:
        raise ValueError(
            f"the distribution table is shaped {values.shape}; its layout in "
            f"reporting/distribution.py calls for {expected_shape} "
            "(year, corporate spread, tax row, group, column)"
        )
    n_years, n_spreads, n_taxes, n_groups, n_columns = expected_shape
    # The array is [year, spread, tax row, group, column]; the file lists the
    # taxes within each group, so the tax and group axes swap before
    # flattening.
    by_row = values.transpose(0, 1, 3, 2, 4).reshape(
        n_years * n_spreads * n_groups * n_taxes, n_columns
    )

    dimensions = np.array([dimension for dimension, _group in distribution.GROUPS])
    groups = np.array([group for _dimension, group in distribution.GROUPS])
    rows_per_spread = n_groups * n_taxes
    keys = pd.DataFrame(
        {
            "group_dimension": np.tile(
                np.repeat(dimensions, n_taxes), n_years * n_spreads
            ),
            "group": np.tile(np.repeat(groups, n_taxes), n_years * n_spreads),
            "tax": np.tile(np.array(distribution.TAX_ROWS), n_years * n_spreads * n_groups),
            "corporate_spread": np.tile(np.repeat(spreads, rows_per_spread), n_years),
            "year": np.repeat(years, n_spreads * rows_per_spread),
        }
    )
    table = pd.concat(
        [keys, pd.DataFrame(by_row, columns=list(value_columns))], axis=1
    )
    return table[list(distribution.table_columns())]


def wealth_diagnostics_table(projection: ProjectionRun) -> pd.DataFrame:
    """The wealth channel's diagnostics, five tables in one long-form file.

    Each row is one table, one measure, one year, and, for the two tables that
    cut the shortfall by age or by net-worth percentile bin, the age or bin the
    number belongs to. Long form keeps five differently shaped tables in one
    file.
    """
    tables = wealth_diagnostics.wealth_diagnostics(projection.wealth_history)
    years = [int(year) for year in np.asarray(projection.wealth_history.years)]
    ages = [int(age) for age in np.asarray(projection.wealth_history.ages)]
    rows: list[dict[str, Any]] = []
    for table_name, measures in tables.items():
        for measure, values in measures.items():
            # These two name the rows and the columns of the other measures
            # rather than reporting anything themselves.
            if measure in ("year", "ages"):
                continue
            array = np.asarray(values)
            if array.ndim == 1:
                rows.extend(
                    {
                        "table": table_name,
                        "measure": measure,
                        "year": year,
                        "cut": "",
                        "value": float(value),
                    }
                    for year, value in zip(years, array, strict=True)
                )
                continue
            if table_name == "shortfall by age":
                cuts = ages
            else:
                cuts = list(range(array.shape[1]))
            for year_index, year in enumerate(years):
                rows.extend(
                    {
                        "table": table_name,
                        "measure": measure,
                        "year": year,
                        "cut": str(cut),
                        "value": float(array[year_index, cut_index]),
                    }
                    for cut_index, cut in enumerate(cuts)
                )
    return pd.DataFrame(rows, columns=["table", "measure", "year", "cut", "value"])


def gains_cells_table(projection: ProjectionRun) -> pd.DataFrame:
    """The capital-gains model's cell tables, one row per year per age cell.

    Every field the stock recurrence emits gets a column under its own name
    (`behavior/kg/stepper.CELL_TABLE_FIELDS`), in dollars and shares. The age
    column is the taxpayer's age: ages run from 18, and everyone at or above
    80 is pooled into the top cell.

    """
    # Imported here so that writing tables does not import the compiled
    # model; the gains model's modules import JAX.
    from taxmodel.behavior.kg import stepper
    from taxmodel.behavior.kg.stock_inputs import KG_AGE_MIN

    tables = np.asarray(projection.gains_cell_tables, dtype=np.float64)
    n_years, n_ages, _n_fields = tables.shape
    years = np.repeat(np.asarray(projection.years, dtype=np.int64), n_ages)
    ages = np.tile(np.arange(KG_AGE_MIN, KG_AGE_MIN + n_ages, dtype=np.int64), n_years)
    flat = tables.reshape(n_years * n_ages, -1)
    return pd.DataFrame(
        {
            "year": years,
            "age": ages,
            **{
                name: flat[:, position]
                for position, name in enumerate(stepper.CELL_TABLE_FIELDS)
            },
        }
    )


def law_table(projection: ProjectionRun) -> pd.DataFrame:
    """The resolved law the scenario ran in each year, baseline and reform."""
    return tax_law_table.tax_law_table(
        projection.compiled_law, projection.compiled_baseline_law
    )


def receipts_tables(
    projection: ProjectionRun,
    years: list[int],
    economy: str | None,
    config_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The receipts table and the revenue-estimates table, or nothing.

    Receipts reads data off disk: CBO's revenue levels and the scores other
    models publish come from the shared model-data tree, at the vintages this
    scenario's economy leg pins. The reading, the year-coverage check and the
    arithmetic are in `reporting/receipts.py`. Nothing is returned when the
    projection lacks the payment columns receipts needs; the run manifest then
    says the receipts files were not written.

    Each world's corporate statutory rate score comes from the projection,
    where `scenario/prepare.prepare_scenario` left it before the run started.
    A scenario that never touched the corporate rate carries None, which is
    read as scoring nothing.
    """
    if receipts.missing_model_columns(projection):
        return None
    inputs = receipts_inputs.load_receipts_inputs(years, economy, config_root)
    lines_by_world = receipts.receipts_by_world(
        projection, inputs, projection.corporate_rate_score_by_world
    )
    return (
        receipts.receipts_table(lines_by_world, inputs),
        receipts.revenue_estimates_table(lines_by_world, inputs),
    )


@dataclass(frozen=True)
class ScenarioAggregates:
    """What one finished scenario leaves in host memory after its files are written.

    The run loop writes each scenario's files as soon as it finishes and then
    frees everything the scenario held. The stacked tables, the stacked
    revenue estimates, and the run manifest need the whole run, and this
    record is what they read: numpy arrays and one DataFrame, never a device
    array.
    """

    spec: ScenarioSpec
    # The row's projection years.
    years: tuple[int, ...]
    # The four channel effects, `projection.channel_totals` with every array
    # brought to host numpy.
    channel_totals: dict
    # The whole reform effect, `projection.total_effect`, likewise numpy.
    total_effect: dict
    # The frame written to this scenario's revenue_estimates.csv, retained so
    # the stacked revenue estimates can subtract rungs without recomputing
    # receipts. None when receipts could not be built for this scenario.
    revenue_estimates: pd.DataFrame | None


def stacked_tables(
    plan: RunPlan, aggregates: dict[str, ScenarioAggregates]
) -> dict[str, stacked.StackedTable]:
    """One stacked table per stack row, keyed by the stack's row ID.

    A stacked package is scored as a ladder of runs, and a provision's own
    effect is the step from one rung to the next. The table is built in
    `reporting/stacked.py`; this says which scenarios each stack's steps are
    taken between, read from the aggregates each scenario retained.
    """
    tables: dict[str, stacked.StackedTable] = {}
    for stack in plan.stacks:
        scenarios = {
            scenario_id: _reported_channels(aggregates[scenario_id])
            for scenario_id in stack.scenario_ids
            if scenario_id in aggregates
        }
        # Every rung a provision is measured against has to be among the runs.
        for increment in stack.increments:
            _rung_measured_against(increment, scenarios)
        years = aggregates[stack.scenario_ids[0]].years
        tables[stack.row_id] = stacked.stacked_table(
            scenarios, stack.increments, years, STACKED_TABLE_COLUMN
        )
    return tables


def stacked_revenue_estimates(
    plan: RunPlan, aggregates: dict[str, ScenarioAggregates]
) -> dict[str, pd.DataFrame]:
    """One stacked budget-effect file per stack row.

    The stacked liability table above decomposes the income tax by channel.
    This file reports what each provision does to the whole budget (every tax,
    on the fiscal-year receipts timing), attributed in scoring order. A
    provision's row is the step between two adjacent rungs' revenue estimates,
    so a tax the income-tax column never sees, such as the wealth tax or the
    estate tax, lands on the provision that levied it.

    Each rung's revenue estimate is the frame that was written to its own
    revenue_estimates.csv, retained in its aggregates; a rung whose receipts
    could not be built leaves the whole stack's file unwritten, and the run
    manifest's not-written entries say why.
    """
    tables: dict[str, pd.DataFrame] = {}
    for stack in plan.stacks:
        estimate_of: dict[str, pd.DataFrame] = {}
        for scenario_id in stack.scenario_ids:
            retained = aggregates.get(scenario_id)
            if retained is None:
                continue
            if retained.revenue_estimates is None:
                break
            estimate_of[scenario_id] = retained.revenue_estimates
        else:
            rows: list[pd.DataFrame] = []
            for increment in stack.increments:
                subtrahend_id = _rung_measured_against(increment, estimate_of)
                minuend = estimate_of[increment.minuend_id]
                if subtrahend_id is None:
                    # A rung's own estimate is already reform minus baseline.
                    step = minuend.copy()
                else:
                    step = _difference_revenue_estimates(
                        minuend, estimate_of[subtrahend_id]
                    )
                step.insert(0, "provision", increment.provision)
                rows.append(step)
            tables[stack.row_id] = pd.concat(rows, ignore_index=True)
    return tables


def _reported_channels(retained: ScenarioAggregates) -> dict[Any, Any]:
    """One scenario's four channels together with the whole reform effect.

    The projection keeps the whole effect apart from the four channels it
    decomposes into; the stacked table reports all five, so they are put back in
    one mapping here.
    """
    return dict(retained.channel_totals) | {
        revenue.TOTAL_DIFFERENCE: retained.total_effect
    }


def _rung_measured_against(
    increment: law_stacks.ProvisionIncrement, available: Mapping[str, Any]
) -> str | None:
    """Which rung a provision's step is subtracted from, or None for baseline.

    None means the provision is measured against the run's own baseline, and
    a rung's own numbers are already reform-minus-baseline comparisons, so
    there is nothing to subtract. A later rung whose predecessor is not among
    the finished scenarios cannot be scored.
    """
    if increment.subtrahend_id is None:
        return None
    if increment.subtrahend_id not in available:
        raise _missing_stack_rung_error(increment)
    return increment.subtrahend_id


def _missing_stack_rung_error(
    increment: law_stacks.ProvisionIncrement,
) -> ValueError:
    """Explain why an absent predecessor makes a stacked estimate impossible."""
    return ValueError(
        f"The stacked package is missing rung '{increment.subtrahend_id}', which "
        f"rung '{increment.minuend_id}' needs to estimate provision "
        f"'{increment.provision}'."
    )


def _difference_revenue_estimates(
    minuend: pd.DataFrame, subtrahend: pd.DataFrame
) -> pd.DataFrame:
    """Subtract two revenue-estimate frames after matching their fiscal-year rows.

    Every published estimate is identified by fiscal year, world, and revenue
    line. The frames may arrive in different row orders, so subtraction must
    match those identifiers instead of their positions in the files.
    """
    keys = ["year", "world", "line"]
    minuend_by_key = minuend.set_index(keys)
    subtrahend_by_key = subtrahend.set_index(keys)
    if not minuend_by_key.index.is_unique or not subtrahend_by_key.index.is_unique:
        raise ValueError(
            "Each revenue estimate must have one row for each fiscal year, "
            "world, and revenue line."
        )
    if not minuend_by_key.index.symmetric_difference(subtrahend_by_key.index).empty:
        raise ValueError(
            "The two revenue estimates do not cover the same fiscal years, "
            "worlds, and revenue lines."
        )

    subtrahend_in_minuend_order = subtrahend_by_key.reindex(minuend_by_key.index)
    step = minuend.copy()
    for column in ("dollars", "share_of_gdp"):
        step[column] = (
            minuend_by_key[column] - subtrahend_in_minuend_order[column]
        ).to_numpy()
    return step


def marginal_rate_table(projection: ProjectionRun) -> pd.DataFrame:
    """The dollar-weighted average marginal rate table, as the reporting layer built it."""
    return pd.DataFrame(
        list(projection.mtr_table), columns=list(mtr_table.TABLE_COLUMNS)
    )
