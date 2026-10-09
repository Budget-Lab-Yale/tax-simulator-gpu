"""Shared pieces for the runscript driver's tests.

The driver's job is to walk a runscript, run what it names, and write the run
directory. What the model computes is tested everywhere else, so these tests
substitute a stand-in for the projection itself: a results object of the right
shape carrying zeros. That keeps the tests off the compiled scan — which does
not fit on a login node — while running the real planning, the real provenance
resolution, the real table writers and the real manifests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxmodel.config import runscript
from taxmodel.law.compile import FILING_STATUSES, CompiledLaw, FieldSpec
from taxmodel.reporting import (
    assemble,
    distribution,
    mtr_table,
    receipts,
    revenue,
    tax_totals,
    wealth_diagnostics,
)
from taxmodel.run.scenarios import ScenarioResult
from taxmodel.years import wealth_arithmetic, world_names

# The stand-in population: five households, which is enough for the
# household-shaped wealth-channel arrays to have a shape at all.
STAND_IN_HOUSEHOLDS = 5
# The net-worth percentile bins the wealth channel cuts its cells into. The
# economy default layer's `wealth.n_pctiles` is 100.
PERCENTILE_BINS = 100

# The five evaluated (household frame, tax law) cells, in world order.
EVALUATED_CELLS = tuple(world_names.WORLD_CELLS.values())

# The runscript's header row, spelled the way the reader defines the columns.
RUNSCRIPT_HEADER = ",".join(runscript.COLUMNS) + "\n"


def stand_in_tax_totals(years: tuple[int, ...]) -> dict[object, tax_totals.WorldTotals]:
    """Zero-filled weighted totals of the contract's shape, one set per world.

    Every column the four files report is present and zero, so the tests can
    check what gets written without running the model.
    """
    n_years = len(years)
    columns = tax_totals.every_totalled_column()
    return {
        cell: tax_totals.WorldTotals(
            sums={name: np.zeros(n_years) for name in columns},
            nonzero_counts={name: np.zeros(n_years) for name in columns},
            n_tax_units=np.zeros(n_years),
            n_returns=np.zeros(n_years),
            n_returns_dep=np.zeros(n_years),
            n_dep=np.zeros(n_years),
        )
        for cell in EVALUATED_CELLS
    }


def stand_in_projection(years: tuple[int, ...]):
    """A projection-shaped result carrying zeros, for testing what gets written."""
    n_years = len(years)
    # The projection publishes a weighted total per world and year for each of
    # these columns. `assemble.REVENUE_COLUMNS` is what a real run totals, and
    # every column `receipts.MODEL_COLUMNS` asks for is among them; both lists
    # are named here so a column dropped from either one is noticed.
    totaled_columns = (*assemble.REVENUE_COLUMNS, *receipts.MODEL_COLUMNS)
    revenue_columns = {name: np.zeros(n_years) for name in totaled_columns}
    cell_shape = (n_years, len(wealth_arithmetic.age_grid()), PERCENTILE_BINS)
    household_shape = (n_years, STAND_IN_HOUSEHOLDS)
    history = wealth_diagnostics.WealthShortfallHistory(
        years=np.asarray(years),
        ages=np.asarray(wealth_arithmetic.age_grid()),
        cell_shortfall=np.zeros(cell_shape),
        cell_inflow=np.zeros(cell_shape),
        growth_factor=np.ones(cell_shape),
        capital_income_per_dollar_of_assets=np.zeros(cell_shape),
        cell_net_worth=np.ones(cell_shape),
        allocated_dollars=np.zeros(household_shape),
        haircut_fraction=np.zeros(household_shape),
        weight=np.ones(household_shape),
        haircut_cap=0.9,
    )
    return assemble.ProjectionRun(
        years=years,
        grid=world_names.ResultGrid(),
        totals={
            cell: {name: values.copy() for name, values in revenue_columns.items()}
            for cell in EVALUATED_CELLS
        },
        channel_totals={
            difference: {name: values.copy() for name, values in revenue_columns.items()}
            for difference in revenue.CHANNEL_DIFFERENCES
        },
        total_effect={name: values.copy() for name, values in revenue_columns.items()},
        wealth_history=history,
        mtr_table=stand_in_marginal_rate_rows(years),
        tax_totals=stand_in_tax_totals(years),
        compiled_law=stand_in_compiled_law(years, STAND_IN_STANDARD_DEDUCTION_REFORM),
        compiled_baseline_law=stand_in_compiled_law(
            years, STAND_IN_STANDARD_DEDUCTION_BASELINE
        ),
        distribution_table=np.zeros(
            (
                n_years,
                len(distribution.CORPORATE_SPREADS),
                len(distribution.TAX_ROWS),
                len(distribution.GROUPS),
                len(distribution.value_column_names()),
            )
        ),
    )


# A real run gets the compiled law from the scenario; the stand-in supplies a
# law with two fields in it, which is enough to exercise the writers without
# the compiled scan.

# The standard deduction the stand-in law states, in dollars. Nothing in the
# model reads it here -- it is a recognisable number to look for on the way
# back out of tax_law.csv, and the reform's differs from the baseline's so the
# two blocks of the file can be told apart.
STAND_IN_STANDARD_DEDUCTION_BASELINE = 15_000.0
STAND_IN_STANDARD_DEDUCTION_REFORM = 25_000.0


def stand_in_compiled_law(years: tuple[int, ...], standard_deduction: float):
    """A compiled law with one dollar field and one on/off switch in it."""
    shape = (len(years), len(FILING_STATUSES), 1)
    values = {
        "deductions.standard": np.full(shape, standard_deduction),
        # Whether the law requires the IRS to certify a claimed child before
        # the earned income credit is paid. Off in the stand-in.
        "eitc.parent_precert": np.zeros(shape),
    }
    schema = {
        name: FieldSpec(
            name=name,
            parameter=name.split(".")[0],
            subparameter=name.split(".")[1],
            dtype="float64",
            units="dollars" if name.endswith("standard") else "indicator",
            slots_used=1,
            varies_by_filing_status=name.endswith("standard"),
            indexed=False,
            from_mapper=False,
            differentiable=name.endswith("standard"),
            padding="nan",
        )
        for name in values
    }
    return CompiledLaw(
        alternative="stand-in",
        years=years,
        max_brackets=1,
        values=values,
        active={name: np.ones(shape, dtype=bool) for name in values},
        schema=schema,
        replacements=[],
    )


def stand_in_marginal_rate_rows(years: tuple[int, ...]) -> list[dict[str, object]]:
    """One average-marginal-rate row per year, carrying the table's own columns."""
    return [
        {
            "year": int(year),
            "group": "Overall",
            "priced_dollar": "wages",
            "taxes_included": "income and payroll tax",
            "base_dollars": 0.0,
            "average_marginal_rate_baseline_law": 0.0,
            "average_marginal_rate_reform_law": 0.0,
            "change_in_average_marginal_rate": 0.0,
        }
        for year in years
    ]


def assert_marginal_rate_columns_are_the_reporting_layers():
    """The stand-in rows carry exactly the columns the real table carries."""
    written = set(stand_in_marginal_rate_rows((2026,))[0])
    assert written == set(mtr_table.TABLE_COLUMNS)


# How many scenarios the stand-in group runner deals into each group. Two, so
# that a run of three or more scenarios exercises several groups and a
# remainder group — the dealing, per-group writing and retention the driver
# does for real even when the compute is a stand-in.
STAND_IN_GROUP_SIZE = 2

# The manifest's counters section for a run whose results came from a test's
# stand-in rather than from the batch runner.
UNMEASURED_COUNTERS = {
    "measured": False,
    "why_not": (
        "no compiled program ran: these results did not come through the "
        "batch runner (years/loop.py), which is where the compile and memory "
        "counters are measured"
    ),
    "compiled_programs": None,
    "families": [],
    "peak_device_bytes_in_use": None,
    "compile_seconds": None,
    "execute_seconds": None,
}


@pytest.fixture
def stand_in_runner():
    """A group runner standing in for the real compute, with the same seam.

    The real group runner (`scenarios.run_scenario_groups`) hands each finished
    group of `ScenarioResult`s to ``deliver`` as soon as it is done and
    returns the run's counters. This one does the same, dealing the scenarios
    into groups of STAND_IN_GROUP_SIZE, without touching JAX, law compilation
    or the population reader.
    """

    def run(scenarios, provenance, *, deliver, population, config_root):
        for start in range(0, len(scenarios), STAND_IN_GROUP_SIZE):
            deliver(
                tuple(
                    ScenarioResult(
                        spec=spec,
                        projection=stand_in_projection(tuple(spec.row.years)),
                        provenance=provenance[spec.id],
                    )
                    for spec in scenarios[start : start + STAND_IN_GROUP_SIZE]
                )
            )
        return dict(UNMEASURED_COUNTERS)

    return run


def read_table(path):
    """Read one of a run's tables, whichever form it was written in.

    Most published tables are CSV. The exceptions are the ones too big to read
    by eye: the per-household detail files, the distribution table and the
    marginal rate table, which are Parquet.
    """
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
