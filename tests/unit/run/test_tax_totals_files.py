"""The four tax-specific totals files: their columns, their rows, their numbers.

The model itself is tested elsewhere. What is checked here is that the four
files carry the columns they promise, one row per world per year, and that the
number in a cell is the number the run handed over, to the cent.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from taxmodel.reporting import tax_totals
from taxmodel.run import output_directory, scenarios
from taxmodel.run.plan import plan_run
from taxmodel.years import world_names
from tests.tolerances import assert_agrees_to_the_cent
from tests.unit.run.conftest import (
    EVALUATED_CELLS,
    RUNSCRIPT_HEADER,
    stand_in_projection,
)

YEARS = (2026, 2027)


@pytest.fixture
def finished_run(tmp_path, stand_in_runner):
    """One baseline scenario, run through the real writers with a stand-in model."""
    runscript = tmp_path / "smoke.csv"
    runscript.write_text(RUNSCRIPT_HEADER + "baseline,default,default,default,2026:2027,\n")
    plan = plan_run(runscript, output_root=tmp_path / "runs")
    return scenarios.execute_run(
        plan,
        population_root=tmp_path / "population",
        group_runner=stand_in_runner,
    )


def scenario_folder(run_directory):
    return run_directory / "scenarios" / "baseline"


# ---------------------------------------------------------------------------
# The columns each file promises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_name", "expected_columns"),
    [
        ("form1040.csv", tax_totals.form_1040_columns()),
        ("payroll.csv", tax_totals.payroll_columns()),
        ("estate.csv", list(tax_totals.ESTATE_COLUMNS)),
        ("wealth.csv", list(tax_totals.WEALTH_COLUMNS)),
    ],
)
def test_each_totals_file_carries_the_columns_it_promises(
    finished_run, file_name, expected_columns
):
    written = pd.read_csv(scenario_folder(finished_run) / file_name)
    assert list(written.columns) == expected_columns


@pytest.mark.parametrize("file_name", output_directory.TAX_TOTALS_FILES)
def test_each_totals_file_has_one_row_per_world_per_year(finished_run, file_name):
    written = pd.read_csv(scenario_folder(finished_run) / file_name)
    assert len(written) == len(EVALUATED_CELLS) * len(YEARS)
    assert list(written["world"].unique()) == list(world_names.WORLD_NAMES)
    assert sorted(written["year"].unique()) == list(YEARS)


def test_form_1040_counts_itemizers_and_does_not_add_the_flag_up_as_dollars():
    """The itemizer count stays; the itemizing flag is not summed as dollars."""
    columns = tax_totals.form_1040_columns()
    assert "n_itemizing" in columns
    assert "itemizing" not in columns


def test_the_marginal_rate_columns_are_gone_from_form_1040():
    """Form 1040 carries no mean marginal rate columns."""
    columns = tax_totals.form_1040_columns()
    assert not [name for name in columns if name.startswith("mtr_")]


# ---------------------------------------------------------------------------
# The numbers in the file are the numbers the run handed over
# ---------------------------------------------------------------------------


def varied_totals(years):
    """Weighted totals with a different number in every place one can go.

    Every world, column and year gets its own value, and the values carry
    fractions of a dollar, so a builder that read the wrong world, the wrong
    year or the wrong column would be caught rather than matching by luck.
    """
    n_years = len(years)
    columns = tax_totals.every_totalled_column()
    built = {}
    for world_position, cell in enumerate(EVALUATED_CELLS):
        offset = 1_000_000.0 * (world_position + 1)
        sums = {}
        counts = {}
        for column_position, name in enumerate(columns):
            base = offset + 1_000.0 * (column_position + 1)
            sums[name] = base + np.arange(n_years) + 0.37
            counts[name] = base / 2.0 + np.arange(n_years) + 0.11
        built[cell] = tax_totals.WorldTotals(
            sums=sums,
            nonzero_counts=counts,
            n_tax_units=offset + np.arange(n_years) + 0.5,
            n_returns=offset + 1.0 + np.arange(n_years) + 0.5,
            n_returns_dep=offset + 2.0 + np.arange(n_years) + 0.5,
            n_dep=offset + 3.0 + np.arange(n_years) + 0.5,
        )
    return built


@pytest.fixture
def projection_with_varied_totals():
    return dataclasses.replace(
        stand_in_projection(YEARS),
        tax_totals=varied_totals(YEARS),
    )


def written_and_read_back(tmp_path, table, name):
    """Round-trip one table through the CSV a run would write."""
    path = tmp_path / name
    table.to_csv(path, index=False)
    return pd.read_csv(path)


def one_world(written, world):
    return written[written["world"] == world].sort_values("year")


def test_form_1040_reports_the_totals_the_run_handed_over(
    tmp_path, projection_with_varied_totals
):
    handed_over = projection_with_varied_totals.tax_totals
    written = written_and_read_back(
        tmp_path, tax_totals.form_1040_table(projection_with_varied_totals), "f.csv"
    )
    for world, cell in world_names.WORLD_CELLS.items():
        totals = handed_over[cell]
        rows = one_world(written, world)
        assert_agrees_to_the_cent(
            rows["n_returns"], totals.n_returns, f"{world}: returns filed"
        )
        assert_agrees_to_the_cent(
            rows["n_dep"], totals.n_dep, f"{world}: dependents on filed returns"
        )
        assert_agrees_to_the_cent(
            rows["n_simple_filers"],
            totals.sums[tax_totals.SIMPLE_FILER],
            f"{world}: simple filers",
        )
        for name in tax_totals.FORM_1040_VARIABLES:
            assert_agrees_to_the_cent(
                rows[f"n_{name}"],
                totals.nonzero_counts[name],
                f"{world}: returns with a nonzero {name}",
            )
            if name not in tax_totals.FORM_1040_COUNT_ONLY_VARIABLES:
                assert_agrees_to_the_cent(
                    rows[name], totals.sums[name], f"{world}: {name} dollars"
                )


def test_payroll_reports_the_totals_the_run_handed_over(
    tmp_path, projection_with_varied_totals
):
    handed_over = projection_with_varied_totals.tax_totals
    written = written_and_read_back(
        tmp_path, tax_totals.payroll_table(projection_with_varied_totals), "p.csv"
    )
    for world, cell in world_names.WORLD_CELLS.items():
        totals = handed_over[cell]
        rows = one_world(written, world)
        assert_agrees_to_the_cent(
            rows["n_tax_units"], totals.n_tax_units, f"{world}: tax units"
        )
        for name in tax_totals.PAYROLL_VARIABLES:
            assert_agrees_to_the_cent(
                rows[f"n_{name}"],
                totals.nonzero_counts[name],
                f"{world}: records with a nonzero {name}",
            )
            assert_agrees_to_the_cent(
                rows[name], totals.sums[name], f"{world}: {name} dollars"
            )


def test_estate_and_wealth_report_the_totals_the_run_handed_over(
    tmp_path, projection_with_varied_totals
):
    handed_over = projection_with_varied_totals.tax_totals
    estate = written_and_read_back(
        tmp_path, tax_totals.estate_table(projection_with_varied_totals), "e.csv"
    )
    wealth = written_and_read_back(
        tmp_path, tax_totals.wealth_table(projection_with_varied_totals), "w.csv"
    )
    for world, cell in world_names.WORLD_CELLS.items():
        totals = handed_over[cell]
        estate_rows = one_world(estate, world)
        assert_agrees_to_the_cent(
            estate_rows[tax_totals.ESTATE_TAX_EXPECTED],
            totals.sums[tax_totals.ESTATE_TAX_EXPECTED],
            f"{world}: expected estate tax",
        )
        assert_agrees_to_the_cent(
            estate_rows[tax_totals.ESTATE_RETURNS_EXPECTED],
            totals.sums[tax_totals.ESTATE_RETURNS_EXPECTED],
            f"{world}: expected taxable estates",
        )
        wealth_rows = one_world(wealth, world)
        assert_agrees_to_the_cent(
            wealth_rows["wealth_tax"],
            totals.sums[tax_totals.WEALTH_LIABILITY],
            f"{world}: wealth tax",
        )
        assert_agrees_to_the_cent(
            wealth_rows["wealth_returns"],
            totals.nonzero_counts[tax_totals.WEALTH_LIABILITY],
            f"{world}: units owing wealth tax",
        )


def test_a_column_the_run_did_not_total_is_named_in_the_error(
    projection_with_varied_totals,
):
    """A missing column says which file wanted it, rather than a bare KeyError."""
    totals = projection_with_varied_totals.tax_totals
    stripped = {
        cell: dataclasses.replace(
            value, sums={k: v for k, v in value.sums.items() if k != "agi"}
        )
        for cell, value in totals.items()
    }
    projection = dataclasses.replace(projection_with_varied_totals, tax_totals=stripped)
    with pytest.raises(KeyError, match="form1040.csv needs the 'agi' column"):
        tax_totals.form_1040_table(projection)
