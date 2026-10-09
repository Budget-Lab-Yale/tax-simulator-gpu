"""The distribution file's layout: each number of the device table lands in
the row of its year, corporate spread, group and tax, and in the column of its
name."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from taxmodel.reporting import distribution, run_tables
from tests.unit.run.conftest import stand_in_projection

YEARS = (2026, 2027)


def _with_table(table):
    return dataclasses.replace(stand_in_projection(YEARS), distribution_table=table)


def test_each_number_lands_in_the_row_and_column_it_belongs_to():
    shape = (
        len(YEARS),
        len(distribution.CORPORATE_SPREADS),
        len(distribution.TAX_ROWS),
        len(distribution.GROUPS),
        len(distribution.value_column_names()),
    )
    # Every entry distinct, so a misplaced number cannot match by accident.
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    table = run_tables.distribution_table(_with_table(values))

    assert list(table.columns) == list(distribution.table_columns())
    indexed = table.set_index(list(distribution.KEY_COLUMNS))
    last_tax = len(distribution.TAX_ROWS) - 1
    last_group = len(distribution.GROUPS) - 1
    for year_index, spread_index, tax_index, group_index, column in (
        (0, 0, 0, 0, "average_tax_change"),
        (1, 1, 3, 17, "net_change"),
        (0, 1, 2, 5, "income_cutoff"),
        (1, 0, last_tax, last_group, "conventional_accrual_etr"),
    ):
        dimension, group = distribution.GROUPS[group_index]
        tax = distribution.TAX_ROWS[tax_index]
        spread = distribution.CORPORATE_SPREADS[spread_index]
        column_index = distribution.value_column_names().index(column)
        key = (dimension, group, tax, spread, YEARS[year_index])
        assert indexed.loc[key, column] == (
            values[year_index, spread_index, tax_index, group_index, column_index]
        )


def test_a_run_without_a_table_gives_no_table():
    assert run_tables.distribution_table(_with_table(None)) is None


def test_a_table_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="distribution table is shaped"):
        run_tables.distribution_table(_with_table(np.zeros((1, 1, 1, 1, 1))))
