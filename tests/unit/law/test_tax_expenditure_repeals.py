"""The SALT and QBI repeal scenarios for the tax expenditure estimates repeal
the whole deduction, including the floor and minimum OBBBA added."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxmodel.law import compile_law
from taxmodel.law.indexes import index_table_from_frame
from tests.conftest import CONFIG_ROOT, INDEX_TABLE

FOLDER = "tests/tax_expenditures"
YEARS = tuple(range(2025, 2030))


@pytest.fixture(scope="module")
def index_table():
    return index_table_from_frame(pd.read_csv(INDEX_TABLE))


def compiled(scenario, index_table):
    """The scenario's law for 2025-2029. The Booker folders have lint findings
    in their child credit files, which restate indexed amounts without their
    indexation, so they are compiled without the lint."""
    return compile_law(
        CONFIG_ROOT,
        YEARS,
        index_table,
        alternative=f"{FOLDER}/{scenario}",
        lint=not scenario.startswith("booker_"),
    )


def every_filer_every_year(law, field: str) -> np.ndarray:
    """A single-valued field for all four filing statuses and every year."""
    return law.values[field][:, :, 0]


@pytest.mark.parametrize("scenario", ["repeal_salt", "booker_repeal_salt"])
def test_no_filer_keeps_any_salt_deduction(scenario, index_table):
    """Neither the cap nor its $10,000 floor (IRC 164(b)(7)(B)(iii)) is left."""
    law = compiled(scenario, index_table)
    assert np.all(every_filer_every_year(law, "item.salt_limit") == 0.0)
    assert np.all(every_filer_every_year(law, "item.salt_floor") == 0.0)


@pytest.mark.parametrize("scenario", ["repeal_qbi", "booker_repeal_qbi"])
def test_no_filer_keeps_any_qbi_deduction(scenario, index_table):
    """Neither the 20 percent deduction nor the $400 minimum (IRC 199A(i)) is left."""
    law = compiled(scenario, index_table)
    assert np.all(every_filer_every_year(law, "qbi.rate") == 0.0)
    assert np.all(every_filer_every_year(law, "qbi.min_value") == 0.0)
