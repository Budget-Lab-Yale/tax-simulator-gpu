"""Statutory amounts in the default law, compiled with the golden index table.

Each test states the amount the statute or IRS guidance gives and checks that
the default law compiles to it. Filing statuses are 1 single, 2 joint, 3
married filing separately, 4 head of household.
"""

from __future__ import annotations

import pandas as pd
import pytest

from taxmodel.law import compile_law
from taxmodel.law.indexes import index_table_from_frame
from tests.conftest import CONFIG_ROOT, INDEX_TABLE

YEARS = (2024, 2025, 2026, 2027)
FILING_STATUSES = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def default_law():
    index_table = index_table_from_frame(pd.read_csv(INDEX_TABLE))
    return compile_law(CONFIG_ROOT, YEARS, index_table)


def by_status(law, field: str, year: int) -> dict[int, float]:
    """One single-valued field in one year, by filing status."""
    row = law.values[field][law.years.index(year), :, 0]
    return {status: float(row[i]) for i, status in enumerate(FILING_STATUSES)}


def test_the_2025_standard_deduction_is_the_obbba_amount(default_law):
    """IRC 63(c)(7) as amended by OBBBA sec. 70102; IRS, tax year 2026
    inflation adjustments (Rev. Proc. 2025-32): $15,750 single and separate,
    $31,500 joint, $23,625 head of household."""
    assert by_status(default_law, "std.value", 2025) == {
        1: 15_750.0,
        2: 31_500.0,
        3: 15_750.0,
        4: 23_625.0,
    }


def test_the_2026_standard_deduction_is_indexed_from_2024(default_law):
    """Rev. Proc. 2025-32: $16,100 single and separate, $32,200 joint,
    $24,150 head of household, which the golden index table reproduces."""
    assert by_status(default_law, "std.value", 2026) == {
        1: 16_100.0,
        2: 32_200.0,
        3: 16_100.0,
        4: 24_150.0,
    }
