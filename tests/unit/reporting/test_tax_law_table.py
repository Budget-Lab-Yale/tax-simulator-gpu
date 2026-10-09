"""tax_law.csv: the law a scenario ran, written out and read back."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxmodel.law.compile import FILING_STATUSES, CompiledLaw, FieldSpec
from taxmodel.reporting import tax_law_table

YEARS = (2026, 2027)

# The two standard deduction amounts the fixture law states for a single
# filer, and the joint amount, which is twice the single one.
SINGLE_STANDARD_DEDUCTION = {2026: 15_000.0, 2027: 16_000.0}
REFORM_STANDARD_DEDUCTION = {2026: 25_000.0, 2027: 26_000.0}
MARRIED_FILING_JOINTLY = 2


def a_law(standard_deduction: dict[int, float]) -> CompiledLaw:
    """A compiled law holding one dollar field that varies by year and status."""
    shape = (len(YEARS), len(FILING_STATUSES), 1)
    values = np.zeros(shape)
    for year_index, year in enumerate(YEARS):
        for status_index, status in enumerate(FILING_STATUSES):
            amount = standard_deduction[year]
            values[year_index, status_index, 0] = (
                amount * 2.0 if status == MARRIED_FILING_JOINTLY else amount
            )
    spec = FieldSpec(
        name="deductions.standard",
        parameter="deductions",
        subparameter="standard",
        dtype="float64",
        units="dollars",
        slots_used=1,
        varies_by_filing_status=True,
        indexed=False,
        from_mapper=False,
        differentiable=True,
        padding="nan",
    )
    return CompiledLaw(
        alternative="fixture",
        years=YEARS,
        max_brackets=1,
        values={"deductions.standard": values},
        active={"deductions.standard": np.ones(shape, dtype=bool)},
        schema={"deductions.standard": spec},
        replacements=[],
    )


def test_the_law_table_carries_both_laws_year_by_year_and_status_by_status(tmp_path):
    table = tax_law_table.tax_law_table(
        a_law(REFORM_STANDARD_DEDUCTION), a_law(SINGLE_STANDARD_DEDUCTION)
    )
    written = tmp_path / "tax_law.csv"
    table.to_csv(written, index=False, na_rep="NA")
    read_back = pd.read_csv(written)

    # Two laws, two years, four filing statuses.
    assert len(read_back) == 2 * len(YEARS) * len(FILING_STATUSES)
    assert list(read_back.columns) == [
        "law",
        "year",
        "filing_status",
        "deductions.standard",
    ]

    def amount(law: str, year: int, filing_status: int) -> float:
        row = read_back[
            (read_back["law"] == law)
            & (read_back["year"] == year)
            & (read_back["filing_status"] == filing_status)
        ]
        return float(row["deductions.standard"].iloc[0])

    assert amount("baseline", 2026, 1) == SINGLE_STANDARD_DEDUCTION[2026]
    assert amount("baseline", 2027, 1) == SINGLE_STANDARD_DEDUCTION[2027]
    assert amount("baseline", 2026, MARRIED_FILING_JOINTLY) == (
        2.0 * SINGLE_STANDARD_DEDUCTION[2026]
    )
    assert amount("reform", 2026, 1) == REFORM_STANDARD_DEDUCTION[2026]
    assert amount("reform", 2027, MARRIED_FILING_JOINTLY) == (
        2.0 * REFORM_STANDARD_DEDUCTION[2027]
    )
