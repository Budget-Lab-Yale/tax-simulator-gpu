"""Mortality above the top-coded age, from the Social Security life tables.

Tax-Data top-codes age at 80, so it can say how likely an 80-year-old
household is to die this year but nothing about an 85-year-old. The
selling choice needs those older ages: a taxpayer decides whether to hold
an appreciated asset by weighing the tax on selling against the chance of
dying and passing it on, and that comparison needs a last age of life to
work backward from. The Social Security life tables run out at age 119,
which the selling choice takes as the end of the age grid. (The published
probability of dying at 119 is high but not one, so the last age is a
stopping point the model imposes, not one the table states.)

These are one person's chances of dying. The age cells below 81 measure
mortality per tax unit, so the rates here are put on that convention before the
selling choice reads them
(``assemble.mortality_above_the_top_code_for_tax_units``).

The two tables here are the period life tables published with the 2024
Trustees Report under the Alternative 2 (intermediate) assumptions, one for
men and one for women. The model does not distinguish the sexes, so it uses
the simple average of the male and female probability of dying within the
year at each age and calendar year, rather than the sex mix of asset holders
at that age, which the model does not carry.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .stock_inputs import KG_AGE_MAX

# Period life tables from the 2024 OASDI Trustees Report, Alternative 2
# assumptions.
MALE_LIFE_TABLE = Path("resources/PerLifeTables_M_Alt2_TR2024.csv")
FEMALE_LIFE_TABLE = Path("resources/PerLifeTables_F_Alt2_TR2024.csv")

# Tax-Data reports one top-coded age for everyone at or above it, so the life
# table is needed from one year above that on up. The tables stop at 119, and
# the selling choice takes that as the last age of life.
FIRST_AGE_ABOVE_TOP_CODE = KG_AGE_MAX + 1
OLDEST_AGE = 119

# The published files carry four lines of title and footnote before the header
# row, and their column names ("q(x)" and so on) are actuarial notation that
# pandas would keep verbatim. Take the three columns that matter by position:
# calendar year, age, and the probability of dying within the year.
LEADING_TEXT_LINES = 4
YEAR_COLUMN = 0
AGE_COLUMN = 1
PROBABILITY_OF_DEATH_COLUMN = 2


def mortality_above_the_top_code(years: Sequence[int]) -> np.ndarray:
    """The chance of dying within the year, by age and calendar year.

    Returns an array with one row per age from 81 through 119 and one column
    per year in `years`, in the order given. Each entry is the average of the
    male and female probability.

    Every age and year asked for has to be in both tables. A gap is an error:
    a missing older age would shorten the age grid the selling choice works
    back along.
    """
    ages = list(range(FIRST_AGE_ABOVE_TOP_CODE, OLDEST_AGE + 1))
    years = [int(year) for year in years]

    male = _read_one_table(MALE_LIFE_TABLE, ages, years)
    female = _read_one_table(FEMALE_LIFE_TABLE, ages, years)
    return 0.5 * male + 0.5 * female


def _read_one_table(
    path: str | Path, ages: Sequence[int], years: Sequence[int]
) -> np.ndarray:
    """One published table, cut down to the ages and years asked for."""
    raw = pd.read_csv(path, skiprows=LEADING_TEXT_LINES)
    table = pd.DataFrame(
        {
            "year": raw.iloc[:, YEAR_COLUMN].astype(int),
            "age": raw.iloc[:, AGE_COLUMN].astype(int),
            "probability_of_death": raw.iloc[:, PROBABILITY_OF_DEATH_COLUMN].astype(
                float
            ),
        }
    ).set_index(["age", "year"])["probability_of_death"]

    wanted = pd.MultiIndex.from_product([ages, years], names=["age", "year"])
    missing = wanted.difference(table.index)
    if len(missing) > 0:
        first = missing[0]
        raise ValueError(
            f"{path} has no mortality rate for age {first[0]} in {first[1]} "
            f"({len(missing)} age-year combinations are missing in all)"
        )
    return table.reindex(wanted).to_numpy().reshape(len(ages), len(years))
