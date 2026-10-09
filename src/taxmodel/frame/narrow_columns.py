"""Whole-number columns stored as 16-bit integers.

Dollar columns stay 64-bit floating point. Ages, counts, flags and codes are
whole numbers with small ranges, and storing them as 16-bit integers saves
host memory. `narrow_whole_numbers` and `widen_whole_numbers` are the two
directions of the conversion, and the second gives back exactly the numbers
the first was given.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Columns stored as whole numbers
# ---------------------------------------------------------------------------
#
# Ages, the count of dependents, filing status, the year a mortgage was taken
# out, and the yes/no flags are whole numbers in a small range. A 16-bit
# integer holds every one of them, and converting one back to a 64-bit float
# returns the same number bit for bit. `narrow_whole_numbers` checks that a
# column holds only whole numbers in range whenever a population is read, so
# a vintage that breaks the assumption stops the run.
#
# `tips_lh1` and `tips_lh2` are left out: they hold 0 or 1 in the current
# vintage but are shares (the fraction of an earner's tips received in a
# traditionally tipped occupation), and the marginal-rate machinery
# differentiates the tax calculation with respect to them
# (`taxes/individual_income_tax.CONTINUOUS_TAX_INPUTS`). `kg_lt_years_held`
# is a holding period but is not a whole number. The random draws are also
# left out.
NARROW_WHOLE_NUMBER_COLUMNS: tuple[str, ...] = (
    # Whether the record files a return, and whether it is claimed as someone
    # else's dependent.
    "filer",
    "dep_status",
    # Filing status, 1 through 4.
    "filing_status",
    # The first earner's sex, which with age decides how likely an unmarried
    # filer is to be a surviving spouse.
    "male1",
    # The two spouses' ages and blindness. The second spouse's columns are
    # missing when there is no second spouse.
    "age1",
    "age2",
    "blind1",
    "blind2",
    # Counts of dependents, in total and for the earned income tax credit.
    "n_dep",
    "n_dep_eitc",
    # Up to three dependents: each one's age, whether the dependent has a
    # Social Security number, and whether the dependent qualifies for the
    # child tax credit. All three are missing when the household has no
    # dependent in that position.
    "dep_age1",
    "dep_age2",
    "dep_age3",
    "dep_ssn1",
    "dep_ssn2",
    "dep_ssn3",
    "dep_ctc1",
    "dep_ctc2",
    "dep_ctc3",
    # Whether each kind of business income comes from a specified service
    # trade or business, which the qualified business income deduction phases
    # out for high earners. Missing when the household has no such business.
    "sstb_sole_prop",
    "sstb_part",
    "sstb_scorp",
    "sstb_farm",
    # The calendar year of any divorce, which decides whether alimony follows
    # pre- or post-2019 law. Missing for a household with no divorce on
    # record.
    "divorce_year",
    # The calendar year each mortgage was taken out, which decides which
    # balance limit applies to the interest deduction.
    "first_mort_year",
    "second_mort_year",
    # Whether the record is a decedent this year. Tax-Data has no decedents of
    # its own, so this arrives zero and the estate side works in expected
    # deaths.
    "decedent_flag",
)

_NARROW_WHOLE_NUMBER_SET = frozenset(NARROW_WHOLE_NUMBER_COLUMNS)

# The storage type for the columns above. One type for all of them, wide
# enough to hold a calendar year.
NARROW_WHOLE_NUMBER_TYPE = np.int16

# The missing value in a narrow column. An integer array has no NaN, and a
# missing value here carries information (an absent second spouse's age, an
# absent third dependent). This is the most negative 16-bit integer, which no
# real age, count, flag or calendar year can equal.
MISSING_WHOLE_NUMBER = np.iinfo(NARROW_WHOLE_NUMBER_TYPE).min


def is_narrow_whole_number_column(name: str) -> bool:
    """Whether this column is stored as a small whole number rather than a float."""
    return name in _NARROW_WHOLE_NUMBER_SET


def narrow_whole_numbers(name: str, values: np.ndarray) -> np.ndarray:
    """One column's float64 values in the narrow type, refusing an inexact cast.

    A value that is not a whole number, or that will not fit, stops the read
    and names the column and the value. A column already in the narrow type
    is returned as it is.
    """
    values = np.asarray(values)
    if values.dtype == NARROW_WHOLE_NUMBER_TYPE:
        return values
    values = values.astype(np.float64)
    missing = np.isnan(values)
    present = values[~missing]
    if present.size:
        if not np.all(present == np.floor(present)):
            offender = present[present != np.floor(present)][0]
            raise ValueError(
                f"column {name} is stored as a whole number, but this vintage "
                f"holds {offender} in it"
            )
        limits = np.iinfo(NARROW_WHOLE_NUMBER_TYPE)
        # The most negative value the type holds is reserved for the missing
        # value.
        if present.min() <= limits.min or present.max() > limits.max:
            raise ValueError(
                f"column {name} runs from {present.min()} to {present.max()}, "
                f"which does not fit the {NARROW_WHOLE_NUMBER_TYPE.__name__} "
                "the population block stores it in"
            )
    return np.where(missing, MISSING_WHOLE_NUMBER, values).astype(
        NARROW_WHOLE_NUMBER_TYPE
    )


def widen_whole_numbers(values: np.ndarray) -> np.ndarray:
    """A narrow column back as float64, with the missing value back as NaN.

    Every whole number a 16-bit integer holds is exactly representable as a
    64-bit float, so this returns the same numbers the block was given.
    """
    widened = np.asarray(values, dtype=np.float64)
    return np.where(values == MISSING_WHOLE_NUMBER, np.nan, widened)
