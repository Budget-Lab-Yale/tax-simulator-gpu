"""Cell-minus-cell differences of a result grid, for tests.

The model reports each revenue channel by subtracting population totals in
`reporting/revenue.py`. These helpers subtract two grid cells column by
column for one year so that tests can check the four channels add up to the
whole reform effect.
"""

from __future__ import annotations

import numpy as np

from taxmodel.years.world_names import (
    Frame,
    Law,
    ResultGrid,
    YearResults,
    describe_cell,
)

ResultDifference = dict[str, np.ndarray]


def difference(
    grid: ResultGrid,
    year: int,
    minuend_frame: Frame,
    minuend_law: Law,
    subtrahend_frame: Frame,
    subtrahend_law: Law,
) -> ResultDifference:
    """Subtract every result column of one evaluated cell from another's."""
    minuend: YearResults = grid.get(minuend_frame, minuend_law, year)
    subtrahend: YearResults = grid.get(subtrahend_frame, subtrahend_law, year)
    if minuend.keys() != subtrahend.keys():
        raise KeyError(
            f"{describe_cell(minuend_frame, minuend_law)} and "
            f"{describe_cell(subtrahend_frame, subtrahend_law)} in {year} "
            "carry different result columns; "
            f"missing from the first: {sorted(subtrahend.keys() - minuend.keys())}; "
            f"missing from the second: {sorted(minuend.keys() - subtrahend.keys())}"
        )
    return {
        name: np.asarray(minuend[name]) - np.asarray(subtrahend[name])
        for name in minuend
    }


def first_order_minus_baseline(grid: ResultGrid, year: int) -> ResultDifference:
    """The tax-law effect on the raw household frame."""
    return difference(grid, year, Frame.RAW, Law.REFORM, Frame.RAW, Law.BASELINE)


def mechanical_ex_saving_minus_first_order(
    grid: ResultGrid, year: int
) -> ResultDifference:
    """The corporate and employer-payroll income-flow effect."""
    return difference(grid, year, Frame.FLOWS, Law.REFORM, Frame.RAW, Law.REFORM)


def mechanical_minus_mechanical_ex_saving(
    grid: ResultGrid, year: int
) -> ResultDifference:
    """The effect of the prior wealth shortfall on current taxes."""
    return difference(grid, year, Frame.MECHANICAL, Law.REFORM, Frame.FLOWS, Law.REFORM)


def conventional_minus_mechanical(grid: ResultGrid, year: int) -> ResultDifference:
    """The tax effect of household behavior."""
    return difference(
        grid, year, Frame.BEHAVIOR, Law.REFORM, Frame.MECHANICAL, Law.REFORM
    )


def conventional_minus_baseline(grid: ResultGrid, year: int) -> ResultDifference:
    """The complete reform effect after income, wealth, and behavior."""
    return difference(grid, year, Frame.BEHAVIOR, Law.REFORM, Frame.RAW, Law.BASELINE)
