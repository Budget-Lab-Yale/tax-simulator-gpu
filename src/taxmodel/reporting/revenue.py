"""Revenue by year: population totals for each (frame, law) cell and the
differences between cells.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from taxmodel.years import world_names
from taxmodel.years.world_names import Frame, Law, ResultGrid

Array = np.ndarray

# A cell of the results grid: which household frame, under which law.
Cell = tuple[Frame, Law]


@dataclass(frozen=True)
class CellDifference:
    """One comparison, written as one grid cell minus another.

    A comparison is named by its two (frame, law) cells. ``description`` says
    in words what the difference measures; no code branches on it.
    """

    minuend: Cell
    subtrahend: Cell
    description: str


# The four channels a reform's revenue effect is decomposed into. Each one is
# the step from the previous frame to the next, all under reform law, so the
# four of them add up to the whole effect (`TOTAL_DIFFERENCE` below). The
# cells are defined in years/world_names.py.
_LAW_EFFECT = CellDifference(
    minuend=(Frame.RAW, Law.REFORM),
    subtrahend=(Frame.RAW, Law.BASELINE),
    description="the tax law's effect on households as Tax-Data delivers them",
)
_INCOME_FLOW_EFFECT = CellDifference(
    minuend=(Frame.FLOWS, Law.REFORM),
    subtrahend=(Frame.RAW, Law.REFORM),
    description="the corporate-tax and employer-payroll income adjustments",
)
_WEALTH_SHORTFALL_EFFECT = CellDifference(
    minuend=(Frame.MECHANICAL, Law.REFORM),
    subtrahend=(Frame.FLOWS, Law.REFORM),
    description="the prior year's wealth shortfall marking the balance sheet down",
)
_BEHAVIOR_EFFECT = CellDifference(
    minuend=(Frame.BEHAVIOR, Law.REFORM),
    subtrahend=(Frame.MECHANICAL, Law.REFORM),
    description="households changing what they do",
)

CHANNEL_DIFFERENCES: tuple[CellDifference, ...] = (
    _LAW_EFFECT,
    _INCOME_FLOW_EFFECT,
    _WEALTH_SHORTFALL_EFFECT,
    _BEHAVIOR_EFFECT,
)


def channel_name(difference: CellDifference) -> str:
    """The reported name of one comparison: the two worlds it subtracts."""
    minuend = world_names.world_name(*difference.minuend)
    subtrahend = world_names.world_name(*difference.subtrahend)
    return f"{minuend}_minus_{subtrahend}"


TOTAL_DIFFERENCE = CellDifference(
    minuend=(Frame.BEHAVIOR, Law.REFORM),
    subtrahend=(Frame.RAW, Law.BASELINE),
    description="the whole reform effect, after income flows, wealth and behavior",
)


def totals_by_year(
    grid: ResultGrid,
    frame: Frame,
    law: Law,
    years: Sequence[int],
    columns: Sequence[str] | None = None,
) -> dict[str, Array]:
    """Annual population totals of one grid cell's result columns.

    The year loop has already weighted and summed the households. Each grid
    entry holds one population total per column. Read the requested columns
    in the requested year order; with no selection, read every column.
    """
    rows = [grid.get(frame, law, year) for year in years]
    if not rows:
        return {}
    names = list(rows[0]) if columns is None else list(columns)
    if columns is None and any(row.keys() != rows[0].keys() for row in rows[1:]):
        raise KeyError("the result columns change from year to year")
    totals = {}
    for name in names:
        values = [np.asarray(row[name], dtype=np.float64) for row in rows]
        if any(value.size != 1 for value in values):
            raise ValueError(f"column {name!r} must hold one population total per year")
        totals[name] = np.asarray([value.item() for value in values], dtype=np.float64)
    return totals


def difference_totals_by_year(
    grid: ResultGrid,
    difference: CellDifference,
    years: Sequence[int],
    columns: Sequence[str] | None = None,
) -> dict[str, Array]:
    """Subtract the selected annual totals of the two named cells."""
    minuend = totals_by_year(grid, *difference.minuend, years, columns)
    subtrahend = totals_by_year(grid, *difference.subtrahend, years, columns)
    if minuend.keys() != subtrahend.keys():
        raise KeyError("the compared cells carry different result columns")
    return {name: minuend[name] - subtrahend[name] for name in minuend}


def channel_totals_by_year(
    grid: ResultGrid,
    years: Sequence[int],
    columns: Sequence[str] | None = None,
) -> dict[CellDifference, dict[str, Array]]:
    """Population totals for each channel of the reform's revenue effect.

    Because each channel is one frame minus the frame before it, all under
    reform law, the intermediate cells cancel and the four channel totals add
    to the total effect year by year.
    """
    return {
        difference: difference_totals_by_year(grid, difference, years, columns)
        for difference in CHANNEL_DIFFERENCES
    }
