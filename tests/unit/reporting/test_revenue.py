"""Annual population totals and the three measures of a revenue change."""

from __future__ import annotations

import numpy as np
import pytest

from taxmodel.reporting import revenue
from taxmodel.years.world_names import WORLD_CELLS, Frame, Law, ResultGrid

YEARS = (2026, 2027)

# Population tax under baseline law, already weighted and summed by the year loop.
LIABILITY_RAW_BASELINE = np.array([7_740_000.0])

# 2027 is 2026 doubled, so every 2027 answer is twice the 2026 answer.
SECOND_YEAR_MULTIPLE = 2.0


def _grid() -> ResultGrid:
    """Baseline law on the household frame as Tax-Data delivers it, both years."""
    grid = ResultGrid()
    for position, year in enumerate(YEARS):
        scale = SECOND_YEAR_MULTIPLE**position
        grid.put(
            Frame.RAW, Law.BASELINE, year, {"liability": LIABILITY_RAW_BASELINE * scale}
        )
    return grid


def test_population_totals_are_preserved_in_the_requested_year_order():
    totals = revenue.totals_by_year(_grid(), Frame.RAW, Law.BASELINE, YEARS[::-1])
    np.testing.assert_array_equal(totals["liability"], [15_480_000.0, 7_740_000.0])


def test_asking_for_a_world_the_runner_did_not_evaluate_says_which_one():
    grid = ResultGrid()
    grid.put(Frame.RAW, Law.BASELINE, 2026, {"liability": LIABILITY_RAW_BASELINE})
    # The whole reform effect ends at the conventional world, which this grid
    # does not hold, and the message says so by name.
    with pytest.raises(KeyError, match="conventional world"):
        revenue.difference_totals_by_year(grid, revenue.TOTAL_DIFFERENCE, [2026])


def test_household_arrays_cannot_be_mistaken_for_population_totals():
    grid = ResultGrid()
    grid.put(Frame.RAW, Law.BASELINE, 2026, {"liability": np.array([100.0, 200.0])})
    with pytest.raises(ValueError, match="one population total"):
        revenue.totals_by_year(grid, Frame.RAW, Law.BASELINE, [2026])


def test_differences_read_only_the_requested_columns():
    grid = ResultGrid()
    grid.put(
        Frame.RAW,
        Law.BASELINE,
        2026,
        {"liability": np.array([100.25]), "baseline_only": np.zeros(3)},
    )
    grid.put(
        Frame.BEHAVIOR,
        Law.REFORM,
        2026,
        {"liability": np.array([90.125]), "reform_only": np.zeros(4)},
    )
    difference = revenue.difference_totals_by_year(
        grid, revenue.TOTAL_DIFFERENCE, [2026], columns=["liability"]
    )
    assert tuple(difference) == ("liability",)
    np.testing.assert_array_equal(difference["liability"], [-10.125])


def test_unselected_totals_refuse_columns_that_change_between_years():
    grid = ResultGrid()
    grid.put(Frame.RAW, Law.BASELINE, 2026, {"liability": np.array([100.0])})
    grid.put(Frame.RAW, Law.BASELINE, 2027, {"other": np.array([100.0])})
    with pytest.raises(KeyError, match="change from year to year"):
        revenue.totals_by_year(grid, Frame.RAW, Law.BASELINE, YEARS)


def test_unselected_differences_refuse_columns_that_change_between_worlds():
    grid = ResultGrid()
    grid.put(Frame.RAW, Law.BASELINE, 2026, {"liability": np.array([100.0])})
    grid.put(Frame.BEHAVIOR, Law.REFORM, 2026, {"other": np.array([100.0])})
    with pytest.raises(KeyError, match="different result columns"):
        revenue.difference_totals_by_year(grid, revenue.TOTAL_DIFFERENCE, [2026])


def test_empty_years_have_no_totals():
    assert revenue.totals_by_year(ResultGrid(), Frame.RAW, Law.BASELINE, []) == {}


def test_totals_and_differences_equal_the_grid_arithmetic():
    # Fractional, negative and large totals over a full projection window and
    # every revenue channel.
    years = tuple(range(2026, 2056))
    rng = np.random.default_rng(2026)
    grid = ResultGrid()
    for frame, law in WORLD_CELLS.values():
        for year in years:
            grid.put(
                frame,
                law,
                year,
                {
                    "liability": rng.uniform(-1e13, 1e13, size=1),
                    "receipts": rng.uniform(-1e6, 1e6, size=1),
                },
            )
    for frame, law in WORLD_CELLS.values():
        totals = revenue.totals_by_year(grid, frame, law, years)
        for column in totals:
            expected = [grid.get(frame, law, year)[column][0] for year in years]
            np.testing.assert_array_equal(totals[column], expected)
    for channel in (*revenue.CHANNEL_DIFFERENCES, revenue.TOTAL_DIFFERENCE):
        totals = revenue.difference_totals_by_year(grid, channel, years)
        for column in totals:
            expected = [
                grid.get(*channel.minuend, year)[column][0]
                - grid.get(*channel.subtrahend, year)[column][0]
                for year in years
            ]
            np.testing.assert_array_equal(totals[column], expected)
