"""The wealth-bathtub diagnostics, on a tub small enough to fill by hand.

Two years, two ages, two net-worth percentile bins, and the ten households of
``conftest.py``. Only the top bin holds any shortfall, which is what a wealth
tax should look like, and only the two richest households take any of it out
of their balance sheets.

The tub, in dollars:

                      year 1                  year 2
    cell (age 0, bin 1)   1,000 shortfall        2,050 shortfall
    cell (age 1, bin 1)   3,000 shortfall        6,120 shortfall
    total                 4,000                  8,170
    of which new saving   4,000                  4,000
    carried from last year        0                4,170

So the carried shortfall grew by 4,170 / 4,000 = 1.0425 between the two
years, and the first year has nothing to carry in.
"""

from __future__ import annotations

import numpy as np
import pytest

from taxmodel.reporting.wealth_diagnostics import (
    WealthShortfallHistory,
    cell_rates_by_year,
    haircut_summary_by_year,
    shortfall_by_age_by_year,
    shortfall_by_percentile_bin_by_year,
    shortfall_totals_by_year,
    wealth_diagnostics,
)

YEARS = np.array([2026, 2027])
AGES = np.array([40, 41])

# The largest fraction of gross assets one year's haircut may move.
HAIRCUT_CAP = 0.2

CELL_SHORTFALL = np.array(
    [
        [[0.0, 1_000.0], [0.0, 3_000.0]],  # 2026
        [[0.0, 2_050.0], [0.0, 6_120.0]],  # 2027
    ]
)
CELL_INFLOW = np.array(
    [
        [[0.0, 1_000.0], [0.0, 3_000.0]],  # the first year is all fresh saving
        [[0.0, 1_000.0], [0.0, 3_000.0]],
    ]
)
GROWTH_FACTOR = np.array(
    [
        [[1.05, 1.05], [1.05, 1.04]],
        [[1.05, 1.05], [1.04, 1.04]],
    ]
)
CAPITAL_INCOME_PER_DOLLAR_OF_ASSETS = np.array(
    [
        [[0.03, 0.03], [0.03, 0.02]],
        [[0.03, 0.03], [0.02, 0.02]],
    ]
)
CELL_NET_WORTH = np.array(
    [
        [[0.0, 100_000.0], [0.0, 300_000.0]],
        [[0.0, 100_000.0], [0.0, 300_000.0]],
    ]
)

# Ten household slots, each weighing ten, so the population weighs 100.
WEIGHT = np.full((2, 10), 10.0)

# In 2026 the whole shortfall reaches the two richest households: 100 dollars
# each of ten in slot 8 and 300 each of ten in slot 9 is 1,000 + 3,000. In
# 2027 the cell holding 2,050 has no household with positive net worth left to
# take it out of, so only slot 9's 6,120 is allocated and the rest leaks away.
ALLOCATED_DOLLARS = np.zeros((2, 10))
ALLOCATED_DOLLARS[0, 8] = 100.0
ALLOCATED_DOLLARS[0, 9] = 300.0
ALLOCATED_DOLLARS[1, 9] = 612.0

HAIRCUT_FRACTION = np.zeros((2, 10))
HAIRCUT_FRACTION[0, 8] = 0.01
HAIRCUT_FRACTION[0, 9] = 0.05
HAIRCUT_FRACTION[1, 8] = -0.02  # a tax cut saved into wealth scales it up
HAIRCUT_FRACTION[1, 9] = HAIRCUT_CAP  # pinned at the cap


def _history() -> WealthShortfallHistory:
    return WealthShortfallHistory(
        years=YEARS,
        ages=AGES,
        cell_shortfall=CELL_SHORTFALL,
        cell_inflow=CELL_INFLOW,
        growth_factor=GROWTH_FACTOR,
        capital_income_per_dollar_of_assets=CAPITAL_INCOME_PER_DOLLAR_OF_ASSETS,
        cell_net_worth=CELL_NET_WORTH,
        allocated_dollars=ALLOCATED_DOLLARS,
        haircut_fraction=HAIRCUT_FRACTION,
        weight=WEIGHT,
        haircut_cap=HAIRCUT_CAP,
    )


def test_the_totals_say_how_much_is_in_the_tub_and_where_it_came_from():
    totals = shortfall_totals_by_year(_history())
    assert totals["total_shortfall_dollars"] == pytest.approx([4_000.0, 8_170.0])
    assert totals["inflow_dollars"] == pytest.approx([4_000.0, 4_000.0])
    assert totals["carried_in_dollars"] == pytest.approx([0.0, 4_170.0])
    # The first year has nothing to carry in, so there is no growth to report.
    assert np.isnan(totals["implied_growth_of_the_carried_shortfall"][0])
    assert totals["implied_growth_of_the_carried_shortfall"][1] == pytest.approx(
        4_170.0 / 4_000.0
    )


def test_the_tub_is_reported_against_the_wealth_it_is_drawn_from():
    totals = shortfall_totals_by_year(_history())
    # 400,000 of weighted positive net worth in both years.
    assert totals["cell_net_worth_dollars"] == pytest.approx(
        [400_000.0, 400_000.0]
    )
    assert totals["shortfall_as_share_of_net_worth"] == pytest.approx(
        [4_000.0 / 400_000.0, 8_170.0 / 400_000.0]
    )


def test_shortfall_that_no_household_can_absorb_is_reported_as_a_leak():
    totals = shortfall_totals_by_year(_history())
    # 2026: 10 x 100 + 10 x 300 = 4,000, the whole shortfall.
    # 2027: 10 x 612 = 6,120 of a shortfall of 8,170.
    assert totals["allocated_to_households_dollars"] == pytest.approx(
        [4_000.0, 6_120.0]
    )
    assert totals["share_of_shortfall_reaching_households"] == pytest.approx(
        [1.0, 6_120.0 / 8_170.0]
    )


def test_the_cell_rates_are_averaged_over_the_cells_holding_the_money():
    rates = cell_rates_by_year(_history())
    # 2026: 1,000 in a cell with G of 1.05 and 3,000 in a cell with G of 1.04.
    # (1,000 x 1.05 + 3,000 x 1.04) / 4,000 = 1.0425.
    assert rates["average_growth_factor"][0] == pytest.approx(1.0425)
    # Capital income per dollar of assets: (1,000 x 0.03 + 3,000 x 0.02) / 4,000.
    assert rates["average_capital_income_per_dollar_of_assets"][0] == pytest.approx(
        90.0 / 4_000.0
    )
    # Only the two top-bin cells hold anything, in either year.
    assert rates["filled_cells"] == pytest.approx([2.0, 2.0])
    assert rates["smallest_growth_factor_in_a_filled_cell"][0] == pytest.approx(1.04)
    assert rates["largest_growth_factor_in_a_filled_cell"][0] == pytest.approx(1.05)


def test_an_empty_cells_rates_do_not_enter_the_average():
    """Both empty cells carry a growth factor of 1.05 and hold nothing.

    If they counted, 2026's average growth factor would be pulled above
    1.0425 towards 1.05. It is not.
    """
    rates = cell_rates_by_year(_history())
    assert rates["average_growth_factor"][0] < 1.05


def test_the_shortfall_is_reported_by_age_and_by_percentile_bin():
    by_age = shortfall_by_age_by_year(_history())
    assert list(by_age["ages"]) == [40, 41]
    assert by_age["shortfall_dollars"][0] == pytest.approx([1_000.0, 3_000.0])

    by_bin = shortfall_by_percentile_bin_by_year(_history())
    # Nothing in the bottom bin; the whole tub sits in the top one.
    assert by_bin["shortfall_dollars"][0] == pytest.approx([0.0, 4_000.0])
    assert by_bin["shortfall_dollars"][1] == pytest.approx([0.0, 8_170.0])


def test_the_haircut_summary_counts_the_households_pinned_at_the_cap():
    haircut = haircut_summary_by_year(_history())
    # Nobody is at the cap in 2026; slot 9 is in 2027, one household of ten
    # out of a population of a hundred.
    assert haircut["share_of_households_at_the_cap"] == pytest.approx([0.0, 0.1])
    # Two of ten slots have a haircut in each year.
    assert haircut["share_of_households_with_any_haircut"] == pytest.approx([0.2, 0.2])
    # 2026: (10 x 0.01 + 10 x 0.05) / 20 = 0.03.
    # 2027: (10 x -0.02 + 10 x 0.20) / 20 = 0.09.
    assert haircut["average_haircut_fraction"] == pytest.approx([0.03, 0.09])
    assert haircut["largest_haircut_fraction"] == pytest.approx([0.05, 0.2])
    # A tax cut saved into wealth scales the balance sheet up, so a negative
    # haircut is a real answer and not an error.
    assert haircut["smallest_haircut_fraction"] == pytest.approx([0.0, -0.02])


def test_every_diagnostic_table_is_produced_together():
    tables = wealth_diagnostics(_history())
    assert set(tables) == {
        "shortfall totals",
        "cell rates",
        "shortfall by age",
        "shortfall by percentile bin",
        "haircut",
    }
    for table in tables.values():
        assert len(table["year"]) == 2


def test_a_history_whose_pieces_disagree_about_the_run_is_refused():
    history = _history()
    wrong = WealthShortfallHistory(
        years=history.years,
        ages=history.ages,
        cell_shortfall=history.cell_shortfall,
        cell_inflow=np.zeros((2, 2, 3)),  # three percentile bins, not two
        growth_factor=history.growth_factor,
        capital_income_per_dollar_of_assets=history.capital_income_per_dollar_of_assets,
        cell_net_worth=history.cell_net_worth,
        allocated_dollars=history.allocated_dollars,
        haircut_fraction=history.haircut_fraction,
        weight=history.weight,
        haircut_cap=history.haircut_cap,
    )
    with pytest.raises(ValueError, match="cell_inflow"):
        shortfall_totals_by_year(wrong)


def test_a_run_in_which_nothing_ever_drains_reports_no_rates_rather_than_zero():
    """With no reform there is no shortfall, and the rates describe nothing."""
    empty = WealthShortfallHistory(
        years=YEARS,
        ages=AGES,
        cell_shortfall=np.zeros((2, 2, 2)),
        cell_inflow=np.zeros((2, 2, 2)),
        growth_factor=GROWTH_FACTOR,
        capital_income_per_dollar_of_assets=CAPITAL_INCOME_PER_DOLLAR_OF_ASSETS,
        cell_net_worth=CELL_NET_WORTH,
        allocated_dollars=np.zeros((2, 10)),
        haircut_fraction=np.zeros((2, 10)),
        weight=WEIGHT,
        haircut_cap=HAIRCUT_CAP,
    )
    rates = cell_rates_by_year(empty)
    assert np.all(np.isnan(rates["average_growth_factor"]))
    assert rates["filled_cells"] == pytest.approx([0.0, 0.0])
    totals = shortfall_totals_by_year(empty)
    assert totals["total_shortfall_dollars"] == pytest.approx([0.0, 0.0])
    assert np.all(np.isnan(totals["share_of_shortfall_reaching_households"]))
