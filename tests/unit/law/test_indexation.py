"""Indexation arithmetic.

The rounding goldens in tests/golden/fixtures/law/rounding_goldens.csv are
frozen cases: made-up base values, index ratios and rounding rules, each with
its expected indexed value.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from taxmodel.law.errors import LawError
from taxmodel.law.indexation import IndexTable, apply_indexation, index_ratios
from tests.conftest import GOLDEN_LAW_FIXTURES


def col(values):
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def read_number(text: str) -> float:
    return math.nan if text in ("NA", "", "NaN") else float(text)


# -- rounding ---------------------------------------------------------------


def test_the_rounding_goldens():
    """Every floor, ceiling and round case in the goldens file, to the last cent."""
    with open(GOLDEN_LAW_FIXTURES / "rounding_goldens.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 500

    got = apply_indexation(
        col([read_number(r["base_value"]) for r in rows]),
        col([read_number(r["i_index"]) for r in rows]),
        col([read_number(r["i_direction"]) for r in rows]),
        col([read_number(r["i_increment"]) for r in rows]),
    )[:, 0]
    want = np.array([read_number(r["value"]) for r in rows])

    mismatch = [
        (rows[i], got[i], want[i])
        for i in range(len(rows))
        if not ((math.isnan(got[i]) and math.isnan(want[i])) or got[i] == want[i])
    ]
    assert not mismatch, mismatch[:5]


def test_a_tie_goes_to_the_even_multiple():
    """Rounding is half-to-even, and the tree hits ties by construction.

    $12.50 at a $25 increment is exactly half a bracket step, and rounds to
    zero, not to $25. Rounding half away from zero here moves the bracket a
    full increment.
    """
    base = col([12.5, 37.5, 62.5, 87.5])
    got = apply_indexation(base, col([1, 1, 1, 1]), col([0, 0, 0, 0]), col([25] * 4))
    assert got[:, 0].tolist() == [0.0, 50.0, 50.0, 100.0]


def test_the_three_directions():
    base = col([9525.0, 9525.0, 9525.0])
    got = apply_indexation(base, col([1.1, 1.1, 1.1]), col([-1, 0, 1]), col([25, 25, 25]))
    # 9525 * 1.1 = 10477.5 -> 419.1 increments
    assert got[:, 0].tolist() == [10475.0, 10475.0, 10500.0]


def test_an_unbounded_limit_stays_unbounded():
    got = apply_indexation(col([math.inf]), col([1.02]), col([-1]), col([100]))
    assert got[0, 0] == math.inf


def test_a_slot_with_no_rounding_rule_keeps_its_base_value():
    """The donut-hole bracket: indexation is off for that slot alone."""
    got = apply_indexation(
        col([60600.0, 400000.0]),
        col([1.5, 1.5]),
        col([1, math.nan]),
        col([300, math.nan]),
    )
    assert got[0, 0] == 90900.0
    assert got[1, 0] == 400000.0


def test_an_unrecognised_rounding_direction_is_refused():
    with pytest.raises(LawError, match="i_direction"):
        apply_indexation(col([1000.0]), col([1.0]), col([7]), col([100]))


# -- the index chain --------------------------------------------------------


@pytest.fixture
def two_percent() -> IndexTable:
    return IndexTable.from_rows([("cpi", y, 0.02) for y in range(2000, 2040)])


def test_the_ratio_is_lagged_a_year(two_percent):
    """Next year's tables are set from this year's inflation.

    With prices growing 2% a year and a 2020 base year, the 2021 amount is the
    base value: growth is measured through 2020, and the 2020 level over the
    2020 level is one. 2022 carries a single year of growth.
    """
    measure = {y: "cpi" for y in range(2000, 2031)}
    years = tuple(range(2018, 2031))
    ratios = index_ratios(measure, np.full((len(years), 1), 2020.0), two_percent, years, "x")
    assert ratios[years.index(2021), 0] == pytest.approx(1.0)
    assert ratios[years.index(2022), 0] == pytest.approx(1.02)
    assert ratios[years.index(2025), 0] == pytest.approx(1.02**4)


def test_a_frozen_measure_stops_the_chain_for_good(two_percent):
    """Writing the measure out is how the TCJA froze a phaseout, and it sticks."""
    measure = {y: ("cpi" if y < 2020 else None) for y in range(2000, 2031)}
    years = tuple(range(2018, 2031))
    ratios = index_ratios(measure, np.full((len(years), 1), 2015.0), two_percent, years, "x")
    assert ratios[years.index(2019), 0] == pytest.approx(1.02**3)
    assert math.isnan(ratios[years.index(2025), 0])


def test_naming_a_measure_the_index_data_does_not_carry_stops_the_compile(two_percent):
    measure = {y: "awi" for y in range(2000, 2031)}
    years = tuple(range(2018, 2031))
    with pytest.raises(LawError, match="no growth rate"):
        index_ratios(measure, np.full((len(years), 1), 2015.0), two_percent, years, "pr.oasdi")


def test_a_window_past_the_end_of_the_projection_stops_the_compile(two_percent):
    """Better to stop than revert the parameter to its nominal base-year level."""
    measure = {y: "cpi" for y in range(2000, 2051)}
    years = tuple(range(2018, 2051))
    with pytest.raises(LawError, match="past the end"):
        index_ratios(measure, np.full((len(years), 1), 2015.0), two_percent, years, "ord.brackets")
