"""Which world gets subtracted from which, in the revenue decomposition.

A reform's revenue effect is reported as four channels, each one the step from
one world to the next under reform law:

    first_order          minus baseline              the tax law's own effect
    mechanical_ex_saving minus first_order           corporate and payroll flows
    mechanical           minus mechanical_ex_saving  the prior wealth shortfall
    conventional         minus mechanical            household behavior

and the four add up to conventional minus baseline, the whole effect. The
arithmetic works because the intermediate worlds cancel, which is exactly why
a subtraction written backwards is easy to miss: the four channels would still
be four plausible-looking dollar figures, and only the total would give the
mistake away.

These tests pin the direction down. The fixture below holds population totals
for one year, with a different number in every world, so no two
channels have the same size and no channel is zero. Reversing any one of the
subtractions changes that channel by twice its own value and throws the sum
off the total by the same amount.
"""

from __future__ import annotations

import numpy as np
import pytest

from taxmodel.reporting import revenue
from taxmodel.years import world_names
from taxmodel.years.world_names import ResultGrid, world_cell
from tests.tolerances import assert_agrees_to_the_cent

YEAR = 2026
YEARS = (YEAR,)

# Population tax totals from three households of weight one in each world.
#
#   baseline              1,000 + 5,000 + 20,000 = 26,000
#   first_order           1,200 + 4,400 + 23,000 = 28,600
#   mechanical_ex_saving  1,250 + 4,500 + 23,150 = 28,900
#   mechanical            1,240 + 4,460 + 23,100 = 28,800
#   conventional          1,260 + 4,490 + 23,200 = 28,950
#
# The middle household pays less under the reform's law while the other two pay
# more, so the law's effect is a real netting rather than three numbers all
# pointing the same way.
LIABILITY_BY_WORLD = {
    "baseline": np.array([26_000.0]),
    "first_order": np.array([28_600.0]),
    "mechanical_ex_saving": np.array([28_900.0]),
    "mechanical": np.array([28_800.0]),
    "conventional": np.array([28_950.0]),
}

# What each channel comes to, worked out from the totals above.
LAW_EFFECT = 2_600.0  # 28,600 - 26,000
INCOME_FLOW_EFFECT = 300.0  # 28,900 - 28,600
WEALTH_SHORTFALL_EFFECT = -100.0  # 28,800 - 28,900
BEHAVIOR_EFFECT = 150.0  # 28,950 - 28,800
WHOLE_EFFECT = 2_950.0  # 28,950 - 26,000, and also 2,600 + 300 - 100 + 150


def _grid() -> ResultGrid:
    """The five worlds of the fixture, in the one year."""
    grid = ResultGrid()
    for world, liability in LIABILITY_BY_WORLD.items():
        frame, law = world_cell(world)
        grid.put(frame, law, YEAR, {"liability": liability})
    return grid


def _world_total(world: str) -> float:
    """One world's population tax total in the fixture year."""
    frame, law = world_cell(world)
    totals = revenue.totals_by_year(_grid(), frame, law, YEARS)
    return float(totals["liability"][0])


def test_each_channel_subtracts_the_two_worlds_its_name_says():
    """The declared minuend and subtrahend of every channel, in world names.

    This states the direction as a fact about the constants, before any
    arithmetic: the second world named is the one being subtracted.
    """
    named = [
        (
            world_names.world_name(*channel.minuend),
            world_names.world_name(*channel.subtrahend),
        )
        for channel in revenue.CHANNEL_DIFFERENCES
    ]
    assert named == [
        ("first_order", "baseline"),
        ("mechanical_ex_saving", "first_order"),
        ("mechanical", "mechanical_ex_saving"),
        ("conventional", "mechanical"),
    ]
    assert world_names.world_name(*revenue.TOTAL_DIFFERENCE.minuend) == "conventional"
    assert world_names.world_name(*revenue.TOTAL_DIFFERENCE.subtrahend) == "baseline"

    # Each channel's reported name spells out the same subtraction in the same
    # order, so a column header in an output file cannot say one thing while
    # the arithmetic behind it does another.
    assert [
        revenue.channel_name(channel) for channel in revenue.CHANNEL_DIFFERENCES
    ] == [
        "first_order_minus_baseline",
        "mechanical_ex_saving_minus_first_order",
        "mechanical_minus_mechanical_ex_saving",
        "conventional_minus_mechanical",
    ]


def test_each_channel_comes_out_at_the_size_and_sign_the_fixture_puts_there():
    """The computed channels against the differences worked out by hand.

    Three of the four channels raise revenue and one lowers it, and no two have
    the same magnitude, so a channel computed backwards, or against the wrong
    world, lands on a number that is not in this list.
    """
    channels = revenue.channel_totals_by_year(_grid(), YEARS)
    expected = {
        revenue.CHANNEL_DIFFERENCES[0]: LAW_EFFECT,
        revenue.CHANNEL_DIFFERENCES[1]: INCOME_FLOW_EFFECT,
        revenue.CHANNEL_DIFFERENCES[2]: WEALTH_SHORTFALL_EFFECT,
        revenue.CHANNEL_DIFFERENCES[3]: BEHAVIOR_EFFECT,
    }
    for channel, amount in expected.items():
        assert channels[channel]["liability"] == pytest.approx([amount])

    # And each channel equals the difference of the two world totals it names,
    # taken in that order.
    for channel in revenue.CHANNEL_DIFFERENCES:
        minuend = _world_total(world_names.world_name(*channel.minuend))
        subtrahend = _world_total(world_names.world_name(*channel.subtrahend))
        assert channels[channel]["liability"][0] == pytest.approx(minuend - subtrahend)


def test_the_four_channels_add_up_to_the_whole_reform_effect_to_the_cent():
    """The decomposition adds back to conventional minus baseline.

    A cent is this model's standard for a test of an accounting identity
    (the reasoning is in tests/tolerances.py).
    """
    grid = _grid()
    channels = revenue.channel_totals_by_year(grid, YEARS)
    added_up = sum(totals["liability"] for totals in channels.values())
    whole = revenue.difference_totals_by_year(grid, revenue.TOTAL_DIFFERENCE, YEARS)[
        "liability"
    ]

    assert whole == pytest.approx([WHOLE_EFFECT])
    assert_agrees_to_the_cent(
        added_up,
        whole,
        "the four channels of the reform's revenue effect against the whole effect",
    )


def test_the_income_flow_channel_is_mechanical_ex_saving_minus_first_order():
    """The one mutation this package exists to catch: that subtraction reversed.

    The two worlds are `mechanical_ex_saving` and `first_order`. The channel is the first minus
    the second — what the corporate-tax and employer-payroll income adjustments
    add on top of the reform's law.

    In the fixture the two worlds are 28,900 and 28,600, so the channel is +300
    and the reversal would be -300. Both halves are stated: that the model
    gives +300, and that with the subtraction written the other way the four
    channels would add to 2,350 instead of the 2,950 that conventional minus
    baseline actually is, short by twice the channel. Because the two worlds
    genuinely differ here, a sign flip cannot slip through by landing on zero.
    """
    grid = _grid()
    channel = revenue.CHANNEL_DIFFERENCES[1]
    assert revenue.channel_name(channel) == "mechanical_ex_saving_minus_first_order"

    computed = revenue.difference_totals_by_year(grid, channel, YEARS)["liability"][0]
    flows = _world_total("mechanical_ex_saving")
    static = _world_total("first_order")
    assert flows != static, "the fixture has to make the two worlds differ"
    assert computed == pytest.approx(flows - static)
    assert computed == pytest.approx(INCOME_FLOW_EFFECT)
    assert computed != pytest.approx(static - flows)

    # What the reversal would do to the published total.
    reversed_sum = (
        LAW_EFFECT + (static - flows) + WEALTH_SHORTFALL_EFFECT + BEHAVIOR_EFFECT
    )
    assert reversed_sum == pytest.approx(WHOLE_EFFECT - 2 * INCOME_FLOW_EFFECT)
    assert reversed_sum != pytest.approx(WHOLE_EFFECT)
