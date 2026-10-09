"""The stacked table, and the identity it rests on.

The identity: the provisions' increments add up to the effect of the scenario
with every provision in force, in every year and in every channel. It is exact
arithmetic — the intermediate rungs cancel — so these tests compare with `==`
and not to a tolerance.

The runs here are hand-written channel totals rather than real simulations.
That is deliberate: the point of these tests is the subtraction and the
arrangement, and a fabricated ladder makes both readable. The identity is
checked again on real numbers from the calculator chain in
tests/unit/years/test_stacked_projection.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from taxmodel.config.law_stacks import ProvisionIncrement
from taxmodel.reporting import revenue
from taxmodel.reporting.revenue import (
    CHANNEL_DIFFERENCES,
    TOTAL_DIFFERENCE,
    Frame,
    Law,
)
from taxmodel.reporting.stacked import (
    REPORTED_CHANNELS,
    TOTAL_ROW,
    provision_increments,
    stacked_table,
)
from taxmodel.years import world_names
from tests.conftest import stacked_table_row
from tests.tolerances import assert_agrees_to_the_cent

YEARS = (2030, 2031, 2032)

# The result column a revenue table reports.
COLUMN = "liab_iit"


def channel_totals(law_effect, flows, wealth, behavior):
    """One run's channel totals: four channels and their sum, by year.

    Each argument is one year series for one channel, in billions of made-up
    dollars. The whole reform effect is the four added up, which is what a real
    run reports too — the intermediate frames cancel.
    """
    law_effect, flows = np.asarray(law_effect), np.asarray(flows)
    wealth, behavior = np.asarray(wealth), np.asarray(behavior)
    channels = dict(
        zip(CHANNEL_DIFFERENCES, (law_effect, flows, wealth, behavior), strict=True)
    )
    totals = {difference: {COLUMN: values} for difference, values in channels.items()}
    totals[TOTAL_DIFFERENCE] = {COLUMN: law_effect + flows + wealth + behavior}
    return totals


@pytest.fixture
def three_rung_ladder():
    """Three cumulative runs, each a whole-package-so-far effect on baseline.

    Rung 1 is the first provision alone, rung 2 is the first two together, rung
    3 is all three. The numbers are made up but they grow the way a real ladder
    does: each rung raises more than the one below it.
    """
    return {
        "p1": channel_totals(
            [100.0, 110.0, 120.0], [10.0, 11.0, 12.0], [1.0, 2.0, 3.0], [-5.0, -6.0, -7.0]
        ),
        "p2": channel_totals(
            [160.0, 175.0, 190.0], [14.0, 15.0, 16.0], [2.0, 4.0, 6.0], [-9.0, -11.0, -13.0]
        ),
        "p3": channel_totals(
            [230.0, 250.0, 271.0], [21.0, 22.0, 24.0], [3.0, 7.0, 10.0], [-15.0, -18.0, -21.0]
        ),
    }


@pytest.fixture
def stacked_increments():
    """What `expand_stack` produces for a three-provision stacked package."""
    return (
        ProvisionIncrement(provision="p1", minuend_id="p1", subtrahend_id=None),
        ProvisionIncrement(provision="p2", minuend_id="p2", subtrahend_id="p1"),
        ProvisionIncrement(provision="p3", minuend_id="p3", subtrahend_id="p2"),
    )


# ---------------------------------------------------------------------------
# The increments
# ---------------------------------------------------------------------------


def test_the_first_provision_is_its_own_run_measured_against_baseline(
    three_rung_ladder, stacked_increments
):
    increments = provision_increments(
        three_rung_ladder, stacked_increments, COLUMN, YEARS
    )
    law_effect = CHANNEL_DIFFERENCES[0]
    assert np.array_equal(increments["p1"][law_effect], [100.0, 110.0, 120.0])


def test_a_later_provision_is_the_step_up_from_the_rung_below_it(
    three_rung_ladder, stacked_increments
):
    increments = provision_increments(
        three_rung_ladder, stacked_increments, COLUMN, YEARS
    )
    law_effect = CHANNEL_DIFFERENCES[0]
    assert np.array_equal(increments["p2"][law_effect], [60.0, 65.0, 70.0])
    assert np.array_equal(increments["p3"][law_effect], [70.0, 75.0, 81.0])


def test_a_missing_run_names_the_scenario_and_the_ones_supplied(
    three_rung_ladder, stacked_increments
):
    del three_rung_ladder["p2"]
    with pytest.raises(KeyError, match="no run was supplied for scenario 'p2'"):
        provision_increments(three_rung_ladder, stacked_increments, COLUMN, YEARS)


def test_a_run_covering_the_wrong_years_is_refused(three_rung_ladder, stacked_increments):
    three_rung_ladder["p2"] = channel_totals([1.0], [1.0], [1.0], [1.0])
    with pytest.raises(ValueError, match="reports 1 years"):
        provision_increments(three_rung_ladder, stacked_increments, COLUMN, YEARS)


def test_a_missing_result_column_is_refused(three_rung_ladder, stacked_increments):
    with pytest.raises(KeyError, match="no result column 'liab_pr'"):
        provision_increments(three_rung_ladder, stacked_increments, "liab_pr", YEARS)


# ---------------------------------------------------------------------------
# The identity
# ---------------------------------------------------------------------------


def test_the_increments_sum_exactly_to_the_combined_scenario_per_year_per_channel(
    three_rung_ladder, stacked_increments
):
    """The acceptance criterion: an identity, not a tolerance.

    The last rung of the ladder is the scenario with every provision in force,
    so its reported effect is the whole package's. Adding the provisions'
    increments has to give back exactly that, in every year and in every
    channel, because the intermediate rungs appear once with each sign.
    """
    increments = provision_increments(
        three_rung_ladder, stacked_increments, COLUMN, YEARS
    )
    combined = three_rung_ladder["p3"]

    for channel in REPORTED_CHANNELS:
        summed = sum(
            increments[increment.provision][channel]
            for increment in stacked_increments
        )
        assert_agrees_to_the_cent(
            summed,
            combined[channel][COLUMN],
            f"the {revenue.channel_name(channel)} channel's increments do not "
            "add up to the combined scenario",
        )


def test_the_tables_total_row_is_the_combined_scenarios_effect(
    three_rung_ladder, stacked_increments
):
    table = stacked_table(
        three_rung_ladder, stacked_increments, YEARS, COLUMN
    )
    combined = three_rung_ladder["p3"]
    for channel in REPORTED_CHANNELS:
        assert_agrees_to_the_cent(
            stacked_table_row(table, TOTAL_ROW, channel),
            combined[channel][COLUMN],
            f"the {revenue.channel_name(channel)} channel's Total row is not "
            "the combined scenario's effect",
        )


def test_the_identity_holds_even_when_the_increments_change_sign(stacked_increments):
    """A provision that loses revenue on top of the ones below it still adds up."""
    ladder = {
        "p1": channel_totals([100.0, 100.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
        # The second provision more than gives back the first's revenue.
        "p2": channel_totals([-30.0, -40.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
        "p3": channel_totals([50.0, 45.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
    }
    increments = provision_increments(ladder, stacked_increments, COLUMN, (2030, 2031))
    law_effect = CHANNEL_DIFFERENCES[0]
    assert np.array_equal(increments["p2"][law_effect], [-130.0, -140.0])
    summed = sum(increments[i.provision][law_effect] for i in stacked_increments)
    assert_agrees_to_the_cent(
        summed,
        ladder["p3"][law_effect][COLUMN],
        "the increments do not add up once one of them changes sign",
    )


# ---------------------------------------------------------------------------
# The table's shape
# ---------------------------------------------------------------------------


def test_the_table_is_wide_in_years_and_long_in_channel_and_provision(
    three_rung_ladder, stacked_increments
):
    table = stacked_table(
        three_rung_ladder, stacked_increments, YEARS, COLUMN
    )

    assert table.header == (
        "channel",
        "provision",
        "2030",
        "2031",
        "2032",
    )
    # One block per channel: a row per provision, then that channel's Total.
    rows_per_block = len(stacked_increments) + 1
    assert len(table.rows) == rows_per_block * len(REPORTED_CHANNELS)

    # The combined total leads the table, before the channels it breaks into.
    first_block = table.rows[:rows_per_block]
    assert [row[1] for row in first_block] == ["p1", "p2", "p3", TOTAL_ROW]
    assert {row[0] for row in first_block} == {"conventional_minus_baseline"}


def test_the_channel_column_uses_the_five_world_names(
    three_rung_ladder, stacked_increments
):
    """The channel column names each channel by the five world names."""
    table = stacked_table(
        three_rung_ladder, stacked_increments, YEARS, COLUMN
    )
    seen: list[str] = []
    for row in table.rows:
        if row[0] not in seen:
            seen.append(row[0])
    assert seen == [
        "conventional_minus_baseline",
        "first_order_minus_baseline",
        "mechanical_ex_saving_minus_first_order",
        "mechanical_minus_mechanical_ex_saving",
        "conventional_minus_mechanical",
    ]


def test_the_world_names_cover_every_evaluated_cell():
    assert world_names.world_name(Frame.RAW, Law.BASELINE) == "baseline"
    assert world_names.world_name(Frame.RAW, Law.REFORM) == "first_order"
    assert world_names.world_name(Frame.FLOWS, Law.REFORM) == "mechanical_ex_saving"
    assert world_names.world_name(Frame.MECHANICAL, Law.REFORM) == "mechanical"
    assert world_names.world_name(Frame.BEHAVIOR, Law.REFORM) == "conventional"


def test_a_cell_that_is_never_evaluated_has_no_world_name():
    with pytest.raises(KeyError, match="not one of the five"):
        world_names.world_name(Frame.BEHAVIOR, Law.BASELINE)


def test_the_table_writes_a_csv_a_reader_can_open(
    three_rung_ladder, stacked_increments, tmp_path
):
    table = stacked_table(
        three_rung_ladder, stacked_increments, YEARS, COLUMN
    )
    path = tmp_path / "stacked.csv"
    table.write_csv(path)
    lines = path.read_text().splitlines()
    assert lines[0] == "channel,provision,2030,2031,2032"
    assert len(lines) == 1 + len(table.rows)
