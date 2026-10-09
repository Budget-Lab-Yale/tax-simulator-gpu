"""A stacked reform run end to end: the identity on real calculator numbers.

Three projections over the same three hand-written households and the same two
years, sharing one baseline: the first rung raises every ordinary rate two
points, the second rung cuts the standard deduction on top of it, and a third
run scores the two provisions' combined law directly. The two provisions'
increments then have to add up to the combined run's effect, in every year and
in every channel, exactly.

Both provisions raise the tax of any household with taxable income, which the
three test households have; a provision that moved none of them would leave the
increments at zero and the identity would hold without proving anything, so one
test checks that they are not.

Everything comes from the repository's own configuration tree — the two
provision folders and the stack file that names them are real files, not
fixtures — so the whole path is exercised: the fold, the schedule lint inside
the compile, the compiled law, the calculator chain, and the cross-run
subtraction.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from taxmodel.config.law_stacks import expand_stack
from taxmodel.reporting import assemble, revenue
from taxmodel.reporting.stacked import (
    REPORTED_CHANNELS,
    TOTAL_ROW,
    provision_increments,
    stacked_table,
)
from taxmodel.scenario import provenance
from tests.conftest import CONFIG_ROOT, stacked_table_row
from tests.tolerances import assert_agrees_to_the_cent
from tests.unit.config.law_stack_helpers import combined_law, stack_file_path

from .test_integration import YEARS, run, synthetic_population

STACK = "tests_two_provisions"

# The income tax liability net of refundable credits: the column both provisions
# move, and the one a revenue table reports.
COLUMN = "liab_iit_net"


@pytest.fixture(scope="module")
def baseline_scenario():
    return provenance.load_scenario(YEARS)


@pytest.fixture(scope="module")
def population():
    return synthetic_population()


@pytest.fixture(scope="module")
def expanded():
    return expand_stack(stack_file_path(CONFIG_ROOT, STACK), CONFIG_ROOT)


@pytest.fixture(scope="module")
def ladder(baseline_scenario, population, expanded):
    """One run per rung, plus a run of the two provisions' combined law.

    Every run shares the same baseline and the same population block, which is
    what makes the increments clean: the baseline cell cancels.
    """
    jax.clear_caches()
    runs = {}
    for scenario in expanded.scenarios:
        reform = provenance.load_scenario(YEARS, resolved_law=scenario.law)
        runs[scenario.id] = channel_totals(
            run(baseline_scenario, population, saving_share=0.0, reform=reform)
        )

    combined = provenance.load_scenario(
        YEARS,
        resolved_law=combined_law(stack_file_path(CONFIG_ROOT, STACK), CONFIG_ROOT),
    )
    runs["combined"] = channel_totals(
        run(baseline_scenario, population, saving_share=0.0, reform=combined)
    )
    return runs


def channel_totals(simulation):
    """One run's four revenue channels and their whole, by year."""
    _per_cell, channels, whole = assemble.revenue_tables(simulation)
    return {**channels, revenue.TOTAL_DIFFERENCE: whole}


def test_neither_provisions_increment_is_zero(ladder, expanded):
    """Otherwise the identity would hold trivially and prove nothing.

    Both provisions raise tax, so both increments should be positive in both
    years. A provision that moved none of the three households would pass the
    identity test while measuring nothing.
    """
    law_effect = revenue.CHANNEL_DIFFERENCES[0]
    increments = provision_increments(ladder, expanded.increments, COLUMN, YEARS)
    for increment in expanded.increments:
        own = increments[increment.provision][law_effect]
        assert np.all(own > 0.0), f"{increment.provision} raises nothing: {own}"


def test_the_increments_sum_to_the_combined_law_per_year_per_channel(
    ladder, expanded
):
    """The provisions' increments add up to the combined law's effect, to the cent.

    The increments are formed by subtracting one cumulative run from the one
    below it, and this adds them back up. In arithmetic the intermediate rungs
    cancel exactly; in floating point they very nearly do. The tolerance is in
    tests/tolerances.py.
    """
    increments = provision_increments(ladder, expanded.increments, COLUMN, YEARS)
    for channel in REPORTED_CHANNELS:
        summed = sum(
            increments[increment.provision][channel]
            for increment in expanded.increments
        )
        combined = ladder["combined"][channel][COLUMN]
        assert_agrees_to_the_cent(
            summed,
            combined,
            f"the {revenue.channel_name(channel)} channel's increments do not "
            "add up to the combined law's effect",
        )


def test_the_last_rung_is_the_combined_law(ladder, expanded):
    """The top of a stacked ladder already has every provision in force.

    So the combined run is the same law as the last rung, and reporting it
    twice is a check on the fold rather than an extra scenario a real run pays
    for.
    """
    last = expanded.scenarios[-1].id
    for channel in REPORTED_CHANNELS:
        assert np.array_equal(
            ladder[last][channel][COLUMN], ladder["combined"][channel][COLUMN]
        )


def test_the_stacked_tables_total_row_is_the_whole_package(ladder, expanded):
    table = stacked_table(ladder, expanded.increments, YEARS, COLUMN)
    for channel in REPORTED_CHANNELS:
        assert np.array_equal(
            stacked_table_row(table, TOTAL_ROW, channel), ladder["combined"][channel][COLUMN]
        )


def test_the_table_has_one_block_per_channel_naming_both_provisions(ladder, expanded):
    table = stacked_table(ladder, expanded.increments, YEARS, COLUMN)
    assert table.header == ("channel", "provision", *(str(y) for y in YEARS))
    first_block = table.rows[: len(expanded.increments) + 1]
    assert [row[1] for row in first_block] == [
        "tests/stack_provisions/1_ord_rates",
        "tests/stack_provisions/2_standard_deduction",
        TOTAL_ROW,
    ]
