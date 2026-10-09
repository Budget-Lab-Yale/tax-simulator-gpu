"""Tests for the group-size arithmetic in taxmodel.run.group_size.

These tests state their expectations in terms of the module's own constants,
so they hold for any measured values in which a gains lane needs more memory
than a plain one. One test checks that directly.
"""

from types import SimpleNamespace

import pytest

from taxmodel.run import group_size, grouping
from taxmodel.years.device import HOST_MEMORY_BUDGET_BYTES

GIB = 1024**3

# Shorthand for calling the function at the household count the constants were
# measured at, where no scaling happens. Eleven years is the test runscript's
# window; one test below checks that the lane cost does not depend on it.
MEASURED = {"n_years": 11, "n_households": group_size.MEASURED_HOUSEHOLD_COUNT}


def population_bytes(n_years: int) -> int:
    """What the resident population block adds, beyond the year in the overhead."""
    return (n_years - 1) * group_size.POPULATION_BYTES_PER_YEAR


def device_bytes_for(
    n_scenarios: float,
    per_lane_bytes: int,
    n_years: int = 11,
) -> int:
    """A card just big enough for the overhead, the population, and n lanes."""
    return int(
        group_size.SHARED_DEVICE_BYTES
        + population_bytes(n_years)
        + n_scenarios * per_lane_bytes
    )


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def test_rounds_down_to_the_largest_menu_entry_that_fits():
    # Room for 11 plain lanes: 11 is not on the menu, so the answer is the
    # next menu entry down, 8.
    device = device_bytes_for(11, group_size.PLAIN_LANE_DEVICE_BYTES)
    assert (
        group_size.scenarios_per_program(
            **MEASURED, runs_the_gains_model=False, device_total_bytes=device
        )
        == 8
    )


def test_an_exact_menu_fit_is_kept():
    device = device_bytes_for(8, group_size.PLAIN_LANE_DEVICE_BYTES)
    assert (
        group_size.scenarios_per_program(
            **MEASURED, runs_the_gains_model=False, device_total_bytes=device
        )
        == 8
    )


def test_room_beyond_the_menu_caps_at_the_largest_entry():
    device = device_bytes_for(1000, group_size.PLAIN_LANE_DEVICE_BYTES)
    assert (
        group_size.scenarios_per_program(
            **MEASURED, runs_the_gains_model=False, device_total_bytes=device
        )
        == max(group_size.GROUP_SIZE_MENU)
    )


def test_a_machine_too_small_for_the_arithmetic_still_runs_groups_of_one():
    # A device smaller than the shared overhead makes the subtraction
    # negative; the answer is still 1, never 0 or negative. This is the
    # laptop, and a group of one is exactly what it runs.
    device = group_size.SHARED_DEVICE_BYTES // 2
    assert (
        group_size.scenarios_per_program(
            **MEASURED, runs_the_gains_model=False, device_total_bytes=device
        )
        == 1
    )
    assert (
        group_size.scenarios_per_program(
            **MEASURED, runs_the_gains_model=True, device_total_bytes=device
        )
        == 1
    )


def test_gains_scenarios_get_smaller_groups_than_plain_ones():
    assert group_size.GAINS_LANE_DEVICE_BYTES > group_size.PLAIN_LANE_DEVICE_BYTES
    device = device_bytes_for(8, group_size.PLAIN_LANE_DEVICE_BYTES)
    plain = group_size.scenarios_per_program(
        **MEASURED, runs_the_gains_model=False, device_total_bytes=device
    )
    gains = group_size.scenarios_per_program(
        **MEASURED, runs_the_gains_model=True, device_total_bytes=device
    )
    assert plain == 8
    assert gains < plain


def test_a_lane_costs_the_same_however_long_the_window_is():
    """The point of compiling one year: a lane holds one year, not thirty.

    A longer window costs only the extra years of the population block, so on
    a device with plenty of room the group size does not move at all.
    """
    device = 200 * GIB
    eleven_years = group_size.scenarios_per_program(
        n_years=11,
        n_households=group_size.MEASURED_HOUSEHOLD_COUNT,
        runs_the_gains_model=True,
        device_total_bytes=device,
    )
    thirty_one_years = group_size.scenarios_per_program(
        n_years=31,
        n_households=group_size.MEASURED_HOUSEHOLD_COUNT,
        runs_the_gains_model=True,
        device_total_bytes=device,
    )
    assert eleven_years == thirty_one_years


def test_fewer_households_leave_room_for_more_lanes():
    device = device_bytes_for(5, group_size.PLAIN_LANE_DEVICE_BYTES)
    at_full_population = group_size.scenarios_per_program(
        **MEASURED, runs_the_gains_model=False, device_total_bytes=device
    )
    at_half_population = group_size.scenarios_per_program(
        n_years=11,
        n_households=group_size.MEASURED_HOUSEHOLD_COUNT // 2,
        runs_the_gains_model=False,
        device_total_bytes=device,
    )
    assert at_full_population == 4
    assert at_half_population > at_full_population


def test_a_machine_with_no_device_memory_runs_one_scenario_at_a_time():
    """The laptop rule, and it does not depend on how big the budget looks.

    On a CPU the width is limited by what the XLA compiler holds while it
    builds the program, not by what the run holds while it executes, which is
    what these constants measure. A four-year, three-scenario run sized at eight
    lanes by the arithmetic below reached 67 GB of host memory compiling it and
    was killed (Slurm job 21946963), so the answer there is one, whatever
    number the machine reports.
    """
    for budget in (12 * GIB, 64 * GIB, 512 * GIB):
        for gains in (False, True):
            assert (
                group_size.scenarios_per_program(
                    n_years=31,
                    n_households=group_size.MEASURED_HOUSEHOLD_COUNT,
                    runs_the_gains_model=gains,
                    device_total_bytes=budget,
                    has_its_own_device_memory=False,
                )
                == 1
            )


# ---------------------------------------------------------------------------
# Reading the device's total memory
# ---------------------------------------------------------------------------


class FakeGpu:
    """A device that reports its memory, the way a real GPU does."""

    def memory_stats(self):
        return {"bytes_limit": 80 * GIB, "bytes_in_use": 5 * GIB}


class FakeCpu:
    """A device that reports nothing, the way JAX's CPU backend does."""

    def memory_stats(self):
        return None


def test_a_device_that_reports_memory_is_believed():
    assert group_size.read_device_total_bytes(FakeGpu()) == 80 * GIB


def test_a_device_that_reports_nothing_gets_the_laptop_budget():
    assert group_size.read_device_total_bytes(FakeCpu()) == HOST_MEMORY_BUDGET_BYTES


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, 1), (3, 3), (8, 8), (9, 5), (10, 5), (15, 8), (16, 8), (17, 6)],
)
def test_family_width_avoids_padding_without_adding_groups(count, expected):
    family = grouping.Family(False, None, True, tuple(range(count)))
    population = SimpleNamespace(
        universe=SimpleNamespace(n_slots=group_size.MEASURED_HOUSEHOLD_COUNT)
    )
    machine = grouping.RunMachine(
        device_total_bytes=device_bytes_for(8, group_size.GAINS_LANE_DEVICE_BYTES),
        has_its_own_device_memory=True,
        program_cache={},
    )
    assert (
        grouping.group_size_for(family, tuple(range(2027, 2038)), population, machine)
        == expected
    )


def test_balancing_keeps_cpu_families_at_one_lane():
    family = grouping.Family(False, None, True, tuple(range(10)))
    population = SimpleNamespace(universe=SimpleNamespace(n_slots=1))
    machine = grouping.RunMachine(512 * GIB, False, {})
    assert grouping.group_size_for(family, (2027,), population, machine) == 1


@pytest.mark.parametrize(("count", "expected"), [(10, 10), (11, 11), (12, 6)])
def test_whole_family_can_use_memory_between_menu_entries(count, expected):
    family = grouping.Family(False, None, True, tuple(range(count)))
    population = SimpleNamespace(
        universe=SimpleNamespace(n_slots=group_size.MEASURED_HOUSEHOLD_COUNT)
    )
    machine = grouping.RunMachine(
        device_bytes_for(11, group_size.GAINS_LANE_DEVICE_BYTES), True, {}
    )
    assert (
        grouping.group_size_for(family, tuple(range(2027, 2038)), population, machine)
        == expected
    )


def test_exact_family_fit_still_respects_maximum_program_width():
    assert (
        group_size.scenarios_per_program(
            **MEASURED,
            runs_the_gains_model=True,
            device_total_bytes=512 * GIB,
            n_scenarios=40,
        )
        == 32
    )
