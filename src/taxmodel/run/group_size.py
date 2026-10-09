"""The number of scenarios one compiled projection year runs at once.

The run loop compiles one projection year and calls it once per year, with a
fixed number of scenarios running side by side as lanes (`years/loop.py`).
This module picks that number from the measured memory constants below. If the
constants drift, the fix is to re-run the measurement job and paste in new
numbers.

One compiled program holds a single year, so the memory cost of a lane does not
depend on the length of the window. The population block does: it sits on the
device for every year of the run on a machine with device memory of its own.
"""

from __future__ import annotations

from taxmodel.years.device import HOST_MEMORY_BUDGET_BYTES

# ---------------------------------------------------------------------------
# Measured memory constants
#
# Measured on an H200 over the full 221,831-household bridge population, from
# production runs of the ordinary command at several lane counts, with and
# without the capital-gains model, over thirty-year and eleven-year windows.
# Production runs are used because the reporting reductions and the run
# directory's writing are part of the peak. Leaving them out understates a
# capital-gains lane by a factor of about 1.8.
#
# The reading is `counters.peak_device_bytes_in_use` in each run's manifest.
# Regressing peaks on lane counts gives a line whose slope is the cost of one
# lane and whose intercept is the memory held at any lane count. The gains and
# plain intercepts agree to within 4%, and the larger is used.
#
# The constants are fitted to the thirty-year runs. The reported tables
# accumulate a year at a time, so an eleven-year run's requirement is
# over-stated by about 6%, which is the safe direction for a memory budget.
# ---------------------------------------------------------------------------

# Device memory one more scenario lane adds to a compiled year that runs no
# behavior modules.
PLAIN_LANE_DEVICE_BYTES = 2_526_631_232  # measured: 2.35 GiB

# The same for a lane running the capital-gains model, which holds extra
# per-household state, mainly the conversion channel's per-household leg rates
# kept from the first execution pass until the behavioral solve reads them, and
# runs the year step twice. A gains lane costs 3.3 times a plain one.
GAINS_LANE_DEVICE_BYTES = 8_328_733_650  # measured: 7.76 GiB

# Device memory held at any lane count: the first year of households, the
# compiled year itself, and the run's fixed structures. The other years of the
# population block are charged below. Of the 0.99 GiB, 0.28 GiB is the first
# year and 0.71 GiB is everything else.
SHARED_DEVICE_BYTES = 1_060_295_437  # measured: 0.99 GiB

# Device memory for one year of the population block: the difference in
# resident bytes between the thirty-year and eleven-year runs, divided by the
# nineteen years between them. The gains and plain runs agree to within 0.1%.
POPULATION_BYTES_PER_YEAR = 305_740_239  # measured: 0.28 GiB

# The population the constants above were measured over. Scaling them in
# proportion to a different household count is an assumption.
MEASURED_HOUSEHOLD_COUNT = 221_831

# Group sizes used when a family has to be split across several programs. A
# family that fits in one program uses its own size, up to the menu's maximum.
GROUP_SIZE_MENU = (1, 2, 4, 8, 16, 32)


def scenarios_per_program(
    n_years: int,
    n_households: int,
    runs_the_gains_model: bool,
    device_total_bytes: int,
    has_its_own_device_memory: bool = True,
    n_scenarios: int | None = None,
) -> int:
    """The number of scenarios that fit in one compiled year on the device.

    The shared overhead and the per-lane cost are scaled from the measured
    household count to this run's. The overhead is subtracted from the device's
    memory and the remainder divided by the per-lane cost. The answer is rounded
    down to the largest entry in GROUP_SIZE_MENU that fits and never falls
    below 1.

    When ``n_scenarios`` is given and the whole family fits within both the
    memory estimate and the largest menu size, the family's own size is
    returned. Otherwise the result is the largest menu size that fits.

    ``has_its_own_device_memory`` comes from `years/loop.py`: whether the
    machine has a card or one memory pool shared with the host. On a card the
    population block stays resident and is charged for here. Without one the
    answer is a single scenario.
    """
    if not has_its_own_device_memory:
        # On a CPU the binding constraint is the memory the XLA compiler uses
        # while building the program, which these constants do not measure. An
        # eight-lane program has been seen to need 67 GB of host memory to
        # compile. A one-lane year compiles inside 16 GB.
        return 1

    scale = n_households / MEASURED_HOUSEHOLD_COUNT
    per_lane = (
        GAINS_LANE_DEVICE_BYTES
        if runs_the_gains_model
        else PLAIN_LANE_DEVICE_BYTES
    ) * scale
    # The shared overhead covers the year the compiled program is working on.
    # The population block's other years are added here.
    overhead = SHARED_DEVICE_BYTES * scale + (
        (n_years - 1) * POPULATION_BYTES_PER_YEAR * scale
    )

    scenarios_that_fit = (device_total_bytes - overhead) / per_lane

    if n_scenarios is not None and 1 <= n_scenarios <= min(
        scenarios_that_fit, max(GROUP_SIZE_MENU)
    ):
        return n_scenarios

    chosen = 1
    for menu_size in GROUP_SIZE_MENU:
        if menu_size <= scenarios_that_fit:
            chosen = max(chosen, menu_size)
    return chosen


def read_device_total_bytes(device) -> int:
    """The device's total memory, or the host budget when it reports none.

    A GPU reports its memory limit through JAX's memory statistics. Under JAX's
    default up-front reservation that limit is three-quarters of the card, the
    same number the constants above were measured against. JAX's CPU backend
    reports nothing, so on a laptop the answer is the host memory budget.
    """
    stats = device.memory_stats()
    if stats is not None and "bytes_limit" in stats:
        return int(stats["bytes_limit"])
    return HOST_MEMORY_BUDGET_BYTES
