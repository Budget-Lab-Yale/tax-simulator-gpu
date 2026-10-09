"""Many scenarios in one compiled program, on the tiny fixed Population block.

The four things these tests prove about the batching machinery:

* a batch of N scenarios equals the same N run one lane at a time, bitwise, at
  any fixed group size;
* a padded final chunk changes no reported number;
* exactly one program is compiled per switch group, shown by a counter and by
  an independent count of how many times the calculators were traced, not by
  waiting to see whether a run feels slow;
* the program cache spares a repeat call its compile — including a call that
  brings fewer scenarios than the fixed size, which pads up to the same shape
  and so needs no new program.

The group size is a required argument here: the caller works it out before the
run from measured per-scenario memory costs (`run/group_size.py`), and this
module compiles at exactly that size.

Every scenario here uses a placeholder law, one number per year that the toy
calculator multiplies income by, so these tests are about the batching
machinery and not about tax. The laws in a batch differ only in values, not in
shape, as with a rate change inside existing brackets.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.reporting import device_sums
from taxmodel.years import device as device_module
from taxmodel.years import loop, one_year
from tests.unit.years import test_one_year as fixture
from tests.unit.years.one_scenario import taxes_then_prices, unchanged_frame

YEARS = fixture.YEARS


def a_program(
    tax_calculator=fixture.taxes, fingerprint: str = "charitable double run: no"
) -> loop.ScenarioProgram:
    """One shared program. Scenarios that share one program run in one batch."""
    return loop.ScenarioProgram(
        switch_fingerprint=fingerprint,
        frame_builder=fixture.FRAME_BUILDER,
        tax_calculator=tax_calculator,
        price_calculator=fixture.prices,
        behavior_builder=unchanged_frame,
        taxes_and_prices=taxes_then_prices(tax_calculator, fixture.prices),
    )


def a_scenario(
    name: str,
    program: loop.ScenarioProgram,
    reform_rate: float,
    saving_share: float = 0.5,
) -> loop.BatchScenario:
    return loop.BatchScenario(
        name=name,
        program=program,
        reform_laws=jnp.asarray([reform_rate] * len(YEARS)),
        baseline_laws=jnp.ones(len(YEARS)),
        wealth_profiles=fixture.profiles(saving_share),
    )


# The scenarios these tests run: the baseline sitting in the batch as an
# ordinary lane, then rate variants. Five is enough that a group size of two
# leaves one padding lane in the last chunk, which is the case the padding test
# exists for; eight fills a whole eight-lane program for the cache test.
EIGHT_SCENARIOS = (
    ("baseline", 1.0, 0.5),
    ("rate_up", 1.1, 0.5),
    ("rate_down", 0.9, 0.3),
    ("rate_up_more", 1.2, 0.5),
    ("saves_nothing", 1.1, 0.0),
    ("rate_down_more", 0.8, 0.5),
    ("saves_a_little", 1.0, 0.1),
    ("rate_up_most", 1.3, 0.5),
)
FIVE_SCENARIOS = EIGHT_SCENARIOS[:5]
THREE_SCENARIOS = EIGHT_SCENARIOS[:3]


def scenarios_from(
    program: loop.ScenarioProgram, spec=THREE_SCENARIOS
) -> list[loop.BatchScenario]:
    return [a_scenario(name, program, rate, saving) for name, rate, saving in spec]


def three_scenarios(program: loop.ScenarioProgram) -> list[loop.BatchScenario]:
    return scenarios_from(program, THREE_SCENARIOS)


@functools.cache
def run_one_scenario(reform_rate: float, saving_share: float, tax_calculator=None):
    """The same scenario on its own, in a batch of a single lane.

    This is the reference the tests below compare a wider batch against: the
    same arithmetic in a program traced one lane wide. Remembered between
    tests, because a run of the toy model is deterministic, several tests
    compare against the same one, and each run costs a compile.
    """
    program = a_program(
        tax_calculator=fixture.taxes if tax_calculator is None else tax_calculator
    )
    batched = loop.run_scenarios(
        fixture.population(),
        [a_scenario("on its own", program, reform_rate, saving_share)],
        years=YEARS,
        scenarios_per_program=1,
    )
    return batched.runs["on its own"]


# How far two runs of the same arithmetic may sit apart. Both sides run one
# compiled projection year, but at different widths, and the compiler is free
# to reassociate a sum inside each program it compiles. That moves the last
# bit or two of a number formed by subtracting two nearly equal totals — a
# distribution table's tax change is exactly that — so the claim these tests
# make is that widening the batch does not change the arithmetic, not that it
# reproduces the rounding.
A_ROUNDING_APART = 1e-12


def assert_runs_are_identical(actual, expected, context: str) -> None:
    """Every reported number of two runs, compared to within the last bits.

    Result grid, distribution table, wealth-channel history and the carried
    stock of unrealized gain: everything a run reports.
    """
    assert actual.years == expected.years, context
    for year in expected.years:
        for _cell, frame, law in device_sums.EVALUATED_CELLS:
            expected_cell = expected.grid.get(frame, law, year)
            actual_cell = actual.grid.get(frame, law, year)
            assert actual_cell.keys() == expected_cell.keys(), context
            for column, value in expected_cell.items():
                np.testing.assert_allclose(
                    np.asarray(actual_cell[column]),
                    np.asarray(value),
                    rtol=A_ROUNDING_APART,
                    err_msg=f"{context}: {frame.value}/{law.value} {year} {column}",
                )
    if expected.distribution_table is None:
        assert actual.distribution_table is None, context
    else:
        np.testing.assert_allclose(
            actual.distribution_table,
            expected.distribution_table,
            rtol=A_ROUNDING_APART,
            err_msg=f"{context}: distribution table",
        )
    assert (
        actual.wealth_channel_history.keys() == expected.wealth_channel_history.keys()
    )
    for name, history in expected.wealth_channel_history.items():
        np.testing.assert_allclose(
            np.asarray(actual.wealth_channel_history[name]),
            np.asarray(history),
            rtol=A_ROUNDING_APART,
            err_msg=f"{context}: wealth channel {name}",
        )
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual.states_after_year),
        jax.tree.leaves(expected.states_after_year),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=A_ROUNDING_APART,
            err_msg=context,
        )


# ---------------------------------------------------------------------------
# A batch equals the same scenarios run one lane at a time
# ---------------------------------------------------------------------------


def test_a_batch_equals_the_same_scenarios_run_one_lane_at_a_time():
    batched = loop.run_scenarios(
        fixture.population(),
        three_scenarios(a_program()),
        years=YEARS,
        scenarios_per_program=3,
    )

    assert list(batched.runs) == [name for name, _rate, _saving in THREE_SCENARIOS]
    for name, rate, saving in THREE_SCENARIOS:
        assert_runs_are_identical(
            batched.runs[name], run_one_scenario(rate, saving), name
        )


def test_a_batch_of_one_scenario_is_still_that_scenario():
    program = a_program()
    batched = loop.run_scenarios(
        fixture.population(),
        [a_scenario("only", program, 1.1)],
        years=YEARS,
        scenarios_per_program=1,
    )

    assert batched.counters.groups[0].chunk_size == 1
    assert_runs_are_identical(batched.runs["only"], run_one_scenario(1.1, 0.5), "only")


def test_the_same_scenarios_give_the_same_answers_at_any_group_size():
    """Chunks of one and one big chunk are the same arithmetic, bitwise.

    This is the property that lets the group size be a pure memory decision:
    the caller can pick any size the machine holds and the answers do not
    move. Three scenarios at size three is one chunk; at size one it is three
    chunks of one lane each.
    """
    program = a_program()
    one_chunk = loop.run_scenarios(
        fixture.population(),
        three_scenarios(program),
        years=YEARS,
        scenarios_per_program=3,
    )
    three_chunks = loop.run_scenarios(
        fixture.population(),
        three_scenarios(program),
        years=YEARS,
        scenarios_per_program=1,
    )

    assert one_chunk.counters.groups[0].chunks == 1
    assert three_chunks.counters.groups[0].chunks == 3
    assert three_chunks.counters.groups[0].padding_lanes == 0
    for name, _rate, _saving in THREE_SCENARIOS:
        assert_runs_are_identical(
            three_chunks.runs[name], one_chunk.runs[name], f"{name} in small chunks"
        )


# ---------------------------------------------------------------------------
# Padding the last chunk
# ---------------------------------------------------------------------------


def test_a_padded_final_chunk_changes_no_reported_number():
    """Five scenarios in chunks of two: two full chunks and one padded chunk."""
    program = a_program()
    chunked = loop.run_scenarios(
        fixture.population(),
        scenarios_from(program, FIVE_SCENARIOS),
        years=YEARS,
        scenarios_per_program=2,
    )

    group = chunked.counters.groups[0]
    assert group.chunk_size == 2
    assert group.chunks == 3
    assert group.padding_lanes == 1
    assert group.chunks * group.chunk_size == group.scenarios + group.padding_lanes
    for name, rate, saving in FIVE_SCENARIOS:
        assert_runs_are_identical(
            chunked.runs[name], run_one_scenario(rate, saving), f"{name} in chunks"
        )


def test_chunk_positions_fills_out_the_last_chunk_by_repeating():
    assert loop.chunk_positions(6, 3) == [[0, 1, 2], [3, 4, 5]]
    assert loop.chunk_positions(7, 3) == [[0, 1, 2], [3, 4, 5], [6, 6, 6]]
    assert loop.chunk_positions(3, 1) == [[0], [1], [2]]
    assert loop.chunk_positions(1, 4) == [[0, 0, 0, 0]]


def test_a_chunk_of_no_scenarios_is_refused():
    with pytest.raises(ValueError, match="at least one scenario"):
        loop.chunk_positions(4, 0)


def test_a_group_size_below_one_is_refused():
    with pytest.raises(ValueError, match="at least one scenario lane"):
        loop.run_scenarios(
            fixture.population(),
            three_scenarios(a_program()),
            years=YEARS,
            scenarios_per_program=0,
        )


# ---------------------------------------------------------------------------
# One compile per switch group
# ---------------------------------------------------------------------------


def counting_calculator():
    """The toy tax calculator, plus a count of how often it was traced.

    The calculator is Python, so it runs while a program is being traced and
    never while a compiled program is executing. Its call count is therefore an
    independent witness to how many programs were traced, arrived at without
    trusting the runner's own counters.
    """
    calls: list[int] = []

    def taxes(frame, law):
        calls.append(1)
        return fixture.taxes(frame, law)

    return taxes, calls


# A group compiles five programs: the projection year, the population totals,
# and the three report programs (the tax at death, the distribution table and
# the marginal-rate sums).
PROGRAMS_PER_GROUP = 5


def test_exactly_one_program_is_compiled_for_a_group_however_many_chunks():
    taxes, calls = counting_calculator()
    program = a_program(tax_calculator=taxes)
    scenarios = three_scenarios(program)

    one_chunk = loop.run_scenarios(
        fixture.population(), scenarios, years=YEARS, scenarios_per_program=3
    )
    traces_for_one_chunk = len(calls)

    calls.clear()
    three_chunks = loop.run_scenarios(
        fixture.population(), scenarios, years=YEARS, scenarios_per_program=1
    )

    assert one_chunk.counters.groups[0].programs_compiled == PROGRAMS_PER_GROUP
    assert one_chunk.counters.programs_compiled == PROGRAMS_PER_GROUP
    # Three one-lane chunks, one compile: every chunk presents the same shape,
    # so all three share the programs compiled for the first one.
    assert three_chunks.counters.groups[0].chunks == 3
    assert (
        three_chunks.counters.groups[0].programs_compiled == PROGRAMS_PER_GROUP
    )
    # The independent witness. The calculator is Python, so it runs while a
    # program is being traced and never while a compiled program executes: the
    # three-chunk run traced it exactly as often as the one-chunk run did,
    # because each run compiled its two programs exactly once. Had every chunk
    # compiled its own, the count would have been three times over.
    assert traces_for_one_chunk > 0
    assert len(calls) == traces_for_one_chunk


def test_two_switch_groups_run_as_two_batches():
    """A scenario that answers a structural question differently runs apart."""

    def taxes_with_the_charitable_choice(frame, law):
        # Stands in for the second code path the charitable double run takes:
        # a different Python function, so a different traced program.
        return fixture.taxes(frame, law)

    without = a_program(fingerprint="charitable double run: no")
    with_the_choice = a_program(
        tax_calculator=taxes_with_the_charitable_choice,
        fingerprint="charitable double run: yes",
    )
    scenarios = [
        a_scenario("plain_baseline", without, 1.0),
        a_scenario("charitable_reform", with_the_choice, 1.1),
        a_scenario("plain_reform", without, 1.2),
    ]

    batched = loop.run_scenarios(
        fixture.population(), scenarios, years=YEARS, scenarios_per_program=2
    )

    assert batched.counters.switch_groups == 2
    assert batched.counters.programs_compiled == 2 * PROGRAMS_PER_GROUP
    by_fingerprint = {
        group.switch_fingerprint: group for group in batched.counters.groups
    }
    assert by_fingerprint["charitable double run: no"].scenarios == 2
    # The one-scenario group still runs the fixed two-lane shape, padded.
    assert by_fingerprint["charitable double run: yes"].scenarios == 1
    assert by_fingerprint["charitable double run: yes"].padding_lanes == 1
    for group in batched.counters.groups:
        assert group.programs_compiled == PROGRAMS_PER_GROUP
    assert list(batched.runs) == ["plain_baseline", "charitable_reform", "plain_reform"]
    assert_runs_are_identical(
        batched.runs["charitable_reform"],
        run_one_scenario(1.1, 0.5, taxes_with_the_charitable_choice),
        "charitable_reform",
    )


def test_scenarios_are_grouped_by_the_program_they_run():
    first = a_program()
    second = a_program(fingerprint="charitable double run: yes")
    scenarios = [
        a_scenario("a", first, 1.0),
        a_scenario("b", second, 1.1),
        a_scenario("c", first, 1.2),
    ]

    groups = loop.group_by_switches(scenarios)

    assert list(groups.values()) == [[0, 2], [1]]


# ---------------------------------------------------------------------------
# The seam with the shared-layout compiler, and the ordinary refusals
# ---------------------------------------------------------------------------


def test_laws_that_do_not_share_a_shape_are_refused_by_name():
    """Two laws that compile to different shapes cannot share one program.

    The refusal names both scenarios and points at the batch law compiler,
    which is what puts a set of laws on one shared layout.
    """
    program = a_program()
    narrow = a_scenario("narrow", program, 1.0)
    wide = loop.BatchScenario(
        name="wide",
        program=program,
        # The same law stated one value per filing status instead of one value
        # shared by all four, which is the shape mismatch the batch refuses.
        reform_laws=jnp.ones((len(YEARS), 4)),
        baseline_laws=jnp.ones(len(YEARS)),
        wealth_profiles=fixture.profiles(0.5),
    )

    with pytest.raises(ValueError, match="compile_batch"):
        loop.run_scenarios(
            fixture.population(),
            [narrow, wide],
            years=YEARS,
            scenarios_per_program=2,
        )


def test_two_scenarios_cannot_share_a_name():
    program = a_program()
    with pytest.raises(ValueError, match="share a name"):
        loop.run_scenarios(
            fixture.population(),
            [a_scenario("same", program, 1.0), a_scenario("same", program, 1.1)],
            years=YEARS,
            scenarios_per_program=2,
        )


def test_an_empty_batch_is_refused():
    with pytest.raises(ValueError, match="at least one scenario"):
        loop.run_scenarios(
            fixture.population(), [], years=YEARS, scenarios_per_program=1
        )


def test_a_scenario_whose_profiles_cover_the_wrong_years_is_named():
    program = a_program()
    wrong_years = loop.BatchScenario(
        name="wrong_years",
        program=program,
        reform_laws=jnp.ones(len(YEARS)),
        baseline_laws=jnp.ones(len(YEARS)),
        wealth_profiles=one_year.stack_wealth_profiles(
            jnp.full((fixture.N_AGES, fixture.N_BINS), 0.5),
            jnp.eye(fixture.N_BINS),
            jnp.zeros(len(YEARS) + 1),
            fmax=0.9,
            pass_through_weight=0.2,
        ),
    )

    with pytest.raises(ValueError, match="wrong_years"):
        loop.run_scenarios(
            fixture.population(),
            [wrong_years],
            years=YEARS,
            scenarios_per_program=1,
        )


# ---------------------------------------------------------------------------
# Reading the machine's memory
# ---------------------------------------------------------------------------


def test_the_budget_falls_back_to_the_stated_host_budget_without_a_card():
    """JAX's CPU backend reports no memory, which is the laptop path."""
    already_committed = 3 * 1024**3
    budget = device_module.read_memory_budget(jax.devices()[0], already_committed)

    if jax.devices()[0].memory_stats() is None:
        assert budget.total_bytes == device_module.HOST_MEMORY_BUDGET_BYTES
        assert budget.committed_bytes == already_committed
        assert budget.free_bytes == device_module.HOST_MEMORY_BUDGET_BYTES - already_committed
        assert "reports no memory statistics" in budget.source
    else:
        assert budget.total_bytes > 0
        assert "memory statistics" in budget.source


def test_free_bytes_is_what_remains_after_the_committed_arrays():
    budget = device_module.MemoryBudget(
        device="a made-up card",
        total_bytes=1_000,
        committed_bytes=400,
        source="this test",
    )

    assert budget.free_bytes == 600


def test_the_counters_say_what_stayed_resident_on_the_device():
    """What the run holds for its whole length, which is what the budget is about.

    On a card that is the population block plus every scenario's own arrays.
    On a machine that reads the population one year at a time and drops it —
    the laptop, and the CPU these tests run on — the block is not resident at
    all, and only the scenarios' arrays are counted.
    """
    population = fixture.population()
    batched = loop.run_scenarios(
        population,
        three_scenarios(a_program()),
        years=YEARS,
        scenarios_per_program=3,
    )

    block_bytes = device_module.bytes_on_device(
        one_year.population_arrays_for_years(population, YEARS)
    )
    device = jax.devices()[0]
    if device_module.has_its_own_device_memory(device):
        assert batched.counters.resident_bytes > block_bytes
    else:
        assert batched.counters.resident_bytes > 0
    assert batched.counters.device
    assert batched.counters.memory_source


# ---------------------------------------------------------------------------
# The program cache
# ---------------------------------------------------------------------------


def test_a_cached_program_is_reused_without_a_compile():
    """A gains scenario's second execution: same program, zero new compiles."""
    program = a_program()
    cache: dict = {}
    first = loop.run_scenarios(
        fixture.population(),
        scenarios_from(program, THREE_SCENARIOS),
        years=YEARS,
        scenarios_per_program=3,
        program_cache=cache,
    )
    second = loop.run_scenarios(
        fixture.population(),
        scenarios_from(program, THREE_SCENARIOS),
        years=YEARS,
        scenarios_per_program=3,
        program_cache=cache,
    )

    assert first.counters.groups[0].programs_compiled == PROGRAMS_PER_GROUP
    assert second.counters.groups[0].programs_compiled == 0
    assert second.counters.groups[0].chunk_size == 3


def test_a_smaller_call_at_the_same_fixed_size_compiles_nothing():
    """Five scenarios after eight, both at size eight: the cache key is the
    fixed size, not the call's scenario count, so the second call pads its
    five scenarios up to the eight-lane shape and reuses the program as it
    stands. This is the everyday case the fixed size exists for: a run's last,
    smaller group of scenarios must not pay a second compile.
    """
    program = a_program()
    cache: dict = {}
    eight = loop.run_scenarios(
        fixture.population(),
        scenarios_from(program, EIGHT_SCENARIOS),
        years=YEARS,
        scenarios_per_program=8,
        program_cache=cache,
    )
    five = loop.run_scenarios(
        fixture.population(),
        scenarios_from(program, FIVE_SCENARIOS),
        years=YEARS,
        scenarios_per_program=8,
        program_cache=cache,
    )

    assert eight.counters.groups[0].programs_compiled == PROGRAMS_PER_GROUP
    assert five.counters.programs_compiled == 0
    assert five.counters.groups[0].chunk_size == 8
    assert five.counters.groups[0].chunks == 1
    assert five.counters.groups[0].padding_lanes == 3
    # And the padded, cache-hit answers are the scenarios' own.
    for name, rate, saving in FIVE_SCENARIOS:
        assert_runs_are_identical(
            five.runs[name], run_one_scenario(rate, saving), f"{name} on a cache hit"
        )
