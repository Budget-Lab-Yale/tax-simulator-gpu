"""The year loop: many scenarios through one compiled projection year.

The unit of compilation is one calendar year: the baseline sweep and the four
reform worlds for a single year, with every reporting reduction. The compiled
year is called once per year by a plain Python loop.

* The law is data and every scenario reads the same population block, so
  ``jax.vmap`` runs one year of N scenarios side by side through the same
  program. Compile time does not grow with the number of lanes, and sixteen
  lanes execute in under twice one lane's time.
* One year's working set (frames, tax results, and the gradient scratch the
  marginal rates need) grows with the number of lanes. The caller chooses the
  number of lanes from measured per-scenario memory costs
  (`run/group_size.py`) and passes it in as ``scenarios_per_program``. This
  module compiles a year of that many lanes and pads the last group out to it.
* Two scenarios share a compiled year only when the same functions run in
  both. The structural-switch fingerprint that `law.batch.compile_batch`
  supplies with each compiled law records that, and scenarios are grouped by
  it. The per-household detail request is part of the same key, because the
  requested columns are extra outputs of the traced program.

The baseline is an ordinary scenario here, one more lane in the batch.

**Array placement.** On a machine that reports its memory (a GPU) the whole
population block is copied to the device once for a run. On one that does not
(a laptop) the year loop asks the population for one year at a time, and that
year is read from the bridge folder when it is asked for
(`frame/population.PopulationFromDisk`). Each scenario's law, wealth
profile and behavioral parameters go to the device once and stay; a
thirty-year law is a few megabytes. Three inputs carry a per-household column
for every year of the window and are kept on the host and sent one year at a
time: the heir structure, the capital-gains cell weights, and each scenario's
capital-gains tables.

**Padding.** Every chunk presents the compiled year exactly
``scenarios_per_program`` lanes. Fifty-three scenarios at ten per program is
five full chunks and one chunk of three; the last chunk is filled out by
repeating the last real scenario, and the repeated lanes are dropped when the
chunk's answers are taken apart. A padded lane is never read, so it cannot
reach a reported number. No lane can see another, so the padding lanes'
values do not matter. The same padding covers a call smaller than the
program: five scenarios through an eight-lane program is one chunk with three
padding lanes, and the compiled program's shape is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from taxmodel.frame.population import PopulationSource
from taxmodel.reporting import assemble, heir_inputs
from taxmodel.reporting import detail as detail_output
from taxmodel.run import host_memory
from taxmodel.years import one_year, wealth_state, worlds
from taxmodel.years.device import (
    GroupPrograms,
    PassThroughTheYears,
    peak_device_bytes_in_use,
    put_the_run_on_the_device,
)
from taxmodel.years.stacking import stack_lanes

Array = jnp.ndarray

# ---------------------------------------------------------------------------
# Scenario inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioProgram:
    """The parts of a scenario that decide which code runs.

    Two scenarios share one compiled program only when everything here is
    identical, so this is the key the batch runner groups by.

    ``switch_fingerprint`` is the label ``law.batch.compile_batch`` supplies
    with each compiled law. This module compares two of them for equality and
    does not read them otherwise.

    The calculators are compared by identity: two different Python functions
    trace to two different programs even when they compute the same thing. The
    runscript driver builds the calculators once and shares them across
    scenarios.

    ``detail_request`` is part of the key because the requested per-household
    columns are extra outputs of the traced program. Requests are compared by
    value, so two scenarios asking for the same columns from the same worlds
    for the same years share a program.

    ``runs_the_gains_model`` composes the gains applier and the rate exports
    into the program, and ``gains_rate_pricers`` is the code that measures the
    exported rates, compared by identity like the calculators. The cell tables
    are per-scenario data on :class:`BatchScenario`. Whether the law deems a
    decedent's gains realized at death is a number, not part of the key: the
    deemed gain is priced by extra calculator calls on the same households.
    """

    switch_fingerprint: str
    frame_builder: worlds.FrameBuilder
    tax_calculator: worlds.TaxCalculator
    price_calculator: worlds.PriceCalculator
    behavior_builder: worlds.BehaviorBuilder
    # The mechanical frame's liabilities and marginal rates under the reform's
    # law from one run of the calculators.
    taxes_and_prices: worlds.TaxesAndPrices
    detail_request: detail_output.DetailRequest | None = None
    # The gains model's pass with no behavioral response, read by the
    # first-order and mechanical worlds: the treatment of gains at death
    # changes while realization behavior stays at baseline. None for a
    # scenario without the gains model.
    frozen_gains_builder: worlds.BehaviorBuilder | None = None
    runs_the_gains_model: bool = False
    gains_rate_pricers: object = None


@dataclass(frozen=True)
class BatchScenario:
    """One scenario: its name, the code it runs, and its own arrays.

    ``reform_laws`` and ``baseline_laws`` are bound annual laws stacked on
    their first axis, one entry per projection year. ``wealth_profiles`` is
    the scenario's saving assumptions, also one entry per year.

    ``initial_state`` is optional; left out, the scenario starts with zero
    wealth shortfalls.

    ``gains_tables`` is a gains scenario's runtime tables
    (`behavior/gains/builders.GainsTables`, stacked on the year axis), mapped
    over the lanes like the laws. None for a scenario without the gains model.

    ``behavior_parameters`` is this scenario's behavioral parameters
    (``behavior/parameters.py``), stacked lane by lane like the law. Every
    margin runs in every program, and the parameter values turn a response on
    or off. Left out, every margin runs at its neutral values.
    """

    name: str
    program: ScenarioProgram
    reform_laws: object
    baseline_laws: object
    wealth_profiles: wealth_state.WealthProfileYear
    initial_state: worlds.SimulationState | None = None
    gains_tables: object = None
    behavior_parameters: object = None
    # The national corporate figures the distribution table spreads
    # (`reporting/corporate_spread.CorporateDollars`, [year] in each field).
    # Left out, every figure is zero.
    corporate_dollars: object = None


# ---------------------------------------------------------------------------
# Compile and memory counters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwitchGroupCounters:
    """One group's compile and memory counters.

    ``programs_compiled`` is the number of programs this call traced and
    compiled: the projection year, the population totals, and the four
    reported table families. It is 6 for a full pass that compiled all of
    them, 0 when the program cache already held them, and 1 for a pass that
    exported the capital-gains rates only, which compiles the year alone.

    ``chunk_size`` is the number of scenario lanes in the compiled program,
    the ``scenarios_per_program`` the caller passed.

    ``padding_lanes`` is the number of lanes across all chunks that repeated a
    real scenario to fill out a chunk's shape.

    ``demand_bytes`` is the memory the compiled program reported needing for
    its temporaries and outputs, read before it ran. ``free_bytes`` is the
    memory the device reported free at that moment. Both are recorded only.

    ``compile_seconds`` is the wall-clock time spent tracing and compiling
    for this group, zero on a program-cache hit.

    ``execute_seconds`` is the wall-clock time the compiled programs spent
    running this group's chunks, from passing a chunk its arrays to the device
    finishing the last answer. Reading answers off the device and assembling
    result grids on the host is not included.
    """

    switch_fingerprint: str
    scenarios: int
    chunk_size: int
    chunks: int
    padding_lanes: int
    programs_compiled: int
    demand_bytes: int
    free_bytes: int
    compile_seconds: float = 0.0
    execute_seconds: float = 0.0


@dataclass(frozen=True)
class BatchCounters:
    """The whole batch's compile and memory counters.

    ``device`` and ``memory_source`` say which device ran the work and where
    the free-memory number came from.

    ``resident_bytes`` is the memory the batch copied to the device once and
    kept there: the population block plus every scenario's law and profile
    arrays.

    ``peak_device_bytes_in_use`` is the device allocator's peak during this
    batch: the resident arrays plus the running chunk's working set. It is
    ``None`` on a device that reports no memory statistics (JAX's CPU
    backend).
    """

    device: str
    memory_source: str
    resident_bytes: int
    groups: tuple[SwitchGroupCounters, ...]
    peak_device_bytes_in_use: int | None = None

    @property
    def programs_compiled(self) -> int:
        """Every program compiled for this batch, across all switch groups."""
        return sum(group.programs_compiled for group in self.groups)

    @property
    def compile_seconds(self) -> float:
        """Wall-clock seconds this batch spent tracing and compiling."""
        return sum(group.compile_seconds for group in self.groups)

    @property
    def execute_seconds(self) -> float:
        """Wall-clock seconds the compiled programs spent running chunks."""
        return sum(group.execute_seconds for group in self.groups)

    @property
    def switch_groups(self) -> int:
        """The number of groups the scenarios were split into."""
        return len(self.groups)


@dataclass(frozen=True)
class BatchRun:
    """Every scenario's results, keyed by name, plus the batch counters.

    ``runs[name]`` is the :class:`~taxmodel.reporting.assemble.SimulationRun`
    for that scenario, in the order the caller listed the scenarios.
    """

    runs: dict[str, assemble.SimulationRun]
    counters: BatchCounters


# ---------------------------------------------------------------------------
# Grouping, stacking, padding
# ---------------------------------------------------------------------------


def group_by_switches(
    scenarios: Sequence[BatchScenario],
) -> dict[ScenarioProgram, list[int]]:
    """Split the scenarios into the groups that can share a compiled program.

    The result maps each distinct program to the positions of the scenarios
    that run it, in the caller's order.
    """
    groups: dict[ScenarioProgram, list[int]] = {}
    for position, scenario in enumerate(scenarios):
        groups.setdefault(scenario.program, []).append(position)
    return groups


def chunk_positions(n_scenarios: int, chunk_size: int) -> list[list[int]]:
    """The scenario positions each chunk runs, with the last chunk padded.

    Every chunk has exactly ``chunk_size`` entries. The last chunk repeats its
    final real scenario to fill out the shape; the caller drops those lanes'
    answers.
    """
    if chunk_size < 1:
        raise ValueError("a chunk holds at least one scenario")
    chunks: list[list[int]] = []
    for start in range(0, n_scenarios, chunk_size):
        real = list(range(start, min(start + chunk_size, n_scenarios)))
        chunks.append(real + [real[-1]] * (chunk_size - len(real)))
    return chunks


def run_scenarios(
    population: PopulationSource,
    scenarios: Sequence[BatchScenario],
    *,
    scenarios_per_program: int,
    years: Sequence[int] = one_year.DEFAULT_YEARS,
    heir_structure: heir_inputs.HeirStructure | None = None,
    gains_cell_weights: object = None,
    program_cache: dict | None = None,
    rates_only: bool = False,
) -> BatchRun:
    """Run every scenario and return each one's results, plus the counters.

    This is the entry point for the run's compute. It takes a population block
    and a list of scenarios and returns per-scenario result grids, with the
    compile and memory counters the run manifest records.

    In order: the loop's inputs are put on the device (`years/device.py`); the
    scenarios are grouped by the code they run; each group's projection year
    is compiled at ``scenarios_per_program`` lanes; and that year is called
    once per projection year by the loop below, followed by the programs that
    form the year's totals and reported tables from its columns. A group
    larger than the fixed size is split into chunks, and the last chunk is
    padded by repeating its final real scenario.

    ``scenarios_per_program`` is the number of scenarios that run side by side
    in one compiled year. The caller computes it before the run from measured
    per-scenario memory costs and the device's total memory
    (`run/group_size.py`). The same number is used for every group and every
    call, so compiled-program shapes are stable across a run. A call with
    fewer scenarios than the fixed size pads up to it and reuses the program.

    ``gains_cell_weights`` is the capital-gains model's per-household cell
    weights, stacked by year (``behavior/kg/prepass.CellWeights``). Like the
    heir structure it depends on the data alone and is the same for every
    scenario, so it is held once on the host and sent to the device one year
    at a time. A batch with no gains scenario leaves this out.

    ``program_cache`` is a dictionary the caller keeps between calls. A gains
    scenario's years run twice around one host solve with the same compiled
    year and different cell tables. The cache is keyed by the program, the
    years and ``scenarios_per_program``, not by the number of scenarios in the
    call, so both executions and any later smaller call at the same fixed size
    hit one entry. An entry is written as soon as the year program exists and
    is written again when the totals and the reported tables are compiled. On
    a hit the counters report zero programs compiled for the programs the
    entry already held.

    ``rates_only`` runs the projection years for their capital-gains rate
    exports alone. The year program runs every year and the wealth state
    carries forward as usual, but no totals, no reported tables and no
    per-household detail are formed, and the returned runs hold the rate
    exports and nothing else. This is the first of a gains scenario's two
    executions, which the host-side selling solve reads and the second
    execution then replaces.
    """
    requested_years = one_year.check_years(years)
    trim_every_year = host_memory.trim_after_every_year()
    n_years = len(requested_years)
    if scenarios_per_program < 1:
        raise ValueError("a compiled program holds at least one scenario lane")
    if not scenarios:
        raise ValueError("a batch needs at least one scenario")
    names = [scenario.name for scenario in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("two scenarios in one batch cannot share a name")

    on_device = put_the_run_on_the_device(
        population, scenarios, requested_years, heir_structure, gains_cell_weights
    )
    host_memory.note("after putting the run on the device", trim=True)

    runs: dict[str, assemble.SimulationRun] = {}
    group_counters: list[SwitchGroupCounters] = []
    for program, positions in group_by_switches(scenarios).items():
        in_group = on_device.lanes(positions, program.switch_fingerprint)

        # A year is compiled once, at the fixed number of lanes, and called
        # once per projection year by the loop below. A year an earlier call
        # compiled is found in ``program_cache``.
        programs = GroupPrograms(
            on_device,
            program,
            in_group,
            requested_years,
            scenarios_per_program,
            program_cache,
        )
        host_memory.note(
            "after the year program compiles"
            if programs.programs_compiled
            else "after the year program is found in the cache",
            trim=True,
        )

        # Scenarios run in chunks of exactly ``scenarios_per_program`` lanes.
        chunks = chunk_positions(len(in_group), scenarios_per_program)
        # The last chunk is padded to the full size by repeating its final
        # real scenario; the padding lanes' answers are never read.
        padding_lanes = len(chunks) * scenarios_per_program - len(in_group)

        for chunk in chunks:
            lanes = [in_group[position] for position in chunk]
            # The wealth stocks every lane starts from, carried year to year.
            state = stack_lanes([lane.initial_state for lane in lanes])
            one_pass = PassThroughTheYears(programs, lanes)
            for index in range(n_years):
                population_this_year = one_pass.read_the_population(index)
                inputs = one_pass.this_years_inputs(index, population_this_year, state)
                state, outputs = one_pass.run_the_year(inputs)
                if rates_only:
                    del inputs
                    one_pass.keep_the_gains_rates(outputs)
                else:
                    totals = one_pass.form_the_totals(outputs, population_this_year)
                    reports = one_pass.form_the_reported_tables(
                        outputs, inputs, population_this_year
                    )
                    # Release this year's inputs and per-household columns
                    # before the next year runs, so the device does not hold
                    # two years of each at once.
                    del inputs
                    one_pass.keep(outputs, totals, reports)
                del outputs
                if trim_every_year:
                    host_memory.give_back_free_memory()
                host_memory.note(f"after projection year {requested_years[index]}")

            # Only the real lanes are read; a padding lane's answers stay in
            # the chunk.
            if rates_only:
                exports = one_pass.finish_the_gains_rates()
                host_memory.note("after the chunk's rate exports are stacked", trim=True)
                for lane, position in _real_lanes(chunk):
                    runs[in_group[position].name] = (
                        assemble.simulation_run_with_only_the_gains_rates(
                            _one_lanes_exports(exports, lane), requested_years
                        )
                    )
            else:
                answers = one_pass.finish()
                host_memory.note("after the chunk's years are stacked", trim=True)
                for lane, position in _real_lanes(chunk):
                    scenario = in_group[position]
                    runs[scenario.name] = assemble.assemble_simulation_run(
                        answers.for_lane(lane),
                        requested_years,
                        program.detail_request,
                        on_device.detail_population,
                    )

        group_counters.append(
            SwitchGroupCounters(
                switch_fingerprint=program.switch_fingerprint,
                scenarios=len(in_group),
                chunk_size=scenarios_per_program,
                chunks=len(chunks),
                padding_lanes=padding_lanes,
                programs_compiled=programs.programs_compiled,
                demand_bytes=programs.demand_bytes(),
                free_bytes=on_device.measured.free_bytes,
                compile_seconds=programs.compile_seconds,
                execute_seconds=programs.execute_seconds,
            )
        )

    return BatchRun(
        runs={name: runs[name] for name in names},
        counters=BatchCounters(
            device=str(on_device.device),
            memory_source=on_device.measured.source,
            resident_bytes=on_device.resident_bytes,
            groups=tuple(group_counters),
            peak_device_bytes_in_use=peak_device_bytes_in_use(on_device.device),
        ),
    )


def _one_lanes_exports(exports: dict, lane: int) -> dict:
    """One lane's capital-gains rate exports, cut out of the chunk's stacked arrays."""
    return jax.tree.map(lambda value: value[:, lane], exports)


def _real_lanes(chunk: Sequence[int]):
    """The lanes of a chunk that hold real scenarios.

    Padding repeats the last real scenario, so a position that appears twice in
    a chunk is padding after its first appearance.
    """
    seen: set[int] = set()
    for lane, position in enumerate(chunk):
        if position not in seen:
            seen.add(position)
            yield lane, position


__all__ = [
    "BatchCounters",
    "BatchRun",
    "BatchScenario",
    "ScenarioProgram",
    "SwitchGroupCounters",
    "chunk_positions",
    "group_by_switches",
    "run_scenarios",
]
