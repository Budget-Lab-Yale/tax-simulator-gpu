"""The scenario loop: running a plan's scenarios in fixed-size groups.

`execute_run` is the second step of a run, after `run/plan.py` has produced
the plan. It resolves every scenario's provenance up front, starts the
population read, and hands the scenarios to `run_scenario_groups`, which does
this for each group of scenarios that can share one compiled program:

1. prepare the group's scenarios on the shared layout;
2. solve the frozen capital-gains pass for the scenarios that run the gains
   model;
3. run the projection years once (`years/loop.py`);
4. solve the selling choice from that first execution
   (`years/gains_solve.py`);
5. run the years again with the solved gains behavior;
6. hand the finished scenarios to the writer and free the group's memory.

Scenarios that do not run the gains model stop after step 3. Grouping, the
group size, the compiled-program cache, and the compile and memory counters
are in `run/grouping.py`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from taxmodel.behavior import loader as behavior_loader
from taxmodel.behavior import parameters as behavior_parameters
from taxmodel.behavior.gains import builders
from taxmodel.behavior.kg import prepass
from taxmodel.config import DEFAULT_LAYER, leg_or_none
from taxmodel.config.interfaces import (
    interface_root,
)
from taxmodel.frame.population import prepare_population
from taxmodel.reporting import assemble, heir_inputs, run_tables
from taxmodel.run import compile_cache, host_memory, output_directory
from taxmodel.run.grouping import (
    CompiledPartition,
    Family,
    compile_partition_laws,
    family_counters,
    free_the_gains_rate_exports,
    group_size_for,
    groups_that_can_share_a_program,
    read_the_machine,
    run_counters,
    runs_the_gains_model,
    share_the_familys_calculators,
    split_into_families,
    timed,
)
from taxmodel.run.plan import RunPlan, ScenarioSpec, check_behavior_legs, population_for
from taxmodel.scenario import gains_inputs, prepare
from taxmodel.scenario import provenance as scenario_provenance
from taxmodel.years import device, gains_solve, loop

# The scenario loop
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioResult:
    """One scenario's finished projection, and the provenance it ran under."""

    spec: ScenarioSpec
    projection: assemble.ProjectionRun
    provenance: scenario_provenance.Provenance

class PendingPopulation:
    """A population being read on a background thread while the run starts up.

    The read needs only the folder and the years, so it starts as soon as
    those are known. The run collects it at the first point that needs a
    household, which is loading the heir structure. Provenance resolution,
    opening the run directory, partitioning the scenarios, and compiling the
    partition's laws all run in the meantime.

    On a machine that holds every year at once this reads the whole
    population block. On one that reads a year at a time it reads only the
    record universe and the random draws, and the year-files stay on disk
    until the year loop reaches them.

    A read that fails raises where the run collects it. A run that fails
    before collecting the read calls `abandon`.
    """

    def __init__(
        self,
        root: str | Path,
        years: Sequence[int],
        read_every_year_up_front: bool = True,
    ) -> None:
        self._started = time.perf_counter()
        self._thread = ThreadPoolExecutor(max_workers=1)
        self._reading: Future = self._thread.submit(
            prepare_population,
            root,
            tuple(years),
            read_every_year_up_front=read_every_year_up_front,
        )

    def block_until_read(self) -> Any:
        """Wait for the read to finish and return the population.

        Prints the time the read took and the part of it the run spent
        waiting. A failed read raises here.
        """
        waiting_since = time.perf_counter()
        try:
            population = self._reading.result()
        finally:
            self.abandon()
        finished = time.perf_counter()
        print(
            "host stopwatch: reading the population took "
            f"{finished - self._started:.1f} s on a background thread, of which "
            f"{finished - waiting_since:.1f} s was spent waiting for it with "
            "nothing else to do"
        )
        return population

    def abandon(self) -> None:
        """Stop waiting on the read.

        Called on the way out of a run that failed before it collected the
        population. A read that has not started is cancelled. One already in
        flight cannot be interrupted inside pyarrow, but the run does not wait
        for it, and Python collects the thread at exit. Calling this twice is
        harmless.
        """
        self._reading.cancel()
        self._thread.shutdown(wait=False, cancel_futures=True)


def _population_in_hand(population: Any) -> Any:
    """The population block, after waiting for a background read if one is running."""
    if isinstance(population, PendingPopulation):
        return population.block_until_read()
    return population


def heir_structure(
    years: Sequence[int],
    population: Any,
    economy: str | None,
    config_root: str | Path,
) -> Any:
    """The heirs of each year's decedents and the amounts they inherit.

    Bequests are measured before tax, so this does not vary by scenario. It
    depends on the economy leg, which pins the Estate-Tax-Distribution
    publication to read.
    """
    with timed("loading the heir structure"):
        heirs = heir_inputs.load_heir_structure(
            years,
            population.universe.ids,
            economy=leg_or_none(economy),
            config_root=config_root,
        )
    return heirs


def run_scenario_groups(
    scenarios: Sequence[ScenarioSpec],
    provenance: dict[str, scenario_provenance.Provenance],
    *,
    deliver: Callable[[tuple[ScenarioResult, ...]], None],
    population: Any,
    config_root: str | Path = scenario_provenance.DEFAULT_CONFIG_ROOT,
) -> dict[str, Any]:
    """Run every scenario of the runscript in fixed-size groups.

    The scenarios are partitioned by projection window and by what their
    economy legs resolve to (`groups_that_can_share_a_program`). Each
    partition's laws are compiled to one shared layout on the host, so every
    scenario in the partition has identical array shapes.

    Each partition is split into families of scenarios that can share one
    compiled program (`split_into_families`), and each family runs in groups
    of a fixed size chosen from the machine's memory (`group_size_for`). Each
    group is prepared, executed, and handed to ``deliver`` when it finishes.
    ``deliver`` is `execute_run`'s writer, which writes each scenario's tables
    and keeps only a small per-scenario summary, so peak memory is one group's
    worth regardless of the runscript's length.

    A grouped run produces the same numbers as the same scenarios run one at a
    time, since every group calls the same device function.

    Returns the counters the run manifest carries: one entry per family with
    the group size chosen for it and its compile and execution time.

    `population` is a population block already in memory, or a
    `PendingPopulation` whose read `execute_run` started on a background
    thread. The read is collected just before the first thing that needs a
    household.
    """
    machine = read_the_machine()
    host_memory.note("run_scenario_groups begins")
    family_entries: list[dict[str, Any]] = []
    every_call: list[loop.BatchCounters] = []

    for (years, economy_key), partition in groups_that_can_share_a_program(
        scenarios, config_root
    ).items():
        # Every economy leg in the partition resolves the growth series and the
        # shared publications identically, so any member's leg serves the
        # shared reads.
        economy = leg_or_none(partition[0].economy) or DEFAULT_LAYER
        with timed("compiling the partition's laws to one shared layout"):
            compiled = compile_partition_laws(partition, years, economy, config_root)
        host_memory.note("after compiling the partition's laws", trim=True)

        # The heir structure is the first thing that needs a household, so the
        # background population read is collected here.
        population = _population_in_hand(population)
        host_memory.note("after the population read is collected", trim=True)

        # Read once per partition, since every economy leg in it resolves to
        # the same Estate-Tax-Distribution publication.
        heirs = heir_structure(years, population, economy, config_root)
        print(heir_inputs.describe(heirs))
        host_memory.note("after the heir structure")

        # The behavior step for scenarios without a gains model. Built once per
        # partition, since the batch runner compares behavior steps by identity
        # when deciding which scenarios share a compiled program.
        plain_behavior = behavior_loader.plain_behavior_builder()

        # The inputs every gains scenario in the partition shares: the
        # household inputs, the program builders, and the cell weights. Built
        # when the first gains family arrives.
        gains_pieces: SharedGainsPieces | None = None

        partition_inputs = PartitionInputs(
            scenarios=tuple(partition),
            compiled=compiled,
            years=years,
            config_root=config_root,
            population=population,
            provenance=provenance,
            heirs=heirs,
            plain_behavior=plain_behavior,
            program_cache=machine.program_cache,
            deliver=deliver,
        )

        for family in split_into_families(
            partition, compiled.charity_double_run, provenance, years
        ):
            if family.runs_the_gains_model and gains_pieces is None:
                with timed("reading the shared gains inputs"):
                    gains_pieces = shared_gains_pieces(
                        population, years, economy, config_root
                    )
                host_memory.note("after the shared gains inputs", trim=True)
            # The group size is chosen from the machine's memory
            # (`run/group_size.py`).
            size = group_size_for(family, years, population, machine)
            per_call = _run_family(
                family,
                partition_inputs,
                gains_pieces if family.runs_the_gains_model else None,
                size,
            )
            family_entries.append(
                family_counters(family, years, economy_key, size, per_call)
            )
            every_call.extend(per_call)

    return run_counters(family_entries, every_call)


@dataclass(frozen=True)
class PartitionInputs:
    """The inputs shared by every group of one partition.

    Everything here is settled once per partition, before its first family
    runs: the scenarios and their laws compiled to one shared layout, the run
    years, the configuration root, the population block, each scenario's
    resolved provenance, the heir structure, the behavior step for scenarios
    without the gains model, the run-wide compiled-program cache, and the
    writer each finished group is handed to. Inputs that vary by family (the
    shared gains pieces, the group size) are passed as separate arguments.
    """

    scenarios: tuple[ScenarioSpec, ...]
    compiled: CompiledPartition
    years: tuple[int, ...]
    config_root: str | Path
    population: Any
    provenance: dict[str, scenario_provenance.Provenance]
    heirs: Any
    plain_behavior: Any
    program_cache: dict
    deliver: Callable[[tuple[ScenarioResult, ...]], None]


def _run_family(
    family: Family,
    partition: PartitionInputs,
    gains_pieces: SharedGainsPieces | None,
    size: int,
) -> list[loop.BatchCounters]:
    """Run one family in groups of at most `size` scenarios.

    The first group compiles the family's program and supplies the shared
    calculators, and every later group reuses both. A last group smaller than
    `size` is padded to `size` inside `years/loop.py`, so every group has the
    same shape and hits the program cache.
    """
    members = [partition.scenarios[position] for position in family.positions]
    laws = [partition.compiled.laws[position] for position in family.positions]
    shared_calculators: dict[str, Any] | None = None
    per_call: list[loop.BatchCounters] = []
    for start in range(0, len(members), size):
        shared_calculators, counters = _run_group(
            members[start : start + size],
            laws[start : start + size],
            family,
            partition,
            shared_calculators,
            gains_pieces,
            size,
        )
        per_call.extend(counters)
    return per_call


def _run_group(
    members: Sequence[ScenarioSpec],
    laws: Sequence[object],
    family: Family,
    partition: PartitionInputs,
    shared_calculators: dict[str, Any] | None,
    gains_pieces: SharedGainsPieces | None,
    size: int,
) -> tuple[dict[str, Any], list[loop.BatchCounters]]:
    """Run one group, start to finish.

    Prepare the scenarios; solve the frozen capital-gains pass for those that
    run the gains model; run the projection years; solve the selling choice
    and run the years again; hand the finished scenarios to the writer.

    Everything built here (the bound laws, the device results, the assembled
    projections) is a local and is freed on return. What outlives a group is
    what the writer behind ``deliver`` keeps, the compiled program in the run's
    program cache, and the shared calculators returned for the family's next
    group.
    """
    provenance = partition.provenance

    with timed("preparing the group's scenarios"):
        prepared = _prepare_the_group(members, laws, family, partition)
    host_memory.note("after preparing the group's scenarios")
    prepared, shared_calculators = share_the_familys_calculators(
        prepared, shared_calculators
    )

    # The frozen capital-gains pass, the law arithmetic of each reform's
    # treatment of gains at death, reads no rate and is solved before any year
    # is simulated. Empty for a family without the gains model.
    gains, solve_group = _gains_states_for_group(
        members, prepared, provenance, gains_pieces
    )
    host_memory.note("after the frozen gains pass")

    # The first execution. Gains scenarios run with the frozen tables solved
    # and the behavioral tables at the do-nothing values. It is run for the
    # per-household rate columns the host-side selling solve reads: the second
    # execution reproduces every world whose numbers are reported, so this one
    # forms no totals and no tables. Other scenarios run only this execution
    # and it forms everything.
    with timed("the first execution, compiles included"):
        first = _run_the_years(
            [
                batch_scenario(
                    spec,
                    ready,
                    gains.get(spec.id),
                    behavior_leg=provenance[spec.id].behavior_leg,
                    plain_behavior_builder=partition.plain_behavior,
                )
                for spec, ready in zip(members, prepared, strict=True)
            ],
            partition,
            gains_pieces.cell_weights if gains else None,
            size,
            rates_only=bool(gains),
        )
    host_memory.note("after the first execution", trim=True)
    per_call = [first.counters]
    final_runs = dict(first.runs)

    # For a gains group, the selling choice is solved on the host from the
    # exported rates and the same compiled program runs again with the
    # behavioral tables filled in. The second execution's conventional world
    # is the answer. Its other worlds do not read the behavioral tables and
    # reproduce the first execution, so every reported number comes from here.
    behavioral_tables: dict[str, Any] = {}
    if gains:
        with timed("the behavioral gains solves"):
            behavioral_tables, second_batch = _solve_the_selling_choice(
                members, prepared, gains, solve_group, gains_pieces, first, partition
            )
        host_memory.note("after the behavioral gains solves")
        free_the_gains_rate_exports(first, behavioral_tables)
        host_memory.note("after freeing the gains rate exports", trim=True)
        with timed("the second execution"):
            second = _run_the_years(
                second_batch, partition, gains_pieces.cell_weights, size
            )
        host_memory.note("after the second execution", trim=True)
        per_call.append(second.counters)
        final_runs.update(second.runs)

    with timed("assembling the projections and rate tables"):
        results = tuple(
            ScenarioResult(
                spec=spec,
                projection=replace(
                    assemble.projection_from_simulation(ready, final_runs[spec.id]),
                    gains_cell_tables=behavioral_tables.get(spec.id),
                ),
                provenance=provenance[spec.id],
            )
            for spec, ready in zip(members, prepared, strict=True)
        )
    host_memory.note("after assembling the projections")
    partition.deliver(results)
    host_memory.note("after the group's folders are built", trim=True)
    return shared_calculators, per_call


def _prepare_the_group(
    members: Sequence[ScenarioSpec],
    laws: Sequence[object],
    family: Family,
    partition: PartitionInputs,
) -> list[prepare.PreparedScenario]:
    """Bind each member's laws and build its calculators (`scenario/prepare.py`)."""
    compiled = partition.compiled
    return [
        prepare.prepare_scenario(
            partition.population,
            partition.years,
            config_root=partition.config_root,
            alternative=spec.alternative_or_stack_name,
            economy=leg_or_none(spec.economy),
            behavior=leg_or_none(spec.behavior),
            detail_request=family.detail_request,
            name=spec.id,
            compiled_law=law,
            compiled_baseline_law=compiled.compiled_baseline,
            provenance=partition.provenance[spec.id],
            shared_baseline=compiled.shared_baseline,
        )
        for spec, law in zip(members, laws, strict=True)
    ]


def _run_the_years(
    batch: Sequence[loop.BatchScenario],
    partition: PartitionInputs,
    cell_weights: Any,
    size: int,
    rates_only: bool = False,
) -> loop.BatchRun:
    """One execution of the projection years for this group (`years/loop.py`).

    With ``rates_only`` the years run for the capital-gains rate exports alone
    and no reported number is formed.
    """
    return loop.run_scenarios(
        partition.population,
        batch,
        years=partition.years,
        heir_structure=partition.heirs,
        gains_cell_weights=cell_weights,
        program_cache=partition.program_cache,
        scenarios_per_program=size,
        rates_only=rates_only,
    )


def _solve_the_selling_choice(
    members: Sequence[ScenarioSpec],
    prepared: Sequence[prepare.PreparedScenario],
    gains: dict[str, GainsScenarioState],
    solve_group: gains_solve.GainsSolveGroup | None,
    gains_pieces: SharedGainsPieces | None,
    first: loop.BatchRun,
    partition: PartitionInputs,
) -> tuple[dict[str, Any], list[loop.BatchScenario]]:
    """Solve every gains scenario's selling choice, and build the second batch.

    The choice is solved from the rates the first execution exported, for
    every scenario in one compiled program, since the scenarios differ only in
    their input numbers. Returns the solved behavioral cell tables by scenario
    and the batch for the second execution, in which each gains scenario
    carries its solved tables.
    """
    assert solve_group is not None
    assert gains_pieces is not None
    in_solve_order = list(solve_group.scenario_ids)
    solved_by_scenario = dict(
        zip(
            in_solve_order,
            gains_solve.solve_behavioral_tables_for_group(
                solve_group,
                [
                    first.runs[scenario_id].gains_rate_exports
                    for scenario_id in in_solve_order
                ],
            ),
            strict=True,
        )
    )
    behavioral_tables: dict[str, Any] = {}
    second_batch = []
    for spec, ready in zip(members, prepared, strict=True):
        state = gains[spec.id]
        solved = solved_by_scenario[spec.id]
        behavioral_tables[spec.id] = solved.cell_tables
        second_batch.append(
            batch_scenario(
                spec,
                ready,
                state,
                behavioral=solved.cell_tables,
                conversions=(
                    solved.conversions_by_household_leg if state.converts else None
                ),
                behavior_leg=partition.provenance[spec.id].behavior_leg,
                plain_behavior_builder=partition.plain_behavior,
            )
        )
    return behavioral_tables, second_batch


@dataclass(frozen=True)
class GainsScenarioState:
    """One gains scenario's host-side inputs, settled before anything runs.

    ``law`` is this scenario's law on gains at death. ``frozen_tables`` is the
    frozen pass solved under it for the first-order world, and
    ``mechanical_frozen_tables`` the same pass with the corporate markdown of
    share prices, for the mechanical worlds. Left at None, the mechanical
    worlds read ``frozen_tables``. ``pricers`` is shared between
    scenarios whose calculators settled the charitable double-run switch the
    same way. ``shared`` is what the scenario has in common with every other
    gains scenario of its partition: the households as the gains model reads
    them, the program pieces, and the charity deflator.
    """

    law: gains_solve.GainsLaw
    frozen_tables: Any
    pricers: builders.GainsRatePricers
    shared: SharedGainsPieces
    mechanical_frozen_tables: Any = None
    # Whether this scenario's kg_dynamics section binds a conversion
    # calibration, so that the behavioral solve prices conversions of salary
    # into equity. The conversion module runs either way, on a table that is
    # all zeros where nothing was priced.
    converts: bool = False

@dataclass(frozen=True)
class SharedGainsPieces:
    """The inputs every gains scenario in one partition shares, read and built once.

    The household inputs come from the data (`scenario/gains_inputs.py`). The
    program builders are shared because every gains scenario runs the same
    chain and the batch runner compares builders by identity. `cell_weights`
    are the weights the compiled program averages the measured rates over:
    each household's age cell and its dollars of realized and unrealized gain.

    `pricers_by_switch` is filled as scenarios arrive. Pricers are shared
    between scenarios whose calculators settled the charitable double-run
    switch the same way.

    `charity_deflator_by_year` brings each run year's estates back to the
    dollars the charitable bequest logits were fit on. It depends on the
    economy leg's macro projection and the run's years.
    """

    years: tuple[int, ...]
    host_records: Any
    frozen_builder: Any
    behavioral_builder: Any
    pricers_by_switch: dict[bool, builders.GainsRatePricers]
    macro_root: Path
    charity_deflator_by_year: Any
    cell_weights: Any


def shared_gains_pieces(
    population: Any,
    years: tuple[int, ...],
    economy: str,
    config_root: str | Path,
) -> SharedGainsPieces:
    """Read and build the `SharedGainsPieces` for one partition."""
    macro_root = interface_root(
        "Macro-Projections", economy=leg_or_none(economy), config_root=config_root
    )
    _estate_parameters, cluster_death_weight_cap = (
        scenario_provenance.read_estate_bridge()
    )
    host_records = gains_inputs.host_records_by_year(
        population, years, cluster_death_weight_cap
    )
    chain_spec = builders.read_gains_chain_spec()
    charity_deflator_by_year = builders.charity_deflator_for_years(macro_root, years)
    return SharedGainsPieces(
        years=years,
        host_records=host_records,
        frozen_builder=builders.frozen_gains_builder_for(chain_spec),
        behavioral_builder=builders.behavioral_gains_builder_for(chain_spec),
        pricers_by_switch={},
        macro_root=macro_root,
        charity_deflator_by_year=charity_deflator_by_year,
        cell_weights=prepass.cell_weights_by_year(host_records),
    )


def _gains_states_for_group(
    group: Sequence[ScenarioSpec],
    prepared: Sequence[prepare.PreparedScenario],
    provenance: dict[str, scenario_provenance.Provenance],
    pieces: SharedGainsPieces | None,
) -> tuple[dict[str, GainsScenarioState], gains_solve.GainsSolveGroup | None]:
    """The inputs this group's gains scenarios need before the first execution.

    The frozen solve reads no rate and runs before any year is simulated. Each
    scenario's cell tables and gains-at-death law are runtime inputs to the
    compiled program, so scenarios that share a behavior leg and a program
    shape share one program. A group with no gains scenarios gets an empty
    answer.
    """
    running = [
        (spec, ready)
        for spec, ready in zip(group, prepared, strict=True)
        if runs_the_gains_model(provenance[spec.id])
    ]
    if not running:
        return {}, None
    assert pieces is not None, "a gains scenario needs the shared gains pieces"

    with timed("the gains frozen solves"):
        # Every scenario's frozen solve, in one compiled program. Each
        # scenario's law on gains at death is an argument to the program.
        solve_group = gains_solve.prepare_gains_solve_group(
            pieces.host_records,
            pieces.years,
            scenario_ids=[spec.id for spec, _ in running],
            compiled_laws=[ready.compiled_law for _, ready in running],
            behavior_legs=[provenance[spec.id].behavior_leg for spec, _ in running],
            macro_root=pieces.macro_root,
            corporate_asset_exposures=[
                ready.corporate_asset_exposure for _, ready in running
            ],
            equity_markdowns=[ready.equity_markdown_by_year for _, ready in running],
            mechanical_equity_markdowns=[
                ready.mechanical_equity_markdown_by_year for _, ready in running
            ],
        )
        frozen_tables = gains_solve.solve_frozen_tables_for_group(solve_group)

        states = {
            spec.id: gains_state_for_scenario(
                pieces,
                ready,
                law=solve_group.laws_by_scenario[position],
                frozen_tables=frozen_tables[position].first_order,
                mechanical_frozen_tables=frozen_tables[position].mechanical,
                converts=provenance[spec.id].behavior_leg.converts,
            )
            for position, (spec, ready) in enumerate(running)
        }
    return states, solve_group


def gains_state_for_scenario(
    pieces: SharedGainsPieces,
    ready: prepare.PreparedScenario,
    *,
    law: gains_solve.GainsLaw,
    frozen_tables: Any,
    converts: bool,
    mechanical_frozen_tables: Any = None,
) -> GainsScenarioState:
    """One gains scenario's state, once its frozen pass is solved.

    The rate pricers are shared between scenarios whose calculators settled
    the charitable double-run switch the same way. The first scenario with
    each answer builds them.
    """
    if ready.charity_double_run not in pieces.pricers_by_switch:
        pieces.pricers_by_switch[ready.charity_double_run] = (
            builders.make_gains_rate_pricers(ready.charity_double_run)
        )
    return GainsScenarioState(
        law=law,
        frozen_tables=frozen_tables,
        pricers=pieces.pricers_by_switch[ready.charity_double_run],
        shared=pieces,
        converts=converts,
        mechanical_frozen_tables=mechanical_frozen_tables,
    )


def batch_scenario(
    spec: ScenarioSpec,
    ready: prepare.PreparedScenario,
    gains_state: GainsScenarioState | None,
    behavioral: Any = None,
    conversions: Any = None,
    behavior_leg: Any = None,
    *,
    plain_behavior_builder: Any,
) -> loop.BatchScenario:
    """One scenario as the batch runner takes it, with or without the gains model.

    Every scenario runs every behavioral margin. A scenario with no gains
    state runs the pinned chain with the applier's slot at neutral and its own
    parameters as data. With a gains state, the program includes the gains
    applier and the rate exports, and this scenario's cell tables and
    gains-at-death law are runtime data. ``behavioral`` is the solved
    behavioral cell tables for the second execution. When it is left out, the
    conventional world runs on the do-nothing tables, which is the first
    execution. ``conversions`` is the solved [year, household, leg]
    conversion table. For the first execution of a converting scenario a
    zero-filled table of the same shape stands in, so both executions present
    the compiled program the same shapes.
    """
    program = loop.ScenarioProgram(
        switch_fingerprint=ready.switch_fingerprint,
        frame_builder=ready.frame_builder,
        tax_calculator=ready.tax_calculator,
        price_calculator=ready.price_calculator,
        taxes_and_prices=ready.taxes_and_prices,
        detail_request=ready.detail_request,
        behavior_builder=plain_behavior_builder,
    )
    gains_tables = None
    if gains_state is not None:
        program = replace(
            program,
            behavior_builder=gains_state.shared.behavioral_builder,
            frozen_gains_builder=gains_state.shared.frozen_builder,
            runs_the_gains_model=True,
            gains_rate_pricers=gains_state.pricers,
        )
        if conversions is None:
            # Nothing priced yet (the first execution, or a scenario that
            # prices no conversions). The conversion module runs in every
            # program and reads this table every year, so it must have its
            # full shape.
            n_years = len(gains_state.shared.host_records)
            n_households = gains_state.shared.host_records[0]["weight"].shape[0]
            conversions = jnp.zeros(
                (n_years, n_households, len(gains_solve.stock_inputs.CONVERSION_LEGS))
            )
        gains_tables = builders.runtime_tables(
            gains_state.law,
            gains_state.frozen_tables,
            behavioral,
            conversions=conversions,
            charity_deflator=gains_state.shared.charity_deflator_by_year,
            mechanical_frozen_cell_tables=gains_state.mechanical_frozen_tables,
        )
    return loop.BatchScenario(
        name=spec.id,
        program=program,
        reform_laws=ready.reform_laws,
        baseline_laws=ready.baseline_laws,
        wealth_profiles=ready.wealth_profiles,
        gains_tables=gains_tables,
        # This scenario's behavioral parameters, by margin and parameter name.
        # They are data, so two scenarios that differ only in an elasticity
        # share one compiled program.
        behavior_parameters=(
            behavior_parameters.all_neutral()
            if behavior_leg is None
            else behavior_parameters.for_leg(behavior_leg)
        ),
        corporate_dollars=ready.corporate_dollars,
    )


def _every_year(scenarios: Sequence[ScenarioSpec]) -> tuple[int, ...]:
    """Every projection year any scenario asks for. One population load serves all."""
    years: set[int] = set()
    for spec in scenarios:
        years.update(int(year) for year in spec.row.years)
    return tuple(sorted(years))

# The group runner `execute_run` calls. It is handed the planned scenarios and
# their resolved provenance, calls ``deliver`` with each finished group of
# `ScenarioResult`s, and returns the counters dictionary the run manifest
# carries. Tests substitute a runner that produces stand-in results without
# touching JAX.
GroupRunner = Callable[..., dict[str, Any]]

# --------------------------------------------------------------------------
# Running a plan
# --------------------------------------------------------------------------


def resolve_scenario_provenance(
    plan: RunPlan,
    provenance_root: str | Path = scenario_provenance.DEFAULT_PROVENANCE_ROOT,
    accept_stale_calibration: bool = False,
) -> dict[str, scenario_provenance.Provenance]:
    """Resolve every scenario's three legs before any of them runs.

    A stale calibrated value stops the run before any scenario is computed.

    Each scenario is resolved against the full folded list of reform folders it
    runs and against its own economy and behavior legs, which a stack rung may
    have changed from the runscript row.
    """
    resolved: dict[str, scenario_provenance.Provenance] = {}
    for spec in plan.scenarios:
        resolved[spec.id] = scenario_provenance.resolve_scenario_provenance(
            plan.config_root,
            economy=leg_or_none(spec.economy),
            behavior=leg_or_none(spec.behavior),
            provenance_root=provenance_root,
            law_layers=spec.law_layers,
            accept_stale_calibration=accept_stale_calibration,
        )
    return resolved


def execute_run(
    plan: RunPlan,
    population_root: str | Path | None = None,
    provenance_root: str | Path = scenario_provenance.DEFAULT_PROVENANCE_ROOT,
    group_runner: GroupRunner = run_scenario_groups,
    execution: dict[str, Any] | None = None,
    population: Any = None,
    accept_stale_calibration: bool = False,
) -> Path:
    """Run the planned scenarios and write the run directory. Returns its path.

    Writing is interleaved with running. `output_directory.begin_run` opens
    the directory under its `.partial` name, every group of finished scenarios
    is written as the group runner hands it over, and
    `output_directory.finish_run` writes the stacked tables and the run
    manifest, then renames the directory to its final name. A directory
    without the `.partial` suffix is a complete run. A stopped run leaves a
    partial directory with finished scenario folders and no run manifest.

    Once a scenario's tables are on disk, only the small `ScenarioAggregates`
    summary is kept, so host memory does not grow with the runscript's length.

    `population` is a block already in memory, handed to the runner instead of
    a folder to read. `population_root` is recorded in the manifests either
    way.

    The population read starts here on a background thread
    (`PendingPopulation`), since the folder and years are known and nothing
    before the heir structure touches a household.
    """
    # Point XLA at the compile cache before anything compiles.
    print(compile_cache.switch_on_compile_cache())
    allocation = host_memory.set_the_c_library_allocation_thresholds()
    if allocation:
        print(allocation)
    host_memory.start_tracing_the_python_heap()
    host_memory.note("the run begins")

    check_behavior_legs(plan)
    root = (
        Path(population_root) if population_root is not None else population_for(plan)
    )
    if population is None:
        # A machine with a graphics card copies the whole population onto the
        # card, so it reads every year up front. One without reads a year when
        # the year loop reaches it.
        population = PendingPopulation(
            root,
            _every_year(plan.scenarios),
            read_every_year_up_front=device.population_stays_resident(),
        )

    # Everything from here on runs while the population is being read, so a
    # failure must also end the read, in the `finally` at the bottom. `state`
    # is named before the try because the run directory is opened inside it.
    state: output_directory.Building | None = None
    try:
        provenance = resolve_scenario_provenance(
            plan, provenance_root, accept_stale_calibration=accept_stale_calibration
        )
        for scenario_id, resolved in provenance.items():
            for line in resolved.staleness.warnings:
                print(f"{scenario_id}: stale (warning): {line}")
            # The gains model's code has changed since its numbers were fit.
            # This warns and never stops, since a git diff cannot tell a
            # comment edit from a change to the solver. Anything that stops the
            # run already raised while the leg was being resolved.
            for line in resolved.calibration_staleness.warnings:
                print(f"{scenario_id}: gains calibration (warning): {line}")
            if accept_stale_calibration and resolved.calibration_staleness.stops:
                for line in resolved.calibration_staleness.stops:
                    print(f"{scenario_id}: stale calibration ACCEPTED: {line}")

        state = output_directory.begin_run(plan)
        aggregates: dict[str, run_tables.ScenarioAggregates] = {}

        def write_finished_group(results: Sequence[ScenarioResult]) -> None:
            # The group runner calls this as each group finishes. The
            # scenario's device arrays are turned into tables here, and the
            # tables are handed to the helper thread that writes them. Only the
            # summary `write_scenario_results` returns is kept.
            with timed(f"building {len(results)} finished scenario folder(s)"):
                for result in results:
                    aggregates[result.spec.id] = (
                        output_directory.write_scenario_results(
                            state, result, root, plan.config_root
                        )
                    )

        counters = group_runner(
            plan.scenarios,
            provenance,
            deliver=write_finished_group,
            population=population,
            config_root=plan.config_root,
        )
        with timed("finishing the run directory"):
            final_path = output_directory.finish_run(
                state, plan, aggregates, counters, root, execution or {}
            )
        host_memory.note("after the run directory is finished", trim=True)
        return final_path
    finally:
        # On the ordinary path `finish_run` has already stopped the file
        # writer and the population has been collected, so both of these do
        # nothing. On the way out of a failed run they end the helper thread
        # and the population read.
        if state is not None:
            state.writer.stop()
        if isinstance(population, PendingPopulation):
            population.abandon()


# --------------------------------------------------------------------------
