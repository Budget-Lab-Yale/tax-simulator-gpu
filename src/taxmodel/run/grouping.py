"""Grouping scenarios into shared compiled programs, and the run's counters.

The scenario loop (`run/scenarios.py`) runs several scenarios through one
compiled program at a time. A partition is the scenarios that share a
projection window and an economy leg, so their laws can be compiled to one
shared layout (`groups_that_can_share_a_program`, `compile_partition_laws`).
A family is the members of a partition that also agree on everything else
that decides which code runs, so one compiled program serves them all
(`split_into_families`).

The rest of this file supports the scenario loop: the host-side stopwatch,
the compile and memory counters the run manifest records per family, the
device's memory and the run's compiled-program cache, the group size chosen
for a family, the calculators one group hands the next, and the release of
device memory between a gains group's two executions.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax

from taxmodel.config import DEFAULT_LAYER, law_layers, leg_or_none
from taxmodel.config.interfaces import (
    interface_root,
)
from taxmodel.law.batch import compile_batch
from taxmodel.law.indexes import index_table_from_frame
from taxmodel.reporting import detail as detail_output
from taxmodel.run import group_size
from taxmodel.run.plan import ScenarioSpec
from taxmodel.scenario import prepare
from taxmodel.scenario import provenance as scenario_provenance
from taxmodel.years import loop
from taxmodel.years.device import has_its_own_device_memory


def stopwatch(label: str, started: float) -> None:
    """Print one line with the elapsed time of a host-side phase of the run.

    Every run logs its own breakdown of the time spent outside the compiled
    program: reading data, compiling law, the gains solves, writing output.
    """
    print(f"host stopwatch: {label} took {time.perf_counter() - started:.1f} s")

@dataclass(frozen=True)
class Family:
    """Scenarios of one partition that can all run in one compiled program.

    Members agree on everything that decides which functions run: the
    projection window and the resolved economy inputs (the partition), whether
    the model computes the 1040 twice for the charitable choice, the
    per-household detail request, and whether the capital-gains model runs.
    `positions` index into the partition's scenario list, in runscript order.
    """

    charity_double_run: bool
    detail_request: detail_output.DetailRequest | None
    runs_the_gains_model: bool
    positions: tuple[int, ...]


def split_into_families(
    partition: Sequence[ScenarioSpec],
    charity_double_run: Sequence[bool],
    provenance: dict[str, scenario_provenance.Provenance],
    years: tuple[int, ...],
) -> list[Family]:
    """Split one partition into the sets that can each share a compiled program.

    Beyond the partition key, three things separate families: (a) whether the
    law computes the 1040 twice for the charitable choice; (b) the detail
    request, since requested per-household columns are extra outputs of the
    traced program; and (c) whether the scenario runs the capital-gains model,
    which adds the gains applier and the rate exports to the program.

    Families come back in order of first appearance, members in runscript
    order, so the run's output order matches the runscript.
    """
    members: dict[tuple[Any, ...], list[int]] = {}
    for position, spec in enumerate(partition):
        request = None if spec.detail is None else spec.detail.request(years)
        key = (
            charity_double_run[position],
            request,
            runs_the_gains_model(provenance[spec.id]),
        )
        members.setdefault(key, []).append(position)
    return [
        Family(
            charity_double_run=double_run,
            detail_request=request,
            runs_the_gains_model=runs_gains,
            positions=tuple(positions),
        )
        for (double_run, request, runs_gains), positions in members.items()
    ]


@dataclass(frozen=True)
class CompiledPartition:
    """One partition's laws on a shared layout, and the baseline they price against.

    `laws[i]` and `charity_double_run[i]` belong to the partition's i-th
    scenario. Every law has identical array shapes, so any group drawn from
    this partition presents the compiled program the same shapes.
    """

    laws: tuple
    charity_double_run: tuple[bool, ...]
    compiled_baseline: object
    shared_baseline: prepare.SharedBaseline


def compile_partition_laws(
    partition: Sequence[ScenarioSpec],
    years: tuple[int, ...],
    economy: str,
    config_root: str | Path,
) -> CompiledPartition:
    """Compile every law in the partition together, on the host, before anything runs.

    Baseline law is compiled once here as well, since every scenario in the
    partition prices its reform against it over the same years and growth
    series.
    """
    scenario_root = Path(config_root)
    resolved = [
        law_layers.resolve_law_stack(scenario_root, spec.law_layers)[-1]
        for spec in partition
    ]
    baseline = law_layers.resolve_law_stack(scenario_root, ())[0]

    growth, _index_source = scenario_provenance.index_series(
        None, leg_or_none(economy), config_root
    )
    compiled = compile_batch(
        [*resolved, baseline],
        years,
        index_table_from_frame(growth),
        lint_scope=sorted({layer for spec in partition for layer in spec.law_layers})
        or None,
    )
    compiled_baseline = compiled.laws[-1]
    return CompiledPartition(
        laws=compiled.laws[:-1],
        charity_double_run=compiled.charity_double_run[:-1],
        compiled_baseline=compiled_baseline,
        shared_baseline=prepare.load_shared_baseline(
            years,
            config_root=config_root,
            economy=leg_or_none(economy),
            compiled_baseline_law=compiled_baseline,
        ),
    )

def runs_the_gains_model(resolved: scenario_provenance.Provenance) -> bool:
    """Whether this scenario's behavior leg binds the capital-gains machinery.

    A leg that binds nothing under `kg_dynamics` runs no gains model.
    """
    leg = resolved.behavior_leg
    return leg is not None and leg.runs_the_gains_model

def _shared_economy_inputs(
    economy: str, config_root: str | Path
) -> tuple[str, str, str]:
    """The three inputs an economy leg resolves that a whole group shares.

    A group reads three things once: the growth series its laws are compiled
    against, the Estate-Tax-Distribution publication that says who inherits,
    and the Macro-Projections publication the gains model reads. Everything
    else an economy leg can change (the financing profile, the return deltas)
    is per-scenario data passed to the compiled program as numbers, so it does
    not constrain who shares a group.
    """
    leg = leg_or_none(economy)
    _growth, growth_source = scenario_provenance.index_series(None, leg, config_root)
    return (
        str(growth_source),
        str(
            interface_root(
                "Estate-Tax-Distribution", economy=leg, config_root=config_root
            )
        ),
        str(interface_root("Macro-Projections", economy=leg, config_root=config_root)),
    )


def groups_that_can_share_a_program(
    scenarios: Sequence[ScenarioSpec],
    config_root: str | Path = scenario_provenance.DEFAULT_CONFIG_ROOT,
) -> dict[tuple[tuple[int, ...], object], list[ScenarioSpec]]:
    """Split the run into partitions whose laws can be compiled together.

    Two things must match: (a) the projection window, since a year loop of a
    different length is a different program; and (b) the inputs the whole
    partition shares, as the scenario's economy leg resolves them: the growth
    series, the heir publication and the macro publication
    (`_shared_economy_inputs`). A partition is then cut further into families
    (`split_into_families`).

    The match is on the resolved inputs, not the leg's name. Two differently
    named legs often point at the same publications and differ only in
    per-scenario data such as the financing profile, and scenarios on such
    legs can share a compiled program.

    When every scenario sits on one economy leg the name is the key, and
    nothing is read from disk here.
    """
    legs = {leg_or_none(spec.economy) or DEFAULT_LAYER for spec in scenarios}
    resolved: dict[str, tuple[str, str, str]] = {}
    if len(legs) > 1:
        resolved = {leg: _shared_economy_inputs(leg, config_root) for leg in legs}
    groups: dict[tuple[tuple[int, ...], object], list[ScenarioSpec]] = {}
    for spec in scenarios:
        # An empty economy cell and the word `default` are the same leg, so
        # they must not be read as two.
        economy = leg_or_none(spec.economy) or DEFAULT_LAYER
        key = (tuple(int(year) for year in spec.row.years), resolved.get(economy, economy))
        groups.setdefault(key, []).append(spec)
    return groups


def family_counters(
    family: Family,
    years: tuple[int, ...],
    economy_key: object,
    size: int,
    per_call: Sequence[loop.BatchCounters],
) -> dict[str, Any]:
    """One family's row in the manifest's counters.

    A family compiles once, on its first group's first execution. A gains
    family may compile once more for the second execution. Every later group
    of the family, a padded remainder group included, hits the program cache.
    A higher compile count means shapes or function identities drifted between
    groups.
    """
    calls = [group for counter in per_call for group in counter.groups]
    return {
        # The family key, part by part: the window, the resolved economy
        # inputs, the charitable double-run switch, the detail request, and
        # whether the gains model ran.
        "first_year": int(years[0]),
        "last_year": int(years[-1]),
        "economy": str(economy_key),
        "charity_double_run": family.charity_double_run,
        "detail": (
            None
            if family.detail_request is None
            else {
                world: list(columns)
                for world, columns in family.detail_request.columns_by_world.items()
            }
        ),
        "runs_the_gains_model": family.runs_the_gains_model,
        "scenarios": len(family.positions),
        "scenarios_per_program": size,
        "groups_run": math.ceil(len(family.positions) / size),
        "padding_lanes": sum(group.padding_lanes for group in calls),
        "programs_compiled": sum(group.programs_compiled for group in calls),
        "compile_seconds": sum(group.compile_seconds for group in calls),
        "execute_seconds": sum(group.execute_seconds for group in calls),
        "demand_bytes": max((group.demand_bytes for group in calls), default=None),
        "free_bytes": min((group.free_bytes for group in calls), default=None),
    }


def run_counters(
    families: list[dict[str, Any]], per_call: Sequence[loop.BatchCounters]
) -> dict[str, Any]:
    """The whole run's compile and memory counters, with one entry per family.

    The performance test reads these from the run manifest: the number of
    programs compiled, the group size of each family, and the peak device
    memory.
    """
    return {
        "measured": True,
        "device": per_call[0].device,
        "memory_source": per_call[0].memory_source,
        "compiled_programs": sum(counter.programs_compiled for counter in per_call),
        "resident_bytes": max(counter.resident_bytes for counter in per_call),
        # The peak is read from the device itself, so every call of this run
        # reports the same number. It is absent on a CPU run, where JAX keeps
        # no device memory statistics.
        "peak_device_bytes_in_use": max(
            (
                counter.peak_device_bytes_in_use
                for counter in per_call
                if counter.peak_device_bytes_in_use is not None
            ),
            default=None,
        ),
        "compile_seconds": sum(counter.compile_seconds for counter in per_call),
        "execute_seconds": sum(counter.execute_seconds for counter in per_call),
        "families": families,
    }


# ---------------------------------------------------------------------------
# The machine, the group size, and the calculators one group hands the next
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def timed(label: str):
    """Time one host-side phase of the run and print the `stopwatch` line."""
    started = time.perf_counter()
    yield
    stopwatch(label, started)


@dataclass(frozen=True)
class RunMachine:
    """The machine's memory and the run's program cache, read once before the first group.

    ``device_total_bytes`` and ``has_its_own_device_memory`` feed the
    group-size arithmetic (`run/group_size.py`): the device's memory, and
    whether it is the device's own (a card) or a pool shared with the host (a
    laptop). The second also decides whether the population block stays
    resident on the device.

    ``program_cache`` is the run's compiled-program cache. A family's first
    group compiles, and every later group of that family reuses the cached
    program.
    """

    device_total_bytes: int
    has_its_own_device_memory: bool
    program_cache: dict


def read_the_machine() -> RunMachine:
    """Read the device's memory and open the run's compiled-program cache."""
    device = jax.devices()[0]
    return RunMachine(
        device_total_bytes=group_size.read_device_total_bytes(device),
        has_its_own_device_memory=has_its_own_device_memory(device),
        program_cache={},
    )


def group_size_for(
    family: Family, years: tuple[int, ...], population: Any, machine: RunMachine
) -> int:
    """The number of scenarios one compiled year carries for this family.

    A family that fits the memory estimate runs as one group of its own size.
    Otherwise it is split into the fewest groups the capacity allows, each of
    the same width: ten scenarios with capacity eight run as two groups of
    five. All groups share one compiled program.
    """
    count = len(family.positions)
    capacity = group_size.scenarios_per_program(
        len(years),
        population.universe.n_slots,
        family.runs_the_gains_model,
        machine.device_total_bytes,
        has_its_own_device_memory=machine.has_its_own_device_memory,
        n_scenarios=count,
    )
    groups = (count + capacity - 1) // capacity
    return (count + groups - 1) // groups


# The program functions a family shares, taken from its first scenario.
#
# Two scenarios share a compiled program only when the functions the year loop
# calls are the same Python objects; two separately built closures trace to
# two programs even when they compute the same thing
# (`years/loop.ScenarioProgram`). Preparing a scenario builds fresh closures,
# so every member of a family is given the first scenario's instead.
#
# The frame builder closes over the first scenario's corporate asset-exposure
# shares (`prepare.prepare_scenario`), so the whole family's frames are built
# with those shares. Separately compiled programs may round in a different
# order, and the resulting differences are floating-point noise.
SHARED_CALCULATOR_FIELDS = (
    "frame_builder",
    "tax_calculator",
    "price_calculator",
    "taxes_and_prices",
)


def share_the_familys_calculators(
    prepared: Sequence[prepare.PreparedScenario],
    shared_calculators: dict[str, Any] | None,
) -> tuple[list[prepare.PreparedScenario], dict[str, Any]]:
    """Give every member of a group the family's shared calculators.

    A family's first group takes them from its first scenario in runscript
    order (see `SHARED_CALCULATOR_FIELDS`). Every later group is handed the
    same ones.
    """
    if shared_calculators is None:
        shared_calculators = {
            field: getattr(prepared[0], field) for field in SHARED_CALCULATOR_FIELDS
        }
    return [replace(ready, **shared_calculators) for ready in prepared], shared_calculators


def free_the_gains_rate_exports(first: loop.BatchRun, scenario_ids: Any) -> None:
    """Release the per-household rate exports once the selling choice is solved.

    The rates themselves are small, since the compiled program averaged them
    into age cells, but the conversion channel's inputs are eighteen columns
    of [year, household] per gains scenario, about a gigabyte a scenario over
    a thirty-year window. They stay on the device until released here, and the
    second execution needs that memory for its own temporaries.
    """
    for scenario_id in scenario_ids:
        first.runs[scenario_id].gains_rate_exports.clear()
