"""Device placement, compiled programs, and memory accounting for the year loop.

The year loop (`years/loop.py`) says what the model computes: one compiled
projection year, called once per year, for several scenarios at a time. This
module holds the device mechanics.

**Scenario lanes.** Each scenario's small arrays (its law, wealth profile and
behavioral parameters) go to the device once and stay for the whole run. Its
capital-gains tables stay on the host. The lanes of one chunk are stacked
side by side once for the whole window, and the loop cuts one year out at a
time.

**Compiled programs.** A year is ``jax.vmap`` over the one-year step with a
scenario axis added, then ``jax.jit``. The population totals and the three
reported table families are compiled as their own programs after the year.
Every compile is timed.

**Memory measurement.** A GPU reports its limit, its use and its high-water
mark; the CPU backend reports nothing, so an assumed host budget stands in. A
compiled program reports what it needs before it runs. These numbers go into
the run's record only; the caller fixed the group size before the run started
(`run/group_size.py`).
"""

from __future__ import annotations

import contextlib
import functools
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.behavior import parameters as behavior_parameters
from taxmodel.reporting import corporate_spread, device_sums
from taxmodel.run import hardware, host_memory
from taxmodel.years import one_year, worlds
from taxmodel.years.stacking import stack_lanes

if TYPE_CHECKING:
    # Only for annotations; `years/loop.py` imports this module.
    from taxmodel.frame.population import PopulationSource
    from taxmodel.years.loop import BatchScenario, ScenarioProgram

Array = jnp.ndarray

# The assumed memory of a machine whose device does not report its memory,
# which is JAX's CPU backend on a laptop: 12 GB of host memory for the whole
# run. Read by the group-size arithmetic (`run/group_size.py`) and by the
# memory report below.
HOST_MEMORY_BUDGET_BYTES = 12 * 1024**3


# ---------------------------------------------------------------------------
# Device memory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryBudget:
    """A device's total and committed memory, for the counters' report.

    ``total_bytes`` is the device's memory and ``committed_bytes`` the part
    already in use, mostly the resident population block. ``source`` says
    whether the numbers came from the device's own statistics or from the
    assumed laptop budget. These numbers go into the report only.
    """

    device: str
    total_bytes: int
    committed_bytes: int
    source: str

    @property
    def free_bytes(self) -> int:
        return max(self.total_bytes - self.committed_bytes, 0)


def read_memory_budget(device: jax.Device, committed_bytes: int) -> MemoryBudget:
    """Read the device's memory statistics, or fall back to the assumed laptop budget.

    A GPU reports its limit and its bytes in use, which already include
    everything this batch made resident. JAX's CPU backend reports nothing, so
    the total is the assumed laptop budget and the committed part is what this
    batch put on the device.
    """
    stats = device.memory_stats()
    if stats is not None and "bytes_limit" in stats:
        return MemoryBudget(
            device=str(device),
            total_bytes=int(stats["bytes_limit"]),
            committed_bytes=int(stats.get("bytes_in_use", committed_bytes)),
            source="the device's own memory statistics",
        )
    return MemoryBudget(
        device=str(device),
        total_bytes=HOST_MEMORY_BUDGET_BYTES,
        committed_bytes=committed_bytes,
        source=(
            "the assumed host budget, because this device reports no memory statistics"
        ),
    )


def peak_device_bytes_in_use(device: jax.Device) -> int | None:
    """The device allocator's peak bytes in use since the process started.

    A GPU keeps this high-water mark. JAX's CPU backend keeps no memory
    statistics, so on a laptop the result is None.
    """
    stats = device.memory_stats()
    if not stats or "peak_bytes_in_use" not in stats:
        return None
    return int(stats["peak_bytes_in_use"])


def bytes_on_device(tree: object) -> int:
    """The device memory a pytree of arrays occupies, from each array's shape and dtype."""
    total = 0
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        total += math.prod(array.shape) * array.dtype.itemsize
    return int(total)


# ---------------------------------------------------------------------------
# The compiled program's own memory account
# ---------------------------------------------------------------------------


def program_demand_bytes(compiled: object) -> int:
    """The memory a compiled program reports needing for its temporaries and outputs.

    Its arguments are left out: the population block, the laws and the
    profiles are already resident and counted as committed.
    """
    analysis = compiled.memory_analysis()
    return int(analysis.temp_size_in_bytes) + int(analysis.output_size_in_bytes)


# ---------------------------------------------------------------------------
# The compiled programs: one year, the totals, and the three reported tables
# ---------------------------------------------------------------------------


def _compiled_year(program: ScenarioProgram):
    """One projection year for a group of scenarios, ready to be compiled.

    ``jax.vmap`` adds the scenario axis. The population, the calendar year,
    the heir structure and the capital-gains cell weights are the same for
    every scenario and are passed once; the law, the wealth profile, the
    behavioral parameters and the gains tables are per scenario and are
    mapped.
    """

    shared = None
    per_lane = 0
    return jax.jit(
        jax.vmap(
            functools.partial(one_year.one_year_step, program),
            in_axes=(
                per_lane,  # the wealth state carried from last year
                shared,  # the population
                per_lane,  # reform law
                per_lane,  # baseline law
                shared,  # who inherits, and how much
                per_lane,  # the wealth profile and return path
                shared,  # the calendar year
                per_lane,  # this scenario's capital-gains tables
                shared,  # the gains model's cell weights
                per_lane,  # the behavioral margins' numbers
            ),
        )
    )


def _compiled_totals():
    """The population totals for one year, for a group of scenarios.

    It takes the per-household columns the year program produced, one set per
    world per scenario lane, and the three population columns the totals read,
    and returns every published total and count. ``jax.vmap`` adds the
    scenario axis over the columns; the population is passed once.

    Compiled as its own program because forming the totals inside the year
    graph made the year's compile much slower. See
    ``device_sums.population_totals_and_counts``.

    The totals leave the program as one array per world, the totals laid end
    to end in the order `totals_layout` records, rather than as one
    one-element array per total. A year has some seven thousand totals, and
    each array that crosses the program boundary is a separate buffer to hand
    over, stack over the years and copy to the host. `totals_by_name` puts the
    names back.
    """
    shared = None
    per_lane = 0

    def one_lane(reported_by_cell, population):
        totals = device_sums.population_totals_and_counts(reported_by_cell, population)
        return {
            world: jnp.concatenate(jax.tree.leaves(one_world))
            for world, one_world in totals.items()
        }

    return jax.jit(jax.vmap(one_lane, in_axes=(per_lane, shared)))


def totals_layout(reported_by_cell, population) -> dict:
    """The structure of each world's totals, for reading `_compiled_totals`' arrays.

    Maps each world to the tree structure of its dictionary of totals, in the
    order `_compiled_totals` lays them end to end. Found from the shapes alone;
    nothing is computed.
    """
    shapes = jax.eval_shape(
        device_sums.population_totals_and_counts, reported_by_cell, population
    )
    return {
        world: jax.tree.structure(one_world) for world, one_world in shapes.items()
    }


def _one_lanes_shapes(columns: object) -> object:
    """The shapes of one lane's arrays, with the leading lane axis dropped."""
    return jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape[1:], value.dtype), columns
    )


def totals_by_name(flat_by_world: dict, layout: dict) -> dict:
    """Each world's totals as a dictionary again, from the end-to-end arrays.

    ``flat_by_world`` maps each world to an array whose last axis runs over the
    totals; the leading axes (years, lanes) are kept. Each total comes back as
    an array with a last axis of length one, the shape the totals had before
    they were laid end to end.
    """
    return {
        world: jax.tree.unflatten(
            layout[world],
            [flat[..., position : position + 1] for position in range(flat.shape[-1])],
        )
        for world, flat in flat_by_world.items()
    }


class _ReportPrograms(NamedTuple):
    """The three programs that form a year's reported tables, for a group.

    Each takes the per-household columns the year program produced
    (``device_sums.ReportColumns``) and returns a small table. They are
    compiled apart from the year, like the population totals, because the
    many whole-population sums the reported tables take made the year's
    compile slow.

    They run in this order, because the distribution table charges heirs the
    tax the allocation assigned them:

    1. ``tax_at_death`` spreads the estate tax and the tax on gains deemed
       realized at death over the heirs who bear them.
    2. ``distribution_table`` cuts the reported worlds into the table's groups
       and sums income and each tax within each.
    3. ``mtr_table_sums`` sums the marginal rates within income quintiles.
    """

    tax_at_death: object
    distribution_table: object
    mtr_table_sums: object


def _compiled_reports() -> _ReportPrograms:
    """The three report programs, each mapped over the scenario lanes.

    The report columns are per scenario; the heir structure and the population
    block are passed once.
    """
    shared = None
    per_lane = 0
    return _ReportPrograms(
        tax_at_death=jax.jit(
            jax.vmap(
                device_sums.tax_at_death_for_year,
                in_axes=(per_lane, shared, shared),
            )
        ),
        distribution_table=jax.jit(
            jax.vmap(
                device_sums.distribution_table_for_year,
                in_axes=(per_lane, shared, per_lane, per_lane),
            )
        ),
        mtr_table_sums=jax.jit(
            jax.vmap(device_sums.mtr_table_sums_for_year, in_axes=(per_lane,))
        ),
    )


class _OneYearsInputs(NamedTuple):
    """One year of one chunk's arrays.

    ``year_arguments`` is everything the compiled year takes, in its argument
    order. ``heirs`` is the heir structure, which the report programs take
    separately.
    """

    year_arguments: tuple
    heirs: object
    # This year's national corporate figures, one scalar per lane in each
    # field (`reporting/corporate_spread.CorporateDollars`).
    corporate: object


class _CompileTimer:
    """The seconds one compile took, filled in when its ``with`` block ends."""

    seconds: float = 0.0


@contextlib.contextmanager
def _timed_compile(what: str):
    """Time one of the group's compiles and print what was compiled.

    The seconds go on the group's compile total, and the caller adds the
    number of programs it made to the group's count.
    """
    timer = _CompileTimer()
    started = time.perf_counter()
    yield timer
    timer.seconds = time.perf_counter() - started
    print(f"compiled {what} in {timer.seconds:.1f} s")


def _compile_reports(
    programs: _ReportPrograms,
    columns: device_sums.ReportColumns,
    prepared: _OneYearsInputs,
    population: dict,
) -> _ReportPrograms:
    """Compile the three report programs on this year's columns.

    The heir allocation is compiled and run first because the distribution
    table takes its result as an argument.
    """
    tax_at_death = (
        programs.tax_at_death.lower(columns, prepared.heirs, population).compile()
    )
    at_death = tax_at_death(columns, prepared.heirs, population)
    return _ReportPrograms(
        tax_at_death=tax_at_death,
        distribution_table=programs.distribution_table.lower(
            columns, prepared.heirs, at_death, prepared.corporate
        ).compile(),
        mtr_table_sums=programs.mtr_table_sums.lower(columns).compile(),
    )


def _run_reports(
    programs: _ReportPrograms,
    columns: device_sums.ReportColumns,
    prepared: _OneYearsInputs,
    population: dict,
) -> dict:
    """One year's reported tables, from the year's columns."""
    at_death = programs.tax_at_death(columns, prepared.heirs, population)
    return {
        "distribution_table": programs.distribution_table(
            columns, prepared.heirs, at_death, prepared.corporate
        ),
        "mtr_table_sums": programs.mtr_table_sums(columns),
        # The heir allocation's balancing totals, which the host checks after
        # the run (`assemble.check_estate_allocation`).
        "estate_allocation": (
            {"estate_allocation": at_death["estate_allocation"]} if at_death else {}
        ),
    }


# ---------------------------------------------------------------------------
# Placement of each scenario's arrays, and stacking the lanes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResidentScenario:
    """One scenario's arrays, placed for the whole run.

    The small ones (the law, the wealth profile, the behavioral parameters)
    are put on the device once and stay there; a thirty-year law is a few
    megabytes. The capital-gains tables are kept on the host: their conversion
    columns are eighteen per-household columns for every year of the window,
    about a gigabyte a scenario, and the year loop needs one year of them at a
    time.
    """

    name: str
    reform_laws: object
    baseline_laws: object
    wealth_profiles: object
    initial_state: worlds.SimulationState
    behavior_parameters: object
    # The national corporate figures the distribution table spreads, [year]
    # in each field.
    corporate_dollars: object
    # Stacked by year and held on the host; see above.
    gains_tables: object


def _on_the_host(tree: object) -> object:
    """A pytree of arrays as plain NumPy, so it occupies no device memory."""
    return jax.device_get(tree)


def _make_resident(
    scenario: BatchScenario, n_years: int, device: jax.Device
) -> _ResidentScenario:
    """Place one scenario's arrays, with defaults filled in."""
    if scenario.wealth_profiles.r_total.shape[0] != n_years:
        raise ValueError(
            f"scenario '{scenario.name}' has wealth profiles for "
            f"{scenario.wealth_profiles.r_total.shape[0]} years, expected "
            f"{n_years}"
        )
    initial_state = (
        one_year.zero_initial_state(scenario.wealth_profiles)
        if scenario.initial_state is None
        else scenario.initial_state
    )
    return _ResidentScenario(
        name=scenario.name,
        gains_tables=_on_the_host(scenario.gains_tables),
        **jax.device_put(
            {
                "reform_laws": scenario.reform_laws,
                "baseline_laws": scenario.baseline_laws,
                "wealth_profiles": scenario.wealth_profiles,
                "initial_state": initial_state,
                "corporate_dollars": (
                    corporate_spread.no_corporate_tax(n_years)
                    if scenario.corporate_dollars is None
                    else scenario.corporate_dollars
                ),
                # Every margin runs in every program; a scenario given no
                # parameters runs all of them at neutral values.
                "behavior_parameters": (
                    behavior_parameters.all_neutral()
                    if scenario.behavior_parameters is None
                    else scenario.behavior_parameters
                ),
            },
            device,
        ),
    )


def _check_lanes_share_shapes(
    resident: Sequence[_ResidentScenario], fingerprint: str
) -> None:
    """Refuse a batch whose laws do not share one shape.

    Laws compiled one at a time may differ in shape, because a parameter that
    one law varies by filing status and another does not compiles to a
    different leaf shape. `law.batch.compile_batch` compiles a set of laws to
    one shared layout. The check is made here so that the message names the
    scenario and the shapes.
    """
    reference = resident[0]
    reference_leaves = jax.tree.leaves(
        (reference.reform_laws, reference.baseline_laws, reference.wealth_profiles)
    )
    for scenario in resident[1:]:
        leaves = jax.tree.leaves(
            (scenario.reform_laws, scenario.baseline_laws, scenario.wealth_profiles)
        )
        if len(leaves) != len(reference_leaves):
            raise ValueError(
                f"scenario '{scenario.name}' and scenario '{reference.name}' "
                "carry different law structures, so they cannot run in one "
                "batch. Compiling a set of laws to one shared layout is "
                "law.batch.compile_batch."
            )
        for position, (leaf, reference_leaf) in enumerate(
            zip(leaves, reference_leaves, strict=True)
        ):
            if jnp.shape(leaf) != jnp.shape(reference_leaf):
                raise ValueError(
                    f"scenario '{scenario.name}' presents its law array number "
                    f"{position} as {jnp.shape(leaf)} where scenario "
                    f"'{reference.name}' presents {jnp.shape(reference_leaf)}; "
                    f"the two share the switch fingerprint '{fingerprint}' but "
                    "not a shape. Compiling a set of laws to one shared layout "
                    "is law.batch.compile_batch."
                )


class _WindowArrays(NamedTuple):
    """The lanes' year-varying arrays for all years, lanes side by side.

    Every array carries the projection year on its first axis and the scenario
    lane on its second, so one slice per array is one year in the shape the
    compiled year takes.
    """

    reform_laws: object
    baseline_laws: object
    wealth_profiles: object
    corporate_dollars: object


@dataclass(frozen=True)
class _LanesSideBySide:
    """One chunk of scenarios' arrays, stacked side by side once for the whole run.

    The compiled year takes the lanes on one axis. The stacking does not
    depend on the year, so it is done once here, and the year loop takes one
    slice per array per year.

    ``by_year`` holds the arrays that change from year to year.
    ``behavior_parameters`` is the same in every year of the window and is
    passed to every year unchanged.
    """

    by_year: _WindowArrays
    behavior_parameters: object


def _lanes_side_by_side(lanes: Sequence[_ResidentScenario]) -> _LanesSideBySide:
    """Stack these scenarios' arrays for the whole window, before the year loop.

    The capital-gains tables are not stacked here: they stay on the host, and
    the year loop cuts one year of them out at a time.
    """
    lanes_beside_years = lambda *per_lane: jnp.stack(per_lane, axis=1)
    return _LanesSideBySide(
        by_year=jax.tree.map(
            lanes_beside_years,
            *[
                _WindowArrays(
                    reform_laws=lane.reform_laws,
                    baseline_laws=lane.baseline_laws,
                    wealth_profiles=lane.wealth_profiles,
                    corporate_dollars=lane.corporate_dollars,
                )
                for lane in lanes
            ],
        ),
        behavior_parameters=stack_lanes([lane.behavior_parameters for lane in lanes]),
    )


def has_its_own_device_memory(device: jax.Device) -> bool:
    """Whether the device reports its own memory limit.

    A GPU reports its limit through JAX's memory statistics. JAX's CPU backend
    reports nothing; on a laptop the compiler, the population and the running
    program share the host's memory.

    Two decisions read this. Here, whether every year of the population block
    is held on the device at once (on a GPU) or read one year at a time (on a
    laptop). In `run/group_size.py`, how many scenarios one compiled year
    carries, which on a laptop is one.
    """
    stats = device.memory_stats()
    return bool(stats and "bytes_limit" in stats)


def population_stays_resident() -> bool:
    """Whether this machine holds every year of the population at once.

    Read before the population is read, so that a run on a laptop builds a
    `population.PopulationFromDisk` instead of the whole block. The answer
    comes from `run/hardware.py` where that is possible, so the background
    population read starts without waiting for the JAX backend.
    """
    card = hardware.graphics_card_is_present()
    if card is not None:
        return card
    return has_its_own_device_memory(jax.devices()[0])


# ---------------------------------------------------------------------------
# Year slices, host-side stacking, and one lane's results
# ---------------------------------------------------------------------------


def _device_outputs_for_lane(
    stacked: one_year.YearOutputs,
    stacked_totals: dict,
    stacked_reports: dict,
    detail_by_year: dict,
    gains_exports_by_year: dict,
    lane: int,
) -> one_year.DeviceOutputs:
    """One scenario's whole projection, cut out of the group's stacked results.

    The year loop stacked its results with the years first and the scenarios
    second. One scenario's projection is that with the scenario axis removed.
    """
    lane_of_year_stacked = lambda tree: jax.tree.map(
        lambda value: value[:, lane], tree
    )
    mine = lane_of_year_stacked(stacked)
    reports = lane_of_year_stacked(stacked_reports)
    return one_year.DeviceOutputs(
        totals_by_cell=lane_of_year_stacked(stacked_totals),
        mtr_table_sums_by_year=reports["mtr_table_sums"] or None,
        wealth_channel_history=mine.wealth_channel_history,
        states_after_year=mine.state_after_year,
        detail_by_year=lane_of_year_stacked(detail_by_year),
        estate_allocation_by_year=reports["estate_allocation"],
        gains_rate_exports_by_year=lane_of_year_stacked(gains_exports_by_year),
        distribution_table_by_year=(
            None
            if isinstance(reports["distribution_table"], dict)
            else reports["distribution_table"]
        ),
    )


def _year_of(tree: object, index: int) -> object:
    """One year's entry out of host arrays stacked on a leading year axis."""
    return jax.tree.map(lambda value: value[index], tree)


@jax.jit
def _year_of_on_the_device(tree: object, index) -> object:
    """One year's entry out of device arrays stacked on a leading year axis.

    The whole tree is cut in one compiled program, with the year's position
    as data. A group's laws and profiles are over a thousand arrays, and a
    Python loop over them would send one slice to the device per array per
    year.
    """
    return jax.tree.map(
        lambda value: jax.lax.dynamic_index_in_dim(value, index, axis=0, keepdims=False),
        tree,
    )


def _stack_on_the_host(trees: Sequence[object]) -> object:
    """Stack pytrees of NumPy arrays on a new leading axis, without the device."""
    return jax.tree.map(lambda *parts: np.stack(parts), *trees)


# ---------------------------------------------------------------------------
# The steps in the order the loop calls them: put the run on the device,
# compile a group's programs, take one chunk of lanes through the years
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOnTheDevice:
    """Everything the year loop reads, placed for the whole run.

    Built once per call of `loop.run_scenarios`, before any group runs. The
    population block is resident on a GPU and read one year at a time on a
    laptop; ``read_population_year`` covers both. The heir structure and the
    gains model's cell weights are held on the host, and one year of each is
    sent to the device per year. Every scenario's small arrays are resident
    (`_ResidentScenario`). ``measured`` is the device's reported free memory
    once all of that was in place, for the counters only.
    ``detail_population`` is the three population columns a per-household
    detail file carries, read only when a run asked for one.
    """

    device: jax.Device
    read_population_year: Callable[[int], dict]
    calendar_years: Array
    heir_by_year: object
    cell_weights_by_year: object
    resident: list[_ResidentScenario]
    resident_bytes: int
    measured: MemoryBudget
    detail_population: dict | None

    def lanes(
        self, positions: Sequence[int], fingerprint: str
    ) -> list[_ResidentScenario]:
        """The resident scenarios at these positions, checked to share one shape."""
        in_group = [self.resident[position] for position in positions]
        _check_lanes_share_shapes(in_group, fingerprint)
        return in_group

    def one_years_inputs(
        self,
        lanes: Sequence[_ResidentScenario],
        side_by_side: _LanesSideBySide,
        index: int,
        state: object,
        raw_year: dict,
    ) -> _OneYearsInputs:
        """The arguments to one call of the compiled year.

        ``side_by_side`` is these lanes' arrays already stacked for the whole
        window (`_lanes_side_by_side`); one year of the law and the wealth
        profiles is cut out here.

        ``raw_year`` is that year's population block, read by the caller
        because the totals program needs the same block.

        This year's capital-gains tables are cut out of the host's arrays,
        stacked across lanes in NumPy, and copied to the device.
        """
        gains_on_the_host = _stack_on_the_host(
            [_year_of(lane.gains_tables, index) for lane in lanes]
        )
        gains_on_the_device = jax.device_put(gains_on_the_host, self.device)
        this_year = _year_of_on_the_device(side_by_side.by_year, index)
        reform = this_year.reform_laws
        baseline = this_year.baseline_laws
        profiles = this_year.wealth_profiles
        behavior_numbers = side_by_side.behavior_parameters
        calendar_year = self.calendar_years[index]
        heirs = jax.device_put(_year_of(self.heir_by_year, index), self.device)
        cell_weights = jax.device_put(
            _year_of(self.cell_weights_by_year, index), self.device
        )
        return _OneYearsInputs(
            year_arguments=(
                state,
                raw_year,
                reform,
                baseline,
                heirs,
                profiles,
                calendar_year,
                gains_on_the_device,
                cell_weights,
                behavior_numbers,
            ),
            heirs=heirs,
            corporate=this_year.corporate_dollars,
        )


def _year_reader(
    population: PopulationSource,
    years: tuple[int, ...],
    device: jax.Device,
    stays_resident: bool,
):
    """A reader for one year of the population, and its resident bytes.

    Returns a function from the year's position to that year's household
    arrays on the device, and the number of bytes held for the whole run
    (zero when each year is read and dropped).
    """
    if stays_resident:
        resident = jax.device_put(
            one_year.population_arrays_for_years(population, years), device
        )
        return (
            lambda index: jax.tree.map(lambda value: value[index], resident),
            bytes_on_device(resident),
        )

    def read_one_year(index: int):
        one = one_year.population_arrays_for_years(population, [years[index]])
        return jax.device_put({name: value[0] for name, value in one.items()}, device)

    return read_one_year, 0


def _detail_columns(
    population: PopulationSource, years: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """The three household columns a per-household detail file carries.

    The identifier and the validity mask come from the record universe. The
    weight is the only column read per year, so a run whose population is on
    disk reads that one column rather than every year's households.
    """
    universe = population.universe
    valid = np.stack([universe.valid_in(year) for year in years])
    weight = np.stack(
        [population.year(year, columns=("weight",)).values["weight"] for year in years]
    )
    return {
        "id": np.broadcast_to(universe.ids, valid.shape).copy(),
        "weight": weight,
        "valid": valid,
    }


def put_the_run_on_the_device(
    population: PopulationSource,
    scenarios: Sequence[BatchScenario],
    requested_years: tuple[int, ...],
    heir_structure: object,
    gains_cell_weights: object,
) -> RunOnTheDevice:
    """Place the population and every scenario's arrays, once per run.

    The population block is copied to the device whole on a GPU and read one
    year at a time on a laptop. The memory reading is taken after everything
    resident is in place.
    """
    n_years = len(requested_years)
    device = jax.devices()[0]
    stays_resident = has_its_own_device_memory(device)
    raw_for_year, population_bytes = _year_reader(
        population, requested_years, device, stays_resident
    )
    calendar_years = jax.device_put(
        jnp.asarray(requested_years, dtype=jnp.float64), device
    )
    # The heir structure and the gains model's cell weights are the same for
    # every scenario and are per household for every year of the window, so
    # they are held on the host and one year of each is sent to the device per
    # year.
    heir_by_year = _on_the_host(one_year.heir_arrays(heir_structure, requested_years))
    cell_weights_by_year = _on_the_host(gains_cell_weights)
    resident = [_make_resident(scenario, n_years, device) for scenario in scenarios]

    resident_bytes = population_bytes + sum(
        bytes_on_device(
            (
                scenario.reform_laws,
                scenario.baseline_laws,
                scenario.wealth_profiles,
                scenario.initial_state,
                scenario.behavior_parameters,
            )
        )
        for scenario in resident
    )
    # For the counters only: which device this ran on and what it reported
    # free.
    measured = read_memory_budget(device, resident_bytes)

    # The population columns the host needs after the run: a detail file
    # carries the household identifier, its weight and whether the slot held a
    # household that year. Read only when a run asked for per-household
    # columns.
    detail_population = None
    if any(scenario.program.detail_request is not None for scenario in scenarios):
        detail_population = _detail_columns(population, requested_years)

    host_memory.note_host_holders(
        {
            "the heir structure": heir_by_year,
            "the gains model's cell weights": cell_weights_by_year,
            "every scenario's gains tables": [r.gains_tables for r in resident],
            "the detail population columns": detail_population,
        }
    )

    return RunOnTheDevice(
        device=device,
        read_population_year=raw_for_year,
        calendar_years=calendar_years,
        heir_by_year=heir_by_year,
        cell_weights_by_year=cell_weights_by_year,
        resident=resident,
        resident_bytes=resident_bytes,
        measured=measured,
        detail_population=detail_population,
    )


class GroupPrograms:
    """The programs one group of scenarios runs, and their compile and run times.

    A group is the scenarios that share one `loop.ScenarioProgram`. Its
    projection year is compiled here, once, at ``scenarios_per_program``
    lanes, unless an earlier call left it in the run's program cache. The
    population totals and the three reported table programs are compiled on
    the first year's columns, inside the pass through the years.

    The cache entry is written as soon as the year program exists and is
    written again when the totals and the reported tables are compiled. A pass
    that only exports the capital-gains rates never compiles those four, so the
    entry it leaves holds the year program alone and the next pass compiles the
    rest. The group's counters
    (`loop.SwitchGroupCounters`) are the number of programs this call
    compiled, the seconds spent compiling, and the seconds the device spent
    running.

    The cache key uses the fixed lane count, not this call's scenario count,
    so a later call with fewer scenarios finds the same entry.
    """

    def __init__(
        self,
        run: RunOnTheDevice,
        program: ScenarioProgram,
        in_group: Sequence[_ResidentScenario],
        requested_years: tuple[int, ...],
        scenarios_per_program: int,
        program_cache: dict | None,
    ) -> None:
        self.run = run
        self.program = program
        self.requested_years = tuple(requested_years)
        self.scenarios_per_program = scenarios_per_program
        self.program_cache = program_cache
        self.execute_seconds = 0.0
        self.cache_key = (program, self.requested_years, scenarios_per_program)
        cached = None if program_cache is None else program_cache.get(self.cache_key)
        self.compiled = None if cached is None else cached[0]
        self.compiled_totals = None if cached is None else cached[1]
        self.compiled_reports = None if cached is None else cached[2]
        self.totals_layout = None if cached is None else cached[3]
        if self.compiled is None:
            # Tracing reads shapes and dtypes only, and every lane in a group
            # shares its shapes, so any lane serves.
            tracing_lanes = [in_group[0]] * scenarios_per_program
            with _timed_compile(
                f"one projection year for {scenarios_per_program} scenario(s) "
                f"(switch fingerprint {program.switch_fingerprint})"
            ) as timer:
                self.compiled = (
                    _compiled_year(program)
                    .lower(
                        *run.one_years_inputs(
                            tracing_lanes,
                            _lanes_side_by_side(tracing_lanes),
                            0,
                            stack_lanes([lane.initial_state for lane in tracing_lanes]),
                            run.read_population_year(0),
                        ).year_arguments
                    )
                    .compile()
                )
            self.compile_seconds = timer.seconds
            self.programs_compiled = 1
            self.store_in_the_cache()
            # The totals program is compiled in the pass, on the first year's
            # columns, and counted in the same two numbers.
        else:
            self.compile_seconds = 0.0
            self.programs_compiled = 0

    def store_in_the_cache(self) -> None:
        """Put the programs compiled so far into the run's program cache."""
        if self.program_cache is not None:
            self.program_cache[self.cache_key] = (
                self.compiled,
                self.compiled_totals,
                self.compiled_reports,
                self.totals_layout,
            )

    def demand_bytes(self) -> int:
        """The sum of the compiled programs' reported memory needs.

        The year program's outputs are still alive while the other four run,
        so the sum is the amount that has to fit. A pass that only exports the
        capital-gains rates compiles the year program alone, and only that
        program's demand is counted.
        """
        year = program_demand_bytes(self.compiled)
        totals = (
            0
            if self.compiled_totals is None
            else program_demand_bytes(self.compiled_totals)
        )
        reports = [
            program_demand_bytes(report) for report in self.compiled_reports or ()
        ]
        if host_memory.ledger_is_on():
            named = [("the projection year", year), ("the totals", totals)]
            named += [(f"report program {n}", size) for n, size in enumerate(reports)]
            for name, size in named:
                print(
                    f"memory ledger | program demand: {name:<28s} "
                    f"{size / 1024**3:6.2f} GB",
                    flush=True,
                )
        return year + totals + sum(reports)


@dataclass(frozen=True)
class StackedAnswers:
    """One chunk's results for the whole window, years first and lanes second."""

    small: one_year.YearOutputs
    # Each world's totals laid end to end, on the host, [year, lane, total];
    # ``totals_layout`` says which total is which (`totals_by_name`).
    totals: dict
    totals_layout: dict
    reports: dict
    detail_by_year: dict
    gains_exports_by_year: dict

    def for_lane(self, lane: int) -> one_year.DeviceOutputs:
        """One lane's whole projection, cut out of the chunk's results."""
        return _device_outputs_for_lane(
            self.small,
            totals_by_name(self.totals, self.totals_layout),
            self.reports,
            self.detail_by_year,
            self.gains_exports_by_year,
            lane,
        )


class PassThroughTheYears:
    """One chunk of lanes going through every projection year.

    The year loop calls the methods in this order, once per year:
    `read_the_population`, `this_years_inputs`, `run_the_year`,
    `form_the_totals`, `form_the_reported_tables`, `keep`;
    and `finish` after the last year.

    A capital-gains scenario runs its years twice, and the first pass is read
    only for the rate columns the host-side selling solve needs. That pass
    calls `keep_the_gains_rates` in place of the two forming methods and
    `keep`, and `finish_the_gains_rates` in place of `finish`. It compiles and
    runs the year program alone; the totals, the reported tables, the detail
    columns and the stacked results of a full pass are all skipped.

    A year's results are kept three ways. The small results (population
    totals, distribution tables, marginal-rate sums) stay on the device and
    are stacked when the window is done; the totals are then copied to the
    host as one array per world. The two per-household results (the detail
    columns and the gains rate exports) come back to the host every year,
    because the host-side realization solve reads them there. The columns the
    totals are formed from never leave the device and are released as soon as
    this year's totals are formed.

    The counters' ``execute_seconds`` is the sum of the year (from just before
    its population is read to the device finishing), the totals, and the
    reported tables.
    """

    def __init__(
        self, programs: GroupPrograms, lanes: Sequence[_ResidentScenario]
    ) -> None:
        self.programs = programs
        self.run = programs.run
        self.lanes = lanes
        self.small_by_year: list[one_year.YearOutputs] = []
        self.totals_by_year: list[dict] = []
        self.reports_by_year: list[dict] = []
        self.per_household_by_year: list[tuple] = []
        self.side_by_side = _lanes_side_by_side(lanes)
        self.year_started = 0.0

    def read_the_population(self, index: int) -> dict:
        """This year's population block, on the device; the year's clock starts here."""
        self.year_started = time.perf_counter()
        return self.run.read_population_year(index)

    def this_years_inputs(
        self, index: int, raw_year: dict, state: object
    ) -> _OneYearsInputs:
        """Cut this year out of the lanes' arrays, with the wealth state to carry in."""
        return self.run.one_years_inputs(
            self.lanes, self.side_by_side, index, state, raw_year
        )

    def run_the_year(
        self, inputs: _OneYearsInputs
    ) -> tuple[object, one_year.YearOutputs]:
        """Call the compiled year and wait for it; the year's clock stops here."""
        state, outputs = self.programs.compiled(*inputs.year_arguments)
        # Waiting here releases this year's temporaries before the next year
        # allocates its own; the group size was chosen on that basis.
        jax.block_until_ready((state, outputs))
        self.programs.execute_seconds += time.perf_counter() - self.year_started
        return state, outputs

    def form_the_totals(self, outputs: one_year.YearOutputs, raw_year: dict) -> dict:
        """The population totals, in their own program, from the year's columns.

        The columns stay on the device. The first year compiles the program.
        """
        programs = self.programs
        population = device_sums.population_columns_for_totals(raw_year)
        if programs.compiled_totals is None:
            with _timed_compile(
                f"the population totals for {programs.scenarios_per_program} "
                "scenario(s)"
            ) as timer:
                programs.compiled_totals = (
                    _compiled_totals()
                    .lower(outputs.reported_columns_by_cell, population)
                    .compile()
                )
                programs.totals_layout = totals_layout(
                    _one_lanes_shapes(outputs.reported_columns_by_cell), population
                )
            programs.compile_seconds += timer.seconds
            programs.programs_compiled += 1
            programs.store_in_the_cache()
        started = time.perf_counter()
        totals = programs.compiled_totals(
            outputs.reported_columns_by_cell, population
        )
        jax.block_until_ready(totals)
        programs.execute_seconds += time.perf_counter() - started
        return totals

    def form_the_reported_tables(
        self, outputs: one_year.YearOutputs, inputs: _OneYearsInputs, raw_year: dict
    ) -> dict:
        """The reported tables, each in its own program.

        The columns stay on the device. The first year compiles the programs
        and writes them into the run's program cache.
        """
        programs = self.programs
        population = device_sums.population_columns_for_reports(raw_year)
        if programs.compiled_reports is None:
            with _timed_compile(
                "the three reported table programs for "
                f"{programs.scenarios_per_program} scenario(s)"
            ) as timer:
                programs.compiled_reports = _compile_reports(
                    _compiled_reports(),
                    outputs.report_columns,
                    inputs,
                    population,
                )
            programs.compile_seconds += timer.seconds
            programs.programs_compiled += len(_ReportPrograms._fields)
            programs.store_in_the_cache()
        started = time.perf_counter()
        reports = _run_reports(
            programs.compiled_reports,
            outputs.report_columns,
            inputs,
            population,
        )
        jax.block_until_ready(reports)
        programs.execute_seconds += time.perf_counter() - started
        return reports

    def keep(
        self, outputs: one_year.YearOutputs, totals: dict, reports: dict
    ) -> None:
        """Keep the small results on the device; copy the per-household ones to the host."""
        self.totals_by_year.append(totals)
        self.reports_by_year.append(reports)
        self.small_by_year.append(
            outputs._replace(
                detail={},
                gains_rate_exports={},
                # The totals and the three reported tables are formed, so the
                # columns are released.
                reported_columns_by_cell={},
                report_columns=device_sums.NO_REPORT_COLUMNS,
            )
        )
        self.per_household_by_year.append(
            _on_the_host((outputs.detail, outputs.gains_rate_exports))
        )

    def keep_the_gains_rates(self, outputs: one_year.YearOutputs) -> None:
        """Copy this year's capital-gains rate exports to the host and keep nothing else."""
        self.per_household_by_year.append(_on_the_host(outputs.gains_rate_exports))

    def finish_the_gains_rates(self) -> dict:
        """The window's rate exports on the host, years first and lanes second."""
        return _stack_on_the_host(self.per_household_by_year)

    def finish(self) -> StackedAnswers:
        """Stack the window's results, years first."""
        small = stack_lanes(self.small_by_year)
        totals = _on_the_host(stack_lanes(self.totals_by_year))
        reports = stack_lanes(self.reports_by_year)
        detail_by_year, gains_exports_by_year = _stack_on_the_host(
            self.per_household_by_year
        )
        return StackedAnswers(
            small=small,
            totals=totals,
            totals_layout=self.programs.totals_layout,
            reports=reports,
            detail_by_year=detail_by_year,
            gains_exports_by_year=gains_exports_by_year,
        )
