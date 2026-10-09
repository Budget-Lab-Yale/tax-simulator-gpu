"""Per-household detail columns for the requested years.

A run may be asked for per-household numbers as well as totals and tables. The
request names calculated columns, the worlds to take them from, and the years
wanted. The compiled year loop accumulates the named columns for every year,
and the years the request did not ask for are dropped on the host.

Only columns the model creates can be asked for: the calculator's results and
the marginal tax rates it prices. Input columns are never exported; a
downstream consumer has them from Tax-Data and joins by household id, which
every detail file carries. A name outside that vocabulary is refused when the
request object is built.

The household id and the household weight are always included. A column's
population total is the weight times the column, added up over every exported
row, so the detail re-adds to the run's own totals.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from taxmodel.taxes import marginal_rates, statutory_order
from taxmodel.years import world_names

Array = jnp.ndarray

# The world whose marginal tax rates a run has. Both sets of rates (under
# baseline law and under the reform's law) are measured on the mechanical
# frame, with the household held fixed and only the law changed, so they are
# offered from the `mechanical` world only. Asking for a rate from a different
# world is refused. Baseline-law rates measured on the baseline household frame
# would be a different number, and no run computes it today.
MARGINAL_RATE_WORLD = "mechanical"


def vocabulary(world: str) -> tuple[str, ...]:
    """The column names a detail request may ask for from one world.

    Every world offers the calculator's results. The `mechanical` world also
    offers every marginal tax rate the run stores, under baseline law and under
    the reform's law. Those are the priced-dollar rates (twenty-seven dollars
    times the two prices, income tax alone and income tax plus the employee's
    payroll tax) and the balance-sheet rates, which give the wealth tax's and
    the estate tax's cost per dollar of each kind of asset and each kind of
    debt, together with the estate tax's rate on a dollar of the taxable estate
    itself.
    """
    # An unknown world name is refused here, naming the five that exist.
    world_names.world_cell(world)
    if world == MARGINAL_RATE_WORLD:
        return (*statutory_order.CALCULATED_COLUMNS, *marginal_rates.ALL_RATE_COLUMNS)
    return statutory_order.CALCULATED_COLUMNS


def _refuse_unknown_columns(world: str, columns: Sequence[str]) -> None:
    """Name every column this world does not offer, and say why."""
    offered = set(vocabulary(world))
    unknown = [name for name in columns if name not in offered]
    if not unknown:
        return
    reasons = []
    for name in unknown:
        if name in marginal_rates.ALL_RATE_COLUMNS:
            reasons.append(
                f"{name}: the marginal tax rates are measured on the "
                f"{MARGINAL_RATE_WORLD} frame, so they can only be asked for "
                f"from the {MARGINAL_RATE_WORLD} world"
            )
        elif name in statutory_order.RECORD_COLUMNS:
            reasons.append(
                f"{name}: an input column, which detail files never carry -- "
                "read it from Tax-Data and join on household_id"
            )
        else:
            reasons.append(f"{name}: not a column the model calculates")
    raise KeyError(
        f"detail request for world {world!r} names "
        f"{len(unknown)} column(s) the model cannot export:\n  - "
        + "\n  - ".join(reasons)
        + "\nA detail request names calculated columns only; the vocabulary is "
        "every column the calculators create, plus the priced marginal rates "
        f"from the {MARGINAL_RATE_WORLD} world."
    )


@dataclass(frozen=True)
class DetailRequest:
    """The per-household columns one run was asked for.

    `columns_by_world` maps a world name (`baseline`, `first_order`,
    `mechanical_ex_saving`, `mechanical`, `conventional`) to the calculated
    columns wanted from that world. `years` are the projection years whose rows
    are written; every other year is computed and dropped.

    Building the object validates it: an unknown world, and a column this world
    does not offer, fail here before the run starts. A year the run did not
    simulate is caught when the run hands its detail over.
    """

    columns_by_world: Mapping[str, tuple[str, ...]]
    years: tuple[int, ...]

    def __post_init__(self) -> None:
        columns_by_world = {
            str(world): tuple(str(name) for name in columns)
            for world, columns in self.columns_by_world.items()
        }
        years = tuple(int(year) for year in self.years)
        if not columns_by_world:
            raise ValueError("a detail request must name at least one world")
        if not years:
            raise ValueError("a detail request must name at least one year")
        if len(set(years)) != len(years):
            raise ValueError("a detail request cannot name a year twice")
        for world, columns in columns_by_world.items():
            world_names.world_cell(world)
            if not columns:
                raise ValueError(
                    f"detail request for world {world!r} names no columns"
                )
            if len(set(columns)) != len(columns):
                raise ValueError(
                    f"detail request for world {world!r} names a column twice"
                )
            _refuse_unknown_columns(world, columns)
        # Store the normalized copies so the frozen object is already checked
        # everywhere downstream.
        object.__setattr__(self, "columns_by_world", columns_by_world)
        object.__setattr__(self, "years", tuple(sorted(years)))

    def __hash__(self) -> int:
        """Two requests that ask for the same thing hash alike.

        A request is part of what decides which compiled program a scenario runs
        (the requested columns leave the year loop as extra outputs), so the
        batch runner groups scenarios by it and uses one as a dictionary key.
        The dictionary of columns by world is unhashable, so `keys()` restates
        it as a tuple.
        """
        return hash((self.keys(), self.years))

    @classmethod
    def same_columns_from_each_world(
        cls,
        columns: Sequence[str],
        worlds: Sequence[str],
        years: Sequence[int],
    ) -> DetailRequest:
        """The common case: one column list taken from several worlds."""
        return cls(
            columns_by_world={world: tuple(columns) for world in worlds},
            years=tuple(years),
        )

    @property
    def worlds(self) -> tuple[str, ...]:
        """The worlds asked for, in the order `world_names.WORLD_NAMES` lists them."""
        return tuple(
            world
            for world in world_names.WORLD_NAMES
            if world in self.columns_by_world
        )

    def keys(self) -> tuple[tuple[str, str], ...]:
        """Every (world, column) pair asked for."""
        return tuple(
            (world, column)
            for world in self.worlds
            for column in self.columns_by_world[world]
        )


def one_year(
    request: DetailRequest,
    available: Mapping[str, Mapping[str, Array]],
) -> dict[tuple[str, str], Array]:
    """The requested columns for one projection year, inside the compiled loop.

    `available` maps each requested world to the columns that world computed
    for this year. The result is one per-household array per (world, column)
    pair, which the year loop hands back as an ordinary output.
    """
    picked: dict[tuple[str, str], Array] = {}
    for world, columns in request.columns_by_world.items():
        cell = available[world]
        missing = [name for name in columns if name not in cell]
        if missing:
            raise KeyError(
                f"this run did not compute {', '.join(missing)} for the "
                f"{world} world, so they cannot be exported. The estate tax "
                "columns need the estate measurement bridge, and the marginal "
                "rates need a run that prices them."
            )
        for name in columns:
            picked[(world, name)] = jnp.asarray(cell[name])
    return picked


@dataclass(frozen=True)
class HouseholdDetail:
    """The per-household columns a run produced, for the years asked for.

    Every array has shape `[year, household slot]`, with the years in
    `years`. `valid` says which slots hold a real household in which year; a
    slot that holds none carries zero weight.
    """

    years: tuple[int, ...]
    household_id: np.ndarray
    weight: np.ndarray
    valid: np.ndarray
    columns: Mapping[tuple[str, str], np.ndarray]

    @property
    def worlds(self) -> tuple[str, ...]:
        """The worlds present, in the order `world_names.WORLD_NAMES` lists them."""
        present = {world for world, _column in self.columns}
        return tuple(
            world for world in world_names.WORLD_NAMES if world in present
        )

    def column_names(self, world: str) -> tuple[str, ...]:
        """The columns present for one world, in alphabetical order.

        The compiled year loop hands its outputs back in a dictionary whose
        keys JAX has sorted, so alphabetical is the order that is stable from
        run to run.
        """
        return tuple(
            sorted(name for key_world, name in self.columns if key_world == world)
        )


def assemble(
    request: DetailRequest,
    run_years: Sequence[int],
    household_id: Array,
    weight: Array,
    valid: Array,
    columns_by_year: Mapping[tuple[str, str], Array],
) -> HouseholdDetail:
    """Copy the year-stacked detail to the host and keep the years asked for.

    `columns_by_year` is what the year loop accumulated: one
    `[year, household slot]` array per (world, column) pair, covering every
    year the run simulated. The years the request did not ask for are dropped
    here.
    """
    run_years = tuple(int(year) for year in run_years)
    outside = [year for year in request.years if year not in run_years]
    if outside:
        raise ValueError(
            f"the detail request asks for {outside}, which this run did not "
            f"simulate; the run covers {run_years[0]} to {run_years[-1]}"
        )
    rows = np.asarray([run_years.index(year) for year in request.years])
    return HouseholdDetail(
        years=request.years,
        household_id=np.asarray(household_id)[rows],
        weight=np.asarray(weight)[rows],
        valid=np.asarray(valid)[rows],
        columns={
            key: np.asarray(value)[rows] for key, value in columns_by_year.items()
        },
    )


def table(detail: HouseholdDetail, world: str) -> pd.DataFrame:
    """One world's detail as a long table: one row per household slot per year.

    The columns the run-directory freeze document requires come first (the
    household id, the calendar year, and the weight), then the calculated
    columns under their own model names, alphabetically, then `valid`.

    Every slot in the population block gets a row, including a slot that holds
    no household in that year. Those rows carry zero weight, so the weighted
    re-addition of this file agrees with the run's own totals. `valid` lets a
    reader drop the empty slots in one step.
    """
    if world not in detail.worlds:
        raise KeyError(f"the run's detail holds no {world!r} world")
    _n_years, n_slots = detail.weight.shape
    years = np.repeat(np.asarray(detail.years, dtype=np.int64), n_slots)
    frame = {
        "household_id": detail.household_id.reshape(-1).astype(np.int64),
        "year": years,
        "weight": detail.weight.reshape(-1),
    }
    for name in detail.column_names(world):
        frame[name] = np.asarray(detail.columns[(world, name)]).reshape(-1)
    frame["valid"] = detail.valid.reshape(-1).astype(bool)
    return pd.DataFrame(frame)


@dataclass(frozen=True)
class DetailFile:
    """One written detail file, described the way a manifest wants it."""

    world: str
    path: Path
    years: tuple[int, ...]
    columns: tuple[str, ...]
    n_rows: int


def tables_to_write(
    detail: HouseholdDetail, directory: str | Path
) -> Iterator[tuple[DetailFile, pd.DataFrame]]:
    """One world's table, and where it goes, for every world asked for.

    Each file is named for the world it holds (`mechanical.parquet` and so
    on), so a consumer that wants only `first_order` reads only that file.

    The directory the files go in, `scenarios/{ID}/detail/` inside a run
    directory, and their listing in the run manifest are decided in
    `run/output_directory.py`. Building the tables and writing
    them are separate because a run writes its files on a helper thread
    (`run/file_writer.py`).

    The worlds come one at a time, and the next world's table is not built
    until the caller has taken the last one. Each table is one row per
    household per year, gigabytes for a thirty-year run.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for world in detail.worlds:
        frame = table(detail, world)
        yield (
            DetailFile(
                world=world,
                path=directory / f"{world}.parquet",
                years=detail.years,
                columns=detail.column_names(world),
                n_rows=len(frame),
            ),
            frame,
        )


def write_parquet(detail: HouseholdDetail, directory: str | Path) -> list[DetailFile]:
    """Write one Parquet file per world, and say what was written.

    Used by the tests. A run writes its detail through `tables_to_write`, one
    world at a time.
    """
    written = []
    for described, frame in tables_to_write(detail, directory):
        frame.to_parquet(described.path, index=False)
        written.append(described)
    return written

