"""A run's population, on NumPy arrays with one stable slot per record.

Every record keeps one slot in every year of the run. A record that is absent
in a year remains in its slot with `valid` false, zero weight, and harmless
values for the frame steps and calculators. Random draws are stored once per
slot because they are determined by record id rather than year.

Two classes hold a run's households, and both return one year's columns laid
on the run's slots through `year`. `Population` holds
every year at once. `PopulationFromDisk` holds only the record universe and
the random draws, and reads a year's Parquet file when it is asked for that
year. Thirty years of the wide household columns come to about 7 GB, which is
most of a laptop's memory; a machine with a graphics card copies the whole
block onto the card and reads every year up front. `run/scenarios.execute_run`
chooses between them.

The arrays stay on the host. The year loop takes the year it needs and turns
it into JAX arrays (`years/one_year.py`).

Dollar amounts are stored as float64. Ages, counts of dependents, filing
status, and the yes/no flags are whole numbers in a small range and are
stored as 16-bit integers; `prepare.NARROW_WHOLE_NUMBER_COLUMNS` lists them
and defines the missing value. Read a value column through `values_as_float`,
which puts a narrow column back into float64 with its missing values as NaN.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from taxmodel.frame import prepare

# A true/false column with a missing value cannot be a NumPy boolean array, so
# pandas leaves it as objects: either the words in the file or the booleans
# pandas parsed them into, depending on how the file was read.
_LOGICAL_VALUES = {
    True: 1.0,
    False: 0.0,
    "True": 1.0,
    "False": 0.0,
    "TRUE": 1.0,
    "FALSE": 0.0,
}

# Filing status 1 means Single. A padded slot needs a real filing status
# because the law table has no row for zero.
FILING_SINGLE = 1.0

# The number of year-files read at the same time. pyarrow releases Python's
# interpreter lock while it reads and decodes a file, so threads overlap the
# waiting on the network filesystem. Eight is the number of cores the cluster
# jobs ask for.
YEAR_FILE_READ_THREADS = 8


def stored_type(name: str) -> type:
    """The NumPy type this field is kept in: a narrow whole number, or float64."""
    if prepare.is_narrow_whole_number_column(name):
        return prepare.NARROW_WHOLE_NUMBER_TYPE
    return np.float64


def stored_values(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Value arrays in the types the block keeps them in.

    Float64 arrays from a test fixture or a scenario that replaces a column
    are converted: the whole-number columns go to their narrow form, with NaN
    becoming the missing-value integer.
    """
    return {
        name: (
            prepare.narrow_whole_numbers(name, array)
            if prepare.is_narrow_whole_number_column(name)
            else np.asarray(array, dtype=np.float64)
        )
        for name, array in values.items()
    }


def as_logical(column: pd.Series) -> np.ndarray | None:
    """A true/false column as 1, 0 and NaN, or None if it is really text.

    A column is identified as logical by inspecting its values, so a logical
    column that is missing everywhere is treated as text.
    """
    present = column.dropna()
    if present.empty:
        return None
    if any(value not in _LOGICAL_VALUES for value in present.unique()):
        return None
    return column.map(_LOGICAL_VALUES).to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class PopulationYear:
    """One run year's households, laid on the run's slots.

    `values` holds one array per value column, with one entry per slot, in the
    types described in this module's header. `valid` says which slots hold a
    household that exists in this year. The random draws do not vary by year
    and are not repeated here; they are read off the population itself.
    """

    year: int
    valid: np.ndarray
    values: dict[str, np.ndarray]

    def values_as_float(self, name: str) -> np.ndarray:
        """One value column as float64, with the missing value as NaN."""
        stored = self.values[name]
        if prepare.is_narrow_whole_number_column(name):
            return prepare.widen_whole_numbers(stored)
        return stored


def _slots_for_year(
    arrays: Mapping[str, np.ndarray],
    universe: prepare.RecordUniverse,
    year_row: int,
    year: int,
) -> np.ndarray:
    """The slot each of this year's records occupies, checked against the universe."""
    ids = np.asarray(arrays["id"], dtype=np.int64)
    unknown = int(np.isin(ids, universe.ids, invert=True).sum())
    if unknown:
        raise ValueError(
            f"{unknown} records in {year} are not members of the record universe"
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"population arrays for {year} contain duplicate ids")
    slots = universe.slots_of(ids)
    expected_valid = universe.valid[year_row]
    if not np.array_equal(np.sort(slots), np.flatnonzero(expected_valid)):
        raise ValueError(
            f"population arrays for {year} do not match that year's validity mask"
        )
    return slots


def _values_on_slots(
    arrays: Mapping[str, np.ndarray],
    universe: prepare.RecordUniverse,
    slots: np.ndarray,
    year: int,
    value_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """One year's value columns, scattered from file order into slot order.

    A slot with no record this year receives zero for every field except
    filing status, which receives 1 for Single because zero is not a filing
    status. A present record whose weight is missing receives zero weight.
    Missing values in all other present fields stay missing, because a missing
    spouse age, for example, carries information.
    """
    columns: dict[str, np.ndarray] = {}
    for name in value_names:
        if prepare.is_narrow_whole_number_column(name):
            incoming = prepare.narrow_whole_numbers(name, arrays[name])
        else:
            incoming = np.asarray(arrays[name], dtype=np.float64)
        if len(incoming) != len(slots):
            raise ValueError(
                f"population field {name} has {len(incoming)} rows in {year}, "
                f"expected {len(slots)}"
            )
        if name == "weight":
            incoming = np.nan_to_num(incoming, nan=0.0)
        column = np.full(
            universe.n_slots,
            FILING_SINGLE if name == "filing_status" else 0.0,
            dtype=stored_type(name),
        )
        column[slots] = incoming
        columns[name] = column
    return columns


def _draws_on_slots(
    draw_arrays_by_year: Mapping[int, Mapping[str, np.ndarray]],
    slots_by_year: Mapping[int, np.ndarray],
    universe: prepare.RecordUniverse,
    draw_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Each record's random draws, gathered from every year it appears in.

    If two years give the same record different values for one draw,
    preparation stops: a draw is keyed by record id and cannot vary by year.
    Every slot must receive each draw because every universe member exists in
    at least one run year.
    """
    draws = {
        name: np.full(universe.n_slots, np.nan, dtype=np.float64)
        for name in draw_names
    }
    seen = {name: np.zeros(universe.n_slots, dtype=bool) for name in draw_names}
    for year in universe.years:
        slots = slots_by_year[year]
        for name in draw_names:
            incoming = np.asarray(draw_arrays_by_year[year][name], dtype=np.float64)
            already_seen = seen[name][slots]
            if already_seen.any():
                old = draws[name][slots[already_seen]]
                new = incoming[already_seen]
                if not np.array_equal(old, new, equal_nan=True):
                    raise ValueError(
                        f"random draw {name} differs between years for the same record"
                    )
            draws[name][slots] = incoming
            seen[name][slots] = True
    for name, column in draws.items():
        if np.isnan(column).any():
            raise ValueError(f"random draw {name} is missing for a record in the run")
    return draws


@dataclass(frozen=True)
class Population:
    """One run's records, held in the same slots in every year.

    `values` holds one array per ordinary field, with shape `[year, slot]`:
    float64 for a dollar amount, and a 16-bit integer for the whole-number
    columns named in `prepare.NARROW_WHOLE_NUMBER_COLUMNS`. `draws` holds one
    float64 array per random field, with one value per slot. `field_names`
    preserves the source column order, and `dropped_text_columns` records source columns that no calculator reads.
    """

    universe: prepare.RecordUniverse
    field_names: tuple[str, ...]
    values: dict[str, np.ndarray]
    draws: dict[str, np.ndarray]
    dropped_text_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_values = (len(self.universe.years), self.universe.n_slots)
        for name, values in self.values.items():
            expected_type = stored_type(name)
            if values.shape != expected_values or values.dtype != expected_type:
                raise ValueError(
                    f"population field {name} is {values.shape} {values.dtype}, "
                    f"expected {expected_values} {np.dtype(expected_type).name}"
                )
        for name, draws in self.draws.items():
            if draws.shape != (self.universe.n_slots,) or draws.dtype != np.float64:
                raise ValueError(
                    f"population draw {name} is {draws.shape} {draws.dtype}, "
                    f"expected {(self.universe.n_slots,)} float64"
                )

    @property
    def years(self) -> tuple[int, ...]:
        """The run years represented by the first axis of each field."""
        return self.universe.years

    def year(
        self, year: int, columns: Sequence[str] | None = None
    ) -> PopulationYear:
        """One run year's columns. `columns` names value columns; all by default."""
        year_row = self.universe.year_index(year)
        wanted = self.values.keys() if columns is None else dict.fromkeys(columns)
        return PopulationYear(
            year=int(year),
            valid=self.universe.valid[year_row],
            values={name: self.values[name][year_row] for name in wanted},
        )



def _frame_arrays(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Convert one pandas frame into the numeric arrays the block accepts."""
    arrays: dict[str, np.ndarray] = {}
    text_columns: list[str] = []
    for name in frame.columns:
        column = frame[name]
        if name == "id":
            arrays[name] = column.to_numpy(dtype=np.int64)
            continue
        if pd.api.types.is_bool_dtype(column) or pd.api.types.is_numeric_dtype(column):
            values = column.to_numpy(dtype=np.float64)
        else:
            values = as_logical(column)
            if values is None:
                text_columns.append(name)
                continue
        if prepare.is_narrow_whole_number_column(name):
            values = prepare.narrow_whole_numbers(name, values)
        arrays[name] = values
    return arrays, tuple(text_columns)


def _ordered_union(groups: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """Names from several years, retaining the first order in which each appears."""
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for name in group:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return tuple(ordered)


def field_names_from_columns(
    columns_by_year: Mapping[int, Sequence[str]],
) -> tuple[str, ...]:
    """The population's field names, from the columns each year's source carries.

    The record id comes first and the calendar year last, and every year must
    carry the same columns in the same order.
    """
    field_names: tuple[str, ...] | None = None
    for year in sorted(columns_by_year):
        source_names = list(columns_by_year[year])
        if "id" not in source_names:
            raise ValueError(f"population arrays for {year} have no id column")
        names = ["id", *(name for name in source_names if name != "id")]
        if "year" not in names:
            names.append("year")
        if field_names is None:
            field_names = tuple(names)
        elif tuple(names) != field_names:
            raise ValueError(
                f"population arrays for {year} do not have the same columns "
                "in the same order as the first run year"
            )
    if field_names is None:
        raise ValueError("a population needs at least one year")
    return field_names


def draw_columns_among(field_names: Sequence[str]) -> tuple[str, ...]:
    """The random-draw columns among these field names."""
    return tuple(name for name in field_names if name in prepare.DRAW_COLUMNS)


def value_columns_among(field_names: Sequence[str]) -> tuple[str, ...]:
    """The ordinary value columns: everything but the id, the year, and the draws."""
    draw_names = draw_columns_among(field_names)
    return tuple(
        name for name in field_names if name not in ("id", "year", *draw_names)
    )


def population_from_arrays(
    year_arrays: Mapping[int, Mapping[str, np.ndarray]],
    universe: prepare.RecordUniverse,
    dropped_text_columns: Sequence[str] = (),
) -> Population:
    """Lay annual numeric arrays onto one record universe.

    How a year's records are placed in slots is in `_values_on_slots`, and how
    the random draws are gathered across years is in `_draws_on_slots`.
    """
    normalized = {int(year): dict(arrays) for year, arrays in year_arrays.items()}
    if set(normalized) != set(universe.years):
        raise ValueError(
            "population arrays cover years "
            f"{tuple(sorted(normalized))}, expected {universe.years}"
        )

    field_names = field_names_from_columns(
        {
            year: [name for name in normalized[year] if name != "valid"]
            for year in universe.years
        }
    )
    draw_names = draw_columns_among(field_names)
    value_names = value_columns_among(field_names)
    shape = (len(universe.years), universe.n_slots)
    values = {
        name: np.full(
            shape,
            FILING_SINGLE if name == "filing_status" else 0.0,
            dtype=stored_type(name),
        )
        for name in value_names
    }

    slots_by_year: dict[int, np.ndarray] = {}
    for year_row, year in enumerate(universe.years):
        arrays = normalized[year]
        slots = _slots_for_year(arrays, universe, year_row, year)
        slots_by_year[year] = slots
        for name, column in _values_on_slots(
            arrays, universe, slots, year, value_names
        ).items():
            values[name][year_row] = column

    draws = _draws_on_slots(normalized, slots_by_year, universe, draw_names)

    return Population(
        universe=universe,
        field_names=field_names,
        values=values,
        draws=draws,
        dropped_text_columns=tuple(dropped_text_columns),
    )


def population_from_frames(
    year_frames: Mapping[int, pd.DataFrame],
    universe: prepare.RecordUniverse,
) -> Population:
    """Build a population from small annual frames, chiefly for tests."""
    converted: dict[int, dict[str, np.ndarray]] = {}
    dropped: list[tuple[str, ...]] = []
    for year, frame in year_frames.items():
        arrays, text_columns = _frame_arrays(frame)
        converted[int(year)] = arrays
        dropped.append(text_columns)
    return _population_dropping_text_columns(converted, universe, dropped)


def _population_dropping_text_columns(
    year_arrays: dict[int, dict[str, np.ndarray]],
    universe: prepare.RecordUniverse,
    dropped: list[tuple[str, ...]],
) -> Population:
    """Build the population, first dropping the text columns any year carried.

    A column that came in as text in one year is dropped from every year, so
    the population holds the same numeric fields throughout.
    """
    dropped_text_columns = _ordered_union(dropped)
    for arrays in year_arrays.values():
        for name in dropped_text_columns:
            arrays.pop(name, None)
    return population_from_arrays(
        year_arrays,
        universe,
        dropped_text_columns=dropped_text_columns,
    )


def prepare_population(
    tax_data_root: str | Path,
    years: Sequence[int],
    seed: int = prepare.DEFAULT_SEED,
    read_every_year_up_front: bool = True,
) -> PopulationSource:
    """Read the run's population and place every year on one set of slots.

    With `read_every_year_up_front` false, bridge input comes back as a
    `PopulationFromDisk`, which reads a year when the run asks for it. Raw
    Tax-Data text is always read whole; only the bridge output is stored one
    file per year.
    """
    if prepare.input_form(tax_data_root) == prepare.BRIDGE_INPUT:
        if read_every_year_up_front:
            return population_from_bridge(tax_data_root, years, seed=seed)
        return population_from_bridge_by_year(tax_data_root, years, seed=seed)
    return population_from_tax_data(tax_data_root, years, seed=seed)


def population_from_tax_data(
    tax_data_root: str | Path,
    years: Sequence[int],
    seed: int = prepare.DEFAULT_SEED,
) -> Population:
    """Read raw Tax-Data text and place its annual records into stable slots."""
    years = [int(year) for year in years]
    year_ids = {
        year: prepare.read_tax_data_year_ids(tax_data_root, year) for year in years
    }
    universe = prepare.build_universe(year_ids)
    draws = prepare.keyed_draws(universe.ids, seed=seed)

    frames: dict[int, pd.DataFrame] = {}
    for year in years:
        frame = prepare.read_tax_data_year_frame(tax_data_root, year, universe.ids)
        frame = frame.drop(
            columns=[
                name
                for name in prepare.NEVER_READ_COLUMNS
                if name in frame.columns
            ]
        )
        frames[year] = frame.merge(draws, on="id", how="left", validate="one_to_one")
    return population_from_frames(frames, universe)


def check_bridge_seed(bridge_root: str | Path, seed: int) -> None:
    """Stop unless the vintage was converted on the seed this run asks for."""
    manifest = prepare.read_bridge_manifest(bridge_root)
    converted_seed = int(manifest["random_draw_seed"])
    if converted_seed != int(seed):
        raise ValueError(
            f"this run asks for random draws on seed {seed}, but the bridge "
            f"output at {bridge_root} was converted with seed {converted_seed}. "
            "The draws are data now: convert the vintage again on the seed you "
            "want, or run on the seed the conversion used"
        )


def population_from_bridge(
    bridge_root: str | Path,
    years: Sequence[int],
    seed: int = prepare.DEFAULT_SEED,
) -> Population:
    """Read bridge Parquet directly into a run's fixed population arrays."""
    years = [int(year) for year in years]
    check_bridge_seed(bridge_root, seed)

    # The year-files are read on a small pool of threads and the results put
    # back in run-year order.
    read_threads = max(1, min(YEAR_FILE_READ_THREADS, len(years)))

    def read_year(year: int) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
        source_names = pq.read_schema(prepare.bridge_year_path(bridge_root, year)).names
        # The columns no part of the model reads are left out of the Parquet
        # read.
        skip = set(prepare.NEVER_READ_COLUMNS)
        selected_columns = [name for name in source_names if name not in skip]
        return prepare.read_bridge_year_arrays(
            bridge_root, year, columns=selected_columns
        )

    with ThreadPoolExecutor(max_workers=read_threads) as pool:
        ids_by_year = list(
            pool.map(lambda year: prepare.read_bridge_year_ids(bridge_root, year), years)
        )
        year_ids = dict(zip(years, ids_by_year))
        universe = prepare.build_universe(year_ids)
        read_years = list(pool.map(read_year, years))

    year_arrays: dict[int, dict[str, np.ndarray]] = {}
    dropped: list[tuple[str, ...]] = []
    for year, (arrays, text_columns) in zip(years, read_years):
        year_arrays[year] = arrays
        dropped.append(text_columns)

    return _population_dropping_text_columns(year_arrays, universe, dropped)


@dataclass(frozen=True)
class PopulationFromDisk:
    """A run's population, read one bridge year-file at a time.

    Holds what spans years and is small: the record universe, each record's
    random draws, and the field names. The wide household columns are not
    held; `year` reads that year's file and lays its households on the run's
    slots, and the caller drops the result when the year is done.

    The same year read twice is read from disk twice. A capital-gains
    scenario, which runs the projection years, solves on the host, and runs
    them again, therefore reads each year's file twice.
    """

    bridge_root: Path
    universe: prepare.RecordUniverse
    field_names: tuple[str, ...]
    draws: dict[str, np.ndarray]
    dropped_text_columns: tuple[str, ...]

    @property
    def years(self) -> tuple[int, ...]:
        """The run years this population covers."""
        return self.universe.years

    @property
    def value_names(self) -> tuple[str, ...]:
        """The value columns a year read carries."""
        return value_columns_among(self.field_names)

    def year(
        self, year: int, columns: Sequence[str] | None = None
    ) -> PopulationYear:
        """One run year's columns, read from that year's file.

        `columns` names value columns; all of them by default. A caller that
        needs only a few — the per-household detail file's weight column, the
        capital-gains model's household inputs — names them, and the Parquet
        read leaves the rest on disk.
        """
        year = int(year)
        year_row = self.universe.year_index(year)
        wanted = self.value_names if columns is None else tuple(dict.fromkeys(columns))
        arrays, _ = prepare.read_bridge_year_arrays(
            self.bridge_root, year, columns=["id", *wanted]
        )
        slots = _slots_for_year(arrays, self.universe, year_row, year)
        return PopulationYear(
            year=year,
            valid=self.universe.valid[year_row],
            values=_values_on_slots(arrays, self.universe, slots, year, wanted),
        )


def population_from_bridge_by_year(
    bridge_root: str | Path,
    years: Sequence[int],
    seed: int = prepare.DEFAULT_SEED,
) -> PopulationFromDisk:
    """Read what spans years out of the bridge folder and leave the rest on disk.

    Two passes over the year-files, both small. The first reads each file's
    schema, which settles the field names and which columns are text. The
    second reads each year's ids and random-draw columns, which settles the
    record universe and each record's draws.
    """
    years = [int(year) for year in years]
    bridge_root = Path(bridge_root)
    check_bridge_seed(bridge_root, seed)
    read_threads = max(1, min(YEAR_FILE_READ_THREADS, len(years)))

    with ThreadPoolExecutor(max_workers=read_threads) as pool:
        columns_by_year = list(
            pool.map(
                lambda year: prepare.read_bridge_year_columns(bridge_root, year), years
            )
        )
        # A column that is text in any year is dropped from every year, so the
        # population holds the same numeric fields throughout.
        dropped_text_columns = _ordered_union([text for _, text in columns_by_year])
        numeric_by_year = {
            year: [
                name
                for name in numeric
                if name != "valid" and name not in dropped_text_columns
            ]
            for year, (numeric, _) in zip(years, columns_by_year)
        }
        field_names = field_names_from_columns(numeric_by_year)
        draw_names = draw_columns_among(field_names)

        read_draws = lambda year: prepare.read_bridge_year_arrays(
            bridge_root, year, columns=["id", *draw_names]
        )
        ids_and_draws = [arrays for arrays, _ in pool.map(read_draws, years)]

    year_ids = {
        year: np.asarray(arrays["id"], dtype=np.int64)
        for year, arrays in zip(years, ids_and_draws)
    }
    universe = prepare.build_universe(year_ids)
    draw_arrays_by_year = dict(zip(years, ids_and_draws))
    slots_by_year = {
        year: _slots_for_year(
            draw_arrays_by_year[year], universe, universe.year_index(year), year
        )
        for year in years
    }
    draws = _draws_on_slots(draw_arrays_by_year, slots_by_year, universe, draw_names)

    return PopulationFromDisk(
        bridge_root=bridge_root,
        universe=universe,
        field_names=field_names,
        draws=draws,
        dropped_text_columns=dropped_text_columns,
    )


# The two ways a run holds its households. Both answer `year`, and both carry
# the record universe, the field names and the random draws.
PopulationSource = Population | PopulationFromDisk
