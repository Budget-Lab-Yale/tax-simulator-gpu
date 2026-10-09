"""The record universe: who is in the run, which years they exist in, and
which random draws belong to them.

This is the host-side preparation layer (ARCHITECTURE D18). It runs once per
run, in pandas and NumPy, before anything is compiled or traced. It settles
four things:

1. The tax units in the run: the union of the ids across every Tax-Data path
   and year the run touches. Tax-Data vintages do not carry the same ids, and
   projection years add new ids within a vintage. Building it from one year
   alone would drop records that exist only in other years.

2. Each id's slot. Every id gets one slot and keeps it in every year of the
   run. A year in which a record does not exist is marked invalid and carries
   zero weight, so a record that appears in 2030 and one that disappears
   after 2028 are both ordinary universe members.

3. Each record's random numbers. Every random column is derived on its own
   from the run seed, the column's name, and the record's id (ARCHITECTURE
   D8). Adding or removing a column does not move any other column's draws,
   and a record's draw does not depend on where it sits in the file or in the
   universe. The arithmetic is in `frame/draws.py`.

4. Each record's earnings, counted once. Tax-Data reports the same dollars
   more than once: an earner's `wages1` already contains that earner's tips
   and overtime, and the household columns `wages`, `sole_prop` and `farm`
   are the two earners' columns added together. This layer replaces those
   overlapping columns with pieces that do not overlap, and the calculators
   add the pieces back up wherever a tax rule wants a total. See
   `split_overlapping_wages`.

A production run reads the wage split and the draws from the bridge output
(`frame/bridge.py`), where they were settled once per Tax-Data vintage. This
module holds the code that computes them, which the bridge calls. The form a
run reads is decided by what is in the folder it is pointed at (see
`input_form`).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Callers read the keyed random draws and the narrow whole-number storage off
# this module, so the two imports below are part of its interface.
from taxmodel.frame.draws import (  # noqa: F401
    DEFAULT_SEED,
    DRAW_COLUMNS,
    draw_uniform,
    keyed_draws,
)
from taxmodel.frame.narrow_columns import (  # noqa: F401
    MISSING_WHOLE_NUMBER,
    NARROW_WHOLE_NUMBER_COLUMNS,
    NARROW_WHOLE_NUMBER_TYPE,
    is_narrow_whole_number_column,
    narrow_whole_numbers,
    widen_whole_numbers,
)

# ---------------------------------------------------------------------------
# The id universe and its slots
# ---------------------------------------------------------------------------


def read_tax_data_year_ids(tax_data_root: str | Path, year: int) -> np.ndarray:
    """The ids in one Tax-Data year file, in file order."""
    path = Path(tax_data_root) / f"tax_units_{year}.csv"
    return pd.read_csv(path, usecols=["id"])["id"].to_numpy(dtype=np.int64)


@dataclass(frozen=True)
class RecordUniverse:
    """Who is in the run and which years each record exists in.

    `ids` holds one id per slot, in ascending order; a record's slot is its
    position in this array in every year. `valid` is a boolean array with one
    row per run year and one column per slot: True where that record exists
    in that year's Tax-Data file.
    """

    ids: np.ndarray
    years: tuple[int, ...]
    valid: np.ndarray

    def __post_init__(self) -> None:
        if self.valid.shape != (len(self.years), len(self.ids)):
            raise ValueError(
                f"validity mask is {self.valid.shape}, expected "
                f"{(len(self.years), len(self.ids))}"
            )

    @property
    def n_slots(self) -> int:
        return len(self.ids)

    def year_index(self, year: int) -> int:
        """The row of `valid` holding this year."""
        try:
            return self.years.index(int(year))
        except ValueError:
            raise KeyError(
                f"year {year} is not in this run ({self.years[0]}-{self.years[-1]})"
            ) from None

    def valid_in(self, year: int) -> np.ndarray:
        """Which slots hold a record that exists in this year."""
        return self.valid[self.year_index(year)]

    def slots_of(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """The slot each of these ids occupies. Unknown ids are an error."""
        wanted = np.asarray(ids, dtype=np.int64)
        slots = np.searchsorted(self.ids, wanted)
        out_of_range = slots >= len(self.ids)
        slots_clipped = np.where(out_of_range, 0, slots)
        if out_of_range.any() or (self.ids[slots_clipped] != wanted).any():
            raise KeyError("some ids are not members of this record universe")
        return slots


def build_universe(year_ids: Mapping[int, np.ndarray]) -> RecordUniverse:
    """Assemble the universe from the ids present in each run year.

    `year_ids` maps each run year to the ids in that year's Tax-Data file.
    Every record present in any requested year is part of the run.
    """
    years = tuple(sorted(int(y) for y in year_ids))
    everyone = np.unique(np.concatenate([np.asarray(year_ids[y]) for y in years]))
    valid = np.zeros((len(years), len(everyone)), dtype=bool)
    for row, year in enumerate(years):
        valid[row] = np.isin(everyone, np.asarray(year_ids[year]))
    return RecordUniverse(ids=everyone.astype(np.int64), years=years, valid=valid)


# The Tax-Data columns that repeat other columns, dropped as the file is read.
# `wages1` and `wages2` are each earner's total pay including that earner's
# tips and overtime; `wages`, `sole_prop` and `farm` are the two earners'
# amounts added up for the household. The record keeps the non-overlapping
# pieces: `wages_base1` and `wages_base2` (pay that is neither tips nor
# overtime), `tips1`/`tips2`, `ot1`/`ot2`, `sole_prop1`/`sole_prop2` and
# `farm1`/`farm2`.
OVERLAPPING_COLUMNS: tuple[str, ...] = (
    "wages",
    "wages1",
    "wages2",
    "sole_prop",
    "farm",
    # Household totals of tips and overtime, repeating `tips1` + `tips2` and
    # `ot1` + `ot2` the same way.
    "tips",
    "ot",
)

# Columns that no part of this model reads: Tax-Data columns, and one random
# draw that older bridge files hold. They are left out of a run's population. The bridge does not write them (`frame/bridge.convert_year`
# drops them), and the reading code skips any that it finds in an older bridge
# file.
NEVER_READ_COLUMNS: tuple[str, ...] = (
    # The second earner's sex. Only the first earner's sex is used, and only to
    # look up how likely an unmarried filer is to be a surviving spouse.
    "male2",
    # Tax-Data's ordering of parents for assigning dependents. This model takes
    # the dependent columns as given, and the distribution tables decide parent
    # status from the dependents' ages.
    "parent_rank",
    # The count of child tax credit dependents. The credit is computed from the
    # three per-dependent flags `dep_ctc1` to `dep_ctc3` instead.
    "n_dep_ctc",
    # Household partnership self-employment income. The payroll tax reads
    # `part_se1` and `part_se2`, which add to it.
    "part_se",
    # The share of the mortgage balance that sits on the primary home. The
    # mortgage interest deduction works from the two balances and the two
    # interest amounts directly.
    "prim_mort_share",
    # Net operating losses carried forward. Nothing in the model reads it; the
    # alternative minimum tax reads its own `amt_nols`. Whether the regular
    # income tax ought to use it is an open question.
    "nols",
    # Consumption by category. These support a value-added or sales tax, which
    # this model does not have.
    "c_clothing",
    "c_motor_vehicles",
    "c_durables",
    "c_other_nondurables",
    "c_food_off_premises",
    "c_gasoline",
    "c_housing_utilities",
    "c_other_services_health",
    # The Forbes billionaire tags: which records were matched to the Forbes
    # list, and what that list said about them. Wealth and estate tax are
    # computed from the balance sheet columns, not from these.
    "forbes_flag",
    "forbes_source_year",
    "forbes_rank",
    "forbes_name",
    "forbes_source_category",
    "forbes_rank_group",
    "forbes_donor_id",
    "forbes_net_worth",
    "forbes_fiscal_income",
    # A random draw for a Social Security claiming year that the model no
    # longer imputes. Bridge files converted before it was dropped still
    # hold it.
    "r.oasdi_exp",
)


# The columns `split_overlapping_wages` reads.
_WAGE_PIECES: tuple[str, ...] = (
    "wages1",
    "wages2",
    "tips1",
    "tips2",
    "ot1",
    "ot2",
)


def split_overlapping_wages(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    """Replace Tax-Data's overlapping wage columns with pieces that don't.

    Tax-Data puts an earner's tips and FLSA overtime inside that earner's
    `wages1` or `wages2`, and repeats the whole household's wages,
    sole-proprietorship income and farm income in `wages`, `sole_prop` and
    `farm`. The earner's pay that is neither tips nor overtime becomes its
    own column,

        wages_base1 = wages1 - tips1 - ot1

    and the same for the second earner, and the overlapping columns listed in
    `OVERLAPPING_COLUMNS` are dropped. A calculator that wants an earner's
    total pay adds the three pieces back; one that wants the household's adds
    all six.

    A handful of records each year report more tips and overtime than total
    wages, a Tax-Data defect.
    Those records keep their negative base pay and are warned about.
    """
    missing = [name for name in _WAGE_PIECES if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{year} Tax-Data cannot be split into wage pieces: it has no "
            + ", ".join(missing)
        )
    base1 = frame["wages1"] - frame["tips1"] - frame["ot1"]
    base2 = frame["wages2"] - frame["tips2"] - frame["ot2"]
    warn_on_negative_base_wages(frame["id"], base1, base2, year)
    # The two new columns are concatenated on in one step, since each
    # column assignment would copy the whole frame.
    added = pd.DataFrame(
        {"wages_base1": base1, "wages_base2": base2}, index=frame.index
    )
    out = pd.concat([frame, added], axis=1)
    return out.drop(columns=[c for c in OVERLAPPING_COLUMNS if c in out.columns])


def warn_on_negative_base_wages(
    ids: pd.Series, base1: pd.Series, base2: pd.Series, year: int
) -> None:
    """Warn about records that report more tips and overtime than wages."""
    for earner, base in ((1, base1), (2, base2)):
        bad = ids[base < 0]
        if bad.empty:
            continue
        warnings.warn(
            f"{year} Tax-Data: {len(bad)} records report more tips and "
            f"overtime for earner {earner} than that earner's total wages, so "
            f"wages_base{earner} is negative. The negative amount is carried "
            "into the model unchanged. This is a Tax-Data problem. Record "
            "ids: " + ", ".join(str(i) for i in bad.to_numpy()),
            stacklevel=2,
        )


def read_tax_data_year_frame(
    tax_data_root: str | Path,
    year: int,
    ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """One year of Tax-Data text, optionally limited to named records, split into pieces.

    The overlapping wage, sole-proprietorship and farm columns are split here,
    where Tax-Data enters the model. See `split_overlapping_wages`.
    """
    path = Path(tax_data_root) / f"tax_units_{year}.csv"
    frame = pd.read_csv(path, low_memory=False)
    frame["id"] = frame["id"].astype(np.int64)
    if ids is not None:
        frame = frame[frame["id"].isin(np.asarray(ids, dtype=np.int64))]
    frame = frame.reset_index(drop=True)
    # Split after selecting named records, so the warning about bad tips and
    # overtime names the records this run taxes.
    frame = split_overlapping_wages(frame, year)
    # `decedent_flag` is zero because Tax-Data has no decedents: every record
    # is alive at the start of the year, and the estate side works in expected
    # deaths.
    added = pd.DataFrame(
        {"year": int(year), "decedent_flag": 0}, index=frame.index
    )
    return pd.concat([frame, added], axis=1)


# ---------------------------------------------------------------------------
# Reading the bridge output
# ---------------------------------------------------------------------------
#
# The bridge (frame/bridge.py) converts one Tax-Data vintage, once, into one
# Parquet file per year holding the columns the model reads, with the wage
# split done and the random draws in columns. The two input forms are told
# apart by what is in the folder.

BRIDGE_INPUT = "bridge"
TAX_DATA_CSV_INPUT = "tax-data csv"
BRIDGE_MANIFEST_NAME = "manifest.yaml"


def bridge_year_path(bridge_root: str | Path, year: int) -> Path:
    """Where the bridge writes one converted year."""
    return Path(bridge_root) / f"tax_units_{year}.parquet"


def input_form(root: str | Path) -> str:
    """The form of Tax-Data this folder holds: bridge output or raw CSV."""
    root = Path(root)
    if (root / BRIDGE_MANIFEST_NAME).exists():
        return BRIDGE_INPUT
    if any(root.glob("tax_units_*.csv")):
        return TAX_DATA_CSV_INPUT
    raise FileNotFoundError(
        f"{root} holds neither a bridge manifest nor any tax_units_<year>.csv, "
        "so it is not a population the model can read"
    )


def read_bridge_manifest(bridge_root: str | Path) -> dict:
    """The bridge manifest for a converted vintage."""
    path = Path(bridge_root) / BRIDGE_MANIFEST_NAME
    return yaml.safe_load(path.read_text())


def population_provenance(tax_data_root: str | Path) -> dict[str, object]:
    """The population's provenance, for the run manifest.

    If the folder is no longer readable (a path that has moved, or a manifest
    written for a run this process did not make), the manifest records that
    instead of failing.
    """
    try:
        form = input_form(tax_data_root)
    except FileNotFoundError:
        return {"population_form": "not readable at the path recorded here"}
    if form != BRIDGE_INPUT:
        return {"population_form": form}
    bridge_manifest = read_bridge_manifest(tax_data_root)
    return {
        "population_form": form,
        "source_vintage": bridge_manifest["source_vintage"],
        "bridge_converted_at": bridge_manifest["converted_at"],
        "random_draw_seed": bridge_manifest["random_draw_seed"],
    }


def read_bridge_year_ids(bridge_root: str | Path, year: int) -> np.ndarray:
    """The ids in one converted year, in file order."""
    path = bridge_year_path(bridge_root, year)
    column = pq.read_table(path, columns=["id"])["id"]
    return column.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def read_bridge_year_columns(
    bridge_root: str | Path, year: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The columns in one converted year, in file order: the numeric ones and the text ones.

    Reads the file's schema only, not its rows. Columns no part of the model
    reads (`NEVER_READ_COLUMNS`) are left out of both lists. Text columns are
    not copied into a run's population.
    """
    schema = pq.read_schema(bridge_year_path(bridge_root, year))
    numeric: list[str] = []
    text: list[str] = []
    for field in schema:
        if field.name in NEVER_READ_COLUMNS:
            continue
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            text.append(field.name)
        else:
            numeric.append(field.name)
    return tuple(numeric), tuple(text)


def read_bridge_year_arrays(
    bridge_root: str | Path,
    year: int,
    columns: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Read one bridge year as NumPy arrays and name the omitted text columns.

    The identifier remains a signed 64-bit integer. The columns listed in
    `NARROW_WHOLE_NUMBER_COLUMNS` come back as 16-bit integers, with
    `MISSING_WHOLE_NUMBER` where the file has no value. Every other numeric,
    category, or true/false column becomes float64, with a missing value kept
    as NaN. Text columns are named in the second return value and are not
    copied into the model's population.
    """
    table = pq.read_table(bridge_year_path(bridge_root, year), columns=columns)
    arrays: dict[str, np.ndarray] = {}
    text_columns: list[str] = []
    for name in table.column_names:
        column = table[name]
        if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
            text_columns.append(name)
        elif name == "id":
            arrays[name] = column.to_numpy(zero_copy_only=False).astype(
                np.int64, copy=False
            )
        else:
            values = column.cast(pa.float64()).to_numpy(zero_copy_only=False)
            if is_narrow_whole_number_column(name):
                values = narrow_whole_numbers(name, values)
            arrays[name] = values
    return arrays, tuple(text_columns)
